"""集めた全レースを使って、モデルの材料が単勝オッズに上乗せできる情報を持っているかを確かめる。

各レースの各馬について
    オッズの確率（の対数）
    今の速さ: そのレースより前に、集めたレースの中で走った芝の直近4走のうち良かった2走の rel の平均
              （馬場差・年齢差を補正。前の走がなければ「データなし」の印を付け、値はレース内の平均）
    騎手の腕: jockey.fit で出した値（予想するグループのレースを除いて計算し直す）
を材料にし、強さ = Σ 重み × 材料 として、1〜3着の順番が起こる確率（Plackett-Luce）が
一番大きくなる重みをニュートン法で探す。

レースを日付順に5つのグループに分け、あるグループを除いたデータで重みを決めて、そのグループを予想する。
「オッズだけ」と「オッズ＋モデルの材料」の当たり具合を比べる。

使い方:
    python3 broad_backtest.py 202608040911   # 先に correct_speed.py を実行しておく

出力:
    data/<race_id>/broad_backtest.csv
"""
import collections
import csv
import math
import os
import sys

import numpy as np

import calibrate_pedigree
import jockey
import netkeiba

FOLDS = 5
TOP = 3
MIN_FIELD = 8
RECENT, BEST = 4, 2
MAX_DAYS = 365
GOING = {"稍": "稍重", "不": "不良"}


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def day_number(d):
    return int((np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}") - np.datetime64("2000-01-01")).astype(int))


def build(out):
    races = read(os.path.join(out, "races.csv"))
    base = {b["course"]: float(b["baseline_sec_per_1000m"]) for b in read(os.path.join(out, "baseline.csv"))}
    effects = {(e["kind"], e["key"]): float(e["sec_per_1000m"]) for e in read(os.path.join(out, "effects.csv"))}
    day = {(d["date"], d["venue"], d["going"]): float(d["sec_per_1000m"]) for d in read(os.path.join(out, "day_variants.csv"))}
    info = {r["race_id"]: r for r in races}

    runs = {}
    history = collections.defaultdict(list)  # 馬 → (日, rel)
    for r in races:
        rr = netkeiba.race_runs(r["race_id"])
        runs[r["race_id"]] = rr
        course = f"{r['venue']}芝{r['inout']}{r['distance']}"
        key = (r["date"], r["venue"], r["going"])
        if course not in base or key not in day:
            continue
        for x in rr:
            age = int(r["date"][:4]) - int(x["horse_id"][:4])
            age_key = "2歳" if age == 2 else "3歳" if age == 3 else "3歳以上"
            pace = x["time"] / int(r["distance"]) * 1000 - day[key] - effects.get(("age", age_key), 0.0)
            history[x["horse_id"]].append((day_number(r["date"]), pace - base[course]))
    return races, info, runs, history


def features(race, rr, history, jockeys):
    """1レース分の材料（オッズの対数確率・今の速さ・データなしの印・騎手の腕）と 1〜3着の並び。"""
    today = day_number(race["date"])
    inv = np.array([1 / x["odds"] for x in rr])
    market = np.log(inv / inv.sum())
    level = []
    for x in rr:
        prior = sorted((h for h in history[x["horse_id"]] if 0 < today - h[0] <= MAX_DAYS), reverse=True)[:RECENT]
        best = sorted(p[1] for p in prior)[:BEST]
        level.append(float(np.mean(best)) if best else None)
    known = [v for v in level if v is not None]
    if len(known) < 3:
        return None
    mean = float(np.mean(known))
    lv = np.array([v if v is not None else mean for v in level])
    missing = np.array([v is None for v in level], float)
    jk = np.array([jockeys.get(x["jockey_id"], 0.0) for x in rr])
    # 強さはプラスほど強い向きにそろえる（rel・騎手の腕はマイナスほど速い）
    X = np.column_stack([market, -(lv - mean), missing, -jk])
    finish = [int(x["finish"]) if x["finish"].isdigit() else 99 for x in rr]
    order = [i for i in np.argsort(finish, kind="stable")[:TOP] if finish[i] < 99]
    return X, order


def loglik(theta, data):
    """順番の確率の対数・勾配・ヘッセ行列。"""
    ll, g, H = 0.0, np.zeros(len(theta)), np.zeros((len(theta), len(theta)))
    for X, order in data:
        s = X @ theta
        alive = np.ones(len(s), bool)
        for i in order:
            sa, Xa = s[alive], X[alive]
            p = np.exp(sa - sa.max())
            p /= p.sum()
            mu = p @ Xa
            ll += s[i] - (sa.max() + math.log(np.sum(np.exp(sa - sa.max()))))
            g += X[i] - mu
            H -= (Xa * p[:, None]).T @ Xa - np.outer(mu, mu)
            alive[i] = False
    return ll, g, H


def fit(data, use):
    """use で選んだ材料だけで重みを探す（ニュートン法）。"""
    theta = np.zeros(4)
    theta[0] = 1.0
    idx = np.array(use)
    for _ in range(30):
        ll, g, H = loglik(theta, data)
        step = np.linalg.solve(H[np.ix_(idx, idx)] - 1e-6 * np.eye(len(idx)), g[idx])
        theta[idx] -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    ll, _, H = loglik(theta, data)
    se = np.full(4, np.nan)
    se[idx] = np.sqrt(np.diag(np.linalg.inv(-H[np.ix_(idx, idx)])))
    return theta, se, ll


def main(race_id):
    out = os.path.join("data", race_id)
    races, info, runs, history = build(out)
    targets = [r for r in races if r["race_id"] not in calibrate_pedigree.PAST_RACES
               and len(runs[r["race_id"]]) >= MIN_FIELD and all(x["odds"] for x in runs[r["race_id"]])]
    targets.sort(key=lambda r: r["date"])
    folds = np.array_split(np.arange(len(targets)), FOLDS)
    distance = {r["race_id"]: int(r["distance"]) for r in races}
    date = {r["race_id"]: r["date"] for r in races}

    models = {"オッズだけ": [0], "オッズ＋今の速さ": [0, 1, 2], "オッズ＋今の速さ＋騎手": [0, 1, 2, 3]}
    cv = {k: [] for k in models}
    rows = []
    for f, test_idx in enumerate(folds):
        test_ids = {targets[i]["race_id"] for i in test_idx}
        jk = jockey.fit([dict(x, race_id=rid) for rid, rr in runs.items() if rid not in test_ids for x in rr],
                        distance, date)["jockey"]
        data = {}
        for r in targets:
            d = features(r, runs[r["race_id"]], history, jk)
            if d:
                data[r["race_id"]] = d
        train = [d for rid, d in data.items() if rid not in test_ids]
        test = [(rid, d) for rid, d in data.items() if rid in test_ids]
        for k, use in models.items():
            theta, _, _ = fit(train, use)
            for rid, d in test:
                ll, _, _ = loglik(theta, [d])
                cv[k].append(ll)
                rows.append({"fold": f, "race_id": rid, "model": k, "loglik": round(ll, 4)})
        print(f"グループ {f + 1}/{FOLDS}: 予想したレース {len(test)}", flush=True)

    print(f"\n答え合わせ（1〜3着の順番の確率の対数、1レースあたりの平均、大きいほど良い）  レース数 {len(cv['オッズだけ'])}")
    ref = np.array(cv["オッズだけ"])
    for k, v in cv.items():
        diff = np.array(v) - ref
        print(f"  {k:20s} {np.mean(v):+.4f}  オッズだけとの差 {np.mean(diff):+.4f} ± {np.std(diff) / math.sqrt(len(diff)):.4f}")

    all_data = list(data.values())
    theta, se, ll1 = fit(all_data, [0, 1, 2, 3])
    _, _, ll0 = fit(all_data, [0])
    print(f"\n全レースで決めた重み（±は誤差）:")
    for name, t, s in zip(["オッズ", "今の速さ（1秒/1000m速いと）", "データなしの印", "騎手の腕（1秒/1000m速くすると）"], theta, se):
        print(f"  {name:28s} {t:+.3f} ± {s:.3f}")
    print(f"  モデルの材料を足した効果: 尤度比 {2 * (ll1 - ll0):.1f}（3つ足したので 7.8 を超えれば偶然とは言いにくい）")
    with open(os.path.join(out, "broad_backtest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
