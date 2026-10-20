"""騎手の腕と、馬と騎手の相性を出す。

集めたレース（races.csv）の全出走馬の走りを使い、1000m あたりのペースを

    ペース = レースの平均 + 馬の実力 + 騎手の腕 + 斤量 1kg あたりの差 × (斤量 - 57)

に分ける。「レースの平均」がコース・馬場・クラスの違いをまとめて吸収する。
騎手の腕は、同じ馬が別の騎手で走ったときとの比べで決まる。
1回しか出てこない馬は比べようがないので使わない。
騎乗数が少ない騎手の値はぶれるので、全騎手のばらつきから信じてよい分だけ残す（平均 0 の方へ寄せる）。
大敗（ゴール前で追うのをやめた馬など）に引っ張られないよう、ずれの大きい走の重みを下げる。

相性は、騎手の腕を除いた残りのずれを (馬, 騎手) の組ごとに平均したもの。
--validate で、各馬の最後の1走を隠して当てられるかを確かめると、相性を足すとかえって外れた。
なので予想に使う上乗せ（jockey_adj）には騎手の腕だけを入れ、相性は参考として残す。

注意: 調子が上がってきた馬ほど上手な騎手に乗り替わりやすいので、上位騎手の腕は少し大きめに出うる。

使い方:
    python3 jockey.py 202608040911              # 先に correct_speed.py を実行しておく
    python3 jockey.py 202608040911 --validate   # 騎手の腕・相性で次の1走を当てられるか確かめる

出力:
    data/<race_id>/jockeys.csv     騎手ごとの腕（秒/1000m、マイナスほど速くする）
    data/<race_id>/jockey_adj.csv  今年の登録馬ごとの、乗り替わりによる上乗せと相性
"""
import collections
import csv
import math
import os
import sys

import numpy as np

import calibrate_pedigree
import netkeiba

ITERATIONS = 60
MIN_RIDES_FOR_TAU = 30
SLOPE_RIDGE = float(os.environ.get("SLOPE_RIDGE", 1.0))  # 馬ごとの傾きを 0 へ寄せる強さ（大きいほど傾きを入れない）


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_runs(race_ids):
    runs = []
    for rid in race_ids:
        for r in netkeiba.race_runs(rid):
            r["race_id"] = rid
            runs.append(r)
    return runs


def fit(runs, distance, date):
    """騎手の腕・相性を当てはめる。distance・date は race_id → 距離・日付（YYYYMMDD）。

    馬ごとに「時間とともに速くなる（遅くなる）傾き」も入れる。成長期にたまたま乗っていた騎手が
    相性が良いように見えてしまうのを防ぐため。
    """
    horse_n = collections.Counter(r["horse_id"] for r in runs)
    runs = [r for r in runs if horse_n[r["horse_id"]] >= 2]
    y = np.array([r["time"] / distance[r["race_id"]] * 1000 for r in runs])
    kg = np.array([r["weight"] - 57 for r in runs])
    days = np.array([(np.datetime64(f"{date[r['race_id']][:4]}-{date[r['race_id']][4:6]}-{date[r['race_id']][6:]}")
                      - np.datetime64("2000-01-01")).astype(int) for r in runs]) / 365.0
    idx = {}
    for key in ("race_id", "horse_id", "jockey_id"):
        values = sorted({r[key] for r in runs})
        pos = {v: i for i, v in enumerate(values)}
        idx[key] = (np.array([pos[r[key]] for r in runs]), values)
    ri, hi, ji = idx["race_id"][0], idx["horse_id"][0], idx["jockey_id"][0]
    race = np.zeros(ri.max() + 1)
    horse = np.zeros(hi.max() + 1)
    jockey = np.zeros(ji.max() + 1)
    beta, lam, w = 0.0, 50.0, np.ones(len(y))
    # 馬ごとの時間の中心からのずれ（年）。傾きは馬の実力と別に推定する
    t = days - (np.bincount(hi, days) / np.bincount(hi))[hi]
    slope = np.zeros(len(horse))

    def group_mean(values, index, size, weights, ridge=0.0):
        return np.bincount(index, weights * values, size) / (np.bincount(index, weights, size) + ridge)

    for it in range(ITERATIONS):
        trend = slope[hi] * t
        race = group_mean(y - horse[hi] - trend - jockey[ji] - beta * kg, ri, len(race), w)
        horse = group_mean(y - race[ri] - trend - jockey[ji] - beta * kg, hi, len(horse), w)
        part = y - race[ri] - horse[hi] - jockey[ji] - beta * kg
        slope = np.bincount(hi, w * t * part, len(horse)) / (np.bincount(hi, w * t * t, len(horse)) + SLOPE_RIDGE)
        trend = slope[hi] * t
        jockey = group_mean(y - race[ri] - horse[hi] - trend - beta * kg, ji, len(jockey), w, lam)
        shift = np.average(jockey[ji], weights=w)  # 騎手の平均は 0、ずれは馬の側へ
        jockey -= shift
        horse += shift
        res = y - race[ri] - horse[hi] - trend - jockey[ji]
        beta = float(np.sum(w * kg * res) / np.sum(w * kg * kg))
        res -= beta * kg
        scale = 1.4826 * np.median(np.abs(res))
        w = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(res), 1e-9))
        if it % 10 == 9:  # 騎手の腕の「本当の差」を見積もり直し、寄せる強さを決める
            sigma2 = float(np.sum(w * res ** 2) / np.sum(w))
            n = np.bincount(ji, w, len(jockey))
            raw = jockey * (n + lam) / n
            many = n >= MIN_RIDES_FOR_TAU
            tau2 = max(float(np.var(raw[many]) - np.mean(sigma2 / n[many])), 1e-4)
            lam = sigma2 / tau2

    sigma2 = float(np.sum(w * res ** 2) / np.sum(w))
    rides = np.bincount(ji, minlength=len(jockey))
    names = {}
    for r in runs:
        names[r["jockey_id"]] = r["jockey"]

    # 相性: (馬, 騎手) の組ごとの残りのずれ
    pair_res = collections.defaultdict(list)
    for r, e in zip(runs, res + 0):
        pair_res[(r["horse_id"], r["jockey_id"])].append(e)
    horse_jockeys = collections.defaultdict(set)
    for h, j in pair_res:
        horse_jockeys[h].add(j)
    usable = {k: v for k, v in pair_res.items() if len(v) >= 2 and len(horse_jockeys[k[0]]) >= 2}
    pm = np.array([np.mean(v) for v in usable.values()])
    pn = np.array([len(v) for v in usable.values()])
    pair_tau2 = max(float(np.var(pm) - np.mean(sigma2 / pn)), 0.0)
    pairs = {k: float(np.mean(v)) * pair_tau2 / (pair_tau2 + sigma2 / len(v)) for k, v in pair_res.items()}

    return {
        "jockey": {jid: float(jockey[i]) for i, jid in enumerate(idx["jockey_id"][1])},
        "rides": {jid: int(rides[i]) for i, jid in enumerate(idx["jockey_id"][1])},
        "names": names, "beta": beta, "sigma": math.sqrt(sigma2), "tau": math.sqrt(sigma2 / lam),
        "pairs": pairs, "pair_tau": math.sqrt(pair_tau2), "pair_noise": math.sqrt(float(np.mean(sigma2 / pn))),
        "pair_obs_sd": float(np.std(pm)), "pair_count": len(usable), "runs": len(runs),
        "race": {v: float(race[i]) for i, v in enumerate(idx["race_id"][1])},
        "horse": {v: float(horse[i]) for i, v in enumerate(idx["horse_id"][1])},
        "slope": {v: float(slope[i]) for i, v in enumerate(idx["horse_id"][1])},
        "center": {v: float(c) for v, c in zip(idx["horse_id"][1], np.bincount(hi, days) / np.bincount(hi))},
    }


def validate(runs, distance, date):
    """各馬の最後の1走を隠して当てはめ、隠した走を当てられるかで騎手の腕・相性の効き目を確かめる。"""
    by_horse = collections.defaultdict(list)
    for r in runs:
        by_horse[r["horse_id"]].append(r)
    test = set()
    for rs in by_horse.values():
        if len(rs) >= 3:
            last = max(rs, key=lambda r: date[r["race_id"]])
            test.add((last["race_id"], last["horse_id"]))
    m = fit([r for r in runs if (r["race_id"], r["horse_id"]) not in test], distance, date)
    err = collections.defaultdict(list)
    for r in runs:
        if (r["race_id"], r["horse_id"]) not in test or r["race_id"] not in m["race"] or r["horse_id"] not in m["horse"]:
            continue
        d = r["race_id"]
        t = (np.datetime64(f"{date[d][:4]}-{date[d][4:6]}-{date[d][6:]}") - np.datetime64("2000-01-01")).astype(int) / 365.0
        y = r["time"] / distance[d] * 1000
        base = m["race"][d] + m["horse"][r["horse_id"]] + m["slope"][r["horse_id"]] * (t - m["center"][r["horse_id"]]) \
            + m["beta"] * (r["weight"] - 57)
        j = m["jockey"].get(r["jockey_id"], 0.0)
        p = m["pairs"].get((r["horse_id"], r["jockey_id"]), 0.0)
        for k, pred in (("馬だけ", base), ("＋騎手の腕", base + j), ("＋騎手の腕＋相性", base + j + p)):
            err[k].append(float(np.clip(y - pred, -1.0, 1.0)) ** 2)  # 大敗は ±1秒/1000m で打ち切る
    ref = np.array(err["馬だけ"])
    print(f"隠した走: {len(ref)}")
    for k, e in err.items():
        diff = np.array(e) - ref
        print(f"  {k:14s} 二乗誤差 {np.mean(e):.5f}  馬だけとの差 {np.mean(diff):+.5f} ± {np.std(diff) / math.sqrt(len(diff)):.5f}")


def jockey_of(race_id, horse_id):
    return next((r["jockey_id"] for r in netkeiba.race_runs(race_id) if r["horse_id"] == horse_id), None)


def main(race_id):
    out = os.path.join("data", race_id)
    races = read(os.path.join(out, "races.csv"))
    # 過去の菊花賞は、あとで答え合わせに使うので腕の計算から外す
    train = [r for r in races if r["race_id"] not in calibrate_pedigree.PAST_RACES]
    distance = {r["race_id"]: int(r["distance"]) for r in races}
    date = {r["race_id"]: r["date"] for r in races}
    all_runs = load_runs([r["race_id"] for r in train])
    if "--validate" in sys.argv:
        validate(all_runs, distance, date)
        return
    model = fit(all_runs, distance, date)

    jockeys = sorted(({"jockey_id": j, "name": model["names"][j], "rides": model["rides"][j],
                       "effect_sec_per_1000m": round(e, 3), "effect_3000m_sec": round(3 * e, 2)}
                      for j, e in model["jockey"].items()), key=lambda r: r["effect_sec_per_1000m"])
    with open(os.path.join(out, "jockeys.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(jockeys[0]))
        w.writeheader()
        w.writerows(jockeys)

    print(f"使った走: {model['runs']}  1走のぶれ ±{model['sigma']:.2f}秒/1000m  "
          f"騎手の腕の本当の差 ±{model['tau']:.3f}秒/1000m  斤量1kgあたり {model['beta']:+.3f}秒/1000m")
    print(f"相性: 組の数 {model['pair_count']}  組ごとの差の実際 ±{model['pair_obs_sd']:.3f} / "
          f"偶然で出る分 ±{model['pair_noise']:.3f} → 本当の差 ±{model['pair_tau']:.3f}")
    print("腕の上位（騎乗200回以上）:")
    for j in [j for j in jockeys if j["rides"] >= 200][:10]:
        print(f"  {j['name']}: {j['effect_3000m_sec']:+.2f}秒/3000m（{j['rides']}回）")

    # 今年の登録馬: 「今の速さ」を出した走の騎手と、菊花賞の騎手との差
    _, entries = netkeiba.entries(race_id)
    base = {b["course"]: float(b["baseline_sec_per_1000m"]) for b in read(os.path.join(out, "baseline.csv"))}
    effects = {(e["kind"], e["key"]): float(e["sec_per_1000m"]) for e in read(os.path.join(out, "effects.csv"))}
    runs = collections.defaultdict(list)
    for r in read(os.path.join(out, "runs_corrected.csv")):
        rel = 3600 / float(r["corrected_kmh"]) - effects.get(("age", r["age"]), 0.0) - base[r["course"]]
        runs[r["name"]].append((r["date"], rel, r["race_id"]))
    rows = []
    for e in entries:
        recent = sorted(runs[e["name"]], reverse=True)[:4]
        best = sorted(recent, key=lambda x: x[1])[:2]
        past = [jockey_of(rid, e["horse_id"]) for _, _, rid in best]
        past_effect = float(np.mean([model["jockey"].get(j, 0.0) for j in past]))
        now = model["jockey"].get(e["jockey_id"], 0.0) if e["jockey_id"] else 0.0
        pair = model["pairs"].get((e["horse_id"], e["jockey_id"]), 0.0)
        rows.append({
            "name": e["name"], "entry_jockey": e["jockey"] or "未定", "entry_jockey_effect": round(now, 3),
            "past_jockeys": "・".join(model["names"].get(j, "?") for j in past), "past_effect": round(past_effect, 3),
            "pair_effect_reference": round(pair, 3),
            # 乗り替わりによる上乗せ（秒/1000m、マイナスほど速くなる）。騎手未定なら 0（平均的な騎手）とみなす
            "jockey_adj": round(now - past_effect, 3),
        })
    with open(os.path.join(out, "jockey_adj.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1])
