"""単勝オッズに上乗せできる情報（候補）を、集めた全レースでまとめて試す実験台。

各候補は「レースの前に分かる情報」だけで作る。当日の馬場の偏りは、その日の“前の”レースの結果だけを使う。
強さ = Σ 重み × 材料 として、1〜3着の順番が起こる確率（Plackett-Luce）を当てはめ、
レースを日付順に5つに分けて「そのグループを除いて重みを決め、そのグループを予想」する。
オッズだけのときより当たり具合が上がった候補だけが、本当に上乗せできる情報。

候補（プラスほど強い向き。馬番・位置取りはレース内で 0〜1 にそろえる: 0 = 内・前）
    draw          (0.5 - 馬番)                          枠の内外（オッズは知っているはず。確かめ用）
    day_inside    当日の内有利度 × (0.5 - 馬番)          当日の内有利度 = 0.5 - 前のレースの1〜3着の馬番の平均
    day_front     当日の前有利度 × (0.5 - いつもの位置)  当日の前有利度 = 0.5 - 前のレースの1〜3着の4角位置の平均
    pace          -(先行馬の多さ) × (0.5 - いつもの位置) 先行馬が多いほど前の馬は不利、後ろの馬は有利
    unlucky       前走の(着順 - 上がりの順位)             上がりは速いのに着順が悪かった（負けすぎ）
    body_gain     馬体重 +10kg 以上                       当日の馬体重
    body_loss     馬体重 -10kg 以下
いつもの位置: 集めたレースの中の前の走（1年以内）での1コーナーの位置の平均。分からない馬は印を付ける。

使い方:
    python3 feature_lab.py 202608040911   # 先に correct_speed.py を実行しておく

出力:
    data/<race_id>/feature_lab.csv
"""
import collections
import csv
import math
import os
import sys

import numpy as np

import calibrate_pedigree
import netkeiba

FOLDS = 5
TOP = 3
MIN_FIELD = 8
MAX_DAYS = 365
PREV_DAYS = int(os.environ.get("PREV_DAYS", 0))  # 前の何日分の同じ競馬場のレースも当日の偏りに使うか
NAMES = ["market", "draw", "day_inside", "day_front", "pace", "unlucky", "body_gain", "body_loss",
         "no_style", "no_last"]
CANDIDATES = {
    "draw（枠の内外）": ["draw"],
    "day_inside（当日の内有利）": ["day_inside"],
    "day_front（当日の前有利）": ["day_front", "no_style"],
    "pace（展開）": ["pace", "no_style"],
    "unlucky（前走の負けすぎ）": ["unlucky", "no_last"],
    "body（馬体重の増減）": ["body_gain", "body_loss"],
}


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def day_number(d):
    return int((np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}") - np.datetime64("2000-01-01")).astype(int))


def norm(rank, n):
    return (rank - 1) / (n - 1) if n > 1 else 0.5


def build(out):
    races = sorted(read(os.path.join(out, "races.csv")), key=lambda r: (r["date"], r["race_id"]))
    runs = {r["race_id"]: netkeiba.race_runs(r["race_id"]) for r in races}

    # 当日の偏り: その日・その競馬場の、そのレースより前のレースの1〜3着の馬番と4角位置
    day_tops = collections.defaultdict(list)  # (日, 競馬場) → [(レース番号, 馬番の平均, 4角位置の平均)]
    for r in races:
        rr = runs[r["race_id"]]
        n = len(rr)
        top = [x for x in rr if x["finish"] in ("1", "2", "3") and x["number"] and x["passing"]]
        if n >= MIN_FIELD and top:
            day_tops[(r["date"], r["venue"])].append((
                int(r["race_id"][-2:]),
                float(np.mean([norm(x["number"], n) for x in top])),
                float(np.mean([norm(x["passing"][-1], n) for x in top]))))

    history = collections.defaultdict(list)  # 馬 → [(日, 1角位置, 着順-上がり順位)]
    data = {}
    for r in races:
        rr = runs[r["race_id"]]
        n = len(rr)
        today = day_number(r["date"])
        if r["race_id"] not in calibrate_pedigree.PAST_RACES and n >= MIN_FIELD and all(x["odds"] for x in rr) \
                and all(x["number"] for x in rr):
            no = int(r["race_id"][-2:])
            earlier = [t for t in day_tops[(r["date"], r["venue"])] if t[0] < no]
            if PREV_DAYS:  # 前の日（同じ競馬場の開催）のレースも、当日の偏りの手がかりに足す
                for back in range(1, PREV_DAYS + 1):
                    prev = str(np.datetime64(f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}") - np.timedelta64(back, "D")).replace("-", "")
                    earlier += day_tops[(prev, r["venue"])]
            inside = 0.5 - float(np.mean([t[1] for t in earlier])) if earlier else 0.0
            front = 0.5 - float(np.mean([t[2] for t in earlier])) if earlier else 0.0
            style, last = [], []
            for x in rr:
                prior = [h for h in history[x["horse_id"]] if 0 < today - h[0] <= MAX_DAYS]
                style.append(float(np.mean([h[1] for h in prior])) if prior else None)
                last.append(max(prior)[2] if prior else None)
            known = [s for s in style if s is not None]
            mean_style = float(np.mean(known)) if known else 0.5
            front_share = float(np.mean([s < 0.25 for s in known])) if known else 0.25
            inv = np.array([1 / x["odds"] for x in rr])
            rows = []
            for x, s, l in zip(rr, style, last):
                d = 0.5 - norm(x["number"], n)
                st = 0.5 - (s if s is not None else mean_style)
                bc = x["body_change"] or 0
                rows.append([0.0, d, inside * d, front * st, -(front_share - 0.25) * st,
                             l if l is not None else 0.0, float(bc >= 10), float(bc <= -10),
                             float(s is None), float(l is None)])
            X = np.array(rows)
            X[:, 0] = np.log(inv / inv.sum())
            finish = [int(x["finish"]) if x["finish"].isdigit() else 99 for x in rr]
            order = [i for i in np.argsort(finish, kind="stable")[:TOP] if finish[i] < 99]
            data[r["race_id"]] = (r["date"], X, order)
        # このレースの結果を、後のレースのための「前の走」として記録する
        ups = sorted((x["last600"], i) for i, x in enumerate(rr) if x["last600"] is not None)
        up_rank = {i: k + 1 for k, (_, i) in enumerate(ups)}
        for i, x in enumerate(rr):
            if x["passing"] and x["finish"].isdigit() and i in up_rank:
                history[x["horse_id"]].append(
                    (today, norm(x["passing"][0], n), norm(int(x["finish"]), n) - norm(up_rank[i], n)))
    return data


def loglik(theta, data):
    ll, g, H = 0.0, np.zeros(len(theta)), np.zeros((len(theta), len(theta)))
    for X, order in data:
        s = X @ theta
        alive = np.ones(len(s), bool)
        for i in order:
            sa, Xa = s[alive], X[alive]
            m = sa.max()
            p = np.exp(sa - m)
            z = p.sum()
            p /= z
            mu = p @ Xa
            ll += s[i] - m - math.log(z)
            g += X[i] - mu
            H -= (Xa * p[:, None]).T @ Xa - np.outer(mu, mu)
            alive[i] = False
    return ll, g, H


def fit(data, cols):
    data = [(X[:, cols], order) for X, order in data]
    theta = np.zeros(len(cols))
    theta[0] = 1.0
    for _ in range(30):
        _, g, H = loglik(theta, data)
        step = np.linalg.solve(H - 1e-6 * np.eye(len(cols)), g)
        theta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    ll, _, H = loglik(theta, data)
    return theta, np.sqrt(np.diag(np.linalg.inv(-H))), ll


def main(race_id):
    out = os.path.join("data", race_id)
    data = build(out)
    ids = sorted(data, key=lambda k: data[k][0])
    folds = np.array_split(np.arange(len(ids)), FOLDS)
    print(f"レース数 {len(ids)}")

    def cv(cols):
        lls = []
        for test in folds:
            tset = {ids[i] for i in test}
            theta, _, _ = fit([data[k][1:] for k in ids if k not in tset], cols)
            lls += [loglik(theta, [(data[k][1][:, cols], data[k][2])])[0] for k in ids if k in tset]
        return np.array(lls)

    base = cv([0])
    _, _, ll_base = fit([d[1:] for d in data.values()], [0])
    rows = []
    print(f"{'候補':24s} {'答え合わせでの上乗せ（1レースあたり）':>18s}   {'重み（全レース）':>12s}   尤度比")
    for name, feats in CANDIDATES.items():
        cols = [0] + [NAMES.index(f) for f in feats]
        diff = cv(cols) - base
        theta, se, ll = fit([d[1:] for d in data.values()], cols)
        lr = 2 * (ll - ll_base)
        print(f"  {name:22s} {np.mean(diff):+.4f} ± {np.std(diff) / math.sqrt(len(diff)):.4f}   "
              f"{theta[1]:+.3f} ± {se[1]:.3f}   {lr:5.1f}")
        rows.append({"candidate": name, "cv_gain": round(float(np.mean(diff)), 5),
                     "cv_gain_se": round(float(np.std(diff) / math.sqrt(len(diff))), 5),
                     "weight": round(float(theta[1]), 4), "weight_se": round(float(se[1]), 4),
                     "likelihood_ratio": round(lr, 2)})
    with open(os.path.join(out, "feature_lab.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
