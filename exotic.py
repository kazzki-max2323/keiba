"""馬券の種類ごとの「値付けのズレ」を測る。

1. 単勝オッズから各馬の勝つ確率 p を出し、1〜3着の並び方の確率を計算する。
   1着は p の比、2着は p^λ2 の比、3着は p^λ3 の比で残りの馬から選ばれるとする。
   λ2・λ3 は集めたレースの実際の着順から決める（λ = 1 だと人気馬が2・3着に来る確率を出しすぎる）。
2. 券種ごとに、全部の組み合わせを確率の大きさでグループに分け、
   「そのグループの組み合わせを全部 100 円ずつ買ったら、いくら戻ったか」（戻り率）を計算する。
   外れた組み合わせのオッズは要らない。当たった組み合わせの払戻金だけで正確に計算できる。
3. 前半の時期と後半の時期で同じ傾向が出るかも見る。

使い方:
    python3 exotic.py 202608040911   # 先に correct_speed.py を実行しておく

出力:
    data/<race_id>/exotic_returns.csv
"""
import collections
import csv
import itertools
import math
import os
import sys

import numpy as np

import netkeiba

MIN_FIELD = 8
SPLIT_DATE = "20250701"
KINDS = ("単勝", "複勝", "ワイド", "馬連", "馬単", "三連複", "三連単")
TAKEOUT = {"単勝": 0.20, "複勝": 0.20, "ワイド": 0.225, "馬連": 0.225, "馬単": 0.25, "三連複": 0.25, "三連単": 0.275}
# 確率のグループの境目（この確率以上〜次の確率未満）
EDGES = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.01]
BOOT = 300
rng = np.random.default_rng(0)


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load(out):
    races = []
    for r in sorted(read(os.path.join(out, "races.csv")), key=lambda r: r["date"]):
        rr = netkeiba.race_runs(r["race_id"])
        if len(rr) < MIN_FIELD or not all(x["odds"] and x["number"] for x in rr):
            continue
        pay = netkeiba.payouts(r["race_id"])
        if not all(k in pay for k in KINDS):
            continue
        inv = np.array([1 / x["odds"] for x in rr])
        finish = [int(x["finish"]) if x["finish"].isdigit() else 99 for x in rr]
        races.append({"race_id": r["race_id"], "date": r["date"], "p": inv / inv.sum(),
                      "number": [x["number"] for x in rr], "order": list(np.argsort(finish, kind="stable")[:3]),
                      "pay": pay})
    return races


def fit_lambda(races):
    """2着・3着の λ を、実際の着順が一番起こりやすくなる値に決める（格子で総当たり）。"""
    def ll(l2, l3):
        total = 0.0
        for r in races:
            p = r["p"]
            a, b, c = r["order"]
            s2, s3 = p ** l2, p ** l3
            total += math.log(p[a]) + math.log(s2[b] / (s2.sum() - s2[a])) \
                + math.log(s3[c] / (s3.sum() - s3[a] - s3[b]))
        return total
    grid = np.round(np.arange(0.5, 1.21, 0.02), 2)
    best2 = max(grid, key=lambda l2: ll(l2, 1.0))
    best3 = max(grid, key=lambda l3: ll(best2, l3))
    return best2, best3, ll(best2, best3) - ll(1.0, 1.0)


def combo_probs(p, l2, l3):
    """券種ごとの {組（馬の位置の組）: 確率}。"""
    n = len(p)
    s2, s3 = p ** l2, p ** l3
    p12 = p[:, None] * s2[None, :] / (s2.sum() - s2[:, None])
    np.fill_diagonal(p12, 0.0)
    rest3 = s3.sum() - s3[:, None, None] - s3[None, :, None]
    p123 = p12[:, :, None] * s3[None, None, :] / rest3
    idx = np.arange(n)
    p123[idx, idx, :] = 0
    p123[:, idx, idx] = 0
    p123[idx, :, idx] = 0
    out = {k: {} for k in KINDS}
    for i in range(n):
        out["単勝"][(i,)] = p[i]
        out["複勝"][(i,)] = p123[i].sum() + p123[:, i].sum() + p123[:, :, i].sum()
    for i, j in itertools.combinations(range(n), 2):
        out["馬連"][(i, j)] = p12[i, j] + p12[j, i]
        out["ワイド"][(i, j)] = sum(p123[a, b, :].sum() + p123[a, :, b].sum() + p123[:, a, b].sum()
                                  for a, b in ((i, j), (j, i)))
        out["馬単"][(i, j)] = p12[i, j]
        out["馬単"][(j, i)] = p12[j, i]
    for c in itertools.combinations(range(n), 3):
        perms = list(itertools.permutations(c))
        out["三連複"][c] = sum(p123[a, b, d] for a, b, d in perms)
        for a, b, d in perms:
            out["三連単"][(a, b, d)] = p123[a, b, d]
    return out


def tally(races, l2, l3):
    """レースごと・券種ごと・確率グループごとの (買った数, 払戻金, 確率の合計, 当たり数)。"""
    nb = len(EDGES) - 1
    stats = {k: np.zeros((len(races), nb, 4)) for k in KINDS}
    for ri, r in enumerate(races):
        probs = combo_probs(r["p"], l2, l3)
        pos = {num: i for i, num in enumerate(r["number"])}
        for k in KINDS:
            paid = {}
            for nums, yen in r["pay"][k]:
                if all(n in pos for n in nums):
                    key = tuple(pos[n] for n in nums)
                    if k not in ("馬単", "三連単"):
                        key = tuple(sorted(key))
                    paid[key] = paid.get(key, 0) + yen
            q = np.array(list(probs[k].values()))
            keys = list(probs[k])
            b = np.clip(np.searchsorted(EDGES, q, side="right") - 1, 0, nb - 1)
            st = stats[k][ri]
            np.add.at(st[:, 0], b, 1)
            np.add.at(st[:, 2], b, q)
            for key, yen in paid.items():
                if key in probs[k]:
                    bi = b[keys.index(key)]
                    st[bi, 1] += yen
                    st[bi, 3] += 1
    return stats


def summarize(stats, mask):
    rows = []
    for k in KINDS:
        s = stats[k][mask]
        tot = s.sum(axis=0)
        for bi in range(len(EDGES) - 1):
            if tot[bi, 0] == 0:
                continue
            ret = tot[bi, 1] / (100 * tot[bi, 0])
            # レースを選び直して戻り率のぶれを見る
            boots = []
            for _ in range(BOOT):
                pick = rng.integers(0, len(s), len(s))
                t = s[pick, bi].sum(axis=0)
                boots.append(t[1] / (100 * t[0]) if t[0] else 0.0)
            rows.append({"kind": k, "prob_from": EDGES[bi], "prob_to": min(EDGES[bi + 1], 1.0),
                         "bets": int(tot[bi, 0]), "hits": int(tot[bi, 3]), "expected_hits": round(tot[bi, 2], 1),
                         "return": round(ret, 3), "return_se": round(float(np.std(boots)), 3),
                         "breakeven_gap": round(ret - 1, 3)})
    return rows


def main(race_id):
    out = os.path.join("data", race_id)
    races = load(out)
    early = np.array([r["date"] < SPLIT_DATE for r in races])
    l2, l3, gain = fit_lambda([r for r, e in zip(races, early) if e])  # 前半の時期だけで決める
    print(f"レース数 {len(races)}（前半 {early.sum()} / 後半 {(~early).sum()}）  λ2 = {l2:.2f}  λ3 = {l3:.2f}"
          f"（λ=1 より対数尤度 +{gain:.1f}）")
    stats = tally(races, l2, l3)
    all_rows = []
    for label, mask in (("全期間", np.ones(len(races), bool)), ("前半", early), ("後半", ~early)):
        rows = summarize(stats, mask)
        for r in rows:
            r["period"] = label
        all_rows += rows
    with open(os.path.join(out, "exotic_returns.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["period"] + [k for k in all_rows[0] if k != "period"])
        w.writeheader()
        w.writerows(all_rows)

    print("\n全部の組み合わせを買った場合の戻り率（券種ごと）")
    for k in KINDS:
        t = stats[k].sum(axis=(0, 1))
        print(f"  {k:4s} {t[1] / (100 * t[0]):.1%}（控除率から見た理論値 {1 - TAKEOUT[k]:.1%}）")
    print("\n確率グループごとの戻り率（全期間 / 前半 / 後半、±はぶれ）")
    by = collections.defaultdict(dict)
    for r in all_rows:
        by[(r["kind"], r["prob_from"])][r["period"]] = r
    for (k, lo), d in by.items():
        a = d["全期間"]
        parts = "  ".join(f"{d[p]['return']:6.1%}±{d[p]['return_se']:5.1%}" if p in d else " " * 14
                          for p in ("全期間", "前半", "後半"))
        print(f"  {k:4s} 確率 {lo:<7g}〜 買った数 {a['bets']:>8d} 当たり {a['hits']:>5d}（予想 {a['expected_hits']:>7.1f}）  {parts}")


if __name__ == "__main__":
    main(sys.argv[1])
