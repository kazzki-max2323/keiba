"""三連単の各組み合わせの確率を出す。

勝つ確率: 強さ = (単勝オッズの確率)^α × (モデルの確率)^β（α・β は blend.py の結果）。
1〜3着の並び: 2着は 勝つ確率^λ2、3着は 勝つ確率^λ3 の比で残りの馬から選ばれるとする（λ は exotic.py と同じ決め方）。
「適正オッズ」は 1 ÷ 確率。実際のオッズがこれを大きく上回らない限り、割に合わない。

使い方:
    python3 trifecta.py 202608040911        # 上位 30 通り
    python3 trifecta.py 202608040911 50     # 上位 50 通り

出力:
    data/<race_id>/trifecta.csv
"""
import csv
import os
import sys

import numpy as np

import exotic
import netkeiba


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main(race_id, top):
    out = os.path.join("data", race_id)
    weight = read(os.path.join(out, "blend_weight.csv"))[0]
    a, b = float(weight["alpha"]), float(weight["beta"])
    model = {p["name"]: max(float(p["win"]), 1e-4) for p in read(os.path.join(out, "probabilities.csv"))}
    odds, status = netkeiba.win_odds(race_id)
    names = sorted(odds, key=lambda n: odds[n])
    inv = np.array([1 / odds[n] for n in names])
    market = inv / inv.sum()
    s = a * np.log(market) + b * np.log([model.get(n, 1e-4) for n in names])
    p = np.exp(s - s.max())
    p /= p.sum()

    l2, l3, _ = exotic.fit_lambda(exotic.load(out))
    probs = exotic.combo_probs(p, l2, l3)["三連単"]
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    rows, total = [], 0.0
    for rank, (key, q) in enumerate(ranked[:top], 1):
        total += q
        rows.append({"rank": rank, "first": names[key[0]], "second": names[key[1]], "third": names[key[2]],
                     "prob": round(float(q), 5), "cumulative": round(float(total), 4),
                     "fair_odds": round(1 / q, 1)})
    with open(os.path.join(out, "trifecta.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cum = np.cumsum([q for _, q in ranked])
    print(f"オッズ: {status}（yoso = 予想オッズ）  出走 {len(names)} 頭  組み合わせ {len(ranked)} 通り  λ2={l2:.2f} λ3={l3:.2f}")
    print("勝つ確率: " + "  ".join(f"{n} {q:.1%}" for n, q in zip(names, p) if q >= 0.02))
    for share in (0.1, 0.25, 0.5):
        print(f"  確率の合計が {share:.0%} になるのは上位 {int(np.searchsorted(cum, share)) + 1} 通り")
    for r in rows:
        print(f"{r['rank']:>3d}. {r['first']} → {r['second']} → {r['third']}  "
              f"{r['prob']:.2%}（累計 {r['cumulative']:.1%}）  適正オッズ {r['fair_odds']}倍")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
