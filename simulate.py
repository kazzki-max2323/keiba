"""レースを何万回もシミュレーションして、各馬の勝つ確率・3着以内の確率を出す。

1回のレースでの各馬の走り（rel、小さいほど速い）を

    走り = c × 今の速さ + p × 血統の距離(100m単位) + 調子のブレの大きさ × その日のぶれ

とする。
- c（今の速さをどれだけ信じるか）と p（血統の効き目）は確定できないので、
  過去の菊花賞から求めた値と誤差の幅で、シミュレーションのたびに選び直す。
- その日のぶれは、過去の菊花賞で実際に出た「予想からのずれ」（大敗も含む）から無作為に取る。
- 調子のブレの大きさは variability.csv の factor（平均的な馬 = 1）。
- jockey_adj.csv があれば、騎手の乗り替わりによる上乗せ（秒/1000m）をそのまま足す（今年の予想のみ）。

過去の出走馬の「今の速さ」も今年と同じく馬場差を補正してあるので、c はそのまま使う。

過去の菊花賞で、その年を除いたデータで作ったモデルに予想させ、実際の結果と比べる（答え合わせ）。
比べる相手は、単勝オッズから出した確率と、全馬同じ確率。

使い方:
    python3 simulate.py 202608040911   # 先に stamina.py・variability.py・calibrate_pedigree.py を実行しておく

出力:
    data/<race_id>/probabilities.csv   今年の各馬の確率
    data/<race_id>/backtest.csv        過去の菊花賞での答え合わせ
"""
import collections
import csv
import math
import os
import sys

import numpy as np

import calibrate_pedigree

SIMS = 20000
PED_CENTER = 1800  # calibrate_pedigree.fit と同じ中心
rng = np.random.default_rng(0)


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def model(rows):
    """過去の出走馬から c・p とその誤差、その日のぶれの見本を作る。"""
    coef, se, _, _, resid = calibrate_pedigree.fit(rows)
    return {"p": coef[0], "p_se": se[0], "c": 1 + coef[1], "c_se": se[1],
            "noise": resid - np.median(resid)}


def simulate(level, ped_units, factor, m, offset=0.0):
    """各馬の (勝つ確率, 3着以内の確率, 平均着順)。offset は秒/1000m でそのまま足す上乗せ（騎手など）。"""
    n = len(level)
    c = rng.normal(m["c"], m["c_se"], (SIMS, 1))
    p = rng.normal(m["p"], m["p_se"], (SIMS, 1))
    noise = rng.choice(m["noise"], (SIMS, n)) * factor
    perf = c * level + p * ped_units + offset + noise
    rank = perf.argsort(axis=1).argsort(axis=1)
    return (rank == 0).mean(axis=0), (rank < 3).mean(axis=0), rank.mean(axis=0) + 1


def backtest(rows):
    out = []
    for year in sorted({r["year"] for r in rows}):
        m = model([r for r in rows if r["year"] != year])
        field = [r for r in rows if r["year"] == year]
        level = np.array([r["level"] for r in field])
        ped = np.array([(r["pedigree_distance"] - PED_CENTER) / 100 for r in field])
        win, top3, _ = simulate(level, ped, np.ones(len(field)), m)
        inv = np.array([1 / r["odds"] if r["odds"] else 0.0 for r in field])
        market = inv / inv.sum()
        for r, w, t, mk in zip(field, win, top3, market):
            out.append({"year": year, "name": r["name"], "finish": r["finish"], "odds": r["odds"],
                        "model_win": round(float(w), 4), "model_top3": round(float(t), 4),
                        "market_win": round(float(mk), 4), "uniform_win": round(1 / len(field), 4)})
    return out


def score(bt):
    """勝ち馬に付けた確率の対数の平均（大きいほど良い）と、3着以内の当たり具合（ブライアスコア、小さいほど良い）。"""
    winners = [r for r in bt if r["finish"] == "1"]
    eps = 1e-4
    res = {k: float(np.mean([math.log(max(r[k], eps)) for r in winners]))
           for k in ("model_win", "market_win", "uniform_win")}
    # 市場と半々に混ぜた確率（対数で平均してから合計 1 に直す）
    by_year = collections.defaultdict(list)
    for r in bt:
        by_year[r["year"]].append(r)
    blend = []
    for rs in by_year.values():
        raw = np.array([math.sqrt(max(r["model_win"], eps) * max(r["market_win"], eps)) for r in rs])
        raw /= raw.sum()
        blend += [math.log(p) for r, p in zip(rs, raw) if r["finish"] == "1"]
    res["blend_win"] = float(np.mean(blend))
    res["model_top3_brier"] = float(np.mean([(r["model_top3"] - (r["finish"] in ("1", "2", "3"))) ** 2 for r in bt]))
    for k, key in (("model_rank_corr", "model_win"), ("market_rank_corr", "market_win")):
        corrs = []
        for rs in by_year.values():
            rs = [r for r in rs if r["finish"].isdigit()]
            pred = np.argsort(np.argsort([-r[key] for r in rs]))
            actual = np.argsort(np.argsort([int(r["finish"]) for r in rs]))
            corrs.append(np.corrcoef(pred, actual)[0, 1])
        res[k] = float(np.mean(corrs))
    res["uniform_top3_brier"] = float(np.mean([(3 / 18 - (r["finish"] in ("1", "2", "3"))) ** 2 for r in bt]))
    return res, winners


def main(race_id):
    out = os.path.join("data", race_id)
    rows = read(os.path.join(out, "pedigree_calibration.csv"))
    for r in rows:
        r["level"], r["rel_kikka"] = float(r["level"]), float(r["rel_kikka"])
        r["pedigree_distance"] = float(r["pedigree_distance"])
        r["odds"] = float(r["odds"]) if r["odds"] else None

    # 1. 答え合わせ
    bt = backtest(rows)
    with open(os.path.join(out, "backtest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(bt[0]))
        w.writeheader()
        w.writerows(bt)
    res, winners = score(bt)
    print("答え合わせ（過去9回の菊花賞、勝ち馬に付けた確率の対数の平均、大きいほど良い）")
    for k in ("uniform_win", "model_win", "market_win", "blend_win"):
        print(f"  {k:12s} {res[k]:+.3f}  （平均すると勝ち馬に {math.exp(res[k]):.1%} を付けた）")
    print(f"  3着以内のブライアスコア: モデル {res['model_top3_brier']:.4f} / 全馬同じ {res['uniform_top3_brier']:.4f}")
    print("  着順との順位相関（年ごと、1 に近いほど着順を当てている）:")
    print(f"    モデル {res['model_rank_corr']:+.3f}   オッズ {res['market_rank_corr']:+.3f}")
    for r in winners:
        print(f"    {r['year']} {r['name']}: モデル {r['model_win']:.1%}  オッズ {r['market_win']:.1%}")

    # 2. 今年の予想
    m = model(rows)
    st = {s["name"]: s for s in read(os.path.join(out, "stamina.csv"))}
    var = {v["name"]: float(v["factor"]) for v in read(os.path.join(out, "variability.csv"))}
    target_base = None
    for b in read(os.path.join(out, "baseline.csv")):
        if b["course"] == calibrate_pedigree.TARGET_COURSE:
            target_base = float(b["baseline_sec_per_1000m"])
    names = list(st)
    # stamina.csv の km/h の差を rel（秒/1000m）に戻す
    level = np.array([3600 / (3600 / target_base + float(st[n]["level_vs_baseline_kmh"])) - target_base
                      for n in names])
    ped = np.array([(float(st[n]["pedigree_distance"]) - PED_CENTER) / 100 for n in names])
    factor = np.array([var.get(n, 1.0) for n in names])
    # 騎手の乗り替わりによる上乗せ（jockey.py）。今の速さを出した走の騎手と、本番の騎手の腕の差
    jpath = os.path.join(out, "jockey_adj.csv")
    jadj = {j["name"]: float(j["jockey_adj"]) for j in read(jpath)} if os.path.exists(jpath) else {}
    offset = np.array([jadj.get(n, 0.0) for n in names])
    print(f"\n今の速さを信じる度合い c = {m['c']:.2f} ± {m['c_se']:.2f}  血統 p = {m['p']:+.3f} ± {m['p_se']:.3f}")
    print(f"その日のぶれ（菊花賞）: 標準偏差 ±{np.std(m['noise']):.2f}秒/1000m（3000mで ±{3 * np.std(m['noise']):.1f}秒）"
          f"  1000mあたり1秒以上の大敗 {np.mean(m['noise'] > 1.0):.0%}")

    win, top3, avg = simulate(level, ped, factor, m, offset)
    result = sorted(({"name": n, "win": round(float(w), 4), "top3": round(float(t), 4),
                      "avg_finish": round(float(a), 1), "fair_odds": round(1 / w, 1) if w > 0 else "",
                      "level_rel": round(float(l), 3), "factor": f, "jockey_adj": float(o)}
                     for n, w, t, a, l, f, o in zip(names, win, top3, avg, level, factor, offset)),
                    key=lambda r: -r["win"])
    with open(os.path.join(out, "probabilities.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(result[0]))
        w.writeheader()
        w.writerows(result)
    for r in result:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1])
