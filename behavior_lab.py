"""市場をつくる人の行動の癖から、単勝オッズに上乗せできる情報を探す。

案1 馬券の種類どうしの食い違い（直近 1000 レース、exotic_ev.py で集めた確定オッズ）
    umatan_gap: 馬単のオッズから逆算した「1着になる確率」と、単勝の確率の対数の差
    trio_gap:   三連複のオッズから逆算した「3着以内の確率」と、単勝の確率から計算した3着以内の確率の対数の差
    単勝の市場とは買う人が違うので、どちらかだけが知っている情報があれば、差が結果を当てるはず。
案2 場面ごとの値付けの狂い（集めた全レース）
    最終レース・G1 の日の他のレース・重い馬場・月曜開催で、オッズの確率の信じ方（傾き）が変わるか。
    馬番 7 番・1 番が買われすぎていないか。

どれも 強さ = Σ 重み × 材料 として、1〜3着の順番の確率（Plackett-Luce）を当てはめ、
日付順に5つに分けた答え合わせで「単勝オッズ（と人気薄の買われすぎの補正）だけ」と比べる。

使い方:
    python3 behavior_lab.py 202608040911

出力:
    data/<race_id>/behavior_lab.csv
"""
import collections
import csv
import math
import os
import sys

import numpy as np

import exotic
import exotic_ev
import feature_lab
import netkeiba

FOLDS = 5


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def base_columns(p):
    lp = np.log(p)
    return [lp, lp ** 2]  # 2乗の項で、人気薄が買われすぎる癖の曲がり方を吸収する


def cross_pool(races, l2, l3):
    """案1 の材料。"""
    data = {}
    for r in races:
        tri = netkeiba.final_odds(r["race_id"], "三連複")
        uma = netkeiba.final_odds(r["race_id"], "馬単")
        if not tri or not uma:
            continue
        num = r["number"]
        pos = {n: i for i, n in enumerate(num)}
        n = len(num)
        win_u = np.zeros(n)
        for (a, _), o in uma.items():
            if a in pos:
                win_u[pos[a]] += 1 / o
        top3_t = np.zeros(n)
        for combo, o in tri.items():
            for c in combo:
                if c in pos:
                    top3_t[pos[c]] += 1 / o
        if np.any(win_u == 0) or np.any(top3_t == 0):
            continue
        win_u /= win_u.sum()
        top3_t = 3 * top3_t / top3_t.sum()
        model_top3 = np.array([v for v in exotic.combo_probs(r["p"], l2, l3)["複勝"].values()])
        X = np.column_stack(base_columns(r["p"]) + [np.log(win_u) - np.log(r["p"]),
                                                    np.log(top3_t) - np.log(model_top3)])
        data[r["race_id"]] = (r["date"], X, list(r["order"]))
    return data, ["umatan_gap", "trio_gap"]


def contexts(races, out):
    """案2 の材料。"""
    info = {r["race_id"]: r for r in read(os.path.join(out, "races.csv"))}
    g1_days = {(r["date"], r["venue"]) for r in info.values() if r["level"] == "G1"}
    last_race = collections.defaultdict(int)
    for rid, r in info.items():
        key = (r["date"], r["venue"])
        last_race[key] = max(last_race[key], int(rid[-2:]))
    data = {}
    for r in races:
        i = info[r["race_id"]]
        key = (i["date"], i["venue"])
        weekday = int((np.datetime64(f"{i['date'][:4]}-{i['date'][4:6]}-{i['date'][6:]}").astype("int64") - 4) % 7)
        lp = np.log(r["p"])
        flags = {
            "final_race": float(int(r["race_id"][-2:]) == last_race[key]),
            "g1_day_other": float(key in g1_days and i["level"] != "G1"),
            "heavy_going": float(i["going"] in ("重", "不良")),
            "monday": float(weekday == 0),
        }
        num = np.array(r["number"])
        X = np.column_stack(base_columns(r["p"]) + [lp * flags[k] for k in flags]
                            + [(num == 7).astype(float), (num == 1).astype(float)])
        data[r["race_id"]] = (r["date"], X, list(r["order"]))
    return data, [f"slope×{k}" for k in ("final_race", "g1_day_other", "heavy_going", "monday")] + ["number_7", "number_1"]


def evaluate(data, names, label):
    ids = sorted(data, key=lambda k: data[k][0])
    folds = np.array_split(np.arange(len(ids)), FOLDS)
    all_data = [data[k][1:] for k in ids]

    def cv(cols):
        lls = []
        for test in folds:
            tset = {ids[i] for i in test}
            theta, _, _ = feature_lab.fit([data[k][1:] for k in ids if k not in tset], cols)
            lls += [feature_lab.loglik(theta, [(data[k][1][:, cols], data[k][2])])[0] for k in ids if k in tset]
        return np.array(lls)

    base_cols = [0, 1]
    base = cv(base_cols)
    _, _, ll0 = feature_lab.fit(all_data, base_cols)
    print(f"\n{label}  レース数 {len(ids)}")
    print(f"  {'材料':22s} {'答え合わせでの上乗せ':>18s}   {'重み（全レース）':>14s}   尤度比")
    rows = []
    for j, name in enumerate(names):
        cols = base_cols + [2 + j]
        diff = cv(cols) - base
        theta, se, ll = feature_lab.fit(all_data, cols)
        lr = 2 * (ll - ll0)
        print(f"  {name:22s} {np.mean(diff):+.4f} ± {np.std(diff) / math.sqrt(len(diff)):.4f}   "
              f"{theta[-1]:+.3f} ± {se[-1]:.3f}   {lr:5.1f}")
        rows.append({"test": label, "feature": name, "races": len(ids),
                     "cv_gain": round(float(np.mean(diff)), 5),
                     "cv_gain_se": round(float(np.std(diff) / math.sqrt(len(diff))), 5),
                     "weight": round(float(theta[-1]), 4), "weight_se": round(float(se[-1]), 4),
                     "likelihood_ratio": round(lr, 2)})
    return rows


def main(race_id):
    out = os.path.join("data", race_id)
    races = exotic.load(out)
    train, test = exotic_ev.split(races)
    l2, l3, _ = exotic.fit_lambda(train)
    rows = evaluate(*cross_pool(test, l2, l3), "案1 馬券の種類どうしの食い違い")
    rows += evaluate(*contexts(races, out), "案2 場面ごとの値付けの狂い")
    with open(os.path.join(out, "behavior_lab.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
