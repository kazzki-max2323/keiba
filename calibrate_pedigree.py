"""過去の菊花賞の出走馬を使って、「血統の距離」が3000mでの走りにどれだけ効くかを測る。

各馬について
    今の速さ (level): 菊花賞より前の芝の直近4走のうち、良かった2走の rel の平均
    本番 (rel_k):     菊花賞での rel
を出し、本番で「今の速さ」からどれだけ上下したか (y = rel_k - level) を血統の距離で説明する。
rel は「そのコース・距離の基準ペースより 1000m あたり何秒遅いか」（マイナスほど速い）。

古い年は出走前のレースの馬場差まで集めると時間がかかりすぎるので、ここでは馬場差を補正しない。
馬場の良し悪しは血統と関係がないので、ぶれは増えるが偏りは生まない。
年ごとの馬場の速さの違い（芝の高速化など）は、年ごとの定数を入れて取り除く。

使い方:
    python3 calibrate_pedigree.py 202608040911   # 先に correct_speed.py と stamina.py を実行しておく

出力:
    data/<race_id>/pedigree_calibration.csv   過去の出走馬ごとの値
    data/<race_id>/pedigree_weight.csv        当てはめた係数
"""
import csv
import os
import sys

import numpy as np

import netkeiba
import pedigree

# 京都で行われた菊花賞（2021・2022年は阪神開催なので除く）
PAST_RACES = ["201508040711", "201608040711", "201708040711", "201808040711", "201908040711",
              "202008040611", "202308020711", "202408050611", "202508030911"]
TARGET_COURSE = "京都芝外3000"
RECENT, BEST = 4, 2
PED_UNIT = 100  # 血統の距離はこの m 単位で効き目を表す


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def course_base(base, venue, distance):
    """内回り・外回りが分からない走は、ある方の基準（両方あれば平均）を使う。"""
    found = [base[k] for k in (f"{venue}芝{distance}", f"{venue}芝外{distance}") if k in base]
    return sum(found) / len(found) if found else None


def rel(pace, birth_year, date, effects, base_pace):
    age = int(date[:4]) - birth_year
    age_effect = effects.get(("age", "2歳" if age == 2 else "3歳" if age == 3 else "3歳以上"), 0.0)
    return pace - age_effect - base_pace


def collect(base, effects):
    rows = []
    for race_id in PAST_RACES:
        runners = netkeiba.race_runners(race_id)
        print(f"{race_id}: {len(runners)}頭", flush=True)
        for horse_id, name, finish, sec, odds in runners:
            birth = int(horse_id[:4])
            history = netkeiba.horse_results(horse_id, cache=True)
            own = next((r for r in history if r["race_id"] == race_id), None)
            if own is None:
                continue
            prior = []
            for r in history:
                venue = netkeiba.VENUES.get(r["race_id"][4:6])
                if r["date"] >= own["date"] or r["surface"] != "芝" or venue is None:
                    continue
                b = course_base(base, venue, r["distance"])
                if b is not None:
                    pace = netkeiba.to_seconds(r["time"]) / r["distance"] * 1000
                    prior.append((r["date"], rel(pace, birth, r["date"], effects, b), r["distance"]))
            prior.sort(reverse=True)
            if len(prior) < BEST:
                continue
            best = sorted(prior[:RECENT], key=lambda p: p[1])[:BEST]
            rows.append({
                "year": own["date"][:4], "race_id": race_id, "name": name, "finish": finish, "odds": odds,
                "level": round(float(np.mean([p[1] for p in best])), 3),
                "rel_kikka": round(rel(sec / 3, birth, own["date"], effects, base[TARGET_COURSE]), 3),
                "longest_before": max(p[2] for p in prior),
                **pedigree.pedigree(horse_id),
            })
    return rows


def fit(rows):
    """y = rel_k - level を、血統の距離・今の速さ・年ごとの定数で当てはめる（外れ値に強い重み付き）。"""
    years = sorted({r["year"] for r in rows})
    y = np.array([r["rel_kikka"] - r["level"] for r in rows])
    ped = np.array([(r["pedigree_distance"] - 1800) / PED_UNIT for r in rows])
    level = np.array([r["level"] for r in rows])
    X = np.column_stack([ped, level] + [np.array([r["year"] == yr for r in rows], float) for yr in years])
    w = np.ones(len(y))
    for _ in range(50):  # 大敗した馬に引っ張られないよう、残差の大きい馬の重みを下げる（Huber）
        W = np.sqrt(w)
        coef = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
        resid = y - X @ coef
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        w = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(resid), 1e-9))
    sigma2 = np.sum(w * resid ** 2) / (len(y) - X.shape[1])
    cov = sigma2 * np.linalg.inv(X.T @ (w[:, None] * X))
    return coef, np.sqrt(np.diag(cov)), years, scale, resid


def main(race_id):
    out = os.path.join("data", race_id)
    base = {b["course"]: float(b["baseline_sec_per_1000m"]) for b in read(os.path.join(out, "baseline.csv"))}
    effects = {(e["kind"], e["key"]): float(e["sec_per_1000m"]) for e in read(os.path.join(out, "effects.csv"))}
    rows = collect(base, effects)
    with open(os.path.join(out, "pedigree_calibration.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    coef, se, years, scale, _ = fit(rows)
    names = ["pedigree_per_100m", "level"] + [f"year_{y}" for y in years]
    with open(os.path.join(out, "pedigree_weight.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["term", "sec_per_1000m", "se"])
        for n, c, s in zip(names, coef, se):
            w.writerow([n, round(float(c), 4), round(float(s), 4)])
    print(f"使えた馬: {len(rows)}頭  1頭ごとのぶれ ±{scale:.2f}秒/1000m")
    print(f"血統の距離 +100m あたり: {coef[0]:+.3f} ± {se[0]:.3f} 秒/1000m（マイナスほど3000mで速くなる）")
    print(f"今の速さ: {coef[1]:+.3f} ± {se[1]:.3f}（-1 に近いほど、今の速さは本番であてにならない）")


if __name__ == "__main__":
    main(sys.argv[1])
