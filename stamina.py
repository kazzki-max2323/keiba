"""スタミナ（距離が延びたときに速さをどれだけ保てるか）を馬ごとに推定し、目標の距離での時速を予想する。

1走ごとに「補正後のペースが、そのコースの基準より 1000m あたり何秒速かったか」（rel）を出す。
若いうちの走りは、年齢区分の差（成長分）を差し引いておく。

- 今の速さ: rel の重み付き平均。近い時期のレースほど重く数える（半年で重さが半分）。
- スタミナ: クラス差も差し引いた rel を距離（対数）に対して直線で当てはめた傾き。
  長い距離はキャリアの後半・上のクラスで走ることが多いので、クラス差を残すと
  「成長して強くなった」ことまで「距離が延びて強くなった」ように見えてしまう。

走った回数が少ない・距離の幅が狭い馬の傾きはぶれやすいので、
全馬の傾きのばらつきから「どの程度信じてよいか」を見積もり、平均的な馬（傾き 0）の方へ寄せる。

使い方:
    python3 stamina.py 202608040911 京都芝外3000 3歳   # 先に correct_speed.py を実行しておく

出力:
    data/<race_id>/stamina.csv
"""
import csv
import datetime
import math
import os
import re
import sys

import numpy as np

HALF_LIFE_DAYS = 180
REF_DISTANCE = 2000  # 傾きの基準にする距離
MIN_TAU2 = 0.05  # 馬ごとの傾きの差の下限（0 にすると全馬が同じスタミナになる）


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def kmh(pace):
    return 3600 / pace


def to_time(sec):
    sec = round(sec, 1)
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def main(race_id, target_course, target_age):
    out = os.path.join("data", race_id)
    runs = read(os.path.join(out, "runs_corrected.csv"))
    base = {b["course"]: float(b["baseline_sec_per_1000m"]) for b in read(os.path.join(out, "baseline.csv"))}
    effects = {(e["kind"], e["key"]): float(e["sec_per_1000m"]) for e in read(os.path.join(out, "effects.csv"))}
    target_distance = int(re.search(r"(\d{4})$", target_course).group(1))
    target_base = base[target_course]
    x_target = math.log(target_distance / REF_DISTANCE)
    today = max(datetime.date.fromisoformat(r["date"].replace("/", "-")) for r in runs)

    horses = {}
    for r in runs:
        rel = kmh(float(r["corrected_kmh"])) - effects.get(("age", r["age"]), 0.0) - base[r["course"]]
        days = (today - datetime.date.fromisoformat(r["date"].replace("/", "-"))).days
        horses.setdefault(r["name"], []).append({
            "x": math.log(int(r["distance"]) / REF_DISTANCE), "distance": int(r["distance"]),
            "rel": rel, "rel_class": rel - effects.get(("level", r["level"]), 0.0),
            "w": 0.5 ** (days / HALF_LIFE_DAYS),
        })

    # 1. 馬ごとに、クラス差を除いた rel を距離に対して直線で当てはめる
    fits, ssr, dof = {}, 0.0, 0
    for name, rows in horses.items():
        x = np.array([r["x"] for r in rows])
        y = np.array([r["rel_class"] for r in rows])
        f = {"rows": rows, "slope": 0.0, "sxx": 0.0}
        if len(rows) >= 3 and np.ptp(x) > 0:
            f["slope"], icept = np.polyfit(x, y, 1)
            f["sxx"] = float(np.sum((x - x.mean()) ** 2))
            ssr += float(np.sum((y - icept - f["slope"] * x) ** 2))
            dof += len(rows) - 2
        fits[name] = f
    sigma2 = ssr / dof  # 1走ごとのぶれ（全馬共通とみなす）

    # 2. 傾きをどこまで信じるか（全馬の傾きのばらつき − 測定のぶれ）
    measured = [f for f in fits.values() if f["sxx"] > 0]
    for f in measured:
        f["var_b"] = sigma2 / f["sxx"]
    slopes = np.array([f["slope"] for f in measured])
    tau2 = max(float(np.var(slopes)) - float(np.mean([f["var_b"] for f in measured])), MIN_TAU2)

    rows_out = []
    for name, f in fits.items():
        rows = f["rows"]
        trust = tau2 / (tau2 + f["var_b"]) if f["sxx"] > 0 else 0.0
        b = f["slope"] * trust
        var_b = tau2 * (1 - trust)
        w = np.array([r["w"] for r in rows])
        xm = float(np.average([r["x"] for r in rows], weights=w))
        level = float(np.average([r["rel"] for r in rows], weights=w))  # 今の速さ（距離 xm での rel）
        var_level = sigma2 * float(np.sum(w ** 2) / np.sum(w) ** 2)
        dx = x_target - xm
        rel_target = level + b * dx
        sd = math.sqrt(var_level + dx ** 2 * var_b)
        pace = target_base + rel_target + effects[("age", target_age)]
        rows_out.append({
            "name": name, "turf_runs": len(rows), "longest": max(r["distance"] for r in rows),
            "avg_distance": round(REF_DISTANCE * math.exp(xm)),
            # 今の速さ: 目標コースの基準と比べて何km/h 速いか（距離を延ばす前）
            "level_vs_baseline_kmh": round(kmh(target_base + level) - kmh(target_base), 2),
            # スタミナ: 距離が 1.5 倍になったとき、平均的な馬と比べて何km/h 得するか（プラスほど長距離向き）
            "stamina_kmh": round(kmh(target_base + b * math.log(1.5)) - kmh(target_base), 2),
            "stamina_raw_kmh": round(kmh(target_base + f["slope"] * math.log(1.5)) - kmh(target_base), 2),
            "trust": round(trust, 2),
            "pred_vs_baseline_kmh": round(kmh(target_base + rel_target) - kmh(target_base), 2),
            "pred_kmh": round(kmh(pace), 2),
            "pred_kmh_low": round(kmh(pace + sd), 2), "pred_kmh_high": round(kmh(pace - sd), 2),
            "pred_time": to_time(pace * target_distance / 1000),
        })
    rows_out.sort(key=lambda r: -r["pred_kmh"])

    with open(os.path.join(out, "stamina.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"{target_course} 基準 {kmh(target_base):.2f}km/h  1走のぶれ ±{math.sqrt(sigma2):.2f}秒/1000m  "
          f"傾き: 全馬平均 {slopes.mean():+.2f} 観測のばらつき ±{slopes.std():.2f} "
          f"測定誤差 ±{math.sqrt(np.mean([f['var_b'] for f in measured])):.2f} → 本当の差 ±{math.sqrt(tau2):.2f}")
    for r in rows_out:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
