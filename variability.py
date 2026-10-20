"""調子のブレ（1走ごとの速さのばらつき）を馬ごとに出す。

クラス差・年齢差・馬場差を除いた rel（そのコースの基準より 1000m あたり何秒遅いか）の、
馬ごとの標準偏差を出す。走った回数が少ないと標準偏差は偶然で大きく上下するので、
「全馬が同じブレ」だった場合に偶然で出るばらつきと、実際のばらつきを比べ、
馬ごとの差として信じてよい分だけを残して全馬の平均の方へ寄せる（対数の分散で計算する）。

使い方:
    python3 variability.py 202608040911   # 先に correct_speed.py を実行しておく

出力:
    data/<race_id>/variability.csv
        sd_raw:     その馬の実際の標準偏差（秒/1000m）
        sd:         平均の方へ寄せた後の標準偏差（予想に使う）
        factor:     全馬の平均と比べたブレの大きさ（1.2 なら平均より 2 割ぶれやすい）
        bad_runs:   自分の平均より 1000m あたり 1 秒以上遅かった走の数（大敗）
"""
import collections
import csv
import math
import os
import sys

import numpy as np


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main(race_id):
    out = os.path.join("data", race_id)
    base = {b["course"]: float(b["baseline_sec_per_1000m"]) for b in read(os.path.join(out, "baseline.csv"))}
    effects = {(e["kind"], e["key"]): float(e["sec_per_1000m"]) for e in read(os.path.join(out, "effects.csv"))}
    horses = collections.defaultdict(list)
    for r in read(os.path.join(out, "runs_corrected.csv")):
        rel = (3600 / float(r["corrected_kmh"]) - base[r["course"]]
               - effects.get(("age", r["age"]), 0.0) - effects.get(("level", r["level"]), 0.0))
        horses[r["name"]].append(rel)

    measured = {n: np.array(v) for n, v in horses.items() if len(v) >= 3}
    log_var = {n: math.log(np.var(v, ddof=1)) for n, v in measured.items()}
    # 正規分布なら log(標本分散) の偶然のばらつきはおよそ 2/(n-1)
    noise = {n: 2 / (len(v) - 1) for n, v in measured.items()}
    center = float(np.mean(list(log_var.values())))
    tau2 = max(float(np.var(list(log_var.values()))) - float(np.mean(list(noise.values()))), 0.0)

    rows = []
    for name, v in horses.items():
        if name in measured:
            trust = tau2 / (tau2 + noise[name])
            log_sd2 = trust * log_var[name] + (1 - trust) * center
            sd_raw = round(float(np.std(v, ddof=1)), 3)
        else:
            trust, log_sd2, sd_raw = 0.0, center, ""
        sd = math.sqrt(math.exp(log_sd2))
        rows.append({
            "name": name, "runs": len(v), "sd_raw": sd_raw, "trust": round(trust, 2),
            "sd": round(sd, 3), "factor": round(sd / math.sqrt(math.exp(center)), 3),
            "bad_runs": int(np.sum(v - np.mean(v) > 1.0)),
        })
    rows.sort(key=lambda r: r["sd"])
    with open(os.path.join(out, "variability.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"全馬の平均のブレ ±{math.sqrt(math.exp(center)):.2f}秒/1000m  "
          f"log分散の実際のばらつき {np.var(list(log_var.values())):.2f} / 偶然で出る分 {np.mean(list(noise.values())):.2f}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1])
