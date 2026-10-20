"""レースの出走（登録）馬の全成績を集め、時速を計算して CSV に保存する。

使い方:
    python3 fetch_race.py 202608040911

出力:
    data/<race_id>/runs.csv     各馬の全レース（1行 = 1走）
    data/<race_id>/summary.csv  馬ごとの時速まとめ
"""
import csv
import os
import statistics
import sys

import netkeiba


def main(race_id):
    title, horses = netkeiba.entries(race_id)
    print(title, f"{len(horses)}頭")
    out = os.path.join("data", race_id)
    os.makedirs(out, exist_ok=True)

    all_runs, summary = [], []
    for h in horses:
        runs = netkeiba.horse_results(h["horse_id"])
        for r in runs:
            all_runs.append({"horse_id": h["horse_id"], "name": h["name"], "entry_jockey": h["jockey"], **r})
        turf = [r for r in runs if r["surface"] == "芝"]
        long_turf = [r for r in turf if r["distance"] >= 2000]
        ups = [r["last600_kmh"] for r in turf if r["last600_kmh"]]
        summary.append({
            "name": h["name"], "jockey": h["jockey"], "runs": len(runs),
            "turf_runs": len(turf),
            "turf_avg_kmh": round(statistics.mean(r["speed_kmh"] for r in turf), 2) if turf else "",
            "turf2000plus_avg_kmh": round(statistics.mean(r["speed_kmh"] for r in long_turf), 2) if long_turf else "",
            "turf2000plus_best_kmh": max((r["speed_kmh"] for r in long_turf), default=""),
            "longest_distance": max((r["distance"] for r in runs), default=""),
            "last600_avg_kmh": round(statistics.mean(ups), 2) if ups else "",
            "last600_best_kmh": max(ups, default=""),
        })
        print(f"  {h['name']}: {len(runs)}走")

    with open(os.path.join(out, "runs.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_runs[0]))
        w.writeheader()
        w.writerows(all_runs)
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main(sys.argv[1])
