"""コース・馬場の補正をかけた時速を出す。

各馬が走った日に、同じ競馬場で行われた芝レースの勝ち時計をすべて集め、
1000mあたりのタイム（ペース）を次の3つの和として推定する。

    ペース = コース・距離の基準 + その日の馬場差 + クラス差 + 年齢区分の差

馬のタイムから「その日の馬場差」だけを差し引くと、平均的な馬場で走った場合の時速になる。
さらに、そのコース・距離の基準（3歳以上1勝クラスの平均的な勝ち時計）と比べた差を
「基準比」として出す。基準比はコースや距離が違っても比べやすい。

使い方:
    python3 correct_speed.py 202608040911   # 先に fetch_race.py を実行しておく

出力:
    data/<race_id>/runs_corrected.csv     1走ごとの補正後の値
    data/<race_id>/summary_corrected.csv  馬ごとのまとめ
    data/<race_id>/baseline.csv           コース・距離ごとの基準タイム
    data/<race_id>/day_variants.csv       開催日ごとの馬場差
"""
import collections
import csv
import os
import statistics
import sys

import netkeiba

REF_LEVEL, REF_AGE = "1勝", "3歳以上"


def collect(runs):
    """馬が走った日・競馬場の芝レースの結果をすべて集める。"""
    days = sorted({(r["race_id"][4:6], r["date"].replace("/", "")) for r in runs
                   if r["surface"] == "芝" and r["race_id"][4:6] in netkeiba.VENUES})
    races = []
    for i, (venue, date) in enumerate(days, 1):
        targets = [rid for rid, surface, _ in netkeiba.day_races(date)
                   if surface == "芝" and rid[4:6] == venue]
        for rid in targets:
            r = netkeiba.race_result(rid)
            if r and r["surface"] == "芝":
                r["venue"], r["date"] = netkeiba.VENUES[venue], date
                races.append(r)
        print(f"  [{i}/{len(days)}] {date} {netkeiba.VENUES[venue]}: 芝{len(targets)}レース", flush=True)
    return races


def fit(races, iterations=200):
    """ペースを基準・馬場差・クラス差・年齢差に分ける（交互に平均を取って合わせる）。"""
    for r in races:
        r["pace"] = r["win_time"] / r["distance"] * 1000
        r["course"] = (r["venue"], r["inout"], r["distance"])
        r["day"] = (r["date"], r["venue"], r["going"])
    base = collections.defaultdict(float)
    day = collections.defaultdict(float)
    level = collections.defaultdict(float)
    age = collections.defaultdict(float)

    def update(effects, key, others):
        groups = collections.defaultdict(list)
        for r in races:
            groups[r[key]].append(r["pace"] - sum(e[r[k]] for e, k in others))
        for k, v in groups.items():
            effects[k] = statistics.mean(v)

    for _ in range(iterations):
        update(base, "course", [(day, "day"), (level, "level"), (age, "age")])
        update(level, "level", [(base, "course"), (day, "day"), (age, "age")])
        update(age, "age", [(base, "course"), (day, "day"), (level, "level")])
        update(day, "day", [(base, "course"), (level, "level"), (age, "age")])
        # 基準は「3歳以上1勝クラス・その競馬場の平均的な馬場」とする
        for k in list(base):
            base[k] += level[REF_LEVEL] + age[REF_AGE]
        lv, ag = level[REF_LEVEL], age[REF_AGE]
        for k in level:
            level[k] -= lv
        for k in age:
            age[k] -= ag
        per_venue = collections.defaultdict(list)
        for k, v in day.items():
            per_venue[k[1]].append(v)
        shift = {v: statistics.mean(x) for v, x in per_venue.items()}
        for k in day:
            day[k] -= shift[k[1]]
        for k in base:
            base[k] += shift[k[0]]
    return base, day, level, age


def main(race_id):
    out = os.path.join("data", race_id)
    with open(os.path.join(out, "runs.csv"), encoding="utf-8-sig") as f:
        runs = list(csv.DictReader(f))
    for r in runs:
        r["distance"] = int(r["distance"])

    print("同じ日の芝レースを集めています…")
    races = collect(runs)
    base, day, level, age = fit(races)
    counts = collections.Counter(r["course"] for r in races)
    by_id = {r["race_id"]: r for r in races}

    corrected = []
    for r in runs:
        venue = netkeiba.VENUES.get(r["race_id"][4:6])
        sec = netkeiba.to_seconds(r["time"])
        going = {"稍": "稍重", "不": "不良"}.get(r["going"], r["going"])
        key_day = (r["date"].replace("/", ""), venue, going)
        own = by_id.get(r["race_id"])  # 内回り・外回りは、その馬が走ったレース自体から判定する
        key_course = own["course"] if own else (venue, "", r["distance"])
        if r["surface"] != "芝" or key_day not in day or key_course not in base:
            continue
        pace = sec / r["distance"] * 1000
        adj_pace = pace - day[key_day]
        corrected.append({
            "name": r["name"], "date": r["date"], "race": r["race"], "course": f"{venue}芝{key_course[1]}{r['distance']}",
            "going": r["going"], "finish": r["finish"], "jockey": r["jockey"], "time": r["time"],
            "speed_kmh": r["speed_kmh"],
            "day_variant_sec_per_1000m": round(day[key_day], 2),
            "corrected_kmh": round(3600 / adj_pace, 2),
            "baseline_kmh": round(3600 / base[key_course], 2),
            "vs_baseline_kmh": round(3600 / adj_pace - 3600 / base[key_course], 2),
            "baseline_races": counts[key_course],
        })

    by_horse = collections.defaultdict(list)
    for c in corrected:
        by_horse[c["name"]].append(c)
    summary = []
    for name, cs in by_horse.items():
        raw = [float(c["speed_kmh"]) for c in cs]
        fixed = [c["corrected_kmh"] for c in cs]
        vs = [c["vs_baseline_kmh"] for c in cs]
        recent = sorted(cs, key=lambda c: c["date"], reverse=True)[:3]
        summary.append({
            "name": name, "turf_runs": len(cs),
            "raw_avg_kmh": round(statistics.mean(raw), 2),
            "corrected_avg_kmh": round(statistics.mean(fixed), 2),
            "correction_kmh": round(statistics.mean(fixed) - statistics.mean(raw), 2),
            "vs_baseline_avg_kmh": round(statistics.mean(vs), 2),
            "vs_baseline_best_kmh": max(vs),
            "vs_baseline_recent3_kmh": round(statistics.mean(c["vs_baseline_kmh"] for c in recent), 2),
        })
    summary.sort(key=lambda s: -s["vs_baseline_recent3_kmh"])

    def write(name, rows):
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("runs_corrected.csv", corrected)
    write("summary_corrected.csv", summary)
    write("baseline.csv", [{"course": f"{v}芝{io}{d}", "races": counts[(v, io, d)],
                            "baseline_sec_per_1000m": round(p, 2), "baseline_kmh": round(3600 / p, 2)}
                           for (v, io, d), p in sorted(base.items())])
    write("day_variants.csv", [{"date": d, "venue": v, "going": g, "sec_per_1000m": round(x, 2)}
                               for (d, v, g), x in sorted(day.items())])
    print("クラス差（秒/1000m、マイナスほど速い）:", {k: round(v, 2) for k, v in sorted(level.items(), key=lambda x: x[1])})
    print("年齢差（秒/1000m）:", {k: round(v, 2) for k, v in age.items()})
    print(f"集めたレース: {len(races)}  補正できた走: {len(corrected)}/{sum(r['surface'] == '芝' for r in runs)}")


if __name__ == "__main__":
    main(sys.argv[1])
