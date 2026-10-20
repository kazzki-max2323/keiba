"""netkeiba からレースの出走（登録）馬の全成績を集め、時速を計算して CSV に保存する。

使い方:
    python3 fetch_race.py 202608040911

出力:
    data/<race_id>/runs.csv     各馬の全レース（1行 = 1走）
    data/<race_id>/summary.csv  馬ごとの時速まとめ
"""
import csv
import html
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
WAIT = 3  # サイトに負担をかけないよう、1リクエストごとに待つ秒数


def get(url, encoding):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                body = res.read()
            break
        except urllib.error.HTTPError:
            if attempt == 4:
                raise
            time.sleep(5 * 2 ** attempt)  # 一時的に断られたら少し待って再試行
    time.sleep(WAIT)
    return body.decode(encoding, "ignore")


def text(cell):
    return html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()


def entries(race_id):
    s = get(f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}", "utf-8")
    title = re.search(r"<title>([^<|]*)", s).group(1).strip()
    horses = []
    for tr in re.findall(r'<tr class="HorseList.*?</tr>', s, re.S):
        h = re.search(r'horse/(\d+)"[^>]*>([^<]*)', tr)
        j = re.search(r'jockey/result/recent/\d+/"[^>]*>\s*([^<]*)', tr)
        horses.append({"horse_id": h.group(1), "name": h.group(2).strip(),
                       "jockey": j.group(1).strip() if j else ""})
    return title, horses


def to_seconds(t):
    m = re.fullmatch(r"(\d+):(\d{2}\.\d)", t)
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def results(horse_id):
    s = get(f"https://db.netkeiba.com/horse/result/{horse_id}/", "euc-jp")
    table = re.search(r'<table[^>]*db_h_race_results.*?</table>', s, re.S)
    if not table:
        return []
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table.group(0), re.S)[1:]:
        c = [text(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(c) < 28:
            continue
        course = re.fullmatch(r"(芝|ダ|障)(\d+)", c[14])
        sec = to_seconds(c[18])
        if not course or sec is None:  # 中止・除外など
            continue
        dist = int(course.group(2))
        up = float(c[27]) if re.fullmatch(r"\d+\.\d", c[27]) else None
        rows.append({
            "date": c[0], "venue": c[1], "race": c[4], "field_size": c[6],
            "finish": c[11], "jockey": c[12], "weight_carried": c[13],
            "surface": course.group(1), "distance": dist, "going": c[16],
            "time": c[18], "margin": c[19], "last600": c[27] if up else "",
            "speed_kmh": round(dist / sec * 3.6, 2),
            "last600_kmh": round(600 / up * 3.6, 2) if up else "",
        })
    return rows


def main(race_id):
    title, horses = entries(race_id)
    print(title, f"{len(horses)}頭")
    out = os.path.join("data", race_id)
    os.makedirs(out, exist_ok=True)

    all_runs, summary = [], []
    for h in horses:
        runs = results(h["horse_id"])
        for r in runs:
            all_runs.append({"horse_id": h["horse_id"], "name": h["name"], **r})
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
