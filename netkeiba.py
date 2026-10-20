"""netkeiba のページ取得と読み取り。

過去レースのページは内容が変わらないので data/cache に保存し、2回目からはサイトに取りに行かない。
"""
import hashlib
import html
import os
import re
import time
import urllib.error
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
WAIT = 3  # サイトに負担をかけないよう、1リクエストごとに待つ秒数
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")

# race_id の5〜6桁目 → 競馬場（JRA のみ）
VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
          "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def get(url, encoding, cache=False):
    path = os.path.join(CACHE, hashlib.md5(url.encode()).hexdigest() + ".html")
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read().decode(encoding, "ignore")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as res:
                body = res.read()
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(5 * 2 ** attempt)  # 一時的に断られたら少し待って再試行
    time.sleep(WAIT)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "wb") as f:
            f.write(body)
    return body.decode(encoding, "ignore")


def text(cell):
    return html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()


def to_seconds(t):
    m = re.fullmatch(r"(\d+):(\d{2}\.\d)", t)
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def entries(race_id):
    """出馬表（登録馬）の一覧。"""
    s = get(f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}", "utf-8")
    title = re.search(r"<title>([^<|]*)", s).group(1).strip()
    horses = []
    for tr in re.findall(r'<tr class="HorseList.*?</tr>', s, re.S):
        h = re.search(r'horse/(\d+)"[^>]*>([^<]*)', tr)
        j = re.search(r'jockey/result/recent/\d+/"[^>]*>\s*([^<]*)', tr)
        horses.append({"horse_id": h.group(1), "name": h.group(2).strip(),
                       "jockey": j.group(1).strip() if j else ""})
    return title, horses


def horse_results(horse_id):
    """馬の全成績。中止・除外などタイムのない走は除く。"""
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
        if not course or sec is None:
            continue
        rid = re.search(r"/race/(\d{12})/", tr)
        dist = int(course.group(2))
        up = float(c[27]) if re.fullmatch(r"\d+\.\d", c[27]) else None
        rows.append({
            "race_id": rid.group(1) if rid else "",
            "date": c[0], "venue": c[1], "race": c[4], "field_size": c[6],
            "finish": c[11], "jockey": c[12], "weight_carried": c[13],
            "surface": course.group(1), "distance": dist, "going": c[16],
            "time": c[18], "margin": c[19], "last600": c[27] if up else "",
            "speed_kmh": round(dist / sec * 3.6, 2),
            "last600_kmh": round(600 / up * 3.6, 2) if up else "",
        })
    return rows


def day_races(date):
    """その日に行われた全レースの (race_id, 芝/ダ/障, 距離)。date は YYYYMMDD。"""
    s = get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}", "utf-8", cache=True)
    races = []
    for li in re.findall(r'<li class="RaceList_DataItem.*?</li>', s, re.S):
        rid = re.search(r"race_id=(\d{12})", li)
        course = re.search(r"(芝|ダ|障)(\d{3,4})m", li)
        if rid and course:
            races.append((rid.group(1), course.group(1), int(course.group(2))))
    return races


def race_level(name, info):
    """クラスと年齢区分。name はレース名、info は「3歳オープン」などの条件欄。"""
    for g in ("GIII", "GII", "GI"):
        if f"({g})" in name or f"(J.{g})" in name:
            level = {"GI": "G1", "GII": "G2", "GIII": "G3"}[g]
            break
    else:
        level = next((v for k, v in [("新馬", "新馬"), ("未勝利", "未勝利"), ("1勝", "1勝"), ("500万", "1勝"),
                                     ("2勝", "2勝"), ("1000万", "2勝"), ("3勝", "3勝"), ("1600万", "3勝"),
                                     ("オープン", "OP")] if k in info), "OP")
    age = "2歳" if "2歳" in info else "3歳以上" if "以上" in info else "3歳"
    return level, age


def race_result(race_id):
    """レース結果ページから条件と勝ち時計を読む。"""
    s = get(f"https://db.netkeiba.com/race/{race_id}/", "euc-jp", cache=True)
    head = re.search(r'<dl class="racedata.*?</dl>', s, re.S)
    small = re.search(r'<p class="smalltxt">(.*?)</p>', s, re.S)
    table = re.search(r'<table[^>]*race_table_01.*?</table>', s, re.S)
    if not (head and small and table):
        return None
    head_txt = text(head.group(0))
    name = text(re.search(r"<h1>(.*?)</h1>", head.group(0), re.S).group(1))
    course = re.search(r"(芝|ダ|障)\S*?(\d{3,4})m", head_txt)
    going = re.search(r"(?:芝|ダート)\s*:\s*(良|稍重|重|不良)", head_txt)
    first = re.findall(r"<tr[^>]*>(.*?)</tr>", table.group(0), re.S)[1]
    cells = [text(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", first, re.S)]
    sec = to_seconds(cells[7])
    if not course or sec is None:
        return None
    level, age = race_level(name, text(small.group(1)))
    return {"race_id": race_id, "name": name, "surface": course.group(1),
            "distance": int(course.group(2)), "going": going.group(1) if going else "",
            "level": level, "age": age, "win_time": sec}
