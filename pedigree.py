"""血統からスタミナの目安を出す。

父と母の父それぞれについて、netkeiba の産駒成績（累計）にある「平均距離(芝)」を取る。
これは「その種牡馬の子どもが芝で勝ったレースの平均距離」で、長いほど長距離向きの血統と言える。
勝利数が少ない種牡馬の値はぶれるので、芝の勝利数に応じて平均的な値（PRIOR_DISTANCE）の方へ寄せる。

使い方:
    python3 pedigree.py 202608040911   # 先に fetch_race.py を実行しておく

出力:
    data/<race_id>/pedigree.csv
"""
import csv
import os
import re
import sys

import netkeiba

PRIOR_DISTANCE = 1800  # 種牡馬の芝平均勝ち距離のおおよその平均（JRA の芝レースの中心的な距離）
PRIOR_WINS = 20  # この勝利数ぶん、平均的な値を混ぜる


def parents(horse_id):
    """(父ID, 父名, 母父ID, 母父名)。"""
    s = netkeiba.get(f"https://db.netkeiba.com/horse/ped/{horse_id}/", "euc-jp", cache=True)
    table = re.search(r'<table[^>]*blood_table.*?</table>', s, re.S).group(0)
    cells = [(int(span), re.search(r'horse/(\w{10})/">\s*([^<]+)', body))
             for span, body in re.findall(r'<td[^>]*rowspan="(\d+)"[^>]*>(.*?)</td>', table, re.S)]
    sire = cells[0][1]
    dam_index = next(i for i, (span, _) in enumerate(cells) if span == 16 and i > 0)
    damsire = cells[dam_index + 1][1]
    clean = lambda m: netkeiba.text(m.group(2)).replace("\xa0", " ").strip()
    return sire.group(1), clean(sire), damsire.group(1), clean(damsire)


def sire_distance(sire_id):
    """種牡馬の産駒の (芝の勝利数, 芝の平均距離)。"""
    s = netkeiba.get(f"https://db.netkeiba.com/horse/sire/{sire_id}/", "euc-jp", cache=True)
    table = re.search(r'<table[^>]*nk_tb_common.*?</table>', s, re.S)
    if not table:
        return 0, None
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table.group(0), re.S):
        c = [netkeiba.text(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if c and c[0] == "累計" and len(c) > 19:
            wins = int(c[13].replace(",", "") or 0)
            dist = float(c[19].replace(",", "")) if c[19] not in ("", "0.0") else None
            return wins, dist
    return 0, None


def shrink(wins, dist):
    if dist is None:
        return PRIOR_DISTANCE
    return (wins * dist + PRIOR_WINS * PRIOR_DISTANCE) / (wins + PRIOR_WINS)


def pedigree(horse_id):
    sire_id, sire, damsire_id, damsire = parents(horse_id)
    sw, sd = sire_distance(sire_id)
    dw, dd = sire_distance(damsire_id)
    return {
        "sire": sire, "sire_turf_wins": sw, "sire_avg_turf_distance": sd or "",
        "damsire": damsire, "damsire_turf_wins": dw, "damsire_avg_turf_distance": dd or "",
        # 父を2、母の父を1の重みで合わせた血統の距離（勝利数の少ない種牡馬は平均へ寄せた後の値）
        "pedigree_distance": round((2 * shrink(sw, sd) + shrink(dw, dd)) / 3),
    }


def main(race_id):
    out = os.path.join("data", race_id)
    with open(os.path.join(out, "runs.csv"), encoding="utf-8-sig") as f:
        horses = {r["horse_id"]: r["name"] for r in csv.DictReader(f)}
    rows = []
    for horse_id, name in horses.items():
        rows.append({"name": name, **pedigree(horse_id)})
        print(rows[-1], flush=True)
    rows.sort(key=lambda r: -r["pedigree_distance"])
    with open(os.path.join(out, "pedigree.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
