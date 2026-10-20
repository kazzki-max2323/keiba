"""期間中の JRA の芝の開催日（競馬場コード:YYYYMMDD）を書き出す。

JRA の開催は土日と祝日の月曜、年始・年末が中心なので、その日付だけ netkeiba のレース一覧を見る。

使い方:
    python3 calendar_days.py 20250101 20261020 > days.txt
    python3 correct_speed.py 202608040911 @days.txt
"""
import sys

import numpy as np

import netkeiba


def candidates(start, end):
    d = np.datetime64(f"{start[:4]}-{start[4:6]}-{start[6:]}")
    last = np.datetime64(f"{end[:4]}-{end[4:6]}-{end[6:]}")
    while d <= last:
        s = str(d).replace("-", "")
        weekday = int((d.astype("datetime64[D]").view("int64") - 4) % 7)  # 0 = 月曜
        if weekday in (0, 5, 6) or s[4:] in ("0104", "0105", "0106", "1228"):
            yield s
        d += np.timedelta64(1, "D")


def main(start, end):
    for date in candidates(start, end):
        venues = sorted({rid[4:6] for rid, surface, _ in netkeiba.day_races(date)
                         if surface == "芝" and rid[4:6] in netkeiba.VENUES})
        for v in venues:
            print(f"{v}:{date}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
