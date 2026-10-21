"""「確率 × オッズ」が大きい組み合わせだけを買った場合の戻り率を、過去レースの確定オッズで確かめる。

確率は exotic.py と同じく単勝オッズから計算する（2・3着の補正 λ は、答え合わせに使うレースより前のレースで決める）。
オッズは確定オッズ（締め切り後の値）。実際に買う時点のオッズとは少し違うので、結果は少し甘めの見積もりになる。

使い方:
    python3 exotic_ev.py 202608040911 collect   # 直近 N レースの三連複・馬単の確定オッズを集める（時間がかかる）
    python3 exotic_ev.py 202608040911           # 答え合わせ

出力:
    data/<race_id>/exotic_ev.csv
"""
import csv
import os
import sys

import numpy as np

import exotic
import netkeiba

N_RACES = 1000
KINDS = ("三連複", "馬単")
THRESHOLDS = (1.0, 1.2, 1.5, 2.0)
MIN_PROBS = (0.0, 0.003, 0.01)
BOOT = 500
rng = np.random.default_rng(0)


def split(races):
    races = sorted(races, key=lambda r: (r["date"], r["race_id"]))
    return races[:-N_RACES], races[-N_RACES:]


def main(race_id, mode):
    out = os.path.join("data", race_id)
    train, test = split(exotic.load(out))
    if mode == "collect":
        for i, r in enumerate(test, 1):
            for k in KINDS:
                netkeiba.final_odds(r["race_id"], k)
            if i % 50 == 0:
                print(f"  [{i}/{len(test)}] {r['date']}", flush=True)
        return

    l2, l3, _ = exotic.fit_lambda(train)
    print(f"確率の計算に使うレース {len(train)}（λ2 = {l2:.2f}  λ3 = {l3:.2f}）  答え合わせ {len(test)} レース "
          f"{test[0]['date']}〜{test[-1]['date']}")
    # レースごと・条件ごとの (買った数, 払戻, 確率の合計, 当たり数)
    conds = [(k, t, m) for k in KINDS for t in THRESHOLDS for m in MIN_PROBS]
    per_race = np.zeros((len(test), len(conds), 4))
    used = np.zeros(len(test), bool)
    for ri, r in enumerate(test):
        probs = exotic.combo_probs(r["p"], l2, l3)
        number = r["number"]
        for k in KINDS:
            odds = netkeiba.final_odds(r["race_id"], k)
            if not odds:
                continue
            used[ri] = True
            hits = {nums: yen for nums, yen in r["pay"][k]}
            for key, q in probs[k].items():
                nums = tuple(number[i] for i in key)
                if k != "馬単":
                    nums = tuple(sorted(nums))
                o = odds.get(nums)
                if o is None:
                    continue
                ev = q * o
                for ci, (ck, t, m) in enumerate(conds):
                    if ck == k and ev > t and q >= m:
                        per_race[ri, ci] += (1, hits.get(nums, 0), q, nums in hits)
    per_race = per_race[used]
    half = len(per_race) // 2
    rows = []
    print(f"\nオッズが取れたレース {used.sum()}")
    print(f"{'券種':4s} {'確率×オッズ':>8s} {'確率の下限':>6s} {'買った数':>7s} {'当たり':>5s}（予想）  戻り率±ぶれ     前半    後半")
    for ci, (k, t, m) in enumerate(conds):
        s = per_race[:, ci]
        tot = s.sum(axis=0)
        if tot[0] == 0:
            continue
        ret = tot[1] / (100 * tot[0])
        boots = []
        for _ in range(BOOT):
            b = s[rng.integers(0, len(s), len(s))].sum(axis=0)
            boots.append(b[1] / (100 * b[0]) if b[0] else 0.0)
        early, late = s[:half].sum(axis=0), s[half:].sum(axis=0)
        r_early = early[1] / (100 * early[0]) if early[0] else float("nan")
        r_late = late[1] / (100 * late[0]) if late[0] else float("nan")
        print(f"{k:4s} {t:>8.1f}超 {m:>8.3f} {int(tot[0]):>8d} {int(tot[3]):>5d}（{tot[2]:6.1f}）  "
              f"{ret:6.1%}±{np.std(boots):5.1%}  {r_early:6.1%}  {r_late:6.1%}")
        rows.append({"kind": k, "ev_over": t, "min_prob": m, "bets": int(tot[0]), "hits": int(tot[3]),
                     "expected_hits": round(tot[2], 1), "return": round(ret, 4),
                     "return_se": round(float(np.std(boots)), 4),
                     "return_first_half": round(r_early, 4), "return_second_half": round(r_late, 4)})
    with open(os.path.join(out, "exotic_ev.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
