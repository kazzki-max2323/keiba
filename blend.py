"""モデルの確率と単勝オッズの確率を組み合わせる。

各馬の強さを

    強さ = (オッズの確率)^α × (モデルの確率)^β

とし、勝つ確率は強さの比、2着以下も残りの馬の強さの比で順に決まるとする（Plackett-Luce）。
α・β は過去の菊花賞の1〜3着の順番が一番起こりやすくなる値を、格子状に総当たりして探す。
勝ち馬だけ（9頭）では少なすぎるので、3着までの順番を使う。

当たり具合は、その年を除いたデータで α・β を決めて予想させて測る。比べる相手は
    オッズそのまま / オッズだけで α を合わせたもの（人気薄が過大評価されがちな癖を直したもの）。
β がはっきり 0 より大きければ、モデルはオッズにない情報を持っている。

使い方:
    python3 blend.py 202608040911   # 先に simulate.py を実行しておく

出力:
    data/<race_id>/blend_weight.csv     α・β と答え合わせ
    data/<race_id>/probabilities_blend.csv  今年の組み合わせ後の確率（オッズは netkeiba から取る）
"""
import collections
import csv
import math
import os
import sys

import numpy as np

import netkeiba

TOP = 3
EPS = 1e-4
SIMS = 100000
ALPHAS = np.round(np.arange(0.3, 1.81, 0.02), 2)
BETAS = np.round(np.arange(-0.5, 1.51, 0.02), 2)
rng = np.random.default_rng(0)


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def races_from_backtest(rows):
    by_year = collections.defaultdict(list)
    for r in rows:
        if r["market_win"] and float(r["market_win"]) > 0:
            by_year[r["year"]].append(r)
    races = {}
    for year, rs in by_year.items():
        lm = np.log(np.maximum([float(r["market_win"]) for r in rs], EPS))
        lo = np.log(np.maximum([float(r["model_win"]) for r in rs], EPS))
        order = [i for i, r in sorted(enumerate(rs), key=lambda x: int(x[1]["finish"]) if x[1]["finish"].isdigit() else 99)]
        races[year] = (lm, lo, order[:TOP])
    return races


def loglik(races, a, b, top=TOP):
    """3着までの順番が起こる確率の対数（全レースの合計）。"""
    total = 0.0
    for lm, lo, order in races.values():
        s = a * lm + b * lo
        alive = np.ones(len(s), bool)
        for i in order[:top]:
            total += s[i] - np.log(np.sum(np.exp(s[alive])))
            alive[i] = False
    return total


def best(races, betas=BETAS):
    grid = [(loglik(races, a, b), a, b) for a in ALPHAS for b in betas]
    return max(grid)


def main(race_id):
    out = os.path.join("data", race_id)
    races = races_from_backtest(read(os.path.join(out, "backtest.csv")))

    # 1. その年を除いて α・β を決め、その年を予想する
    scores = {"オッズそのまま": [], "オッズ（α調整）": [], "オッズ＋モデル": []}
    wins = {k: [] for k in scores}
    for year in sorted(races):
        train = {y: r for y, r in races.items() if y != year}
        test = {year: races[year]}
        _, a0, _ = best(train, betas=[0.0])
        _, a1, b1 = best(train)
        for k, (a, b) in (("オッズそのまま", (1.0, 0.0)), ("オッズ（α調整）", (a0, 0.0)), ("オッズ＋モデル", (a1, b1))):
            scores[k].append(loglik(test, a, b))
            wins[k].append(loglik(test, a, b, top=1))
    print("答え合わせ（その年を除いて重みを決めて予想、大きいほど良い）")
    print(f"  {'':16s} 3着までの順番  勝ち馬に付けた確率（平均）")
    for k in scores:
        print(f"  {k:16s} {np.mean(scores[k]):+.3f}        {math.exp(np.mean(wins[k])):.1%}")

    # 2. 全年で α・β を決める。β = 0 と比べてどれだけ良くなるか（尤度比）
    ll1, a, b = best(races)
    ll0, a0, _ = best(races, betas=[0.0])
    lr = 2 * (ll1 - ll0)
    print(f"\n全年で決めた重み: α = {a:.2f}（オッズ）  β = {b:.2f}（モデル）")
    print(f"モデルを足した効果: 尤度比 {lr:.2f}（3.84 を超えれば偶然とは言いにくい）")
    with open(os.path.join(out, "blend_weight.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "beta", "alpha_market_only", "likelihood_ratio"]
                   + [f"cv_top3_{k}" for k in scores] + [f"cv_win_{k}" for k in scores])
        w.writerow([a, b, a0, round(lr, 3)] + [round(float(np.mean(v)), 4) for v in scores.values()]
                   + [round(math.exp(np.mean(v)), 4) for v in wins.values()])

    # 3. 今年の確率
    odds, status = netkeiba.win_odds(race_id)
    probs = read(os.path.join(out, "probabilities.csv"))
    names = [p["name"] for p in probs if p["name"] in odds]
    model = np.array([max(float(p["win"]), EPS) for p in probs if p["name"] in odds])
    inv = np.array([1 / odds[n] for n in names])
    market = inv / inv.sum()
    s = a * np.log(market) + b * np.log(model)
    # 強さに従って着順を何度も作る（Gumbel の性質: 強さの対数 + Gumbel の乱数の大きい順が Plackett-Luce になる）
    rank = (-(s + rng.gumbel(size=(SIMS, len(s))))).argsort(axis=1).argsort(axis=1)
    win, top3 = (rank == 0).mean(axis=0), (rank < 3).mean(axis=0)
    rows = sorted(({"name": n, "odds": odds[n], "market_win": round(float(m), 4), "model_win": round(float(o), 4),
                    "blend_win": round(float(w), 4), "blend_top3": round(float(t), 4),
                    "expected_return": round(float(w) * odds[n], 2)}
                   for n, m, o, w, t in zip(names, market, model, win, top3)), key=lambda r: -r["blend_win"])
    with open(os.path.join(out, "probabilities_blend.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n今年のオッズ: {status}（yoso = netkeiba の予想オッズ）")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1])
