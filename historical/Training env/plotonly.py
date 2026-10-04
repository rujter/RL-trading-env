import numpy as np
import matplotlib.pyplot as plt


def plot_results(seed, results_path="eval_results.npz", out_dir=".", show=False):
    def label(ax, y, text):
        ax.text(0.02, y, text, transform=ax.transAxes, color="green", fontsize=10, va="top")

    def mark_episodes(ax):
        top = ax.get_ylim()[1]
        for es, t in zip(episode_starts, tickers):
            ax.axvline(x=es, color="gray", linestyle="--", alpha=0.4)
            ax.text(es + 5, top, t, fontsize=9, va="top", color="gray")

    data = np.load(results_path)
    ypoints = data["ypoints"]
    buy_x, buy_y = data["buy_x"], data["buy_y"]
    sell_x, sell_y = data["sell_x"], data["sell_y"]
    hold_x, hold_y = data["hold_x"], data["hold_y"]
    profit = data["profitabilitypercentage_arr"]
    episode_starts = data["episode_starts"]
    n = len(episode_starts)
    tickers = [str(t) for t in data["tickers"]] if "tickers" in data.files else [str(i) for i in range(n)]

    # per-ticker buy-and-hold profit
    bounds = list(episode_starts) + [len(ypoints) + 1]
    hold_profit = np.array([
        (ypoints[bounds[i + 1] - 2] - ypoints[bounds[i] - 1]) / ypoints[bounds[i] - 1] * 100
        for i in range(n)
    ])

    # metrics
    avg_profitability = profit.mean()
    sharpe = avg_profitability / (profit.std() + 1e-8)
    T = 0.0
    downside_dev = np.sqrt(np.mean(np.minimum(0, profit - T) ** 2))
    sortino = (avg_profitability - T) / (downside_dev + 1e-8)
    avg_hold = hold_profit.mean()

    # orders plot
    fig1, ax = plt.subplots(figsize=(20, 6))
    ax.plot(buy_x, buy_y, linestyle="", marker="o", ms=4, mec="r", mfc="r", label="buy")
    ax.plot(sell_x, sell_y, linestyle="", marker="o", ms=4, mec="g", mfc="g", label="sell")
    ax.plot(hold_x, hold_y, linestyle="", marker="o", ms=1, mec="b", mfc="b", label="hold")
    mark_episodes(ax)
    label(ax, 0.5, f"numhold: {hold_x.size}")
    label(ax, 0.45, f"buynum: {buy_x.size}")
    label(ax, 0.40, f"numsell: {sell_x.size}")
    ax.legend()
    fig1.savefig(f"{out_dir}/orders{seed}.png")

    # profit vs hold plot
    idx = np.arange(n)
    width = 0.38
    fig2, ax = plt.subplots(figsize=(14, 6))
    ax.bar(idx - width / 2, profit, width, label="real profit (agent)")
    ax.bar(idx + width / 2, hold_profit, width, label="profit by holding")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{i}\n{t}" for i, t in zip(idx, tickers)])
    ax.set_xlabel("ticker number")
    ax.set_ylabel("profit (%)")
    ax.set_title("Agent profit vs buy-and-hold per ticker")
    label(ax, 0.98, f"avg agent: {avg_profitability:.2f}%   avg hold: {avg_hold:.2f}%")
    label(ax, 0.93, f"sharpe: {sharpe:.2f}   sortino: {sortino:.2f}")
    ax.legend(loc="upper right")
    fig2.savefig(f"{out_dir}/p vs h{seed}.png")

    if show:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)

    return {"avg_profit": avg_profitability, "avg_hold": avg_hold, "sharpe": sharpe, "sortino": sortino}