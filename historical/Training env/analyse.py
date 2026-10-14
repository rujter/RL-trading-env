import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots without opening a window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from metrics import episode_metrics

RNG = np.random.default_rng(0)


def load_run(path):
    rows = []

    with np.load(path) as data:
        starting_cash = float(data["startingcash"])

        for episode_id in np.unique(data["ep_id"]):
            episode = data["ep_id"] == episode_id
            metrics = episode_metrics(
                equity=data["equity"][episode],
                price=data["price"][episode],
                shares=data["shares"][episode],
                orders=data["order"][episode],
                start_cash=starting_cash,
                ppy=config.PERIODS_PER_YEAR,
                cost_bps=config.COST_BPS,
            )
            metrics["ticker"] = str(data["tickers"][episode_id])
            rows.append(metrics)

    return rows


def collect(eval_dir):
    agent_results = []
    random_results = []
    missing_files = 0

    for seed in config.SEEDS:
        for a, b in config.TARGET_CONFIGS:
            path = eval_dir / (
                f"eval_results_{config.MODEL_PREFIX}_s{seed}_a{a:.2f}_b{b:.2f}.npz"
            )

            for result in load_run(path):
                agent_results.append({**result, "seed": seed, "a": a, "b": b})

        random_path = eval_dir / f"eval_results_random_s{seed}.npz"
        if random_path.exists():
            for result in load_run(random_path):
                random_results.append({**result, "seed": seed})
    return pd.DataFrame(agent_results), pd.DataFrame(random_results)


def seed_level(results):
    averages = results.groupby(["seed", "a", "b"], as_index=False).agg({
        "ret": "mean",
        "hold_ret": "mean",
        "exposure": "mean",
        "turnover": "mean",
        "buys": "mean",
        "sells": "mean",
        "holds": "mean",
        "maxdd": "median",
        "hold_maxdd": "median",
        "sharpe": "median",
        "sortino": "median",
        "calmar": "median",
        "omega": "median",
        "beta": "median",
        "alpha": "median",
        "hold_sharpe": "median",
    })
    averages["excess"] = averages["ret"] - averages["hold_ret"]
    return averages


def cell_summary(seed_results, random_return):
    rows = []
    for (a, b), group in seed_results.groupby(["a", "b"]):

        rows.append({
            "a": a,
            "b": b,
            "n_seeds": len(group),
            "ret_med": group["ret"].median(),
            "hold": group["hold_ret"].median(),
            "random": random_return,
            "excess_med": group["excess"].median(),
            "p_beat_hold": (group["excess"] > 0).mean(),
            "sharpe": group["sharpe"].median(),
            "hold_sharpe": group["hold_sharpe"].median(),
            "sortino": group["sortino"].median(),
            "calmar": group["calmar"].median(),
            "omega": group["omega"].median(),
            "maxdd": group["maxdd"].median(),
            "hold_maxdd": group["hold_maxdd"].median(),
            "beta": group["beta"].median(),
            "alpha": group["alpha"].median(),
            "exposure": group["exposure"].median(),
            "turnover": group["turnover"].median(),
            "buys": group["buys"].mean(),
            "sells": group["sells"].mean(),
            "holds": group["holds"].mean(),
        })

    return pd.DataFrame(rows)


def return_label(result):
    return (
        f"Agent: {result.ret_med:.1f}% [{result.ret_lo:.1f}, {result.ret_hi:.1f}]\n"
        f"Hold: {result.hold:.1f}% | Rand: {result.random:.1f}%\n"
        f"Excess: {result.excess_med:+.1f} [{result.excess_lo:.1f}, {result.excess_hi:.1f}]\n"
        f"P(beat): {result.p_beat_hold:.0%} (n={int(result.n_seeds)})\n"
        f"Exp: {result.exposure:.0%} | Turn: {result.turnover:.1f}x\n"
        f"B/S/H: {result.buys:.0f}/{result.sells:.0f}/{result.holds:.0f}"
    )


def risk_label(result):
    return (
        f"Sortino: {result.sortino:.2f}\n"
        f"Sharpe: {result.sharpe:.2f} (hold {result.hold_sharpe:.2f})\n"
        f"Calmar: {result.calmar:.2f} | Omega: {result.omega:.2f}\n"
        f"MaxDD: {result.maxdd:.1f}% (hold {result.hold_maxdd:.1f}%)\n"
        f"Beta: {result.beta:.2f} | Alpha: {result.alpha:.1f}%/yr"
    )


def heatmap(summary, metric, label, title, colorbar_label, cmap, output_path, center=None):
    values = summary.pivot(index="b", columns="a", values=metric)
    values = values.sort_index().sort_index(axis=1)
    labels = pd.DataFrame("", index=values.index, columns=values.columns)

    for _, result in summary.iterrows():
        labels.loc[result["b"], result["a"]] = label(result)

    color_limits = {}
    if center is not None:
        limit = np.nanmax(np.abs(values.to_numpy())) or 1.0
        color_limits = {"center": center, "vmin": -limit, "vmax": limit}

    figure, axis = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        values,
        annot=labels.to_numpy(),
        fmt="",
        cmap=cmap,
        ax=axis,
        xticklabels=[f"{a:.2f}" for a in values.columns],
        yticklabels=[f"{b:.2f}" for b in values.index],
        cbar_kws={"label": colorbar_label},
        annot_kws={"size": 8, "weight": "bold"},
        **color_limits,
    )
    axis.set_title(title, fontsize=13, pad=12)
    axis.set_xlabel("Profit Coefficient (a)")
    axis.set_ylabel("Benchmark Coefficient (b)")
    axis.invert_yaxis()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_equity_curves(eval_dir, out_dir, seed=None, a=None, b=None):
    if seed is None:
        seed = config.SEEDS[0]
    if a is None or b is None:
        a, b = config.TARGET_CONFIGS[0]

    path = eval_dir / (
        f"eval_results_{config.MODEL_PREFIX}_s{seed}_a{a:.2f}_b{b:.2f}.npz"
    )
    if not path.exists():
        return

    with np.load(path) as data:
        starting_cash = float(data["startingcash"])
        episodes = np.unique(data["ep_id"])
        columns = 4
        rows = int(np.ceil(len(episodes) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows))
        axes = axes.ravel()

        for axis, episode_id in zip(axes, episodes):
            episode = data["ep_id"] == episode_id
            prices = data["price"][episode]
            hold_equity = starting_cash * prices / prices[0]

            axis.plot(data["equity"][episode], label="Agent")
            axis.plot(hold_equity, label="Buy & Hold", alpha=0.7)

            orders = data["order"][episode]
            buys = int((orders == 1).sum())
            sells = int((orders == 2).sum())
            holds = len(orders) - buys - sells
            ticker = data["tickers"][episode_id]
            axis.set_title(
                f"{ticker}\nBuy: {buys} | Sell: {sells} | Hold: {holds}",
                fontsize=9,
            )
            axis.legend(fontsize=7)

    for axis in axes[len(episodes):]:
        axis.axis("off")

    figure.suptitle(f"Equity Curves (Seed {seed}, a={a:.2f}, b={b:.2f})")
    figure.tight_layout()
    figure.savefig(out_dir / "equity_curves.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=config.EVAL_DIR, type=Path)
    parser.add_argument("--out", default="eval_plots", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    agent_results, random_results = collect(args.dir)
    seed_results = seed_level(agent_results)

    random_return = np.nan
    if not random_results.empty:
        random_return = random_results.groupby("seed")["ret"].mean().median()

    summary = cell_summary(seed_results, random_return)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.round(2).to_string(index=False))

    heatmap(
        summary,
        metric="excess_med",
        label=return_label,
        title="Agent vs Buy & Hold (Median over seeds, 95% Bootstrap CI)",
        colorbar_label="Median excess return vs hold (%)",
        cmap="RdYlGn",
        output_path=args.out / "heatmap_agent_return.png",
        center=0.0,
    )
    heatmap(
        summary,
        metric="sortino",
        label=risk_label,
        title="Risk Metrics",
        colorbar_label="Median Sortino",
        cmap="YlGnBu",
        output_path=args.out / "heatmap_risk.png",
    )
    plot_equity_curves(args.dir, args.out)


if __name__ == "__main__":
    main()
