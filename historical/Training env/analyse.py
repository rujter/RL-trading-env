import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as C
from metrics import episode_metrics

RNG = np.random.default_rng(0)


def load_run(path):
    d = np.load(path)
    start = float(d["startingcash"])
    rows = []
    
    for e in np.unique(d["ep_id"]):
        m = d["ep_id"] == e
        row = episode_metrics(
            d["equity"][m], d["price"][m], d["shares"][m],
            d["order"][m], start, C.PERIODS_PER_YEAR, C.COST_BPS
        )
        row["ticker"] = str(d["tickers"][e])
        rows.append(row)
        
    return rows


def boot_ci(x, stat=np.median, n=5000, level=95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), (n, len(x)))
    s = stat(x[idx], axis=1)
    alpha = (100 - level) / 2
    return tuple(np.percentile(s, [alpha, 100 - alpha]))


def collect(eval_dir):
    rows, rand_rows, missing = [], [], 0

    for seed in C.SEEDS:
        for a, b in C.TARGET_CONFIGS:
            f = eval_dir / f"eval_results_{C.MODEL_PREFIX}_s{seed}_a{a:.2f}_b{b:.2f}.npz"
            if not f.exists():
                missing += 1
                continue
            for r in load_run(f):
                rows.append({**r, "seed": seed, "a": a, "b": b})

        rand_f = eval_dir / f"eval_results_random_s{seed}.npz"
        if rand_f.exists():
            for r in load_run(rand_f):
                rand_rows.append({**r, "seed": seed})

    if missing:
        print(f"Warning: {missing} evaluation files were missing.")
    if not rows:
        raise SystemExit(f"No result files found in {eval_dir}")

    return pd.DataFrame(rows), pd.DataFrame(rand_rows)


def seed_level(df):
    """Aggregate per (seed, a, b): mean for returns/actions, median for ratios."""
    out = df.groupby(["seed", "a", "b"]).agg(
        ret=("ret", "mean"), hold_ret=("hold_ret", "mean"),
        exposure=("exposure", "mean"), turnover=("turnover", "mean"),
        buys=("buys", "mean"), sells=("sells", "mean"), holds=("holds", "mean"),
        maxdd=("maxdd", "median"), hold_maxdd=("hold_maxdd", "median"),
        sharpe=("sharpe", "median"), sortino=("sortino", "median"),
        calmar=("calmar", "median"), omega=("omega", "median"),
        beta=("beta", "median"), alpha=("alpha", "median"),
        hold_sharpe=("hold_sharpe", "median"),
    ).reset_index()

    out["excess"] = out["ret"] - out["hold_ret"]
    return out


def cell_summary(sl, rand_med):
    rows = []
    for (a, b), g in sl.groupby(["a", "b"]):
        ex_lo, ex_hi = boot_ci(g["excess"])
        rt_lo, rt_hi = boot_ci(g["ret"])
        rows.append({
            "a": a, "b": b, "n_seeds": len(g),
            "ret_med": g["ret"].median(), "ret_lo": rt_lo, "ret_hi": rt_hi,
            "hold": g["hold_ret"].median(), "random": rand_med,
            "excess_med": g["excess"].median(), "excess_lo": ex_lo, "excess_hi": ex_hi,
            "p_beat_hold": (g["excess"] > 0).mean(),
            "sharpe": g["sharpe"].median(), "hold_sharpe": g["hold_sharpe"].median(),
            "sortino": g["sortino"].median(), "calmar": g["calmar"].median(),
            "omega": g["omega"].median(), "maxdd": g["maxdd"].median(),
            "hold_maxdd": g["hold_maxdd"].median(),
            "beta": g["beta"].median(), "alpha": g["alpha"].median(),
            "exposure": g["exposure"].median(), "turnover": g["turnover"].median(),
            "buys": g["buys"].mean(), "sells": g["sells"].mean(), "holds": g["holds"].mean(),
        })
    return pd.DataFrame(rows)


def heatmap(cs, value_col, annot_fn, title, cbar_label, cmap, fname, out_dir, center=None):
    a_vals = sorted(cs["a"].unique())
    b_vals = sorted(cs["b"].unique())

    val_matrix = cs.pivot(index="b", columns="a", values=value_col).reindex(index=b_vals, columns=a_vals)
    
    annot_matrix = pd.DataFrame("", index=b_vals, columns=a_vals, dtype=object)
    for _, r in cs.iterrows():
        annot_matrix.loc[r["b"], r["a"]] = annot_fn(r)

    kw = {}
    if center is not None:
        vmax = np.nanmax(np.abs(val_matrix.values)) or 1.0
        kw = {"center": center, "vmin": -vmax, "vmax": vmax}

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        val_matrix, annot=annot_matrix.values, fmt="", cmap=cmap, ax=ax,
        xticklabels=[f"{a:.2f}" for a in a_vals],
        yticklabels=[f"{b:.2f}" for b in b_vals],
        cbar_kws={"label": cbar_label},
        annot_kws={"size": 8, "weight": "bold"},
        **kw
    )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Profit Coefficient (a)")
    ax.set_ylabel("Benchmark Coefficient (b)")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=200)
    plt.close(fig)


def plot_equity_curves(eval_dir, out_dir, seed=None, a=None, b=None):
    seed = seed if seed is not None else C.SEEDS[0]
    a, b = (a, b) if (a is not None and b is not None) else C.TARGET_CONFIGS[0]

    f = eval_dir / f"eval_results_{C.MODEL_PREFIX}_s{seed}_a{a:.2f}_b{b:.2f}.npz"
    if not f.exists():
        return

    d = np.load(f)
    start = float(d["startingcash"])
    eps = np.unique(d["ep_id"])

    ncols = 4
    nrows = int(np.ceil(len(eps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, e in zip(axes, eps):
        m = d["ep_id"] == e
        hold = start * d["price"][m] / d["price"][m][0]

        ax.plot(d["equity"][m], label="Agent")
        ax.plot(hold, label="Buy & Hold", alpha=0.7)

        o = d["order"][m]
        nb, ns = int((o == 1).sum()), int((o == 2).sum())
        nh = int(len(o) - nb - ns)

        ax.set_title(f"{d['tickers'][e]}\nBuy: {nb} | Sell: {ns} | Hold: {nh}", fontsize=9)
        ax.legend(fontsize=7)

    for ax in axes[len(eps):]:
        ax.axis("off")

    fig.suptitle(f"Equity Curves (Seed {seed}, a={a:.2f}, b={b:.2f})")
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curves.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=C.EVAL_DIR, type=Path)
    p.add_argument("--out", default="eval_plots", type=Path)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df, rdf = collect(args.dir)

    spread = df.groupby("ticker")["hold_ret"].agg(lambda s: s.max() - s.min())

    sl = seed_level(df)
    rand_med = rdf.groupby("seed")["ret"].mean().median() if len(rdf) else np.nan

    cs = cell_summary(sl, rand_med)
    cs.to_csv(args.out / "summary.csv", index=False)
    print("\nSummary Results:")
    print(cs.round(2).to_string(index=False))

    heatmap(
        cs, "excess_med",
        lambda r: (f"Agent: {r.ret_med:.1f}% [{r.ret_lo:.1f}, {r.ret_hi:.1f}]\n"
                   f"Hold: {r.hold:.1f}% | Rand: {r.random:.1f}%\n"
                   f"Excess: {r.excess_med:+.1f} [{r.excess_lo:.1f}, {r.excess_hi:.1f}]\n"
                   f"P(beat): {r.p_beat_hold:.0%} (n={int(r.n_seeds)})\n"
                   f"Exp: {r.exposure:.0%} | Turn: {r.turnover:.1f}x\n"
                   f"B/S/H: {r.buys:.0f}/{r.sells:.0f}/{r.holds:.0f}"),
        "Agent vs Buy & Hold (Median over seeds, 95% Bootstrap CI)",
        "Median excess return vs hold (%)", "RdYlGn",
        "heatmap_agent_return.png", args.out, center=0.0
    )

    heatmap(
        cs, "sortino",
        lambda r: (f"Sortino: {r.sortino:.2f}\nSharpe: {r.sharpe:.2f} (hold {r.hold_sharpe:.2f})\n"
                   f"Calmar: {r.calmar:.2f} | Omega: {r.omega:.2f}\n"
                   f"MaxDD: {r.maxdd:.1f}% (hold {r.hold_maxdd:.1f}%)\n"
                   f"Beta: {r.beta:.2f} | Alpha: {r.alpha:.1f}%/yr"),
        "Risk Metrics",
        "Median Sortino", "YlGnBu",
        "heatmap_risk.png", args.out
    )

    plot_equity_curves(args.dir, args.out)


if __name__ == "__main__":
    main()