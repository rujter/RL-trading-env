import numpy as np
def _div(a, b):
    return float(a / b) if b != 0 else np.nan


def step_returns(equity, start_cash):
    eq = np.insert(equity, 0, start_cash)
    return np.diff(eq) / eq[:-1]


def max_drawdown(equity, start_cash):
    eq = np.insert(equity, 0, start_cash)
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / peak))


def sharpe(r, ppy):
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return _div(r.mean(), sd) * np.sqrt(ppy)


def sortino(r, ppy, target=0.0):
    diff = r - target
    downside_std = np.sqrt(np.mean(np.minimum(0.0, diff) ** 2))
    return _div(diff.mean(), downside_std) * np.sqrt(ppy)


def omega(r, target=0.0):
    diff = r - target
    gains = np.maximum(0.0, diff).sum()
    losses = np.maximum(0.0, -diff).sum()
    return _div(gains, losses)


def cagr(equity, start_cash, ppy):
    years = len(equity) / ppy
    total_return = float(equity[-1]) / start_cash
    if total_return <= 0 or years <= 0:
        return -1.0
    return total_return ** (1.0 / years) - 1.0


def beta_alpha(r_agent, r_bench, ppy):
    if len(r_bench) <= 1:
        return np.nan, np.nan
    var_b = r_bench.var(ddof=1)
    if var_b == 0:
        return np.nan, np.nan

    beta = np.cov(r_agent, r_bench, ddof=1)[0, 1] / var_b
    alpha_ann = (r_agent.mean() - beta * r_bench.mean()) * ppy
    return float(beta), float(alpha_ann * 100.0)


def episode_metrics(equity, price, shares, orders, start_cash, ppy=252, cost_bps=0.0):
    equity = np.asarray(equity, dtype=float)
    price = np.asarray(price, dtype=float)
    shares = np.asarray(shares, dtype=float)
    orders = np.asarray(orders)

    prev_eq = np.insert(equity[:-1], 0, start_cash)
    dshares = np.abs(np.diff(shares, prepend=0.0))
    turnover = dshares * price / prev_eq
    
    r_gross = step_returns(equity, start_cash)
    r = r_gross - (cost_bps / 1e4) * turnover
    net_eq = start_cash * np.cumprod(1.0 + r)

    hold_eq = start_cash * price / price[0]
    r_hold = step_returns(hold_eq, start_cash)

    beta, alpha = beta_alpha(r, r_hold, ppy)
    mdd = max_drawdown(net_eq, start_cash)

    buys = int((orders == 1).sum())
    sells = int((orders == 2).sum())
    holds = len(orders) - buys - sells

    return {
        "ret": (net_eq[-1] / start_cash - 1.0) * 100.0,
        "hold_ret": (hold_eq[-1] / start_cash - 1.0) * 100.0,
        "sharpe": sharpe(r, ppy),
        "sortino": sortino(r, ppy),
        "omega": omega(r),
        "maxdd": mdd * 100.0,
        "hold_maxdd": max_drawdown(hold_eq, start_cash) * 100.0,
        "calmar": _div(cagr(net_eq, start_cash, ppy), mdd),
        "hold_sharpe": sharpe(r_hold, ppy),
        "beta": beta,
        "alpha": alpha,
        "exposure": float(np.mean(shares * price / equity)),
        "turnover": float(turnover.sum()),
        "buys": buys,
        "sells": sells,
        "holds": holds,
        "steps": len(equity),
    }