"""Portfolio metrics for walk-forward backtest summaries."""

import numpy as np


def compound_return_pct(period_returns_pct: list[float]) -> float:
    """Compound a list of period returns (each in %, e.g. 10.0 = +10%)."""
    if not period_returns_pct:
        return 0.0
    growth = 1.0
    for r in period_returns_pct:
        growth *= 1.0 + r / 100.0
    return (growth - 1.0) * 100.0


def summarize_cycle_returns(returns_pct: list[float]) -> dict:
    """
    Summarize independent per-cycle returns (each cycle resets capital).

    Arithmetic mean overstates growth when returns vary; geometric mean is the
    constant per-cycle return that would match the same compounded growth.
    """
    if not returns_pct:
        return {
            'n_cycles': 0,
            'mean_return_pct': 0.0,
            'geometric_mean_return_pct': 0.0,
            'compounded_all_cycles_pct': 0.0,
        }
    n = len(returns_pct)
    mean = float(np.mean(returns_pct))
    geo = compound_return_pct(returns_pct)
    geo_per_cycle = ((1.0 + geo / 100.0) ** (1.0 / n) - 1.0) * 100.0 if n else 0.0
    compounded = compound_return_pct(returns_pct)
    return {
        'n_cycles': n,
        'mean_return_pct': mean,
        'geometric_mean_return_pct': geo_per_cycle,
        'compounded_all_cycles_pct': compounded,
    }


def verify_backtest_return(weekly_returns: list[float], initial_capital: float,
                           final_capital: float, tol_pct: float = 0.05) -> float | None:
    """
    Return mismatch in percentage points if compounded weekly returns
    do not match (final - initial) / initial; else None.
    """
    if not weekly_returns or initial_capital <= 0:
        return None
    compounded = initial_capital * float(np.prod(1.0 + np.array(weekly_returns)))
    reported = (final_capital - initial_capital) / initial_capital
    expected_pct = (compounded / initial_capital - 1.0) * 100.0
    reported_pct = reported * 100.0
    diff = abs(expected_pct - reported_pct)
    return diff if diff > tol_pct else None
