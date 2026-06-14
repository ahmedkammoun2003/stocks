"""Shared portfolio rules used by backtest and the RL training environment."""

import numpy as np
import pandas as pd

MAX_ASSETS = 5
TRANSACTION_FEE_RATE = 0.001
TRAILING_STOP_DD = 0.08
TRAILING_STOP_RECOVERY = 0.97
MARKET_BEARISH_THRESHOLD = -0.0005
MAX_ANNUAL_VOL = 0.35
MIN_PRED_RETURN = 0.002
VOLUME_RATIO_MIN = 0.70
REBALANCE_DAYS = 5


def build_market_indicators(pivoted: pd.DataFrame, vol_pivoted: pd.DataFrame) -> dict:
    daily_returns = pivoted.pct_change()
    return {
        'rolling_vol': daily_returns.rolling(window=30).std() * np.sqrt(252),
        'sma20': pivoted.rolling(window=20).mean(),
        'sma50': pivoted.rolling(window=50).mean(),
        'vol_ma20': vol_pivoted.rolling(window=20).mean(),
        'market_return_20d': daily_returns.rolling(window=20).mean().median(axis=1),
    }


def normalized_ranks(values: dict, tickers: list) -> dict:
    """Map tickers to percentile ranks in (0, 1]; higher raw value → higher rank."""
    if not tickers:
        return {}
    arr = np.array([values.get(t, 0.0) for t in tickers], dtype=float)
    order = np.argsort(arr, kind='mergesort')
    ranks = np.empty(len(tickers), dtype=float)
    ranks[order] = np.arange(1, len(tickers) + 1)
    denom = max(len(tickers), 1)
    return {tickers[i]: ranks[i] / denom for i in range(len(tickers))}


def sharpe_model_weights(recent_sharpes: dict | None) -> tuple[float, float]:
    """Weight XGB / RL by rolling Sharpe (last 3 cycles, floored at 0)."""
    if not recent_sharpes:
        return 0.5, 0.5

    def _sharpes(key: str) -> list:
        entry = recent_sharpes.get(key, [])
        if isinstance(entry, dict):
            return entry.get('sharpe', [])
        return entry

    def _avg(key: str) -> float:
        vals = [s for s in _sharpes(key)[-3:] if s is not None]
        return max(0.0, float(np.mean(vals))) if vals else 0.0

    xgb_s, rl_s = _avg('xgboost'), _avg('rl')
    total = xgb_s + rl_s
    if total <= 0:
        return 0.5, 0.5

    floor = 0.01
    w_x = max(floor, xgb_s / total)
    w_r = max(floor, rl_s / total)
    norm = w_x + w_r
    return w_x / norm, w_r / norm


def filter_tickers_by_val_loss(tickers: list, xgboost_models: dict) -> list:
    """Keep tickers at or below cross-sectional median XGB validation error."""
    if not tickers:
        return tickers

    xgb_losses = {t: xgboost_models[t][1] for t in tickers if t in xgboost_models}
    if not xgb_losses:
        return tickers

    xgb_med = float(np.median(list(xgb_losses.values())))
    kept = [t for t in tickers if t in xgb_losses and xgb_losses[t] <= xgb_med]
    return kept if kept else tickers


def passes_eligibility(
    ticker: str,
    date_start,
    prices_start,
    indicators: dict,
    vol_pivoted: pd.DataFrame,
    strategy: str,
    xgb_signal: float = 0.0,
    rl_allocation: float = 0.0,
    w_xgb: float = 0.5,
    w_rl: float = 0.5,
) -> bool:
    rolling_vol = indicators['rolling_vol']
    sma20 = indicators['sma20']
    sma50 = indicators['sma50']
    vol_ma20 = indicators['vol_ma20']

    vol = rolling_vol.loc[date_start, ticker] if date_start in rolling_vol.index else 1.0
    price = prices_start[ticker]
    s20 = sma20.loc[date_start, ticker] if date_start in sma20.index else price
    s50 = sma50.loc[date_start, ticker] if date_start in sma50.index else price

    if vol > MAX_ANNUAL_VOL:
        return False
    if price < s20:
        return False
    if price < s50 * 0.95:
        return False
    if date_start in vol_pivoted.index and ticker in vol_pivoted.columns:
        cur_vol = vol_pivoted.loc[date_start, ticker]
        avg_vol = vol_ma20.loc[date_start, ticker] if date_start in vol_ma20.index else 0
        if avg_vol > 0 and cur_vol < VOLUME_RATIO_MIN * avg_vol:
            return False

    if strategy == 'xgboost' and xgb_signal < MIN_PRED_RETURN:
        return False
    if strategy == 'rl' and rl_allocation <= 0:
        return False
    if strategy == 'combined':
        if w_rl > 0 and rl_allocation <= 0:
            return False
        if w_xgb > 0 and xgb_signal < MIN_PRED_RETURN:
            return False
    return True


def select_target_weights(
    tickers: list,
    scores: dict,
    date_start,
    prices_start,
    indicators: dict,
    vol_pivoted: pd.DataFrame,
    strategy: str,
    market_bearish: bool,
    xgb_signals: dict | None = None,
    rl_allocations: dict | None = None,
    w_xgb: float = 0.5,
    w_rl: float = 0.5,
    rank_thresholds: tuple = (0.75, 0.55, 0.35, 0.0),
    zscore_thresholds: tuple = (0.8, 0.5, 0.2, 0.0),
) -> dict:
    """Return portfolio weights (including Cash) after filters and top-N selection."""
    xgb_signals = xgb_signals or {}
    rl_allocations = rl_allocations or {}
    thresholds = rank_thresholds if strategy == 'combined' else zscore_thresholds

    eligible = []
    if not market_bearish:
        for threshold in thresholds:
            eligible = [
                t for t in tickers
                if passes_eligibility(
                    t, date_start, prices_start, indicators, vol_pivoted,
                    strategy, xgb_signals.get(t, 0.0), rl_allocations.get(t, 0.0),
                    w_xgb, w_rl,
                ) and scores.get(t, 0.0) >= threshold
            ]
            if eligible:
                break

    if eligible:
        top = sorted(eligible, key=lambda t: scores[t], reverse=True)[:MAX_ASSETS]
        sum_scores = sum(scores[t] for t in top)
        weights = {
            t: (scores[t] / sum_scores if sum_scores > 0 else 1.0 / len(top))
            for t in top
        }
        weights['Cash'] = 0.0
        return weights
    return {'Cash': 1.0}


def apply_turnover_fee(
    target_weights: dict,
    current_weights: dict,
    tickers: list,
    capital: float,
) -> tuple[float, float]:
    all_assets = set(list(tickers) + ['Cash'])
    t_w = {a: target_weights.get(a, 0.0) for a in all_assets}
    c_w = {a: current_weights.get(a, 0.0) for a in all_assets}
    turnover = 0.5 * sum(abs(t_w[a] - c_w[a]) for a in all_assets)
    fee = turnover * capital * TRANSACTION_FEE_RATE
    return fee, turnover
