"""Cross-sectional signals and ranks for backtest-aligned strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.feature_columns import FEATURE_COLUMNS
from trading_rules import (
    MARKET_BEARISH_THRESHOLD,
    build_market_indicators,
    filter_tickers_by_val_loss,
    normalized_ranks,
    passes_eligibility,
    sharpe_model_weights,
)


def _z_score(arr: np.ndarray) -> np.ndarray:
    std = np.std(arr)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / (std + 1e-8)


def build_pivoted(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivoted = (
        df.pivot(index="Date", columns="Ticker", values="Close")
        .ffill()
        .bfill()
        .sort_index()
    )
    vol_pivoted = (
        df.pivot(index="Date", columns="Ticker", values="Volume")
        .ffill()
        .bfill()
        .sort_index()
    )
    return pivoted, vol_pivoted


def compute_xgb_signals(
    df: pd.DataFrame,
    xgboost_models: dict,
    tickers: list[str],
    as_of_date: pd.Timestamp,
) -> dict[str, float]:
    signals: dict[str, float] = {}
    for ticker in tickers:
        if ticker not in xgboost_models:
            signals[ticker] = 0.0
            continue
        stock_rows = df[(df["Ticker"] == ticker) & (df["Date"] <= as_of_date)]
        if stock_rows.empty:
            signals[ticker] = 0.0
            continue
        latest_row = stock_rows.iloc[-1]
        feat = latest_row[FEATURE_COLUMNS].values.reshape(1, -1)
        model, _ = xgboost_models[ticker]
        signals[ticker] = float(model.predict(feat)[0])
    return signals


def build_rl_observation(
    pivoted: pd.DataFrame,
    pivot_columns: list[str],
    as_of_date: pd.Timestamp,
    current_weights: dict | None = None,
    window_size: int = 10,
) -> np.ndarray:
    """Build PPO observation vector at as_of_date (matches backtest / RL env)."""
    if as_of_date not in pivoted.index:
        valid = pivoted.index[pivoted.index <= as_of_date]
        if len(valid) == 0:
            raise ValueError(f"No market data on or before {as_of_date}")
        as_of_date = valid[-1]

    start_idx = pivoted.index.get_loc(as_of_date)
    n_assets = len(pivot_columns)
    current_weights = current_weights or {}

    if start_idx >= window_size:
        window_prices = pivoted.iloc[start_idx - window_size : start_idx].values
    else:
        window_prices = pivoted.iloc[:start_idx].values
        window_prices = np.pad(
            window_prices,
            ((window_size - len(window_prices), 0), (0, 0)),
            mode="edge",
        )

    norm_prices = (window_prices / (window_prices[0] + 1e-8)).flatten().astype(np.float32)
    if start_idx >= 5:
        ret_5d = (
            (pivoted.iloc[start_idx - 1].values - pivoted.iloc[start_idx - 6].values)
            / (pivoted.iloc[start_idx - 6].values + 1e-8)
        ).astype(np.float32)
    else:
        ret_5d = np.zeros(len(pivot_columns), dtype=np.float32)

    daily_rets = np.diff(window_prices, axis=0) / (window_prices[:-1] + 1e-8)
    vol_10d = np.std(daily_rets, axis=0).astype(np.float32)
    cur_w = np.array(
        [current_weights.get(t, 0.0) for t in pivot_columns],
        dtype=np.float32,
    )
    cur_w /= cur_w.sum() + 1e-8
    return np.concatenate([norm_prices, ret_5d, vol_10d, cur_w]).astype(np.float32)


def compute_rl_allocations(
    rl_model,
    pivoted: pd.DataFrame,
    pivot_columns: list[str],
    tickers: list[str],
    as_of_date: pd.Timestamp,
    current_weights: dict | None = None,
) -> dict[str, float]:
    if rl_model is None:
        return {t: 0.0 for t in tickers}

    obs = build_rl_observation(pivoted, pivot_columns, as_of_date, current_weights)
    action, _ = rl_model.predict(obs, deterministic=True)
    rl_raw = action / (np.sum(action) + 1e-8)
    out: dict[str, float] = {}
    for ticker in tickers:
        if ticker in pivot_columns:
            idx = pivot_columns.index(ticker)
            out[ticker] = float(rl_raw[idx]) if idx < len(rl_raw) else 0.0
        else:
            out[ticker] = 0.0
    return out


def compute_scores(
    strategy: str,
    tickers: list[str],
    xgb_signals: dict[str, float],
    rl_allocations: dict[str, float],
    w_xgb: float = 0.5,
    w_rl: float = 0.5,
) -> dict[str, float]:
    xgb_arr = np.array([xgb_signals[t] for t in tickers])
    rl_arr = np.array([rl_allocations[t] for t in tickers])
    xgb_norm = _z_score(xgb_arr)
    rl_norm = _z_score(rl_arr)

    if strategy == "combined":
        xgb_ranks = normalized_ranks(xgb_signals, tickers)
        rl_ranks = normalized_ranks(rl_allocations, tickers)
        return {t: w_xgb * xgb_ranks[t] + w_rl * rl_ranks[t] for t in tickers}

    scores: dict[str, float] = {}
    for j, ticker in enumerate(tickers):
        if strategy == "xgboost":
            scores[ticker] = float(xgb_norm[j])
        else:
            scores[ticker] = float(rl_norm[j])
    return scores


def rank_stocks(
    df: pd.DataFrame,
    xgboost_models: dict,
    rl_model,
    *,
    strategy: str = "combined",
    as_of_date: pd.Timestamp | None = None,
    pivot_columns: list[str] | None = None,
    recent_strategy_sharpes: dict | None = None,
    apply_val_loss_filter: bool = True,
) -> pd.DataFrame:
    """
    Rank tickers by model scores at as_of_date (default: latest date in df).

    Returns a DataFrame sorted by score descending.
    """
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    pivoted, vol_pivoted = build_pivoted(df)
    if pivoted.empty:
        return pd.DataFrame()

    all_cols = list(pivot_columns or pivoted.columns)
    indicators = build_market_indicators(pivoted, vol_pivoted)

    if as_of_date is None:
        as_of_date = df["Date"].max()
    as_of_date = pd.Timestamp(as_of_date)

    tickers = [t for t in all_cols if t in xgboost_models]
    if apply_val_loss_filter:
        tickers = filter_tickers_by_val_loss(tickers, xgboost_models)
    if not tickers:
        return pd.DataFrame()

    if strategy == "combined":
        w_xgb, w_rl = sharpe_model_weights(recent_strategy_sharpes)
    else:
        w_xgb = w_rl = 0.5

    xgb_signals = compute_xgb_signals(df, xgboost_models, tickers, as_of_date)
    rl_allocations = compute_rl_allocations(
        rl_model, pivoted, all_cols, tickers, as_of_date,
    )
    scores = compute_scores(
        strategy, tickers, xgb_signals, rl_allocations, w_xgb, w_rl,
    )

    if as_of_date not in pivoted.index:
        valid = pivoted.index[pivoted.index <= as_of_date]
        price_date = valid[-1] if len(valid) else as_of_date
    else:
        price_date = as_of_date

    prices_start = pivoted.loc[price_date]
    mkt_trend = indicators["market_return_20d"].get(price_date, 0.0)
    market_bearish = bool(mkt_trend < MARKET_BEARISH_THRESHOLD)

    rows = []
    for ticker in tickers:
        eligible = passes_eligibility(
            ticker,
            price_date,
            prices_start,
            indicators,
            vol_pivoted,
            strategy,
            xgb_signals.get(ticker, 0.0),
            rl_allocations.get(ticker, 0.0),
            w_xgb,
            w_rl,
        )
        rows.append(
            {
                "Ticker": ticker,
                "Score": scores[ticker],
                "XGB_5d_pred": xgb_signals[ticker],
                "RL_weight": rl_allocations[ticker],
                "Eligible": eligible and not market_bearish,
                "Val_MSE": xgboost_models[ticker][1],
            }
        )

    out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    out["AsOfDate"] = price_date.date().isoformat()
    out["Strategy"] = strategy
    out["Market_Bearish"] = market_bearish
    return out
