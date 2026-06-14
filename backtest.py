import numpy as np
import pandas as pd
from tqdm import tqdm

from models.feature_columns import FEATURE_COLUMNS
from metrics import verify_backtest_return
from trading_rules import (
    MARKET_BEARISH_THRESHOLD,
    TRAILING_STOP_DD,
    TRAILING_STOP_RECOVERY,
    build_market_indicators,
    normalized_ranks,
    sharpe_model_weights,
    filter_tickers_by_val_loss,
    select_target_weights,
    apply_turnover_fee,
)


def _z_score(arr):
    std = np.std(arr)
    return np.zeros_like(arr) if std == 0 else (arr - np.mean(arr)) / (std + 1e-8)


def run_backtest(df, xgboost_models, rl_model, test_start_date, test_end_date,
                 strategy='combined', recent_strategy_returns=None,
                 recent_strategy_sharpes=None):
    print(f"\n==================================================")
    print(f"       STARTING BACKTEST - STRATEGY: {strategy.upper()}      ")
    print(f"==================================================")

    pivoted = df.pivot(index='Date', columns='Ticker', values='Close').ffill().bfill().sort_index()
    vol_pivoted = df.pivot(index='Date', columns='Ticker', values='Volume').ffill().bfill().sort_index()
    all_pivot_cols = list(pivoted.columns)
    indicators = build_market_indicators(pivoted, vol_pivoted)

    tickers = [t for t in all_pivot_cols if t in xgboost_models]
    if not tickers:
        print("Warning: No valid tickers with trained models for this cycle. Skipping backtest.")
        return 0.0, [], 0.0, 0.0, 0.0

    filtered = filter_tickers_by_val_loss(tickers, xgboost_models)
    if len(filtered) < len(tickers):
        print(f"Val-loss filter: {len(filtered)}/{len(tickers)} tickers kept for backtest.")
    tickers = filtered

    test_dates = pivoted.index[(pivoted.index >= test_start_date) & (pivoted.index <= test_end_date)]
    if len(test_dates) < 5:
        print("Warning: Test period is too short. Skipping.")
        return 0.0, [], 0.0, 0.0, 0.0

    test_prices = pivoted.loc[test_dates]
    capital = 1000.0
    initial_capital = capital
    peak_capital = capital
    current_weights = {"Cash": 1.0}
    trades = []
    week_steps = range(0, len(test_dates) - 5, 5)
    weekly_capitals = [capital]
    weekly_returns = []
    trailing_stop_active = False

    print(f"Backtesting over {len(week_steps)} weeks "
          f"({test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')})...")

    if strategy == 'combined':
        w_xgb, w_rl = sharpe_model_weights(recent_strategy_sharpes)
        print(f"Combined Sharpe weights -> XGB: {w_xgb:.2f}, RL: {w_rl:.2f}")
    else:
        w_xgb = w_rl = 0.5

    for i in tqdm(week_steps, desc=f"[{strategy.upper()}] Weekly Backtest", unit="week"):
        start_idx = pivoted.index.get_loc(test_dates[i])
        date_start = test_dates[i]
        date_end = test_dates[i + 5]
        prices_start = test_prices.loc[date_start]
        prices_end = test_prices.loc[date_end]

        peak_capital = max(peak_capital, capital)
        drawdown_from_peak = (peak_capital - capital) / (peak_capital + 1e-8)
        if drawdown_from_peak > TRAILING_STOP_DD:
            trailing_stop_active = True
        elif trailing_stop_active and capital >= peak_capital * TRAILING_STOP_RECOVERY:
            trailing_stop_active = False

        if trailing_stop_active:
            target_weights = {"Cash": 1.0}
            fee, turnover = apply_turnover_fee(target_weights, current_weights, tickers, capital)
            if turnover > 0.01:
                capital -= fee
            current_weights = {t: target_weights.get(t, 0.0) for t in tickers}
            current_weights['Cash'] = 1.0
            weekly_returns.append(0.0)
            weekly_capitals.append(capital)
            continue

        mkt_trend = indicators['market_return_20d'].get(date_start, 0.0)
        market_bearish = bool(mkt_trend < MARKET_BEARISH_THRESHOLD)

        xgb_signals = {}
        for ticker in tickers:
            stock_rows = df[(df['Ticker'] == ticker) & (df['Date'] <= date_start)]
            if stock_rows.empty:
                xgb_signals[ticker] = 0.0
                continue
            latest_row = stock_rows.iloc[-1]
            xgb_feat = latest_row[FEATURE_COLUMNS].values.reshape(1, -1)
            xgb_signals[ticker] = float(xgboost_models[ticker][0].predict(xgb_feat)[0])

        n_assets_full = len(all_pivot_cols)
        window_size = 10
        if start_idx >= window_size:
            window_prices = pivoted.iloc[start_idx - window_size:start_idx].values
        else:
            window_prices = pivoted.iloc[:start_idx].values
            window_prices = np.pad(
                window_prices,
                ((window_size - len(window_prices), 0), (0, 0)),
                mode='edge',
            )

        norm_prices = (window_prices / (window_prices[0] + 1e-8)).flatten().astype(np.float32)
        if start_idx >= 5:
            ret_5d = (
                (pivoted.iloc[start_idx - 1].values - pivoted.iloc[start_idx - 6].values)
                / (pivoted.iloc[start_idx - 6].values + 1e-8)
            ).astype(np.float32)
        else:
            ret_5d = np.zeros(n_assets_full, dtype=np.float32)

        daily_rets_window = np.diff(window_prices, axis=0) / (window_prices[:-1] + 1e-8)
        vol_10d = np.std(daily_rets_window, axis=0).astype(np.float32)
        cur_w_full = np.array(
            [current_weights.get(t, 0.0) for t in all_pivot_cols], dtype=np.float32,
        )
        cur_w_full /= cur_w_full.sum() + 1e-8

        obs = np.concatenate([norm_prices, ret_5d, vol_10d, cur_w_full]).astype(np.float32)
        rl_action, _ = rl_model.predict(obs, deterministic=True)
        rl_raw = rl_action / (np.sum(rl_action) + 1e-8)
        rl_allocations = {}
        for ticker in tickers:
            col_idx = all_pivot_cols.index(ticker)
            rl_allocations[ticker] = float(rl_raw[col_idx]) if col_idx < len(rl_raw) else 0.0

        xgb_arr = np.array([xgb_signals[t] for t in tickers])
        rl_arr = np.array([rl_allocations[t] for t in tickers])
        xgb_norm = _z_score(xgb_arr)
        rl_norm = _z_score(rl_arr)
        xgb_norm_dict = {tickers[j]: xgb_norm[j] for j in range(len(tickers))}
        rl_norm_dict = {tickers[j]: rl_norm[j] for j in range(len(tickers))}

        if strategy == 'combined':
            xgb_ranks = normalized_ranks(xgb_signals, tickers)
            rl_ranks = normalized_ranks(rl_allocations, tickers)
            scores = {
                t: w_xgb * xgb_ranks[t] + w_rl * rl_ranks[t]
                for t in tickers
            }
        else:
            scores = {}
            for ticker in tickers:
                if strategy == 'xgboost':
                    scores[ticker] = xgb_norm_dict[ticker]
                else:
                    scores[ticker] = rl_norm_dict[ticker]

        target_weights = select_target_weights(
            tickers, scores, date_start, prices_start,
            indicators, vol_pivoted, strategy, market_bearish,
            xgb_signals, rl_allocations, w_xgb, w_rl,
        )

        fee, turnover = apply_turnover_fee(target_weights, current_weights, tickers, capital)
        capital -= fee

        if turnover > 0.01:
            all_assets = set(list(tickers) + ['Cash'])
            t_w = {a: target_weights.get(a, 0.0) for a in all_assets}
            active_allocs = {k: f"{v:.1%}" for k, v in t_w.items() if v > 0}
            trades.append({
                'Date': date_start.strftime('%Y-%m-%d'),
                'Turnover': turnover,
                'Fee': fee,
                'Capital': capital,
                'Allocations': active_allocs,
                'Market_Bearish': market_bearish,
            })

        current_weights = {t: target_weights.get(t, 0.0) for t in tickers}
        current_weights['Cash'] = target_weights.get('Cash', 0.0)

        capital_before = capital
        ending_capital = 0.0
        for asset, weight in current_weights.items():
            if asset == 'Cash':
                ending_capital += weight * capital_before
            else:
                weekly_return = (
                    (prices_end[asset] - prices_start[asset]) / prices_start[asset]
                )
                ending_capital += weight * capital_before * (1 + weekly_return)

        weekly_return_pct = (ending_capital - capital_before) / (capital_before + 1e-8)
        weekly_returns.append(weekly_return_pct)
        capital = ending_capital
        weekly_capitals.append(capital)

    weekly_returns_arr = np.array(weekly_returns)
    mean_return = np.mean(weekly_returns_arr)
    std_return = np.std(weekly_returns_arr)
    sharpe_ratio = np.sqrt(52) * (mean_return / (std_return + 1e-8))

    weekly_capitals_arr = np.array(weekly_capitals)
    peaks = np.maximum.accumulate(weekly_capitals_arr)
    drawdowns = (peaks - weekly_capitals_arr) / (peaks + 1e-8)
    max_drawdown_pct = -float(np.max(drawdowns)) * 100.0

    print("\n================ BACKTEST RESULTS ================")
    print(f"Initial Capital:  1,000.00 TND")
    print(f"Final Capital:    {capital:.2f} TND")
    total_return = ((capital - initial_capital) / initial_capital) * 100
    mismatch = verify_backtest_return(weekly_returns, initial_capital, capital)
    if mismatch is not None:
        print(f"WARNING: Return check failed — weekly compound vs total "
              f"differs by {mismatch:.3f} pp")
    print(f"Total Return:     {total_return:.2f}%")
    print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {max_drawdown_pct:.2f}%")
    print(f"Number of trades: {len(trades)}")
    print("\nTrades Made:")
    for t in trades[:15]:
        bear_tag = " [BEAR]" if t.get('Market_Bearish') else ""
        print(f"  {t['Date']}: Rebalanced (Turnover: {t['Turnover']:.1%}, "
              f"Fee: {t['Fee']:.2f}, Capital: {t['Capital']:.2f}, "
              f"Allocs: {t['Allocations']}){bear_tag}")
    if len(trades) > 15:
        print(f"  ... and {len(trades) - 15} more rebalances.")
    print("==================================================")

    return total_return, trades, sharpe_ratio, max_drawdown_pct, capital
