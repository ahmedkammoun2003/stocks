"""
main.py — Multi-strategy algorithmic trading system.

Parallelism architecture
------------------------
  Main process
  │
  ├─ ProcessPoolExecutor(spawn|fork, N workers)
  │    Each worker trains XGBoost for one ticker independently.
  │
  ├─ threading.Thread("RL")                 ← concurrent with pool
  │    PPO + DummyVecEnv/SubprocVecEnv
  │
  └─ join() before backtest
"""

import os
import sys
import threading
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

_N_CPU = os.cpu_count() or 4
os.environ.setdefault('OMP_NUM_THREADS',        str(_N_CPU))
os.environ.setdefault('MKL_NUM_THREADS',        str(_N_CPU))
os.environ.setdefault('OPENBLAS_NUM_THREADS',   str(_N_CPU))
os.environ.setdefault('NUMEXPR_NUM_THREADS',    str(_N_CPU))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(_N_CPU))
torch.set_num_threads(_N_CPU)
torch.set_num_interop_threads(max(2, _N_CPU // 4))

from data_loader import load_tunisian_stocks
from features import preprocess_features
from parallel_pool import get_shared_dataframe, init_pool_dataframe, pickle_dataframe
from models.xgboost_model import train_xgboost
from models.rl_model import train_rl_model
from backtest import run_backtest
from memory_manager import MemoryManager
from metrics import summarize_cycle_returns
from model_store import save_cycle_models

TRAIN_DAYS = 4 * 365
VAL_DAYS   = 2 * 365
TEST_DAYS  = 2 * 365
STEP_DAYS  = 4 * 365


def _process_pool_start_method() -> str:
    """CUDA cannot be re-initialized after fork; use spawn when a GPU is present."""
    if sys.platform == 'win32':
        return 'spawn'
    if torch.cuda.is_available():
        return 'spawn'
    return 'fork'


def _train_ticker_worker(args: tuple):
    """Train XGBoost for one ticker inside a worker process."""
    ticker, train_start, val_start, test_start, test_end, xgb_params, n_xgb_jobs = args

    df = get_shared_dataframe()
    os.environ['OMP_NUM_THREADS'] = str(n_xgb_jobs)
    os.environ['MKL_NUM_THREADS'] = str(n_xgb_jobs)

    try:
        xgb_out = train_xgboost(
            df, ticker, train_start, val_start, test_start, test_end,
            hyperparams=xgb_params,
            n_jobs_override=n_xgb_jobs,
        )
    except Exception as exc:
        print(f"\n  [XGB-{ticker}] Error: {exc}", flush=True)
        return ticker, None

    return ticker, xgb_out


def _print_resource_plan(mm: MemoryManager, df_ram_gb: float = 0.0) -> tuple:
    """Returns (n_workers, n_xgb_jobs)."""
    n_workers  = mm.process_pool_size(df_ram_gb=df_ram_gb)
    n_xgb_jobs = mm.xgb_n_jobs_parallel(n_workers)

    print("┌─────────────────────────────────────────────────────┐")
    print("│              RESOURCE ALLOCATION PLAN               │")
    print("├─────────────────────────────────────────────────────┤")
    print(f"│  DataFrame RAM (per worker, spawn)      : {df_ram_gb:>5.2f} GB       │")
    print(f"│  Ticker processes (ProcessPoolExecutor) : {n_workers:<4}       │")
    print(f"│  XGBoost threads per process            : {n_xgb_jobs:<4}       │")
    print(f"│  RL parallel environments (VecEnv)      : {mm.rl_n_envs():<4}       │")
    print(f"│  RL total timesteps                     : {mm.rl_total_timesteps():<8}   │")
    print("└─────────────────────────────────────────────────────┘\n")
    return n_workers, n_xgb_jobs


def main():
    mm = MemoryManager()
    mm.report()

    print("Loading Data...")
    df = load_tunisian_stocks(years=30)
    print(
        f"Loaded {len(df)} total rows of data "
        f"({df['Ticker'].nunique()} tickers, "
        f"{df['Date'].min().date()} → {df['Date'].max().date()})."
    )

    df_ram_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)
    n_workers, n_xgb_jobs = _print_resource_plan(mm, df_ram_gb)

    min_obs         = 756
    ticker_counts   = df['Ticker'].value_counts()
    tickers_to_keep = ticker_counts[ticker_counts >= min_obs].index
    dropped         = list(set(ticker_counts.index) - set(tickers_to_keep))
    if dropped:
        print(f"\nDropping tickers with insufficient data: {dropped}")
    df = df[df['Ticker'].isin(tickers_to_keep)]
    print(f"\nProceeding with {len(tickers_to_keep)} tickers.")
    print(list(tickers_to_keep))

    print("\nPreprocessing Features (Fourier, HMM, etc.)...")
    df = preprocess_features(df)
    print("Feature Engineering completed.")

    tickers  = df['Ticker'].unique()
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    current_start   = min_date
    cycle           = 1
    returns_history = {'xgboost': [], 'rl': [], 'combined': []}
    risk_history = {
        s: {'sharpe': [], 'max_drawdown_pct': []}
        for s in ['xgboost', 'rl', 'combined']
    }

    pool_start_method = _process_pool_start_method()
    pool_ctx = mp.get_context(pool_start_method)
    df_pickle = pickle_dataframe(df)
    print(
        f"[PARALLEL] DataFrame pickled once ({len(df_pickle) / 1024**2:.1f} MB) "
        f"for worker initialisation.\n"
    )

    while True:
        train_start = current_start
        val_start   = train_start + pd.Timedelta(days=TRAIN_DAYS)
        test_start  = val_start   + pd.Timedelta(days=VAL_DAYS)
        test_end    = test_start  + pd.Timedelta(days=TEST_DAYS)

        if test_start >= max_date:
            break
        if test_end > max_date:
            test_end = max_date

        print(f"\n{'═' * 54}")
        print(f"               CYCLE {cycle}")
        print(f" Train: {train_start.date()} → {val_start.date()}")
        print(f" Val:   {val_start.date()} → {test_start.date()}")
        print(f" Test:  {test_start.date()} → {test_end.date()}")
        print(f"{'═' * 54}")

        mm_c         = MemoryManager()
        n_workers_c  = mm_c.process_pool_size(df_ram_gb=df_ram_gb)
        n_xgb_jobs_c = mm_c.xgb_n_jobs_parallel(n_workers_c)

        print(
            f"[MEM] Avail: {mm_c.available_gb:.1f} GB | "
            f"Budget: {mm_c.budget_gb:.1f} GB | "
            f"GPU free: {mm_c.gpu_free_gb:.1f} GB | "
            f"Procs: {n_workers_c} | "
            f"XGB-threads/proc: {n_xgb_jobs_c}"
        )

        xgboost_models: dict = {}

        rl_result: list = [None]
        rl_error:  list = [None]

        def _rl_bg():
            try:
                rl_result[0] = train_rl_model(
                    df, train_start, val_start, test_start,
                    memory_manager=mm_c,
                )
            except Exception as exc:
                rl_error[0] = exc

        rl_thread = threading.Thread(target=_rl_bg, daemon=True,
                                     name=f"RL-C{cycle}")
        print(f"\n[PARALLEL] RL thread '{rl_thread.name}' starting...")
        rl_thread.start()

        print(
            f"[PARALLEL] ProcessPoolExecutor: {len(tickers)} tickers "
            f"× {n_workers_c} processes ({pool_start_method})\n"
        )

        args_list = [
            (ticker, train_start, val_start, test_start, test_end, None, n_xgb_jobs_c)
            for ticker in tickers
        ]

        completed = 0
        with ProcessPoolExecutor(
            max_workers=n_workers_c,
            mp_context=pool_ctx,
            initializer=init_pool_dataframe,
            initargs=(df_pickle,),
        ) as pool:
            future_map = {
                pool.submit(_train_ticker_worker, args): args[0]
                for args in args_list
            }
            pbar = tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc=f"Cycle {cycle} — XGBoost training",
                unit="ticker",
            )
            for future in pbar:
                ticker = future_map[future]
                try:
                    t, xgb_out = future.result()
                    if xgb_out is not None:
                        xgboost_models[t] = xgb_out
                        completed += 1
                    pbar.set_postfix(ready=completed, last=t)
                except Exception as exc:
                    print(f"\n  [Worker-{ticker}] Error: {exc}")

        print(
            f"\n[PARALLEL] Ticker pool done: "
            f"{completed}/{len(tickers)} XGB models ready. "
            f"Waiting for RL thread…"
        )

        rl_thread.join()
        if rl_error[0]:
            print(f"[WARNING] RL failed: {rl_error[0]}")
            rl_model = None
        else:
            rl_model = rl_result[0]
            print(f"[PARALLEL] RL thread joined. All models ready.\n")

        for strat in ['xgboost', 'rl', 'combined']:
            if strat == 'rl' and rl_model is None:
                returns_history[strat].append(0.0)
                risk_history[strat]['sharpe'].append(0.0)
                risk_history[strat]['max_drawdown_pct'].append(0.0)
                print(f"Cycle {cycle} Return ({strat}): N/A")
                continue
            ret, _, sharpe, max_dd, _final_cap = run_backtest(
                df, xgboost_models, rl_model,
                test_start, test_end,
                strategy=strat,
                recent_strategy_returns=returns_history,
                recent_strategy_sharpes=risk_history,
            )
            returns_history[strat].append(ret)
            risk_history[strat]['sharpe'].append(sharpe)
            risk_history[strat]['max_drawdown_pct'].append(max_dd)
            print(
                f"Cycle {cycle} ({strat}): return {ret:.2f}% | "
                f"Sharpe {sharpe:.2f} | max DD {max_dd:.2f}%"
            )

        pivot_cols = list(
            df.pivot(index='Date', columns='Ticker', values='Close').columns
        )
        save_cycle_models(
            cycle,
            xgboost_models,
            rl_model,
            train_start=train_start,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            pivot_columns=pivot_cols,
            risk_history=risk_history,
        )

        current_start += pd.Timedelta(days=STEP_DAYS)
        cycle += 1

    print("\n" + "=" * 50)
    print("             OVERALL CYCLE RESULTS")
    print("=" * 50)
    for idx in range(len(returns_history['combined'])):
        print(f"\n--- Cycle {idx + 1} ---")
        for strat in ['xgboost', 'rl', 'combined']:
            ret = returns_history[strat][idx]
            sh = risk_history[strat]['sharpe'][idx]
            dd = risk_history[strat]['max_drawdown_pct'][idx]
            print(
                f"  {strat.capitalize()}: return {ret:.2f}% | "
                f"Sharpe {sh:.2f} | max drawdown {dd:.2f}%"
            )

    print("\n--- Summary Across Cycles ---")
    print("  (Each cycle backtest starts at 1,000 TND; test windows are separate.)")
    for strat in ['xgboost', 'rl', 'combined']:
        rets = returns_history[strat]
        if not rets:
            continue
        n = len(rets)
        stats = summarize_cycle_returns(rets)
        avg_sh = sum(risk_history[strat]['sharpe']) / n
        avg_dd = sum(risk_history[strat]['max_drawdown_pct']) / n
        worst_dd = min(risk_history[strat]['max_drawdown_pct'])
        print(f"\n  {strat.capitalize()} ({n} cycles):")
        print(f"    Mean cycle return:      {stats['mean_return_pct']:>7.2f}%  "
              f"(arithmetic average of per-cycle totals)")
        print(f"    Geometric mean / cycle:  {stats['geometric_mean_return_pct']:>7.2f}%  "
              f"(typical compounded rate)")
        print(f"    Chained compound:       {stats['compounded_all_cycles_pct']:>7.2f}%  "
              f"(1,000 TND through all cycles in sequence)")
        print(f"    Mean Sharpe (weekly):   {avg_sh:>7.2f}")
        print(f"    Mean / worst max DD:    {avg_dd:>7.2f}% / {worst_dd:.2f}%")

    print("\n--- Project Execution Completed Successfully ---")


if __name__ == "__main__":
    main()
