#!/usr/bin/env python3
"""
Rank Tunisian stocks (BVMT) using saved walk-forward models.

Fetches the latest OHLCV via data_loader.load_tunisian_stocks() (ilboursa.com),
rebuilds features, loads saved_models/latest/, and prints a ranked table.

Build standalone executable:
  ./scripts/build_predictor.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Project root (or PyInstaller extract dir for imports)
if getattr(sys, "frozen", False):
    _ROOT = Path(sys.executable).resolve().parent
else:
    _ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loader import load_bvmt_stocks, load_tunisian_stocks
from features import preprocess_features
from model_store import DEFAULT_MODELS_ROOT, load_models, resolve_models_dir
from scoring import rank_stocks


MIN_OBS = 756


def _default_models_dir() -> Path:
    if getattr(sys, "frozen", False):
        bundled = _ROOT / "saved_models" / "latest"
        if bundled.is_dir() and (bundled / "metadata.json").is_file():
            return bundled
    return DEFAULT_MODELS_ROOT / "latest"


def refresh_market_data(years: int = 30) -> pd.DataFrame:
    """Download / incrementally update BVMT quotes through ilboursa (existing loader)."""
    print("Fetching latest BVMT market data (ilboursa.com)...")
    df = load_bvmt_stocks(years=years)
    if not isinstance(df, pd.DataFrame) or df.empty or 'Date' not in df.columns:
        print("Warning: market data fetch returned no usable rows.")
        return pd.DataFrame(columns=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    try:
        min_date = pd.to_datetime(df['Date']).min()
        max_date = pd.to_datetime(df['Date']).max()
        print(
            f"Data: {len(df)} rows, {df['Ticker'].nunique()} tickers, "
            f"{min_date.date()} → {max_date.date()}"
        )
    except Exception:
        print(f"Data: {len(df)} rows, {df['Ticker'].nunique()} tickers, invalid Date column")
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ticker_counts = df["Ticker"].value_counts()
    keep = ticker_counts[ticker_counts >= MIN_OBS].index
    dropped = set(ticker_counts.index) - set(keep)
    if dropped:
        print(f"Dropping thin-history tickers: {sorted(dropped)}")
    df = df[df["Ticker"].isin(keep)].copy()
    print("Engineering features (Fourier, HMM, technicals)...")
    return preprocess_features(df)


def run_predict(
    strategy: str = "combined",
    models_dir: Path | None = None,
    years: int = 30,
    top_n: int | None = None,
    output_csv: Path | None = None,
    skip_refresh: bool = False,
    cache_csv: Path | None = None,
) -> pd.DataFrame:
    models_path = resolve_models_dir(models_dir or _default_models_dir())
    xgboost_models, rl_model, meta = load_models(models_path)

    if skip_refresh and cache_csv and cache_csv.is_file():
        print(f"Loading cached CSV: {cache_csv}")
        df = pd.read_csv(cache_csv)
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df = refresh_market_data(years=years)

    df = prepare_dataframe(df)

    pivot_columns = meta.get("pivot_columns")
    if not pivot_columns:
        pivot_columns = sorted(df["Ticker"].unique())

    risk_history = meta.get("risk_history")

    if strategy == "rl" and rl_model is None:
        print("WARNING: No RL model in bundle; falling back to xgboost ranking.")
        strategy = "xgboost"

    ranked = rank_stocks(
        df,
        xgboost_models,
        rl_model,
        strategy=strategy,
        pivot_columns=pivot_columns,
        recent_strategy_sharpes=risk_history,
    )

    if ranked.empty:
        print("No tickers to rank (check saved models vs current universe).")
        return ranked

    if top_n is not None:
        ranked = ranked.head(top_n)

    cycle = meta.get("cycle", "?")
    print(f"\n{'=' * 60}")
    print(f" STOCK RANKINGS — strategy={strategy}  (models: cycle {cycle})")
    print(f" As of: {ranked['AsOfDate'].iloc[0]}  |  "
          f"Market bearish: {ranked['Market_Bearish'].iloc[0]}")
    print(f"{'=' * 60}\n")

    display_cols = [
        "Rank", "Ticker", "Score", "XGB_5d_pred", "RL_weight", "Eligible",
    ]
    print(ranked[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    eligible = ranked[ranked["Eligible"]]
    if not eligible.empty:
        print(f"\nTop eligible picks ({len(eligible)} passed filters):")
        print(eligible[display_cols].head(min(10, len(eligible))).to_string(
            index=False, float_format=lambda x: f"{x:.4f}",
        ))

    if output_csv:
        ranked.to_csv(output_csv, index=False)
        print(f"\nWrote {output_csv}")

    return ranked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank BVMT stocks using saved XGBoost + RL models.",
    )
    parser.add_argument(
        "--strategy",
        choices=["combined", "xgboost", "rl"],
        default="combined",
        help="Ranking method (default: combined)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=f"Path to saved cycle (default: {_default_models_dir()})",
    )
    parser.add_argument("--years", type=int, default=30, help="History years for data load")
    parser.add_argument("--top", type=int, default=None, help="Show only top N rows")
    parser.add_argument("--csv", type=Path, default=None, help="Write rankings to CSV")
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Use local tunisian_stocks_30y.csv only (no ilboursa fetch)",
    )
    parser.add_argument(
        "--cache-csv",
        type=Path,
        default=Path("tunisian_stocks_30y.csv"),
        help="CSV used with --skip-refresh",
    )
    args = parser.parse_args(argv)

    try:
        run_predict(
            strategy=args.strategy,
            models_dir=args.models_dir,
            years=args.years,
            top_n=args.top,
            output_csv=args.csv,
            skip_refresh=args.skip_refresh,
            cache_csv=args.cache_csv,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
