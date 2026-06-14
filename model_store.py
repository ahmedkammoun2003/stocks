"""Save and load walk-forward models (XGBoost per ticker + PPO RL)."""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path

import xgboost as xgb

DEFAULT_MODELS_ROOT = Path("saved_models")


def _json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_cycle_models(
    cycle: int,
    xgboost_models: dict,
    rl_model,
    *,
    train_start,
    val_start,
    test_start,
    test_end,
    pivot_columns: list[str],
    risk_history: dict | None = None,
    models_root: Path | None = None,
) -> Path:
    """
    Persist one walk-forward cycle under saved_models/cycle_NNN/ and refresh latest/.
    """
    root = Path(models_root or DEFAULT_MODELS_ROOT)
    cycle_dir = root / f"cycle_{cycle:03d}"
    latest_dir = root / "latest"

    if cycle_dir.exists():
        shutil.rmtree(cycle_dir)
    cycle_dir.mkdir(parents=True, exist_ok=True)

    xgb_dir = cycle_dir / "xgb"
    xgb_dir.mkdir()

    xgb_val_mse: dict[str, float] = {}
    for ticker, entry in xgboost_models.items():
        if entry is None:
            continue
        model, val_mse = entry
        model.save_model(str(xgb_dir / f"{ticker}.json"))
        xgb_val_mse[ticker] = float(val_mse)

    rl_path = cycle_dir / "rl_model.zip"
    if rl_model is not None:
        rl_model.save(str(rl_path))

    meta = {
        "cycle": cycle,
        "saved_at": datetime.now().isoformat(),
        "train_start": train_start,
        "val_start": val_start,
        "test_start": test_start,
        "test_end": test_end,
        "pivot_columns": list(pivot_columns),
        "xgb_tickers": sorted(xgb_val_mse.keys()),
        "xgb_val_mse": xgb_val_mse,
        "has_rl": rl_model is not None,
        "risk_history": risk_history or {},
    }
    with open(cycle_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(cycle_dir, latest_dir)

    print(f"[MODELS] Saved cycle {cycle} → {cycle_dir} (latest → {latest_dir})")
    return cycle_dir


def load_models(models_dir: Path | None = None) -> tuple[dict, object | None, dict]:
    """
    Load XGBoost models, optional RL model, and metadata from a saved cycle directory.

    Returns (xgboost_models, rl_model, metadata) where xgboost_models maps
    ticker -> (XGBRegressor, val_mse).
    """
    base = Path(models_dir or DEFAULT_MODELS_ROOT / "latest")
    meta_path = base / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No saved models at {base}. Run main.py first to train and save models."
        )

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    xgb_dir = base / "xgb"
    xgb_val_mse = meta.get("xgb_val_mse", {})
    xgboost_models: dict = {}

    for path in sorted(xgb_dir.glob("*.json")):
        ticker = path.stem
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        xgboost_models[ticker] = (model, float(xgb_val_mse.get(ticker, float("inf"))))

    rl_model = None
    rl_path = base / "rl_model.zip"
    if meta.get("has_rl") and rl_path.is_file():
        from stable_baselines3 import PPO

        rl_model = PPO.load(str(rl_path))

    return xgboost_models, rl_model, meta


def resolve_models_dir(path: str | Path | None) -> Path:
    """Resolve models directory; supports PyInstaller bundle layout."""
    if path is not None:
        return Path(path)
    return DEFAULT_MODELS_ROOT / "latest"
