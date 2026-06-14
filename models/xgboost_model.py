import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error

from models.feature_columns import FEATURE_COLUMNS


_DEFAULT_PARAMS = {
    'n_estimators':     2000,
    'learning_rate':    0.03,
    'max_depth':        4,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'min_child_weight': 1,
    'gamma':            0.0,
}


def train_xgboost(df, ticker, train_start, val_start, test_start, test_end,
                  hyperparams: dict = None, memory_manager=None,
                  n_jobs_override: int = None):
    """
    Train an XGBoost regressor to predict the 5-day forward return.
    Uses tree_method='hist' for fast histogram-based training.

    n_jobs_override : explicit thread count (used when running inside a
                      process pool so CPUs aren't over-subscribed).
    """
    print(f"Training XGBoost for {ticker}...")
    stock_df = df[df['Ticker'] == ticker].copy()

    stock_df['Target'] = (
        (stock_df['Close'].shift(-5) - stock_df['Close']) / stock_df['Close']
    )
    stock_df.dropna(inplace=True)

    train_df = stock_df[(stock_df['Date'] >= train_start) & (stock_df['Date'] < val_start)]
    val_df   = stock_df[(stock_df['Date'] >= val_start)   & (stock_df['Date'] < test_start)]
    test_df  = stock_df[(stock_df['Date'] >= test_start)  & (stock_df['Date'] <= test_end)]

    print(f"  Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("  Warning: Not enough data for 3-way split. Falling back to default split.")
        train_size = int(len(stock_df) * 0.7)
        val_size   = int(len(stock_df) * 0.15)
        train_df   = stock_df.iloc[:train_size]
        val_df     = stock_df.iloc[train_size:train_size + val_size]
        test_df    = stock_df.iloc[train_size + val_size:]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df['Target']
    X_val,   y_val   = val_df[FEATURE_COLUMNS],   val_df['Target']
    X_test,  y_test  = test_df[FEATURE_COLUMNS],  test_df['Target']

    if len(X_train) == 0:
        print(f"  Warning: Insufficient data to train XGBoost for {ticker}. Skipping.")
        return None

    p = {**_DEFAULT_PARAMS, **(hyperparams or {})}

    # Thread count priority: explicit override > memory manager > all cores
    if n_jobs_override is not None:
        n_jobs = n_jobs_override
    elif memory_manager is not None:
        n_jobs = memory_manager.xgb_n_jobs()
    else:
        import os; n_jobs = os.cpu_count() or -1

    # Shared kwargs for both model variants
    base_kwargs = dict(
        objective        = 'reg:squarederror',
        tree_method      = 'hist',      # fastest CPU method
        max_bin          = 512,         # more bins → better splits
        n_estimators     = p['n_estimators'],
        learning_rate    = p['learning_rate'],
        max_depth        = p['max_depth'],
        subsample        = p['subsample'],
        colsample_bytree = p['colsample_bytree'],
        reg_alpha        = p['reg_alpha'],
        reg_lambda       = p['reg_lambda'],
        min_child_weight = p['min_child_weight'],
        gamma            = p['gamma'],
        n_jobs           = n_jobs,
    )

    if len(X_val) == 0:
        model = xgb.XGBRegressor(**base_kwargs)
        model.fit(X_train, y_train, verbose=False)
        val_mse = float('inf')
        print("  XGBoost: No validation data, trained without early stopping.")
    else:
        model = xgb.XGBRegressor(**base_kwargs, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_mse = mean_squared_error(y_val, model.predict(X_val))
        print(f"  XGBoost Best Iteration: {model.best_iteration}")

    if len(X_test) > 0:
        mse = mean_squared_error(y_test, model.predict(X_test))
        print(f"  XGBoost Val MSE: {val_mse:.4f}, Test MSE for {ticker}: {mse:.4f}")
    else:
        print(f"  XGBoost Val MSE: {val_mse:.4f}, No test data for {ticker}.")

    return model, val_mse
