"""Feature columns used by XGBoost."""

FEATURE_COLUMNS = (
    ['Fourier_Dominant', 'HMM_State', 'RSI', 'MACD', 'MACD_Signal',
     'BB_Upper', 'BB_Lower']
    + [f'Close_Lag_{i}' for i in range(1, 6)]
)

N_FEATURES = len(FEATURE_COLUMNS)
