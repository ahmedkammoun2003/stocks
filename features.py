import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.fft import fft
from hmmlearn import hmm
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

HMM_N_STATES = 4
HMM_N_ITER = 500
HMM_N_RESTARTS = 3
HMM_TOL = 1e-3


@contextmanager
def _quiet_hmmlearn():
    """Suppress benign EM log-likelihood oscillation warnings from hmmlearn."""
    logger = logging.getLogger('hmmlearn')
    prev = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(prev)


def add_fourier_features(df, window=20):
    """
    Computes window-based Fourier frequency features.
    """
    prices = df['Close'].values
    fourier_features = np.zeros(len(prices))

    for i in range(window, len(prices)):
        window_data = prices[i-window:i]
        fft_vals = np.abs(fft(window_data))
        if window > 2:
            fourier_features[i] = np.max(fft_vals[1:window//2])

    df['Fourier_Dominant'] = fourier_features
    return df


def _kmeans_hmm_init(X: np.ndarray, labels: np.ndarray, n_states: int):
    """Build HMM parameters from KMeans cluster labels."""
    startprob = np.bincount(labels, minlength=n_states).astype(float)
    startprob = np.maximum(startprob, 1.0)
    startprob /= startprob.sum()

    trans = np.full((n_states, n_states), 1.0 / n_states)
    for i in range(len(labels) - 1):
        trans[labels[i], labels[i + 1]] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    trans /= row_sums

    means = np.zeros((n_states, X.shape[1]))
    covars = np.zeros((n_states, X.shape[1]))
    for k in range(n_states):
        pts = X[labels == k]
        if len(pts) == 0:
            means[k] = X.mean(axis=0)
            covars[k] = X.var(axis=0) + 1e-2
        else:
            means[k] = pts.mean(axis=0)
            covars[k] = pts.var(axis=0) + 1e-2

    return startprob, trans, means, covars


def _fit_hmm_states(returns: np.ndarray, n_states: int) -> np.ndarray:
    """
    Fit a Gaussian HMM on [return, rolling-vol] with KMeans warm-start.
    Returns integer state labels aligned with ``returns``.
    """
    n = len(returns)
    if n < 50:
        return np.zeros(n, dtype=int)

    roll_vol = pd.Series(returns).rolling(20, min_periods=5).std().to_numpy()
    fallback_vol = float(np.std(returns)) if np.std(returns) > 1e-8 else 0.01
    roll_vol = np.where(np.isfinite(roll_vol), roll_vol, fallback_vol)
    roll_vol = np.where(roll_vol > 1e-8, roll_vol, fallback_vol)

    X = np.column_stack([returns, roll_vol])
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X_scaled = np.clip((X - mu) / sigma, -5.0, 5.0)

    n_states = min(n_states, max(2, n // 80))

    best_model = None
    best_score = -np.inf

    with _quiet_hmmlearn():
        for restart in range(HMM_N_RESTARTS):
            seed = 42 + restart
            try:
                km = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
                labels = km.fit_predict(X_scaled)
                startprob, trans, means, covars = _kmeans_hmm_init(
                    X_scaled, labels, n_states,
                )

                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type='diag',
                    n_iter=HMM_N_ITER,
                    tol=HMM_TOL,
                    min_covar=1e-2,
                    random_state=seed,
                    init_params='',
                    implementation='scaling',
                )
                model.startprob_ = startprob
                model.transmat_ = trans
                model.means_ = means
                model.covars_ = covars
                model.fit(X_scaled)

                score = model.score(X_scaled)
                if score > best_score:
                    best_score = score
                    best_model = model
            except (ValueError, np.linalg.LinAlgError):
                continue

    if best_model is None:
        return np.zeros(n, dtype=int)

    return best_model.predict(X_scaled)


def add_hmm_states(df, n_states=HMM_N_STATES):
    """
    Fits an HMM to returns + rolling volatility and predicts hidden states.
    """
    returns = df['Close'].pct_change().fillna(0).values
    df['HMM_State'] = _fit_hmm_states(returns, n_states)
    return df


def add_technical_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)

    return df


def preprocess_features(df):
    """
    Applies feature engineering for each stock separately.
    """
    processed_dfs = []
    for ticker, group in df.groupby('Ticker'):
        group = group.copy()
        group = add_fourier_features(group, window=30)
        group = add_hmm_states(group, n_states=HMM_N_STATES)
        group = add_technical_indicators(group)

        for i in range(1, 6):
            group[f'Close_Lag_{i}'] = group['Close'].shift(i)

        group.dropna(inplace=True)
        processed_dfs.append(group)

    return pd.concat(processed_dfs, ignore_index=True)
