import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from trading_rules import (
    MARKET_BEARISH_THRESHOLD,
    REBALANCE_DAYS,
    TRAILING_STOP_DD,
    TRAILING_STOP_RECOVERY,
    build_market_indicators,
    normalized_ranks,
    select_target_weights,
    apply_turnover_fee,
)


_DEFAULT_PARAMS = {
    'learning_rate': 1e-4,
    'ent_coef':      0.005,
    'clip_range':    0.2,
    'gae_lambda':    0.95,
    'n_epochs':      10,
    'net_arch_key':  'medium',
    'eval_patience': 5,
}

_NET_ARCHS = {
    'small':  [128, 64],
    'medium': [256, 128],
    'large':  [512, 256, 128],
    'xlarge': [512, 256, 256, 128],
}


class MultiStockTradingEnv(gym.Env):
    """
    RL environment aligned with backtest rules:
    max 5 assets (confidence-weighted), trend/vol/volume filters,
    transaction fees, trailing stop.
    """

    def __init__(self, df, start_date, end_date):
        super().__init__()
        pivoted = (
            df.pivot(index='Date', columns='Ticker', values='Close')
            .ffill().bfill().sort_index()
        )
        vol_pivoted = (
            df.pivot(index='Date', columns='Ticker', values='Volume')
            .ffill().bfill().sort_index()
        )
        mask = (pivoted.index >= start_date) & (pivoted.index <= end_date)
        self.dates = pivoted.index[mask]
        self.prices = pivoted.loc[mask].values.astype(np.float32)
        self.tickers = list(pivoted.columns)
        self.n_assets = len(self.tickers)
        self.n_steps = len(self.prices)
        self.window = 10

        self._vol_pivoted = vol_pivoted.loc[mask]
        self._indicators = build_market_indicators(
            pivoted.loc[:end_date], vol_pivoted.loc[:end_date],
        )

        self.action_space = spaces.Box(0, 1, (self.n_assets,), dtype=np.float32)
        obs_size = self.n_assets * self.window + self.n_assets * 2 + self.n_assets
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), dtype=np.float32)
        self._reset_state()

    def _reset_state(self):
        self.step_idx = self.window
        self.capital = 1000.0
        self.peak_capital = 1000.0
        self.weights = {'Cash': 1.0}
        self.trailing_stop_active = False
        self.recent = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def _obs(self):
        wp = self.prices[self.step_idx - self.window: self.step_idx]
        norm = (wp / (wp[0] + 1e-8)).flatten()
        if self.step_idx >= 5:
            r5 = (
                (self.prices[self.step_idx - 1] - self.prices[self.step_idx - 6])
                / (self.prices[self.step_idx - 6] + 1e-8)
            )
        else:
            r5 = np.zeros(self.n_assets, dtype=np.float32)
        dr = np.diff(wp, axis=0) / (wp[:-1] + 1e-8)
        vol = np.std(dr, axis=0).astype(np.float32)
        cur_w = np.array([self.weights.get(t, 0.0) for t in self.tickers], dtype=np.float32)
        cur_w /= cur_w.sum() + 1e-8
        return np.concatenate([norm, r5.astype(np.float32), vol, cur_w]).astype(np.float32)

    def _apply_backtest_rules(self, action: np.ndarray) -> dict:
        date_start = self.dates[self.step_idx]
        prices_s = pd.Series(self.prices[self.step_idx], index=self.tickers)
        mkt_trend = self._indicators['market_return_20d'].get(date_start, 0.0)
        market_bearish = bool(mkt_trend < MARKET_BEARISH_THRESHOLD)

        rl_alloc = {
            self.tickers[i]: float(action[i] / (action.sum() + 1e-8))
            for i in range(self.n_assets)
        }
        rl_ranks = normalized_ranks(rl_alloc, self.tickers)
        scores = {t: rl_ranks[t] for t in self.tickers}

        return select_target_weights(
            self.tickers, scores, date_start, prices_s,
            self._indicators, self._vol_pivoted,
            strategy='rl', market_bearish=market_bearish,
            rl_allocations=rl_alloc,
        )

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        target = self._apply_backtest_rules(action)

        fee, turnover = apply_turnover_fee(
            target, self.weights, self.tickers, self.capital,
        )
        self.capital -= fee
        self.weights = {t: target.get(t, 0.0) for t in self.tickers}
        self.weights['Cash'] = target.get('Cash', 0.0)

        step_end = min(self.step_idx + REBALANCE_DAYS, self.n_steps - 1)
        prices_start = self.prices[self.step_idx]
        prices_end = self.prices[step_end]

        ending = 0.0
        for i, ticker in enumerate(self.tickers):
            w = self.weights.get(ticker, 0.0)
            if w > 0:
                ret = (prices_end[i] - prices_start[i]) / (prices_start[i] + 1e-8)
                ending += w * self.capital * (1 + ret)
        ending += self.weights.get('Cash', 0.0) * self.capital

        port_ret = (ending - self.capital) / (self.capital + 1e-8)
        self.capital = ending

        self.peak_capital = max(self.peak_capital, self.capital)
        dd = (self.peak_capital - self.capital) / (self.peak_capital + 1e-8)
        if dd > TRAILING_STOP_DD:
            self.trailing_stop_active = True
        elif self.trailing_stop_active and self.capital >= self.peak_capital * TRAILING_STOP_RECOVERY:
            self.trailing_stop_active = False

        if self.trailing_stop_active:
            self.weights = {'Cash': 1.0}
            for t in self.tickers:
                self.weights[t] = 0.0

        self.recent.append(port_ret)
        if len(self.recent) > 20:
            self.recent.pop(0)
        if len(self.recent) > 2:
            arr = np.array(self.recent)
            dv = np.std(arr[arr < 0]) if (arr < 0).any() else 0.0
            reward = port_ret * 10.0 - 3.0 * dv - turnover * 0.5
        else:
            reward = port_ret * 10.0 - turnover * 0.5

        self.step_idx = step_end
        done = self.step_idx >= self.n_steps - REBALANCE_DAYS
        return self._obs(), float(reward), done, False, {}


def _make_env(df, start_date, end_date):
    def _init():
        return MultiStockTradingEnv(df, start_date, end_date)
    return _init


def train_rl_model(df, train_start, val_start, test_start,
                   hyperparams: dict = None, memory_manager=None):
    """Train PPO with backtest-aligned env and validation early stopping."""
    print(
        f"Training RL Model from {train_start.date()} to {test_start.date()} "
        f"(eval: {val_start.date()} → {test_start.date()})..."
    )

    p = {**_DEFAULT_PARAMS, **(hyperparams or {})}

    if memory_manager is not None:
        total_timesteps = memory_manager.rl_total_timesteps(base=80_000)
        batch_size = memory_manager.rl_batch_size(base=256)
        n_steps = memory_manager.rl_n_steps(base=1024)
        n_envs = memory_manager.rl_n_envs(base=8)
    else:
        total_timesteps = 80_000
        batch_size = 256
        n_steps = 1024
        n_envs = 1

    net_arch = _NET_ARCHS.get(p.get('net_arch_key', 'medium'), [256, 128])

    import threading
    in_daemon = threading.current_thread().daemon

    def _build_vec(start, end, env_n):
        fns = [_make_env(df, start, end) for _ in range(env_n)]
        if env_n > 1 and not in_daemon:
            import sys
            import torch as _torch
            start_method = 'spawn' if sys.platform == 'win32' or _torch.cuda.is_available() else 'fork'
            return SubprocVecEnv(fns, start_method=start_method)
        return DummyVecEnv(fns)

    train_env = _build_vec(train_start, test_start, n_envs)
    eval_env = DummyVecEnv([_make_env(df, val_start, test_start)])

    import torch as _torch
    _device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"  RL: total_timesteps={total_timesteps}, batch={batch_size}, "
          f"n_steps={n_steps}, n_envs={n_envs}, device={_device}")

    eval_freq = max(n_steps, total_timesteps // 15)
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=p['eval_patience'], min_evals=3,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        callback_on_new_best=stop_cb,
    )

    model = PPO(
        "MlpPolicy", train_env,
        verbose=0,
        device=_device,
        batch_size=batch_size,
        n_steps=n_steps,
        learning_rate=p['learning_rate'],
        ent_coef=p['ent_coef'],
        clip_range=p['clip_range'],
        n_epochs=p['n_epochs'],
        gae_lambda=p['gae_lambda'],
        policy_kwargs=dict(net_arch=net_arch),
    )
    try:
        import rich  # noqa: F401
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=use_progress_bar,
        )
    finally:
        train_env.close()
        eval_env.close()
        if _device.startswith('cuda'):
            from models.cuda_utils import release_cuda_memory
            release_cuda_memory()

    print("RL Model Training Completed.")
    return model
