#trainer/data_engine_trainer.py
import argparse
import gc
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import pickle
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
import ta
from tqdm import tqdm

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REQUIRED_COLS = {
    "datetime_ist", "open", "high", "low", "close",
    "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"
}
EPS = 1e-10

PREDEFINED_FEATURES = [
    "rsi_14", "macd", "macd_signal", "obv", "atr_14", "momentum_5",
    "taker_buy_ratio", "volume_spike_5", "large_trade_ratio",
    "session_asia", "session_london", "session_ny",
    "bullish_engulfing", "hidden_divergence",
    "weighted_rsi", "weighted_volume",
    "realized_vol_30", "garman_klass_30", "parkinson_30"
]


def setup_logger(log_path: str):
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("DataEngineTrainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    fh = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


class DataEngineTrainer:
    def __init__(
        self,
        input_path: str,
        output_dir: str = "processed/",
        scaler_dir: str = "scalers/",
        chunk_size: int = 50_000,
        n_features: int = 15,
        n_folds: int = 5,
        missing_threshold: float = 0.05,
        exp_lookback: int = 50,
        oos_fraction: float = 0.2,
        mse_threshold: float = 0.1,
        convergence_tol: float = 0.01,
        save_scaled: bool = True,
        log_path: str = "logs/data_engine_trainer.log",
        version: str = "1.0.1",
        volatility_adjustment: bool = True,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.scaler_dir = scaler_dir
        self.chunk_size = chunk_size
        self.n_features = n_features
        self.n_folds = n_folds
        self.missing_threshold = missing_threshold
        self.exp_weights = np.exp(np.linspace(-1, 0, exp_lookback))
        self.exp_weights /= self.exp_weights.sum()
        self.oos_fraction = oos_fraction
        self.mse_threshold = mse_threshold
        self.convergence_tol = convergence_tol
        self.save_scaled = save_scaled
        self.version = version
        self.volatility_adjustment = volatility_adjustment

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.scaler_dir).mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(log_path)

        self.scaler: MinMaxScaler | None = None
        self.top_features: list[str] = []
        self.feature_scores: pd.Series | None = None
        self.volatility_factor: float = 1.0

    # --------- IO & Utilities ---------
    def _downcast_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["float"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["integer"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        return df

    def load_data(self) -> pd.DataFrame:
        try:
            self.logger.info(f"Loading data from {self.input_path}")
            df = pd.read_csv(self.input_path)

            assert "datetime_ist" in df.columns, "datetime_ist column is missing from input data"
            df["datetime"] = pd.to_datetime(df["datetime_ist"], format="%Y-%m-%d %H:%M:%S")
            assert REQUIRED_COLS.issubset(df.columns), f"Missing columns: {REQUIRED_COLS - set(df.columns)}"

            if (df[["open", "high", "low", "close"]] <= 0).any().any():
                raise ValueError("Non-positive prices detected")

            df = df.sort_values("datetime").reset_index(drop=True)
            df = self._downcast_numeric(df)
            self.logger.info(f"Loaded {len(df)} rows successfully")
            return df
        except Exception as e:
            self.logger.error(f"Failed loading data: {e}")
            raise

    def handle_nulls(self, df: pd.DataFrame, phase: str = "basic") -> pd.DataFrame:
        try:
            self.logger.info(f"Handling nulls ({phase})")
            core_cols = [
                "open", "high", "low", "close",
                "volume", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote",
            ]
            if phase == "basic":
                df[core_cols] = df[core_cols].ffill().bfill().interpolate(limit_direction="both")
                null_frac = df[core_cols].isnull().mean()
                if null_frac.any():
                    self.logger.warning(f"Core null fractions after fill: {null_frac.to_dict()}")
            else:
                num_cols = df.select_dtypes(include=[np.number]).columns
                if df[num_cols].isnull().any().any():
                    imputer = IterativeImputer(random_state=42, max_iter=10, initial_strategy="median")
                    df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols, index=df.index)
            null_frac_all = df.isnull().mean()
            drop_cols = null_frac_all[null_frac_all > self.missing_threshold].index.tolist()
            if drop_cols:
                self.logger.warning(
                    f"Dropping columns with >{self.missing_threshold*100:.1f}% nulls: {drop_cols}"
                )
                df = df.drop(columns=drop_cols)
            return df
        except Exception as e:
            self.logger.error(f"Null handling error: {e}")
            raise

    # ---------- Volatility estimators ----------
    @staticmethod
    def realized_volatility(log_returns: pd.Series, window: int) -> pd.Series:
        # realized volatility: sqrt(sum(ret^2)) over window
        return np.sqrt(log_returns.pow(2).rolling(window).sum())

    @staticmethod
    def garman_klass(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        # Continuous variant of Garman-Klass estimator (variance), return sqrt to get vol
        rs = (high / low).apply(np.log)
        ks = (close / ( (high + low) / 2 ) ).apply(np.log)  # approximate
        var = (0.5 * rs.pow(2) - (2 * np.log(2) - 1) * ks.pow(2)).rolling(window).mean()
        var = var.clip(lower=EPS)
        return np.sqrt(var)

    @staticmethod
    def parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        # Parkinson (1980) estimator
        rs = (high / low).apply(np.log)
        var = (1.0 / (4.0 * np.log(2))) * rs.pow(2).rolling(window).mean()
        var = var.clip(lower=EPS)
        return np.sqrt(var)

    # ---------- Feature Engineering ----------
    def _exp_weighted(self, arr: np.ndarray) -> float:
        n = len(arr)
        w = self.exp_weights[-n:]
        return float(np.dot(arr, w)) if n else np.nan

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Calculating features...")

            df["returns"] = df["close"].pct_change()
            df["log_ret"] = np.log(df["close"]).diff()
            df["volatility"] = df["high"] - df["low"]

            # TA features
            try:
                df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
            except Exception as e:
                self.logger.warning(f"RSI calc failed: {e}")
                df["rsi_14"] = np.nan

            try:
                macd_ind = ta.trend.MACD(df["close"])  # fast=12, slow=26, signal=9
                df["macd"] = macd_ind.macd()
                df["macd_signal"] = macd_ind.macd_signal()
            except Exception as e:
                self.logger.warning(f"MACD calc failed: {e}")
                df["macd"] = np.nan
                df["macd_signal"] = np.nan

            try:
                df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
            except Exception as e:
                self.logger.warning(f"OBV calc failed: {e}")
                df["obv"] = np.nan

            try:
                df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
            except Exception as e:
                self.logger.warning(f"ATR calc failed: {e}")
                df["atr_14"] = np.nan

            try:
                df["momentum_5"] = ta.momentum.ROCIndicator(df["close"], window=5).roc()
            except Exception as e:
                self.logger.warning(f"Momentum calc failed: {e}")
                df["momentum_5"] = np.nan

            # Microstructure / order-flow proxies
            df["taker_buy_ratio"] = df["taker_buy_base"] / (df["volume"] + EPS)
            df["volume_spike_5"] = df["volume"] / (df["volume"].rolling(5).mean().replace(0, EPS) + EPS)
            trades_roll_std = df["trades"].rolling(5).std().fillna(0.0)
            trades_roll_mean = df["trades"].rolling(5).mean().replace(0, EPS)
            df["large_trade_ratio"] = trades_roll_std / trades_roll_mean

            # Sessions (IST based)
            df["hour_ist"] = df["datetime"].dt.hour
            df["weekday"] = df["datetime"].dt.weekday
            df["session_asia"] = ((df["hour_ist"] >= 0) & (df["hour_ist"] < 8)).astype(int)
            df["session_london"] = ((df["hour_ist"] >= 8) & (df["hour_ist"] < 16)).astype(int)
            df["session_ny"] = ((df["hour_ist"] >= 16) & (df["hour_ist"] <= 23)).astype(int)

            # Extended candlestick patterns
            # Bullish Engulfing
            df["bullish_engulfing"] = (
                (df["close"] > df["open"]) &
                (df["close"].shift(1) < df["open"].shift(1)) &
                (df["close"] > df["open"].shift(1)) &
                (df["open"] < df["close"].shift(1))
            ).astype(int)

            # Doji
            df["doji"] = ( (df["close"] - df["open"]).abs() <= (df["high"] - df["low"]) * 0.1 ).astype(int)

            # Hammer / Hanging man (simple)
            body = (df["close"] - df["open"]).abs()
            lower_shadow = df["open"].where(df["close"]>=df["open"], df["close"]) - df["low"]
            df["hammer"] = ((lower_shadow > 2 * body) & (body <= (df["high"] - df["low"]) * 0.3)).astype(int)

            # Shooting Star
            upper_shadow = df["high"] - df["close"].where(df["close"]>=df["open"], df["open"])
            df["shooting_star"] = ((upper_shadow > 2 * body) & (body <= (df["high"] - df["low"]) * 0.3)).astype(int)

            # Hidden divergence (as before)
            df["price_low_14"] = df["low"].rolling(14).min()
            df["rsi_low_14"] = df["rsi_14"].rolling(14).min()
            df["hidden_divergence"] = (
                (df["low"] > df["price_low_14"].shift(1)) &
                (df["rsi_14"] < df["rsi_low_14"].shift(1))
            ).astype(int)

            # Realized / GK / Parkinson volatility over 30-periods (1m -> 30m)
            df["realized_vol_30"] = self.realized_volatility(df["log_ret"].fillna(0.0), 30)
            df["garman_klass_30"] = self.garman_klass(df["high"], df["low"], df["close"], 30)
            df["parkinson_30"] = self.parkinson(df["high"], df["low"], 30)

            # Exponentially weighted variants with volatility adjustment
            for c in ["rsi_14", "volume"]:
                if c in df.columns:
                    weight_factor = self.volatility_factor if self.volatility_adjustment else 1.0
                    adjusted_weights = self.exp_weights ** weight_factor
                    adjusted_weights /= adjusted_weights.sum()

                    df[f"weighted_{c}"] = (
                        df[c].rolling(len(adjusted_weights), min_periods=1)
                        .apply(lambda x: float(np.dot(x, adjusted_weights[-len(x):])) if len(x) > 0 else np.nan, raw=False)
                    )

            self.logger.info(f"Generated {len(df.columns)} columns (features + base)")
            return df
        except Exception as e:
            self.logger.error(f"Feature calculation error: {e}")
            raise

    # ---------- Feature Selection ----------
    def select_top_features(self, df: pd.DataFrame, directional: bool = False) -> list[str]:
        try:
            self.logger.info("Selecting top features using mutual information with returns")
            target = df["returns"] if directional else df["returns"].abs()

            exclude = ["datetime", "hour_ist", "returns", "volatility", "log_ret", "week_day"]
            feature_cols = [col for col in df.columns if col not in exclude and df[col].dtype != "O"]

            available_features = [f for f in PREDEFINED_FEATURES if f in feature_cols]
            self.logger.info(f"Available predefined features: {len(available_features)}/{len(PREDEFINED_FEATURES)}")

            X = df[available_features].shift(1)
            y = target

            valid_mask = X.notna().all(axis=1) & y.notna()
            Xf = X.loc[valid_mask]
            yf = y.loc[valid_mask]

            if Xf.isnull().any().any():
                imputer = IterativeImputer(random_state=42, max_iter=10, initial_strategy="median")
                Xf = pd.DataFrame(imputer.fit_transform(Xf), columns=available_features, index=Xf.index)

            mi_scores = mutual_info_regression(Xf, yf, random_state=42)
            mi_series = pd.Series(mi_scores, index=available_features, name="mi_score")
            mi_series = mi_series.sort_values(ascending=False)

            self.feature_scores = mi_series
            self.top_features = mi_series.head(self.n_features).index.tolist()

            out_fp = os.path.join(self.output_dir, "feature_importance.csv")
            mi_series.to_csv(out_fp, index_label="feature")
            self.logger.info(f"Top {self.n_features} features selected and saved -> {out_fp}")
            return self.top_features
        except Exception as e:
            self.logger.error(f"Feature selection error: {e}")
            self.top_features = [f for f in PREDEFINED_FEATURES if f in df.columns][:self.n_features]
            self.logger.warning(f"Using predefined features due to error: {self.top_features}")
            return self.top_features

    # ---------- Scaler & Validation ----------
    def fit_scaler(self, df: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
        try:
            X = df[feature_cols].values
            scaler = MinMaxScaler()

            tscv = TimeSeriesSplit(n_splits=self.n_folds)
            prev_bounds = None
            last_train = None
            for fold, (train_idx, _) in enumerate(tscv.split(X), start=1):
                last_train = X[train_idx]
                cur_min = np.nanmin(last_train, axis=0)
                cur_max = np.nanmax(last_train, axis=0)
                bounds = np.concatenate([cur_min, cur_max])
                if prev_bounds is not None and np.all(np.abs(bounds - prev_bounds) < self.convergence_tol):
                    self.logger.info(f"Early stopping at fold {fold}: scaler bounds converged")
                    break
                prev_bounds = bounds

            scaler.fit(last_train if last_train is not None else X)
            self.scaler = scaler
            return scaler
        except Exception as e:
            self.logger.error(f"Scaler fitting error: {e}")
            raise

    def monitor_drift(self, df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list[str]) -> float:
        try:
            X = df[feature_cols].values
            X_scaled = scaler.transform(X)
            mse = float(np.mean((X_scaled - 0.5) ** 2))
            if mse > self.mse_threshold:
                self.logger.warning(f"Data drift detected (MSE from 0.5) = {mse:.4f}")
            else:
                self.logger.info(f"No significant data drift (MSE from 0.5) = {mse:.4f}")
            return mse
        except Exception as e:
            self.logger.error(f"Drift monitoring error: {e}")
            raise

    # ---------- Batch Processing ----------
    def process_in_batches(self, df: pd.DataFrame, batch_size: int = 10_000) -> pd.DataFrame:
        results = []
        n_batches = int(np.ceil(len(df) / batch_size))
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(df))
            batch = df.iloc[start:end].copy()
            self.logger.info(f"Processing batch {i + 1}/{n_batches} (rows {start}:{end})")
            batch = self.handle_nulls(batch, phase="post_feat")
            results.append(batch)
            del batch
            gc.collect()
        return pd.concat(results, ignore_index=True)

    # ---------- Feature details ----------
    def generate_feature_details(self, out_fp: str):
        # Produce a CSV describing features and logic
        rows = []
        def add(name, category, logic, purpose):
            rows.append({"feature_name": name, "category": category, "logic": logic, "purpose": purpose})

        add("rsi_14", "technical", "TA RSI 14-period", "Momentum / overbought-oversold")
        add("macd", "technical", "MACD line (12,26)", "Trend strength")
        add("macd_signal", "technical", "MACD signal (9)", "Trend signal")
        add("obv", "volume", "On-Balance Volume", "Volume flow direction")
        add("atr_14", "volatility", "Average True Range 14", "Average range for SL/TP and volatility scaling")
        add("momentum_5", "technical", "5-period ROC", "Short-term momentum")
        add("taker_buy_ratio", "microstructure", "taker_buy_base / volume", "Buy pressure proxy")
        add("volume_spike_5", "microstructure", "volume / rolling_mean(5)", "Detect volume surges")
        add("large_trade_ratio", "microstructure", "rolling_std(trades,5)/rolling_mean(trades,5)", "Trade-size volatility proxy")
        add("session_asia", "session", "IST hour in [0,7]", "Session indicator for sizing/liquidity")
        add("session_london", "session", "IST hour in [8,15]", "Session indicator for sizing/liquidity")
        add("session_ny", "session", "IST hour in [16,23]", "Session indicator for sizing/liquidity")
        add("bullish_engulfing", "price_action", "two-candle engulfing pattern", "Entry pattern signal")
        add("doji", "price_action", "body small relative to range", "Indecision candle")
        add("hammer", "price_action", "long lower shadow relative to body", "Reversal signal")
        add("shooting_star", "price_action", "long upper shadow relative to body", "Reversal signal")
        add("hidden_divergence", "price_action", "price higher low & RSI lower low", "Bullish hidden divergence")
        add("weighted_rsi", "technical", "exp-weighted rsi adjusted by volatility", "Recency-weighted RSI")
        add("weighted_volume", "volume", "exp-weighted volume adjusted by volatility", "Recency-weighted volume")
        add("realized_vol_30", "volatility", "sqrt(sum(log_ret^2) over 30)", "Short-term realized volatility")
        add("garman_klass_30", "volatility", "Garman-Klass estimator over 30", "Range-based volatility estimator")
        add("parkinson_30", "volatility", "Parkinson estimator over 30", "High-low based volatility estimator")

        df = pd.DataFrame(rows)
        df.to_csv(out_fp, index=False)
        self.logger.info(f"Feature details saved -> {out_fp}")

    # ---------- Main runner ----------
    def process(self):
        try:
            df = self.load_data()
            df = self.handle_nulls(df, phase="basic")

            # Pre-calc volatility measures to decide adaptive multiplier
            # compute realized volatility (30) over log returns after minimal cleaning
            df["log_ret"] = np.log(df["close"]).diff()
            rv30 = self.realized_volatility(df["log_ret"].fillna(0.0), 30)
            # choose a robust quantile-based multiplier
            q = max(0.01, min(0.99, 0.75))
            rv_q = np.nanquantile(rv30.replace(0, np.nan).dropna(), q)
            baseline = rv_q if not np.isnan(rv_q) else (rv30.mean() if len(rv30.dropna())>0 else 0.001)
            # adaptive multiplier: smaller in low vol, larger in high vol
            volatility_multiplier = float(np.clip(20 + 200 * baseline, 10, 200))
            self.logger.info(f"Adaptive volatility_multiplier set to {volatility_multiplier:.4f}")

            # calculate features (this will use volatility_factor but we set initial estimate)
            # initial normalized volatility estimate
            high_low_range = (df['high'] - df['low']).tail(100)
            price_volatility = high_low_range.mean() / df['close'].iloc[-1]
            returns_volatility = df['close'].pct_change().tail(100).std()
            normalized_volatility = max(price_volatility, returns_volatility, baseline)
            self.volatility_factor = float(max(0.5, min(3.0, normalized_volatility * volatility_multiplier)))
            self.logger.info(f"Initial volatility_factor: {self.volatility_factor:.4f}")

            df = self.calculate_features(df)
            df = self.handle_nulls(df, phase="post_feat")

            df_processed = self.process_in_batches(df)

            top_features = self.select_top_features(df_processed, directional=False)

            # Fit canonical scaler using top features
            scaler = self.fit_scaler(df_processed, top_features)

            # Drift check
            self.monitor_drift(df_processed, scaler, top_features)

            # Manifest / schema
            manifest = {
                "version": self.version,
                "purpose": "Module 1: Data Engine Trainer - Feature engineering and normalization for BTC/USD AI Trading System",
                "input_data": self.input_path,
                "top_features": top_features,
                "n_features": len(top_features),
                "feature_selection_method": "mutual_information_with_returns_abs",
                "exp_lookback": len(self.exp_weights),
                "save_scaled": self.save_scaled,
                "volatility_factor": self.volatility_factor,
                "missing_value_threshold": self.missing_threshold,
                "scaler_type": "MinMaxScaler",
                "output_description": "Processed data chunks with normalized features for downstream modules",
                "next_steps": "Use output chunks with Module 2 (Regime Detector) and Module 3 (Specialist Team)",
                "notes": "Features selected by mutual information with absolute returns. Volatility estimators: realized,gk,parkinson",
                "time_reference": "datetime_ist (Indian Standard Time)"
            }
            with open(os.path.join(self.output_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

            # Save chunks (scaled by canonical scaler)
            n_chunks = int(np.ceil(len(df_processed) / self.chunk_size))
            for i in tqdm(range(n_chunks), desc="Processing chunks"):
                start = i * self.chunk_size
                end = min((i + 1) * self.chunk_size, len(df_processed))
                chunk = df_processed.iloc[start:end].copy()
                if len(chunk) < 100:
                    continue

                output_cols = ["datetime_ist"] + top_features
                out_df = chunk[output_cols].copy()

                if self.save_scaled and self.scaler is not None:
                    X_scaled = self.scaler.transform(chunk[top_features].values)
                    scaled_cols = [f"{c}_scaled" for c in top_features]
                    for j, col in enumerate(scaled_cols):
                        out_df[col] = X_scaled[:, j]

                out_fp = os.path.join(self.output_dir, f"chunk_{i}.csv")
                out_df.to_csv(out_fp, index=False)
                self.logger.info(f"Saved chunk {i} with {len(top_features)} features (scaled={self.save_scaled})")

            # Save canonical scaler
            if self.scaler is not None:
                with open(os.path.join(self.scaler_dir, "scaler_canonical.pkl"), "wb") as f:
                    pickle.dump(self.scaler, f)

            # Save top-features text
            with open(os.path.join(self.output_dir, "top_features.txt"), "w") as f:
                f.write("Top Features Selected for Model Training\n")
                f.write("========================================\n\n")
                f.write(f"Selection Method: Mutual Information with Absolute Returns\n")
                f.write(f"Total Features Selected: {len(top_features)}\n")
                f.write(f"Volatility Factor: {self.volatility_factor:.4f}\n")
                f.write(f"Time Reference: datetime_ist (Indian Standard Time)\n\n")
                f.write("Feature List (in order of importance):\n")
                f.write("-------------------------------------\n")
                for ii, feat in enumerate(top_features, 1):
                    f.write(f"{ii:2d}. {feat}\n")

            # Generate feature details CSV
            self.generate_feature_details(os.path.join(self.output_dir, "details.csv"))

            self.logger.info("DataEngineTrainer completed successfully")
        except Exception as e:
            self.logger.error(f"DataEngineTrainer failed: {e}")
            raise


# ---------- CLI ----------

def parse_args_train():
    p = argparse.ArgumentParser(description="Module 1 — Data Engine Trainer (Production)")
    p.add_argument("--input_path", default="data/historical/BTCUSDT_1m_200000.csv")
    p.add_argument("--output_dir", default="processed/")
    p.add_argument("--scaler_dir", default="scalers/")
    p.add_argument("--chunk_size", type=int, default=50_000)
    p.add_argument("--n_features", type=int, default=15)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--missing_threshold", type=float, default=0.05)
    p.add_argument("--exp_lookback", type=int, default=50)
    p.add_argument("--oos_fraction", type=float, default=0.2)
    p.add_argument("--mse_threshold", type=float, default=0.1)
    p.add_argument("--convergence_tol", type=float, default=0.01)
    p.add_argument("--save_scaled", action="store_true")
    p.add_argument("--no-save_scaled", dest="save_scaled", action="store_false")
    p.add_argument("--volatility_adjustment", action="store_true")
    p.add_argument("--no-volatility_adjustment", dest="volatility_adjustment", action="store_false")
    p.set_defaults(
        save_scaled=True,
        volatility_adjustment=True
    )
    p.add_argument("--log_path", default="logs/data_engine_trainer.log")
    p.add_argument("--version", default="1.0.1")
    return p.parse_args()


if __name__ == "__main__" and os.getenv("MODULE_ENTRY", "train") == "train":
    args = parse_args_train()
    trainer = DataEngineTrainer(
        input_path=args.input_path,
        output_dir=args.output_dir,
        scaler_dir=args.scaler_dir,
        chunk_size=args.chunk_size,
        n_features=args.n_features,
        n_folds=args.n_folds,
        missing_threshold=args.missing_threshold,
        exp_lookback=args.exp_lookback,
        oos_fraction=args.oos_fraction,
        mse_threshold=args.mse_threshold,
        convergence_tol=args.convergence_tol,
        save_scaled=args.save_scaled,
        log_path=args.log_path,
        version=args.version,
        volatility_adjustment=args.volatility_adjustment,
    )
    trainer.process()
