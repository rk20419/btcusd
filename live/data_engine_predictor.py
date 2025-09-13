from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import ta

# sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.feature_selection import mutual_info_regression

# optional prometheus metrics
try:
    from prometheus_client import Gauge, start_http_server, Counter  # type: ignore
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

# tqdm optional
try:
    from tqdm import tqdm  # noqa: F401
except Exception:
    pass

# Constants
REQUIRED_COLS = {
    "datetime_ist", "open", "high", "low", "close",
    "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"
}
EPS = 1e-10
DEFAULT_TOP_K = 15
HEALTH_FILE = "processed/live/health.json"

# Predefined features from trainer
PREDEFINED_FEATURES = [
    "rsi_14", "macd", "macd_signal", "obv", "atr_14", "momentum_5",
    "taker_buy_ratio", "volume_spike_5", "large_trade_ratio",
    "session_asia", "session_london", "session_ny",
    "bullish_engulfing", "doji", "hammer", "shooting_star", "hidden_divergence",
    "weighted_rsi", "weighted_volume",
    "realized_vol_30", "garman_klass_30", "parkinson_30"
]

# -------------------------
# Logger
# -------------------------
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_path: str = "logs/data_engine_predictor.log") -> logging.Logger:
    Path(os.path.dirname(log_path) or ".").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("DataEnginePredictor")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

        fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return logger

# -------------------------
# Predictor class
# -------------------------
class DataEnginePredictor:
    def __init__(
        self,
        live_data_path: str = "data/live/1m.csv",
        output_path: str = "processed/live/live_features.csv",
        scaler_path: str = "scalers/scaler_canonical.pkl",
        feature_importance_path: str = "processed/feature_importance.csv",
        top_features_path: Optional[str] = None,
        log_path: str = "logs/data_engine_predictor.log",
        missing_threshold: float = 0.05,
        exp_lookback: int = 50,
        monitor_interval: int = 10,
        clip_bounds: Tuple[float, float] = (0.0, 1.0),
        save_scaled: bool = True,
        prometheus_port: Optional[int] = None,
        fallback_use_identity_scaler: bool = True,
        volatility_adjustment: bool = True,
        n_features: int = 15,
        mse_threshold: float = 0.1,
        version: str = "1.0.1",
    ):
        self.live_data_path = Path(live_data_path)
        self.output_path = Path(output_path)
        self.scaler_path = Path(scaler_path)
        self.feature_importance_path = Path(feature_importance_path)
        self.top_features_path = Path(top_features_path) if top_features_path else None
        self.missing_threshold = missing_threshold
        self.monitor_interval = max(1, int(monitor_interval))
        self.clip_lo, self.clip_hi = clip_bounds
        self.save_scaled = save_scaled
        self.prometheus_port = prometheus_port
        self.fallback_use_identity_scaler = fallback_use_identity_scaler
        self.volatility_adjustment = volatility_adjustment
        self.n_features = n_features
        self.mse_threshold = mse_threshold
        self.version = version
        self.volatility_factor = 1.0

        # Validate configuration
        self._validate_configuration()

        # make directories
        Path(self.output_path.parent).mkdir(parents=True, exist_ok=True)
        Path(self.live_data_path.parent).mkdir(parents=True, exist_ok=True)
        Path(self.scaler_path.parent).mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(log_path)

        # metrics
        self.last_good_df: Optional[pd.DataFrame] = None
        self.last_latency = None
        self.last_processed_ts: Optional[int] = None
        self.processed_count = 0
        self.error_count = 0

        # prom counters
        if PROM_AVAILABLE and self.prometheus_port:
            try:
                start_http_server(self.prometheus_port)
                self.logger.info(f"Prometheus metrics exported on port {self.prometheus_port}")
            except Exception as e:
                self.logger.warning(f"Failed to start Prometheus server: {e}")

        # Optional Gauges
        if PROM_AVAILABLE:
            self.gauge_latency = Gauge("data_engine_predictor_latency_seconds", "Latency seconds for processing")
            self.gauge_null_frac = Gauge("data_engine_predictor_null_fraction", "Null fraction in core features")
            self.gauge_processed_rows = Gauge("data_engine_processed_rows", "Rows processed in last run")
            self.gauge_volatility_factor = Gauge("data_engine_volatility_factor", "Current volatility adjustment factor")
            self.gauge_feature_coverage = Gauge("data_engine_feature_coverage", "Percentage of top features available")
            self.counter_processed = Counter("data_engine_processed_total", "Total processing cycles")
            self.counter_errors = Counter("data_engine_errors_total", "Total processing errors")
        else:
            self.gauge_latency = self.gauge_null_frac = self.gauge_processed_rows = None
            self.gauge_volatility_factor = self.gauge_feature_coverage = None
            self.counter_processed = self.counter_errors = None

        # exponential weights
        self.exp_weights = np.exp(np.linspace(-1, 0, exp_lookback))
        self.exp_weights /= self.exp_weights.sum()

        # load scaler & feature list
        self.scaler: Optional[MinMaxScaler] = None
        self.top_features: List[str] = []
        self._load_artifacts(exp_lookback, n_features)

        # watch file mtime
        self.last_mtime = self.live_data_path.stat().st_mtime if self.live_data_path.exists() else 0

    def _validate_configuration(self):
        """Validate configuration parameters for sanity"""
        if self.monitor_interval < 1:
            self.logger.warning(f"Monitor interval {self.monitor_interval} too low, setting to 1")
            self.monitor_interval = 1
        
        if not (0 <= self.clip_lo < self.clip_hi <= 1):
            raise ValueError(f"Invalid clip bounds: {self.clip_lo} <= lo < hi <= {self.clip_hi}")
        
        if self.missing_threshold <= 0 or self.missing_threshold > 0.5:
            raise ValueError(f"Missing threshold {self.missing_threshold} should be between 0 and 0.5")

    # -------------------------
    # Artifact loading
    # -------------------------
    def _load_artifacts(self, exp_lookback: int, default_k: int) -> None:
        # load scaler
        try:
            if self.scaler_path.exists():
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"Loaded scaler from {self.scaler_path}")
            else:
                raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load scaler: {e}")
            if self.fallback_use_identity_scaler:
                self.logger.warning("Using identity MinMaxScaler fallback (fit on data later).")
                self.scaler = MinMaxScaler()
            else:
                raise

        # load top features from feature_importance CSV (recommended)
        try:
            if self.feature_importance_path.exists():
                mi_df = pd.read_csv(self.feature_importance_path)
                if "feature" in mi_df.columns and "mi_score" in mi_df.columns:
                    self.top_features = mi_df.sort_values("mi_score", ascending=False)["feature"].head(default_k).tolist()
                elif "feature" in mi_df.columns:
                    self.top_features = mi_df["feature"].head(default_k).tolist()
                else:
                    # take first column
                    self.top_features = mi_df.iloc[:, 0].head(default_k).tolist()
                self.logger.info(f"Top features loaded from {self.feature_importance_path}: {self.top_features}")
            elif self.top_features_path and self.top_features_path.exists():
                # fallback: top_features.txt
                self.top_features = [l.strip() for l in open(self.top_features_path) if l.strip()]
                self.top_features = self.top_features[:default_k]
                self.logger.info(f"Top features loaded from {self.top_features_path}: {self.top_features}")
            else:
                raise FileNotFoundError("Feature importance not found")
        except Exception as e:
            self.logger.warning(f"Failed to load feature importance: {e}")
            # fallback to predefined from trainer
            self.top_features = PREDEFINED_FEATURES[:default_k]
            self.logger.warning(f"Falling back to default top features: {self.top_features}")

        if not self.top_features:
            self.logger.error("No top_features available after all fallbacks. Aborting.")
            raise RuntimeError("No top_features available")

    # -------------------------
    # Utilities
    # -------------------------
    def _downcast_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["float"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["integer"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        return df

    def _exp_weighted(self, arr: np.ndarray) -> float:
        n = len(arr)
        w = self.exp_weights[-n:]
        return float(np.dot(arr, w)) if n else np.nan

    # ---------- Volatility estimators ----------
    @staticmethod
    def realized_volatility(log_returns: pd.Series, window: int) -> pd.Series:
        # realized volatility: sqrt(sum(ret^2)) over window
        return np.sqrt(log_returns.pow(2).rolling(window).sum())

    @staticmethod
    def garman_klass(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        # Continuous variant of Garman-Klass estimator (variance), return sqrt to get vol
        rs = (high / low).apply(np.log)
        ks = (close / ((high + low) / 2)).apply(np.log)  # approximate
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

    def _calculate_volatility_factor(self, df: pd.DataFrame) -> float:
        """Calculate dynamic volatility adjustment factor matching trainer logic"""
        if len(df) < 100:
            return 1.0
        try:
            df["log_ret"] = np.log(df["close"]).diff()
            rv30 = self.realized_volatility(df["log_ret"].fillna(0.0), 30)
            q = max(0.01, min(0.99, 0.75))
            rv_q = np.nanquantile(rv30.replace(0, np.nan).dropna(), q)
            baseline = rv_q if not np.isnan(rv_q) else (rv30.mean() if len(rv30.dropna()) > 0 else 0.001)
            volatility_multiplier = float(np.clip(20 + 200 * baseline, 10, 200))

            high_low_range = (df['high'] - df['low']).tail(100)
            price_volatility = high_low_range.mean() / df['close'].iloc[-1]
            returns_volatility = df['close'].pct_change().tail(100).std()
            normalized_volatility = max(price_volatility, returns_volatility, baseline)
            return max(0.5, min(3.0, normalized_volatility * volatility_multiplier))
        except Exception as e:
            self.logger.warning(f"Volatility factor calculation failed: {e}")
            return 1.0

    # -------------------------
    # IO & Null handling
    # -------------------------
    def load_live_data(self) -> pd.DataFrame:
        if not self.live_data_path.exists():
            raise FileNotFoundError(f"Live data file not found: {self.live_data_path}")
        
        self.logger.info(f"Loading data from {self.live_data_path}")
        df = pd.read_csv(self.live_data_path)

        assert "datetime_ist" in df.columns, "datetime_ist column is missing from input data"
        df["datetime"] = pd.to_datetime(df["datetime_ist"], format="%Y-%m-%d %H:%M:%S")
        
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("Non-positive prices detected")

        df = df.sort_values("datetime").reset_index(drop=True)
        df = self._downcast_numeric(df)
        self.logger.info(f"Loaded {len(df)} rows successfully")
        return df

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

    # -------------------------
    # Feature calculation (safe)
    # -------------------------
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Calculating features...")

            # Calculate volatility factor for dynamic weighting
            if self.volatility_adjustment:
                self.volatility_factor = self._calculate_volatility_factor(df)
                self.logger.info(f"Volatility factor calculated: {self.volatility_factor:.3f}")

            df["returns"] = df["close"].pct_change()
            df["log_ret"] = np.log(df["close"]).diff()
            df["volatility"] = df["high"] - df["low"]

            # TA features
            def safe_assign(col_name: str, func):
                try:
                    df[col_name] = func()
                except Exception as e:
                    self.logger.warning(f"Indicator {col_name} failed: {e}")
                    df[col_name] = np.nan

            safe_assign("rsi_14", lambda: ta.momentum.RSIIndicator(df["close"], window=14).rsi())
            safe_assign("macd", lambda: ta.trend.MACD(df["close"]).macd())
            safe_assign("macd_signal", lambda: ta.trend.MACD(df["close"]).macd_signal())
            safe_assign("obv", lambda: ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume())
            safe_assign("atr_14", lambda: ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range())
            safe_assign("momentum_5", lambda: ta.momentum.ROCIndicator(df["close"], window=5).roc())

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

            # Hidden divergence
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
                        .apply(lambda x: self._exp_weighted(x) if len(x) > 0 else np.nan, raw=False)
                    )

            self.logger.info(f"Generated {len(df.columns)} columns (features + base)")
            return df
        except Exception as e:
            self.logger.error(f"Feature calculation error: {e}")
            raise

    # -------------------------
    # Post-feature impute/validation
    # -------------------------
    def impute_post_features(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if df[num_cols].isnull().any().any():
            self.logger.info("Post-feature imputation required - using small-window iterative/KNN imputer")
            # use recent window for speed
            window = max(150, len(self.exp_weights) + 50)
            sub = df[num_cols].tail(window)
            try:
                imputer = IterativeImputer(random_state=42, max_iter=10, initial_strategy="median")
                sub_imputed = pd.DataFrame(imputer.fit_transform(sub), columns=sub.columns, index=sub.index)
            except Exception as e:
                self.logger.warning(f"IterativeImputer failed: {e} - falling back to KNNImputer")
                imputer2 = KNNImputer()
                sub_imputed = pd.DataFrame(imputer2.fit_transform(sub), columns=sub.columns, index=sub.index)
            df.loc[sub_imputed.index, sub_imputed.columns] = sub_imputed

        # Check top features presence
        missing_top = [c for c in self.top_features if c not in df.columns]
        if missing_top:
            self.logger.warning(f"Top features missing after feature calc: {missing_top}")

        null_frac = df[self.top_features].isnull().mean().mean() if all(f in df.columns for f in self.top_features) else 1.0
        if null_frac > self.missing_threshold:
            self.logger.warning(f"High null fraction after impute: {null_frac:.2%} > {self.missing_threshold*100:.1f}%")
        return df

    # -------------------------
    # Normalize / scale
    # -------------------------
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure top features exist
        for f in self.top_features:
            if f not in df.columns:
                self.logger.debug(f"Filling missing feature {f} with NaN")
                df[f] = np.nan

        X = df[self.top_features].values
        if self.scaler is None:
            self.logger.warning("Scaler is None; fitting a temporary scaler on current data (not persisted)")
            scaler = MinMaxScaler()
            # handle NaNs by filling median for temporary fit
            X_fit = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
            scaler.fit(X_fit)
        else:
            scaler = self.scaler

        # If scaler not yet fitted (fallback), try to fit on non-nan recent rows
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            self.logger.warning(f"Scaler transform failed: {e} - attempting to fit scaler on data")
            # fit on non-nan rows
            try:
                valid_rows = ~np.isnan(X).any(axis=1)
                if valid_rows.sum() >= 10:
                    scaler.fit(X[valid_rows])
                    X_scaled = scaler.transform(np.where(np.isnan(X), 0.0, X))
                    self.logger.info("Fitted fallback scaler on recent non-nan rows")
                else:
                    # fill with zeros if not enough rows
                    self.logger.warning("Not enough valid rows to fit scaler; filling zeros")
                    X_scaled = np.zeros_like(X, dtype=float)
            except Exception as e2:
                self.logger.error(f"Fallback scaler fit failed: {e2}")
                X_scaled = np.zeros_like(X, dtype=float)

        # clip
        X_scaled = np.clip(X_scaled, self.clip_lo, self.clip_hi)

        # Build output
        out = df[["datetime_ist"]].copy()
        for i, col in enumerate(self.top_features):
            out[col] = df[col].values
        if self.save_scaled:
            for i, col in enumerate(self.top_features):
                out[f"{col}_scaled"] = X_scaled[:, i]
        return out

    # -------------------------
    # Process single update
    # -------------------------
    def process_update(self) -> bool:
        start_t = time.time()
        try:
            df = self.load_live_data()
            rows = len(df)
            self.logger.info(f"Loaded live data ({rows} rows) from {self.live_data_path}")

            df = self.handle_nulls(df, phase="basic")
            df = self.calculate_features(df)
            df = self.handle_nulls(df, phase="post_feat")
            out_df = self.normalize_features(df)

            # Calculate feature coverage for monitoring
            feature_coverage = (
                len([f for f in self.top_features if f in out_df.columns]) / 
                len(self.top_features) if self.top_features else 0
            )

            # Save output atomically
            tmp_fp = self.output_path.with_suffix(".tmp")
            out_df.to_csv(tmp_fp, index=False)
            tmp_fp.replace(self.output_path)
            latency = time.time() - start_t
            self.last_latency = latency
            self.last_good_df = out_df
            self.last_processed_ts = int(df["datetime"].max().timestamp()) if "datetime" in df.columns else None
            self.processed_count += 1

            # update health file
            self._write_health(success=True, latency=latency, rows=rows, feature_coverage=feature_coverage)

            # prometheus metrics
            if PROM_AVAILABLE:
                if self.gauge_latency:
                    self.gauge_latency.set(latency)
                if self.gauge_processed_rows:
                    self.gauge_processed_rows.set(rows)
                if self.gauge_null_frac:
                    null_frac = df[self.top_features].isnull().mean().mean() if all(f in df.columns for f in self.top_features) else 1.0
                    self.gauge_null_frac.set(float(null_frac))
                if self.gauge_volatility_factor:
                    self.gauge_volatility_factor.set(float(self.volatility_factor))
                if self.gauge_feature_coverage:
                    self.gauge_feature_coverage.set(float(feature_coverage))
                if self.counter_processed:
                    self.counter_processed.inc()

            self.logger.info(f"Processed & saved live features to {self.output_path} (latency {latency:.3f}s)")
            if latency > 1.0:
                self.logger.warning(f"High processing latency: {latency:.3f}s")

            return True
        except Exception as exc:
            self.logger.error(f"Processing update failed: {exc}\n{traceback.format_exc()}")
            self.error_count += 1
            
            # prometheus metrics
            if PROM_AVAILABLE and self.counter_errors:
                self.counter_errors.inc()
                
            # fallback: persist last good data if exists
            if self.last_good_df is not None:
                try:
                    self.last_good_df.to_csv(self.output_path, index=False)
                    self.logger.warning("Wrote last-good data as fallback output")
                except Exception as e2:
                    self.logger.error(f"Failed to write last-good fallback: {e2}")
            self._write_health(success=False, message=str(exc))
            return False

    def _write_health(self, success: bool, latency: Optional[float] = None, 
                     rows: Optional[int] = None, feature_coverage: Optional[float] = None,
                     message: Optional[str] = None):
        try:
            h = {
                "success": bool(success),
                "timestamp_utc": int(time.time()),
                "latency_s": float(latency) if latency is not None else None,
                "rows": int(rows) if rows is not None else None,
                "last_processed_timestamp": int(self.last_processed_ts) if self.last_processed_ts else None,
                "volatility_factor": float(self.volatility_factor),
                "feature_coverage": float(feature_coverage) if feature_coverage is not None else 0.0,
                "processed_count": int(self.processed_count),
                "error_count": int(self.error_count),
                "message": message or None,
                "version": self.version,
            }
            Path(HEALTH_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(HEALTH_FILE, "w") as hf:
                json.dump(h, hf, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed writing health file: {e}")

    # -------------------------
    # Run loop
    # -------------------------
    def run(self, once: bool = False):
        if once:
            self.logger.info("Running single update (once mode)")
            self.process_update()
            return

        self.logger.info("Starting continuous live monitor loop")
        try:
            while True:
                try:
                    if not self.live_data_path.exists():
                        self.logger.warning(f"Live file missing: {self.live_data_path}; sleeping {self.monitor_interval}s")
                        time.sleep(self.monitor_interval)
                        continue
                    mtime = self.live_data_path.stat().st_mtime
                    if mtime > self.last_mtime:
                        self.last_mtime = mtime
                        self.logger.info("Change detected in live CSV - processing")
                        self.process_update()
                    else:
                        self.logger.debug("No change detected")
                except Exception as e:
                    self.logger.error(f"Error in monitor loop: {e}\n{traceback.format_exc()}")
                time.sleep(self.monitor_interval)
        except KeyboardInterrupt:
            self.logger.info("Monitoring interrupted by user. Exiting gracefully.")
        except Exception as e:
            self.logger.error(f"Monitor fatal error: {e}\n{traceback.format_exc()}")
            raise

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Module 1 — Live Data Engine Predictor (production)")
    p.add_argument("--live_data_path", default="data/live/1m.csv")
    p.add_argument("--output_path", default="processed/live/live_features.csv")
    p.add_argument("--scaler_path", default="scalers/scaler_canonical.pkl")
    p.add_argument("--feature_importance_path", default="processed/feature_importance.csv")
    p.add_argument("--top_features_path", default=None)
    p.add_argument("--log_path", default="logs/data_engine_predictor.log")
    p.add_argument("--missing_threshold", type=float, default=0.05)
    p.add_argument("--exp_lookback", type=int, default=50)
    p.add_argument("--monitor_interval", type=int, default=10)
    p.add_argument("--clip_lo", type=float, default=0.0)
    p.add_argument("--clip_hi", type=float, default=1.0)
    p.add_argument("--once", action="store_true", help="Process a single update and exit")
    p.add_argument("--save_scaled", action="store_true", help="Persist scaled columns")
    p.add_argument("--no-save_scaled", dest="save_scaled", action="store_false")
    p.add_argument("--prometheus_port", type=int, default=None, help="Optional Prometheus exporter port")
    p.add_argument("--fallback_use_identity_scaler", action="store_true", help="Allow identity scaler fallback")
    p.add_argument("--volatility_adjustment", action="store_true", help="Enable dynamic volatility adjustment")
    p.add_argument("--no-volatility_adjustment", dest="volatility_adjustment", action="store_false")
    p.add_argument("--n_features", type=int, default=15, help="Number of top features to use")
    p.add_argument("--mse_threshold", type=float, default=0.1, help="MSE threshold for drift detection")
    p.add_argument("--version", type=str, default="1.0.1", help="Version identifier")
    p.set_defaults(save_scaled=True, fallback_use_identity_scaler=True, volatility_adjustment=True)
    return p.parse_args()

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    predictor = DataEnginePredictor(
        live_data_path=args.live_data_path,
        output_path=args.output_path,
        scaler_path=args.scaler_path,
        feature_importance_path=args.feature_importance_path,
        top_features_path=args.top_features_path,
        log_path=args.log_path,
        missing_threshold=args.missing_threshold,
        exp_lookback=args.exp_lookback,
        monitor_interval=args.monitor_interval,
        clip_bounds=(args.clip_lo, args.clip_hi),
        save_scaled=args.save_scaled,
        prometheus_port=args.prometheus_port,
        fallback_use_identity_scaler=args.fallback_use_identity_scaler,
        volatility_adjustment=args.volatility_adjustment,
        n_features=args.n_features,
        mse_threshold=args.mse_threshold,
        version=args.version,
    )
    predictor.run(once=args.once)