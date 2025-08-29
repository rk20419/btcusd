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
from typing import List, Tuple

import numpy as np
import pandas as pd
import ta
from tqdm import tqdm

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# ----------------------
# Global settings
# ----------------------
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REQUIRED_COLS = {
    "timestamp", "open", "high", "low", "close",
    "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"
}
EPS = 1e-10


def setup_logger(log_path: str):
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("DataEngineTrainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
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
        target_move: float = 300.0,
        prediction_window: int = 5,
        oos_fraction: float = 0.2,
        mse_threshold: float = 0.1,
        convergence_tol: float = 0.01,
        save_scaled: bool = True,
        log_path: str = "logs/data_engine_trainer.log",
        version: str = "1.0.0",
        use_permutation_importance: bool = True,
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
        self.target_move = target_move
        self.prediction_window = prediction_window
        self.oos_fraction = oos_fraction
        self.mse_threshold = mse_threshold
        self.convergence_tol = convergence_tol
        self.save_scaled = save_scaled
        self.version = version
        self.use_permutation_importance = use_permutation_importance
        self.volatility_adjustment = volatility_adjustment

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.scaler_dir).mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(log_path)

        self.scalers = {}
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
            assert REQUIRED_COLS.issubset(df.columns), f"Missing columns: {REQUIRED_COLS - set(df.columns)}"
            if (df[["open", "high", "low", "close"]] <= 0).any().any():
                raise ValueError("Non-positive prices detected")

            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["datetime_ist"] = df["datetime"].dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d %H:%M:%S")
            df = df.sort_values("datetime").reset_index(drop=True)
            df = self._downcast_numeric(df)
            
            # Calculate initial volatility for dynamic adjustments
            if self.volatility_adjustment:
                recent_volatility = df['high'].tail(100) - df['low'].tail(100)
                self.volatility_factor = max(0.5, min(2.0, recent_volatility.mean() / 100))
                self.logger.info(f"Volatility factor calculated: {self.volatility_factor:.4f}")
                
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
                # Post-feature phase: MICE imputation for numerical columns
                num_cols = df.select_dtypes(include=[np.number]).columns
                if df[num_cols].isnull().any().any():
                    imputer = IterativeImputer(random_state=42, max_iter=10, initial_strategy="median")
                    df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols, index=df.index)
            # Drop columns with > threshold missing (precautionary)
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

    # ---------- Feature Engineering ----------
    def _exp_weighted(self, arr: np.ndarray) -> float:
        # Align weights to the right (recent candles get larger weights)
        n = len(arr)
        w = self.exp_weights[-n:]
        return float(np.dot(arr, w)) if n else np.nan

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Calculating features...")
            # Price & returns
            df["returns"] = df["close"].pct_change()
            df["volatility"] = df["high"] - df["low"]  # High-Low range

            # TA features - protected calls (single-indicator failure shouldn't crash pipeline)
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

            try:
                df["mfi_14"] = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=14).money_flow_index()
            except Exception as e:
                self.logger.warning(f"MFI calc failed: {e}")
                df["mfi_14"] = np.nan

            try:
                stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
                df["stoch_k"] = stoch.stoch()
                df["stoch_d"] = stoch.stoch_signal()
            except Exception as e:
                self.logger.warning(f"Stochastic calc failed: {e}")
                df["stoch_k"] = np.nan
                df["stoch_d"] = np.nan

            try:
                df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
                df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
            except Exception as e:
                self.logger.warning(f"EMA calc failed: {e}")
                df["ema_9"] = np.nan
                df["ema_21"] = np.nan

            try:
                bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
                df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df["close"] + EPS)
            except Exception as e:
                self.logger.warning(f"Bollinger calc failed: {e}")
                df["bb_width"] = np.nan

            # Microstructure / order-flow proxies
            df["taker_buy_ratio"] = df["taker_buy_base"] / (df["volume"] + EPS)
            df["volume_spike_5"] = df["volume"] / (df["volume"].rolling(5).mean().replace(0, EPS) + EPS)
            # large_trade_ratio: rolling std of trades / rolling mean of trades (windowed)
            trades_roll_std = df["trades"].rolling(5).std().fillna(0.0)
            trades_roll_mean = df["trades"].rolling(5).mean().replace(0, EPS)
            df["large_trade_ratio"] = trades_roll_std / trades_roll_mean

            # Sessions (IST based)
            df["hour_ist"] = pd.to_datetime(df["datetime_ist"]).dt.hour
            df["session_asia"] = ((df["hour_ist"] >= 0) & (df["hour_ist"] < 8)).astype(int)
            df["session_london"] = ((df["hour_ist"] >= 8) & (df["hour_ist"] < 16)).astype(int)
            df["session_ny"] = ((df["hour_ist"] >= 16) & (df["hour_ist"] <= 23)).astype(int)

            # Price action flags
            df["bullish_engulfing"] = (
                (df["close"] > df["open"]) &
                (df["close"].shift(1) < df["open"].shift(1)) &
                (df["close"] > df["open"].shift(1)) &
                (df["open"] < df["close"].shift(1))
            ).astype(int)

            df["price_low_14"] = df["low"].rolling(14).min()
            df["rsi_low_14"] = df["rsi_14"].rolling(14).min()
            df["hidden_divergence"] = (
                (df["low"] > df["price_low_14"].shift(1)) &
                (df["rsi_14"] < df["rsi_low_14"].shift(1))
            ).astype(int)

            # Exponentially weighted variants with volatility adjustment
            for c in ["rsi_14", "volume", "returns", "volatility"]:
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

    # ---------- Targets ----------
    def _forward_rolling_max(self, s: pd.Series, window: int) -> pd.Series:
        # Max over [t+1, t+window]
        rev = s.iloc[::-1]
        fwd = rev.rolling(window=window, min_periods=1).max().iloc[::-1]
        return fwd.shift(-1)

    def _forward_rolling_min(self, s: pd.Series, window: int) -> pd.Series:
        # Min over [t+1, t+window]
        rev = s.iloc[::-1]
        fwd = rev.rolling(window=window, min_periods=1).min().iloc[::-1]
        return fwd.shift(-1)

    def assign_target(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Assigning target variables (future window) — full-data operation")
            
            # Dynamic target adjustment based on volatility
            adjusted_target_move = self.target_move * self.volatility_factor
            
            future_max = self._forward_rolling_max(df["close"], self.prediction_window)
            future_min = self._forward_rolling_min(df["close"], self.prediction_window)

            df["target_buy"] = (future_max >= df["close"] + adjusted_target_move).astype(int)
            df["target_sell"] = (future_min <= df["close"] - adjusted_target_move).astype(int)
            # drop rows that cannot have forward window assigned (at tail)
            df = df.dropna(subset=["target_buy", "target_sell"]).reset_index(drop=True)

            self.logger.info(f"Adjusted target move: {adjusted_target_move:.2f} (volatility factor: {self.volatility_factor:.2f})")
            self.logger.info(f"Buy target positive ratio: {df['target_buy'].mean():.2%}")
            self.logger.info(f"Sell target positive ratio: {df['target_sell'].mean():.2%}")
            return df
        except Exception as e:
            self.logger.error(f"Target assignment error: {e}")
            raise

    # ---------- Feature Selection ----------
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> pd.Series:
        """Calculate feature importance using multiple methods"""
        # Method 1: Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=feature_names, name="mi_score")
        
        if self.use_permutation_importance:
            # Method 2: Permutation Importance
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X, y)
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            perm_series = pd.Series(perm_importance.importances_mean, index=feature_names, name="perm_score")
            
            # Combine scores (weighted average)
            combined_scores = (mi_series + perm_series) / 2
            return combined_scores
        else:
            return mi_series

    def select_top_features(self, df: pd.DataFrame, target_cols=("target_buy", "target_sell")) -> list[str]:
        try:
            self.logger.info("Selecting top features via advanced feature importance (leak-safe)")
            exclude = ["timestamp", "datetime", "datetime_ist", "hour_ist"] + list(target_cols)
            # exclude any already-scaled columns if present
            exclude += [c for c in df.columns if c.endswith("_scaled")]

            feature_cols = [col for col in df.columns if col not in exclude and df[col].dtype != "O"]

            # Leak-safe shift: use t-1 features to predict t target
            X = df[feature_cols].shift(1)
            y_buy = df[target_cols[0]]
            y_sell = df[target_cols[1]]

            valid_mask = X.notna().all(axis=1) & y_buy.notna() & y_sell.notna()
            Xf = X.loc[valid_mask]
            yb = y_buy.loc[valid_mask]
            ys = y_sell.loc[valid_mask]

            # Impute if necessary
            if Xf.isnull().any().any():
                imputer = IterativeImputer(random_state=42, max_iter=10, initial_strategy="median")
                Xf = pd.DataFrame(imputer.fit_transform(Xf), columns=feature_cols, index=Xf.index)

            # Calculate feature importance for both targets
            importance_buy = self.calculate_feature_importance(Xf, yb, feature_cols)
            importance_sell = self.calculate_feature_importance(Xf, ys, feature_cols)
            
            # Combine importance scores
            combined_importance = (importance_buy + importance_sell) / 2.0
            combined_series = combined_importance.sort_values(ascending=False)

            self.feature_scores = combined_series
            self.top_features = combined_series.head(self.n_features).index.tolist()

            # Persist feature importance report
            out_fp = os.path.join(self.output_dir, "feature_importance.csv")
            combined_series.to_csv(out_fp, index_label="feature")
            self.logger.info(f"Top {self.n_features} features selected and saved -> {out_fp}")
            return self.top_features
        except Exception as e:
            self.logger.error(f"Feature selection error: {e}")
            raise

    # ---------- Scaler & Validation ----------
    def fit_scaler(self, df: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
        try:
            X = df[feature_cols].values
            scaler = MinMaxScaler()

            # Early convergence: check running min/max stability across folds
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
            return scaler
        except Exception as e:
            self.logger.error(f"Scaler fitting error: {e}")
            raise

    def monitor_drift(self, df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list[str]) -> float:
        try:
            X = df[feature_cols].values
            X_scaled = scaler.transform(X)
            # Use midpoint (0.5) as reference (MinMaxScaler maps training min->0, max->1)
            mse = float(np.mean((X_scaled - 0.5) ** 2))
            if mse > self.mse_threshold:
                self.logger.warning(f"Data drift detected (MSE from 0.5) = {mse:.4f}")
            else:
                self.logger.info(f"No significant data drift (MSE from 0.5) = {mse:.4f}")
            return mse
        except Exception as e:
            self.logger.error(f"Drift monitoring error: {e}")
            raise

    def validate_oos(self, df: pd.DataFrame, feature_cols: list[str], scaler: MinMaxScaler) -> None:
        try:
            self.logger.info("Performing out-of-sample validation")
            train_df, oos_df = train_test_split(df, test_size=self.oos_fraction, shuffle=False)
            X_oos = oos_df[feature_cols].values
            X_oos_scaled = scaler.transform(X_oos)
            out_of_bounds = (X_oos_scaled < -0.01) | (X_oos_scaled > 1.01)
            if np.any(out_of_bounds):
                self.logger.warning("Out-of-sample scaling out of bounds detected")
            else:
                self.logger.info("Out-of-sample validation passed")
            self.logger.info(f"OOS Buy target ratio: {oos_df['target_buy'].mean():.2%}")
            self.logger.info(f"OOS Sell target ratio: {oos_df['target_sell'].mean():.2%}")
        except Exception as e:
            self.logger.error(f"OOS validation error: {e}")
            raise

    # ---------- Batch Processing ----------
    def process_in_batches(self, df: pd.DataFrame, batch_size: int = 10_000) -> pd.DataFrame:
        """
        NOTE: df must already include target columns (assign_target called on full df).
        This method applies post-feature null handling and splits into batches to reduce peak memory
        while returning the concatenated processed rows.
        """
        results = []
        n_batches = int(np.ceil(len(df) / batch_size))
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(df))
            batch = df.iloc[start:end].copy()
            self.logger.info(f"Processing batch {i + 1}/{n_batches} (rows {start}:{end})")
            batch = self.handle_nulls(batch, phase="post_feat")
            # targets are already assigned globally; don't recompute targets per-batch
            results.append(batch)
            del batch
            gc.collect()
        return pd.concat(results, ignore_index=True)

    # ---------- Main runner ----------
    def process(self):
        try:
            df = self.load_data()
            df = self.handle_nulls(df, phase="basic")
            df = self.calculate_features(df)
            df = self.handle_nulls(df, phase="post_feat")

            # IMPORTANT: compute targets on the full dataframe BEFORE chunking/batching
            df = self.assign_target(df)

            # Build supervised dataset (with targets) in memory-efficient batches
            df_processed = self.process_in_batches(df)

            # Feature selection
            top_features = self.select_top_features(df_processed)

            # Fit scaler on selected features
            scaler = self.fit_scaler(df_processed, top_features)

            # Drift + OOS checks
            self.monitor_drift(df_processed, scaler, top_features)
            self.validate_oos(df_processed, top_features, scaler)

            # Persist artifacts & chunks
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.scaler_dir).mkdir(parents=True, exist_ok=True)

            # Manifest / schema
            manifest = {
                "version": self.version,
                "top_features": top_features,
                "n_features": len(top_features),
                "target_move": self.target_move,
                "prediction_window": self.prediction_window,
                "exp_lookback": len(self.exp_weights),
                "save_scaled": self.save_scaled,
                "volatility_factor": self.volatility_factor,
            }
            with open(os.path.join(self.output_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

            # Save chunks with both raw + scaled (optional)
            n_chunks = int(np.ceil(len(df_processed) / self.chunk_size))
            for i in tqdm(range(n_chunks), desc="Processing chunks"):
                start = i * self.chunk_size
                end = min((i + 1) * self.chunk_size, len(df_processed))
                chunk = df_processed.iloc[start:end].copy()
                if len(chunk) < 100:
                    continue

                output_cols = ["timestamp", "datetime_ist"] + top_features + ["target_buy", "target_sell"]
                out_df = chunk[output_cols].copy()

                if self.save_scaled:
                    X_scaled = scaler.transform(chunk[top_features].values)
                    scaled_cols = [f"{c}_scaled" for c in top_features]
                    for j, col in enumerate(scaled_cols):
                        out_df[col] = X_scaled[:, j]

                out_fp = os.path.join(self.output_dir, f"chunk_{i}.csv")
                out_df.to_csv(out_fp, index=False)

                # Persist the (same) scaler for reproducibility (named by chunk)
                with open(os.path.join(self.scaler_dir, f"scaler_{i}.pkl"), "wb") as f:
                    pickle.dump(scaler, f)

                self.logger.info(f"Saved chunk {i} with {len(top_features)} features (scaled={self.save_scaled})")

            # Also save top-features list for convenience
            with open(os.path.join(self.output_dir, "top_features.txt"), "w") as f:
                for feat in top_features:
                    f.write(f"{feat}\n")

            # Save a single canonical scaler too
            with open(os.path.join(self.scaler_dir, "scaler_canonical.pkl"), "wb") as f:
                pickle.dump(scaler, f)

            self.logger.info("DataEngineTrainer completed successfully")
        except Exception as e:
            self.logger.error(f"DataEngineTrainer failed: {e}")
            raise


# ---------- CLI ----------

def parse_args_train():
    p = argparse.ArgumentParser(description="Module 1 — Data Engine Trainer")
    p.add_argument("--input_path", default="data/historical/BTCUSDT_1m_200000.csv")
    p.add_argument("--output_dir", default="processed/")
    p.add_argument("--scaler_dir", default="scalers/")
    p.add_argument("--chunk_size", type=int, default=50_000)
    p.add_argument("--n_features", type=int, default=15)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--missing_threshold", type=float, default=0.05)
    p.add_argument("--exp_lookback", type=int, default=50)
    p.add_argument("--target_move", type=float, default=300.0)
    p.add_argument("--prediction_window", type=int, default=5)
    p.add_argument("--oos_fraction", type=float, default=0.2)
    p.add_argument("--mse_threshold", type=float, default=0.1)
    p.add_argument("--convergence_tol", type=float, default=0.01)
    p.add_argument("--save_scaled", action="store_true")
    p.add_argument("--no-save_scaled", dest="save_scaled", action="store_false")
    p.add_argument("--use_permutation_importance", action="store_true")
    p.add_argument("--no-use_permutation_importance", dest="use_permutation_importance", action="store_false")
    p.add_argument("--volatility_adjustment", action="store_true")
    p.add_argument("--no-volatility_adjustment", dest="volatility_adjustment", action="store_false")
    p.set_defaults(
        save_scaled=True,
        use_permutation_importance=True,
        volatility_adjustment=True
    )
    p.add_argument("--log_path", default="logs/data_engine_trainer.log")
    p.add_argument("--version", default="1.0.0")
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
        target_move=args.target_move,
        prediction_window=args.prediction_window,
        oos_fraction=args.oos_fraction,
        mse_threshold=args.mse_threshold,
        convergence_tol=args.convergence_tol,
        save_scaled=args.save_scaled,
        log_path=args.log_path,
        version=args.version,
        use_permutation_importance=args.use_permutation_importance,
        volatility_adjustment=args.volatility_adjustment,
    )
    trainer.process()