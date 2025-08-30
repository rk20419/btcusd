# live/regime_predictor.py
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import joblib

# sklearn & hmmlearn
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

# logging helpers
from logging.handlers import RotatingFileHandler

# Optional prometheus
try:
    from prometheus_client import Gauge, start_http_server
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False

# Constants & defaults
LIVE_FEATURES_DEFAULT = "processed/live/live_features.csv"
GMM_MODEL_DEFAULT = "models/gmm_model.pkl"
HMM_MODEL_DEFAULT = "models/hmm_model.pkl"
SCALER_REGIME_DEFAULT = "models/scaler_regime.pkl"
TOP_FEATURES_DEFAULT = "processed/top_features.txt"

OUT_CURRENT = "processed/live/current_regime.csv"
OUT_NEXT_PROBS = "processed/live/next_regime_probs.csv"
OUT_WARN = "processed/live/early_warnings.csv"
HEALTH_FILE = "processed/live/regime_predictor_health.json"

REQUIRED_COLS = {"timestamp", "datetime_ist"}
EPS = 1e-10

# Regime mapping for interpretability
# REGIME_MAPPING = {
#     0: "RANGE_LOW",
#     1: "RANGE_HIGH", 
#     2: "TREND_UP",
#     3: "TREND_DOWN",
#     4: "VOLATILE_UP", 
#     5: "VOLATILE_DOWN",
#     6: "STABLE_HIGH",
#     7: "STABLE_LOW"
# }

REGIME_MAPPING = {
    # Original 8 regimes with detailed comments
    0: "RANGE_LOW",          # Low volatility range-bound market (21.97 volatility, 15,984 occurrences)
    1: "RANGE_HIGH",         # High volatility range-bound market (35.40 volatility, 11,453 occurrences) 
    2: "TREND_UP",           # Strong upward trend (44.96 volatility, 42,643 occurrences - most frequent)
    3: "TREND_DOWN",         # Strong downward trend (28.38 volatility, 30,281 occurrences)
    4: "VOLATILE_UP",        # Highly volatile upward moves (177.37 volatility, 1,062 occurrences - extreme volatility)
    5: "VOLATILE_DOWN",      # Highly volatile downward moves (92.27 volatility, 18,497 occurrences)
    6: "STABLE_HIGH",        # Stable high price levels (66.66 volatility, 7,315 occurrences)
    7: "STABLE_LOW",         # Stable low price levels (62.36 volatility, 35,294 occurrences - 2nd most frequent)
    
    # Newly identified patterns with detailed comments  
    8: "EXTENDED_TREND",     # Long-lasting trends (2.73 candles duration - longest, 32.74 volatility, 7,182 occurrences)
    9: "HIGH_VOLATILITY",    # Extreme volatility events (146.86 volatility - highest, 95.98% confidence - highest, 1,201 occurrences)
    10: "CONSOLIDATION",     # Sideways consolidation patterns (31.62 volatility - low, 16,298 occurrences - high frequency)
    11: "TRANSITION"         # Short-term transition phases (1.11 candles duration - shortest, 40.60 volatility, 12,790 occurrences)
}
# ---------------- logger ----------------
def setup_logger(log_path: str = "logs/regime_predictor.log") -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("RegimePredictor")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger

# ---------------- predictor ----------------
class RegimePredictor:
    def __init__(
        self,
        live_features_path: str = LIVE_FEATURES_DEFAULT,
        gmm_path: str = GMM_MODEL_DEFAULT,
        hmm_path: str = HMM_MODEL_DEFAULT,
        scaler_path: str = SCALER_REGIME_DEFAULT,
        top_features_path: str = TOP_FEATURES_DEFAULT,
        out_current: str = OUT_CURRENT,
        out_next_probs: str = OUT_NEXT_PROBS,
        out_warn: str = OUT_WARN,
        log_path: str = "logs/regime_predictor.log",
        monitor_interval: int = 10,
        prometheus_port: Optional[int] = None,
        base_threshold: float = 0.7,
        atr_threshold: float = 1.5,
        asia_boost: float = 0.0,
        london_boost: float = 0.0,
        ny_boost: float = 0.15,
        min_warning_confidence: float = 0.6,
    ):
        self.live_features_path = Path(live_features_path)
        self.gmm_path = Path(gmm_path)
        self.hmm_path = Path(hmm_path)
        self.scaler_path = Path(scaler_path)
        self.top_features_path = Path(top_features_path)

        self.out_current = Path(out_current)
        self.out_next_probs = Path(out_next_probs)
        self.out_warn = Path(out_warn)

        self.monitor_interval = max(1, int(monitor_interval))
        self.base_threshold = float(base_threshold)
        self.atr_threshold = float(atr_threshold)
        self.session_boosts = {"asia": asia_boost, "london": london_boost, "ny": ny_boost}
        self.min_warning_confidence = min_warning_confidence

        # ensure output dirs
        for p in (self.out_current.parent, self.out_next_probs.parent, self.out_warn.parent):
            p.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(log_path)

        # Prometheus metrics
        if PROM_AVAILABLE and prometheus_port:
            try:
                start_http_server(prometheus_port)
                self.logger.info(f"Prometheus metrics exported on port {prometheus_port}")
            except Exception as e:
                self.logger.warning(f"Prometheus start failed: {e}")
        if PROM_AVAILABLE:
            self.g_latency = Gauge("regime_predictor_latency_seconds", "Regime predictor latency seconds")
            self.g_confidence = Gauge("regime_predictor_confidence", "Current regime confidence")
            self.g_warn = Gauge("regime_predictor_warning_flag", "1 if warning emitted")
            self.g_regime = Gauge("regime_predictor_current_regime", "Current regime ID")
        else:
            self.g_latency = self.g_confidence = self.g_warn = self.g_regime = None

        # load artifacts
        self.gmm: Optional[GaussianMixture] = None
        self.hmm: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.top_features: List[str] = []

        self._load_models()
        self._load_top_features()

        # watch file
        self.last_mtime = self.live_features_path.stat().st_mtime if self.live_features_path.exists() else 0
        self.last_good_current: Optional[pd.DataFrame] = None
        self.last_regime: Optional[int] = None
        self.last_confidence: Optional[float] = None

    def _load_models(self):
        # load GMM - joblib se load karo
        try:
            # Pehle joblib se try karo
            try:
                self.gmm = joblib.load(self.gmm_path)
                self.logger.info(f"Loaded GMM from {self.gmm_path} with {self.gmm.n_components} components using joblib")
            except:
                # Agar joblib fail hota hai to pickle try karo
                with open(self.gmm_path, "rb") as f:
                    self.gmm = pickle.load(f)
                self.logger.info(f"Loaded GMM from {self.gmm_path} with {self.gmm.n_components} components using pickle")
        except Exception as e:
            self.logger.error(f"Failed to load GMM ({self.gmm_path}): {e}")
            raise

        # load HMM (optional) - joblib se load karo
        try:
            if self.hmm_path.exists():
                try:
                    self.hmm = joblib.load(self.hmm_path)
                    self.logger.info(f"Loaded HMM from {self.hmm_path} using joblib")
                except:
                    with open(self.hmm_path, "rb") as f:
                        self.hmm = pickle.load(f)
                    self.logger.info(f"Loaded HMM from {self.hmm_path} using pickle")
            else:
                self.logger.warning("HMM model not found; transitions will be estimated from data")
                self.hmm = None
        except Exception as e:
            self.logger.warning(f"Failed to load HMM ({self.hmm_path}): {e} - HMM transitions will be unavailable")
            self.hmm = None

        # load scaler (optional) - joblib se load karo
        try:
            if self.scaler_path.exists():
                try:
                    self.scaler = joblib.load(self.scaler_path)
                    self.logger.info(f"Loaded scaler from {self.scaler_path} using joblib")
                except:
                    with open(self.scaler_path, "rb") as f:
                        self.scaler = pickle.load(f)
                    self.logger.info(f"Loaded scaler from {self.scaler_path} using pickle")
            else:
                self.logger.warning("Regime scaler not found; will attempt to use raw features")
                self.scaler = None
        except Exception as e:
            self.logger.warning(f"Failed to load scaler ({self.scaler_path}): {e} - will attempt to use raw features")
            self.scaler = None

    def _load_top_features(self):
        """Load top features from file to ensure consistency with training"""
        try:
            if self.top_features_path.exists():
                with open(self.top_features_path, "r") as f:
                    self.top_features = [line.strip() for line in f if line.strip()]
                self.logger.info(f"Loaded {len(self.top_features)} top features from {self.top_features_path}")
            else:
                self.logger.warning(f"Top features file not found: {self.top_features_path}")
                # Fallback: use GMM dimension to determine number of features
                if self.gmm:
                    self.top_features = [f"feature_{i}" for i in range(self.gmm.means_.shape[1])]
        except Exception as e:
            self.logger.warning(f"Failed to load top features: {e}")
            if self.gmm:
                self.top_features = [f"feature_{i}" for i in range(self.gmm.means_.shape[1])]

    # --------------- utilities ---------------
    def _session_from_hour(self, hour_ist: int) -> str:
        if 0 <= hour_ist < 8:
            return "asia"
        if 8 <= hour_ist < 16:
            return "london"
        return "ny"

    def _compute_dynamic_threshold(self, session: str, atr_val: Optional[float]) -> float:
        thr = self.base_threshold + self.session_boosts.get(session, 0.0)
        if atr_val is not None and atr_val > self.atr_threshold:
            thr += 0.1
        return min(0.99, thr)

    def _handle_nan_values(self, X: np.ndarray) -> np.ndarray:
        """Handle NaN values in the feature matrix using median imputation"""
        nan_count = np.isnan(X).sum()
        
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values in features, imputing with column medians")
            
            # Calculate column medians, ignoring NaN values
            col_medians = np.nanmedian(X, axis=0)
            
            # Find indices of NaN values
            nan_indices = np.where(np.isnan(X))
            
            # Replace NaN values with column medians
            X[nan_indices] = np.take(col_medians, nan_indices[1])
        
        return X

    # --------------- I/O ---------------
    def load_live_features(self) -> pd.DataFrame:
        if not self.live_features_path.exists():
            raise FileNotFoundError(f"Live features file not found: {self.live_features_path}")
        df = pd.read_csv(self.live_features_path)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Required columns missing in live features: {missing}")
        return df

    # --------------- core logic ---------------
    def _prepare_obs(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare observations for prediction, prioritizing scaled features,
        then raw features with scaler, then raw features without scaler
        """
        # First try: use scaled features if available
        scaled_cols = [f"{feat}_scaled" for feat in self.top_features if f"{feat}_scaled" in df.columns]
        if scaled_cols and len(scaled_cols) >= self.gmm.means_.shape[1]:
            X = df[scaled_cols].astype(float).values
            self.logger.info(f"Using {len(scaled_cols)} scaled features for prediction")
            X = self._handle_nan_values(X)  # Handle NaN values
            return X[:, :self.gmm.means_.shape[1]]  # Ensure correct dimensions
        
        # Second try: use raw features with scaler
        raw_cols = [feat for feat in self.top_features if feat in df.columns]
        if raw_cols and self.scaler and len(raw_cols) >= self.gmm.means_.shape[1]:
            try:
                X_raw = df[raw_cols].astype(float).values
                X_raw = self._handle_nan_values(X_raw)  # Handle NaN values before scaling
                X = self.scaler.transform(X_raw)
                self.logger.info(f"Using {len(raw_cols)} raw features with scaler for prediction")
                return X[:, :self.gmm.means_.shape[1]]  # Ensure correct dimensions
            except Exception as e:
                self.logger.warning(f"Scaler transform failed: {e}")
        
        # Third try: use whatever numeric features we have
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("timestamp", "target_buy", "target_sell")]
        
        if len(numeric_cols) >= self.gmm.means_.shape[1]:
            X = df[numeric_cols].astype(float).values
            X = self._handle_nan_values(X)  # Handle NaN values
            self.logger.info(f"Using {len(numeric_cols)} numeric features for prediction")
            return X[:, :self.gmm.means_.shape[1]]  # Ensure correct dimensions
        
        # Final fallback: pad with zeros if needed
        X = np.zeros((len(df), self.gmm.means_.shape[1]))
        self.logger.warning(f"Could not find enough features, using zero-padded array")
        return X

    def _estimate_transmat_from_sequence(self, probs: np.ndarray) -> np.ndarray:
        # probs: (n_obs, n_states)
        labels = probs.argmax(axis=1)
        n_states = self.gmm.n_components
        counts = np.zeros((n_states, n_states), dtype=float)
        for a, b in zip(labels[:-1], labels[1:]):
            counts[a, b] += 1.0
        counts += 1e-6  # Laplace smoothing
        trans = counts / counts.sum(axis=1, keepdims=True)
        return trans

    def run_once(self) -> bool:
        t0 = time.time()
        try:
            df = self.load_live_features()
            if len(df) == 0:
                self.logger.warning("Empty live features DataFrame")
                return False

            # Prepare observations for prediction
            X = self._prepare_obs(df)

            # compute GMM posterior on all rows
            gmm_probs = self.gmm.predict_proba(X)  # (n_obs, n_states)
            gmm_labels = gmm_probs.argmax(axis=1)
            gmm_conf = gmm_probs.max(axis=1)

            # write current regimes
            curr_df = pd.DataFrame({
                "timestamp": df["timestamp"].values,
                "datetime_ist": df["datetime_ist"].values,
                "regime": gmm_labels,
                "regime_confidence": gmm_conf,
                "regime_name": [REGIME_MAPPING.get(label, f"UNKNOWN_{label}") for label in gmm_labels]
            })
            curr_df.to_csv(self.out_current, index=False)
            self.last_good_current = curr_df.copy()
            self.logger.info(f"Wrote current regimes to {self.out_current} ({len(curr_df)} rows)")

            # derive transition matrix
            if self.hmm is not None and hasattr(self.hmm, "transmat_"):
                trans = np.asarray(self.hmm.transmat_)
                self.logger.info("Using HMM transition matrix")
            else:
                n_recent = min(500, len(gmm_probs))
                trans = self._estimate_transmat_from_sequence(gmm_probs[-n_recent:])
                self.logger.info("Using estimated transition matrix from recent data")

            # next probs per-row
            next_probs = gmm_probs @ trans  # (n_obs, n_states)

            # save next probs in long format
            rows = []
            n_states = trans.shape[0]
            for i in range(len(df)):
                ts = int(df["timestamp"].iat[i])
                dt = df["datetime_ist"].iat[i]
                for to_s in range(n_states):
                    rows.append({
                        "timestamp": ts,
                        "datetime_ist": dt,
                        "from_regime": int(gmm_labels[i]),
                        "from_regime_name": REGIME_MAPPING.get(int(gmm_labels[i]), f"UNKNOWN_{int(gmm_labels[i])}"),
                        "to_regime": int(to_s),
                        "to_regime_name": REGIME_MAPPING.get(to_s, f"UNKNOWN_{to_s}"),
                        "prob": float(next_probs[i, to_s])
                    })
            next_df = pd.DataFrame(rows)
            next_df.to_csv(self.out_next_probs, index=False)
            self.logger.info(f"Wrote next-regime probabilities to {self.out_next_probs}")

            # early warnings: evaluate last row
            last_idx = len(df) - 1
            last_hour = pd.to_datetime(df["datetime_ist"].iat[last_idx]).hour if "datetime_ist" in df.columns else None
            session = self._session_from_hour(last_hour) if last_hour is not None else "asia"
            atr_val = float(df["atr_14"].iat[last_idx]) if "atr_14" in df.columns and not pd.isna(df["atr_14"].iat[last_idx]) else None
            dyn_thr = self._compute_dynamic_threshold(session, atr_val)

            warnings = []
            last_next = next_probs[last_idx]
            current_confidence = gmm_conf[last_idx]
            
            # Only generate warnings if confidence is above minimum threshold
            if current_confidence >= self.min_warning_confidence:
                for to_s, prob in enumerate(last_next):
                    if prob >= dyn_thr:
                        warnings.append({
                            "timestamp": int(df["timestamp"].iat[last_idx]),
                            "datetime_ist": df["datetime_ist"].iat[last_idx],
                            "warning_type": "high_next_regime_prob",
                            "score": float(prob),
                            "current_regime": int(gmm_labels[last_idx]),
                            "current_regime_name": REGIME_MAPPING.get(int(gmm_labels[last_idx]), f"UNKNOWN_{int(gmm_labels[last_idx])}"),
                            "predicted_regime": int(to_s),
                            "predicted_regime_name": REGIME_MAPPING.get(to_s, f"UNKNOWN_{to_s}"),
                            "confidence": float(current_confidence),
                            "dynamic_threshold": float(dyn_thr),
                            "session": session,
                            "atr_14": float(atr_val) if atr_val is not None else None
                        })
            
            warn_df = pd.DataFrame(warnings) if warnings else pd.DataFrame(columns=[
                "timestamp", "datetime_ist", "warning_type", "score", "current_regime", 
                "current_regime_name", "predicted_regime", "predicted_regime_name",
                "confidence", "dynamic_threshold", "session", "atr_14"
            ])
            warn_df.to_csv(self.out_warn, index=False)
            self.logger.info(f"Wrote {len(warn_df)} early warnings to {self.out_warn}")

            # Update last values for health monitoring
            self.last_regime = int(gmm_labels[last_idx])
            self.last_confidence = float(current_confidence)

            # prometheus metrics
            latency = time.time() - t0
            if PROM_AVAILABLE:
                try:
                    self.g_latency.set(latency)
                    self.g_confidence.set(float(current_confidence))
                    self.g_warn.set(1.0 if not warn_df.empty else 0.0)
                    self.g_regime.set(float(self.last_regime))
                except Exception as e:
                    self.logger.warning(f"Prometheus metric update failed: {e}")

            # health
            self._write_health(
                success=True, 
                latency=latency, 
                rows=len(df), 
                last_timestamp=int(df["timestamp"].iat[last_idx]),
                current_regime=self.last_regime,
                current_confidence=self.last_confidence,
                warnings_count=len(warn_df)
            )
            
            self.logger.info(
                f"Regime prediction completed (latency {latency:.3f}s). "
                f"Current: {REGIME_MAPPING.get(self.last_regime, 'UNKNOWN')} "
                f"(conf: {self.last_confidence:.3f}), "
                f"Session: {session}, DynThr: {dyn_thr:.3f}, "
                f"Warnings: {len(warn_df)}"
            )
            return True
            
        except Exception as exc:
            self.logger.error(f"Regime predict failed: {exc}\n{traceback.format_exc()}")
            if self.last_good_current is not None:
                try:
                    self.last_good_current.to_csv(self.out_current, index=False)
                    self.logger.warning("Wrote last-good current_regime as fallback")
                except Exception as e2:
                    self.logger.error(f"Fallback write failed: {e2}")
            self._write_health(success=False, message=str(exc))
            return False

    def _write_health(
        self, 
        success: bool, 
        latency: Optional[float] = None, 
        rows: Optional[int] = None, 
        last_timestamp: Optional[int] = None,
        current_regime: Optional[int] = None,
        current_confidence: Optional[float] = None,
        warnings_count: Optional[int] = None,
        message: Optional[str] = None
    ):
        try:
            data = {
                "success": bool(success),
                "timestamp_utc": int(time.time()),
                "latency_s": float(latency) if latency is not None else None,
                "rows": int(rows) if rows is not None else None,
                "last_timestamp": int(last_timestamp) if last_timestamp else None,
                "current_regime": int(current_regime) if current_regime is not None else None,
                "current_regime_name": REGIME_MAPPING.get(current_regime, f"UNKNOWN_{current_regime}") if current_regime is not None else None,
                "current_confidence": float(current_confidence) if current_confidence is not None else None,
                "warnings_count": int(warnings_count) if warnings_count is not None else None,
                "message": str(message)[:200] if message else None
            }
            Path(HEALTH_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(HEALTH_FILE, "w") as fh:
                json.dump(data, fh, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to write health file: {e}")

    # ------------- monitor loop -------------
    def run(self, once: bool = False):
        if once:
            self.logger.info("Running single prediction (once mode)")
            self.run_once()
            return
            
        self.logger.info("Starting regime_predictor monitor loop")
        try:
            while True:
                try:
                    if not self.live_features_path.exists():
                        self.logger.warning(f"No live features file at {self.live_features_path}; sleeping {self.monitor_interval}s")
                        time.sleep(self.monitor_interval)
                        continue
                        
                    mtime = self.live_features_path.stat().st_mtime
                    if mtime > self.last_mtime:
                        self.last_mtime = mtime
                        self.logger.info("Change detected in live features; running prediction")
                        self.run_once()
                    else:
                        self.logger.debug("No change detected in live features")
                        
                except Exception as e:
                    self.logger.error(f"Monitor loop error: {e}\n{traceback.format_exc()}")
                    
                time.sleep(self.monitor_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal run error: {e}\n{traceback.format_exc()}")
            raise

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser("Module 2 — Regime Predictor (live)")
    p.add_argument("--live_features_path", default=LIVE_FEATURES_DEFAULT)
    p.add_argument("--gmm_path", default=GMM_MODEL_DEFAULT)
    p.add_argument("--hmm_path", default=HMM_MODEL_DEFAULT)
    p.add_argument("--scaler_path", default=SCALER_REGIME_DEFAULT)
    p.add_argument("--top_features_path", default=TOP_FEATURES_DEFAULT)
    p.add_argument("--out_current", default=OUT_CURRENT)
    p.add_argument("--out_next_probs", default=OUT_NEXT_PROBS)
    p.add_argument("--out_warn", default=OUT_WARN)
    p.add_argument("--log_path", default="logs/regime_predictor.log")
    p.add_argument("--monitor_interval", type=int, default=10)
    p.add_argument("--prometheus_port", type=int, default=None)
    p.add_argument("--base_threshold", type=float, default=0.7)
    p.add_argument("--atr_threshold", type=float, default=1.5)
    p.add_argument("--asia_boost", type=float, default=0.0, help="Threshold boost for Asia session")
    p.add_argument("--london_boost", type=float, default=0.0, help="Threshold boost for London session")
    p.add_argument("--ny_boost", type=float, default=0.15, help="Threshold boost for NY session")
    p.add_argument("--min_warning_confidence", type=float, default=0.6, help="Minimum confidence to generate warnings")
    p.add_argument("--once", action="store_true", help="Process a single update and exit")
    return p.parse_args()

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    args = parse_args()
    predictor = RegimePredictor(
        live_features_path=args.live_features_path,
        gmm_path=args.gmm_path,
        hmm_path=args.hmm_path,
        scaler_path=args.scaler_path,
        top_features_path=args.top_features_path,
        out_current=args.out_current,
        out_next_probs=args.out_next_probs,
        out_warn=args.out_warn,
        log_path=args.log_path,
        monitor_interval=args.monitor_interval,
        prometheus_port=args.prometheus_port,
        base_threshold=args.base_threshold,
        atr_threshold=args.atr_threshold,
        asia_boost=args.asia_boost,
        london_boost=args.london_boost,
        ny_boost=args.ny_boost,
        min_warning_confidence=args.min_warning_confidence,
    )
    predictor.run(once=args.once)