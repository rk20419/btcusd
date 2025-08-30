from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import traceback
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# sklearn
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# hmmlearn
from hmmlearn.hmm import GaussianHMM

# logging setup
from logging.handlers import RotatingFileHandler

pd.options.mode.chained_assignment = None

# Regime type mapping for interpretability
REGIME_MAPPING = {
    0: "RANGE_LOW",
    1: "RANGE_HIGH", 
    2: "TREND_UP",
    3: "TREND_DOWN",
    4: "VOLATILE_UP", 
    5: "VOLATILE_DOWN",
    6: "STABLE_HIGH",
    7: "STABLE_LOW"
}

def setup_logger(log_path: str = "logs/regime_trainer.log") -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("RegimeTrainer")
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


class RegimeTrainer:
    def __init__(
        self,
        processed_glob: str = "processed/chunk_*.csv",
        single_processed: Optional[str] = None,
        top_features_path: str = "processed/top_features.txt",
        model_dir: str = "models/",
        n_components_search: Tuple[int, int] = (4, 12),
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        hmm_max_iter: int = 200,
        hmm_tol: float = 1e-3,
        n_splits: int = 5,
        random_state: int = 42,
        log_path: str = "logs/regime_trainer.log",
        manifest_name: str = "models/manifest_regime.json",
        min_regime_duration: int = 5,  # Minimum candles for a regime to be valid
        target_regimes: int = 8,  # Target number of regimes to identify
    ):
        self.processed_glob = processed_glob
        self.single_processed = single_processed
        self.top_features_path = top_features_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.nmin, self.nmax = n_components_search
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.hmm_max_iter = hmm_max_iter
        self.hmm_tol = hmm_tol
        self.n_splits = n_splits
        self.random_state = random_state
        self.manifest_name = manifest_name
        self.min_regime_duration = min_regime_duration
        self.target_regimes = target_regimes

        self.logger = setup_logger(log_path)

        self.top_features: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.gmm: Optional[GaussianMixture] = None
        self.hmm: Optional[GaussianHMM] = None
        self.regime_stats: Dict = {}

    # ---------------- I/O ----------------
    def _collect_processed_files(self) -> List[str]:
        if self.single_processed:
            if Path(self.single_processed).exists():
                return [self.single_processed]
            else:
                raise FileNotFoundError(f"Provided processed file not found: {self.single_processed}")
        files = sorted(glob(self.processed_glob))
        if not files:
            raise FileNotFoundError(f"No processed chunk files found with pattern: {self.processed_glob}")
        return files

    def _load_top_features(self) -> List[str]:
        # prefer top_features.txt; else manifest.json; else first file columns
        if self.top_features_path and Path(self.top_features_path).exists():
            with open(self.top_features_path, "r") as f:
                feats = [l.strip() for l in f if l.strip()]
                self.logger.info(f"Loaded top_features from {self.top_features_path}: {feats[:20]}")
                return feats
        # try manifest
        manifest_candidates = list(Path("processed").glob("manifest*.json")) + list(Path("models").glob("manifest*.json"))
        for m in manifest_candidates:
            try:
                j = json.load(open(m, "r"))
                if "top_features" in j:
                    self.logger.info(f"Loaded top_features from manifest {m}")
                    return j["top_features"]
            except Exception:
                continue
        # fallback: inspect a processed file
        files = self._collect_processed_files()
        df = pd.read_csv(files[0], nrows=10)
        # deduce top features as numeric non-target columns
        cols = [c for c in df.columns if c not in ("timestamp", "datetime_ist", "target_buy", "target_sell") and not c.endswith("_scaled")]
        self.logger.warning(f"Falling back to inferred top_features: {cols[:20]}")
        return cols

    # ---------------- Load & merge ----------------
    def load_all_processed(self) -> pd.DataFrame:
        files = self._collect_processed_files()
        parts = []
        for fp in tqdm(files, desc="Loading processed chunks"):
            self.logger.info(f"Loading processed chunk: {fp}")
            df = pd.read_csv(fp)
            parts.append(df)
        df_all = pd.concat(parts, ignore_index=True)
        self.logger.info(f"Total processed rows loaded: {len(df_all)}")
        return df_all

    # ---------------- Preprocess ----------------
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        # Ensure top_features available
        if not self.top_features:
            self.top_features = self._load_top_features()
        # Filter only top features that exist
        features = [f for f in self.top_features if f in df.columns]
        if not features:
            raise ValueError("No overlapping top_features found in processed data.")
        X = df[features].astype(float).values
        # Impute simple nan strategy (median) before scaling
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = np.take(col_medians, inds[1])
            self.logger.info(f"Imputed {len(inds[0])} NaNs with column medians before scaling")
        # Standardize for clustering & HMM training
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        # persist scaler
        joblib.dump(self.scaler, self.model_dir / "scaler_regime.pkl")
        self.logger.info("Saved scaler_regime.pkl")
        # return scaled + DataFrame with timeline
        return Xs, df.reset_index(drop=True)

    # ---------------- GMM model selection & fit ----------------
    def select_and_fit_gmm(self, X: np.ndarray, n_min: int = None, n_max: int = None) -> GaussianMixture:
        n_min = n_min or self.nmin
        n_max = n_max or self.nmax
        best = None
        best_bic = np.inf
        best_aic = np.inf
        results = []
        
        self.logger.info(f"Searching GMM components in [{n_min}, {n_max}] using BIC/AIC")
        for k in tqdm(range(n_min, n_max + 1), desc="GMM Component Search"):
            try:
                g = GaussianMixture(
                    n_components=k,
                    covariance_type=self.covariance_type,
                    reg_covar=self.reg_covar,
                    random_state=self.random_state,
                    max_iter=500,
                    verbose=0,
                )
                g.fit(X)
                bic = g.bic(X)
                aic = g.aic(X)
                results.append((k, bic, aic))
                self.logger.info(f"GMM k={k}: BIC={bic:.1f}, AIC={aic:.1f}")
                
                # Prefer models closer to target regime count
                regime_distance = abs(k - self.target_regimes)
                score = bic + (regime_distance * 100)  # Penalize being far from target
                
                if score < best_bic:
                    best_bic = score
                    best_aic = aic
                    best = g
            except Exception as e:
                self.logger.warning(f"GMM k={k} failed: {e}")
                
        if best is None:
            raise RuntimeError("All GMM fits failed")
            
        # Save component selection results
        results_df = pd.DataFrame(results, columns=["n_components", "bic", "aic"])
        results_df.to_csv(self.model_dir / "gmm_component_selection.csv", index=False)
        
        self.logger.info(f"Selected GMM with {best.n_components} components (BIC {best_bic:.1f}, AIC {best_aic:.1f})")
        self.gmm = best
        joblib.dump(self.gmm, self.model_dir / "gmm_model.pkl")
        self.logger.info("Saved gmm_model.pkl")
        return best

    # ---------------- Build regime assignments ----------------
    def assign_regimes(self, gmm: GaussianMixture, X_scaled: np.ndarray, df_meta: pd.DataFrame) -> pd.DataFrame:
        # responsibilities -> confidence as max probability
        probs = gmm.predict_proba(X_scaled)
        labels = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        
        # Calculate regime durations and filter short regimes
        filtered_labels = self._filter_short_regimes(labels)
        
        df_out = pd.DataFrame({
            "timestamp": df_meta["timestamp"].values,
            "datetime_ist": df_meta["datetime_ist"].values if "datetime_ist" in df_meta.columns else None,
            "regime": filtered_labels,
            "regime_confidence": confidences,
            "regime_name": [REGIME_MAPPING.get(l, f"UNKNOWN_{l}") for l in filtered_labels]
        })
        
        # Calculate regime statistics
        self._calculate_regime_statistics(df_out, df_meta)
        
        # Save assignments
        out_fp = self.model_dir / "regime_assignments.csv"
        df_out.to_csv(out_fp, index=False)
        self.logger.info(f"Saved regime assignments -> {out_fp} ({df_out.shape[0]} rows)")
        return df_out

    def _filter_short_regimes(self, labels: np.ndarray) -> np.ndarray:
        """Filter out regimes that are too short to be meaningful"""
        filtered = labels.copy()
        current_regime = labels[0]
        start_idx = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_regime:
                duration = i - start_idx
                if duration < self.min_regime_duration:
                    # Mark short regime for filtering (assign previous regime)
                    filtered[start_idx:i] = labels[start_idx-1] if start_idx > 0 else labels[i]
                
                current_regime = labels[i]
                start_idx = i
        
        # Check last regime
        duration = len(labels) - start_idx
        if duration < self.min_regime_duration and start_idx > 0:
            filtered[start_idx:] = labels[start_idx-1]
            
        return filtered

    def _calculate_regime_statistics(self, df_assignments: pd.DataFrame, df_meta: pd.DataFrame):
        """Calculate statistics for each regime"""
        stats = {}
        for regime in df_assignments["regime"].unique():
            regime_mask = df_assignments["regime"] == regime
            stats[regime] = {
                "count": int(regime_mask.sum()),
                "duration_avg": float(regime_mask.groupby((regime_mask != regime_mask.shift()).cumsum()).sum().mean()),
                "confidence_avg": float(df_assignments.loc[regime_mask, "regime_confidence"].mean()),
                "volatility_avg": float(df_meta.loc[regime_mask, "volatility"].mean()) if "volatility" in df_meta.columns else None,
                "returns_avg": float(df_meta.loc[regime_mask, "returns"].mean()) if "returns" in df_meta.columns else None,
            }
        
        self.regime_stats = stats
        stats_df = pd.DataFrame.from_dict(stats, orient="index")
        stats_df.to_csv(self.model_dir / "regime_statistics.csv", index_label="regime")
        self.logger.info("Saved regime statistics")

    # ---------------- HMM training ----------------
    def fit_hmm(self, X_scaled: np.ndarray, n_components: Optional[int] = None) -> GaussianHMM:
        if n_components is None:
            if self.gmm is None:
                raise RuntimeError("GMM must be trained before HMM initialization to choose n_components")
            n_components = self.gmm.n_components
            
        self.logger.info(f"Initializing HMM with {n_components} states (Baum-Welch)")
        
        # Initialize HMM parameters using GMM if available
        try:
            # Create HMM and set means/covars from GMM for fast convergence
            hmm = GaussianHMM(
                n_components=n_components,
                covariance_type=self.covariance_type,
                n_iter=self.hmm_max_iter,
                tol=self.hmm_tol,
                verbose=True,  # Enable verbose output for monitoring
                random_state=self.random_state,
            )
            
            # Set startprob uniformly, transmat random (will be re-estimated)
            startprob = np.full(n_components, 1.0 / n_components)
            transmat = np.full((n_components, n_components), 1.0 / n_components)
            hmm.startprob_ = startprob
            hmm.transmat_ = transmat
            
            # Initialize means/covars by splitting GMM means to HMM (if available)
            if self.gmm is not None:
                try:
                    hmm.means_ = self.gmm.means_.copy()
                    # covariance shape handling
                    if self.covariance_type == "full":
                        covs = []
                        # hmmlearn expects covars_ shape (n_components, n_features, n_features)
                        # GMM has covariances_ similarly
                        for c in range(self.gmm.n_components):
                            cov = self.gmm.covariances_[c]
                            # ensure positive definite by adding reg_covar
                            covs.append(cov + self.reg_covar * np.eye(cov.shape[0]))
                        hmm.covars_ = np.array(covs)
                    else:
                        # diag / tied etc. let HMM initialize
                        pass
                except Exception as e:
                    self.logger.warning(f"Failed to initialize HMM means/covars from GMM: {e}")
            
            # Fit HMM on full sequence (Baum-Welch)
            self.logger.info("Fitting HMM (this may take a while)...")
            hmm.fit(X_scaled)
            self.hmm = hmm
            joblib.dump(self.hmm, self.model_dir / "hmm_model.pkl")
            self.logger.info("Saved hmm_model.pkl")
            return hmm
            
        except Exception as e:
            self.logger.error(f"HMM training failed: {e}\n{traceback.format_exc()}")
            raise

    # ---------------- Transition probabilities ----------------
    def compute_transition_matrix(self, assignments: pd.DataFrame, n_states: int) -> pd.DataFrame:
        seq = assignments["regime"].values.astype(int)
        counts = np.zeros((n_states, n_states), dtype=float)
        
        for (a, b) in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1
        
        # Laplace smoothing to avoid zeros
        counts += 1e-6
        probs = counts / counts.sum(axis=1, keepdims=True)
        
        rows = []
        for i in range(n_states):
            for j in range(n_states):
                rows.append({
                    "from_regime": int(i),
                    "from_regime_name": REGIME_MAPPING.get(i, f"UNKNOWN_{i}"),
                    "to_regime": int(j),
                    "to_regime_name": REGIME_MAPPING.get(j, f"UNKNOWN_{j}"),
                    "prob": float(probs[i, j])
                })
                
        dfp = pd.DataFrame(rows)
        dfp.to_csv(self.model_dir / "transition_probs.csv", index=False)
        self.logger.info(f"Saved transition probabilities -> {self.model_dir / 'transition_probs.csv'}")
        return dfp

    # ---------------- Drift / model checks ----------------
    def monitor_feature_drift(self, X_scaled: np.ndarray) -> float:
        # measure MSE from zero mean (after scaler standardization mean=0)
        mse = float(np.mean((X_scaled - 0.0) ** 2))
        self.logger.info(f"Feature drift proxy (post-StdScaler MSE): {mse:.6f}")
        return mse

    # ---------------- Cross-validation checks ----------------
    def time_series_cv_gmm_stability(self, X: np.ndarray):
        # run TimeSeriesSplit and check GMM label stability across folds
        self.logger.info("Running TimeSeriesSplit stability checks (GMM)")
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        label_changes = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            try:
                g = GaussianMixture(
                    n_components=self.gmm.n_components, 
                    covariance_type=self.covariance_type, 
                    reg_covar=self.reg_covar, 
                    random_state=self.random_state
                )
                g.fit(X[train_idx])
                labels_train = g.predict(X[train_idx])
                labels_test = g.predict(X[test_idx])
                
                # Calculate stability metrics
                train_stability = self._calculate_regime_stability(labels_train)
                test_stability = self._calculate_regime_stability(labels_test)
                
                label_changes.append({
                    "fold": fold,
                    "train_stability": train_stability,
                    "test_stability": test_stability
                })
                
                self.logger.info(f"Fold {fold}: Train stability={train_stability:.3f}, Test stability={test_stability:.3f}")
                
            except Exception as e:
                self.logger.warning(f"CV fold GMM failed: {e}")
                
        # Save CV results
        cv_df = pd.DataFrame(label_changes)
        cv_df.to_csv(self.model_dir / "cv_stability_results.csv", index=False)
        
        self.logger.info(f"GMM CV completed ({len(label_changes)} folds)")
        return label_changes

    def _calculate_regime_stability(self, labels_series):
        """Calculate regime persistence metrics"""
        if len(labels_series) < 2:
            return 1.0
            
        changes = np.diff(labels_series)
        stability = 1.0 - (np.sum(changes != 0) / len(changes))
        return stability

    # ---------------- Main runner ----------------
    def run(self):
        start_all = time.time()
        try:
            # Load and prepare data
            df_all = self.load_all_processed()
            X_scaled, df_meta = self.prepare_features(df_all)
            self.monitor_feature_drift(X_scaled)

            # GMM search & fit
            gmm = self.select_and_fit_gmm(X_scaled)
            
            # Cross-validation stability check
            self.time_series_cv_gmm_stability(X_scaled)
            
            # assignments and save
            assignments = self.assign_regimes(gmm, X_scaled, df_meta)

            # HMM train
            hmm = self.fit_hmm(X_scaled, n_components=gmm.n_components)

            # compute transition probabilities (from GMM labels as baseline)
            trans_df = self.compute_transition_matrix(assignments, n_states=gmm.n_components)

            # Save manifest with enhanced information
            manifest = {
                "n_rows": int(df_all.shape[0]),
                "n_features": int(X_scaled.shape[1]),
                "gmm_n_components": int(gmm.n_components),
                "gmm_covariance_type": str(self.covariance_type),
                "hmm_n_components": int(hmm.n_components) if hmm else None,
                "hmm_tol": float(self.hmm_tol),
                "hmm_max_iter": int(self.hmm_max_iter),
                "regime_stats": {str(k): v for k, v in self.regime_stats.items()},
                "regime_mapping": REGIME_MAPPING,
                "timestamp_utc": int(time.time()),
                "training_duration_seconds": float(time.time() - start_all),
            }
            
            with open(self.manifest_name, "w") as mf:
                json.dump(manifest, mf, indent=2)
            self.logger.info(f"Saved manifest -> {self.manifest_name}")

            total_time = time.time() - start_all
            self.logger.info(f"RegimeTrainer completed successfully in {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"RegimeTrainer failed: {e}\n{traceback.format_exc()}")
            raise


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser("Module 2 — Regime Trainer (GMM + HMM)")
    p.add_argument("--processed_glob", default="processed/chunk_*.csv", help="Glob for processed chunk CSVs")
    p.add_argument("--single_processed", default=None, help="Path to single processed CSV (optional)")
    p.add_argument("--top_features_path", default="processed/top_features.txt", help="Top features list")
    p.add_argument("--model_dir", default="models/", help="Directory to save models and outputs")
    p.add_argument("--nmin", type=int, default=4, help="Minimum GMM components to search")
    p.add_argument("--nmax", type=int, default=12, help="Maximum GMM components to search")
    p.add_argument("--covariance_type", default="full", choices=["full", "diag", "tied", "spherical"])
    p.add_argument("--reg_covar", type=float, default=1e-6)
    p.add_argument("--hmm_max_iter", type=int, default=200)
    p.add_argument("--hmm_tol", type=float, default=1e-3)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--log_path", default="logs/regime_trainer.log")
    p.add_argument("--manifest_name", default="models/manifest_regime.json")
    p.add_argument("--min_regime_duration", type=int, default=5, help="Minimum candles for a regime to be valid")
    p.add_argument("--target_regimes", type=int, default=8, help="Target number of regimes to identify")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = RegimeTrainer(
        processed_glob=args.processed_glob,
        single_processed=args.single_processed,
        top_features_path=args.top_features_path,
        model_dir=args.model_dir,
        n_components_search=(args.nmin, args.nmax),
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        hmm_max_iter=args.hmm_max_iter,
        hmm_tol=args.hmm_tol,
        n_splits=args.n_splits,
        random_state=args.random_state,
        log_path=args.log_path,
        manifest_name=args.manifest_name,
        min_regime_duration=args.min_regime_duration,
        target_regimes=args.target_regimes,
    )
    trainer.run()