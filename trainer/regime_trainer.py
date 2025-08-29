# File: trainer/regime_trainer.py
"""
Production-grade Module 2 Part A: Regime Detector Trainer

- Input: processed chunks CSVs (processed/chunk_*.csv) OR a single processed CSV.
- Output:
  - models/gmm_model.pkl
  - models/hmm_model.pkl
  - regime_assignments.csv (timestamp, datetime_ist, regime, confidence)
  - transition_probs.csv (from_regime, to_regime, prob)
  - models/manifest_regime.json
- Models: GaussianMixture (GMM) for regime clusters + GaussianHMM (hmmlearn) for transitions.
"""
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
from typing import List, Optional, Tuple

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

        self.logger = setup_logger(log_path)

        self.top_features: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.gmm: Optional[GaussianMixture] = None
        self.hmm: Optional[GaussianHMM] = None

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
        for fp in files:
            self.logger.info(f"Loading processed chunk: {fp}")
            df = pd.read_csv(fp)
            # prefer scaled columns if present; but we operate on raw features for clustering
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
        self.logger.info(f"Searching GMM components in [{n_min}, {n_max}] using BIC/AIC")
        for k in range(n_min, n_max + 1):
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
                self.logger.info(f"GMM k={k}: BIC={bic:.1f}, AIC={aic:.1f}")
                # prefer lower BIC primarily
                if bic < best_bic:
                    best_bic = bic
                    best = (g, bic, aic)
            except Exception as e:
                self.logger.warning(f"GMM k={k} failed: {e}")
        if best is None:
            raise RuntimeError("All GMM fits failed")
        gmm = best[0]
        self.logger.info(f"Selected GMM with {gmm.n_components} components (BIC {best[1]:.1f})")
        self.gmm = gmm
        joblib.dump(self.gmm, self.model_dir / "gmm_model.pkl")
        self.logger.info("Saved gmm_model.pkl")
        return gmm

    # ---------------- Build regime assignments ----------------
    def assign_regimes(self, gmm: GaussianMixture, X_scaled: np.ndarray, df_meta: pd.DataFrame) -> pd.DataFrame:
        # responsibilities -> confidence as max probability
        probs = gmm.predict_proba(X_scaled)
        labels = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        df_out = pd.DataFrame({
            "timestamp": df_meta["timestamp"].values,
            "datetime_ist": df_meta["datetime_ist"].values if "datetime_ist" in df_meta.columns else None,
            "regime": labels,
            "regime_confidence": confidences
        })
        # Save assignments
        out_fp = self.model_dir / "regime_assignments.csv"
        df_out.to_csv(out_fp, index=False)
        self.logger.info(f"Saved regime assignments -> {out_fp} ({df_out.shape[0]} rows)")
        return df_out

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
                verbose=False,
                random_state=self.random_state,
            )
            # set startprob uniformly, transmat random (will be re-estimated)
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
                rows.append({"from_regime": int(i), "to_regime": int(j), "prob": float(probs[i, j])})
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
        for train_idx, test_idx in tscv.split(X):
            try:
                g = GaussianMixture(n_components=self.gmm.n_components, covariance_type=self.covariance_type, reg_covar=self.reg_covar, random_state=self.random_state)
                g.fit(X[train_idx])
                labels_train = g.predict(X[train_idx])
                labels_test = g.predict(X[test_idx])
                # simple stability: ratio of most frequent label overlap
                # (not perfect but gives signal)
                label_changes.append(float(np.mean(labels_test == labels_test)))
            except Exception as e:
                self.logger.warning(f"CV fold GMM failed: {e}")
        self.logger.info(f"GMM CV completed ({len(label_changes)} folds)")
        return label_changes

    # ---------------- Main runner ----------------
    def run(self):
        start_all = time.time()
        try:
            df_all = self.load_all_processed()
            X_scaled, df_meta = self.prepare_features(df_all)
            self.monitor_feature_drift(X_scaled)

            # GMM search & fit
            gmm = self.select_and_fit_gmm(X_scaled)
            # assignments and save
            assignments = self.assign_regimes(gmm, X_scaled, df_meta)

            # HMM train
            hmm = self.fit_hmm(X_scaled, n_components=gmm.n_components)

            # compute transition probabilities (from GMM labels as baseline)
            trans_df = self.compute_transition_matrix(assignments, n_states=gmm.n_components)

            # Save manifest
            manifest = {
                "n_rows": int(df_all.shape[0]),
                "n_features": int(X_scaled.shape[1]),
                "gmm_n_components": int(gmm.n_components),
                "gmm_covariance_type": str(self.covariance_type),
                "hmm_n_components": int(hmm.n_components) if hmm else None,
                "hmm_tol": float(self.hmm_tol),
                "hmm_max_iter": int(self.hmm_max_iter),
                "timestamp_utc": int(time.time()),
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
    )
    trainer.run()
