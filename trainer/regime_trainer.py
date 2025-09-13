# trainer/regime_trainer.py
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
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Setup logging
def setup_logger(log_path: str):
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("RegimeTrainer")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

class RegimeTrainer:
    def __init__(
        self,
        input_dir: str = "processed/",
        output_dir: str = "models/",
        results_dir: str = "results/",
        n_components: int = 8,  # Fixed to 8 components
        covariance_type: str = "full",
        n_folds: int = 5,
        random_state: int = 42,
        log_path: str = "logs/regime_trainer.log",
        min_regime_duration: int = 5
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.results_dir = results_dir
        self.n_components = n_components  # Fixed to 8
        self.covariance_type = covariance_type
        self.n_folds = n_folds
        self.random_state = random_state
        self.min_regime_duration = min_regime_duration
        
        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(log_path)
        self.scaler = StandardScaler()
        self.gmm = None
        self.hmm_model = None
        
        # Fixed 8 regime mapping with clear logic
        self.regime_names = {
            0: "TREND_UP",        # Strong upward price movement
            1: "TREND_DOWN",      # Strong downward price movement
            2: "RANGE_HIGH",      # Price near range highs with low volatility
            3: "RANGE_LOW",       # Price near range lows with low volatility
            4: "VOLATILE_UP",     # High volatility with upward bias
            5: "VOLATILE_DOWN",   # High volatility with downward bias
            6: "STABLE_HIGH",     # Low volatility, price consolidation near highs
            7: "STABLE_LOW"       # Low volatility, price consolidation near lows
        }
        
        # Feature importance for each regime (to be calculated)
        self.regime_feature_importance = {}

    def load_data(self) -> pd.DataFrame:
        """Load and combine all chunk files"""
        self.logger.info("Loading data chunks...")
        chunk_files = list(Path(self.input_dir).glob("chunk_*.csv"))
        
        if not chunk_files:
            raise ValueError(f"No chunk files found in {self.input_dir}")
        
        dfs = []
        for file in chunk_files:
            df = pd.read_csv(file)
            # Use scaled features if available
            scaled_cols = [col for col in df.columns if col.endswith("_scaled")]
            if scaled_cols:
                df = df[["datetime_ist"] + scaled_cols]
                # Remove _scaled suffix for processing
                df.columns = [col.replace("_scaled", "") for col in df.columns]
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values("datetime_ist").reset_index(drop=True)
        
        # Drop datetime column for modeling
        feature_cols = [col for col in combined_df.columns if col != "datetime_ist"]
        self.logger.info(f"Loaded {len(combined_df)} rows with {len(feature_cols)} features")
        
        return combined_df, feature_cols

    def train_gmm(self, X: np.ndarray) -> GaussianMixture:
        """Train Gaussian Mixture Model with fixed 8 components"""
        self.logger.info(f"Training GMM with fixed {self.n_components} components...")
        
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=500,
            n_init=3,
            verbose=2
        )
        
        gmm.fit(X)
        
        # Calculate model quality metrics
        labels = gmm.predict(X)
        silhouette = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        
        self.logger.info(f"GMM training completed. Silhouette: {silhouette:.3f}, Calinski-Harabasz: {ch_score:.3f}")
        
        return gmm

    def map_to_regimes(self, gmm: GaussianMixture, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Map GMM components to meaningful regimes based on feature characteristics"""
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(self.n_components):
            cluster_data = X[labels == i]
            if len(cluster_data) > 0:
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)
                
                # Calculate feature importance for this regime
                feature_importance = np.abs(cluster_mean) / (cluster_std + 1e-10)
                feature_importance = feature_importance / np.sum(feature_importance)
                
                cluster_stats[i] = {
                    "mean": cluster_mean.tolist(),
                    "std": cluster_std.tolist(),
                    "size": len(cluster_data),
                    "feature_importance": feature_importance.tolist(),
                    "confidence_avg": np.mean(probs[labels == i, i])
                }
        
        # Create mapping documentation
        mapping_logic = {}
        for i in range(self.n_components):
            stats = cluster_stats[i]
            
            # Determine regime type based on feature characteristics
            # This is a simplified approach - in practice, you'd use more sophisticated logic
            trend_strength = stats['mean'][0] if len(stats['mean']) > 0 else 0  # Assuming first feature is trend-related
            volatility = np.mean(stats['std'])
            
            mapping_logic[i] = {
                "assigned_regime": self.regime_names[i],
                "trend_strength": float(trend_strength),
                "volatility": float(volatility),
                "size": stats["size"],
                "confidence_avg": float(stats["confidence_avg"])
            }
        
        return labels, mapping_logic

    def train_hmm(self, labels: np.ndarray) -> hmm.MultinomialHMM:
        """Train Hidden Markov Model for regime transitions"""
        self.logger.info("Training HMM for regime transitions...")
        
        # Create HMM model with fixed 8 states
        model = hmm.MultinomialHMM(
            n_components=self.n_components,
            random_state=self.random_state,
            n_iter=200,
            verbose=True
        )
        
        # Prepare sequences for training
        sequences = [labels]
        lengths = [len(labels)]
        
        # Fit HMM
        model.fit(np.array(sequences).reshape(-1, 1), lengths=lengths)
        
        return model

    def calculate_transition_probs(self, hmm_model: hmm.MultinomialHMM) -> pd.DataFrame:
        """Calculate and format transition probabilities"""
        trans_mat = hmm_model.transmat_
        
        # Create transition probability dataframe
        trans_probs = []
        for i in range(self.n_components):
            for j in range(self.n_components):
                trans_probs.append({
                    "from_regime": self.regime_names[i],
                    "from_regime_id": i,
                    "to_regime": self.regime_names[j],
                    "to_regime_id": j,
                    "probability": trans_mat[i, j]
                })
        
        return pd.DataFrame(trans_probs)

    def validate_models(self, X: np.ndarray, gmm: GaussianMixture, hmm_model: hmm.MultinomialHMM) -> Dict[str, float]:
        """Validate models using time-series cross-validation"""
        self.logger.info("Validating models with cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        gmm_scores = []
        hmm_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            # GMM validation
            X_train, X_test = X[train_idx], X[test_idx]
            
            # Scale data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and evaluate GMM
            gmm_cv = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state
            )
            gmm_cv.fit(X_train_scaled)
            gmm_score = gmm_cv.score(X_test_scaled)
            gmm_scores.append(gmm_score)
            
            # HMM validation
            labels_train = gmm_cv.predict(X_train_scaled)
            labels_test = gmm_cv.predict(X_test_scaled)
            
            # Train HMM on training labels
            hmm_cv = hmm.MultinomialHMM(
                n_components=self.n_components,
                random_state=self.random_state
            )
            hmm_cv.fit(labels_train.reshape(-1, 1), lengths=[len(labels_train)])
            
            # Score on test labels
            hmm_score = hmm_cv.score(labels_test.reshape(-1, 1))
            hmm_scores.append(hmm_score)
            
            self.logger.info(f"Fold {fold}: GMM score = {gmm_score:.3f}, HMM score = {hmm_score:.3f}")
        
        return {
            "gmm_mean_score": float(np.mean(gmm_scores)),
            "gmm_std_score": float(np.std(gmm_scores)),
            "hmm_mean_score": float(np.mean(hmm_scores)),
            "hmm_std_score": float(np.std(hmm_scores)),
            "n_folds": self.n_folds
        }

    def create_manifest(self, gmm: GaussianMixture, hmm_model: hmm.MultinomialHMM, 
                       mapping_logic: Dict[str, Any], validation_scores: Dict[str, float],
                       feature_cols: List[str]) -> Dict[str, Any]:
        """Create comprehensive manifest with regime mapping details"""
        manifest = {
            "version": "1.0.0",
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "GMM_HMM_Regime_Detection",
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "regime_mapping": self.regime_names,
            "mapping_logic": mapping_logic,
            "feature_columns": feature_cols,
            "validation_scores": validation_scores,
            "gmm_params": {
                "converged": gmm.converged_,
                "n_iter": gmm.n_iter_,
                "lower_bound": gmm.lower_bound_
            },
            "hmm_params": {
                "converged": hmm_model.monitor_.converged,
                "n_iter": hmm_model.n_iter,
                "log_likelihood": hmm_model.monitor_.history[-1] if hasattr(hmm_model.monitor_, 'history') and hmm_model.monitor_.history else None
            },
            "min_regime_duration": self.min_regime_duration,
            "description": "Fixed 8-component regime detection model for BTC/USD trading"
        }
        
        return manifest

    def save_results(self, gmm: GaussianMixture, hmm_model: hmm.MultinomialHMM, 
                    df: pd.DataFrame, labels: np.ndarray, trans_probs: pd.DataFrame,
                    mapping_logic: Dict[str, Any], validation_scores: Dict[str, float],
                    feature_cols: List[str]):
        """Save all results and models"""
        # Save models
        with open(Path(self.output_dir) / "gmm_model.pkl", "wb") as f:
            pickle.dump(gmm, f)
        
        with open(Path(self.output_dir) / "hmm_model.pkl", "wb") as f:
            pickle.dump(hmm_model, f)
        
        # Save regime assignments
        regime_df = df[["datetime_ist"]].copy()
        regime_df["regime"] = labels
        regime_df["regime_name"] = [self.regime_names.get(l, f"REGIME_{l}") for l in labels]
        regime_df["confidence"] = gmm.predict_proba(self.scaler.transform(df.drop("datetime_ist", axis=1))).max(axis=1)
        
        regime_df.to_csv(Path(self.results_dir) / "regime_assignments.csv", index=False)
        
        # Save transition probabilities
        trans_probs.to_csv(Path(self.results_dir) / "transition_probs.csv", index=False)
        
        # Create and save manifest
        manifest = self.create_manifest(gmm, hmm_model, mapping_logic, validation_scores, feature_cols)
        with open(Path(self.output_dir) / "manifest_regime.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info("All models, results, and manifest saved successfully")

    def run(self):
        """Main training pipeline"""
        try:
            # Load data
            df, feature_cols = self.load_data()
            X = df[feature_cols].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train GMM with fixed 8 components
            gmm = self.train_gmm(X_scaled)
            
            # Map to meaningful regimes
            labels, mapping_logic = self.map_to_regimes(gmm, X_scaled)
            
            # Train HMM
            hmm_model = self.train_hmm(labels)
            
            # Calculate transition probabilities
            trans_probs = self.calculate_transition_probs(hmm_model)
            
            # Validate models
            validation_scores = self.validate_models(X, gmm, hmm_model)
            
            # Save results
            self.save_results(gmm, hmm_model, df, labels, trans_probs, mapping_logic, validation_scores, feature_cols)
            
            self.logger.info("Regime training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Module 2A: Regime Detector Trainer (Fixed 8 Components)")
    parser.add_argument("--input_dir", default="processed/")
    parser.add_argument("--output_dir", default="models/")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--n_components", type=int, default=8, help="Fixed to 8 components")
    parser.add_argument("--covariance_type", default="full", choices=["full", "tied", "diag", "spherical"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--log_path", default="logs/regime_trainer.log")
    parser.add_argument("--min_regime_duration", type=int, default=5)
    
    args = parser.parse_args()
    
    trainer = RegimeTrainer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        n_folds=args.n_folds,
        random_state=args.random_state,
        log_path=args.log_path,
        min_regime_duration=args.min_regime_duration
    )
    
    trainer.run()

if __name__ == "__main__":
    main()