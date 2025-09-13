from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Setup logging
def setup_logger(log_path: str):
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("RegimePredictor")
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

class CSVFileHandler(FileSystemEventHandler):
    """Watchdog handler to detect changes in CSV file"""
    def __init__(self, predictor):
        self.predictor = predictor
        self.last_processed_time = 0
        
    def on_modified(self, event):
        if event.src_path.endswith('.csv') and event.src_path == self.predictor.input_path:
            current_time = time.time()
            # Debounce to avoid multiple rapid triggers
            if current_time - self.last_processed_time > 2:
                self.last_processed_time = current_time
                self.predictor.logger.info("CSV file modified, processing new data")
                self.predictor.run()

class RegimePredictor:
    def __init__(
        self,
        model_dir: str = "models/",
        input_path: str = "processed/live/live_features.csv",
        output_dir: str = "results/live/",
        threshold: float = 0.7,
        atr_threshold: float = 1.5,
        log_path: str = "logs/regime_predictor.log"
    ):
        self.model_dir = model_dir
        self.input_path = input_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.atr_threshold = atr_threshold
        
        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(log_path)
        self.scaler = None
        self.gmm = None
        self.hmm_model = None
        self.regime_history = []
        self.manifest = None
        self.observer = None
        
        # Load models, manifest, and scaler
        self.load_models()
        self.load_manifest()
        self.load_scaler()

    def load_models(self):
        """Load trained GMM and HMM models"""
        try:
            # Load GMM model
            with open(Path(self.model_dir) / "gmm_model.pkl", "rb") as f:
                self.gmm = pickle.load(f)
            
            # Load HMM model
            with open(Path(self.model_dir) / "hmm_model.pkl", "rb") as f:
                self.hmm_model = pickle.load(f)
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise

    def load_scaler(self):
        """Load the canonical scaler from Module 1"""
        try:
            # Use the canonical scaler from Module 1
            scaler_path = Path("scalers/scaler_canonical.pkl")
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            self.logger.info(f"Scaler loaded successfully from {scaler_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {str(e)}")
            raise

    def load_manifest(self):
        """Load regime manifest file"""
        try:
            manifest_path = Path(self.model_dir) / "manifest_regime.json"
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
            
            self.logger.info("Manifest loaded successfully")
            self.logger.info(f"Model has fixed {self.manifest['n_components']} components")
            
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {str(e)}")
            raise

    def load_live_data(self) -> pd.DataFrame:
        """Load live features data"""
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Live data file not found: {self.input_path}")
        
        df = pd.read_csv(self.input_path)
        
        # Use scaled features if available
        scaled_cols = [col for col in df.columns if col.endswith("_scaled")]
        if scaled_cols:
            df = df[["datetime_ist"] + scaled_cols]
            # Remove _scaled suffix for processing
            df.columns = [col.replace("_scaled", "") for col in df.columns]
        
        return df

    def predict_current_regime(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict current regime using GMM"""
        # Scale features using the canonical scaler
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict regime
        regime_probs = self.gmm.predict_proba(features_scaled)[0]
        regime = np.argmax(regime_probs)
        confidence = regime_probs[regime]
        
        return regime, confidence

    def predict_next_regime(self, current_regime: int) -> Dict[str, float]:
        """Predict next regime probabilities using HMM"""
        # Get transition probabilities from HMM
        trans_probs = self.hmm_model.transmat_[current_regime]
        
        # Format results using regime names from manifest
        next_regime_probs = {}
        for i, prob in enumerate(trans_probs):
            regime_name = self.manifest["regime_mapping"].get(str(i), f"REGIME_{i}")
            next_regime_probs[regime_name] = float(prob)
        
        return next_regime_probs

    def check_early_warnings(self, next_regime_probs: Dict[str, float], 
                            current_regime: int, confidence: float,
                            atr_value: float) -> List[Dict[str, Any]]:
        """Check for early warning signals"""
        warnings = []
        
        # Get current regime name
        current_regime_name = self.manifest["regime_mapping"].get(str(current_regime), f"REGIME_{current_regime}")
        
        # Check for high probability regime transitions
        for regime, prob in next_regime_probs.items():
            if prob > self.threshold:
                # Adjust threshold based on volatility
                adjusted_threshold = self.threshold
                if atr_value > self.atr_threshold:
                    adjusted_threshold = 0.85  # Higher threshold in high volatility
                
                if prob > adjusted_threshold:
                    warnings.append({
                        "type": "HIGH_PROBABILITY_TRANSITION",
                        "message": f"High probability transition from {current_regime_name} to {regime}: {prob:.2f}",
                        "from_regime": current_regime_name,
                        "to_regime": regime,
                        "probability": float(prob),
                        "threshold": float(adjusted_threshold)
                    })
        
        # Check for low confidence in current regime
        if confidence < 0.6:
            warnings.append({
                "type": "LOW_CONFIDENCE",
                "message": f"Low confidence in current regime {current_regime_name}: {confidence:.2f}",
                "regime": current_regime_name,
                "confidence": float(confidence),
                "threshold": 0.6
            })
        
        return warnings

    def save_results(self, current_regime: int, confidence: float, 
                    next_regime_probs: Dict[str, float], warnings: List[Dict[str, Any]]):
        """Save prediction results"""
        # Get regime name from manifest
        current_regime_name = self.manifest["regime_mapping"].get(str(current_regime), f"REGIME_{current_regime}")
        
        # Save current regime
        current_regime_df = pd.DataFrame({
            "timestamp": [pd.Timestamp.now()],
            "datetime_ist": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
            "regime": [current_regime_name],
            "regime_id": [current_regime],
            "confidence": [confidence]
        })
        
        # Append to existing file or create new
        output_file = Path(self.output_dir) / "current_regime.csv"
        if output_file.exists():
            current_regime_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            current_regime_df.to_csv(output_file, index=False)
        
        # Save next regime probabilities
        next_regime_df = pd.DataFrame(list(next_regime_probs.items()), columns=["regime", "probability"])
        next_regime_df["timestamp"] = pd.Timestamp.now()
        
        next_output_file = Path(self.output_dir) / "next_regime_probs.csv"
        if next_output_file.exists():
            next_regime_df.to_csv(next_output_file, mode='a', header=False, index=False)
        else:
            next_regime_df.to_csv(next_output_file, index=False)
        
        # Save early warnings
        if warnings:
            warnings_df = pd.DataFrame(warnings)
            warnings_df["timestamp"] = pd.Timestamp.now()
            
            warnings_output_file = Path(self.output_dir) / "early_warnings.csv"
            if warnings_output_file.exists():
                warnings_df.to_csv(warnings_output_file, mode='a', header=False, index=False)
            else:
                warnings_df.to_csv(warnings_output_file, index=False)
        
        # Update regime history
        self.regime_history.append({
            "timestamp": pd.Timestamp.now(),
            "regime": current_regime,
            "regime_name": current_regime_name,
            "confidence": confidence
        })
        
        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        self.logger.info("Results saved successfully")

    def run(self):
        """Main prediction pipeline"""
        try:
            # Load live data
            df = self.load_live_data()
            
            if len(df) == 0:
                self.logger.warning("No data available for prediction")
                return
            
            # Get the latest data point
            latest_data = df.iloc[-1]
            datetime_str = latest_data["datetime_ist"]
            features = latest_data.drop("datetime_ist").values
            
            # Extract ATR value for volatility adjustment
            atr_value = latest_data.get("atr_14", 1.0) if "atr_14" in latest_data else 1.0
            
            # Predict current regime
            current_regime, confidence = self.predict_current_regime(features)
            
            # Predict next regime probabilities
            next_regime_probs = self.predict_next_regime(current_regime)
            
            # Check for early warnings
            warnings = self.check_early_warnings(next_regime_probs, current_regime, confidence, atr_value)
            
            # Save results
            self.save_results(current_regime, confidence, next_regime_probs, warnings)
            
            # Log results
            current_regime_name = self.manifest["regime_mapping"].get(str(current_regime), f"REGIME_{current_regime}")
            self.logger.info(f"Current regime: {current_regime_name} (confidence: {confidence:.2f})")
            self.logger.info(f"Next regime probabilities: {json.dumps(next_regime_probs, indent=2)}")
            
            if warnings:
                for warning in warnings:
                    self.logger.warning(warning["message"])
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def start_monitoring(self):
        """Start monitoring the CSV file for changes"""
        self.logger.info("Starting file monitoring with watchdog...")
        
        # Set up watchdog observer
        event_handler = CSVFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, os.path.dirname(self.input_path), recursive=False)
        self.observer.start()
        
        try:
            # Run once initially to process any existing data
            self.run()
            
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Stopped by user")
        finally:
            self.observer.stop()
            self.observer.join()

def main():
    parser = argparse.ArgumentParser(description="Module 2B: Regime Predictor (Fixed 8 Components)")
    parser.add_argument("--model_dir", default="models/")
    parser.add_argument("--input_path", default="processed/live/live_features.csv")
    parser.add_argument("--output_dir", default="results/live/")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--atr_threshold", type=float, default=1.5)
    parser.add_argument("--log_path", default="logs/regime_predictor.log")
    
    args = parser.parse_args()
    
    predictor = RegimePredictor(
        model_dir=args.model_dir,
        input_path=args.input_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        atr_threshold=args.atr_threshold,
        log_path=args.log_path
    )
    
    # Start monitoring for file changes
    predictor.start_monitoring()

if __name__ == "__main__":
    main()