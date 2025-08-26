import pandas as pd
import numpy as np
import pickle
import json
import logging
import argparse
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegimePredictor:
    def __init__(self, models_dir, output_dir, logs_dir):
        # Updated paths as per your requirements
        self.models_dir = os.path.join(models_dir, 'module2')  # models/module2/
        self.output_dir = os.path.join(output_dir, 'live_predictor')  # data/live_predictor/
        self.logs_dir = logs_dir  # logs/
        
        self.gmm = None
        self.hmm_model = None
        self.transition_probs = None
        self.feature_selector = None
        self.selected_features = None
        self.regime_history = []
        self.current_regime = None
        self.regime_confidence = 0.0
        
        self.regime_names = {
            0: "TREND_UP", 1: "TREND_DOWN", 2: "RANGE_HIGH", 
            3: "RANGE_LOW", 4: "VOLATILE_UP", 5: "VOLATILE_DOWN",
            6: "STABLE_HIGH", 7: "STABLE_LOW"
        }
        
        # Create directories with updated paths
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.load_models()
    
    def load_models(self):
        """Load trained GMM and HMM models"""
        try:
            # Load GMM model
            gmm_path = os.path.join(self.models_dir, 'gmm_model.pkl')
            if not os.path.exists(gmm_path):
                raise FileNotFoundError(f"GMM model not found at {gmm_path}")
                
            with open(gmm_path, 'rb') as f:
                self.gmm = pickle.load(f)
            
            # Load HMM model
            hmm_path = os.path.join(self.models_dir, 'hmm_model.pkl')
            if not os.path.exists(hmm_path):
                raise FileNotFoundError(f"HMM model not found at {hmm_path}")
                
            with open(hmm_path, 'rb') as f:
                self.hmm_model = pickle.load(f)
            
            # Load feature selector if exists
            selector_path = os.path.join(self.models_dir, 'feature_selector.pkl')
            if os.path.exists(selector_path):
                with open(selector_path, 'rb') as f:
                    self.feature_selector = pickle.load(f)
            
            # Load selected features
            features_path = os.path.join(self.models_dir, 'selected_features.pkl')
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.selected_features = pickle.load(f)
            
            # Load transition probabilities - updated path
            trans_probs_path = os.path.join(self.logs_dir, 'regime', 'transition_probs.json')
            if os.path.exists(trans_probs_path):
                with open(trans_probs_path, 'r') as f:
                    self.transition_probs = json.load(f)
            else:
                logger.warning("Transition probabilities file not found, using HMM transition matrix")
                self.transition_probs = self.get_transition_probs_from_hmm()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_transition_probs_from_hmm(self):
        """Get transition probabilities directly from HMM model"""
        transition_matrix = self.hmm_model.transmat_
        transition_probs = {}
        
        for i in range(transition_matrix.shape[0]):
            transition_probs[self.regime_names[i]] = {
                self.regime_names[j]: float(transition_matrix[i, j])
                for j in range(transition_matrix.shape[1])
            }
        
        return transition_probs
    
    def select_features(self, live_features):
        """Select only the features used during training"""
        if self.selected_features is not None:
            # Ensure we have all the required features
            missing_features = [f for f in self.selected_features if f not in live_features.index]
            if missing_features:
                logger.warning(f"Missing features in live data: {missing_features}")
                # Add missing features with default values
                for f in missing_features:
                    live_features[f] = 0
            
            # Select only the features used during training
            live_features = live_features[self.selected_features]
        
        return live_features
    
    def calculate_dynamic_threshold(self, atr_ratio, session):
        """Calculate dynamic confidence threshold based on volatility and session"""
        base_threshold = 0.75
        
        # Adjust for volatility
        if atr_ratio > 1.5:  # High volatility
            volatility_adj = 0.1
        elif atr_ratio < 0.5:  # Low volatility
            volatility_adj = -0.05
        else:
            volatility_adj = 0
        
        # Adjust for trading session
        session_thresholds = {
            'asia': 0.7,    # Lower threshold for Asia session
            'london': 0.75,
            'ny': 0.8       # Higher threshold for NY session
        }
        
        session_adj = session_thresholds.get(session, 0.75)
        
        return max(0.6, min(0.95, base_threshold + volatility_adj + (session_adj - 0.75)))
    
    def generate_early_warnings(self, next_regime_probs, current_features):
        """Generate early warnings for high-probability regime transitions"""
        warnings = []
        
        # Get current session and volatility for dynamic threshold
        session = self.get_current_session()
        atr_ratio = current_features.get('atr_14', 1.0) / current_features.get('atr_14_avg', 1.0) if 'atr_14_avg' in current_features else 1.0
        
        dynamic_threshold = self.calculate_dynamic_threshold(atr_ratio, session)
        
        # Check for high-probability transitions
        for regime, prob in next_regime_probs.items():
            if prob > dynamic_threshold:
                warning = {
                    "type": "regime_transition",
                    "expected_regime": regime,
                    "probability": float(prob),
                    "threshold": float(dynamic_threshold),
                    "triggers": []
                }
                
                # Add triggers based on technical factors
                if current_features.get('volume_spike_5', 1) > 2.0:
                    warning["triggers"].append("volume_spike")
                
                if current_features.get('rsi_14', 50) > 70:
                    warning["triggers"].append("overbought")
                elif current_features.get('rsi_14', 50) < 30:
                    warning["triggers"].append("oversold")
                
                if current_features.get('macd', 0) > current_features.get('macd_signal', 0):
                    warning["triggers"].append("macd_bullish")
                else:
                    warning["triggers"].append("macd_bearish")
                
                warnings.append(warning)
        
        return warnings
    
    def get_current_session(self):
        """Get current trading session based on UTC time"""
        current_hour = datetime.utcnow().hour
        
        if 0 <= current_hour < 8:    # Asia session
            return 'asia'
        elif 8 <= current_hour < 13: # London session
            return 'london'
        else:                        # NY session
            return 'ny'
    
    def process_live_data(self, live_features):
        """Process live feature data and predict regimes"""
        try:
            # Handle missing values
            live_features = live_features.fillna(0)
            
            # Select only the features used during training
            live_features = self.select_features(live_features)
            
            # Get current regime
            regime_probs = self.gmm.predict_proba(live_features.values.reshape(1, -1))[0]
            current_regime_idx = np.argmax(regime_probs)
            current_regime = self.regime_names[current_regime_idx]
            confidence = float(regime_probs[current_regime_idx])
            
            # Predict next regime probabilities
            if self.transition_probs:
                # Use pre-calculated transition probabilities
                next_regime_probs = self.transition_probs.get(current_regime, {})
            else:
                # Fallback: Use HMM transition matrix directly
                next_state_probs = self.hmm_model.transmat_[current_regime_idx]
                next_regime_probs = {
                    self.regime_names[i]: float(next_state_probs[i])
                    for i in range(len(self.regime_names))
                }
            
            # Generate early warnings
            current_features_dict = live_features.to_dict()
            warnings = self.generate_early_warnings(next_regime_probs, current_features_dict)
            
            # Update state
            self.current_regime = current_regime
            self.regime_confidence = confidence
            self.regime_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'regime': current_regime,
                'confidence': confidence,
                'next_regime_probs': next_regime_probs
            })
            
            # Keep only last 1000 records
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            return {
                'current_regime': current_regime,
                'confidence': confidence,
                'next_regime_probs': next_regime_probs,
                'warnings': warnings,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing live data: {e}")
            return None
    
    def save_results(self, results):
        """Save prediction results to files"""
        if not results:
            return
        
        try:
            # Save current regime
            current_regime_path = os.path.join(self.output_dir, 'current_regime.json')
            with open(current_regime_path, 'w') as f:
                json.dump({
                    'regime': results['current_regime'],
                    'confidence': results['confidence'],
                    'timestamp': results['timestamp']
                }, f, indent=2)
            
            # Save next regime probabilities
            next_regime_path = os.path.join(self.output_dir, 'next_regime_probs.json')
            with open(next_regime_path, 'w') as f:
                json.dump(results['next_regime_probs'], f, indent=2)
            
            # Save warnings
            warnings_path = os.path.join(self.output_dir, 'early_warnings.json')
            with open(warnings_path, 'w') as f:
                json.dump(results['warnings'], f, indent=2)
            
            # Log history - updated path
            log_path = os.path.join(self.logs_dir, 'regime_history.jsonl')
            with open(log_path, 'a') as f:
                log_entry = {
                    'timestamp': results['timestamp'],
                    'results': results
                }
                f.write(json.dumps(log_entry) + '\n')
                
            logger.debug(f"Results saved to {self.output_dir}")
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def monitor_live_features(self, features_path, poll_interval=5):
        """Monitor live features and predict regimes"""
        logger.info(f"Monitoring live features: {features_path}")
        
        last_processed_time = 0
        
        while True:
            try:
                if os.path.exists(features_path):
                    # Check file modification time to avoid reprocessing
                    file_mtime = os.path.getmtime(features_path)
                    
                    if file_mtime > last_processed_time:
                        # Load live features
                        live_features = pd.read_pickle(features_path)
                        
                        if not live_features.empty:
                            # Process features - use the latest row (assuming chronological order)
                            results = self.process_live_data(live_features.iloc[-1])
                            
                            if results:
                                # Save results
                                self.save_results(results)
                                
                                logger.info(
                                    f"Regime: {results['current_regime']} "
                                    f"(Confidence: {results['confidence']:.2f}) | "
                                    f"Next: {max(results['next_regime_probs'].items(), key=lambda x: x[1])[0]}"
                                )
                                
                                if results['warnings']:
                                    for warning in results['warnings']:
                                        logger.warning(
                                            f"EARLY WARNING: {warning['expected_regime']} "
                                            f"(Prob: {warning['probability']:.2f}, Threshold: {warning['threshold']:.2f})"
                                        )
                                
                                last_processed_time = file_mtime
                
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(poll_interval * 2)

def main():
    parser = argparse.ArgumentParser(description='Predictor for Module 2: Regime Detector')
    parser.add_argument('--models_dir', type=str, default='models')
    parser.add_argument('--features_path', type=str, default='processed/live/live_features.pkl')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--poll_interval', type=int, default=5)
    
    args = parser.parse_args()
    
    predictor = RegimePredictor(args.models_dir, args.output_dir, args.logs_dir)
    predictor.monitor_live_features(args.features_path, args.poll_interval)

if __name__ == '__main__':
    main()