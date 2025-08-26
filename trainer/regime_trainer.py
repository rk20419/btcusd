import pandas as pd
import numpy as np
import pickle
import json
import logging
import argparse
import os
import gc
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegimeTrainer:
    def __init__(self, processed_dir, models_dir, logs_dir):
        self.processed_dir = processed_dir
        self.models_dir = os.path.join(models_dir, 'module2')  # Updated path
        self.logs_dir = logs_dir
        self.gmm = None
        self.hmm_model = None
        self.regime_names = {
        0: "TREND_UP", 1: "TREND_DOWN", 2: "RANGE_HIGH", 
        3: "RANGE_LOW", 4: "VOLATILE_UP", 5: "VOLATILE_DOWN",
        6: "STABLE_HIGH", 7: "STABLE_LOW",
        8: "VOLATILE_RANGE_HIGH", 9: "VOLATILE_RANGE_LOW"  # New regimes added
    }
        self.feature_selector = None
        
        os.makedirs(self.models_dir, exist_ok=True)  # Updated path
        os.makedirs(logs_dir, exist_ok=True)
        
    def load_processed_data(self):
        """Load all processed chunks from Module 1 with memory optimization"""
        try:
            chunk_files = [f for f in os.listdir(self.processed_dir) 
                          if f.startswith('chunk_') and f.endswith('.pkl')]
            chunk_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            if not chunk_files:
                raise ValueError("No processed chunks found in processed_dir")
            
            all_data = []
            for i, file in enumerate(chunk_files):
                chunk_path = os.path.join(self.processed_dir, file)
                logger.info(f"Loading chunk {i+1}/{len(chunk_files)}: {file}")
                
                # Memory-efficient loading with reduced precision
                chunk_data = pd.read_pickle(chunk_path)
                
                # Convert to float32 to save memory
                float_cols = chunk_data.select_dtypes(include=['float64']).columns
                chunk_data[float_cols] = chunk_data[float_cols].astype(np.float32)
                
                all_data.append(chunk_data)
                
                # Clear memory after every 2 chunks
                if (i + 1) % 2 == 0:
                    gc.collect()
                    
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(combined_data)} samples from {len(chunk_files)} chunks")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def remove_low_variance_features(self, data, threshold=0.01):
        """Remove features with very low variance that cause GMM issues"""
        try:
            # Initialize variance threshold selector
            self.feature_selector = VarianceThreshold(threshold=threshold)
            
            # Fit and transform the data
            selected_data = self.feature_selector.fit_transform(data)
            
            # Get selected feature names
            selected_features = data.columns[self.feature_selector.get_support()]
            removed_features = [col for col in data.columns if col not in selected_features]
            
            if removed_features:
                logger.warning(f"Removed low-variance features: {removed_features}")
            
            logger.info(f"Original features: {len(data.columns)}, Selected features: {len(selected_features)}")
            
            return pd.DataFrame(selected_data, columns=selected_features)
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return data
    
    def handle_special_features(self, data):
        """Handle binary and categorical features that might cause GMM issues"""
        # Identify binary features (0/1 values)
        binary_features = []
        for col in data.columns:
            unique_vals = data[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                binary_features.append(col)
        
        if binary_features:
            logger.info(f"Binary features detected: {binary_features}")
            
            # Apply small noise to binary features to avoid perfect separation
            for col in binary_features:
                noise = np.random.normal(0, 0.01, len(data))
                data[col] = np.clip(data[col] + noise, 0, 1)
        
        return data
    
    def optimize_gmm_components(self, data, max_components=12):
        """Find optimal number of GMM components using BIC with memory optimization"""
        try:
            bic_scores = []
            n_components_range = range(2, max_components + 1)
            
            # Use smaller subset for faster optimization
            sample_size = min(15000, len(data))
            sample_data = data.sample(sample_size, random_state=42) if len(data) > sample_size else data
            
            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components, 
                                     covariance_type='full',
                                     random_state=42,
                                     n_init=2,  # Reduced for memory
                                     reg_covar=1e-3,  # Increased regularization
                                     max_iter=100)  # Reduced for memory
                gmm.fit(sample_data)
                bic_scores.append(gmm.bic(sample_data))
                
            optimal_components = n_components_range[np.argmin(bic_scores)]
            logger.info(f"Optimal GMM components: {optimal_components} (BIC: {min(bic_scores):.2f})")
            return optimal_components
            
        except Exception as e:
            logger.error(f"Error optimizing GMM components: {e}")
            return 8  # Fallback to default
    
    def train_gmm(self, data, n_components=8):
        """Train Gaussian Mixture Model for regime detection with memory optimization"""
        try:
            logger.info(f"Training GMM with {n_components} components...")
            
            self.gmm = GaussianMixture(n_components=n_components,
                                      covariance_type='full',
                                      random_state=42,
                                      n_init=3,        # Reduced for memory
                                      max_iter=150,     # Reduced for memory
                                      reg_covar=1e-3)   # Increased regularization
            
            self.gmm.fit(data)
            
            # Get regime assignments
            regime_labels = self.gmm.predict(data)
            regime_probs = self.gmm.predict_proba(data)
            
            logger.info(f"GMM training completed. Convergence: {self.gmm.converged_}")
            logger.info(f"Regime distribution: {pd.Series(regime_labels).value_counts().to_dict()}")
            
            return regime_labels, regime_probs
            
        except Exception as e:
            logger.error(f"Error training GMM: {e}")
            raise
    
    def train_hmm(self, regime_labels, n_states=8):
        """Train Hidden Markov Model for regime transitions with memory optimization"""
        try:
            logger.info("Training HMM for regime transitions...")
            
            # Prepare sequences for HMM (handle chunk boundaries)
            sequences = []
            current_sequence = []
            
            for i, label in enumerate(regime_labels):
                if i > 0 and i % 50000 == 0:  # Respect chunk boundaries
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = []
                current_sequence.append(label)
            
            if current_sequence:
                sequences.append(current_sequence)
            
            # Convert to numpy arrays with lengths
            X = np.concatenate(sequences)
            lengths = [len(seq) for seq in sequences]
            
            self.hmm_model = hmm.MultinomialHMM(n_components=n_states,
                                               n_iter=50,  # Reduced for memory
                                               random_state=42,
                                               verbose=False)
            
            # Reshape for HMM (samples, 1)
            X_reshaped = X.reshape(-1, 1)
            self.hmm_model.fit(X_reshaped, lengths=lengths)
            
            logger.info(f"HMM training completed. Score: {self.hmm_model.score(X_reshaped, lengths=lengths):.2f}")
            return self.hmm_model
            
        except Exception as e:
            logger.error(f"Error training HMM: {e}")
            raise
    
    def calculate_transition_probs(self):
        """Calculate transition probabilities"""
        try:
            transition_matrix = self.hmm_model.transmat_
            
            # Base transition probabilities
            transition_probs = {}
            for i in range(transition_matrix.shape[0]):
                transition_probs[self.regime_names[i]] = {
                    self.regime_names[j]: float(transition_matrix[i, j])
                    for j in range(transition_matrix.shape[1])
                }
            
            return transition_probs
            
        except Exception as e:
            logger.error(f"Error calculating transition probabilities: {e}")
            raise
    
    def save_models_and_results(self, regime_labels, regime_probs, transition_probs, feature_names):
        """Save models, assignments, and probabilities"""
        try:
            # Save GMM model
            gmm_path = os.path.join(self.models_dir, 'gmm_model.pkl')
            with open(gmm_path, 'wb') as f:
                pickle.dump(self.gmm, f)
            
            # Save HMM model
            hmm_path = os.path.join(self.models_dir, 'hmm_model.pkl')
            with open(hmm_path, 'wb') as f:
                pickle.dump(self.hmm_model, f)
            
            # Save feature selector
            if self.feature_selector:
                selector_path = os.path.join(self.models_dir, 'feature_selector.pkl')
                with open(selector_path, 'wb') as f:
                    pickle.dump(self.feature_selector, f)
            
            # Save feature names
            features_path = os.path.join(self.models_dir, 'selected_features.pkl')
            with open(features_path, 'wb') as f:
                pickle.dump(feature_names, f)
            
            # Save regime assignments (in chunks to avoid memory issues)
            assignments_path = os.path.join(self.logs_dir, 'regime_assignments.json')
            
            # Save in chunks for large data
            chunk_size = 50000
            assignments_data = {
                'timestamps': [],
                'regimes': [],
                'regime_names': []
            }
            
            for i in range(0, len(regime_labels), chunk_size):
                chunk_end = min(i + chunk_size, len(regime_labels))
                assignments_data['timestamps'].extend(list(range(i, chunk_end)))
                assignments_data['regimes'].extend(regime_labels[i:chunk_end].tolist())
                assignments_data['regime_names'].extend([self.regime_names[x] for x in regime_labels[i:chunk_end]])
                
                # Clear memory
                if i % 100000 == 0:
                    gc.collect()
            
            with open(assignments_path, 'w') as f:
                json.dump(assignments_data, f, indent=2)
            
            # Save transition probabilities
            trans_probs_path = os.path.join(self.logs_dir, 'transition_probs.json')
            with open(trans_probs_path, 'w') as f:
                json.dump(transition_probs, f, indent=2)
            
            logger.info(f"Models saved to {self.models_dir}")
            logger.info(f"Results saved to {self.logs_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models and results: {e}")
            raise
    
    def validate_models(self, data, regime_labels):
        """Validate model performance with memory optimization"""
        try:
            # Silhouette score (on a sample to save memory)
            if len(np.unique(regime_labels)) > 1:
                sample_size = min(10000, len(data))
                sample_data = data.sample(sample_size, random_state=42)
                sample_labels = regime_labels[:sample_size]
                silhouette = silhouette_score(sample_data, sample_labels)
                logger.info(f"Silhouette Score: {silhouette:.3f}")
            else:
                logger.warning("Only one regime cluster found, skipping silhouette score")
            
            # Cross-validation for HMM (simplified)
            tscv = TimeSeriesSplit(n_splits=2)  # Reduced for memory
            hmm_scores = []
            
            for train_idx, test_idx in tscv.split(data):
                if len(train_idx) > 1000 and len(test_idx) > 1000:  # Ensure enough data
                    train_labels = regime_labels[train_idx]
                    test_labels = regime_labels[test_idx]
                    
                    if len(np.unique(train_labels)) > 1 and len(np.unique(test_labels)) > 1:
                        # Train temporary HMM on a subset
                        temp_hmm = hmm.MultinomialHMM(n_components=8, n_iter=30, random_state=42)
                        temp_hmm.fit(train_labels[:5000].reshape(-1, 1))
                        
                        # Score on test data subset
                        score = temp_hmm.score(test_labels[:5000].reshape(-1, 1))
                        hmm_scores.append(score)
            
            if hmm_scores:
                logger.info(f"HMM Cross-validation scores: {hmm_scores}")
                logger.info(f"Average HMM score: {np.mean(hmm_scores):.3f}")
            else:
                logger.warning("HMM cross-validation skipped due to insufficient regime diversity")
                
        except Exception as e:
            logger.warning(f"Model validation skipped due to error: {e}")
    
    def run_training(self):
        """Main training pipeline with memory optimization"""
        try:
            logger.info("Starting Regime Trainer...")
            
            # Load processed data
            data = self.load_processed_data()
            
            # Use only feature columns (exclude timestamp and other metadata)
            feature_cols = [col for col in data.columns if col not in 
                           ['timestamp', 'datetime_ist', 'hidden_divergence_bear']]
            feature_data = data[feature_cols]
            
            # Handle missing values
            feature_data = feature_data.fillna(feature_data.median())
            
            # Handle special features (binary, categorical)
            feature_data = self.handle_special_features(feature_data)
            
            # Remove low variance features that cause GMM issues
            feature_data = self.remove_low_variance_features(feature_data, threshold=0.01)
            
            # Optimize GMM components
            optimal_components = self.optimize_gmm_components(feature_data)
            
            # Train GMM
            regime_labels, regime_probs = self.train_gmm(feature_data, optimal_components)
            
            # Train HMM
            self.train_hmm(regime_labels, optimal_components)
            
            # Calculate transition probabilities
            transition_probs = self.calculate_transition_probs()
            
            # Validate models
            self.validate_models(feature_data, regime_labels)
            
            # Save results
            self.save_models_and_results(pd.Series(regime_labels), regime_probs, 
                                       transition_probs, feature_data.columns.tolist())
            
            logger.info("Regime training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Trainer for Module 2: Regime Detector')
    parser.add_argument('--processed_dir', type=str, default='processed')
    parser.add_argument('--models_dir', type=str, default='models')
    parser.add_argument('--logs_dir', type=str, default='logs/regime')
    
    args = parser.parse_args()
    
    trainer = RegimeTrainer(args.processed_dir, args.models_dir, args.logs_dir)
    trainer.run_training()

if __name__ == '__main__':
    main()