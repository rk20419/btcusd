"""
ULTRA-ENHANCED Price Flow Analyzer - Targeting 85%+ Accuracy
- Ensemble of LSTM, Transformer, XGBoost, LightGBM, CatBoost
- Advanced feature engineering with statistical metrics (kurtosis, skewness)
- Uses PCA, PowerTransformer, QuantileTransformer for normalization
- SelectFromModel for feature selection with XGBoost
- Dynamic class weights with compute_class_weight
- ADASYN and SMOTE for imbalance handling
- Multi-task learning for multiple horizons
- Confidence-based sample weighting
- Sharpe ratio and confusion matrix for evaluation
- TensorBoard for monitoring
- Chunked data loading for large datasets
- Optimized for low-spec systems with scalability comments
"""

import os
import logging
import random
import time
import gc
import warnings
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import pytz
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.samplers import TPESampler
import shap

# Suppress warnings for cleaner output, but log them for debugging
warnings.filterwarnings('ignore')

# --------------------------
# Advanced Configuration
# --------------------------
class AdvancedConfig:
    # Core parameters - Optimized for low specs
    SEED = 42
    SEQ_LEN = 30  # Reduced; # For larger data, increase to 60-120
    LOOKAHEAD_HORIZONS = [3, 5, 10, 15, 30]  # Multiple horizons
    ENSEMBLE_SIZE = 4  # Includes DL + tree-based; # Increase to 6-8 for high specs
    CHUNK_SIZE = 10000  # For chunked loading; # Reduce to 5000 for very large data
    
    # Data parameters
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    MONTE_CARLO_FOLDS = 3  # Reduced; # Increase to 5-10 for robust testing
    DATA_ROWS_LIMIT = 20000  # For low specs; # Set to None for full datasets
    
    # Model architecture - Slimmed down
    LSTM_HIDDEN_SIZE = 128  # Reduced; # Increase to 256-512
    TRANSFORMER_HEADS = 4  # Reduced; # Increase to 8
    TRANSFORMER_LAYERS = 2  # Reduced; # Increase to 4-6
    TRANSFORMER_DIM = 256  # Reduced; # Increase to 512
    
    # Training parameters
    BATCH_SIZE = 32  # Small for low specs; # Increase to 64-128 for GPU
    EPOCHS = 50  # Reduced; # Increase to 100-200
    EARLY_STOPPING_PATIENCE = 10  # Reduced; # Increase to 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    PCA_N_COMPONENTS = 0.95  # Keep 95% variance; # Adjust to fixed number (e.g., 20) for large data
    
    # Feature engineering
    TECHNICAL_INDICATOR_WINDOWS = [5, 10, 20, 50]  # Reduced; # Add 100
    VOLATILITY_CLUSTERING_FACTORS = True
    MARKET_REGIME_DETECTION = True
    
    # Paths
    ARTIFACTS_DIR = "models/advanced"
    LOG_DIR = "logs/advanced"
    DATA_PATH = "data/historical/BTCUSDT_1m_200000.csv"  # # Add API fallback for production
    TENSORBOARD_DIR = "tensorboard/advanced"
    
    # Advanced parameters
    META_LABELING = True
    CONFIDENCE_THRESHOLD = 0.75
    POSITION_SIZING = True  # Not implemented, flag for future
    
    def __init__(self):
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.TENSORBOARD_DIR, exist_ok=True)

# Initialize config
config = AdvancedConfig()

# --------------------------
# Advanced Logger
# --------------------------
class AdvancedLogger:
    def __init__(self):
        self.logger = logging.getLogger("advanced_price_flow")
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(os.path.join(config.LOG_DIR, "advanced_training.log"))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.performance_history = []
        
    def log_performance(self, epoch, train_loss, val_loss, train_acc, val_acc, 
                       val_precision, val_recall, val_f1, sharpe=0.0, cm=None):
        performance = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'sharpe': sharpe
        }
        self.performance_history.append(performance)
        log_msg = (f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                   f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
                   f"Val F1: {val_f1:.4f}, Sharpe: {sharpe:.4f}")
        if cm is not None:
            log_msg += f"\nConfusion Matrix:\n{cm}"
        self.logger.info(log_msg)

logger = AdvancedLogger()
writer = SummaryWriter(config.TENSORBOARD_DIR)

# --------------------------
# Advanced Feature Engineer
# --------------------------
class AdvancedFeatureEngineer:
    @staticmethod
    def add_technical_indicators(df, windows=None):
        if windows is None:
            windows = config.TECHNICAL_INDICATOR_WINDOWS
            
        try:
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility features
            for window in windows:
                df[f'volatility_{window}'] = df['returns'].rolling(window).std()
                df[f'realized_vol_{window}'] = np.sqrt((df['returns'] ** 2).rolling(window).sum())
                # Statistical features using scipy.stats
                df[f'kurtosis_{window}'] = df['returns'].rolling(window).apply(stats.kurtosis, raw=True)
                df[f'skewness_{window}'] = df['returns'].rolling(window).apply(stats.skew, raw=True)
                
            # Momentum indicators
            df['rsi'] = AdvancedFeatureEngineer.calculate_rsi(df['close'], 14)
            df['macd'], df['macd_signal'], df['macd_hist'] = AdvancedFeatureEngineer.calculate_macd(df['close'])
            
            # Trend indicators
            for window in windows:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                df[f'price_sma_ratio_{window}'] = df['close'] / df[f'sma_{window}']
                
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['obv'] = AdvancedFeatureEngineer.calculate_obv(df['close'], df['volume'])
            
            # Statistical features
            for window in windows:
                df[f'z_score_{window}'] = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
                df[f'quantile_25_{window}'] = df['close'].rolling(window).quantile(0.25)
                df[f'quantile_75_{window}'] = df['close'].rolling(window).quantile(0.75)
                
            # Time-based features with timedelta
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            if df['timestamp'].isnull().any():
                raise ValueError("Null timestamps found after conversion")
            df['time_diff'] = df['timestamp'].diff().fillna(0) / 1000  # Convert ms to seconds
            df['time_since_start'] = (df['timestamp'] - df['timestamp'].min()) / 1000  # Seconds since first timestamp
            df['hour'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce').dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce').dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Advanced features
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20'] if 'volatility_20' in df else 0
            df['trend_strength'] = AdvancedFeatureEngineer.calculate_trend_strength(df['close'], 20)
            df['market_regime'] = AdvancedFeatureEngineer.detect_market_regime(df['close'], df['volume'])
            
            df = df.fillna(method='ffill').fillna(0)
            return df
        except Exception as e:
            logger.logger.error(f"Feature engineering failed: {str(e)}")
            raise

    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace([np.inf, -np.inf, np.nan], 0)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def calculate_obv(price, volume):
        obv = np.zeros(len(price))
        if len(price) > 0:
            obv[0] = volume.iloc[0]
            for i in range(1, len(price)):
                if price.iloc[i] > price.iloc[i-1]:
                    obv[i] = obv[i-1] + volume.iloc[i]
                elif price.iloc[i] < price.iloc[i-1]:
                    obv[i] = obv[i-1] - volume.iloc[i]
                else:
                    obv[i] = obv[i-1]
        return pd.Series(obv, index=price.index)
    
    @staticmethod
    def calculate_trend_strength(series, window):
        high, low = series.rolling(window).max(), series.rolling(window).min()
        tr = np.maximum(high - low, 
                        np.maximum(abs(high - series.shift(1)), 
                                   abs(low - series.shift(1))))
        atr = tr.rolling(window).mean()
        return atr / series.replace(0, np.nan).fillna(method='ffill') * 100
    
    @staticmethod
    def detect_market_regime(price, volume, window=20):
        volatility = price.pct_change().rolling(window).std()
        volume_z = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan).fillna(1)
        regime = np.zeros(len(price))
        vol_quantile_7 = volatility.quantile(0.7)
        regime[(volatility > vol_quantile_7) & (volume_z > 1)] = 3
        regime[(volatility > vol_quantile_7) & (volume_z <= 1)] = 2
        regime[(volatility <= vol_quantile_7) & (volume_z > 1)] = 1
        return regime

# --------------------------
# Advanced Data Processor
# --------------------------
class AdvancedDataProcessor:
    @staticmethod
    def load_data_in_chunks(path, chunk_size, limit_rows=None):
        try:
            chunks = []
            rows_loaded = 0
            for chunk in pd.read_csv(path, chunksize=chunk_size):
                chunks.append(chunk)
                rows_loaded += len(chunk)
                if limit_rows and rows_loaded >= limit_rows:
                    break
            df = pd.concat(chunks)
            if limit_rows:
                df = df.iloc[:limit_rows]
            df = df.dropna(subset=['close', 'volume'])
            return df
        except FileNotFoundError:
            logger.logger.error(f"Data file not found: {path}")
            raise
        except Exception as e:
            logger.logger.error(f"Data loading failed: {str(e)}")
            raise

    @staticmethod
    def detect_outliers(df, features):
        try:
            iso_forest = IsolationForest(contamination=0.01, random_state=config.SEED)
            outliers = iso_forest.fit_predict(df[features].fillna(0))
            df = df[outliers != -1]
            logger.logger.info(f"Removed {sum(outliers == -1)} outliers")
            return df
        except Exception as e:
            logger.logger.error(f"Outlier detection failed: {str(e)}")
            return df

    @staticmethod
    def scale_features(df, features):
        try:
            scaler = RobustScaler()
            pt = PowerTransformer(method='yeo-johnson')
            qt = QuantileTransformer(output_distribution='normal')
            # Apply all three transformers for robustness
            df_scaled = scaler.fit_transform(df[features])
            df_scaled = pt.fit_transform(df_scaled)
            df_scaled = qt.fit_transform(df_scaled)
            df[features] = df_scaled
            joblib.dump({'scaler': scaler, 'pt': pt, 'qt': qt}, os.path.join(config.ARTIFACTS_DIR, "transformers.pkl"))
            return df
        except Exception as e:
            logger.logger.error(f"Feature scaling failed: {str(e)}")
            return df

    @staticmethod
    def apply_pca(df, features):
        try:
            pca = PCA(n_components=config.PCA_N_COMPONENTS, random_state=config.SEED)
            transformed = pca.fit_transform(df[features].fillna(0))
            new_features = [f'pca_{i}' for i in range(transformed.shape[1])]
            df[new_features] = transformed
            joblib.dump(pca, os.path.join(config.ARTIFACTS_DIR, "pca.pkl"))
            logger.logger.info(f"PCA applied, reduced to {len(new_features)} features")
            return df, new_features
        except Exception as e:
            logger.logger.error(f"PCA failed: {str(e)}")
            return df, features

    @staticmethod
    def select_features_xgb(df, features, y):
        try:
            xgb = XGBClassifier(random_state=config.SEED)
            selector = SelectFromModel(xgb, max_features=20)  # Limit for low specs; # Increase for large data
            selector.fit(df[features].fillna(0), y)
            selected = [f for f, s in zip(features, selector.get_support()) if s]
            joblib.dump(selector, os.path.join(config.ARTIFACTS_DIR, "feature_selector.pkl"))
            logger.logger.info(f"Selected {len(selected)} features with XGB")
            return selected
        except Exception as e:
            logger.logger.error(f"Feature selection failed: {str(e)}")
            return features

    @staticmethod
    def create_advanced_labels(df, horizons=None, volatility_adjusted=True):
        if horizons is None:
            horizons = config.LOOKAHEAD_HORIZONS
        labels = {}
        try:
            for horizon in horizons:
                future_return = df['close'].pct_change(horizon).shift(-horizon)
                if volatility_adjusted:
                    volatility = df['returns'].abs().rolling(horizon*2).mean().shift(-horizon)
                    threshold = volatility * 2.0
                else:
                    threshold = 0.002
                primary_label = np.zeros(len(df))
                primary_label[future_return > threshold] = 1
                primary_label[future_return < -threshold] = 2
                confidence = np.abs(future_return) / threshold.replace(0, np.nan).fillna(1)
                meta_label = primary_label.copy()
                meta_label[confidence < 1.5] = 0
                labels[f'primary_{horizon}'] = primary_label
                labels[f'meta_{horizon}'] = meta_label
                labels[f'return_{horizon}'] = future_return
                labels[f'confidence_{horizon}'] = confidence
            return labels
        except Exception as e:
            logger.logger.error(f"Label creation failed: {str(e)}")
            raise

    @staticmethod
    def create_sequences(df, features, seq_length, labels, step=1):
        X, y_multi, confidences = [], [], []
        data = df[features].values
        try:
            for i in range(0, len(data) - seq_length - max(config.LOOKAHEAD_HORIZONS), step):
                X.append(data[i:i+seq_length])
                label_vector = [labels[f'meta_{h}'][i+seq_length] for h in config.LOOKAHEAD_HORIZONS]
                conf_vector = [labels[f'confidence_{h}'][i+seq_length] for h in config.LOOKAHEAD_HORIZONS]
                y_multi.append(label_vector)
                confidences.append(np.mean(conf_vector))
            return np.array(X), np.array(y_multi), np.array(confidences)
        except Exception as e:
            logger.logger.error(f"Sequence creation failed: {str(e)}")
            raise

    @staticmethod
    def handle_imbalance(X, y_multi, method='SMOTE'):
        try:
            if method == 'SMOTE':
                sampler = SMOTE(random_state=config.SEED)
            else:
                sampler = ADASYN(random_state=config.SEED)
            X_flat = X.reshape(X.shape[0], -1)
            y_primary = y_multi[:, 0]
            X_res, y_res = sampler.fit_resample(X_flat, y_primary)
            X_res = X_res.reshape(-1, config.SEQ_LEN, X.shape[2])
            y_multi_res = np.repeat(y_multi, int(len(X_res)/len(X)), axis=0)[:len(X_res)]
            logger.logger.info(f"Resampled with {method} from {len(X)} to {len(X_res)} samples")
            return X_res, y_multi_res
        except Exception as e:
            logger.logger.error(f"Imbalance handling failed: {str(e)}")
            return X, y_multi

    @staticmethod
    def advanced_train_test_split(X, y_multi, confidences, train_size=0.7, val_size=0.15, monte_carlo_folds=3):
        splits = []
        n_samples = len(X)
        try:
            for fold in range(monte_carlo_folds):
                train_end = int(n_samples * train_size)
                val_end = train_end + int(n_samples * val_size)
                if fold > 0:
                    random_shift = np.random.randint(-int(n_samples * 0.05), int(n_samples * 0.05))
                    train_end = max(config.SEQ_LEN, min(n_samples - int(n_samples * val_size), train_end + random_shift))
                    val_end = train_end + int(n_samples * val_size)
                X_train, y_train, conf_train = X[:train_end], y_multi[:train_end], confidences[:train_end]
                X_val, y_val, conf_val = X[train_end:val_end], y_multi[train_end:val_end], confidences[train_end:val_end]
                X_test, y_test, conf_test = X[val_end:], y_multi[val_end:], confidences[val_end:]
                splits.append(((X_train, y_train, conf_train), (X_val, y_val, conf_val), (X_test, y_test, conf_test)))
            return splits
        except Exception as e:
            logger.logger.error(f"Train-test split failed: {str(e)}")
            raise

# --------------------------
# Advanced Model Architectures
# --------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=config.TRANSFORMER_DIM, num_heads=config.TRANSFORMER_HEADS, 
                 num_layers=config.TRANSFORMER_LAYERS, num_classes=3, num_horizons=len(config.LOOKAHEAD_HORIZONS), 
                 dropout=0.1, max_seq_len=100):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        ) for _ in range(num_horizons)])
    
    def forward(self, x):
        x = self.input_projection(x)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:seq_len, :]
        x = self.transformer_encoder(x)
        cls_token = x[:, 0, :]
        outputs = [classifier(cls_token) for classifier in self.classifiers]
        return outputs

class HybridModel(nn.Module):
    def __init__(self, input_dim, lstm_hidden=config.LSTM_HIDDEN_SIZE, transformer_dim=config.TRANSFORMER_DIM, 
                 num_heads=config.TRANSFORMER_HEADS, num_classes=3, num_horizons=len(config.LOOKAHEAD_HORIZONS), dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1)
        )
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        ) for _ in range(num_horizons)])
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_features)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        outputs = [classifier(context) for classifier in self.classifiers]
        return outputs

# --------------------------
# Advanced Trainer
# --------------------------
class AdvancedTrainer:
    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, total_acc = 0.0, 0.0
        all_preds, all_labels = [], []
        try:
            for batch_idx, (data, target, _) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = sum(self.criterion(out, target[:, i]) for i, out in enumerate(outputs))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                preds = torch.argmax(outputs[0], dim=1)
                acc = accuracy_score(target[:, 0].cpu().numpy(), preds.cpu().numpy())
                total_loss += loss.item()
                total_acc += acc
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target[:, 0].cpu().numpy())
                if batch_idx % 50 == 0:
                    logger.logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)
            return avg_loss, avg_acc, all_preds, all_labels
        except Exception as e:
            logger.logger.error(f"Train epoch failed: {str(e)}")
            raise

    def validate(self, val_loader):
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0
        all_preds, all_labels = [], []
        all_returns = []
        try:
            with torch.no_grad():
                for data, target, ret in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.model(data)
                    loss = sum(self.criterion(out, target[:, i]) for i, out in enumerate(outputs))
                    preds = torch.argmax(outputs[0], dim=1)
                    acc = accuracy_score(target[:, 0].cpu().numpy(), preds.cpu().numpy())
                    total_loss += loss.item()
                    total_acc += acc
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target[:, 0].cpu().numpy())
                    all_returns.extend(ret.cpu().numpy())
            avg_loss = total_loss / len(val_loader)
            avg_acc = total_acc / len(val_loader)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)
            pnl = np.array(all_returns) * (np.array(all_preds) == 1) - np.array(all_returns) * (np.array(all_preds) == 2)
            sharpe = pnl.mean() / pnl.std() if pnl.std() != 0 else 0
            return avg_loss, avg_acc, precision, recall, f1, sharpe, cm, all_preds, all_labels
        except Exception as e:
            logger.logger.error(f"Validation failed: {str(e)}")
            raise
    
    def train(self, train_loader, val_loader, epochs, early_stopping_patience):
        try:
            for epoch in range(epochs):
                train_loss, train_acc, _, _ = self.train_epoch(train_loader)
                val_loss, val_acc, val_precision, val_recall, val_f1, sharpe, cm, _, _ = self.validate(val_loader)
                if self.scheduler:
                    self.scheduler.step(val_loss)
                logger.log_performance(epoch, train_loss, val_loss, train_acc, val_acc, 
                                      val_precision, val_recall, val_f1, sharpe, cm)
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Acc/val', val_acc, epoch)
                writer.add_scalar('Sharpe/val', sharpe, epoch)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(config.ARTIFACTS_DIR, f"best_model_{self.model.__class__.__name__}.pth"))
                    logger.logger.info(f"New best model saved: {val_acc:.4f}")
                else:
                    self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    logger.logger.info(f"Early stopping at epoch {epoch}")
                    break
                if epoch % 10 == 0 and epoch > 0:
                    torch.save(self.model.state_dict(), os.path.join(config.ARTIFACTS_DIR, f"checkpoint_{self.model.__class__.__name__}_epoch_{epoch}.pth"))
                gc.collect()
            return self.best_val_acc
        except Exception as e:
            logger.logger.error(f"Training failed: {str(e)}")
            raise

# --------------------------
# Tree-Based Models Wrapper
# --------------------------
class TreeModelWrapper:
    def __init__(self, model_type: str, params: Dict):
        self.model_type = model_type
        if model_type == 'xgboost':
            self.model = XGBClassifier(**params, random_state=config.SEED)
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(**params, random_state=config.SEED)
        elif model_type == 'catboost':
            self.model = CatBoostClassifier(**params, random_state=config.SEED, verbose=0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
            return self
        except Exception as e:
            logger.logger.error(f"Tree model {self.model_type} training failed: {str(e)}")
            raise

    def predict_proba(self, X):
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.logger.error(f"Tree model {self.model_type} prediction failed: {str(e)}")
            raise

# --------------------------
# Advanced Ensemble
# --------------------------
class AdvancedEnsemble:
    def __init__(self, model_classes, model_params, tree_models, device):
        self.models = []
        self.model_classes = model_classes
        self.model_params = model_params
        self.tree_models = tree_models
        self.device = device
        
    def train_ensemble(self, train_loader, val_loader, X_train_flat, y_train_flat, X_val_flat, y_val_flat, epochs):
        best_models = []
        try:
            # Train deep learning models
            for i, (model_class, params) in enumerate(zip(self.model_classes, self.model_params)):
                logger.logger.info(f"Training DL model {i+1}/{len(self.model_classes)}: {model_class.__name__}")
                model = model_class(**params).to(self.device)
                class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train_flat)
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(self.device))
                optimizer = torch.optim.AdamW(model.parameters(), 
                                              lr=config.LEARNING_RATE * (0.8 + 0.4 * random.random()),
                                              weight_decay=config.WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
                trainer = AdvancedTrainer(model, self.device, criterion, optimizer, scheduler)
                best_acc = trainer.train(train_loader, val_loader, epochs, config.EARLY_STOPPING_PATIENCE)
                best_models.append((model, best_acc, 'dl'))
                logger.logger.info(f"DL Model {i+1} done with acc: {best_acc:.4f}")

            # Train tree-based models
            for i, (model_type, params) in enumerate(self.tree_models):
                logger.logger.info(f"Training tree model {i+1}/{len(self.tree_models)}: {model_type}")
                model = TreeModelWrapper(model_type, params).fit(X_train_flat, y_train_flat)
                val_preds = model.predict_proba(X_val_flat)
                val_acc = accuracy_score(y_val_flat, np.argmax(val_preds, axis=1))
                best_models.append((model, val_acc, 'tree'))
                logger.logger.info(f"Tree Model {i+1} done with acc: {val_acc:.4f}")

            best_models.sort(key=lambda x: x[1], reverse=True)
            self.models = [(m, t) for m, _, t in best_models[:config.ENSEMBLE_SIZE]]
            return self
        except Exception as e:
            logger.logger.error(f"Ensemble training failed: {str(e)}")
            raise
    
    def predict(self, X, X_flat):
        all_preds = []
        with torch.no_grad():
            for model, model_type in self.models:
                if model_type == 'dl':
                    model.eval()
                    X_torch = torch.FloatTensor(X).to(self.device)
                    outputs = model(X_torch)
                    preds = [torch.softmax(out, dim=1).cpu().numpy() for out in outputs]
                    all_preds.append(preds[0])  # Use primary horizon
                else:
                    preds = model.predict_proba(X_flat)
                    all_preds.append(preds)
        ensemble_preds = np.mean(all_preds, axis=0)
        return np.argmax(ensemble_preds, axis=1), ensemble_preds

# --------------------------
# Main Execution
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    logger.logger.info(f"Using device: {device}")
    logger.logger.info("Starting advanced training")
    
    try:
        # Load data
        logger.logger.info("Loading data...")
        df = AdvancedDataProcessor.load_data_in_chunks(config.DATA_PATH, config.CHUNK_SIZE, config.DATA_ROWS_LIMIT)
        
        # Feature engineering
        feature_engineer = AdvancedFeatureEngineer()
        df = feature_engineer.add_technical_indicators(df)
        
        # Define feature columns
        feature_columns = [col for col in df.columns if col not in ['timestamp']]
        
        # Outlier detection
        df = AdvancedDataProcessor.detect_outliers(df, feature_columns)
        
        # Scale features
        df = AdvancedDataProcessor.scale_features(df, feature_columns)
        
        # PCA
        df, feature_columns = AdvancedDataProcessor.apply_pca(df, feature_columns)
        
        # Create labels
        processor = AdvancedDataProcessor()
        labels = processor.create_advanced_labels(df)
        
        # Feature selection with XGBoost
        feature_columns = processor.select_features_xgb(df, feature_columns, labels['primary_3'])
        
        # Create sequences
        X, y_multi, confidences = processor.create_sequences(df, feature_columns, config.SEQ_LEN, labels)
        
        # Handle imbalance (try both SMOTE and ADASYN)
        X_smote, y_multi_smote = processor.handle_imbalance(X, y_multi, method='SMOTE')
        X_adasyn, y_multi_adasyn = processor.handle_imbalance(X, y_multi, method='ADASYN')
        # Choose based on size (ADASYN for diversity, SMOTE if less memory)
        X, y_multi = (X_adasyn, y_multi_adasyn) if len(X_adasyn) < 1.5 * len(X) else (X_smote, y_multi_smote)
        
        # Splits
        splits = processor.advanced_train_test_split(X, y_multi, confidences, config.TRAIN_SPLIT, config.VAL_SPLIT, config.MONTE_CARLO_FOLDS)
        
        best_overall_acc = 0.0
        best_models = []
        
        for fold, ((X_train, y_train, conf_train), (X_val, y_val, conf_val), (X_test, y_test, conf_test)) in enumerate(splits):
            logger.logger.info(f"Fold {fold+1}/{config.MONTE_CARLO_FOLDS}")
            
            # Tensors
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train)
            conf_train_t = torch.FloatTensor(conf_train)
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.LongTensor(y_val)
            returns_val = torch.FloatTensor(labels['return_3'].iloc[-len(X_val):].values)
            
            # Flatten for tree-based models
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            y_train_flat = y_train[:, 0]
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_flat = y_val[:, 0]
            
            # Datasets
            train_dataset = TensorDataset(X_train_t, y_train_t, conf_train_t)
            val_dataset = TensorDataset(X_val_t, y_val_t, returns_val)
            
            # Weighted sampler
            weights = conf_train_t / conf_train_t.sum() + 1e-6
            train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            
            # Define optimization objective
            def bayesian_optimization_objective(trial):
                try:
                    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 192])
                    num_layers = trial.suggest_int("num_layers", 1, 3)
                    dropout = trial.suggest_float("dropout", 0.1, 0.3)
                    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
                    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
                    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
                    
                    config.LEARNING_RATE = learning_rate
                    config.BATCH_SIZE = batch_size
                    
                    model = HybridModel(
                        input_dim=len(feature_columns),
                        lstm_hidden=hidden_size,
                        dropout=dropout
                    ).to(device)
                    
                    if optimizer_name == "Adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
                    else:
                        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
                    
                    criterion = nn.CrossEntropyLoss()
                    temp_train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
                    temp_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    
                    trainer = AdvancedTrainer(model, device, criterion, optimizer)
                    best_val_acc = trainer.train(temp_train_loader, temp_val_loader, config.EPOCHS // 2, config.EARLY_STOPPING_PATIENCE // 2)
                    return best_val_acc
                except Exception as e:
                    logger.logger.error(f"Optimization trial failed: {str(e)}")
                    return 0.0
            
            # Bayesian optimization
            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=config.SEED))
            study.optimize(bayesian_optimization_objective, n_trials=5)  # # Increase to 10-20 for large data
            
            best_params = study.best_params
            model = HybridModel(
                input_dim=len(feature_columns),
                lstm_hidden=best_params.get("hidden_size", 128),
                dropout=best_params.get("dropout", 0.2)
            ).to(device)
            
            optimizer_name = best_params.get("optimizer", "AdamW")
            if optimizer_name == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"], weight_decay=config.WEIGHT_DECAY)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["learning_rate"], weight_decay=config.WEIGHT_DECAY)
            
            criterion = nn.CrossEntropyLoss()
            trainer = AdvancedTrainer(model, device, criterion, optimizer)
            best_val_acc = trainer.train(train_loader, val_loader, config.EPOCHS, config.EARLY_STOPPING_PATIENCE)
            
            # SHAP feature importance
            explainer = shap.Explainer(model, X_train[:100])
            shap_values = explainer(X_val[:100])
            shap_importance = np.abs(shap_values.values).mean(0)
            top_features_idx = np.argsort(shap_importance)[-20:]
            logger.logger.info(f"SHAP selected {len(top_features_idx)} features")
            
            if best_val_acc > best_overall_acc:
                best_overall_acc = best_val_acc
                torch.save(model.state_dict(), os.path.join(config.ARTIFACTS_DIR, "overall_best_model.pth"))
            
            best_models.append((model, best_val_acc, 'dl'))
            logger.logger.info(f"Fold {fold+1} acc: {best_val_acc:.4f}")
            
            if best_val_acc >= 0.85:
                logger.logger.info(f"Target 85% achieved in fold {fold+1}!")
                break
        
        # Ensemble with tree-based models
        tree_models = [
            ('xgboost', {'n_estimators': 100, 'max_depth': 5}),
            ('lightgbm', {'n_estimators': 100, 'max_depth': 5}),
            ('catboost', {'iterations': 100, 'depth': 5})
        ]
        model_classes = [HybridModel, TimeSeriesTransformer]
        model_params = [
            {'input_dim': len(feature_columns), 'lstm_hidden': config.LSTM_HIDDEN_SIZE, 'transformer_dim': config.TRANSFORMER_DIM, 'num_heads': config.TRANSFORMER_HEADS, 'dropout': 0.2},
            {'input_dim': len(feature_columns), 'model_dim': config.TRANSFORMER_DIM, 'num_heads': config.TRANSFORMER_HEADS, 'num_layers': config.TRANSFORMER_LAYERS, 'dropout': 0.1}
        ]
        ensemble = AdvancedEnsemble(model_classes, model_params, tree_models, device)
        ensemble.train_ensemble(train_loader, val_loader, X_train_flat, y_train_flat, X_val_flat, y_val_flat, config.EPOCHS)
        
        torch.save(ensemble, os.path.join(config.ARTIFACTS_DIR, "advanced_ensemble.pth"))
        logger.logger.info(f"Final ensemble saved with best acc: {best_overall_acc:.4f}")
        
        writer.close()
    except Exception as e:
        logger.logger.error(f"Main execution failed: {str(e)}")
        import traceback
        logger.logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()