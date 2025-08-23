# trainer/module2_model_trainer.py
"""
PRODUCTION-READY Price Flow Analyzer Module - OPTIMIZED FOR 100K
- Memory-efficient training for large datasets
- Progressive loading and training in chunks of 5000
- Optimized for low-spec systems
- Enhanced fail-safes and monitoring
"""

# Section 1: Imports
# For larger data (>100k rows), increase MIN_IPCA_BATCH to 10000+ for PCA stability, 
# increase MAX_SAMPLES_PER_CLASS to 10000+ for more balanced data, 
# increase BATCH_SIZE to 256+ if RAM allows, decrease SEQ_LEN to 20 if memory errors occur.
import os
import logging
import random
from typing import Tuple, List, Optional, Union
from collections import Counter
from dataclasses import dataclass
import gc
import numpy as np
import pandas as pd
import joblib
import pytz
from datetime import datetime

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Section 2: Configuration
# For larger data, increase EPOCHS to 100 if training time allows, increase HYPER_OPT_TRIALS to 50 for better optimization.
# Increased MIN_STABLE_CANDLES to 4 for stricter stability, reducing false positives.
@dataclass
class Config:
    SEED = 42
    SEQ_LEN = 30  # Reduced for faster response
    MAX_LOOKAHEAD = 30  # Reduced for faster response
    MIN_STABLE_CANDLES = 4  # Increased for better stability detection
    MAX_STABLE_CANDLES = 5
    PRICE_RANGE = 300
    EPOCHS = 50
    BATCH_SIZE = 128
    LR = 1e-3
    HIDDEN_SIZE = 192  # Increased for better capacity
    NUM_LAYERS = 2
    EMBED_DIM = 32
    ANN_CANDIDATES = 100  # Reduced for speed
    TOP_M_FOR_DTW = 20  # Reduced for speed
    DTW_TOPK = 3
    TRAIN_SPLIT_FRAC = 0.80
    VAL_SPLIT_FRAC = 0.15
    ARTIFACTS_DIR = "models/production"
    LOG_DIR = "logs/production"
    DATA_PATH = "data/historical/BTCUSDT_1m_100000.csv"  # Use larger dataset
    MIN_IPCA_BATCH = 8000
    EARLY_STOP_PATIENCE = 25  # Increased for more training
    MAX_SAMPLES_PER_CLASS = 8000  # Increased to retain more data
    REAL_TIME_UPDATE_INTERVAL = 30
    MAX_RETRIES = 3
    REALTIME_SEQ_LEN = 30
    
    CONFIDENCE_THRESHOLD = 0.65
    MIN_DATA_QUALITY_SCORE = 0.8
    MAX_LATENCY_MS = 500
    PERFORMANCE_WINDOW = 100
    ALERT_THRESHOLD = 5
    ATR_PERIOD = 14  # For dynamic thresholds
    HYPER_OPT_TRIALS = 20  # For Optuna

# Initialize config
config = Config()
os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs("data/real_time", exist_ok=True)

# Section 3: Enhanced Logging (health check removed)
class ProductionLogger:
    def __init__(self):
        self.logger = logging.getLogger("price_flow_production")
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(os.path.join(config.LOG_DIR, "production.log"))
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

logger = ProductionLogger()

# Section 4: Enhanced Model
class ProductionPriceFlowLSTM(nn.Module):
    def __init__(self, input_size: int = 12, hidden_size: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS, output_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.dropout(context)
        return self.fc(out)

# Section 5: Data Quality System
class DataQualityValidator:
    @staticmethod
    def validate_real_time_data(df: pd.DataFrame) -> Tuple[bool, float]:
        if df.empty:
            return False, 0.0
        quality_score = 1.0
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        quality_score -= missing_ratio * 0.3
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        zero_ratio = (df[critical_cols] == 0).sum().sum() / (len(critical_cols) * df.shape[0])
        quality_score -= zero_ratio * 0.4
        if 'timestamp' in df.columns:
            time_diffs = df['timestamp'].diff().dropna()
            if not time_diffs.empty:
                abnormal_gaps = (time_diffs > 120000).sum()  # >2 minutes
                gap_penalty = abnormal_gaps / len(time_diffs) * 0.3
                quality_score -= gap_penalty
        if 'close' in df.columns:
            outliers = (df['high'] - df['low'] > 0.05 * df['close']).sum() / df.shape[0]
            quality_score -= outliers * 0.2
        return quality_score >= config.MIN_DATA_QUALITY_SCORE, quality_score

# Section 6: Enhanced Utilities
def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

def detect_session_from_timestamp(ts_ms: int) -> Tuple[str, bool]:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=pytz.UTC)
    hour = dt.hour
    is_sunday = (dt.weekday() == 6)
    if 0 <= hour < 8:
        return "Asia", is_sunday
    elif 8 <= hour < 16:
        return "London", is_sunday
    else:
        return "NY", is_sunday

def ensure_volatility(df: pd.DataFrame):
    if 'volatility' not in df.columns or df['volatility'].isnull().all():
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        df['volatility'] = df['tr'].rolling(config.ATR_PERIOD).mean().fillna(0) * 100
        df.drop(columns=['tr'], inplace=True)
    else:
        df['volatility'] = pd.to_numeric(df['volatility'], errors='coerce').fillna(0)

def calculate_rsi(df: pd.DataFrame, period: int = 14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)  # Neutral default

def calculate_macd(df: pd.DataFrame, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
    short_ema = df['close'].ewm(span=short_period, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    df['macd'] = macd_line - signal_line
    df['macd'] = df['macd'].fillna(0)

def calculate_stability_raw(df_raw: pd.DataFrame, start_idx: int, window: int = 3) -> Tuple[Optional[float], bool]:
    if start_idx + window > len(df_raw):
        return None, False
    subset = df_raw.iloc[start_idx:start_idx+window]
    if start_idx < 19:
        vol20 = df_raw['volatility'].iloc[:start_idx+1].mean() if start_idx >= 0 else df_raw['volatility'].iloc[0]
    else:
        vol20 = df_raw['volatility'].rolling(20).mean().iloc[start_idx]
    if np.isnan(vol20) or vol20 <= 0:
        vol20 = subset['volatility'].mean() or 1e-5  # Fallback to avoid nan/zero
    current_vol = subset['volatility'].mean()
    price_range = subset['high'].max() - subset['low'].min()
    is_stable = (current_vol < 0.5 * vol20) and (price_range < config.PRICE_RANGE)
    avg_close = subset['close'].mean() if is_stable else None
    return avg_close, is_stable

def compute_dtw_confidence_on_candidates(current_seq_norm: np.ndarray,
                                        candidate_idxs: List[int],
                                        X_norm: np.ndarray,
                                        labels: np.ndarray,
                                        topk: int = config.DTW_TOPK) -> float:
    distances = []
    for i in candidate_idxs:
        dist, _ = fastdtw(current_seq_norm, X_norm[i], dist=euclidean)
        distances.append((dist, int(labels[i])))
    distances.sort(key=lambda x: x[0])
    top = distances[:topk]
    if not top:
        return 0.5
    label_scores = {}
    for d, lab in top[:3]:  # Top-3 for ensemble voting
        w = 1.0 / (d + 1e-8)
        label_scores[lab] = label_scores.get(lab, 0.0) + w
    majority_label = max(label_scores, key=label_scores.get)
    conf = label_scores[majority_label] / sum(label_scores.values())
    top_labels = [lab for _, lab in top[:3]]
    majority_vote = Counter(top_labels).most_common(1)[0][0]
    if majority_vote != majority_label:
        conf *= 0.8  # Penalize if vote differs
    return float(conf)

def check_price_stability(df_raw: pd.DataFrame, start_idx: int, current_price: float) -> Tuple[bool, int]:
    if start_idx >= len(df_raw):
        return False, 0
    stable_count = 0
    for j in range(start_idx, min(start_idx + config.MAX_STABLE_CANDLES, len(df_raw))):
        high = df_raw['high'].iat[j]
        low = df_raw['low'].iat[j]
        if abs(high - current_price) <= config.PRICE_RANGE and abs(low - current_price) <= config.PRICE_RANGE:
            stable_count += 1
            if stable_count >= config.MIN_STABLE_CANDLES:
                return True, stable_count
        else:
            stable_count = 0
    return False, stable_count

def balance_classes(X, y, metadata, max_samples_per_class=config.MAX_SAMPLES_PER_CLASS):
    logger.logger.info(f"Original class distribution: {Counter(y)}")
    smote = SMOTE(random_state=config.SEED)
    X_flat = X.reshape(X.shape[0], -1)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    X_resampled = X_resampled.reshape(-1, config.SEQ_LEN, X.shape[2])
    
    balanced_X, balanced_y, balanced_meta = [], [], []
    for class_id in [0, 1, 2]:
        indices = np.where(y_resampled == class_id)[0]
        if len(indices) > max_samples_per_class:
            indices = np.random.choice(indices, max_samples_per_class, replace=False)
        balanced_X.append(X_resampled[indices])
        balanced_y.append(y_resampled[indices])
        balanced_meta.extend([metadata[i % len(metadata)] for i in indices])
    
    balanced_X = np.vstack(balanced_X)
    balanced_y = np.concatenate(balanced_y)
    logger.logger.info(f"Balanced class distribution: {Counter(balanced_y)}")
    return balanced_X, balanced_y, balanced_meta

# Section 7: Enhanced Sequence Building (chunk size 5000)
def load_and_build_sequences(filepath: str, seq_len: int = config.SEQ_LEN, chunksize=5000):
    sequences_raw_indices = []
    metadata = []
    labels = []
    chunk_num = 0
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        df_raw = chunk.reset_index(drop=True)
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'datetime_ist']
        for c in required:
            if c not in df_raw.columns:
                raise ValueError(f"Missing column {c} in chunk {chunk_num}")
                
        ensure_volatility(df_raw)
        calculate_rsi(df_raw)
        calculate_macd(df_raw)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'volatility', 'rsi', 'macd']
        df_raw[numeric_cols] = df_raw[numeric_cols].ffill().fillna(0).astype(float)
        df_raw['session'], df_raw['is_sunday'] = zip(*df_raw['timestamp'].apply(detect_session_from_timestamp))

        # Outlier removal
        df_raw = df_raw[(df_raw['high'] - df_raw['low'] <= 0.05 * df_raw['close'])]

        N = len(df_raw)
        num_seq = max(0, N - seq_len - config.MAX_LOOKAHEAD)
        logger.logger.info(f"Chunk {chunk_num} rows: {N}, sequences: {num_seq}")

        for i in range(0, num_seq):
            idx_entry = i + seq_len - 1
            current_close = df_raw['close'].iat[idx_entry]
            current_low = df_raw['low'].iat[idx_entry]
            current_high = df_raw['high'].iat[idx_entry]
            current_volume = df_raw['volume'].iat[idx_entry]
            current_trades = int(df_raw['trades'].iat[idx_entry])
            current_vol = df_raw['volatility'].iat[idx_entry]
            current_rsi = df_raw['rsi'].iat[idx_entry]
            current_macd = df_raw['macd'].iat[idx_entry]
            current_session = df_raw['session'].iat[idx_entry]
            is_sunday = bool(df_raw['is_sunday'].iat[idx_entry])

            # Dynamic move_threshold with volatility (increased to 2.0)
            vol20 = df_raw['volatility'].rolling(20).mean().iloc[idx_entry]
            base_threshold = 300 if current_session == 'Asia' else 400 if current_session == 'London' else 500
            move_threshold = 2.0 * vol20 if not np.isnan(vol20) and vol20 > 0 else base_threshold
            if current_vol > vol20.mean() + vol20.std():
                move_threshold *= 1.2  # Boost for high volatility

            if is_sunday:
                move_threshold *= 0.8

            if current_session == 'NY':
                volume_factor = 1.8; volatility_factor = 1.5
            elif current_session == 'Asia':
                volume_factor = 1.5; volatility_factor = 1.2
            else:
                volume_factor = 1.6; volatility_factor = 1.4

            # Dynamic lookahead detection
            move_type = 'neutral'
            move_size = 0.0
            min_low = float('inf')
            max_high = -float('inf')
            actual_lookahead = 0
            price_stabilized = False
            
            for j in range(idx_entry + 1, min(idx_entry + 1 + config.MAX_LOOKAHEAD, N)):
                actual_lookahead = j - idx_entry
                fc = df_raw['close'].iat[j]
                min_low = min(min_low, df_raw['low'].iat[j])
                max_high = max(max_high, df_raw['high'].iat[j])
                move_size = fc - current_close
                
                # Trap detection: Reversal > 50% within 5 candles
                if actual_lookahead > 5 and j >= 5 and abs(move_size) > move_threshold * 0.5 and abs(fc - df_raw['close'].iat[j-5]) < move_threshold * 0.5:
                    move_type = 'neutral'
                    break
                
                if (move_size >= move_threshold) and (min_low >= current_low + 100):
                    move_type = 'buy'
                    break
                if (move_size <= -move_threshold) and (max_high <= current_high - 100):
                    move_type = 'sell'
                    break
                    
                price_stabilized, stable_count = check_price_stability(df_raw, j, fc)
                if price_stabilized:
                    break

            if move_type == 'neutral' and price_stabilized:
                if move_size >= move_threshold * 0.6:
                    move_type = 'buy'
                elif move_size <= -move_threshold * 0.6:
                    move_type = 'sell'

            # Spikes and indicators
            start_avg = max(0, idx_entry - 19)
            avg_volume = df_raw['volume'].iloc[start_avg:idx_entry+1].mean() if idx_entry >= start_avg else current_volume
            avg_trades = df_raw['trades'].iloc[start_avg:idx_entry+1].mean() if idx_entry >= start_avg else current_trades
            vol20 = df_raw['volatility'].iloc[start_avg:idx_entry+1].mean() if idx_entry >= start_avg else current_vol
            volume_spike = (current_volume > volume_factor * avg_volume) if avg_volume > 0 else False
            trade_spike = (current_trades > volume_factor * avg_trades) if avg_trades > 0 else False
            high_volatility = (current_vol > volatility_factor * vol20) if vol20 > 0 else False
            taker_pressure = df_raw['taker_buy_base'].iat[idx_entry] / (current_volume + 1e-10)
            rsi_overbought = current_rsi > 70
            rsi_oversold = current_rsi < 30
            macd_positive = current_macd > 0

            # Compute label with RSI/MACD
            if (move_type == 'buy' or 
                ((volume_spike or trade_spike) and (taker_pressure > 0.55) and high_volatility and not rsi_overbought) or
                (move_size >= move_threshold * 0.5 and taker_pressure > 0.6 and macd_positive) or
                (volume_spike and current_vol > vol20 * 1.8 and not rsi_overbought)):
                label = 0
            elif (move_type == 'sell' or 
                  ((volume_spike or trade_spike) and (taker_pressure < 0.45) and high_volatility and not rsi_oversold) or
                  (move_size <= -move_threshold * 0.5 and taker_pressure < 0.4 and not macd_positive) or
                  (volume_spike and current_vol > vol20 * 1.8 and not rsi_oversold)):
                label = 1
            else:
                label = 2

            # Compute SL/TP1/TP2 for metadata
            sl = None; tp1 = None; tp2 = None
            if label in [0, 1]:
                vol = current_vol
                sl = current_close - 2 * vol * 100 if label == 0 else current_close + 2 * vol * 100
                found_tp1_idx = None
                for j in range(idx_entry + 1, min(idx_entry + 1 + config.MAX_LOOKAHEAD, N)):
                    avg_close_raw, is_stable = calculate_stability_raw(df_raw, j)
                    if is_stable:
                        tp1 = avg_close_raw; found_tp1_idx = j; break
                if tp1 is not None and found_tp1_idx is not None:
                    for j in range(found_tp1_idx + 1, min(found_tp1_idx + 1 + 30, N)):
                        avg_close_raw, is_stable = calculate_stability_raw(df_raw, j)
                        if is_stable and ((label == 0 and avg_close_raw > tp1) or (label == 1 and avg_close_raw < tp1)):
                            tp2 = avg_close_raw; break

            sequences_raw_indices.append(i + chunk_num * chunksize)  # Adjust for chunk offset
            metadata.append({
                'timestamp': int(df_raw['timestamp'].iat[idx_entry]),
                'datetime_ist': str(df_raw['datetime_ist'].iat[idx_entry]),
                'session': current_session,
                'move_size': abs(move_size),
                'actual_lookahead': actual_lookahead,
                'price_stabilized': price_stabilized,
                'trigger_method': 'momentum' if move_type != 'neutral' else ('volume' if (volume_spike or trade_spike) and high_volatility else 'no_move'),
                'sl': sl, 'tp1': tp1, 'tp2': tp2,
                'entry_row_idx': int(idx_entry + chunk_num * chunksize)  # Adjust for chunk
            })
            labels.append(int(label))
        chunk_num += 1
        gc.collect()  # Clear memory after each chunk

    labels = np.array(labels, dtype=np.int64)
    logger.logger.info(f"Built sequences indices: {len(sequences_raw_indices)} labels distribution: {np.unique(labels, return_counts=True)}")
    return df_raw, sequences_raw_indices, metadata, labels

def build_norm_sequences_and_labels(df_raw: pd.DataFrame, seq_len: int, sequences_raw_indices: List[int],
                                    metadata: List[dict], labels: np.ndarray, scaler: RobustScaler):
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'volatility', 'rsi', 'macd']
    features_all = scaler.transform(df_raw[feature_cols].values)
    num_seq = len(sequences_raw_indices)
    X = np.zeros((num_seq, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    for idx_i, start in enumerate(sequences_raw_indices):
        seq_norm = features_all[start:start+seq_len]
        X[idx_i] = seq_norm
    return X, y, metadata

# Section 8: Enhanced Embeddings
def build_sequence_embeddings(X: np.ndarray, train_cutoff_seq: int, embed_dim: int = config.EMBED_DIM):
    num_seq = X.shape[0]
    flat = X.reshape(num_seq, -1)
    if num_seq >= config.MIN_IPCA_BATCH:
        logger.logger.info("Using IncrementalPCA for embeddings")
        ipca = IncrementalPCA(n_components=embed_dim, batch_size=config.MIN_IPCA_BATCH)
        ipca.partial_fit(flat[:train_cutoff_seq])
        embeddings = ipca.transform(flat)
        pca_obj = ipca
    else:
        logger.logger.info("Using PCA for embeddings")
        pca = PCA(n_components=embed_dim, random_state=config.SEED)
        pca.fit(flat[:train_cutoff_seq])
        embeddings = pca.transform(flat)
        pca_obj = pca
    gc.collect()
    return embeddings, pca_obj

def embeddings_transform(data: np.ndarray, pca_obj: Union[PCA, IncrementalPCA]) -> np.ndarray:
    return pca_obj.transform(data)

# Section 9: Enhanced Training
def prepare_dataloaders_chrono(X: np.ndarray, y: np.ndarray, batch_size: int = config.BATCH_SIZE,
                               train_frac: float = config.TRAIN_SPLIT_FRAC, val_frac: float = config.VAL_SPLIT_FRAC):
    num = X.shape[0]
    train_end = int(num * train_frac)
    val_end = int(num * (train_frac + val_frac))
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, (train_end, val_end, num)

def calculate_sharpe_ratio(returns):
    if len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    return mean_return / std_dev if std_dev > 0 else 0.0

def objective(trial):
    config.LR = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config.BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    config.HIDDEN_SIZE = trial.suggest_categorical("hidden_size", [64, 128, 256])

    model = ProductionPriceFlowLSTM(input_size=12, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, output_size=3).to(device)
    trained_model = train_loop(model, train_loader, val_loader, X_train_norm, y_train, X_norm_all, labels_all, embeddings_obj, nn_idx, device, epochs=config.EPOCHS, lr=config.LR)
    
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            out = trained_model(Xb)
            test_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            test_labels.extend(yb.numpy())
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    return test_f1

def train_loop(model, train_loader, val_loader, X_train_norm, y_train, X_norm_all, labels_all, embeddings_obj,
               nn_idx, device, epochs=config.EPOCHS, lr=config.LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    y_train_all = np.concatenate([y_batch.numpy() for _, y_batch in train_loader])
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_all), y=y_train_all)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_hybrid_f1 = 0.0
    no_improve = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        model.eval()
        val_labels = []
        val_lstm_preds = []
        val_hybrid_preds = []
        val_probs = []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb_cpu = Xb.numpy()
                Xb = Xb.to(device)
                out = model(Xb)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                lstm_pred = np.argmax(probs, axis=1)
                val_lstm_preds.extend(lstm_pred.tolist())
                val_labels.extend(yb.numpy().tolist())
                val_probs.append(probs)

        lstm_features = np.vstack(val_probs)
        xgb = XGBClassifier(random_state=config.SEED)
        xgb.fit(lstm_features, val_labels)
        hybrid_preds = xgb.predict(lstm_features)

        val_lstm_acc = accuracy_score(val_labels, val_lstm_preds)
        val_lstm_f1 = f1_score(val_labels, val_lstm_preds, average='weighted')
        val_hybrid_acc = accuracy_score(val_labels, hybrid_preds)
        val_hybrid_f1 = f1_score(val_labels, hybrid_preds, average='weighted')
        
        scheduler.step(val_hybrid_f1)

        logger.logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val LSTM Acc: {val_lstm_acc:.4f}, Val LSTM F1: {val_lstm_f1:.4f}, Val Hybrid Acc: {val_hybrid_acc:.4f}, Val Hybrid F1: {val_hybrid_f1:.4f}")

        if val_hybrid_f1 > best_val_hybrid_f1:
            best_val_hybrid_f1 = val_hybrid_f1
            torch.save(model.state_dict(), os.path.join(config.ARTIFACTS_DIR, "module2_model.pth"))
            joblib.dump(xgb, os.path.join(config.ARTIFACTS_DIR, "module2_xgb.save"))
            logger.logger.info(f"Saved best model (hybrid val F1={best_val_hybrid_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOP_PATIENCE:
                logger.logger.info(f"Early stopping at epoch {epoch}")
                break
        gc.collect()

    logger.logger.info(f"Training completed. Best hybrid val F1: {best_val_hybrid_f1:.4f}")
    return model

# Section 10: Main Training Function
def main():
    global train_loader, val_loader, test_loader, X_train_norm, y_train, X_norm_all, labels_all, embeddings_obj, nn_idx, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    logger.logger.info(f"Using device: {device}")

    try:
        # Load full data for scaler (training portion only)
        df_raw_full = pd.read_csv(config.DATA_PATH)
        train_row_cutoff = int(len(df_raw_full) * config.TRAIN_SPLIT_FRAC)
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'volatility', 'rsi', 'macd']
        scaler = RobustScaler()
        scaler.fit(df_raw_full[feature_cols].iloc[:train_row_cutoff].values)
        joblib.dump(scaler, os.path.join(config.ARTIFACTS_DIR, "module2_scaler.save"))
        logger.logger.info("Fitted scaler on train rows only")
        gc.collect()

        # Build sequences in chunks
        _, seq_indices, metadata, labels = load_and_build_sequences(config.DATA_PATH, seq_len=config.SEQ_LEN)
        N = len(df_raw_full)
        if len(seq_indices) == 0:
            raise RuntimeError("Not enough sequences. Increase data or reduce SEQ_LEN/MAX_LOOKAHEAD.")

        # Build norm with full df_raw_full
        X_norm, y, metadata = build_norm_sequences_and_labels(df_raw_full, config.SEQ_LEN, seq_indices, metadata, labels, scaler)
        
        X_balanced, y_balanced, metadata_balanced = balance_classes(X_norm, y, metadata)
        
        noise = np.random.normal(0, 0.01, X_balanced.shape)
        X_aug = X_balanced + noise
        X_balanced = np.concatenate([X_balanced, X_aug])
        y_balanced = np.concatenate([y_balanced, y_balanced])
        metadata_balanced.extend(metadata_balanced)
        
        num_seq = X_balanced.shape[0]
        train_seq_cutoff = int(num_seq * config.TRAIN_SPLIT_FRAC)
        logger.logger.info(f"Balanced and augmented sequences shape: {X_balanced.shape}")

        embeddings_arr, pca_obj = build_sequence_embeddings(X_balanced, train_seq_cutoff, embed_dim=config.EMBED_DIM)
        embeddings_obj = {'embeddings': embeddings_arr, 'pca_obj': pca_obj}
        joblib.dump(pca_obj, os.path.join(config.ARTIFACTS_DIR, "module2_pca.save"))
        np.save(os.path.join(config.ARTIFACTS_DIR, "historical_embeddings.npy"), embeddings_arr)
        np.save(os.path.join(config.ARTIFACTS_DIR, "historical_sequences_norm.npy"), X_balanced)
        np.save(os.path.join(config.ARTIFACTS_DIR, "historical_labels.npy"), y_balanced)
        logger.logger.info("Saved embeddings, sequences, labels")

        nn_idx = NearestNeighbors(n_neighbors=min(config.ANN_CANDIDATES, embeddings_arr.shape[0]), algorithm='auto', metric='euclidean')
        nn_idx.fit(embeddings_arr)
        joblib.dump(nn_idx, os.path.join(config.ARTIFACTS_DIR, "module2_nn_idx.save"))
        logger.logger.info("Fitted and saved NearestNeighbors index")

        train_loader, val_loader, test_loader, (train_end, val_end, total_seq) = prepare_dataloaders_chrono(X_balanced, y_balanced)
        X_train_norm = X_balanced[:train_end]
        y_train = y_balanced[:train_end]
        X_norm_all = X_balanced
        labels_all = y_balanced
        logger.logger.info(f"Prepared dataloaders: train_end_seq={train_end}, val_end_seq={val_end}")

        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=config.HYPER_OPT_TRIALS)
            config.LR = study.best_params['lr']
            config.BATCH_SIZE = study.best_params['batch_size']
            config.HIDDEN_SIZE = study.best_params['hidden_size']
        except Exception as e:
            logger.logger.error(f"Optuna optimization failed: {e}")
            config.LR = 1e-3  # Fallback
            config.BATCH_SIZE = 128
            config.HIDDEN_SIZE = 192

        model = ProductionPriceFlowLSTM(input_size=X_balanced.shape[2], hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, output_size=3).to(device)
        trained_model = train_loop(model, train_loader, val_loader, X_train_norm, y_train,
                                 X_norm_all, labels_all, embeddings_obj, nn_idx, device, epochs=config.EPOCHS, lr=config.LR)

        # Final evaluation on test set and log what model learned
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                out = trained_model(Xb)
                test_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                test_labels.extend(yb.numpy())
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='weighted')
        logger.logger.info(f"Model learned: Test Accuracy: {test_acc * 100:.2f}%, Test F1 Score: {test_f1 * 100:.2f}%")

        logger.logger.info("Training completed successfully")
        
    except Exception as e:
        logger.logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()