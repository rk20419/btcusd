"""
PRODUCTION-READY Price Flow Analyzer Module
- Real-time optimized with fail-safes
- Enhanced data validation
- Performance monitoring
- Comprehensive logging
- Graceful error handling
- Updated thresholds for BTCUSDT volatility
"""

import os
import logging
import math
import random
import time
import json
from typing import Tuple, List, Any, Optional, Dict, Union
from collections import Counter
from dataclasses import dataclass
import threading

import numpy as np
import pandas as pd
import joblib
import pytz
from datetime import datetime, timedelta

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --------------------------
# Production Configuration
# --------------------------
@dataclass
class Config:
    SEED = 42
    SEQ_LEN = 60  # Reduced for real-time performance
    MAX_LOOKAHEAD = 50
    MIN_STABLE_CANDLES = 3
    MAX_STABLE_CANDLES = 5
    PRICE_RANGE = 300  # Increased from 100 to 300 for BTCUSDT volatility
    EPOCHS = 50
    BATCH_SIZE = 64  # Increased for efficiency
    LR = 1e-3
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    EMBED_DIM = 32
    ANN_CANDIDATES = 150  # Reduced for speed
    TOP_M_FOR_DTW = 30  # Reduced for speed
    DTW_TOPK = 5
    TRAIN_SPLIT_FRAC = 0.80
    VAL_SPLIT_FRAC = 0.10
    ARTIFACTS_DIR = "models/production"
    LOG_DIR = "logs/production"
    DATA_PATH = "data/historical/BTCUSDT_1m_5000.csv"
    MIN_IPCA_BATCH = 2000
    EARLY_STOP_PATIENCE = 7
    MAX_SAMPLES_PER_CLASS = 8000
    REAL_TIME_UPDATE_INTERVAL = 30  # seconds
    MAX_RETRIES = 3
    REALTIME_SEQ_LEN = 60  # Optimized for real-time
    
    # Real-time thresholds
    CONFIDENCE_THRESHOLD = 0.65  # Higher threshold for production
    MIN_DATA_QUALITY_SCORE = 0.8
    MAX_LATENCY_MS = 500
    
    # Monitoring
    PERFORMANCE_WINDOW = 100  # predictions to track
    ALERT_THRESHOLD = 5  # consecutive errors before alert

# Initialize config
config = Config()
os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs("data/real_time", exist_ok=True)

# --------------------------
# Enhanced Logging
# --------------------------
class ProductionLogger:
    def __init__(self):
        self.logger = logging.getLogger("price_flow_production")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(config.LOG_DIR, "production.log"))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Performance metrics
        self.performance_metrics = {
            'prediction_times': [],
            'confidence_scores': [],
            'recent_errors': 0,
            'last_alert_time': None
        }
    
    def log_performance(self, prediction_time: float, confidence: float):
        self.performance_metrics['prediction_times'].append(prediction_time)
        self.performance_metrics['confidence_scores'].append(confidence)
        
        # Keep only recent window
        if len(self.performance_metrics['prediction_times']) > config.PERFORMANCE_WINDOW:
            self.performance_metrics['prediction_times'].pop(0)
            self.performance_metrics['confidence_scores'].pop(0)
    
    def check_health(self):
        avg_time = np.mean(self.performance_metrics['prediction_times']) if self.performance_metrics['prediction_times'] else 0
        avg_confidence = np.mean(self.performance_metrics['confidence_scores']) if self.performance_metrics['confidence_scores'] else 0
        
        if avg_time > config.MAX_LATENCY_MS / 1000:
            self.logger.warning(f"High latency detected: {avg_time:.3f}s")
            return False
        
        if avg_confidence < config.CONFIDENCE_THRESHOLD - 0.1:
            self.logger.warning(f"Low confidence detected: {avg_confidence:.3f}")
            return False
            
        return True

logger = ProductionLogger()

# --------------------------
# Enhanced Model
# --------------------------
class ProductionPriceFlowLSTM(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS, output_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0 if num_layers==1 else 0.1)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.dropout(context)
        return self.fc(out)

# --------------------------
# Data Quality System
# --------------------------
class DataQualityValidator:
    @staticmethod
    def validate_real_time_data(df: pd.DataFrame) -> Tuple[bool, float]:
        """Validate real-time data quality"""
        if df.empty:
            return False, 0.0
        
        quality_score = 1.0
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        quality_score -= missing_ratio * 0.3
        
        # Check for zeros in critical columns
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        zero_ratio = (df[critical_cols] == 0).sum().sum() / (len(critical_cols) * df.shape[0])
        quality_score -= zero_ratio * 0.4
        
        # Check timestamp continuity
        if 'timestamp' in df.columns:
            time_diffs = df['timestamp'].diff().dropna()
            if not time_diffs.empty:
                abnormal_gaps = (time_diffs > 120000).sum()  # >2 minutes
                gap_penalty = abnormal_gaps / len(time_diffs) * 0.3
                quality_score -= gap_penalty
        
        return quality_score >= config.MIN_DATA_QUALITY_SCORE, quality_score

# --------------------------
# Enhanced Utilities
# --------------------------
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
    if 'volatility' not in df.columns:
        df['returns'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['returns'].rolling(10).std().fillna(0) * 100
        df.drop(columns=['returns'], inplace=True)

def calculate_stability_raw(df_raw: pd.DataFrame, start_idx: int, window: int = 3) -> Tuple[Optional[float], bool]:
    if start_idx + window > len(df_raw):
        return None, False
    subset = df_raw.iloc[start_idx:start_idx+window]
    if start_idx < 19:
        vol20 = df_raw['volatility'].iloc[:start_idx+1].mean() if start_idx >= 0 else df_raw['volatility'].iloc[0]
    else:
        vol20 = df_raw['volatility'].rolling(20).mean().iloc[start_idx]
    if np.isnan(vol20) or vol20 <= 0:
        return None, False
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
    weights = []
    label_scores = {}
    for d, lab in top:
        w = 1.0 / (d + 1e-8)
        label_scores[lab] = label_scores.get(lab, 0.0) + w
        weights.append(w)
    majority_label = max(label_scores, key=label_scores.get)
    conf = label_scores[majority_label] / sum(label_scores.values())
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
    count = Counter(y)
    logger.logger.info(f"Original class distribution: {count}")
    
    min_samples = min(max_samples_per_class, min(count.values()))
    
    balanced_X, balanced_y, balanced_meta = [], [], []
    
    for class_id in [0, 1, 2]:
        indices = np.where(y == class_id)[0]
        if len(indices) > min_samples:
            indices = np.random.choice(indices, min_samples, replace=False)
        
        balanced_X.append(X[indices])
        balanced_y.append(y[indices])
        balanced_meta.extend([metadata[i] for i in indices])
    
    balanced_X = np.vstack(balanced_X)
    balanced_y = np.concatenate(balanced_y)
    
    logger.logger.info(f"Balanced class distribution: {Counter(balanced_y)}")
    return balanced_X, balanced_y, balanced_meta

# --------------------------
# Enhanced Sequence Building
# --------------------------
def load_and_build_sequences(filepath: str, seq_len: int = config.SEQ_LEN):
    try:
        df_raw = pd.read_csv(filepath).reset_index(drop=True)
        required = ['timestamp','open','high','low','close','volume','quote_volume','trades','taker_buy_base','taker_buy_quote','datetime_ist']
        
        for c in required:
            if c not in df_raw.columns:
                raise ValueError(f"Missing column {c} in {filepath}")
                
        ensure_volatility(df_raw)
        numeric_cols = ['open','high','low','close','volume','quote_volume','trades','taker_buy_base','taker_buy_quote','volatility']
        df_raw[numeric_cols] = df_raw[numeric_cols].ffill().fillna(0).astype(float)
        df_raw['session'], df_raw['is_sunday'] = zip(*df_raw['timestamp'].apply(detect_session_from_timestamp))

        N = len(df_raw)
        num_seq = max(0, N - seq_len - config.MAX_LOOKAHEAD)
        logger.logger.info(f"Rows: {N}, sequences: {num_seq}")

        sequences_raw_indices = []
        metadata = []
        labels = []

        for i in range(0, num_seq):
            idx_entry = i + seq_len - 1
            current_close = df_raw['close'].iat[idx_entry]
            current_low = df_raw['low'].iat[idx_entry]
            current_high = df_raw['high'].iat[idx_entry]
            current_volume = df_raw['volume'].iat[idx_entry]
            current_trades = int(df_raw['trades'].iat[idx_entry])
            current_vol = df_raw['volatility'].iat[idx_entry]
            current_session = df_raw['session'].iat[idx_entry]
            is_sunday = bool(df_raw['is_sunday'].iat[idx_entry])

            # Session thresholds (increased for BTCUSDT volatility)
            if is_sunday:
                move_threshold = 350  # $350 for Sundays
            else:
                move_threshold = 500 if current_session == 'NY' else 300 if current_session == 'Asia' else 400  # NY: $500, Asia: $300, London: $400
            # Optional: Dynamic threshold based on ATR
            # move_threshold = 1.5 * df_raw['close'].rolling(20).std().iloc[idx_entry] * 100

            if current_session == 'NY':
                volume_factor = 1.8; volatility_factor = 1.5
            elif current_session == 'Asia':
                volume_factor = 1.5; volatility_factor = 1.2
            else:
                volume_factor = 1.6; volatility_factor = 1.4
            if is_sunday:
                volume_factor *= 0.8

            # Dynamic lookahead detection
            move_type = 'neutral'
            move_size = 0.0
            min_low = float('inf')
            max_high = -float('inf')
            actual_lookahead = 0
            price_stabilized = False
            
            for j in range(idx_entry+1, min(idx_entry+1+config.MAX_LOOKAHEAD, N)):
                actual_lookahead = j - idx_entry
                fc = df_raw['close'].iat[j]
                min_low = min(min_low, df_raw['low'].iat[j])
                max_high = max(max_high, df_raw['high'].iat[j])
                move_size = fc - current_close
                
                if (move_size >= move_threshold) and (min_low >= current_low + 150):  # Adjusted to +150
                    move_type = 'buy'
                    break
                if (move_size <= -move_threshold) and (max_high <= current_high - 150):  # Adjusted to -150
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

            # spikes
            start_avg = max(0, idx_entry - 19)
            avg_volume = df_raw['volume'].iloc[start_avg:idx_entry+1].mean() if idx_entry >= start_avg else current_volume
            avg_trades = df_raw['trades'].iloc[start_avg:idx_entry+1].mean() if idx_entry >= start_avg else current_trades
            volume_spike = (current_volume > volume_factor * avg_volume) if avg_volume > 0 else False
            trade_spike = (current_trades > volume_factor * avg_trades) if avg_trades > 0 else False
            vol20 = df_raw['volatility'].iloc[start_avg:idx_entry+1].mean() if idx_entry >= start_avg else current_vol
            high_volatility = (current_vol > volatility_factor * vol20) if vol20 > 0 else False
            taker_pressure = df_raw['taker_buy_base'].iat[idx_entry] / (current_volume + 1e-10)

            # compute label
            if (move_type == 'buy' or 
                ((volume_spike or trade_spike) and (taker_pressure > 0.55) and high_volatility) or
                (move_size >= move_threshold * 0.5 and taker_pressure > 0.6) or
                (volume_spike and current_vol > vol20 * 1.8)):
                label = 0
            elif (move_type == 'sell' or 
                  ((volume_spike or trade_spike) and (taker_pressure < 0.45) and high_volatility) or
                  (move_size <= -move_threshold * 0.5 and taker_pressure < 0.4) or
                  (volume_spike and current_vol > vol20 * 1.8 and taker_pressure < 0.45)):
                label = 1
            else:
                label = 2

            # compute sl/tp1/tp2 for metadata
            sl = None; tp1 = None; tp2 = None
            if label in [0,1]:
                vol = current_vol
                sl = current_close - 2 * vol * 100 if label == 0 else current_close + 2 * vol * 100
                found_tp1_idx = None
                
                for j in range(idx_entry+1, min(idx_entry+1+config.MAX_LOOKAHEAD, N)):
                    avg_close_raw, is_stable = calculate_stability_raw(df_raw, j)
                    if is_stable:
                        tp1 = avg_close_raw; found_tp1_idx = j; break
                        
                if tp1 is not None and found_tp1_idx is not None:
                    for j in range(found_tp1_idx+1, min(found_tp1_idx+1+30, N)):
                        avg_close_raw, is_stable = calculate_stability_raw(df_raw, j)
                        if is_stable and ((label==0 and avg_close_raw>tp1) or (label==1 and avg_close_raw<tp1)):
                            tp2 = avg_close_raw; break

            sequences_raw_indices.append(i)
            metadata.append({
                'timestamp': int(df_raw['timestamp'].iat[idx_entry]),
                'datetime_ist': str(df_raw['datetime_ist'].iat[idx_entry]),
                'session': current_session,
                'move_size': abs(move_size),
                'actual_lookahead': actual_lookahead,
                'price_stabilized': price_stabilized,
                'trigger_method': 'momentum' if move_type != 'neutral' else ('volume' if (volume_spike or trade_spike) and high_volatility else 'no_move'),
                'sl': sl, 'tp1': tp1, 'tp2': tp2,
                'entry_row_idx': int(idx_entry)
            })
            labels.append(int(label))

        labels = np.array(labels, dtype=np.int64)
        logger.logger.info(f"Built sequences indices: {len(sequences_raw_indices)} labels distribution: {np.unique(labels, return_counts=True)}")
        return df_raw, sequences_raw_indices, metadata, labels
        
    except Exception as e:
        logger.logger.error(f"Error building sequences: {e}")
        raise

def build_norm_sequences_and_labels(df_raw: pd.DataFrame, seq_len: int, sequences_raw_indices: List[int],
                                    metadata: List[dict], labels: np.ndarray, scaler: RobustScaler):
    feature_cols = ['open','high','low','close','volume','quote_volume','trades','taker_buy_base','taker_buy_quote','volatility']
    features_all = scaler.transform(df_raw[feature_cols].values)
    num_seq = len(sequences_raw_indices)
    X = np.zeros((num_seq, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    for idx_i, start in enumerate(sequences_raw_indices):
        seq_norm = features_all[start:start+seq_len]
        X[idx_i] = seq_norm
    return X, y, metadata

# --------------------------
# Enhanced Embeddings
# --------------------------
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
    return embeddings, pca_obj

def embeddings_transform(data: np.ndarray, pca_obj: Union[PCA, IncrementalPCA]) -> np.ndarray:
    return pca_obj.transform(data)

# --------------------------
# Enhanced Training
# --------------------------
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
            all_preds.extend(preds); all_labels.extend(y_batch.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        # Validation
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
                val_probs.append((Xb_cpu, probs))

        hybrid_preds = []
        for Xb_cpu, probs in val_probs:
            dtw_confs = []
            batch_size_local = Xb_cpu.shape[0]
            flat = Xb_cpu.reshape(batch_size_local, -1)
            emb_batch = embeddings_transform(flat, embeddings_obj['pca_obj'])
            n_neighbors = min(config.ANN_CANDIDATES, embeddings_obj['embeddings'].shape[0])
            _, idxs = nn_idx.kneighbors(emb_batch, n_neighbors=n_neighbors)
            for i in range(batch_size_local):
                candidate_idxs = idxs[i][:config.TOP_M_FOR_DTW].tolist()
                dtw_conf = compute_dtw_confidence_on_candidates(Xb_cpu[i], candidate_idxs, X_norm_all, labels_all, topk=config.DTW_TOPK)
                dtw_confs.append(dtw_conf)
            dtw_confs = np.array(dtw_confs)
            for j in range(len(probs)):
                lstm_probs = probs[j]
                lstm_pred = int(np.argmax(lstm_probs))
                lstm_conf = float(lstm_probs[lstm_pred])
                dtw_conf = float(dtw_confs[j])
                combined_conf = 0.6 * lstm_conf + 0.4 * dtw_conf
                if combined_conf < config.CONFIDENCE_THRESHOLD:
                    hybrid_preds.append(2)
                else:
                    hybrid_preds.append(lstm_pred)

        val_lstm_acc = accuracy_score(val_labels, val_lstm_preds)
        val_lstm_f1 = f1_score(val_labels, val_lstm_preds, average='weighted')
        val_hybrid_acc = accuracy_score(val_labels, hybrid_preds)
        val_hybrid_f1 = f1_score(val_labels, hybrid_preds, average='weighted')
        
        scheduler.step(val_hybrid_f1)

        logger.logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val LSTM Acc: {val_lstm_acc:.4f}, Val LSTM F1: {val_lstm_f1:.4f}, Val Hybrid Acc: {val_hybrid_acc:.4f}, Val Hybrid F1: {val_hybrid_f1:.4f}")

        if val_hybrid_f1 > best_val_hybrid_f1:
            best_val_hybrid_f1 = val_hybrid_f1
            torch.save(model.state_dict(), os.path.join(config.ARTIFACTS_DIR, "module2_model.pth"))
            logger.logger.info(f"Saved best model (hybrid val F1={best_val_hybrid_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOP_PATIENCE:
                logger.logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.logger.info(f"Training completed. Best hybrid val F1: {best_val_hybrid_f1:.4f}")
    return model

# --------------------------
# Production Predictor
# --------------------------
class ProductionPriceFlowPredictor:
    def __init__(self, artifacts_dir: str = config.ARTIFACTS_DIR, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.artifacts_loaded = False
        self.load_artifacts(artifacts_dir)
        self.cache = {}
        self.last_prediction_time = 0
        self.prediction_lock = threading.Lock()
        
    def load_artifacts(self, artifacts_dir: str):
        """Load model artifacts with retry logic"""
        for attempt in range(config.MAX_RETRIES):
            try:
                self.scaler = joblib.load(os.path.join(artifacts_dir, "module2_scaler.save"))
                self.pca = joblib.load(os.path.join(artifacts_dir, "module2_pca.save"))
                self.nn_idx = joblib.load(os.path.join(artifacts_dir, "module2_nn_idx.save"))
                self.X_norm_all = np.load(os.path.join(artifacts_dir, "historical_sequences_norm.npy"))
                self.labels_all = np.load(os.path.join(artifacts_dir, "historical_labels.npy"))
                
                self.model = ProductionPriceFlowLSTM(
                    input_size=self.X_norm_all.shape[2], 
                    hidden_size=config.HIDDEN_SIZE, 
                    num_layers=config.NUM_LAYERS, 
                    output_size=3
                ).to(self.device)
                
                self.model.load_state_dict(torch.load(
                    os.path.join(artifacts_dir, "module2_model.pth"), 
                    map_location=self.device
                ))
                self.model.eval()
                
                self.artifacts_loaded = True
                logger.logger.info("PriceFlowPredictor initialized successfully")
                return
                
            except Exception as e:
                logger.logger.error(f"Attempt {attempt + 1} failed to load artifacts: {e}")
                if attempt == config.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def prepare_latest_sequence(self, df: pd.DataFrame) -> Union[np.ndarray, None]:
        """Prepare the latest sequence with data validation"""
        if len(df) < config.REALTIME_SEQ_LEN:
            logger.logger.warning(f"Not enough data. Need {config.REALTIME_SEQ_LEN} rows, got {len(df)}")
            return None
            
        # Validate data quality
        is_valid, quality_score = DataQualityValidator.validate_real_time_data(df)
        if not is_valid:
            logger.logger.warning(f"Poor data quality: {quality_score:.3f}")
            return None
            
        ensure_volatility(df)
        
        latest_data = df.iloc[-config.REALTIME_SEQ_LEN:].copy()
        
        feature_cols = ['open','high','low','close','volume','quote_volume','trades',
                       'taker_buy_base','taker_buy_quote','volatility']
        
        try:
            latest_norm = self.scaler.transform(latest_data[feature_cols].values)
            return latest_norm.reshape(1, config.REALTIME_SEQ_LEN, -1)
        except Exception as e:
            logger.logger.error(f"Error normalizing data: {e}")
            return None
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make a prediction with thread safety and performance monitoring"""
        with self.prediction_lock:
            start_time = time.time()
            
            if not self.artifacts_loaded:
                return {"prediction": "NEUTRAL", "confidence": 0.0, "error": "Artifacts not loaded"}
            
            if time.time() - self.last_prediction_time < 1.0:
                return {"prediction": "NEUTRAL", "confidence": 0.0, "error": "Rate limited"}
            
            try:
                # Prepare sequence
                sequence = self.prepare_latest_sequence(df)
                if sequence is None:
                    return {"prediction": "NEUTRAL", "confidence": 0.0, "error": "Invalid sequence"}
                
                # Check cache
                cache_key = hash(sequence.tobytes())
                if cache_key in self.cache and time.time() - self.cache[cache_key]['timestamp'] < 30:
                    return self.cache[cache_key]['result']
                
                # LSTM prediction
                with torch.no_grad():
                    tensor_seq = torch.tensor(sequence, dtype=torch.float32).to(self.device)
                    output = self.model(tensor_seq)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    lstm_pred = int(np.argmax(probs))
                    lstm_conf = float(probs[lstm_pred])
                
                # DTW confidence
                flat_seq = sequence.reshape(1, -1)
                emb = embeddings_transform(flat_seq, self.pca)
                n_neighbors = min(config.ANN_CANDIDATES, self.X_norm_all.shape[0])
                _, idxs = self.nn_idx.kneighbors(emb, n_neighbors=n_neighbors)
                candidate_idxs = idxs[0][:config.TOP_M_FOR_DTW].tolist()
                dtw_conf = compute_dtw_confidence_on_candidates(
                    sequence[0], candidate_idxs, self.X_norm_all, self.labels_all, topk=config.DTW_TOPK
                )
                
                # Combine confidences
                combined_conf = 0.6 * lstm_conf + 0.4 * dtw_conf
                
                # Final prediction
                if combined_conf < config.CONFIDENCE_THRESHOLD:
                    final_pred = 2
                    final_conf = 1.0 - combined_conf
                else:
                    final_pred = lstm_pred
                    final_conf = combined_conf
                
                pred_labels = {0: "BUY", 1: "SELL", 2: "NEUTRAL"}
                
                result = {
                    "prediction": pred_labels[final_pred],
                    "confidence": float(final_conf),
                    "lstm_prediction": pred_labels[lstm_pred],
                    "lstm_confidence": float(lstm_conf),
                    "dtw_confidence": float(dtw_conf),
                    "combined_confidence": float(combined_conf),
                    "timestamp": time.time(),
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
                
                # Update cache
                self.cache[cache_key] = {
                    'timestamp': time.time(),
                    'result': result
                }
                
                # Clean old cache entries
                self.clean_cache()
                
                # Log performance
                logger.log_performance((time.time() - start_time), final_conf)
                self.last_prediction_time = time.time()
                
                return result
                
            except Exception as e:
                logger.logger.error(f"Prediction error: {e}")
                return {"prediction": "NEUTRAL", "confidence": 0.0, "error": str(e)}
    
    def clean_cache(self):
        """Remove old cache entries"""
        current_time = time.time()
        keys_to_remove = [k for k, v in self.cache.items() 
                         if current_time - v['timestamp'] > 300]  # 5 minutes
        for k in keys_to_remove:
            del self.cache[k]
    
    def health_check(self) -> Dict[str, Any]:
        """System health check"""
        return {
            "artifacts_loaded": self.artifacts_loaded,
            "cache_size": len(self.cache),
            "system_health": logger.check_health(),
            "last_prediction_time": self.last_prediction_time,
            "device": str(self.device)
        }

# --------------------------
# Main Training Function
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    logger.logger.info(f"Using device: {device}")

    try:
        df_raw, seq_indices, metadata, labels = load_and_build_sequences(config.DATA_PATH, seq_len=config.SEQ_LEN)
        N = len(df_raw)
        if len(seq_indices) == 0:
            raise RuntimeError("Not enough sequences. Increase data or reduce SEQ_LEN/MAX_LOOKAHEAD.")

        # Fit scaler on train rows only
        train_row_cutoff = int(N * config.TRAIN_SPLIT_FRAC)
        feature_cols = ['open','high','low','close','volume','quote_volume','trades','taker_buy_base','taker_buy_quote','volatility']
        scaler = RobustScaler()  # More robust to outliers
        scaler.fit(df_raw[feature_cols].iloc[:train_row_cutoff].values)
        joblib.dump(scaler, os.path.join(config.ARTIFACTS_DIR, "module2_scaler.save"))
        logger.logger.info("Fitted scaler on train rows only")

        # Build normalized sequences
        X_norm, y, metadata = build_norm_sequences_and_labels(df_raw, config.SEQ_LEN, seq_indices, metadata, labels, scaler)
        
        # Balance classes
        X_balanced, y_balanced, metadata_balanced = balance_classes(X_norm, y, metadata)
        
        num_seq = X_balanced.shape[0]
        train_seq_cutoff = int(num_seq * config.TRAIN_SPLIT_FRAC)
        logger.logger.info(f"Balanced sequences shape: {X_balanced.shape}")

        # Build embeddings
        embeddings_arr, pca_obj = build_sequence_embeddings(X_balanced, train_seq_cutoff, embed_dim=config.EMBED_DIM)
        embeddings_obj = {'embeddings': embeddings_arr, 'pca_obj': pca_obj}
        joblib.dump(pca_obj, os.path.join(config.ARTIFACTS_DIR, "module2_pca.save"))
        np.save(os.path.join(config.ARTIFACTS_DIR, "historical_embeddings.npy"), embeddings_arr)
        np.save(os.path.join(config.ARTIFACTS_DIR, "historical_sequences_norm.npy"), X_balanced)
        np.save(os.path.join(config.ARTIFACTS_DIR, "historical_labels.npy"), y_balanced)
        logger.logger.info("Saved embeddings, sequences, labels")

        # Fit ANN index
        nn_idx = NearestNeighbors(n_neighbors=min(config.ANN_CANDIDATES, embeddings_arr.shape[0]), algorithm='auto', metric='euclidean')
        nn_idx.fit(embeddings_arr)
        joblib.dump(nn_idx, os.path.join(config.ARTIFACTS_DIR, "module2_nn_idx.save"))
        logger.logger.info("Fitted and saved NearestNeighbors index")

        # Prepare dataloaders
        train_loader, val_loader, test_loader, (train_end, val_end, total_seq) = prepare_dataloaders_chrono(X_balanced, y_balanced)
        logger.logger.info(f"Prepared dataloaders: train_end_seq={train_end}, val_end_seq={val_end}")

        # Initialize and train model
        model = ProductionPriceFlowLSTM(input_size=X_balanced.shape[2], hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, output_size=3).to(device)

        trained_model = train_loop(model, train_loader, val_loader, X_balanced[:train_end], y_balanced[:train_end],
                                 X_balanced, y_balanced, embeddings_obj, nn_idx, device, epochs=config.EPOCHS, lr=config.LR)

        logger.logger.info("Training completed successfully")
        
    except Exception as e:
        logger.logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()