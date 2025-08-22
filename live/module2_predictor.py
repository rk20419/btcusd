# live/module2_predictor.py
"""
PRODUCTION-READY Price Flow Predictor Module
- Real-time prediction from live CSV data
- Integrates with paper trader for simulation
- Enhanced with hybrid LSTM + XGBoost
- Comprehensive logging and health monitoring
- Aligned with trainer configurations and logic
"""

import os
import logging
import math
import time
import csv
from typing import Tuple, List, Dict, Any, Optional, Union
from collections import Counter
from datetime import datetime
import pytz
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --------------------------
# Production Configuration (fully matched with trainer)
# --------------------------
@dataclass
class Config:
    SEED = 42
    SEQ_LEN = 30  # Reduced for faster response to quick changes
    MAX_LOOKAHEAD = 30  # Reduced for faster response
    MIN_STABLE_CANDLES = 3
    MAX_STABLE_CANDLES = 5
    PRICE_RANGE = 300
    EPOCHS = 50
    BATCH_SIZE = 128
    LR = 1e-3
    HIDDEN_SIZE = 128  # Increased for better capacity
    NUM_LAYERS = 2
    EMBED_DIM = 32
    ANN_CANDIDATES = 100  # Reduced for speed
    TOP_M_FOR_DTW = 20  # Reduced for speed
    DTW_TOPK = 3
    TRAIN_SPLIT_FRAC = 0.80
    VAL_SPLIT_FRAC = 0.15
    ARTIFACTS_DIR = "models/production"
    LOG_DIR = "logs/production"
    DATA_PATH = "data/historical/BTCUSDT_1m_5000.csv"  # Not used in predictor, but included for completeness
    MIN_IPCA_BATCH = 5000
    EARLY_STOP_PATIENCE = 10  # Increased for more training
    MAX_SAMPLES_PER_CLASS = 5000  # Increased to retain more data
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

    # Predictor-specific
    CSV_PATH = "data/live/1m.csv"  # Live CSV updated by WebSocket
    PRED_FILE = "data/price_flow_detections.csv"  # Changed to CSV
    POLL_INTERVAL = 1  # Changed to 1 second
    MAX_CACHE_AGE = 300  # 5 minutes
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(config.PRED_FILE), exist_ok=True)

# --------------------------
# Enhanced Logging (matches trainer)
# --------------------------
class ProductionLogger:
    def __init__(self):
        self.logger = logging.getLogger("price_flow_predictor")
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(os.path.join(config.LOG_DIR, "predictor.log"))
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.performance_metrics = {
            'prediction_times': [],
            'confidence_scores': [],
            'recent_errors': 0,
            'last_alert_time': None
        }
    
    def log_performance(self, prediction_time: float, confidence: float):
        self.performance_metrics['prediction_times'].append(prediction_time)
        self.performance_metrics['confidence_scores'].append(confidence)
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
# Enhanced Model (matches trainer)
# --------------------------
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

# --------------------------
# Data Quality System (matches trainer)
# --------------------------
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

# --------------------------
# Enhanced Utilities (matched with trainer)
# --------------------------
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

# --------------------------
# Predictor Class (enhanced to match trainer, including hybrid XGBoost)
# --------------------------
class ProductionPriceFlowPredictor:
    def __init__(self, artifacts_dir: str = config.ARTIFACTS_DIR):
        self.device = config.DEVICE
        self.artifacts_loaded = False
        self.load_artifacts(artifacts_dir)
        self.cache = {}
        self.last_prediction_time = 0
        self.prediction_lock = threading.Lock()

    def load_artifacts(self, artifacts_dir: str):
        for attempt in range(config.MAX_RETRIES):
            try:
                self.scaler = joblib.load(os.path.join(artifacts_dir, "module2_scaler.save"))
                self.pca = joblib.load(os.path.join(artifacts_dir, "module2_pca.save"))
                self.nn_idx = joblib.load(os.path.join(artifacts_dir, "module2_nn_idx.save"))
                self.X_norm_all = np.load(os.path.join(artifacts_dir, "historical_sequences_norm.npy"))
                self.labels_all = np.load(os.path.join(artifacts_dir, "historical_labels.npy"))
                self.model = ProductionPriceFlowLSTM(input_size=self.X_norm_all.shape[2]).to(self.device)
                self.model.load_state_dict(torch.load(os.path.join(artifacts_dir, "module2_model.pth"), map_location=self.device))
                self.model.eval()
                self.xgb = joblib.load(os.path.join(artifacts_dir, "module2_xgb.save"))
                self.artifacts_loaded = True
                logger.logger.info("Predictor initialized successfully")
                return
            except Exception as e:
                logger.logger.error(f"Attempt {attempt + 1} failed to load artifacts: {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError("Failed to load artifacts after retries")

    def prepare_latest_sequence(self, df: pd.DataFrame) -> Union[np.ndarray, None]:
        if len(df) < config.SEQ_LEN:
            logger.logger.warning(f"Not enough data. Need {config.SEQ_LEN} rows, got {len(df)}")
            return None
        is_valid, quality_score = DataQualityValidator.validate_real_time_data(df)
        if not is_valid:
            logger.logger.warning(f"Poor data quality: {quality_score:.3f}")
            return None
        ensure_volatility(df)
        calculate_rsi(df)
        calculate_macd(df)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'volatility', 'rsi', 'macd']
        df[numeric_cols] = df[numeric_cols].ffill().fillna(0).astype(float)
        df = df[(df['high'] - df['low'] <= 0.05 * df['close'])]  # Outlier removal matching trainer
        latest_data = df.iloc[-config.SEQ_LEN:].copy()
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'volatility', 'rsi', 'macd']
        try:
            latest_norm = self.scaler.transform(latest_data[feature_cols].values)
            return latest_norm.reshape(1, config.SEQ_LEN, -1)
        except Exception as e:
            logger.logger.error(f"Error normalizing data: {e}")
            return None

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        with self.prediction_lock:
            start_time = time.time()
            if not self.artifacts_loaded:
                return {"signal": "NEUTRAL", "confidence": 0.0, "error": "Artifacts not loaded"}
            if time.time() - self.last_prediction_time < config.REAL_TIME_UPDATE_INTERVAL:
                return {"signal": "NEUTRAL", "confidence": 0.0, "error": "Rate limited"}
            try:
                sequence = self.prepare_latest_sequence(df)
                if sequence is None:
                    return {"signal": "NEUTRAL", "confidence": 0.0, "error": "Invalid sequence"}
                cache_key = hash(sequence.tobytes())
                if cache_key in self.cache and time.time() - self.cache[cache_key]['timestamp'] < config.MAX_CACHE_AGE:
                    return self.cache[cache_key]['result']

                # LSTM prediction
                with torch.no_grad():
                    tensor_seq = torch.tensor(sequence, dtype=torch.float32).to(self.device)
                    output = self.model(tensor_seq)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    lstm_pred = int(np.argmax(probs))
                    lstm_conf = float(probs[lstm_pred])

                # Hybrid with XGBoost (matching trainer's hybrid approach)
                lstm_features = probs.reshape(1, -1)
                hybrid_pred = int(self.xgb.predict(lstm_features)[0])
                hybrid_conf = lstm_conf  # Since XGBoost refines, but confidence from probs

                # DTW confidence
                flat_seq = sequence.reshape(1, -1)
                emb = self.pca.transform(flat_seq)
                n_neighbors = min(config.ANN_CANDIDATES, self.X_norm_all.shape[0])
                _, idxs = self.nn_idx.kneighbors(emb, n_neighbors=n_neighbors)
                candidate_idxs = idxs[0].tolist()
                dtw_conf = compute_dtw_confidence_on_candidates(sequence[0], candidate_idxs, self.X_norm_all, self.labels_all)

                # Combine confidences (weighted as in predictor, but using hybrid)
                combined_conf = 0.6 * hybrid_conf + 0.4 * dtw_conf
                final_pred = 2 if combined_conf < config.CONFIDENCE_THRESHOLD else hybrid_pred
                final_conf = 1.0 - combined_conf if final_pred == 2 else combined_conf

                # Session and dynamic thresholds (fixed bug from trainer/predictor)
                df_tail = df.iloc[-config.REALTIME_SEQ_LEN:].reset_index(drop=True)
                timestamp = int(df_tail['timestamp'].iloc[-1])
                session, is_sunday = detect_session_from_timestamp(timestamp)
                vol_series = df_tail['volatility']
                vol20 = vol_series.rolling(20).mean().iloc[-1]
                vol_std = vol_series.rolling(20).std().iloc[-1]
                base_threshold = 300 if session == 'Asia' else 400 if session == 'London' else 500
                move_threshold = 1.5 * vol20 if not np.isnan(vol20) and vol20 > 0 else base_threshold
                current_vol = float(df_tail['volatility'].iloc[-1])
                if current_vol > vol20 + vol_std:
                    move_threshold *= 1.2  # Boost for high volatility
                if is_sunday:
                    move_threshold *= 0.8

                # Volume and volatility checks (matching trainer)
                current_close = float(df_tail['close'].iloc[-1])
                current_low = float(df_tail['low'].iloc[-1])
                current_high = float(df_tail['high'].iloc[-1])
                current_volume = float(df_tail['volume'].iloc[-1])
                current_trades = int(df_tail['trades'].iloc[-1])
                current_rsi = float(df_tail['rsi'].iloc[-1])
                current_macd = float(df_tail['macd'].iloc[-1])
                taker_pressure = float(df_tail['taker_buy_base'].iloc[-1]) / (current_volume + 1e-10)
                start_avg = max(0, len(df_tail) - 19)
                avg_volume = df_tail['volume'].iloc[start_avg:].mean()
                avg_trades = df_tail['trades'].iloc[start_avg:].mean()
                avg_vol = df_tail['volatility'].iloc[start_avg:].mean()
                if session == 'NY':
                    volume_factor = 1.8; volatility_factor = 1.5
                elif session == 'Asia':
                    volume_factor = 1.5; volatility_factor = 1.2
                else:
                    volume_factor = 1.6; volatility_factor = 1.4
                if is_sunday:
                    volume_factor *= 0.8
                volume_spike = (current_volume > volume_factor * avg_volume) if avg_volume > 0 else False
                trade_spike = (current_trades > volume_factor * avg_trades) if avg_trades > 0 else False
                high_volatility = (current_vol > volatility_factor * avg_vol) if avg_vol > 0 else False
                rsi_overbought = current_rsi > 70
                rsi_oversold = current_rsi < 30
                macd_positive = current_macd > 0

                # Move size and override (using historical lookback for confirmation, as no future data)
                lookback_df = df_tail.tail(config.MAX_LOOKAHEAD)
                min_low = float(lookback_df['low'].min())
                max_high = float(lookback_df['high'].max())
                move_size = 0.0
                pred_labels = {0: "BUY", 1: "SELL", 2: "NEUTRAL"}
                signal = pred_labels[final_pred]
                if signal == 'BUY':
                    move_size = max_high - current_close
                    if move_size < move_threshold * 0.5 or (volume_spike and taker_pressure < 0.5) or rsi_overbought:
                        signal = 'NEUTRAL'
                        final_conf = 0.9
                elif signal == 'SELL':
                    move_size = current_close - min_low
                    if move_size < move_threshold * 0.5 or (volume_spike and taker_pressure > 0.5) or rsi_oversold:
                        signal = 'NEUTRAL'
                        final_conf = 0.9

                # Trigger method (aligned with trainer's labeling logic)
                trigger_method = 'no_move'
                if signal == 'BUY':
                    if move_size >= move_threshold:
                        trigger_method = 'momentum'
                    elif (volume_spike or trade_spike) and (taker_pressure > 0.55) and high_volatility and not rsi_overbought:
                        trigger_method = 'volume'
                    elif move_size >= move_threshold * 0.5 and taker_pressure > 0.6 and macd_positive:
                        trigger_method = 'indicator'
                    else:
                        trigger_method = 'volatility'
                elif signal == 'SELL':
                    if move_size >= move_threshold:
                        trigger_method = 'momentum'
                    elif (volume_spike or trade_spike) and (taker_pressure < 0.45) and high_volatility and not rsi_oversold:
                        trigger_method = 'volume'
                    elif move_size <= -move_threshold * 0.5 and taker_pressure < 0.4 and not macd_positive:
                        trigger_method = 'indicator'
                    else:
                        trigger_method = 'volatility'

                # SL, TP1, TP2 (using historical stable periods for estimation, as in trainer metadata)
                sl = None
                tp1 = None
                tp2 = None
                if signal in ['BUY', 'SELL']:
                    vol = current_vol
                    sl = current_close - 2 * vol * 100 if signal == 'BUY' else current_close + 2 * vol * 100
                    found_tp1_idx = None
                    for j in range(len(df_tail) - config.MAX_LOOKAHEAD, len(df_tail)):
                        if j < 0: continue
                        avg_close_raw, is_stable = calculate_stability_raw(df_tail, j)
                        if is_stable:
                            tp1 = avg_close_raw
                            found_tp1_idx = j
                            break
                    if tp1 is not None and found_tp1_idx is not None:
                        for j in range(found_tp1_idx + 1, len(df_tail)):
                            avg_close_raw, is_stable = calculate_stability_raw(df_tail, j)
                            if is_stable and ((signal == 'BUY' and avg_close_raw > tp1) or (signal == 'SELL' and avg_close_raw < tp1)):
                                tp2 = avg_close_raw
                                break

                # Output (enhanced details)
                result = {
                    "timestamp": int(timestamp),
                    "datetime_ist": str(df_tail['datetime_ist'].iloc[-1]),
                    "session": session,
                    "signal": signal,
                    "confidence": float(final_conf),
                    "sl": float(sl) if sl is not None else None,
                    "tp1": float(tp1) if tp1 is not None else None,
                    "tp2": float(tp2) if tp2 is not None else None,
                    "trigger_method": trigger_method,
                    "details": {
                        "move_size": float(move_size),
                        "volatility": float(current_vol),
                        "range": float(df_tail['high'].max() - df_tail['low'].min()),
                        "lstm_confidence": float(lstm_conf),
                        "dtw_confidence": float(dtw_conf),
                        "hybrid_pred": hybrid_pred
                    }
                }
                self.cache[cache_key] = {'timestamp': time.time(), 'result': result}
                self.clean_cache()
                prediction_time = time.time() - start_time
                logger.log_performance(prediction_time, final_conf)
                self.last_prediction_time = time.time()

                # Log detailed metrics for debugging
                logger.logger.info(f"Prediction: {signal}, Combined Conf: {combined_conf:.3f}, LSTM Conf: {lstm_conf:.3f}, "
                                   f"DTW Conf: {dtw_conf:.3f}, Hybrid Pred: {hybrid_pred}, Move Size: {move_size:.2f}, "
                                   f"Min Low: {min_low:.2f}, Max High: {max_high:.2f}, Current Close: {current_close:.2f}")

                return result
            except Exception as e:
                logger.logger.error(f"Prediction error: {e}")
                return {"signal": "NEUTRAL", "confidence": 0.0, "error": str(e)}

    def clean_cache(self):
        current_time = time.time()
        keys_to_remove = [k for k, v in self.cache.items() if current_time - v['timestamp'] > config.MAX_CACHE_AGE]
        for k in keys_to_remove:
            del self.cache[k]

    def health_check(self) -> Dict[str, Any]:
        return {
            "artifacts_loaded": self.artifacts_loaded,
            "cache_size": len(self.cache),
            "system_health": logger.check_health(),
            "last_prediction_time": self.last_prediction_time,
            "device": str(self.device)
        }

# --------------------------
# Paper Trader (enhanced with Sharpe calculation for stats and fixed long/short logic)
# --------------------------
def calculate_sharpe_ratio(returns):
    if len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    return mean_return / std_dev if std_dev > 0 else 0.0

class PaperTrader:
    def __init__(self, cash=10000.0, commission_pct=0.001, slippage_pct=0.0005, max_positions=1):
        self.cash = cash
        self.commission = commission_pct
        self.slippage = slippage_pct
        self.positions = []
        self.trade_log = []
        self.max_positions = max_positions

    def enter(self, side, size_units, price, sl=None, tp1=None, tp2=None, timestamp=None):
        if len(self.positions) >= self.max_positions:
            logger.logger.warning("PaperTrader: Max positions reached")
            return False
        fee = 0.0
        entry_price = 0.0
        if side == 'BUY':
            entry_price = price * (1 + self.slippage)
            cost = entry_price * size_units
            fee = cost * self.commission
            if cost + fee > self.cash:
                logger.logger.warning("PaperTrader: Insufficient cash for entry")
                return False
            self.cash -= cost + fee
        else:  # SELL (short)
            entry_price = price * (1 - self.slippage)
            proceeds = entry_price * size_units
            fee = proceeds * self.commission
            self.cash += proceeds - fee
        pos = {'side': side, 'size': size_units, 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'entry_ts': timestamp, 'entry_fee': fee}
        self.positions.append(pos)
        logger.logger.info(f"PaperTrader ENTER {side} size {size_units:.4f} @ {entry_price:.2f}")
        return True

    def update_and_settle(self, latest_high, latest_low, latest_close, timestamp):
        for pos in list(self.positions):
            side = pos['side']
            entry = pos['entry_price']
            sl = pos['sl']
            tp1 = pos['tp1']
            tp2 = pos['tp2']
            exit_price = None
            if side == 'BUY':
                if tp1 and latest_high >= tp1:
                    exit_price = tp1 * (1 - self.slippage)
                elif tp2 and latest_high >= tp2:
                    exit_price = tp2 * (1 - self.slippage)
                elif sl and latest_low <= sl:
                    exit_price = sl * (1 - self.slippage)
            else:
                if tp1 and latest_low <= tp1:
                    exit_price = tp1 * (1 + self.slippage)
                elif tp2 and latest_low <= tp2:
                    exit_price = tp2 * (1 + self.slippage)
                elif sl and latest_high >= sl:
                    exit_price = sl * (1 + self.slippage)
            if exit_price is not None:
                size = pos['size']
                pnl = 0.0
                exit_fee = 0.0
                if side == 'BUY':
                    proceeds = exit_price * size
                    exit_fee = proceeds * self.commission
                    self.cash += proceeds - exit_fee
                    pnl = (exit_price - entry) * size - pos['entry_fee'] - exit_fee
                else:
                    cost = exit_price * size
                    exit_fee = cost * self.commission
                    self.cash -= cost + exit_fee
                    pnl = (entry - exit_price) * size - pos['entry_fee'] - exit_fee
                self.trade_log.append({
                    'entry_ts': pos['entry_ts'],
                    'exit_ts': timestamp,
                    'side': side,
                    'entry': entry,
                    'exit': exit_price,
                    'pnl': pnl,
                    'size': size
                })
                self.positions.remove(pos)
                logger.logger.info(f"PaperTrader EXIT {side} @ {exit_price:.2f}, PNL: {pnl:.2f}")

    def get_stats(self):
        total_trades = len(self.trade_log)
        if total_trades == 0:
            return {
                'cash': self.cash,
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'open_positions': len(self.positions)
            }
        total_pnl = sum(t['pnl'] for t in self.trade_log)
        win_trades = len([t for t in self.trade_log if t['pnl'] > 0])
        win_rate = win_trades / total_trades
        returns = [t['pnl'] / (t['entry'] * t['size']) for t in self.trade_log]
        sharpe = calculate_sharpe_ratio(returns)
        return {
            'cash': self.cash,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'open_positions': len(self.positions)
        }

# --------------------------
# Save Prediction to CSV (enhanced with all details)
# --------------------------
def save_prediction_to_csv(output: dict, filepath=config.PRED_FILE):
    try:
        file_exists = os.path.exists(filepath)
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "datetime_ist", "session", "signal", "confidence", "sl", "tp1", "tp2", "trigger_method", "move_size", "volatility", "range", "lstm_confidence", "dtw_confidence", "hybrid_pred"])
            if not file_exists:
                writer.writeheader()
            row = {
                "timestamp": output["timestamp"],
                "datetime_ist": output["datetime_ist"],
                "session": output["session"],
                "signal": output["signal"],
                "confidence": output["confidence"],
                "sl": output["sl"],
                "tp1": output["tp1"],
                "tp2": output["tp2"],
                "trigger_method": output["trigger_method"],
                "move_size": output["details"]["move_size"],
                "volatility": output["details"]["volatility"],
                "range": output["details"]["range"],
                "lstm_confidence": output["details"]["lstm_confidence"],
                "dtw_confidence": output["details"]["dtw_confidence"],
                "hybrid_pred": output["details"]["hybrid_pred"]
            }
            writer.writerow(row)
        logger.logger.info(f"Saved prediction @ {output['datetime_ist']} signal={output['signal']} conf={output['confidence']:.3f}")
    except Exception as e:
        logger.logger.error(f"Failed to save prediction to CSV: {e}")

# --------------------------
# Main Monitor Loop (enhanced with health checks)
# --------------------------
def monitor_csv_and_predict(csv_path=config.CSV_PATH, poll_interval=config.POLL_INTERVAL):
    predictor = ProductionPriceFlowPredictor()
    paper = PaperTrader()
    last_row_count = 0
    logger.logger.info(f"Monitoring CSV: {csv_path} (poll every {poll_interval}s). Waiting for new candles...")
    while True:
        try:
            if not os.path.exists(csv_path):
                logger.logger.warning(f"CSV {csv_path} not found. Waiting...")
                time.sleep(poll_interval)
                continue
            df = pd.read_csv(csv_path)
            required = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'datetime_ist']
            if not all(col in df.columns for col in required):
                logger.logger.error(f"CSV missing required columns: {required}")
                time.sleep(poll_interval)
                continue
            current_len = len(df)
            if current_len > last_row_count:
                new_candles = current_len - last_row_count
                logger.logger.info(f"Detected {new_candles} new candle(s). Running prediction.")
                pred = predictor.predict(df)
                if pred.get('error'):
                    logger.logger.warning(f"Prediction failed: {pred['error']}")
                else:
                    save_prediction_to_csv(pred)
                    if pred['signal'] in ['BUY', 'SELL'] and pred['confidence'] >= config.CONFIDENCE_THRESHOLD:
                        size = 0.001  # Example: 0.001 BTC
                        price_for_entry = float(df['close'].iloc[-1])  # Use close for entry
                        paper.enter(pred['signal'], size, price_for_entry, sl=pred['sl'], tp1=pred['tp1'], tp2=pred['tp2'], timestamp=pred['timestamp'])
                    latest_high = float(df['high'].iloc[-1])
                    latest_low = float(df['low'].iloc[-1])
                    latest_close = float(df['close'].iloc[-1])
                    paper.update_and_settle(latest_high, latest_low, latest_close, pred['timestamp'])
                    stats = paper.get_stats()
                    logger.logger.info(f"PaperTrader stats: {stats}")
                last_row_count = current_len
            else:
                logger.logger.debug("No new candles.")
            if not predictor.health_check()['system_health']:
                logger.logger.warning("System health check failed. Consider restarting.")
            time.sleep(poll_interval)
        except Exception as e:
            logger.logger.error(f"Error in monitor loop: {e}")
            time.sleep(poll_interval)

# --------------------------
# Entrypoint
# --------------------------
if __name__ == "__main__":
    monitor_csv_and_predict()