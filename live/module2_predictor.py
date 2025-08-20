"""
PRODUCTION-READY Price Flow Predictor Module
- Real-time predictions using trained Module 2 model
- Matches trainer's architecture, thresholds, and configuration
- Robust data validation, thread safety, caching, and paper trading
- Compatible with BTCUSDT 1-minute candles
"""

import os
import time
import json
import logging
import threading
from typing import Tuple, List, Dict, Any, Optional, Union
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, IncrementalPCA
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# -----------------------
# Configuration (matched with trainer)
# -----------------------
class Config:
    CSV_PATH = "data/live/1m.csv"  # Live CSV updated by WebSocket
    PRED_FILE = "data/price_flow_detections.json"
    ARTIFACTS_DIR = "models/production"
    LOG_DIR = "logs"
    POLL_INTERVAL = 5  # Seconds
    SEQ_LEN = 60  # Matches trainer
    ANN_CANDIDATES = 150  # Matches trainer
    TOP_M_FOR_DTW = 30  # Matches trainer
    DTW_TOPK = 5  # Matches trainer
    CONFIDENCE_THRESHOLD = 0.65  # Matches trainer
    MIN_DATA_QUALITY_SCORE = 0.8  # Matches trainer
    MAX_LATENCY_MS = 500  # Matches trainer
    PERFORMANCE_WINDOW = 100  # Matches trainer
    MAX_CACHE_AGE = 300  # 5 minutes
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(config.PRED_FILE), exist_ok=True)

# -----------------------
# Logging
# -----------------------
class ProductionLogger:
    def __init__(self):
        self.logger = logging.getLogger("module2_predictor")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(config.LOG_DIR, "module2_predictor.log"))
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

# -----------------------
# Model Architecture (matches trainer)
# -----------------------
class ProductionPriceFlowLSTM(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 2, output_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0 if num_layers == 1 else 0.1)
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

# -----------------------
# Data Quality Validator (matches trainer)
# -----------------------
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
        return quality_score >= config.MIN_DATA_QUALITY_SCORE, quality_score

# -----------------------
# Helpers (aligned with trainer)
# -----------------------
def detect_session_from_timestamp(ts_ms: int) -> Tuple[str, bool]:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=pytz.UTC)
    hour = dt.hour
    is_sunday = dt.weekday() == 6
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
    subset = df_raw.iloc[start_idx:start_idx + window]
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

def compute_dtw_confidence_on_candidates(current_seq_norm: np.ndarray, candidate_idxs: List[int],
                                        X_norm: np.ndarray, labels: np.ndarray, topk: int = config.DTW_TOPK) -> float:
    distances = []
    for i in candidate_idxs[:config.TOP_M_FOR_DTW]:
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

# -----------------------
# Predictor Class
# -----------------------
class ProductionPriceFlowPredictor:
    def __init__(self, artifacts_dir: str = config.ARTIFACTS_DIR):
        self.device = config.DEVICE
        self.artifacts_loaded = False
        self.load_artifacts(artifacts_dir)
        self.cache = {}
        self.last_prediction_time = 0
        self.prediction_lock = threading.Lock()

    def load_artifacts(self, artifacts_dir: str):
        for attempt in range(3):
            try:
                self.scaler = joblib.load(os.path.join(artifacts_dir, "module2_scaler.save"))
                self.pca = joblib.load(os.path.join(artifacts_dir, "module2_pca.save"))
                self.nn_idx = joblib.load(os.path.join(artifacts_dir, "module2_nn_idx.save"))
                self.X_norm_all = np.load(os.path.join(artifacts_dir, "historical_sequences_norm.npy"))
                self.labels_all = np.load(os.path.join(artifacts_dir, "historical_labels.npy"))
                self.model = ProductionPriceFlowLSTM(input_size=self.X_norm_all.shape[2]).to(self.device)
                self.model.load_state_dict(torch.load(os.path.join(artifacts_dir, "module2_model.pth"), map_location=self.device))
                self.model.eval()
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
        latest_data = df.iloc[-config.SEQ_LEN:].copy()
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'volatility']
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
                return {"prediction": "NEUTRAL", "confidence": 0.0, "error": "Artifacts not loaded"}
            if time.time() - self.last_prediction_time < 1.0:
                return {"prediction": "NEUTRAL", "confidence": 0.0, "error": "Rate limited"}
            try:
                sequence = self.prepare_latest_sequence(df)
                if sequence is None:
                    return {"prediction": "NEUTRAL", "confidence": 0.0, "error": "Invalid sequence"}
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

                # DTW confidence
                flat_seq = sequence.reshape(1, -1)
                emb = self.pca.transform(flat_seq)
                n_neighbors = min(config.ANN_CANDIDATES, self.X_norm_all.shape[0])
                _, idxs = self.nn_idx.kneighbors(emb, n_neighbors=n_neighbors)
                candidate_idxs = idxs[0].tolist()
                dtw_conf = compute_dtw_confidence_on_candidates(sequence[0], candidate_idxs, self.X_norm_all, self.labels_all)
                
                # Combine confidences
                combined_conf = 0.6 * lstm_conf + 0.4 * dtw_conf
                final_pred = 2 if combined_conf < config.CONFIDENCE_THRESHOLD else lstm_pred
                final_conf = 1.0 - combined_conf if final_pred == 2 else combined_conf

                # Session and thresholds
                timestamp = int(df['timestamp'].iloc[-1])
                session, is_sunday = detect_session_from_timestamp(timestamp)
                if is_sunday:
                    move_threshold = 350
                else:
                    move_threshold = 500 if session == 'NY' else 300 if session == 'Asia' else 400

                # Volume and volatility checks
                df_tail = df.tail(config.SEQ_LEN).reset_index(drop=True)
                current_close = float(df_tail['close'].iloc[-1])
                current_low = float(df_tail['low'].iloc[-1])
                current_high = float(df_tail['high'].iloc[-1])
                current_volume = float(df_tail['volume'].iloc[-1])
                current_trades = float(df_tail['trades'].iloc[-1])
                current_vol = float(df_tail['volatility'].iloc[-1])
                taker_pressure = float(df_tail['taker_buy_base'].iloc[-1]) / (current_volume + 1e-10)
                start_idx = max(0, len(df_tail) - 20)
                avg_volume = df_tail['volume'].iloc[start_idx:].mean()
                avg_trades = df_tail['trades'].iloc[start_idx:].mean()
                avg_vol = df_tail['volatility'].iloc[start_idx:].mean()
                volume_factor = 1.8 if session == 'NY' else 1.5 if session == 'Asia' else 1.6
                volatility_factor = 1.5 if session == 'NY' else 1.2 if session == 'Asia' else 1.4
                if is_sunday:
                    volume_factor *= 0.8
                volume_spike = (current_volume > volume_factor * avg_volume) if avg_volume > 0 else False
                trade_spike = (current_trades > volume_factor * avg_trades) if avg_trades > 0 else False
                high_volatility = (current_vol > volatility_factor * avg_vol) if avg_vol > 0 else False

                # Move size and override
                lookback_df = df_tail.tail(15)  # LOOKBACK_MOVE = 15 for consistency
                min_low = float(lookback_df['low'].min())
                max_high = float(lookback_df['high'].max())
                move_size = 0.0
                pred_labels = {0: "BUY", 1: "SELL", 2: "NEUTRAL"}
                signal = pred_labels[final_pred]
                if signal == 'BUY' and (min_low >= current_low + 150):
                    move_size = max_high - current_close
                    if move_size < move_threshold:
                        signal = 'NEUTRAL'
                        final_conf = 0.9
                elif signal == 'SELL' and (max_high <= current_high - 150):
                    move_size = current_close - min_low
                    if move_size < move_threshold:
                        signal = 'NEUTRAL'
                        final_conf = 0.9

                # Trigger method
                trigger_method = 'no_move'
                if signal == 'BUY':
                    if move_size >= move_threshold:
                        trigger_method = 'momentum'
                    elif (volume_spike or trade_spike) and (taker_pressure > 0.55) and high_volatility:
                        trigger_method = 'volume'
                    else:
                        trigger_method = 'volatility'
                elif signal == 'SELL':
                    if move_size >= move_threshold:
                        trigger_method = 'momentum'
                    elif (volume_spike or trade_spike) and (taker_pressure < 0.45) and high_volatility:
                        trigger_method = 'volume'
                    else:
                        trigger_method = 'volatility'

                # SL, TP1, TP2
                sl = None
                tp1 = None
                tp2 = None
                if signal in ['BUY', 'SELL']:
                    multiplier = 1.5 if final_conf > 0.9 else 2.0
                    sl = current_close - multiplier * current_vol * 100 if signal == 'BUY' else current_close + multiplier * current_vol * 100
                    search_start = max(0, len(df_tail) - 60)
                    found_tp1_idx = None
                    for j in range(search_start, len(df_tail)):
                        avg_c, is_stable = calculate_stability_raw(df_tail, j)
                        if is_stable:
                            tp1 = avg_c
                            found_tp1_idx = j
                            break
                    if tp1 is not None and found_tp1_idx is not None:
                        for j in range(found_tp1_idx + 1, len(df_tail)):
                            avg_c, is_stable = calculate_stability_raw(df_tail, j)
                            if is_stable and ((signal == 'BUY' and avg_c > tp1) or (signal == 'SELL' and avg_c < tp1)):
                                tp2 = avg_c
                                break

                # Output
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
                        "dtw_confidence": float(dtw_conf)
                    }
                }
                self.cache[cache_key] = {'timestamp': time.time(), 'result': result}
                self.clean_cache()
                logger.log_performance((time.time() - start_time), final_conf)
                self.last_prediction_time = time.time()
                return result
            except Exception as e:
                logger.logger.error(f"Prediction error: {e}")
                return {"prediction": "NEUTRAL", "confidence": 0.0, "error": str(e)}

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

# -----------------------
# Paper Trader
# -----------------------
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
        entry_price = price * (1 + self.slippage) if side == 'BUY' else price * (1 - self.slippage)
        cost = entry_price * size_units
        fee = cost * self.commission
        if cost + fee > self.cash:
            logger.logger.warning("PaperTrader: Insufficient cash for entry")
            return False
        self.cash -= (cost + fee)
        pos = {'side': side, 'size': size_units, 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'entry_ts': timestamp}
        self.positions.append(pos)
        logger.logger.info(f"PaperTrader ENTER {side} size {size_units:.4f} @ {entry_price:.2f}")
        return True

    def update_and_settle(self, latest_high, latest_low, latest_close, timestamp):
        closed = []
        for pos in list(self.positions):
            side = pos['side']
            entry = pos['entry_price']
            sl = pos['sl']
            tp1 = pos['tp1']
            tp2 = pos['tp2']
            exit_price = None
            pnl = 0.0
            if side == 'BUY':
                if tp1 and latest_high >= tp1:
                    exit_price = tp1 * (1 - self.slippage)
                elif tp2 and latest_high >= tp2:
                    exit_price = tp2 * (1 - self.slippage)
                elif sl and latest_low <= sl:
                    exit_price = sl * (1 + self.slippage)
            else:
                if tp1 and latest_low <= tp1:
                    exit_price = tp1 * (1 + self.slippage)
                elif tp2 and latest_low <= tp2:
                    exit_price = tp2 * (1 + self.slippage)
                elif sl and latest_high >= sl:
                    exit_price = sl * (1 - self.slippage)
            if exit_price is not None:
                size = pos['size']
                fee = exit_price * size * self.commission
                if side == 'BUY':
                    pnl = (exit_price - entry) * size - fee
                    self.cash += exit_price * size - fee
                else:
                    pnl = (entry - exit_price) * size - fee
                    self.cash += exit_price * size - fee
                self.trade_log.append({
                    'entry_ts': pos['entry_ts'],
                    'exit_ts': timestamp,
                    'side': side,
                    'entry': entry,
                    'exit': exit_price,
                    'pnl': pnl
                })
                self.positions.remove(pos)
                logger.logger.info(f"PaperTrader EXIT {side} @ {exit_price:.2f}, PNL: {pnl:.2f}")
        return

    def get_stats(self):
        total_trades = len(self.trade_log)
        total_pnl = sum(t['pnl'] for t in self.trade_log)
        win_trades = len([t for t in self.trade_log if t['pnl'] > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0
        return {
            'cash': self.cash,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'open_positions': len(self.positions)
        }

# -----------------------
# Save Prediction
# -----------------------
def save_prediction(output: dict, filepath=config.PRED_FILE):
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                arr = json.load(f)
        else:
            arr = []
        arr.append(output)
        with open(filepath, 'w') as f:
            json.dump(arr, f, indent=2)
        logger.logger.info(f"Saved prediction @ {output['datetime_ist']} signal={output['signal']} conf={output['confidence']:.3f}")
    except Exception as e:
        logger.logger.error(f"Failed to save prediction: {e}")

# -----------------------
# Main Monitor Loop
# -----------------------
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
            required = ['timestamp', 'datetime_ist', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
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
                    save_prediction(pred)
                    if pred['signal'] in ['BUY', 'SELL'] and pred['confidence'] >= config.CONFIDENCE_THRESHOLD:
                        size = 0.001  # Example: 0.001 BTC
                        price_for_entry = float(df['open'].iloc[-1])
                        paper.enter(pred['signal'], size, price_for_entry, sl=pred['sl'], tp1=pred['tp1'], tp2=pred['tp2'], timestamp=pred['timestamp'])
                    latest_high = float(df['high'].iloc[-1])
                    latest_low = float(df['low'].iloc[-1])
                    latest_close = float(df['close'].iloc[-1])
                    paper.update_and_settle(latest_high, latest_low, latest_close, pred['timestamp'])
                    logger.logger.info(f"PaperTrader stats: {paper.get_stats()}")
                last_row_count = current_len
            else:
                logger.logger.debug("No new candles.")
            time.sleep(poll_interval)
        except Exception as e:
            logger.logger.error(f"Error in monitor loop: {e}")
            time.sleep(poll_interval)

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    monitor_csv_and_predict()