"""
Real-time Predictor for ProductionPriceFlowLSTM + XGBoost hybrid
- Watches data/live/1m.csv for new candles
- Loads artifacts from models/production/
- Computes indicators (volatility if needed, RSI, MACD)
- Builds last SEQ_LEN=30 window, scales with RobustScaler
- Runs LSTM -> softmax, then optional XGBoost (hybrid)
- **Strategy Gate**: Only emit BUY/SELL if confidence >= threshold AND expected_move >= MIN_MOVE_THRESHOLD (≈300)
- Appends prediction to data/predictor.csv

Run:
    python realtime_predictor_final.py

Dependencies:
    pandas, numpy, torch, joblib, xgboost, pytz
"""
import os
import time
import gc
import sys
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# ------------------------
# Config
# ------------------------
class Config:
    SEQ_LEN = 30
    FEATURE_COLS = [
        'open','high','low','close','volume','quote_volume','trades',
        'taker_buy_base','taker_buy_quote','volatility','rsi','macd'
    ]
    LIVE_CSV = 'data/live/1m.csv'
    OUTPUT_CSV = 'data/predictor.csv'
    ARTIFACTS_DIR = 'models/production'
    MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'module2_model.pth')
    XGB_PATH = os.path.join(ARTIFACTS_DIR, 'module2_xgb.save')
    SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'module2_scaler.save')
    POLL_SEC = 1.0
    CONFIDENCE_THRESHOLD = 0.70   # stricter than training default 0.65
    MIN_MOVE_THRESHOLD = 300.0    # enforce 300+ move chance
    ATR_PERIOD = 14
    VOL_WINDOW = 20               # to approximate vol20 used in training
    LOG_PATH = 'logs/production/realtime_predictor.log'

config = Config()

# ------------------------
# Logging
# ------------------------
logger = logging.getLogger('realtime_predictor')
logger.setLevel(logging.INFO)
if not logger.handlers:
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    fh = logging.FileHandler(config.LOG_PATH)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# ------------------------
# Model (must match training)
# ------------------------
class ProductionPriceFlowLSTM(nn.Module):
    def __init__(self, input_size: int = 12, hidden_size: int = 192, num_layers: int = 2,
                 output_size: int = 3, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        mult = 2 if bidirectional else 1
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * mult, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * mult, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.dropout(context)
        return self.fc(out)

# ------------------------
# Utilities (mirror training)
# ------------------------

def ensure_volatility(df: pd.DataFrame, atr_period: int = config.ATR_PERIOD) -> None:
    # Recompute if missing/NaN/all-zero; otherwise coerce numeric and fillna
    if 'volatility' not in df.columns or df['volatility'].isnull().all() or (df['volatility'].abs().sum() == 0):
        close_shift = df['close'].shift()
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum((df['high'] - close_shift).abs(), (df['low'] - close_shift).abs())
        )
        df['volatility'] = tr.rolling(atr_period).mean().fillna(0) * 100
    else:
        df['volatility'] = pd.to_numeric(df['volatility'], errors='coerce').fillna(0)


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> None:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)


def calculate_macd(df: pd.DataFrame, short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> None:
    short_ema = df['close'].ewm(span=short_period, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    df['macd'] = (macd_line - signal_line).fillna(0)


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(path, error_bad_lines=False)
    except Exception as e:
        logger.warning(f"Failed reading {path}: {e}")
        return None
    return df

# Infer LSTM architecture from saved state_dict

def infer_arch_from_state(state_dict: dict) -> Tuple[int, int, int, bool]:
    # Returns (input_size, hidden_size, num_layers, bidirectional)
    w_ih0 = state_dict.get('lstm.weight_ih_l0')
    if w_ih0 is None:
        raise RuntimeError('Could not find lstm.weight_ih_l0 in state_dict')
    input_size = w_ih0.shape[1]
    hidden_size = w_ih0.shape[0] // 4
    # Count layers by keys like lstm.weight_ih_l{idx}
    layer_indices = []
    for k in state_dict.keys():
        if k.startswith('lstm.weight_ih_l') and 'reverse' not in k:
            try:
                idx = int(k.split('lstm.weight_ih_l')[-1])
                layer_indices.append(idx)
            except Exception:
                pass
    num_layers = (max(layer_indices) + 1) if layer_indices else 1
    bidirectional = any('reverse' in k for k in state_dict.keys() if k.startswith('lstm.weight_ih_l'))
    return input_size, hidden_size, num_layers, bidirectional


def load_artifacts(device: torch.device):
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Missing model weights: {config.MODEL_PATH}")
    if not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError(f"Missing scaler: {config.SCALER_PATH}")

    scaler = joblib.load(config.SCALER_PATH)

    state = torch.load(config.MODEL_PATH, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state_dict = state['state_dict']
    elif isinstance(state, dict):
        state_dict = state
    else:
        raise RuntimeError('Unexpected model checkpoint format')

    in_size, hid_size, n_layers, bidir = infer_arch_from_state(state_dict)
    model = ProductionPriceFlowLSTM(
        input_size=in_size, hidden_size=hid_size, num_layers=n_layers, output_size=3,
        dropout=0.2, bidirectional=bidir
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    xgb = None
    if os.path.exists(config.XGB_PATH):
        try:
            xgb = joblib.load(config.XGB_PATH)
            logger.info('Loaded XGBoost hybrid model.')
        except Exception as e:
            logger.warning(f"Failed to load XGBoost model, will use LSTM-only. Error: {e}")
            xgb = None
    else:
        logger.info('XGBoost model not found; using LSTM-only predictions.')

    return model, xgb, scaler


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    required = ['timestamp','open','high','low','close','volume','quote_volume','trades','taker_buy_base','taker_buy_quote']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Coerce numeric
    num_cols = ['open','high','low','close','volume','quote_volume','trades','taker_buy_base','taker_buy_quote']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    ensure_volatility(df)
    calculate_rsi(df)
    calculate_macd(df)

    # Order & fill
    for c in config.FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Feature column missing after compute: {c}")
    feat = df[config.FEATURE_COLS].copy()
    feat = feat.ffill().bfill().fillna(0)
    return feat


def build_window_scaled(feat: pd.DataFrame, scaler) -> Optional[np.ndarray]:
    if len(feat) < config.SEQ_LEN:
        return None
    last = feat.iloc[-config.SEQ_LEN:]
    arr = scaler.transform(last.values)
    arr = arr.astype(np.float32).reshape(1, config.SEQ_LEN, len(config.FEATURE_COLS))
    return arr


def predict_one(model, xgb, device, window_tensor: np.ndarray) -> Tuple[int, float, np.ndarray, Optional[np.ndarray]]:
    with torch.no_grad():
        X = torch.tensor(window_tensor, dtype=torch.float32, device=device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        lstm_label = int(np.argmax(probs))
        hybrid_label, hybrid_probs = lstm_label, None
        if xgb is not None:
            try:
                x_input = probs.reshape(1, -1)
                proba_func = getattr(xgb, 'predict_proba', None)
                if callable(proba_func):
                    hybrid_probs = proba_func(x_input)[0]
                    hybrid_label = int(np.argmax(hybrid_probs))
                else:
                    hybrid_label = int(xgb.predict(x_input)[0])
            except Exception as e:
                logger.warning(f"XGB predict failed; falling back to LSTM-only. Error: {e}")
                hybrid_label = lstm_label
                hybrid_probs = None
        conf = float(np.max(hybrid_probs if hybrid_probs is not None else probs))
        return hybrid_label, conf, probs, (hybrid_probs if hybrid_probs is not None else None)


def apply_strategy_gate(raw_label: int, conf: float, feat: pd.DataFrame) -> Tuple[int, str, float]:
    """
    Gate BUY/SELL decisions using:
    - Confidence >= CONFIDENCE_THRESHOLD
    - Expected move proxy >= MIN_MOVE_THRESHOLD
    Expected move proxy = 2.0 * mean(last VOL_WINDOW of 'volatility')
    (Matches training's move_threshold ≈ 2*vol20 when valid.)
    Returns: (final_label, reason, expected_move)
    """
    reason = []
    # Compute expected move proxy
    vol_ser = feat['volatility'].tail(config.VOL_WINDOW)
    vol20 = float(vol_ser.mean()) if len(vol_ser) > 0 else 0.0
    expected_move = 2.0 * vol20 if vol20 > 0 else 0.0

    # Confidence gate
    if conf < config.CONFIDENCE_THRESHOLD:
        reason.append(f"conf({conf:.3f})<thr({config.CONFIDENCE_THRESHOLD:.2f})")
        return 2, ';'.join(reason) or 'low_confidence', expected_move

    # Move-size gate
    if expected_move < config.MIN_MOVE_THRESHOLD:
        reason.append(f"exp_move({expected_move:.1f})<min({config.MIN_MOVE_THRESHOLD:.0f})")
        return 2, ';'.join(reason) or 'weak_move', expected_move

    # Passed both gates → keep BUY/SELL; if raw was NEUTRAL, keep it
    if raw_label in (0, 1):
        reason.append('passed')
        return raw_label, ';'.join(reason), expected_move
    else:
        return 2, 'model_neutral', expected_move


def append_output(row: dict):
    os.makedirs(os.path.dirname(config.OUTPUT_CSV), exist_ok=True)
    write_header = not os.path.exists(config.OUTPUT_CSV)
    df = pd.DataFrame([row])
    df.to_csv(config.OUTPUT_CSV, mode='a', header=write_header, index=False)


LABEL_MAP = {0: 'BUY', 1: 'SELL', 2: 'NEUTRAL'}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        model, xgb, scaler = load_artifacts(device)
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        sys.exit(1)

    last_ts = None
    logger.info(f"Watching {config.LIVE_CSV} ...")

    while True:
        try:
            df_live = safe_read_csv(config.LIVE_CSV)
            if df_live is None or df_live.empty:
                time.sleep(config.POLL_SEC)
                continue

            # make sure timestamp exists and numeric
            if 'timestamp' not in df_live.columns:
                logger.warning('timestamp column missing in live data')
                time.sleep(config.POLL_SEC)
                continue
            df_live['timestamp'] = pd.to_numeric(df_live['timestamp'], errors='coerce')
            df_live = df_live.dropna(subset=['timestamp'])
            if df_live.empty:
                time.sleep(config.POLL_SEC)
                continue

            cur_ts = int(df_live['timestamp'].iloc[-1])
            # Process only when a new candle appears
            if last_ts is not None and cur_ts == last_ts:
                time.sleep(config.POLL_SEC)
                continue

            # Build features for the whole dataframe, then take last window
            feat = prepare_features(df_live.copy())
            window = build_window_scaled(feat, scaler)
            if window is None:
                logger.info(f"Waiting for {config.SEQ_LEN} candles; have {len(feat)}")
                time.sleep(config.POLL_SEC)
                continue

            raw_label, conf, lstm_probs, hybrid_probs = predict_one(model, xgb, device, window)
            gated_label, gate_reason, expected_move = apply_strategy_gate(raw_label, conf, feat)

            # Names & probs
            raw_name = LABEL_MAP.get(raw_label, str(raw_label))
            final_name = LABEL_MAP.get(gated_label, str(gated_label))

            # Pull friendly time if available
            dt_col = 'datetime_ist' if 'datetime_ist' in df_live.columns else None
            dt_val = df_live[dt_col].iloc[-1] if dt_col else ''

            out = {
                'timestamp': cur_ts,
                'datetime_ist': dt_val,
                'raw_prediction': raw_label,
                'raw_prediction_name': raw_name,
                'final_prediction': gated_label,
                'final_prediction_name': final_name,
                'confidence': round(conf, 6),
                'expected_move_proxy': round(float(expected_move), 3),
                'gate_reason': gate_reason,
                'prob_buy_lstm': round(float(lstm_probs[0]), 6),
                'prob_sell_lstm': round(float(lstm_probs[1]), 6),
                'prob_neutral_lstm': round(float(lstm_probs[2]), 6),
                'prob_buy_hybrid': round(float(hybrid_probs[0]), 6) if isinstance(hybrid_probs, (np.ndarray, list)) else '',
                'prob_sell_hybrid': round(float(hybrid_probs[1]), 6) if isinstance(hybrid_probs, (np.ndarray, list)) else '',
                'prob_neutral_hybrid': round(float(hybrid_probs[2]), 6) if isinstance(hybrid_probs, (np.ndarray, list)) else '',
                'source': 'hybrid' if isinstance(hybrid_probs, (np.ndarray, list)) else 'lstm_only'
            }

            append_output(out)
            last_ts = cur_ts

            logger.info(
                f"New candle @ {cur_ts} -> RAW={raw_name}, FINAL={final_name} "
                f"(conf={conf:.3f}, exp_move≈{expected_move:.1f}) | saved to {config.OUTPUT_CSV}"
            )

            gc.collect()
            time.sleep(config.POLL_SEC)
        except KeyboardInterrupt:
            logger.info('Stopping realtime predictor (KeyboardInterrupt).')
            break
        except Exception as e:
            logger.error(f"Realtime loop error: {e}")
            time.sleep(config.POLL_SEC)
            continue


if __name__ == '__main__':
    main()
