# live/data_engine_predictor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import logging
import argparse
from scipy.signal import argrelextrema
import time
import threading
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for state management
last_processed_timestamp = 0
processing_lock = threading.Lock()

def calculate_rsi(series, period=14):
    """Calculate RSI for real-time data."""
    if len(series) < period:
        return pd.Series([np.nan] * len(series))
    
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26):
    """Calculate MACD line for real-time data."""
    if len(series) < slow:
        return pd.Series([np.nan] * len(series))
    
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def calculate_atr(df, period=14):
    """Calculate ATR for real-time data."""
    if len(df) < period:
        return pd.Series([np.nan] * len(df))
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.ewm(span=period, min_periods=1).mean()
    return atr

def calculate_obv(df):
    """Calculate OBV for real-time data."""
    sign = np.sign(df['close'].diff(1))
    obv = (sign * df['volume']).fillna(0).cumsum()
    return obv

def detect_bullish_engulfing(df):
    """Detect bullish engulfing pattern for real-time data."""
    if len(df) < 2:
        return pd.Series([0] * len(df))
    
    prev_open, prev_close = df['open'].shift(1), df['close'].shift(1)
    curr_open, curr_close = df['open'], df['close']
    
    prev_bearish = prev_close < prev_open
    curr_bullish = curr_close > curr_open
    body_engulf = (curr_open <= prev_close) & (curr_close >= prev_open)
    
    is_bull_eng = prev_bearish & curr_bullish & body_engulf
    return is_bull_eng.astype(int)

def detect_hidden_divergence_realtime(df, order=5):
    """Detect hidden divergence for real-time data with adaptive order."""
    data_length = len(df)
    adaptive_order = max(2, min(order, data_length // 100))
    
    if len(df) < adaptive_order * 2 + 1:
        return np.zeros(len(df)), np.zeros(len(df))
    
    # Find peaks and troughs for price and RSI
    price_highs = argrelextrema(df['close'].values, np.greater, order=adaptive_order)[0]
    price_lows = argrelextrema(df['close'].values, np.less, order=adaptive_order)[0]
    rsi_highs = argrelextrema(df['rsi_14'].values, np.greater, order=adaptive_order)[0]
    rsi_lows = argrelextrema(df['rsi_14'].values, np.less, order=adaptive_order)[0]
    
    hidden_div_bullish = np.zeros(len(df))
    hidden_div_bearish = np.zeros(len(df))
    
    # Bullish hidden divergence: Price higher low, RSI lower low
    for i in range(1, min(len(price_lows), len(rsi_lows))):
        if (df['close'].iloc[price_lows[i]] > df['close'].iloc[price_lows[i-1]] and
            df['rsi_14'].iloc[rsi_lows[i]] < df['rsi_14'].iloc[rsi_lows[i-1]]):
            hidden_div_bullish[price_lows[i]] = 1
    
    # Bearish hidden divergence: Price lower high, RSI higher high
    for i in range(1, min(len(price_highs), len(rsi_highs))):
        if (df['close'].iloc[price_highs[i]] < df['close'].iloc[price_highs[i-1]] and
            df['rsi_14'].iloc[rsi_highs[i]] > df['rsi_14'].iloc[rsi_highs[i-1]]):
            hidden_div_bearish[price_highs[i]] = 1
    
    return hidden_div_bullish, hidden_div_bearish

def get_session_flags(df):
    """Assign session flags for real-time data."""
    dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    hour = dt.dt.hour
    
    df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['session_london'] = ((hour >= 8) & (hour < 13)).astype(int)
    df['session_ny'] = ((hour >= 13) & (hour < 22)).astype(int)
    
    return df

def calculate_weighted_avg_realtime(series, window=50):
    """Calculate weighted average for real-time  window."""
    weights = np.exp(np.linspace(-1, 0, window))
    
    def weighted_func(x):
        if len(x) < len(weights):
            return np.nan
        return np.dot(x[-len(weights):], weights) / weights.sum()
    
    return series.rolling(window=window, min_periods=1).apply(weighted_func, raw=True)

def load_latest_scaler(scalers_dir):
    """Load the latest scaler with validation."""
    try:
        scaler_files = [f for f in os.listdir(scalers_dir) 
                       if f.startswith('scaler_') and f.endswith('.pkl')]
        if not scaler_files:
            return None
        
        latest_scaler = max(scaler_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        scaler_path = os.path.join(scalers_dir, latest_scaler)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        logger.info(f"Loaded scaler: {latest_scaler}")
        return scaler
        
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        return None

def process_new_data(args):
    """Process new data with proper locking and timestamp tracking."""
    global last_processed_timestamp
    
    # Acquire lock to prevent concurrent processing
    if not processing_lock.acquire(blocking=False):
        logger.debug("Another process is already running, skipping...")
        return False
    
    try:
        # Check if file exists
        if not os.path.exists(args.data_path):
            logger.warning(f"Data file not found: {args.data_path}")
            return False
        
        # Load latest data
        df = pd.read_csv(args.data_path)
        
        if len(df) == 0:
            logger.warning("Data file is empty")
            return False
        
        # Get the latest timestamp
        latest_timestamp = df['timestamp'].iloc[-1]
        
        # Check if we have already processed this timestamp
        if latest_timestamp <= last_processed_timestamp:
            return False
        
        # Take last 60 candles for processing
        recent_data = df.tail(60).copy()
        
        # Load latest scaler
        scaler = load_latest_scaler(args.scalers_dir)
        if scaler is None:
            logger.warning("No scaler available, skipping processing")
            return False
        
        # Calculate features
        recent_data['volatility'] = recent_data['high'] - recent_data['low']
        recent_data['rsi_14'] = calculate_rsi(recent_data['close'], 14)
        recent_data['macd'] = calculate_macd(recent_data['close'])
        recent_data['obv'] = calculate_obv(recent_data)
        recent_data['atr_14'] = calculate_atr(recent_data, 14)
        recent_data['momentum_5'] = recent_data['close'].pct_change(5)
        
        # Microstructure features
        recent_data['taker_buy_ratio'] = recent_data['taker_buy_base'] / recent_data['volume'].replace(0, np.nan)
        recent_data['volume_spike_5'] = recent_data['volume'] / recent_data['volume'].rolling(5, min_periods=1).mean().replace(0, np.nan)
        recent_data['large_trade_ratio'] = recent_data['quote_volume'] / recent_data['trades'].replace(0, np.nan)
        
        # Session flags
        recent_data = get_session_flags(recent_data)
        
        # Price action patterns
        recent_data['bullish_engulfing'] = detect_bullish_engulfing(recent_data)
        recent_data['hidden_divergence_bull'], _ = detect_hidden_divergence_realtime(recent_data, order=2)
        
        # Weighted features (smaller window for real-time)
        recent_data['weighted_rsi'] = calculate_weighted_avg_realtime(recent_data['rsi_14'], window=20)
        recent_data['weighted_volume'] = calculate_weighted_avg_realtime(recent_data['volume'], window=20)
        
        # Handle missing values
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'volatility', 'rsi_14', 'macd',
            'obv', 'atr_14', 'momentum_5', 'taker_buy_ratio', 'volume_spike_5',
            'large_trade_ratio', 'session_asia', 'session_london', 'session_ny',
            'bullish_engulfing', 'hidden_divergence_bull', 'weighted_rsi', 'weighted_volume'
        ]
        
        recent_data[feature_cols] = recent_data[feature_cols].ffill().bfill()
        
        # Normalize features
        normalized_features = scaler.transform(recent_data[feature_cols].iloc[-1:].values)
        normalized_features = np.clip(normalized_features, 0, 1)
        
        # Create output DataFrame
        output_df = pd.DataFrame(normalized_features, columns=feature_cols)
        output_df['timestamp'] = recent_data['timestamp'].iloc[-1]
        ist_time = pd.to_datetime(recent_data['timestamp'].iloc[-1], unit='ms').tz_localize('UTC').tz_convert('Asia/Kolkata')
        output_df['datetime_ist'] = ist_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'live_features.pkl')
        output_df.to_pickle(output_path)
        
        # Update last processed timestamp
        last_processed_timestamp = latest_timestamp
        
        logger.info(f"Processed new data at {output_df['datetime_ist'].iloc[0]}")
        logger.info(f"Features saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return False
    finally:
        # Always release the lock
        processing_lock.release()

def main(args):
    """Main function for continuous predictor."""
    try:
        # Setup logging
        os.makedirs(args.logs_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(args.logs_dir, 'data_engine_predictor.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info("Starting Continuous Data Engine Predictor")
        logger.info(f"Monitoring: {args.data_path}")
        
        # Initialize last processed timestamp
        global last_processed_timestamp
        if os.path.exists(args.data_path):
            try:
                df = pd.read_csv(args.data_path)
                if len(df) > 0:
                    last_processed_timestamp = df['timestamp'].iloc[-1]
                    logger.info(f"Initialized with last timestamp: {last_processed_timestamp}")
            except Exception as e:
                logger.error(f"Error initializing timestamp: {e}")
        
        # Main monitoring loop
        logger.info("Starting monitoring loop...")
        
        while True:
            try:
                process_new_data(args)
                # Sleep for the specified interval
                time.sleep(args.poll_interval)
            except KeyboardInterrupt:
                logger.info("Predictor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(args.poll_interval * 2)  # Longer sleep on error
        
    except Exception as e:
        logger.error(f"Predictor failed: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continuous Predictor for Module 1: Smart Data Engine')
    parser.add_argument('--data_path', type=str, default='data/live/1m.csv')
    parser.add_argument('--scalers_dir', type=str, default='scalers')
    parser.add_argument('--output_dir', type=str, default='processed/live')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--poll_interval', type=int, default=5, help='Polling interval in seconds')
    
    args = parser.parse_args()
    main(args)