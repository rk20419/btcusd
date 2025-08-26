# trainer/data_engine_trainer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import logging
import argparse
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_rsi(series, period=14):
    """Calculate RSI using pandas vectorized operations."""
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26):
    """Calculate MACD line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.ewm(span=period, min_periods=period).mean()
    return atr

def calculate_obv(df):
    """Calculate On-Balance Volume (OBV)."""
    sign = np.sign(df['close'].diff(1))
    obv = (sign * df['volume']).fillna(0).cumsum()
    return obv

def detect_bullish_engulfing(df):
    """Detect bullish engulfing pattern."""
    prev_open, prev_close = df['open'].shift(1), df['close'].shift(1)
    curr_open, curr_close = df['open'], df['close']
    
    prev_bearish = prev_close < prev_open
    curr_bullish = curr_close > curr_open
    body_engulf = (curr_open <= prev_close) & (curr_close >= prev_open)
    
    is_bull_eng = prev_bearish & curr_bullish & body_engulf
    return is_bull_eng.astype(int)

def detect_hidden_divergence(df, order=5):
    """
    Detect hidden divergence with adaptive order based on data length.
    Returns both bullish and bearish divergence.
    """
    data_length = len(df)
    adaptive_order = max(2, min(order, data_length // 100))
    
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
    """Assign session flags based on UTC hour."""
    dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    hour = dt.dt.hour
    
    df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['session_london'] = ((hour >= 8) & (hour < 13)).astype(int)
    df['session_ny'] = ((hour >= 13) & (hour < 22)).astype(int)
    
    return df

def calculate_weighted_avg(series, weights):
    """Calculate rolling weighted average."""
    def weighted_func(x):
        if len(x) < len(weights):
            return np.nan
        valid_weights = weights[-len(x):] if len(x) < len(weights) else weights
        return np.dot(x, valid_weights) / valid_weights.sum()
    
    return series.rolling(window=len(weights), min_periods=1).apply(weighted_func, raw=True)

def calculate_features(df):
    """Compute all 24 features as per summary."""
    # Base features
    df['volatility'] = df['high'] - df['low']
    
    # Technical indicators
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['macd'] = calculate_macd(df['close'])
    df['obv'] = calculate_obv(df)
    df['atr_14'] = calculate_atr(df, 14)
    df['momentum_5'] = df['close'].pct_change(5)
    
    # Microstructure features
    df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, np.nan)
    df['volume_spike_5'] = df['volume'] / df['volume'].rolling(5, min_periods=1).mean().replace(0, np.nan)
    df['large_trade_ratio'] = df['quote_volume'] / df['trades'].replace(0, np.nan)
    
    # Session flags
    df = get_session_flags(df)
    
    # Price action patterns
    df['bullish_engulfing'] = detect_bullish_engulfing(df)
    df['hidden_divergence_bull'], df['hidden_divergence_bear'] = detect_hidden_divergence(df)
    
    # Weighted features
    exp_weights = np.exp(np.linspace(-1, 0, 50))
    df['weighted_rsi'] = calculate_weighted_avg(df['rsi_14'], exp_weights)
    df['weighted_volume'] = calculate_weighted_avg(df['volume'], exp_weights)
    
    return df

def validate_data(df):
    """Validate input DataFrame."""
    required_cols = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'
    ]
    
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not df['timestamp'].is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonically increasing.")
    
    return True

def handle_missing_values(df, feature_cols):
    """Handle missing values in features."""
    # Forward fill then backward fill
    df[feature_cols] = df[feature_cols].ffill().bfill()
    
    # Binary features: fill with 0
    binary_cols = ['session_asia', 'session_london', 'session_ny', 
                  'bullish_engulfing', 'hidden_divergence_bull', 'hidden_divergence_bear']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Ratio features: fill with median
    ratio_cols = ['taker_buy_ratio', 'volume_spike_5', 'large_trade_ratio']
    for col in ratio_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)
    
    return df

def main(args):
    try:
        # Setup logging
        os.makedirs(args.logs_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(args.logs_dir, 'data_engine_trainer.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info("Starting Data Engine Trainer")
        
        # Load data
        logger.info(f"Loading data from {args.data_path}")
        df = pd.read_csv(args.data_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Validate data
        validate_data(df)
        
        # Define feature columns (24 features as per summary)
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'volatility', 'rsi_14', 'macd',
            'obv', 'atr_14', 'momentum_5', 'taker_buy_ratio', 'volume_spike_5',
            'large_trade_ratio', 'session_asia', 'session_london', 'session_ny',
            'bullish_engulfing', 'hidden_divergence_bull', 'weighted_rsi', 'weighted_volume'
        ]
        
        # Create output directories
        os.makedirs(args.processed_dir, exist_ok=True)
        os.makedirs(args.scalers_dir, exist_ok=True)
        
        # Process in chunks
        chunk_idx = 0
        for start_idx in range(0, len(df), args.chunk_size):
            end_idx = min(start_idx + args.chunk_size + args.overlap, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            
            logger.info(f"Processing chunk {chunk_idx} (rows {start_idx}-{end_idx})")
            
            # Calculate features
            chunk = calculate_features(chunk)
            
            # Handle missing values
            chunk = handle_missing_values(chunk, feature_cols)
            
            # Use only the actual chunk (excluding overlap for training)
            actual_chunk = chunk.iloc[:args.chunk_size] if chunk_idx > 0 else chunk
            
            # Fit scaler
            scaler = MinMaxScaler()
            scaler.fit(actual_chunk[feature_cols])
            
            # Save processed chunk
            chunk_path = os.path.join(args.processed_dir, f'chunk_{chunk_idx}.pkl')
            actual_chunk.to_pickle(chunk_path)
            
            # Save scaler
            scaler_path = os.path.join(args.scalers_dir, f'scaler_{chunk_idx}.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info(f"Saved chunk {chunk_idx} with {len(actual_chunk)} rows")
            chunk_idx += 1
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer for Module 1: Smart Data Engine')
    parser.add_argument('--data_path', type=str, default='data/historical/BTCUSDT_1m_200000.csv')
    parser.add_argument('--chunk_size', type=int, default=50000)
    parser.add_argument('--overlap', type=int, default=100)
    parser.add_argument('--processed_dir', type=str, default='processed')
    parser.add_argument('--scalers_dir', type=str, default='scalers')
    parser.add_argument('--logs_dir', type=str, default='logs')
    
    args = parser.parse_args()
    main(args)