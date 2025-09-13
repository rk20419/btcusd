import pandas as pd
import numpy as np
import ta
from datetime import datetime
import os

def calculate_advanced_indicators(file_path):
    """
    Calculate comprehensive technical indicators for trading
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    # Convert all numerical columns to float
    numerical_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                     'trades', 'taker_buy_base', 'taker_buy_quote']
    
    for col in numerical_cols:
        df[col] = df[col].astype(float)
    
    print("Calculating trend indicators...")
    # 1. Trend Indicators
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    
    print("Calculating momentum indicators...")
    # 2. Momentum Indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['awesome_oscillator'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
    
    print("Calculating volatility indicators...")
    # 3. Volatility Indicators
    df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
    df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20'].replace(0, 1)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['kc_upper'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20)
    df['kc_lower'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20)
    
    print("Calculating volume indicators...")
    # 4. Volume-based Indicators
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum().replace(0, 1)
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
    df['volume_obv'] = df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, -1)
    
    print("Calculating microstructure indicators...")
    # 5. Microstructure Indicators
    df['price_range'] = (df['high'] - df['low']) / df['low'].replace(0, 1) * 100
    df['body_size'] = abs(df['close'] - df['open']) / df['open'].replace(0, 1) * 100
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['low'].replace(0, 1) * 100
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['low'].replace(0, 1) * 100
    df['price_velocity'] = df['close'].pct_change(periods=5)
    
    print("Calculating order flow indicators...")
    # 6. Order Flow Analysis
    df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, 1)
    df['taker_sell_volume'] = df['volume'] - df['taker_buy_base']
    df['taker_sell_ratio'] = df['taker_sell_volume'] / df['volume'].replace(0, 1)
    df['buy_sell_imbalance'] = (df['taker_buy_base'] - df['taker_sell_volume']) / df['volume'].replace(0, 1)
    df['taker_buy_quote_ratio'] = df['taker_buy_quote'] / df['quote_volume'].replace(0, 1)
    df['quote_volume_ratio'] = df['quote_volume'] / df['quote_volume'].rolling(20).mean().replace(0, 1)
    df['avg_trade_size'] = df['volume'] / df['trades'].replace(0, 1)
    df['avg_taker_buy_size'] = df['taker_buy_base'] / df['trades'].replace(0, 1)
    
    print("Identifying price patterns...")
    # 7. Price Action Patterns
    doji_threshold = 0.1
    df['is_doji'] = (df['body_size'] < doji_threshold).astype(int)
    
    df['prev_close'] = df['close'].shift(1)
    df['prev_open'] = df['open'].shift(1)
    
    # Bullish Engulfing
    df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                               (df['prev_close'] < df['prev_open']) & 
                               (df['open'] < df['prev_close']) & 
                               (df['close'] > df['prev_open'])).astype(int)
    
    # Bearish Engulfing
    df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                               (df['prev_close'] > df['prev_open']) & 
                               (df['open'] > df['prev_close']) & 
                               (df['close'] < df['prev_open'])).astype(int)
    
    # Hammer pattern
    df['is_hammer'] = ((df['close'] > df['open']) & 
                       (df['lower_shadow'] > 2 * df['body_size']) & 
                       (df['upper_shadow'] < df['body_size'] * 0.1)).astype(int)
    
    # Shooting star pattern
    df['is_shooting_star'] = ((df['close'] < df['open']) & 
                              (df['upper_shadow'] > 2 * df['body_size']) & 
                              (df['lower_shadow'] < df['body_size'] * 0.1)).astype(int)
    
    print("Calculating advanced features...")
    # 8. Advanced Microstructure Features
    df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
    df['hilo_ratio'] = (df['high'] - df['low']) / df['weighted_close'].replace(0, 1) * 100
    df['price_momentum'] = df['close'] / df['close'].shift(5) - 1
    
    # 9. Time-based Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_london_open'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    df['is_ny_open'] = ((df.index.hour >= 13) | (df.index.hour < 21)).astype(int)
    df['is_asian_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
    
    print("Calculating volatility features...")
    # 10. Dynamic Volatility Features
    df['volatility_5'] = df['close'].pct_change().rolling(5).std()
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df['volatility_ratio'] = df['volatility_5'] / df['volatility_20'].replace(0, 1)
    df['volatility_regime'] = pd.cut(df['volatility_20'], bins=5, labels=[1, 2, 3, 4, 5])
    
    print("Calculating momentum features...")
    # 11. Price Momentum Features
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_3'] = df['close'].pct_change(3)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_acceleration'] = df['price_change_1'] - df['price_change_1'].shift(1)
    
    print("Calculating market regime features...")
    # 12. Market Regime Features
    df['above_sma_20'] = (df['close'] > df['sma_20']).astype(int)
    df['above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
    df['trend_strength'] = (df['close'] - df['sma_20']) / df['sma_20'].replace(0, 1) * 100
    df['trend_direction'] = np.where(df['close'] > df['sma_20'], 1, -1)
    
    print("Calculating advanced order flow features...")
    # 13. Advanced Order Flow Features
    df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])
    df['taker_buy_volume_ratio'] = df['taker_buy_base'] / df['volume'].rolling(20).mean().replace(0, 1)
    df['large_trades_ratio'] = (df['trades'] > df['trades'].rolling(20).mean() * 1.5).astype(int)
    
    # 14. Market Microstructure Features
    df['liquidity_ratio'] = df['quote_volume'] / df['volume'].replace(0, 1)
    df['efficiency_ratio'] = abs(df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min()).replace(0, 1)
    
    # Remove NaN values
    df = df.dropna()
    
    # Define feature columns
    feature_columns = [
        # Trend indicators
        'sma_20', 'sma_50', 'ema_12', 'ema_26', 'adx', 'cci',
        # Momentum indicators
        'rsi', 'stoch_k', 'stoch_d', 'macd', 'awesome_oscillator',
        # Volatility indicators
        'bb_upper', 'bb_lower', 'bb_width', 'atr', 'kc_upper', 'kc_lower',
        # Volume indicators
        'obv', 'vwap', 'volume_ratio', 'volume_obv',
        # Microstructure indicators
        'price_range', 'body_size', 'upper_shadow', 'lower_shadow', 'price_velocity',
        # Order flow indicators
        'taker_buy_ratio', 'taker_sell_ratio', 'buy_sell_imbalance', 
        'taker_buy_quote_ratio', 'quote_volume_ratio', 'avg_trade_size', 'avg_taker_buy_size',
        # Price patterns
        'is_doji', 'bullish_engulfing', 'bearish_engulfing', 'is_hammer', 'is_shooting_star',
        # Advanced features
        'hilo_ratio', 'price_momentum',
        # Time-based features
        'hour', 'day_of_week', 'is_london_open', 'is_ny_open', 'is_asian_session',
        # Volatility features
        'volatility_5', 'volatility_20', 'volatility_ratio',
        # Momentum features
        'price_change_1', 'price_change_3', 'price_change_5', 'price_acceleration',
        # Market regime features
        'above_sma_20', 'above_sma_50', 'trend_strength', 'trend_direction',
        # Advanced order flow
        'volume_price_correlation', 'taker_buy_volume_ratio', 'large_trades_ratio',
        # Market microstructure
        'liquidity_ratio', 'efficiency_ratio'
    ]
    
    # Add encoded volatility regime
    df = pd.get_dummies(df, columns=['volatility_regime'], prefix='vol_regime')
    volatility_regime_cols = [col for col in df.columns if col.startswith('vol_regime_')]
    feature_columns.extend(volatility_regime_cols)
    
    # Final DataFrame with all features and original price data
    final_columns = feature_columns + ['open', 'high', 'low', 'close', 'volume']
    final_df = df[final_columns]
    
    return final_df, feature_columns

def main():
    # Create directories if they don't exist
    os.makedirs('data/historical', exist_ok=True)
    
    # File path
    file_path = "data/historical/BTCUSDT_1m_200000.csv"
    output_path = "data/historical/BTCUSDT_1m_200000_with_indicators.csv"
    
    try:
        # Calculate indicators
        df, feature_columns = calculate_advanced_indicators(file_path)
        
        # Save to CSV
        df.to_csv(output_path)
        
        print("Indicators calculated and saved successfully!")
        print(f"Total indicators: {len(feature_columns)}")
        print(f"Data shape: {df.shape}")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()