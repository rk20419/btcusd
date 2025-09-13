import pandas as pd
import numpy as np
import ta
import time
import os
from datetime import datetime
import hashlib

class LiveDataProcessor:
    def __init__(self, input_file, output_file, buffer_size=200, process_existing=False):
        self.input_file = input_file
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.last_processed_time = 0
        self.processed_rows = set()
        self.process_existing = process_existing
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    def calculate_indicators(self, df):
        """
        Calculate all technical indicators to match historical processing format
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                       'trades', 'taker_buy_base', 'taker_buy_quote']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Trend Indicators
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        
        # 2. Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['awesome_oscillator'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
        
        # 3. Volatility Indicators
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20'].replace(0, 1)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['kc_upper'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20)
        df['kc_lower'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20)
        
        # 4. Volume-based Indicators
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum().replace(0, 1)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
        df['volume_obv'] = df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, -1)
        
        # 5. Microstructure Indicators
        df['price_range'] = (df['high'] - df['low']) / df['low'].replace(0, 1) * 100
        df['body_size'] = abs(df['close'] - df['open']) / df['open'].replace(0, 1) * 100
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['low'].replace(0, 1) * 100
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['low'].replace(0, 1) * 100
        df['price_velocity'] = df['close'].pct_change(periods=5)
        
        # 6. Order Flow Analysis
        df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, 1)
        df['taker_sell_volume'] = df['volume'] - df['taker_buy_base']
        df['taker_sell_ratio'] = df['taker_sell_volume'] / df['volume'].replace(0, 1)
        df['buy_sell_imbalance'] = (df['taker_buy_base'] - df['taker_sell_volume']) / df['volume'].replace(0, 1)
        df['taker_buy_quote_ratio'] = df['taker_buy_quote'] / df['quote_volume'].replace(0, 1)
        df['quote_volume_ratio'] = df['quote_volume'] / df['quote_volume'].rolling(20).mean().replace(0, 1)
        df['avg_trade_size'] = df['volume'] / df['trades'].replace(0, 1)
        df['avg_taker_buy_size'] = df['taker_buy_base'] / df['trades'].replace(0, 1)
        
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
        
        # 10. Dynamic Volatility Features
        df['volatility_5'] = df['close'].pct_change().rolling(5).std()
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20'].replace(0, 1)
        df['volatility_regime'] = pd.cut(df['volatility_20'], bins=5, labels=[1, 2, 3, 4, 5])
        
        # 11. Price Momentum Features
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_3'] = df['close'].pct_change(3)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_acceleration'] = df['price_change_1'] - df['price_change_1'].shift(1)
        
        # 12. Market Regime Features
        df['above_sma_20'] = (df['close'] > df['sma_20']).astype(int)
        df['above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
        df['trend_strength'] = (df['close'] - df['sma_20']) / df['sma_20'].replace(0, 1) * 100
        df['trend_direction'] = np.where(df['close'] > df['sma_20'], 1, -1)
        
        # 13. Advanced Order Flow Features
        df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])
        df['taker_buy_volume_ratio'] = df['taker_buy_base'] / df['volume'].rolling(20).mean().replace(0, 1)
        df['large_trades_ratio'] = (df['trades'] > df['trades'].rolling(20).mean() * 1.5).astype(int)
        
        # 14. Market Microstructure Features
        df['liquidity_ratio'] = df['quote_volume'] / df['volume'].replace(0, 1)
        df['efficiency_ratio'] = abs(df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min()).replace(0, 1)
        
        # Remove NaN values
        df = df.dropna()
        
        # Define feature columns to match historical format
        feature_columns = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'adx', 'cci',
            'rsi', 'stoch_k', 'stoch_d', 'macd', 'awesome_oscillator',
            'bb_upper', 'bb_lower', 'bb_width', 'atr', 'kc_upper', 'kc_lower',
            'obv', 'vwap', 'volume_ratio', 'volume_obv',
            'price_range', 'body_size', 'upper_shadow', 'lower_shadow', 'price_velocity',
            'taker_buy_ratio', 'taker_sell_ratio', 'buy_sell_imbalance', 
            'taker_buy_quote_ratio', 'quote_volume_ratio', 'avg_trade_size', 'avg_taker_buy_size',
            'is_doji', 'bullish_engulfing', 'bearish_engulfing', 'is_hammer', 'is_shooting_star',
            'hilo_ratio', 'price_momentum',
            'hour', 'day_of_week', 'is_london_open', 'is_ny_open', 'is_asian_session',
            'volatility_5', 'volatility_20', 'volatility_ratio',
            'price_change_1', 'price_change_3', 'price_change_5', 'price_acceleration',
            'above_sma_20', 'above_sma_50', 'trend_strength', 'trend_direction',
            'volume_price_correlation', 'taker_buy_volume_ratio', 'large_trades_ratio',
            'liquidity_ratio', 'efficiency_ratio'
        ]
        
        # Add encoded volatility regime (one-hot encoding to match historical format)
        df = pd.get_dummies(df, columns=['volatility_regime'], prefix='vol_regime')
        volatility_regime_cols = [col for col in df.columns if col.startswith('vol_regime_')]
        feature_columns.extend(volatility_regime_cols)
        
        # Final DataFrame with all features and original price data
        final_columns = feature_columns + ['open', 'high', 'low', 'close', 'volume']
        final_df = df[final_columns]
        
        # Reset index to include datetime as a column
        final_df.reset_index(inplace=True)
        
        return final_df
    
    def get_row_hash(self, row):
        """Create a unique hash for a row to detect duplicates"""
        row_str = ''.join([str(x) for x in row.values if pd.notna(x)])
        return hashlib.md5(row_str.encode()).hexdigest()
    
    def process_existing_file(self):
        """
        Process the entire existing file and apply indicators to all candles
        """
        try:
            if not os.path.exists(self.input_file):
                print(f"Input file {self.input_file} does not exist")
                return
                
            print(f"Processing existing file: {self.input_file}")
            
            # Read the entire file
            df = pd.read_csv(self.input_file)
            
            if len(df) == 0:
                print("File is empty")
                return
                
            print(f"Found {len(df)} rows in the existing file")
            
            # Calculate indicators for the entire dataset
            result_df = self.calculate_indicators(df)
            
            # Save the entire result to the output file
            result_df.to_csv(self.output_file, index=False)
            print(f"Processed {len(result_df)} rows and saved to {self.output_file}")
            
            # Add all rows to processed_rows to avoid reprocessing
            for _, row in df.iterrows():
                row_hash = self.get_row_hash(row)
                self.processed_rows.add(row_hash)
                
            # Set the last processed time to now
            self.last_processed_time = time.time()
            
        except Exception as e:
            print(f"Error processing existing file: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def process_new_data(self):
        """
        Check for new data and process it
        """
        try:
            # Read the latest data
            if not os.path.exists(self.input_file):
                return
                
            # Get the current file modification time
            current_mod_time = os.path.getmtime(self.input_file)
            
            # Only process if the file has been modified since last check
            if current_mod_time <= self.last_processed_time:
                return
                
            self.last_processed_time = current_mod_time
            
            # Read the CSV file
            df = pd.read_csv(self.input_file)
            
            # Check if we have new data
            if len(df) == 0:
                return
                
            # Get the last row (most recent data)
            last_row = df.iloc[-1:]
            
            # Create a unique identifier for this row
            row_hash = self.get_row_hash(last_row.iloc[0])
            
            # Skip if we've already processed this row
            if row_hash in self.processed_rows:
                return
                
            # Add to processed rows
            self.processed_rows.add(row_hash)
            
            # Keep only the most recent buffer_size rows to avoid memory issues
            if len(self.processed_rows) > self.buffer_size:
                # Remove the oldest entry (FIFO)
                self.processed_rows.remove(next(iter(self.processed_rows)))
            
            # To calculate indicators properly, we need some historical context
            # So we'll use the last buffer_size rows from the file
            historical_data = df.tail(self.buffer_size).copy()
            
            # Calculate indicators
            result_df = self.calculate_indicators(historical_data)
            
            # Get only the last row (the new data with indicators)
            new_row_with_indicators = result_df.iloc[-1:]
            
            # Ensure all columns are present and in the right order
            if os.path.exists(self.output_file):
                existing_df = pd.read_csv(self.output_file, nrows=1)
                expected_columns = existing_df.columns.tolist()
                
                # Add missing columns to new row
                for col in expected_columns:
                    if col not in new_row_with_indicators.columns:
                        new_row_with_indicators[col] = np.nan
                
                # Reorder columns to match existing file
                new_row_with_indicators = new_row_with_indicators[expected_columns]
            
            # Append to output file
            if os.path.exists(self.output_file):
                new_row_with_indicators.to_csv(self.output_file, mode='a', header=False, index=False)
            else:
                new_row_with_indicators.to_csv(self.output_file, mode='w', header=True, index=False)
                
            print(f"Processed new data at {datetime.now()}")
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """
        Main loop to continuously check for new data
        """
        print("Starting live data processor...")
        print(f"Input file: {self.input_file}")
        print(f"Output file: {self.output_file}")
        
        # Process existing file if requested
        if self.process_existing:
            self.process_existing_file()
        
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.process_new_data()
                time.sleep(1)  # Check every second
        except KeyboardInterrupt:
            print("Stopping live data processor...")

def main():
    # Configuration
    input_file = "data/live/1m.csv"  # Changed to process your am.csv file
    output_file = "data/live/indicator.csv"
    
    # Create processor with process_existing=True to process all existing data
    processor = LiveDataProcessor(input_file, output_file, process_existing=True)
    
    # Run the processor
    processor.run()

if __name__ == "__main__":
    main()