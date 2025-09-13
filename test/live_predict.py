import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os
import logging
from datetime import datetime
import json
import csv

class RealTimePredictor:
    def __init__(self, model_path, scaler_path, config_path, live_data_path, output_path):
        """
        Initialize the real-time predictor
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the scaler used during training
            config_path: Path to the model configuration file
            live_data_path: Path to the live indicator data CSV
            output_path: Path to save predictions
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        self.live_data_path = live_data_path
        self.output_path = output_path
        self.model = None
        self.scaler = None
        self.config = None
        self.feature_columns = None
        self.time_steps = None
        self.data_buffer = None
        self.last_processed_time = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Load model, scaler, and config
        self.load_resources()
        
    def setup_logging(self):
        """Setup logging for the prediction system"""
        self.logger = logging.getLogger('RealTimePredictor')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        log_file = f"logs/prediction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def load_resources(self):
        """Load the trained model, scaler, and configuration"""
        try:
            self.logger.info("Loading model resources...")
            
            # Load model
            self.model = load_model(self.model_path)
            self.logger.info(f"Model loaded from {self.model_path}")
            
            # Load scaler
            self.scaler = joblib.load(self.scaler_path)
            self.logger.info(f"Scaler loaded from {self.scaler_path}")
            
            # Load config
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            self.feature_columns = self.config['feature_columns']
            self.time_steps = self.config['time_steps']
            self.logger.info(f"Configuration loaded: {len(self.feature_columns)} features, {self.time_steps} time steps")
            
        except Exception as e:
            self.logger.error(f"Error loading resources: {str(e)}")
            raise
            
    def initialize_prediction_file(self):
        """Initialize the prediction output file with headers"""
        headers = [
            'datetime', 'prediction_buy', 'prediction_sell', 
            'buy_confidence', 'sell_confidence', 'signal_strength',
            'current_price', 'price_change_1', 'price_change_5',
            'timestamp'
        ]
        
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
        self.logger.info(f"Prediction file initialized at {self.output_path}")
        
    def load_live_data(self):
        """Load the latest live data from CSV"""
        try:
            df = pd.read_csv(self.live_data_path)
            
            # Convert datetime if needed
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading live data: {str(e)}")
            return pd.DataFrame()
            
    def prepare_prediction_data(self, df):
        """
        Prepare the data for prediction by selecting the required features
        and creating sequences of the required length
        """
        if len(df) < self.time_steps:
            self.logger.warning(f"Insufficient data for prediction. Need {self.time_steps} rows, have {len(df)}")
            return None
            
        # Select the most recent time_steps rows
        recent_data = df.tail(self.time_steps).copy()
        
        # Ensure we have all required features
        missing_features = set(self.feature_columns) - set(recent_data.columns)
        if missing_features:
            self.logger.warning(f"Missing features in live data: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                recent_data[feature] = 0
        
        # Select and order features exactly as during training
        X = recent_data[self.feature_columns].values
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Reshape for LSTM input (samples, time_steps, features)
        X_reshaped = X_scaled.reshape(1, self.time_steps, len(self.feature_columns))
        
        return X_reshaped, recent_data
        
    def make_prediction(self, X):
        """Make prediction using the loaded model"""
        try:
            predictions = self.model.predict(X, verbose=0)
            buy_pred = predictions[0][0][0]
            sell_pred = predictions[1][0][0]
            
            return buy_pred, sell_pred
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return 0, 0
            
    def calculate_confidence_metrics(self, buy_pred, sell_pred, recent_data):
        """Calculate additional confidence metrics for the prediction"""
        # Calculate signal strength (difference between buy and sell probabilities)
        signal_strength = buy_pred - sell_pred
        
        # Get current price and recent price changes
        current_price = recent_data['close'].iloc[-1] if 'close' in recent_data.columns else 0
        price_change_1 = recent_data['price_change_1'].iloc[-1] if 'price_change_1' in recent_data.columns else 0
        price_change_5 = recent_data['price_change_5'].iloc[-1] if 'price_change_5' in recent_data.columns else 0
        
        return signal_strength, current_price, price_change_1, price_change_5
        
    def save_prediction(self, datetime_str, buy_pred, sell_pred, signal_strength, 
                       current_price, price_change_1, price_change_5):
        """Save the prediction to the output CSV file"""
        # Calculate confidence levels
        buy_confidence = "HIGH" if buy_pred > 0.7 else "MEDIUM" if buy_pred > 0.5 else "LOW"
        sell_confidence = "HIGH" if sell_pred > 0.7 else "MEDIUM" if sell_pred > 0.5 else "LOW"
        
        # Prepare data row
        row = [
            datetime_str, 
            float(buy_pred), 
            float(sell_pred),
            buy_confidence,
            sell_confidence,
            float(signal_strength),
            float(current_price),
            float(price_change_1),
            float(price_change_5),
            int(time.time() * 1000)  # Current timestamp in milliseconds
        ]
        
        # Append to CSV
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        self.logger.info(f"Prediction saved: Buy={buy_pred:.4f}, Sell={sell_pred:.4f}, Signal={signal_strength:.4f}")
        
    def process_new_data(self):
        """Check for new data and process it for prediction"""
        try:
            # Check if file has been modified
            current_mod_time = os.path.getmtime(self.live_data_path)
            
            if current_mod_time <= self.last_processed_time:
                return False
                
            self.last_processed_time = current_mod_time
            
            # Load the latest data
            df = self.load_live_data()
            if df.empty:
                return False
                
            # Prepare data for prediction
            result = self.prepare_prediction_data(df)
            if result is None:
                return False
                
            X, recent_data = result
            
            # Make prediction
            buy_pred, sell_pred = self.make_prediction(X)
            
            # Calculate additional metrics
            signal_strength, current_price, price_change_1, price_change_5 = self.calculate_confidence_metrics(
                buy_pred, sell_pred, recent_data
            )
            
            # Get the datetime of the most recent data point
            if hasattr(recent_data.index, 'values'):
                latest_datetime = recent_data.index.values[-1]
                if isinstance(latest_datetime, np.datetime64):
                    datetime_str = pd.to_datetime(latest_datetime).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    datetime_str = str(latest_datetime)
            else:
                datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save prediction
            self.save_prediction(
                datetime_str, buy_pred, sell_pred, signal_strength,
                current_price, price_change_1, price_change_5
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing new data: {str(e)}")
            return False
            
    def run(self, check_interval=1):
        """
        Run the real-time prediction system
        
        Args:
            check_interval: Time in seconds between checks for new data
        """
        self.logger.info("Starting real-time prediction system...")
        self.logger.info(f"Monitoring: {self.live_data_path}")
        self.logger.info(f"Output: {self.output_path}")
        self.logger.info("Press Ctrl+C to stop")
        
        # Initialize prediction file
        self.initialize_prediction_file()
        
        try:
            while True:
                self.process_new_data()
                time.sleep(check_interval)
        except KeyboardInterrupt:
            self.logger.info("Prediction system stopped by user")
        except Exception as e:
            self.logger.error(f"Error in prediction system: {str(e)}")

def main():
    # Configuration
    model_path = "models/best_model.h5"  # or "models/final_model.h5"
    scaler_path = "models/scaler.pkl"
    config_path = "models/model_config.json"
    live_data_path = "data/live/indicator.csv"
    output_path = "data/live/predictions.csv"
    
    # Create predictor
    predictor = RealTimePredictor(
        model_path=model_path,
        scaler_path=scaler_path,
        config_path=config_path,
        live_data_path=live_data_path,
        output_path=output_path
    )
    
    # Start prediction system
    predictor.run(check_interval=1)  # Check every second

if __name__ == "__main__":
    main()