import os
import time
import json
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import joblib

# Setup logging
os.makedirs("logs", exist_ok=True)
log_path = "logs/module2_live_predictor.logs"

# Create a custom logger
logger = logging.getLogger('Module2LivePredictor')
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Define LSTM model (identical to training code)
class PriceFlowLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, output_size=3):
        super(PriceFlowLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Load scaler from training
def load_scaler(filepath="models/module2_scaler.save"):
    try:
        scaler = joblib.load(filepath)
        logger.info(f"Loaded scaler from {filepath}")
        return scaler
    except FileNotFoundError:
        logger.error(f"Scaler file {filepath} not found. Please train the model first.")
        raise

# Preprocess new candle data
def preprocess_candle(df, seq_len=100, scaler=None):
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'trades', 'taker_buy_base', 'taker_buy_quote', 'volatility']
    df = df[feature_cols].ffill().fillna(0)
    
    if scaler is None:
        logger.warning("No scaler provided, creating new one (not recommended for live prediction)")
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, "models/module2_scaler.save")
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    seq = df[feature_cols].tail(seq_len).values
    if len(seq) < seq_len:
        logger.warning(f"Insufficient data: {len(seq)} candles, padding with zeros")
        seq = np.pad(seq, ((seq_len - len(seq), 0), (0, 0)), mode='constant')
    
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0), scaler

# Predict on new candle
def predict(model, seq, device):
    model.eval()
    with torch.no_grad():
        seq = seq.to(device)
        output = model(seq)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    return pred, confidence

# Save prediction to JSON
def save_prediction(pred, confidence, timestamp, datetime_ist):
    output = {
        'timestamp': int(timestamp),
        'datetime_ist': str(datetime_ist),
        'prediction': ['Buy', 'Sell', 'Neutral'][pred],
        'confidence': confidence
    }
    output_file = "data/move_detections.json"
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                predictions = json.load(f)
        else:
            predictions = []
        predictions.append(output)
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Saved prediction: {output}")
    except Exception as e:
        logger.error(f"Failed to save prediction: {str(e)}")

# Monitor CSV for new candles
def monitor_csv(csv_path, model, scaler, device, seq_len=100, poll_interval=5):
    last_row_count = 0
    logger.info(f"Monitoring {csv_path} for new candles...")
    
    while True:
        try:
            if not os.path.exists(csv_path):
                logger.warning(f"CSV file {csv_path} not found, waiting...")
                time.sleep(poll_interval)
                continue
            
            df = pd.read_csv(csv_path)
            current_row_count = len(df)
            
            if current_row_count > last_row_count:
                logger.info(f"Detected {current_row_count - last_row_count} new candles")
                seq, scaler = preprocess_candle(df, seq_len, scaler)
                pred, confidence = predict(model, seq, device)
                timestamp = df['timestamp'].iloc[-1]
                datetime_ist = df['datetime_ist'].iloc[-1]
                save_prediction(pred, confidence, timestamp, datetime_ist)
                last_row_count = current_row_count
            else:
                logger.debug("No new candles detected")
            
            time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            time.sleep(poll_interval)

def main():
    # Configuration
    csv_path = "data/live/1m.csv"
    model_path = "models/module2_model.pth"
    scaler_path = "models/module2_scaler.save"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 100  # Match training sequence length
    
    # Load model
    model = PriceFlowLSTM(input_size=10, output_size=3).to(device)
    try:
        model.load_state_dict(torch.load(model_path))
        logger.info(f"Loaded model from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file {model_path} not found. Train the model first.")
        return
    
    # Load scaler
    scaler = load_scaler(scaler_path)
    
    # Start monitoring
    monitor_csv(csv_path, model, scaler, device, seq_len)

if __name__ == "__main__":
    main()