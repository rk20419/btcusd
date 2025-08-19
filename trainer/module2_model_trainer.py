# trainer/module2_model_trainer.py

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib  # For saving scaler

# Setup logging
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
log_path = "logs/module2_training.logs"
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


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


def load_and_preprocess_data(filepath: str, seq_len: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV data, normalize features, generate sequences, and label them.

    Labeling logic:
    - Buy (0): Close diff >= 300 or volume spike + high taker pressure + volatility
    - Sell (1): Close diff <= -300 or volume spike + low taker pressure + volatility
    - Neutral (2): Otherwise
    """
    df = pd.read_csv(filepath)

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'trades', 'taker_buy_base', 'taker_buy_quote', 'volatility']

    # Forward fill missing data, then fill any remaining with 0
    df[feature_cols] = df[feature_cols].ffill().fillna(0)

    # Normalize features between 0 and 1
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save scaler for live prediction use
    joblib.dump(scaler, "models/module2_scaler.save")

    features = df[feature_cols].values
    close_prices = df['close'].values
    volumes = df['volume'].values
    trades = df['trades'].values
    taker_buy_base = df['taker_buy_base'].values
    volatility = df['volatility'].values

    avg_volume = volumes.mean()
    avg_trades = trades.mean()
    avg_volatility = volatility.mean()

    X, y = [], []
    for i in range(len(df) - seq_len - 1):
        seq = features[i:i + seq_len]
        future_close = close_prices[i + seq_len]
        current_close = close_prices[i + seq_len - 1]
        price_diff = (future_close - current_close) / scaler.scale_[3]  # Un-normalize close diff approx

        current_volume = volumes[i + seq_len - 1]
        current_trades = trades[i + seq_len - 1]
        volume_spike = current_volume > 2 * avg_volume
        trade_spike = current_trades > 2 * avg_trades
        current_volatility = volatility[i + seq_len - 1]
        taker_pressure = taker_buy_base[i + seq_len - 1] / (current_volume + 1e-10)
        high_volatility = current_volatility > avg_volatility

        if price_diff >= 300 or (volume_spike and taker_pressure > 0.5 and high_volatility):
            label = 0  # Buy
        elif price_diff <= -300 or (volume_spike and taker_pressure < 0.5 and high_volatility):
            label = 1  # Sell
        else:
            label = 2  # Neutral

        X.append(seq)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def prepare_dataloaders(X: np.ndarray, y: np.ndarray, batch_size=32, test_size=0.2, seed=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_loop(model, train_loader, val_loader, device, epochs=30, lr=0.001):
    y_train = np.concatenate([y_batch.numpy() for _, y_batch in train_loader])
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    patience = 5
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        all_preds, all_labels = [], []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        logging.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/module2_model.pth")
            logging.info(f"Saved best model with val accuracy {best_val_acc:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Training completed. Best Validation Accuracy: {best_val_acc:.4f}")
    logging.info(f"Training completed. Best Validation Accuracy: {best_val_acc:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    # Adjust filepath as needed
    X, y = load_and_preprocess_data("data/historical/BTCUSDT_1m_5000.csv", seq_len=100)
    print(f"Data shape X: {X.shape}, y: {y.shape}")
    logging.info(f"Loaded data X shape: {X.shape}, y shape: {y.shape}")

    train_loader, val_loader = prepare_dataloaders(X, y)
    model = PriceFlowLSTM(input_size=10, output_size=3).to(device)

    train_loop(model, train_loader, val_loader, device, epochs=30, lr=0.001)


if __name__ == "__main__":
    main()
