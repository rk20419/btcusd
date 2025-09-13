import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
import os
import json
import logging
from datetime import datetime

# Set up comprehensive logging
def setup_logging():
    """Setup comprehensive logging for the training process"""
    os.makedirs('models', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler
    log_file = f"models/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Automatic data type detection and conversion
def ensure_consistent_dtypes(X, y_buy, y_sell):
    """Ensure consistent data types across all arrays"""
    # Detect the appropriate float type based on TensorFlow's backend
    float_type = tf.keras.backend.floatx()
    
    logging.info(f"Using float type: {float_type}")
    
    # Convert all arrays to the detected float type
    X = X.astype(float_type)
    y_buy = y_buy.astype(float_type)
    y_sell = y_sell.astype(float_type)
    
    return X, y_buy, y_sell

def create_dynamic_targets(df, future_window=5, volatility_multiplier=2.0):
    """
    Create dynamic targets based on future price movement and volatility
    """
    logging.info("Creating dynamic targets...")
    
    # Calculate future price extremes
    df['future_high'] = df['high'].rolling(future_window).max().shift(-future_window)
    df['future_low'] = df['low'].rolling(future_window).min().shift(-future_window)
    
    # Calculate dynamic threshold based on volatility
    df['volatility_threshold'] = df['atr'] * volatility_multiplier
    
    # Create buy signal (price goes up by at least volatility threshold)
    df['target_buy'] = ((df['future_high'] - df['close']) >= df['volatility_threshold']).astype(int)
    
    # Create sell signal (price goes down by at least volatility threshold)
    df['target_sell'] = ((df['close'] - df['future_low']) >= df['volatility_threshold']).astype(int)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df

def create_sequences(X, y_buy, y_sell, time_steps=60):
    """
    Create sequences for training
    """
    logging.info(f"Creating sequences with time steps: {time_steps}")
    
    Xs, y_buys, y_sells = [], [], []
    
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        y_buys.append(y_buy[i + time_steps])
        y_sells.append(y_sell[i + time_steps])
    
    return np.array(Xs), np.array(y_buys), np.array(y_sells)

def create_advanced_model(input_shape):
    """
    Create advanced LSTM model with attention mechanism
    """
    logging.info("Creating advanced model architecture...")
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    lstm1 = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
                kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    batch_norm1 = BatchNormalization()(lstm1)
    
    # Second LSTM layer
    lstm2 = LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
                kernel_regularizer=tf.keras.regularizers.l2(0.001))(batch_norm1)
    batch_norm2 = BatchNormalization()(lstm2)
    
    # Attention mechanism
    attention = Attention()([batch_norm2, batch_norm2])
    
    # Third LSTM layer
    lstm3 = LSTM(32, dropout=0.3, 
                kernel_regularizer=tf.keras.regularizers.l2(0.001))(attention)
    batch_norm3 = BatchNormalization()(lstm3)
    
    # Dense layers
    dense1 = Dense(64, activation='relu', 
                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(batch_norm3)
    dropout1 = Dropout(0.4)(dense1)
    
    dense2 = Dense(32, activation='relu', 
                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(dropout1)
    dropout2 = Dropout(0.4)(dense2)
    
    # Output layers
    buy_output = Dense(1, activation='sigmoid', name='buy')(dropout2)
    sell_output = Dense(1, activation='sigmoid', name='sell')(dropout2)
    
    return Model(inputs=inputs, outputs=[buy_output, sell_output])

def main():
    # Setup logging
    logger = setup_logging()
    logging.info("Starting model training...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load data with indicators
    logging.info("Loading data with indicators...")
    df = pd.read_csv("data/historical/BTCUSDT_1m_200000_with_indicators.csv", index_col='datetime', parse_dates=True)
    
    # Log dataset information
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Dataset columns: {list(df.columns)}")
    
    # Define feature columns (from the indicator calculation)
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
    
    # Add volatility regime columns
    volatility_regime_cols = [col for col in df.columns if col.startswith('vol_regime_')]
    feature_columns.extend(volatility_regime_cols)
    
    # Log feature information
    logging.info(f"Total features: {len(feature_columns)}")
    logging.info(f"Feature list: {feature_columns}")
    
    # Create targets
    df = create_dynamic_targets(df)
    df = df.dropna()
    
    # Check class distribution
    buy_signals = df['target_buy'].sum()
    sell_signals = df['target_sell'].sum()
    total_samples = len(df)
    
    logging.info(f"Buy signals: {buy_signals} ({buy_signals/total_samples*100:.2f}%)")
    logging.info(f"Sell signals: {sell_signals} ({sell_signals/total_samples*100:.2f}%)")
    logging.info(f"Total samples: {total_samples}")
    
    # Prepare features and targets
    X = df[feature_columns].values
    y_buy = df['target_buy'].values
    y_sell = df['target_sell'].values
    
    # Ensure consistent data types
    X, y_buy, y_sell = ensure_consistent_dtypes(X, y_buy, y_sell)
    
    # Normalize features
    logging.info("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to consistent data type after scaling
    float_type = tf.keras.backend.floatx()
    X_scaled = X_scaled.astype(float_type)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    logging.info("Scaler saved successfully")
    
    # Create sequences
    time_steps = 60
    X_seq, y_buy_seq, y_sell_seq = create_sequences(X_scaled, y_buy, y_sell, time_steps)
    
    logging.info(f"Sequences created: {X_seq.shape}")
    
    # Train-test split (time-series aware)
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_buy_train, y_buy_test = y_buy_seq[:split_idx], y_buy_seq[split_idx:]
    y_sell_train, y_sell_test = y_sell_seq[:split_idx], y_sell_seq[split_idx:]
    
    logging.info(f"Training data: {X_train.shape}")
    logging.info(f"Test data: {X_test.shape}")
    
    # Calculate class weights
    logging.info("Calculating class weights...")
    buy_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_buy_train)
    sell_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_sell_train)
    
    # Convert to appropriate float type
    buy_weights = buy_weights.astype(float_type)
    sell_weights = sell_weights.astype(float_type)
    
    logging.info(f"Buy weights: {buy_weights}")
    logging.info(f"Sell weights: {sell_weights}")
    
    # Create model
    model = create_advanced_model((X_train.shape[1], X_train.shape[2]))
    
    # Log model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    logging.info("Model architecture:\n" + "\n".join(model_summary))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'buy': 'binary_crossentropy', 'sell': 'binary_crossentropy'},
        metrics={'buy': ['accuracy', 'precision', 'recall', 'auc'], 
                 'sell': ['accuracy', 'precision', 'recall', 'auc']},
        loss_weights={'buy': float(buy_weights[1]), 'sell': float(sell_weights[1])}
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1),
        ModelCheckpoint('models/best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # Train model
    logging.info("Training model...")
    history = model.fit(
        X_train,
        {'buy': y_buy_train, 'sell': y_sell_train},
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
        shuffle=False
    )
    
    # Save final model
    model.save('models/final_model.h5')
    logging.info("Model saved successfully!")
    
    # Evaluate model
    logging.info("Evaluating model...")
    test_results = model.evaluate(X_test, {'buy': y_buy_test, 'sell': y_sell_test}, verbose=0)
    
    # Print results
    logging.info("\n=== TEST RESULTS ===")
    metric_names = model.metrics_names
    for i, name in enumerate(metric_names):
        logging.info(f"{name}: {test_results[i]:.4f}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('models/training_history.csv', index=False)
    logging.info("Training history saved")
    
    # Save model configuration
    config = {
        'feature_columns': feature_columns,
        'input_shape': (X_train.shape[1], X_train.shape[2]),
        'time_steps': time_steps,
        'class_weights': {
            'buy': buy_weights.tolist(),
            'sell': sell_weights.tolist()
        },
        'test_results': dict(zip(metric_names, [float(r) for r in test_results])),
        'dataset_info': {
            'total_samples': total_samples,
            'buy_signals': int(buy_signals),
            'sell_signals': int(sell_signals),
            'buy_percentage': float(buy_signals/total_samples*100),
            'sell_percentage': float(sell_signals/total_samples*100)
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('models/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("Model configuration saved")
    
    # Plot training history
    plt.figure(figsize=(20, 12))
    
    # Accuracy plots
    plt.subplot(2, 3, 1)
    plt.plot(history.history['buy_accuracy'], label='Buy Train Accuracy')
    plt.plot(history.history['val_buy_accuracy'], label='Buy Val Accuracy')
    plt.title('Buy Accuracy')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['sell_accuracy'], label='Sell Train Accuracy')
    plt.plot(history.history['val_sell_accuracy'], label='Sell Val Accuracy')
    plt.title('Sell Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(2, 3, 3)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    # Precision plots
    plt.subplot(2, 3, 4)
    plt.plot(history.history['buy_precision'], label='Buy Train Precision')
    plt.plot(history.history['val_buy_precision'], label='Buy Val Precision')
    plt.title('Buy Precision')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(history.history['sell_precision'], label='Sell Train Precision')
    plt.plot(history.history['val_sell_precision'], label='Sell Val Precision')
    plt.title('Sell Precision')
    plt.legend()
    
    # Learning rate plot
    plt.subplot(2, 3, 6)
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    
    logging.info("Training visualization saved")
    
    # Create comprehensive performance report
    report = f"""
    === MODEL TRAINING COMPLETE ===
    Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    === DATASET INFORMATION ===
    Total Samples: {total_samples}
    Buy Signals: {buy_signals} ({buy_signals/total_samples*100:.2f}%)
    Sell Signals: {sell_signals} ({sell_signals/total_samples*100:.2f}%)
    Features Used: {len(feature_columns)}
    Sequence Length: {time_steps}
    
    === MODEL ARCHITECTURE ===
    Input Shape: {X_train.shape[1:]}
    Layers: 3 LSTM layers with Attention mechanism
    Regularization: Dropout, L2 regularization, Batch Normalization
    
    === TRAINING RESULTS ===
    Final Validation Loss: {history.history['val_loss'][-1]:.4f}
    Final Buy Accuracy: {history.history['val_buy_accuracy'][-1]:.4f}
    Final Sell Accuracy: {history.history['val_sell_accuracy'][-1]:.4f}
    
    === TEST RESULTS ===
    """
    
    for i, name in enumerate(metric_names):
        report += f"{name}: {test_results[i]:.4f}\n"
    
    # Save report
    with open('models/training_report.txt', 'w') as f:
        f.write(report)
    
    logging.info(report)
    logging.info("Training completed successfully!")
    logging.info("Model files saved in 'models' directory")

if __name__ == "__main__":
    main()