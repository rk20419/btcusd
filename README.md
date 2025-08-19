BTC/USD Expert Prediction System
Overview
The BTC/USD Expert Prediction System is a self-thinking AI engine designed to predict significant price movements (300+ points, including continuations like 2000+ points) in Bitcoin (BTC) against USD on 1-minute (1m) Binance candles. It analyzes 12-value candle data, generates Buy/Sell/Neutral signals with Stop Loss (SL), Take Profit (TP), confidence scores, and average time per move. The system integrates external data (whale detection via CBLOF, news sentiment via Bi-LSTM/TAM) and uses machine learning (LSTM/Transformer), hybrid methods (Random Forest/XGBoost), backtesting, session adaptations, and logging. Built entirely with free, open-source tools and APIs for accessibility, targeting ~85% accuracy.
Purpose: Educational and paper trading; not financial advice. Cryptocurrency trading is high-risk.

Features

Flow-Based Analysis: Detects price moves using momentum, volume, and volatility/taker pressure.
Data Format: Processes 12-value candles:
timestamp (int, Unix ms)
open, high, low, close (float, prices)
volume, quote_volume, taker_buy_base, taker_buy_quote (float, volumes)
trades (int, number of trades)
volatility (float, rolling std dev of returns * 100)
datetime_ist (string, IST format YYYY-MM-DD HH:MM:SS)


External Integration: Whale detection (on-chain anomalies) and news sentiment (text trajectories).
ML Models: LSTM/Transformer for patterns, Random Forest for feature ranking, XGBoost for ensembles.
Session Adaptation: Adjusts for trading sessions (Asia: 00-08 UTC, London: 08-16 UTC, NY: 16-00 UTC).
Dynamic Metrics: Risk Score (0-100), Drift Detection (MSE-based retraining), Avg Time/Move.
Backtesting: Validates on 50,000 candles.
Logging: Detailed JSON/CSV logs with visualizations.
Free Resources: No paid APIs; uses free tiers and open-source libraries.


Prerequisites

Python 3.8+: Core language.
Libraries:
pandas, numpy: Data handling.
scikit-learn: CBLOF, Random Forest.
pytorch: LSTM/Transformer models.
xgboost: Ensemble classification.
websocket-client: Binance WebSocket streaming.
backtrader or zipline: Backtesting.
matplotlib: Visualizations.
pytz: Timezone handling.
requests: API calls.
Install via: pip install pandas numpy scikit-learn torch xgboost websocket-client backtrader matplotlib pytz requests


APIs (free tiers):
Binance API/WebSocket: Klines and live data (wss://stream.binance.com:9443).
Blockchain.com/Blockchair/CryptoQuant/CoinGecko: On-chain data.
NewsAPI.org/CryptoNews-API.com/NewsData.io/CoinDesk/Finnhub/Twinword: News/sentiment.
X API: Social media posts (within 9k words/month limit).


Data: Historical BTC/USD data from Kaggle or CryptoDataDownload.
Environment: Google Colab or AWS Free Tier for cloud deployment; local machine for development.


Installation

Clone Repository (if hosted):git clone <repository-url>
cd btc-usd-prediction-system


Install Dependencies:pip install -r requirements.txt

Sample requirements.txt:pandas
numpy
scikit-learn
torch
xgboost
websocket-client
backtrader
matplotlib
pytz
requests


Set Up Directories:
Run websocket_client.py to initialize data/live and logs directories.
Ensure write permissions for CSV/JSON outputs.


API Keys (optional for free tiers):
Configure Binance API keys (if exceeding rate limits).
Set up X API/NewsAPI.org keys in a .env file or script configuration.




Usage

Data Collection:
Run websocket_client.py to stream live 1m BTC/USDT candles:python websocket_client.py


Outputs: data/live/1m.csv (candles), data/live/live_raw_data.csv (ticks).
Logs: logs/live_stream_data.log.


Training:
Preprocess 50,000 candles (Module 1).
Train models: LSTM (Module 2), CBLOF (Module 3), Bi-LSTM/TAM (Module 4).
Fuse (Module 6) and backtest (Module 8).
Use Jupyter Notebooks for experimentation.


Live Prediction:
Stream data, process through modules, generate signals (Module 5).
Output: JSON/CSV files (signals.json, fused_decisions.json, etc.).


Visualization:
Use matplotlib for flow charts (Module 9).
Example: Plot price moves with signals.




System Architecture
The system is modular, with each module handling a specific task. All modules use the 12-value candle format and free resources.
1️⃣ Module 1 — Data Pipeline

Purpose: Organize 1m candles, normalize, create sequences (100-200 candles), detect sessions.
Input: Binance API/WebSocket data (12 values).
Process: Normalize floats, convert timestamp to datetime_ist, add whale/sentiment features, log in JSON/CSV.
Output: flow_sequences.pkl.

2️⃣ Module 2 — Price Flow Analyzer

Purpose: Detect 300+ point moves via momentum (close), volume (volume/trades), volatility/taker pressure.
Process: LSTM with Attention, session-based thresholds, backtesting.
Output: move_detections.json.

3️⃣ Module 3 — Whale Detection

Purpose: Identify large transfers using CBLOF on on-chain data.
Process: Cluster anomalies, Transformer for sequences, fuse into flow.
Output: whale_alerts.json.

4️⃣ Module 4 — News Sentiment

Purpose: Extract pos/neg/neutral sentiment from news/X posts.
Process: Bi-LSTM/TAM, fine-tune on crypto data, calculate volatility.
Output: sentiment_trajectories.json.

5️⃣ Module 5 — Signal Generator

Purpose: Generate Buy/Sell/Neutral signals.
Process: Ensemble (LSTM + XGBoost), session rules, risk score.
Output: signals.json.

6️⃣ Module 6 — Meta-Flow Fusion

Purpose: Combine all inputs for final bias.
Process: Transformer weighting, drift detection, avg time/move.
Output: fused_decisions.json.

7️⃣ Module 7 — Risk Manager

Purpose: Set SL/TP, manage exposure.
Process: Dynamic SL/TP, trailing SL, risk score.
Output: risk_params.json.

8️⃣ Module 8 — Backtest & Optimization

Purpose: Validate and tune system.
Process: Simulate on 50,000 candles, grid search for hyperparameters.
Output: backtest_report.json.

9️⃣ Module 9 — Output & Logging

Purpose: Export predictions, log data/signals.
Process: Save JSON/CSV, visualize with matplotlib.
Output: final_output.json.


Neutral Waiting Logic

If no 300+ point move and market is stable (low volatility): Monitor flow with whale/sentiment alerts.

Training vs Live Flow

Training:
Preprocess sequences (Module 1).
Train LSTM/CBLOF/Bi-LSTM (Modules 2-4).
Fuse (Module 6), backtest (Module 8).


Live:
Stream data via WebSocket.
Process through modules, output signals/logs.




Free Resources

Data:
Binance API/WebSocket: Free klines and streaming.
Blockchain.com/Blockchair/CryptoQuant/CoinGecko: On-chain data.
NewsAPI.org/CryptoNews-API.com/NewsData.io/CoinDesk/Finnhub/Twinword: News/sentiment.
X API: Social media (free tier).
Kaggle/CryptoDataDownload: Historical data.


Tools:
Python, pandas, numpy: Data processing.
scikit-learn: CBLOF/Random Forest.
PyTorch/Hugging Face: LSTM/Transformer models.
backtrader/Zipline: Backtesting.
matplotlib: Visualizations.
Jupyter Notebooks: Development.
Google Colab/AWS Free Tier: Deployment.




Limitations

Relies on free API limits (e.g., X API 9k words/month).
No internet installs in some execution environments.
Requires fine-tuning for 2025 crypto trends.
Designed for educational/paper trading; real trading risks capital.


Future Enhancements

Add a user interface (e.g., Flask/Dash) for manual overrides.
Expand to other crypto pairs (e.g., ETH/USD).
Integrate more data sources if free tiers expand.
Optimize for lower latency in live mode.


Contributing

Fork the repository, submit pull requests.
Report issues via GitHub Issues (if hosted).
Suggest improvements for models or data sources.


License
MIT License – free to use, modify, and distribute. See LICENSE file for details.

Contact
For questions or feedback, reach out via GitHub or X (@your_handle). Built for educational purposes by a crypto enthusiast.