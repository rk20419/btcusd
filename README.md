BTC/USD Multi-Timeframe Thinker Prediction System: Comprehensive In-Depth Summary
1. System Introduction
The BTC/USD Multi-Timeframe Thinker Prediction System is an unsupervised, real-time framework that emulates expert trader decision-making to predict 300+ point price moves ($115,000–$125,000 in 2–4 hours) with ≥70% accuracy. Built on a modular directory structure (BTC/), it uses historical and live data in standardized formats:

OHLCV Data: timestamp (milliseconds), open, high, low, close, volume.
Tick Data: timestamp, Price, Volume (live raw data).
News Data: timestamp, Text.

Key techniques include sliding windows (20–200 candles, KMeans-clustered), volatility transitions, hierarchical TF integration (1m flows embedded into higher TFs), news sentiment refinement (FinBERT-tone ~0.85 for ETF inflows), and fakeout detection (volume persistence >3 ticks, wick/body ratios >1.5, sentiment mismatches). The system is optimized for Windows (MinGW), uses cache_system.py for real-time access, eliminates CSV crawling, and provides cache_view.csv for monitoring. Outputs are JSON signals saved to predictions_20250813.parquet (updated for current date August 13, 2025).
1.1 Market Context (August 13, 2025)

Price Range: BTC/USD ~$118,000–$122,321, with bullish ETF inflows and Fed rate cut optimism pushing toward $123,200–$128,000 (sources: CryptoRank, Investing.com).
Risks: Double top rejection could pull back to $110,000–$114,000 (Seeking Alpha, Cointelegraph).
Sentiment: FinBERT-tone ~0.85 positive for ETF approvals (2.5 impact score), driving 300+ point moves; negative sentiment (e.g., cycle breaking, CNBC) signals fakeouts.
Volatility: High-volume spikes (e.g., 55.17505 for 1m at timestamp=1100603) indicate breakouts, validated by 1m flows.

1.2 Core Requirements

Multi-TF Predictions: Independent models per TF, fused with 1m embedded in higher TFs.
Dynamic Calculations: Windows (20–200), thresholds, avg_move_time via KMeans/HMM.
300+ Point Moves: Non-retracing, ≥3 USD profit on 0.01 lots (100 points = 1 USD).
TP/SL Logic: TP1 (100-point centroid), TP2 (200–500 points), SL (swing low + 1.2x 1m ATR), sentiment-adjusted.
Fakeout Detection: Volume persistence (>3 ticks), wick/body ratios (>1.5), mismatches.
Accuracy: ≥70% via backtesting on historical data, +10% with sentiment.
Data Handling: Historical/live CSVs, atomic writes to avoid errors.
Sentiment: FinBERT-tone boosts 4h/1d accuracy.
Output: JSON (signal, confidence, tp1, tp2, sl, avg_move_points, avg_move_time, reasons, fakeout_risk).
Unsupervised Models: CNN-LSTM autoencoders, KMeans, HMM, zero-shot FinBERT.
Real-Time: Tick-by-tick, <50ms latency.
Workflow: Pipeline → Analysis (Modules 2–5) → Fakeout → Fusion → Confirmation → Risk → Drift → Output.

1.3 Data Specifications
Both historical and live datasets have identical candle counts per TF:

1m: 25,000 candles (~17.36 days).
3m: 15,000 candles (~31.25 days).
5m: 15,000 candles (~52.08 days).
15m: 5,000 candles (~52.08 days).
30m: 5,000 candles (~104.17 days).
1h: 5,000 candles (~208.33 days).
4h: 5,000 candles (~2.28 years).
1d: 5,000 candles (~13.7 years).
Live Updates: Data appended in real-time via websocket_client.py; raw tick data streamed to live_raw_data.csv.
Sufficiency: Supports training (historical) and prediction (live), with ~17-day overlap for 1m.

2. Directory Structure

Data:
Historical: data/historical/{tf}.csv.
Live: data/live/{tf}.csv, live_raw_data.csv, news_data.csv, cache_view.csv, predictions_20250813.parquet, archive/ticks_YYYYMMDD.csv.


Utils: cache_system.py, helpers.py, feature_engineering.py, logging_config.py, constants.py, logs/.
Stream: websocket_client.py, tick_processor.py, queue_manager.py.
Modules: Folders 1–10 with scripts, logs, artifacts.
Config: config.yaml, requirements.txt.

3. Modules
Module 1: Data Pipeline

File: module1/data_pipeline.py.
Execution Flow:
Load from cache["historical"] (training) or cache["live"] (prediction).
Resample 1m to higher TFs.
Compute features via feature_engineering.py.
Handle gaps/interpolation via helpers.py.
Merge live ticks, buffer for TF completion.


Models: KMeans for windows.
Outputs: Multi-TF DataFrames with features.
Logging: utils/logs/data_pipeline.log.
Artifacts: Scalers (scaler_{tf}.pkl).

Module 2: Price Action Model

File: module2/price_action.py.
Execution Flow:
Input hierarchical windows from cache.
Train CNN-LSTM on historical; infer on live.
Cluster embeddings for direction; calculate TP/SL.


Models: CNN-LSTM, KMeans (3 clusters: UP/DOWN/NEUTRAL).
Outputs: Signal with confidence, TP/SL, avg_move_points/time.
Logging: data/live/module{tf}_predictions.log.
Artifacts: autoenc_{tf}.h5, kmeans_patterns.pkl.

Module 3: Sentiment Analyzer

File: module3/sentiment_analyzer.py.
Execution Flow:
Input last 10–20 news texts.
Score sentiment with FinBERT; cluster events with GPT-2.
Check fakeouts using raw ticks.


Models: FinBERT-tone, GPT-2 (KMeans).
Outputs: Sentiment score, event, impact, avg_move_time.
Logging: utils/logs/sentiment_analyzer.log.
Artifacts: kmeans_events.pkl.

Module 4: Session Behavior Model

File: module4/session_behavior.py.
Execution Flow:
Input 4–8h windows.
Train/infer LSTM for BREAKOUT/RANGE.


Models: Time-Aware LSTM, KMeans.
Outputs: Session type, confidence, avg_move_time.
Logging: utils/logs/session_behavior.log.
Artifacts: lstm_session.h5.

Module 5: Regime Detector

File: module5/regime_detector.py.
Execution Flow:
Input volatility/returns.
Train/infer HMM for regimes.


Models: GaussianHMM, KMeans.
Outputs: Regime, confidence, avg_move_time.
Logging: utils/logs/regime_detector.log.
Artifacts: hmm_regime.pkl.

Module 6: Meta-Brain

File: module6/meta_brain.py.
Execution Flow:
Fuse outputs from Modules 2–5.
Apply attention, penalize conflicting TFs.


Models: Attention layer, KMeans.
Outputs: Fused signal with TP/SL, avg_move_points/time.
Logging: utils/logs/meta_brain.log.
Artifacts: attention_weights.pkl.

Module 7: Confirmation Filter

File: module7/confirmation_filter.py.
Execution Flow:
Validate fused signals (ATR ≥300, TF agreement ≥70%, stability >3 ticks).


Outputs: Approved or rejected signal.
Logging: utils/logs/confirmation_filter.log.
Artifacts: thresholds.pkl.

Module 8: Risk Advisory

File: module8/risk_advisory.py.
Execution Flow:
Calculate TP/SL/sizing for 0.01 lots.


Outputs: Risk-integrated signal.
Logging: utils/logs/risk_advisory.log.
Artifacts: risk_params.pkl.

Module 9: Drift Detection

File: module9/drift_detection.py.
Execution Flow:
Monitor 20–30 signals; flag if accuracy drops >10%.


Outputs: Status, accuracy.
Logging: utils/logs/drift_detection.log.
Artifacts: drift_metrics.pkl.

Module 10: Output & Logging

File: module10/output_logging.py.

Execution Flow:

Compile and save JSON predictions.


Outputs: predictions_20250813.parquet.

Logging: utils/logs/output_logging.log.

Artifacts: output_schema.pkl.

Sample Output:
{
  "timestamp": "2025-08-13T03:46:56.000Z",
  "tf": "fused",
  "signal": "BUY",
  "confidence": 91,
  "tp1": 122500.0,
  "tp2": 123000.0,
  "sl": 121500.0,
  "avg_move_points": 450,
  "avg_move_time": "2h",
  "reasons": ["Stable 1m flow", "TF-aligned", "ETF sentiment: 0.85"],
  "fakeout_risk": "Cleared",
  "module_stats": {
    "1m": {"signal": "BUY", "conf": 85},
    "5m": {"signal": "BUY", "conf": 78},
    "sentiment": {"score": 0.85, "event": "ETF_APPROVAL"}
  }
}



Utilities
Cache System (utils/cache_system.py)

Functionality: Caches live data (cache["live"][tf], cache["raw_ticks"]), updates every 1 second, writes cache_view.csv.
Gap Handling: Detects >1.2x TF interval gaps, logs during market hours.
Integration: Modules access cache, eliminating CSV reads; fixes KeyError: 'Timestamp'.
Logging: utils/logs/cache_system.log.

Constants (utils/constants.py)

Defines: TIMEFRAMES, TF_INTERVALS, FEATURES, WINDOW_SIZES, DATA_DIR, RAW_TICK_FILE, MAX_GAP_MS=5000, MIN_300_POINTS=300, MIN_CONFIDENCE=70, ATR_MULTIPLIER=1.2, STABILITY_WINDOW=3, EPOCHS=100, BATCH_SIZE=32.

Feature Engineering (utils/feature_engineering.py)

Computes: RSI (14), MACD (12,26,9), ATR (14), vol_delta, cumulative_delta, whale_flag (KMeans), stability_score (3 ticks).
Logging: utils/logs/feature_engineering.log.

Helpers (utils/helpers.py)

Functions: standardize_timestamps, interpolate_gaps (<5s forward-fill), align_timeframes (~17 days), fallback_to_lower_tf (3m/5m for 1m).
Logging: utils/logs/helpers.log.

Logging Config (utils/logging_config.py)

Sets up loggers for modules/utilities, per-TF for price action.
Format: %(asctime)s - %(name)s - %(levelname)s - %(message)s.

WebSocket Client (stream/websocket_client.py)

Streams BTC/USDT kline (OHLCV) and tick data from Binance WebSocket.
Saves: Completed candles to data/live/{tf}.csv, ticks to data/live/live_raw_data.csv.
Logging: logs/live_stream_data.log.

Data Flow

Historical (Training): Load data/historical/{tf}.csv (25,000 1m, etc.), preprocess (helpers.py), compute features (feature_engineering.py), train models (Modules 2, 4, 5).
Live (Prediction): Stream via websocket_client.py, cache (cache_system.py), preprocess, predict (Modules 2–10), save to predictions_20250813.parquet.
Real-Time Updates: Live data appended; cache updated every 1s; cache_view.csv for monitoring.

Improvements & Tweaks

Latency: Use asyncio.Queue in queue_manager.py.
Alerts: Drift window to 20 signals; pause if accuracy <60%.
Features: Add funding rates/open interest.
Penalty: Reject if 1m opposes higher TFs (≥70% conf).
Archiving: Daily ticks to archive/ticks_YYYYMMDD.csv.

Execution Steps

Run websocket_client.py for live data.
Run cache_system.py for caching.
Train models with historical data.
Generate predictions with live data.
Monitor logs and cache_view.csv.

The system is ready for deployment. Share issues or logs for further assistance!