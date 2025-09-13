# analysis/trading_signal.py
import pandas as pd
import numpy as np
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
from datetime import datetime
from collections import deque

# ================= Configuration =================
@dataclass
class AnalysisConfig:
    DATA_DIR: str = "data/live"
    SYMBOL: str = "btcusdt"
    TIMEFRAME: str = "1m"  # Primary timeframe for analysis
    SMA_SHORT: int = 10    # Short-term SMA period
    SMA_LONG: int = 30     # Long-term SMA period
    RSI_PERIOD: int = 14   # RSI period
    RSI_OVERBOUGHT: int = 70  # RSI overbought level
    RSI_OVERSOLD: int = 30    # RSI oversold level
    VOLUME_SPIKE: float = 2.0  # Volume spike multiplier
    CHECK_INTERVAL: int = 5   # Check for new data every N seconds
    SIGNAL_CONFIRMATION_COUNT: int = 3  # Number of consecutive signals needed for confirmation
    TREND_CONFIRMATION_PERIOD: int = 20  # Period for trend confirmation
    MIN_SIGNAL_DURATION: int = 60  # Minimum time in seconds before signal can change

config = AnalysisConfig()

# ================= Logger Setup =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_signals.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= Technical Indicators =================
class TechnicalIndicators:
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

# ================= Market Data Analysis =================
class MarketAnalyzer:
    def __init__(self):
        self.previous_volume = 0
        self.previous_signals = {}
        self.signal_history = deque(maxlen=config.SIGNAL_CONFIRMATION_COUNT)
        self.current_trend = "SIDEWAYS"
        self.trend_strength = 0
        self.last_confirmed_signal = "HOLD"
        self.last_signal_time = time.time()
        self.last_confirmed_signal_time = time.time()
        self.signal_lock = False  # Lock signal to prevent frequent changes
        
    def load_latest_data(self, filepath: str, n_rows: int = 100) -> Optional[pd.DataFrame]:
        """Load the latest N rows from a CSV file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                return None
                
            # Read the last n_rows from the CSV
            df = pd.read_csv(filepath)
            if len(df) < n_rows:
                return df
            return df.tail(n_rows)
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            return None
    
    def determine_market_trend(self, ohlcv_data: pd.DataFrame) -> Tuple[str, float]:
        """Determine the overall market trend using multiple timeframes and indicators"""
        if ohlcv_data is None or len(ohlcv_data) < config.TREND_CONFIRMATION_PERIOD:
            return "SIDEWAYS", 0
        
        close_prices = ohlcv_data['close']
        
        # Calculate trend using multiple methods
        sma_short = TechnicalIndicators.calculate_sma(close_prices, config.SMA_SHORT)
        sma_long = TechnicalIndicators.calculate_sma(close_prices, config.SMA_LONG)
        
        # Price position relative to SMAs
        price_above_sma_short = close_prices.iloc[-1] > sma_short.iloc[-1] if not pd.isna(sma_short.iloc[-1]) else False
        price_above_sma_long = close_prices.iloc[-1] > sma_long.iloc[-1] if not pd.isna(sma_long.iloc[-1]) else False
        sma_short_above_long = sma_short.iloc[-1] > sma_long.iloc[-1] if not pd.isna(sma_short.iloc[-1]) and not pd.isna(sma_long.iloc[-1]) else False
        
        # Simple trend determination
        bullish_factors = 0
        bearish_factors = 0
        
        if price_above_sma_short:
            bullish_factors += 1
        else:
            bearish_factors += 1
            
        if price_above_sma_long:
            bullish_factors += 1
        else:
            bearish_factors += 1
            
        if sma_short_above_long:
            bullish_factors += 1
        else:
            bearish_factors += 1
        
        # Determine trend based on factors
        if bullish_factors >= 2:
            return "UPTREND", bullish_factors * 33.33
        elif bearish_factors >= 2:
            return "DOWNTREND", bearish_factors * 33.33
        else:
            return "SIDEWAYS", 0
    
    def analyze_price_action(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price action and generate signals with trend context"""
        if ohlcv_data is None or len(ohlcv_data) < max(config.SMA_LONG, config.RSI_PERIOD):
            return {
                "signal": "NEUTRAL", 
                "confidence": 0, 
                "reason": "Insufficient data",
                "price": 0,
                "rsi": 0,
                "volume_ratio": 0,
                "sma_short": 0,
                "sma_long": 0,
                "reasons": ["Insufficient data for analysis"]
            }
        
        # Check if the required columns exist
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in ohlcv_data.columns:
                return {
                    "signal": "NEUTRAL", 
                    "confidence": 0, 
                    "reason": f"Missing column: {col}",
                    "price": 0,
                    "rsi": 0,
                    "volume_ratio": 0,
                    "sma_short": 0,
                    "sma_long": 0,
                    "reasons": [f"Missing column: {col} in data"]
                }
        
        # Determine market trend first
        market_trend, trend_strength = self.determine_market_trend(ohlcv_data)
        self.current_trend = market_trend
        self.trend_strength = trend_strength
        
        # Calculate technical indicators
        close_prices = ohlcv_data['close']
        volumes = ohlcv_data['volume']
        
        # SMA analysis
        sma_short = TechnicalIndicators.calculate_sma(close_prices, config.SMA_SHORT)
        sma_long = TechnicalIndicators.calculate_sma(close_prices, config.SMA_LONG)
        
        # RSI analysis
        rsi = TechnicalIndicators.calculate_rsi(close_prices, config.RSI_PERIOD)
        
        # Get latest values
        current_price = close_prices.iloc[-1]
        current_volume = volumes.iloc[-1]
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        current_sma_short = sma_short.iloc[-1] if not pd.isna(sma_short.iloc[-1]) else current_price
        current_sma_long = sma_long.iloc[-1] if not pd.isna(sma_long.iloc[-1]) else current_price
        
        # Volume analysis
        avg_volume = volumes.tail(20).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Generate signals based on multiple factors with trend context
        signals = []
        confidence = 0
        reasons = []
        
        # Trend analysis (SMA crossover)
        if current_sma_short > current_sma_long:
            signals.append("BULLISH")
            confidence += 0.3
            reasons.append(f"Uptrend: SMA{config.SMA_SHORT} > SMA{config.SMA_LONG}")
        else:
            signals.append("BEARISH")
            confidence -= 0.3
            reasons.append(f"Downtrend: SMA{config.SMA_SHORT} < SMA{config.SMA_LONG}")
        
        # RSI analysis
        if current_rsi < config.RSI_OVERSOLD:
            signals.append("BULLISH")
            confidence += 0.2
            reasons.append(f"Oversold: RSI {current_rsi:.2f} < {config.RSI_OVERSOLD}")
        elif current_rsi > config.RSI_OVERBOUGHT:
            signals.append("BEARISH")
            confidence -= 0.2
            reasons.append(f"Overbought: RSI {current_rsi:.2f} > {config.RSI_OVERBOUGHT}")
        else:
            signals.append("NEUTRAL")
            reasons.append(f"RSI in neutral range: {current_rsi:.2f}")
        
        # Volume spike analysis
        if volume_ratio > config.VOLUME_SPIKE:
            if 'open' in ohlcv_data.columns and current_price > ohlcv_data['open'].iloc[-1]:  # Green candle
                signals.append("BULLISH")
                confidence += 0.1
                reasons.append(f"High volume buying: {volume_ratio:.2f}x avg volume")
            else:  # Red candle
                signals.append("BEARISH")
                confidence -= 0.1
                reasons.append(f"High volume selling: {volume_ratio:.2f}x avg volume")
        
        # Add trend context to reasons
        reasons.append(f"Market Trend: {market_trend} (Strength: {trend_strength:.2f})")
        
        # Count signals
        bull_count = signals.count("BULLISH")
        bear_count = signals.count("BEARISH")
        neutral_count = signals.count("NEUTRAL")
        
        # Determine final signal with trend consideration
        if bull_count > bear_count and bull_count > neutral_count:
            final_signal = "BUY"
            confidence = min(0.9, max(0.1, confidence))
        elif bear_count > bull_count and bear_count > neutral_count:
            final_signal = "SELL"
            confidence = min(0.9, max(0.1, abs(confidence)))
        else:
            final_signal = "HOLD"
            confidence = 0.5
        
        return {
            "signal": final_signal,
            "confidence": round(confidence, 2),
            "price": current_price,
            "rsi": round(current_rsi, 2),
            "volume_ratio": round(volume_ratio, 2),
            "sma_short": round(current_sma_short, 2),
            "sma_long": round(current_sma_long, 2),
            "reasons": reasons,
            "market_trend": market_trend,
            "trend_strength": round(trend_strength, 2)
        }
    
    def analyze_order_book(self, order_book_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze order book depth for market sentiment"""
        if order_book_data is None or len(order_book_data) == 0:
            return {"order_book_signal": "NEUTRAL", "bid_ask_imbalance": 0}
        
        # Get the latest order book snapshot
        latest = order_book_data.iloc[-1]
        
        try:
            # Check if required columns exist
            if 'bids' not in latest or 'asks' not in latest:
                return {"order_book_signal": "NEUTRAL", "bid_ask_imbalance": 0}
            
            # Parse bids and asks from JSON strings
            bids = json.loads(latest['bids']) if isinstance(latest['bids'], str) else []
            asks = json.loads(latest['asks']) if isinstance(latest['asks'], str) else []
            
            # Calculate total bid and ask volume
            bid_volume = sum(float(bid[1]) for bid in bids[:5]) if bids else 0  # Top 5 bids
            ask_volume = sum(float(ask[1]) for ask in asks[:5]) if asks else 0  # Top 5 asks
            
            # Calculate imbalance
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Determine signal based on imbalance
            if imbalance > 0.1:
                return {"order_book_signal": "BULLISH", "bid_ask_imbalance": round(imbalance, 3)}
            elif imbalance < -0.1:
                return {"order_book_signal": "BEARISH", "bid_ask_imbalance": round(imbalance, 3)}
            else:
                return {"order_book_signal": "NEUTRAL", "bid_ask_imbalance": round(imbalance, 3)}
                
        except Exception as e:
            logger.error(f"Error analyzing order book: {str(e)}")
            return {"order_book_signal": "NEUTRAL", "bid_ask_imbalance": 0}
    
    def confirm_signal(self, new_signal: str, confidence: float) -> Tuple[str, float]:
        """Confirm signal with history and trend context"""
        current_time = time.time()
        
        # Add new signal to history
        self.signal_history.append((new_signal, confidence, current_time))
        
        # If signal is locked and minimum duration hasn't passed, return last confirmed signal
        if self.signal_lock and current_time - self.last_confirmed_signal_time < config.MIN_SIGNAL_DURATION:
            return self.last_confirmed_signal, confidence
        
        # If we don't have enough history, return the new signal
        if len(self.signal_history) < config.SIGNAL_CONFIRMATION_COUNT:
            return new_signal, confidence
        
        # Check if all recent signals are the same
        recent_signals = [signal for signal, conf, time in self.signal_history]
        if all(s == new_signal for s in recent_signals):
            # Strong confirmation - increase confidence and lock signal
            confirmed_confidence = min(0.95, confidence * 1.2)
            self.last_confirmed_signal = new_signal
            self.last_confirmed_signal_time = current_time
            self.signal_lock = True
            return new_signal, confirmed_confidence
        
        # Check if signal is opposite to trend
        if (new_signal == "BUY" and self.current_trend == "DOWNTREND") or \
           (new_signal == "SELL" and self.current_trend == "UPTREND"):
            # Against trend - reduce confidence
            against_trend_confidence = max(0.1, confidence * 0.7)
            return self.last_confirmed_signal, against_trend_confidence
        
        # If signal changed recently, be cautious
        if current_time - self.last_confirmed_signal_time < 30:  # 30 seconds since last signal change
            cautious_confidence = max(0.1, confidence * 0.8)
            return self.last_confirmed_signal, cautious_confidence
        
        # Default: return the new signal with original confidence
        self.last_confirmed_signal = new_signal
        self.last_confirmed_signal_time = current_time
        self.signal_lock = False
        return new_signal, confidence
    
    def get_trading_signal(self) -> Dict[str, Any]:
        """Generate a comprehensive trading signal based on multiple data sources"""
        # Load OHLCV data
        ohlcv_file = os.path.join(config.DATA_DIR, f"{config.TIMEFRAME}.csv")
        ohlcv_data = self.load_latest_data(ohlcv_file, 100)
        
        # Load order book data
        order_book_file = os.path.join(config.DATA_DIR, "order_book", "depth.csv")
        order_book_data = self.load_latest_data(order_book_file, 10)
        
        # Analyze price action
        price_analysis = self.analyze_price_action(ohlcv_data)
        
        # Analyze order book
        order_book_analysis = self.analyze_order_book(order_book_data)
        
        # Confirm signal with history and trend
        confirmed_signal, confirmed_confidence = self.confirm_signal(
            price_analysis["signal"], 
            price_analysis["confidence"]
        )
        
        # Combine analyses
        combined_signal = price_analysis.copy()
        combined_signal.update(order_book_analysis)
        combined_signal["signal"] = confirmed_signal
        combined_signal["confidence"] = confirmed_confidence
        
        # Add final_signal key
        combined_signal["final_signal"] = confirmed_signal
        
        # Final decision with order book confirmation
        if (confirmed_signal == "BUY" and 
            order_book_analysis["order_book_signal"] == "BULLISH"):
            combined_signal["final_signal"] = "STRONG_BUY"
            combined_signal["confidence"] = min(0.95, confirmed_confidence + 0.1)
        elif (confirmed_signal == "SELL" and 
              order_book_analysis["order_book_signal"] == "BEARISH"):
            combined_signal["final_signal"] = "STRONG_SELL"
            combined_signal["confidence"] = min(0.95, confirmed_confidence + 0.1)
        elif (confirmed_signal == "BUY" and 
              order_book_analysis["order_book_signal"] == "BEARISH"):
            combined_signal["final_signal"] = "WEAK_BUY"
            combined_signal["confidence"] = max(0.1, confirmed_confidence - 0.1)
        elif (confirmed_signal == "SELL" and 
              order_book_analysis["order_book_signal"] == "BULLISH"):
            combined_signal["final_signal"] = "WEAK_SELL"
            combined_signal["confidence"] = max(0.1, confirmed_confidence - 0.1)
        
        return combined_signal

# ================= Trading Signal Monitor =================
def monitor_trading_signals():
    """Continuously monitor market data and generate trading signals"""
    analyzer = MarketAnalyzer()
    
    logger.info("Starting BTC/USDT Trading Signal Monitor...")
    logger.info("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Get trading signal
            signal_data = analyzer.get_trading_signal()
            
            # Log the signal
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Use get() to avoid KeyError if any keys are missing
            logger.info(f"{timestamp} | Signal: {signal_data.get('final_signal', 'N/A')} | "
                       f"Confidence: {signal_data.get('confidence', 0)} | "
                       f"Price: ${signal_data.get('price', 0)} | "
                       f"RSI: {signal_data.get('rsi', 0)} | "
                       f"Volume Ratio: {signal_data.get('volume_ratio', 0)}x | "
                       f"Trend: {signal_data.get('market_trend', 'N/A')} ({signal_data.get('trend_strength', 0)})")
            
            # Log reasons for the signal
            for reason in signal_data.get('reasons', []):
                logger.info(f"  - {reason}")
            
            # Log order book analysis if available
            if 'bid_ask_imbalance' in signal_data:
                logger.info(f"  - Order Book: {signal_data.get('order_book_signal', 'N/A')} "
                           f"(Imbalance: {signal_data.get('bid_ask_imbalance', 0)})")
            
            logger.info("-" * 80)
            
            # Wait before next check
            time.sleep(config.CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("Trading signal monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring: {str(e)}")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    monitor_trading_signals()