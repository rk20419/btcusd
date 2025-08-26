# fetch/5k.py
import requests
import pandas as pd
import time
import os
import numpy as np
from datetime import datetime
import pytz
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field

# ================= Configuration =================
@dataclass
class Config:
    BASE_URL: str = "https://api.binance.com/api/v3/klines"
    SYMBOL: str = "BTCUSDT"
    TIMEFRAMES: Dict[str, int] = field(default_factory=lambda: {
        "1m":100
    })
    LIMIT: int = 1000  # Max candles per request
    OUTPUT_DIR: str = "data/historical"
    REQUEST_DELAY: float = 0.3
    MAX_RETRIES: int = 5
    TIMEZONE: str = "Asia/Kolkata"  # IST timezone
    VOLATILITY_WINDOW: int = 10     # Rolling window for volatility %

config = Config()

# ================= Logger Setup =================
class SafeLogger:
    @staticmethod
    def setup():
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/historical_data.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

logger = SafeLogger.setup()

# ================= Data Handling =================
class DataManager:
    @staticmethod
    def initialize_directories():
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    @staticmethod
    def save_data(df: pd.DataFrame, timeframe: str) -> bool:
        """Save processed data to CSV in same format as live stream output"""
        try:
            filename = os.path.join(
                config.OUTPUT_DIR,
                f"{config.SYMBOL}_{timeframe}_{len(df)}.csv"
            )
            # Live format only
            columns_to_save = [
                "timestamp","open","high","low","close","volume",
                "quote_volume","trades","taker_buy_base","taker_buy_quote",
                "volatility","datetime_ist"
                ]
            df[columns_to_save].to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} candles to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {timeframe} data: {str(e)}")
            return False

# ================= Data Processing =================
class DataProcessor:
    @staticmethod
    def validate_klines(data: List[List[Any]]) -> bool:
        """Validate Binance kline structure"""
        return all(len(candle) == 12 for candle in data)

    @staticmethod
    def calculate_volatility(df: pd.DataFrame, period: int = config.VOLATILITY_WINDOW) -> pd.Series:
        """
        Calculate volatility (%) as rolling std deviation of returns * 100
        Same as in stream/websocket_client.py
        """
        prices = df["close"]
        returns = prices.pct_change().fillna(0)
        vol = returns.rolling(window=period, min_periods=1).std() * 100
        return vol.fillna(0.0).round(5)

    @staticmethod
    def process_klines(data: List[List[Any]], timeframe: str) -> pd.DataFrame:
        """Convert raw klines to standardized format + volatility, matching stream output"""
        if not DataProcessor.validate_klines(data):
            raise ValueError("Invalid kline data structure")

        # TF in ms (unused but can check gaps)
        interval_min = int(timeframe[:-1])
        interval_ms = interval_min * 60 * 1000

        # Raw -> DF
        df = pd.DataFrame(data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Volume", "Trades", "Taker Buy Base",
            "Taker Buy Quote", "Ignore"
        ])

        processed = pd.DataFrame()
        processed["timestamp"] = df["Open Time"].astype("int64")
        # Datetime in IST
        tz = pytz.timezone(config.TIMEZONE)
        processed["datetime_ist"] = processed["timestamp"].apply(
            lambda x: datetime.fromtimestamp(x / 1000, tz).strftime('%Y-%m-%d %H:%M:%S')
        )


        # sabhi numeric fields
        processed["open"] = df["Open"].astype(float)
        processed["high"] = df["High"].astype(float)
        processed["low"] = df["Low"].astype(float)
        processed["close"] = df["Close"].astype(float)
        processed["volume"] = df["Volume"].astype(float)
        processed["quote_volume"] = df["Quote Volume"].astype(float)
        processed["trades"] = df["Trades"].astype(int)
        processed["taker_buy_base"] = df["Taker Buy Base"].astype(float)
        processed["taker_buy_quote"] = df["Taker Buy Quote"].astype(float)

        # volatility
        processed["volatility"] = DataProcessor.calculate_volatility(processed, period=config.VOLATILITY_WINDOW)


        # Gap log (optional)
        diffs = processed["timestamp"].diff().dropna()
        if (diffs != interval_ms).any():
            gaps = (diffs != interval_ms).sum()
            logger.warning(f"{timeframe}: {gaps} gaps detected in historical data")

        return processed

# ================= Data Fetcher =================
class HistoricalDataFetcher:
    def __init__(self):
        self.session = requests.Session()
        DataManager.initialize_directories()

    def fetch_timeframe(self, timeframe: str, candle_count: int) -> bool:
        logger.info(f"\nFetching {candle_count} {timeframe} candles for {config.SYMBOL}")

        all_candles = []
        end_time = int(time.time() * 1000)
        retries = 0

        while len(all_candles) < candle_count and retries < config.MAX_RETRIES:
            try:
                url = (f"{config.BASE_URL}?symbol={config.SYMBOL}"
                       f"&interval={timeframe}&limit={config.LIMIT}"
                       f"&endTime={end_time}")

                resp = self.session.get(url)
                resp.raise_for_status()
                new_data = resp.json()

                if not new_data:
                    logger.info("Reached beginning of available data")
                    break

                all_candles = new_data + all_candles
                end_time = new_data[0][0] - 1
                retries = 0

                tz = pytz.timezone(config.TIMEZONE)
                first_dt = datetime.fromtimestamp(new_data[0][0] / 1000, tz).strftime('%d-%m-%y %H:%M')
                last_dt = datetime.fromtimestamp(new_data[-1][0] / 1000, tz).strftime('%d-%m-%y %H:%M')
                logger.info(f"Fetched {len(new_data)} | Total: {len(all_candles)} | Range: {first_dt} → {last_dt} IST")

                time.sleep(config.REQUEST_DELAY)

            except Exception as e:
                retries += 1
                logger.error(f"Attempt {retries}/{config.MAX_RETRIES} failed: {str(e)}")
                time.sleep(5 * retries)

        if not all_candles:
            logger.error(f"No data fetched for {timeframe}")
            return False

        if len(all_candles) > candle_count:
            all_candles = all_candles[-candle_count:]

        try:
            df = DataProcessor.process_klines(all_candles, timeframe)
            return DataManager.save_data(df, timeframe)
        except Exception as e:
            logger.error(f"Processing failed for {timeframe}: {str(e)}")
            return False

# ================= Application Control =================
class Application:
    def __init__(self):
        self.fetcher = HistoricalDataFetcher()

    def run(self):
        logger.info("Starting historical data collection...")
        for timeframe, count in config.TIMEFRAMES.items():
            if not self.fetcher.fetch_timeframe(timeframe, count):
                logger.error(f"Skipping {timeframe} due to errors")
        logger.info("Historical data collection complete!")

if __name__ == "__main__":
    app = Application()
    app.run()
