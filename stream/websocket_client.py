# stream/websocket_client.py
import websocket
import json
import pandas as pd
import os
import time
import logging
import threading
import signal
import sys
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import pytz

# ================= Configuration =================
@dataclass
class Config:
    SYMBOL: str = "btcusdt"
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["1m"])
    BASE_URL: str = "wss://stream.binance.com:9443/stream?streams="
    RAW_DATA_FILE: str = "data/live/live_raw_data.csv"
    OUTPUT_DIR: str = "data/live"
    RECONNECT_DELAY: int = 5
    PING_INTERVAL: int = 30
    PING_TIMEOUT: int = 10
    MAX_RECONNECTS: int = 10
    # VOLATILITY_WINDOW: int = 10
    PRICE_HISTORY_MAX_LEN: int = 100
    TIMEZONE: str = "Asia/Kolkata"  # Match historical fetcher

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
                logging.FileHandler('logs/live_stream_data.log', encoding='utf-8'),
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
        os.makedirs(os.path.dirname(config.RAW_DATA_FILE), exist_ok=True)

    @staticmethod
    def initialize_files():
        # Raw tick data
        if not os.path.exists(config.RAW_DATA_FILE):
            pd.DataFrame(columns=["timestamp", "price", "datetime_ist"]).to_csv(
                config.RAW_DATA_FILE, index=False
            )
        # OHLCV candles per timeframe
        columns_to_save = [
            "timestamp","open","high","low","close","volume",
            "quote_volume","trades","taker_buy_base","taker_buy_quote",
            "datetime_ist"
        ]
        for tf in config.TIMEFRAMES:
            filepath = os.path.join(config.OUTPUT_DIR, f"{tf}.csv")
            if not os.path.exists(filepath):
                pd.DataFrame(columns=columns_to_save).to_csv(filepath, index=False)

    @staticmethod
    def save_raw_tick(data: Dict[str, float]) -> bool:
        """Append tick data with IST datetime"""
        try:
            tz = pytz.timezone(config.TIMEZONE)
            data["datetime_ist"] = datetime.fromtimestamp(data["timestamp"] / 1000, tz).strftime(
                '%Y-%m-%d %H:%M:%S'
            )
            df = pd.DataFrame([data])
            df.to_csv(config.RAW_DATA_FILE, mode='a', header=False, index=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save raw tick: {str(e)}")
            return False

    @staticmethod
    def save_candle(tf: str, data: Dict[str, float]) -> bool:
        """Append OHLCV candle with IST datetime - includes """
        try:
            filepath = os.path.join(config.OUTPUT_DIR, f"{tf}.csv")
            tz = pytz.timezone(config.TIMEZONE)
            data["datetime_ist"] = datetime.fromtimestamp(data["timestamp"] / 1000, tz).strftime(
                '%Y-%m-%d %H:%M:%S'
            )
            df = pd.DataFrame([data])
            df.to_csv(filepath, mode='a', header=False, index=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save {tf} candle: {str(e)}")
            return False

# ================= WebSocket Client =================
class BinanceWebSocket:
    def __init__(self):
        self.ws: Optional[websocket.WebSocketApp] = None
        self.active = True
        self.last_message_time = time.time()
        self.reconnect_attempts = 0
        self.lock = threading.Lock()
        self.price_history = []
        DataManager.initialize_directories()
        DataManager.initialize_files()

    # def _calculate_volatility(self, current_price: float) -> float:
    #     """
    #     Calculate volatility (%) as rolling std deviation of returns, multiplied by 100
    #     Using configured window length.
    #     """
    #     self.price_history.append(current_price)
    #     if len(self.price_history) > config.PRICE_HISTORY_MAX_LEN:
    #         self.price_history.pop(0)

    #     if len(self.price_history) >= config.VOLATILITY_WINDOW:
    #         prices = pd.Series(self.price_history)
    #         returns = prices.pct_change().dropna()
    #         if len(returns) >= 2:
    #             vol = returns.std() * 100
    #             return float(round(vol, 5))  # rounded to 5 decimals for CSV simplicity
    #     return 0.0

    def _handle_message(self, ws, message: str):
        with self.lock:
            self.last_message_time = time.time()
            try:
                msg = json.loads(message)
                if not isinstance(msg, dict) or 'stream' not in msg:
                    return
                stream = msg.get('stream', '')
                if not stream.startswith(f"{config.SYMBOL}@kline_"):
                    return
                data = msg.get('data', {})
                kline = data.get('k', {})
                if not all(key in kline for key in ['t', 'o', 'h', 'l', 'c', 'v', 'x']):
                    logger.warning("Incomplete kline data.")
                    return
                tf = stream.split('@')[1].replace('kline_', '')
                self._process_kline(kline, tf)
            except Exception as e:
                logger.error(f"Message processing error: {str(e)}")

    def _process_kline(self, kline: Dict, timeframe: str):
        try:
            current_price = float(kline["c"])
            # volatility = self._calculate_volatility(current_price)
            tick_data = {"timestamp": int(kline["t"]), "price": current_price,}
            DataManager.save_raw_tick(tick_data)

            if kline.get("x", False):  # Closed candle
                candle_data = {
                    "timestamp": int(kline["t"]),
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": current_price,
                    "volume": float(kline["v"]),
                    "quote_volume": float(kline.get("q", 0.0)),
                    "trades": int(kline.get("n", 0)),
                    "taker_buy_base": float(kline.get("V", 0.0)),
                    "taker_buy_quote": float(kline.get("Q", 0.0)),
                    # "volatility": volatility
                }
                if DataManager.save_candle(timeframe, candle_data):
                    logger.info(f"Saved {timeframe} candle | Price: {current_price:.2f}")
        except Exception as e:
            logger.error(f"Kline processing error: {str(e)}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {str(error)}")
        self._schedule_reconnect()

    def _on_close(self, ws, code, msg):
        logger.info(f"Connection closed: {code} {msg}")
        self._schedule_reconnect()

    def _on_open(self, ws):
        self.reconnect_attempts = 0
        logger.info("WebSocket connection established.")
        streams = [f"{config.SYMBOL}@kline_{tf}" for tf in config.TIMEFRAMES]
        ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": int(time.time())}))

    def _schedule_reconnect(self):
        if self.active and self.reconnect_attempts < config.MAX_RECONNECTS:
            self.reconnect_attempts += 1
            delay = min(config.RECONNECT_DELAY * self.reconnect_attempts, 60)
            logger.info(f"Reconnecting in {delay}s...")
            time.sleep(delay)
            self.run()

    def _monitor_connection(self):
        while self.active:
            if time.time() - self.last_message_time > 60:
                logger.warning("No messages for 60s. reconnecting...")
                if self.ws:
                    self.ws.close()
                break
            time.sleep(10)

    def run(self):
        streams = [f"{config.SYMBOL}@kline_{tf}" for tf in config.TIMEFRAMES]
        url = config.BASE_URL + "/".join(streams)
        logger.info("Connecting to Binance WebSocket...")
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._handle_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        threading.Thread(target=self._monitor_connection, daemon=True).start()
        self.ws.run_forever(ping_interval=config.PING_INTERVAL, ping_timeout=config.PING_TIMEOUT)

    def shutdown(self):
        self.active = False
        if self.ws:
            self.ws.close()

# ================= Application Control =================
def shutdown_handler(sig, frame):
    logger.info("Shutdown signal received.")
    ws_client.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    ws_client = BinanceWebSocket()
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    logger.info("Starting live WebSocket BTC/USDT data collection...")
    try:
        ws_client.run()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)