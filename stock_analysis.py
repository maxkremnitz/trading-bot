import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time as dt_time
import pytz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import requests
import os
import time
import threading
import sys
import json
import traceback
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string
import logging
from functools import wraps
import sqlite3
from contextlib import contextmanager
import warnings
import gc  # Memory Management
warnings.filterwarnings('ignore')
load_dotenv()

# Logging Setup mit Memory-optimierten Settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_bot.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Memory Management für Render
def cleanup_memory():
    """Aggressive Memory Cleanup für Render"""
    try:
        gc.collect()
        # Clear matplotlib cache
        if 'matplotlib.pyplot' in sys.modules:
            plt.close('all')
        return True
    except Exception as e:
        logger.error(f"Memory cleanup error: {e}")
        return False

# RATE LIMITER (OPTIMIERT)
class RateLimiter:
    """Capital.com API Rate Limiter - Memory optimized"""
    def __init__(self):
        self.general_requests = []
        self.trading_requests = []
        self.session_requests = []
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # 5 Minuten
        self.last_cleanup = time.time()
        logger.info("Rate Limiter initialisiert")
    
    def _cleanup_old_requests(self):
        """Cleanup alter Requests für Memory Management"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            with self.lock:
                # Nur letzte 100 Requests behalten
                self.general_requests = self.general_requests[-100:]
                self.trading_requests = self.trading_requests[-50:]
                self.session_requests = self.session_requests[-20:]
                self.last_cleanup = current_time
    
    def can_make_request(self, request_type="general"):
        """Prüft ob Request gemacht werden darf"""
        self._cleanup_old_requests()
        
        with self.lock:
            now = time.time()
            if request_type == "trading":
                self.trading_requests = [t for t in self.trading_requests if now - t < 0.1]
                if len(self.trading_requests) >= 1:
                    return False, 0.1 - (now - self.trading_requests[-1])
                self.trading_requests.append(now)
            elif request_type == "session":
                self.session_requests = [t for t in self.session_requests if now - t < 1.0]
                if len(self.session_requests) >= 1:
                    return False, 1.0 - (now - self.session_requests[-1])
                self.session_requests.append(now)
            else:
                self.general_requests = [t for t in self.general_requests if now - t < 1.0]
                if len(self.general_requests) >= 10:
                    return False, 1.0 - (now - self.general_requests[0])
                self.general_requests.append(now)
            return True, 0
    
    def wait_if_needed(self, request_type="general"):
        """Wartet falls Rate Limit erreicht"""
        can_proceed, wait_time = self.can_make_request(request_type)
        if not can_proceed:
            logger.info(f"Rate Limit: Warte {wait_time:.2f}s für {request_type} Request")
            time.sleep(wait_time + 0.01)

# TRADING HOURS MANAGER
class TradingHoursManager:
    """Handelszeitenbeschränkungen"""
    def __init__(self):
        self.market_hours = {
            'NYSE': {
                'timezone': pytz.timezone('US/Eastern'),
                'open_time': dt_time(9, 30),
                'close_time': dt_time(16, 0),
                'weekdays': [0, 1, 2, 3, 4]
            },
            'XETRA': {
                'timezone': pytz.timezone('Europe/Berlin'),
                'open_time': dt_time(9, 0),
                'close_time': dt_time(17, 30),
                'weekdays': [0, 1, 2, 3, 4]
            },
            'FOREX': {
                'timezone': pytz.timezone('UTC'),
                'open_time': dt_time(21, 0),
                'close_time': dt_time(22, 0),
                'weekdays': [0, 1, 2, 3, 4, 6]
            }
        }
        logger.info("Trading Hours Manager initialisiert")
    
    def is_market_open(self, market='NYSE'):
        """Prüft ob Markt geöffnet ist"""
        try:
            market_info = self.market_hours.get(market, self.market_hours['NYSE'])
            market_tz = market_info['timezone']
            now_utc = datetime.now(pytz.UTC)
            now_market = now_utc.astimezone(market_tz)
            
            if now_market.weekday() not in market_info['weekdays']:
                return False, f"{market} geschlossen (Wochenende)"
            
            current_time = now_market.time()
            open_time = market_info['open_time']
            close_time = market_info['close_time']
            
            if market == 'FOREX':
                if now_market.weekday() == 6:
                    is_open = current_time >= open_time
                elif now_market.weekday() == 4:
                    is_open = current_time <= close_time
                else:
                    is_open = True
            else:
                is_open = open_time <= current_time <= close_time
            
            status = f"{market} {'OFFEN' if is_open else 'GESCHLOSSEN'}"
            return is_open, status
        except Exception as e:
            logger.error(f"Marktzeit-Prüfung Fehler: {e}")
            return False, "Zeitprüfung fehlgeschlagen"
    
    def get_trading_status(self):
        """Umfassender Trading-Status"""
        nyse_open, nyse_status = self.is_market_open('NYSE')
        xetra_open, xetra_status = self.is_market_open('XETRA')
        forex_open, forex_status = self.is_market_open('FOREX')
        
        main_strategy_allowed = nyse_open or xetra_open
        gold_silver_allowed = forex_open
        any_market_open = main_strategy_allowed or gold_silver_allowed
        
        return {
            'analysis_allowed': any_market_open,
            'main_strategy_trading': main_strategy_allowed,
            'gold_silver_trading': gold_silver_allowed,
            'nyse_status': nyse_status,
            'xetra_status': xetra_status,
            'forex_status': forex_status
        }

# SAFE EXECUTION HELPERS
def safe_execute(func):
    """Decorator für sichere Funktionsausführung"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fehler in {func.__name__}: {str(e)}")
            return None
    return wrapper

def safe_float(value, default=0.0):
    """Sichere Float-Konvertierung"""
    try:
        if pd.isna(value) or value is None or value == "":
            return default
        if isinstance(value, (pd.Series, np.ndarray)):
            return default
        return float(value)
    except:
        return default

def safe_round(value, decimals=2):
    """Sichere Round-Funktion"""
    try:
        return round(safe_float(value), decimals)
    except:
        return 0.0

# DATABASE MANAGER (OPTIMIERT)
class DatabaseManager:
    """SQLite Datenbank mit Memory Optimization"""
    def __init__(self, db_path="trading_bot.db"):
        self.db_path = db_path
        self.init_database()
        self.cleanup_interval = 3600  # 1 Stunde
        self.last_cleanup = time.time()
        logger.info("Database Manager initialisiert")
    
    def init_database(self):
        """Datenbank und Tabellen erstellen"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=-64000")  # 64MB Cache
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        price REAL,
                        score REAL,
                        rating TEXT,
                        rsi REAL,
                        macd REAL,
                        volatility REAL,
                        strategy TEXT,
                        account_type TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        action TEXT NOT NULL,
                        score REAL,
                        position_size REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        status TEXT,
                        deal_reference TEXT,
                        deal_id TEXT,
                        account_type TEXT,
                        strategy TEXT,
                        result TEXT
                    )
                """)
                
                conn.commit()
                logger.info("Datenbank initialisiert")
        except Exception as e:
            logger.error(f"DB Init Fehler: {e}")
    
    @contextmanager
    def get_connection(self):
        """Sichere Datenbankverbindung"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            logger.error(f"DB Connection Error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def cleanup_old_data(self):
        """Cleanup alter Daten für Memory Management"""
        try:
            if time.time() - self.last_cleanup > self.cleanup_interval:
                with self.get_connection() as conn:
                    if conn:
                        # Nur letzte 1000 Einträge behalten
                        conn.execute("DELETE FROM analysis_history WHERE id NOT IN (SELECT id FROM analysis_history ORDER BY timestamp DESC LIMIT 1000)")
                        conn.execute("DELETE FROM trades WHERE id NOT IN (SELECT id FROM trades ORDER BY timestamp DESC LIMIT 500)")
                        conn.commit()
                        self.last_cleanup = time.time()
                        logger.info("DB Cleanup durchgeführt")
        except Exception as e:
            logger.error(f"DB Cleanup Fehler: {e}")
    
    @safe_execute
    def save_analysis(self, ticker, data, strategy="main", account_type="demo"):
        """Analyse-Daten speichern"""
        self.cleanup_old_data()
        try:
            with self.get_connection() as conn:
                if conn:
                    conn.execute("""
                        INSERT INTO analysis_history
                        (timestamp, ticker, price, score, rating, rsi, macd, volatility, strategy, account_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        ticker,
                        data.get('price', 0),
                        data.get('score', 0),
                        data.get('rating', 'Unknown'),
                        data.get('rsi', 0),
                        data.get('macd', 0),
                        data.get('volatility', 0),
                        strategy,
                        account_type
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Save Analysis Fehler: {e}")
    
    @safe_execute
    def save_trade(self, trade_data):
        """Trade-Daten speichern"""
        try:
            with self.get_connection() as conn:
                if conn:
                    conn.execute("""
                        INSERT INTO trades
                        (timestamp, ticker, action, score, position_size, stop_loss, take_profit,
                         status, deal_reference, deal_id, account_type, strategy, result)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        trade_data.get('ticker', ''),
                        trade_data.get('action', ''),
                        trade_data.get('score', 0),
                        trade_data.get('position_size', 0),
                        trade_data.get('stop_loss', 0),
                        trade_data.get('take_profit', 0),
                        trade_data.get('status', 'pending'),
                        trade_data.get('deal_reference', ''),
                        trade_data.get('deal_id', ''),
                        trade_data.get('account_type', 'demo'),
                        trade_data.get('strategy', 'main'),
                        json.dumps(trade_data.get('result', {}))
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Save Trade Fehler: {e}")

# CAPITAL.COM API CLIENT (FIXED)
class CapitalComAPI:
    """Capital.com API Client mit Position Size Fix"""
    def __init__(self, rate_limiter, account_type="main"):
        self.api_key = os.getenv('CAPITAL_API_KEY')
        self.password = os.getenv('CAPITAL_PASSWORD')
        self.email = os.getenv('CAPITAL_EMAIL')
        self.account_type = account_type
        self.rate_limiter = rate_limiter
        
        self.base_url = "https://demo-api-capital.backend-capital.com"
        
        # Session Tokens
        self.cst_token = None
        self.security_token = None
        self.account_id = None
        self.last_auth_time = 0
        self.auth_timeout = 600
        
        # Account Info
        self.available_accounts = []
        self.current_account = None
        
        # Position Sync (FIXED - kein Endlos-Loop)
        self.last_position_sync = 0
        self.position_sync_interval = 300  # 5 Minuten statt bei jedem Fehler
        
        # Epic Mapping
        self.epic_mapping = {
            'AAPL': 'AAPL', 'MSFT': 'MSFT', 'AMZN': 'AMZN', 'TSLA': 'TSLA',
            'NVDA': 'NVDA', 'GOOGL': 'GOOGL', 'META': 'META', 'NFLX': 'NFLX',
            'SAP.DE': 'SAP', 'DTE.DE': 'DTE',
            'GOLD': 'GOLD', 'SILVER': 'SILVER', 'GLD': 'GOLD', 'SLV': 'SILVER'
        }
        logger.info(f"Capital.com API initialisiert (Account: {account_type})")
    
    def is_authenticated(self):
        """Prüft Session-Status"""
        if not self.cst_token or not self.security_token:
            return False
        return (time.time() - self.last_auth_time) < self.auth_timeout
    
    @safe_execute
    def authenticate(self):
        """Session erstellen"""
        if not self.api_key or not self.password:
            logger.error("API Credentials fehlen")
            return False
        
        if self.is_authenticated():
            return True
        
        try:
            self.rate_limiter.wait_if_needed("session")
            
            headers = {
                'X-CAP-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'identifier': self.email,
                'password': self.password,
                'encryptedPassword': False
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/session",
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                self.cst_token = response.headers.get('CST')
                self.security_token = response.headers.get('X-SECURITY-TOKEN')
                
                if self.cst_token and self.security_token:
                    self.last_auth_time = time.time()
                    session_data = response.json()
                    self.available_accounts = session_data.get('accounts', [])
                    self.current_account = session_data.get('currentAccountId')
                    logger.info("Capital.com authentifiziert")
                    return True
            else:
                logger.error(f"Auth fehlgeschlagen: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Auth Fehler: {e}")
        
        return False
    
    @safe_execute
    def switch_account(self, target_account_type="demo1"):
        """Account wechseln"""
        if not self.is_authenticated():
            return False
        
        target_account = None
        for acc in self.available_accounts:
            acc_name = acc.get('accountName', '').lower()
            if target_account_type == "demo1" and "account 1" in acc_name:
                target_account = acc
                break
            elif target_account_type == "main" and "account 1" not in acc_name:
                target_account = acc
                break
        
        if not target_account:
            return False
        
        target_id = target_account.get('accountId')
        if target_id == self.current_account:
            return True
        
        try:
            self.rate_limiter.wait_if_needed("general")
            headers = self._get_auth_headers()
            data = {'accountId': target_id}
            
            response = requests.put(
                f"{self.base_url}/api/v1/session",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.current_account = target_id
                self.account_id = target_id
                logger.info(f"Account gewechselt zu: {target_account.get('accountName', 'Unknown')}")
                return True
        except Exception as e:
            logger.error(f"Account Switch Fehler: {e}")
        
        return False
    
    @safe_execute
    def get_positions(self):
        """Positionen abrufen"""
        if not self.is_authenticated():
            return []
        
        try:
            self.rate_limiter.wait_if_needed("general")
            headers = self._get_auth_headers()
            
            response = requests.get(
                f"{self.base_url}/api/v1/positions",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('positions', [])
        except Exception as e:
            logger.error(f"Get Positions Fehler: {e}")
        
        return []
    
    @safe_execute
    def get_available_balance(self):
        """Verfügbares Kapital abrufen"""
        try:
            account_info = self.get_account_info()
            if account_info and 'accounts' in account_info:
                for acc in account_info['accounts']:
                    if acc.get('accountId') == self.current_account:
                        balance_info = acc.get('balance', {})
                        available = balance_info.get('available', 0)
                        total_balance = balance_info.get('balance', 0)
                        logger.info(f"Balance - Total: {total_balance}, Available: {available}")
                        return float(available) if available else float(total_balance)
            
            return 10000.0  # Demo Account Fallback
        except Exception as e:
            logger.error(f"Balance Fehler: {e}")
            return 10000.0
    
    def calculate_position_size_percentage(self, target_percentage=7.5):
        """FIXED: Position Size Berechnung mit höherer Mindestgröße"""
        try:
            available_balance = self.get_available_balance()
            target_amount = available_balance * (target_percentage / 100)
            
            # FIXED: Höhere Mindestgröße für Capital.com
            if target_amount < 50:  # Weniger als $50
                position_size = 0.5  # Minimum
            elif target_amount < 100:  # $50-100
                position_size = 1.0
            elif target_amount < 500:  # $100-500
                position_size = round(target_amount / 100, 1)
            else:  # > $500
                position_size = round(target_amount / 200, 1)
            
            # Absolutes Minimum für Capital.com Demo Account
            position_size = max(0.5, position_size)
            
            logger.info(f"Position Size: {target_percentage}% von ${available_balance} = {position_size}")
            return position_size
        
        except Exception as e:
            logger.error(f"Position Size Error: {e}")
            return 1.0  # Sicherheits-Fallback erhöht
    
    @safe_execute
    def place_order(self, ticker, direction, size=None, stop_distance=None, profit_distance=None, use_percentage_size=False, percentage=7.5):
        """Order platzieren mit FIXED Position Size"""
        if not self.is_authenticated():
            logger.error("Nicht authentifiziert")
            return None
        
        epic = self.epic_mapping.get(ticker, ticker)
        
        try:
            # Position Size berechnen
            if use_percentage_size or size is None:
                size = self.calculate_position_size_percentage(percentage)
            else:
                size = max(0.5, float(size))  # Minimum 0.5 sicherstellen
            
            self.rate_limiter.wait_if_needed("trading")
            headers = self._get_auth_headers()
            
            order_data = {
                'epic': epic,
                'direction': str(direction).upper(),
                'size': size,
                'guaranteedStop': False
            }
            
            if stop_distance:
                order_data['stopDistance'] = max(10, int(stop_distance))
            if profit_distance:
                order_data['profitDistance'] = max(10, int(profit_distance))
            
            logger.info(f"Order: {epic} {direction} Size: {size}")
            
            response = requests.post(
                f"{self.base_url}/api/v1/positions",
                headers=headers,
                json=order_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                deal_reference = result.get('dealReference')
                logger.info(f"Order erfolgreich: {deal_reference}")
                
                if deal_reference:
                    time.sleep(1)
                    confirmation = self.check_deal_confirmation(deal_reference)
                    result['confirmation'] = confirmation
                
                return result
            else:
                logger.error(f"Order fehlgeschlagen ({response.status_code}): {response.text}")
                # REMOVED: Kein automatischer Sync mehr bei Order-Fehlern
        
        except Exception as e:
            logger.error(f"Order Fehler: {e}")
        
        return None
    
    @safe_execute
    def check_deal_confirmation(self, deal_reference):
        """Deal Confirmation prüfen"""
        try:
            self.rate_limiter.wait_if_needed("general")
            headers = self._get_auth_headers()
            
            response = requests.get(
                f"{self.base_url}/api/v1/confirms/{deal_reference}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                confirmation = response.json()
                deal_status = confirmation.get('dealStatus', 'UNKNOWN')
                logger.info(f"Deal Status: {deal_status}")
                return confirmation
        except Exception as e:
            logger.error(f"Deal Confirmation Fehler: {e}")
        
        return None
    
    @safe_execute
    def get_account_info(self):
        """Account Info abrufen"""
        if not self.is_authenticated():
            return None
        
        try:
            self.rate_limiter.wait_if_needed("general")
            headers = self._get_auth_headers()
            
            response = requests.get(
                f"{self.base_url}/api/v1/accounts",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Account Info Fehler: {e}")
        
        return None
    
    def _get_auth_headers(self):
        """Auth Headers"""
        if not self.cst_token or not self.security_token:
            return None
        
        return {
            'X-CAP-API-KEY': self.api_key,
            'CST': self.cst_token,
            'X-SECURITY-TOKEN': self.security_token,
            'Content-Type': 'application/json'
        }
    
    def get_current_account_name(self):
        """Aktueller Account Name"""
        for acc in self.available_accounts:
            if acc.get('accountId') == self.current_account:
                return acc.get('accountName', 'Unknown')
        return 'Unknown'
    
    def ensure_authenticated(self):
        """Auto-Reconnect"""
        if not self.is_authenticated():
            logger.info(f"Session abgelaufen - neu authentifizieren...")
            return self.authenticate()
        return True

# === MAIN TRADING STRATEGY (MEMORY OPTIMIZED) ===
class MainTradingStrategy:
    """Haupt-Trading-Strategie mit Memory Optimization"""
    def __init__(self):
        self.name = "Main Strategy"
        self.stocks_data = {}
        self.data_lock = threading.Lock()
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.hours_since_market_open = 0
        logger.info(f"{self.name} initialisiert")
    
    def get_stock_list(self):
        """Reduzierte Aktienliste für Memory Optimization"""
        return [
            {"Name": "Apple Inc.", "Ticker": "AAPL", "Currency": "USD"},
            {"Name": "Microsoft Corporation", "Ticker": "MSFT", "Currency": "USD"},
            {"Name": "Tesla Inc.", "Ticker": "TSLA", "Currency": "USD"},
            {"Name": "NVIDIA Corporation", "Ticker": "NVDA", "Currency": "USD"},
            {"Name": "Meta Platforms Inc.", "Ticker": "META", "Currency": "USD"},
            {"Name": "SAP SE", "Ticker": "SAP.DE", "Currency": "EUR"}
        ]
    
    def update_daily_trade_tracking(self):
        """Daily Trade Tracking"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
        
        current_hour = datetime.now().hour
        if 15 <= current_hour <= 23:
            self.hours_since_market_open = current_hour - 15
        else:
            self.hours_since_market_open = 0
    
    @safe_execute
    def fetch_historical_data(self, period="6mo"):  # Reduziert von 1y auf 6mo
        """Historische Daten laden (Memory optimized)"""
        stocks_list = self.get_stock_list()
        success_count = 0
        
        # Memory cleanup vor Datenload
        cleanup_memory()
        
        with self.data_lock:
            # Clear alte Daten
            self.stocks_data.clear()
        
        for stock in stocks_list:
            ticker = stock["Ticker"]
            try:
                data = yf.download(
                    ticker,
                    period=period,
                    progress=False,
                    auto_adjust=True,
                    repair=True
                )
                
                if not data.empty and len(data) > 20:
                    current_price = 0.0
                    if 'Close' in data.columns and len(data['Close'].dropna()) > 0:
                        current_price = float(data['Close'].dropna().iloc[-1])
                    
                    current_price = max(0.01, current_price)
                    
                    stock_info = stock.copy()
                    stock_info["CurrentPrice"] = current_price
                    stock_info["LastUpdate"] = datetime.now().isoformat()
                    
                    with self.data_lock:
                        self.stocks_data[ticker] = {
                            "HistoricalData": data.tail(200),  # Nur letzten 200 Tage behalten
                            "Info": stock_info
                        }
                    
                    success_count += 1
                    logger.info(f"{ticker}: ${current_price:.2f}")
                
                # Memory cleanup zwischen Downloads
                if success_count % 3 == 0:
                    cleanup_memory()
            
            except Exception as e:
                logger.error(f"Data Error {ticker}: {e}")
        
        logger.info(f"Daten geladen: {success_count}/{len(stocks_list)}")
        return success_count > 0
    
    @safe_execute
    def calculate_technical_indicators(self):
        """Technische Indikatoren (Memory optimized)"""
        with self.data_lock:
            for ticker, stock_data in self.stocks_data.items():
                try:
                    historical_data = stock_data["HistoricalData"].copy()
                    technical = {}
                    
                    if len(historical_data) < 15:
                        continue
                    
                    # Moving Averages
                    historical_data['MA20'] = historical_data['Close'].rolling(window=20, min_periods=15).mean()
                    historical_data['MA50'] = historical_data['Close'].rolling(window=50, min_periods=20).mean()
                    
                    current_price = safe_float(historical_data['Close'].iloc[-1])
                    ma20_current = safe_float(historical_data['MA20'].iloc[-1])
                    ma50_current = safe_float(historical_data['MA50'].iloc[-1])
                    
                    technical["Price_vs_MA20"] = safe_round((current_price / ma20_current - 1) * 100) if ma20_current > 0 else 0.0
                    technical["Price_vs_MA50"] = safe_round((current_price / ma50_current - 1) * 100) if ma50_current > 0 else 0.0
                    
                    # RSI (Simplified)
                    try:
                        delta = historical_data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=10).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=10).mean()
                        rs = gain / loss.replace(0, 0.0001)
                        rsi_series = 100 - (100 / (1 + rs))
                        technical["RSI"] = safe_round(rsi_series.iloc[-1])
                    except:
                        technical["RSI"] = 50.0
                    
                    # MACD (Simplified)
                    try:
                        if len(historical_data) >= 26:
                            ema12 = historical_data['Close'].ewm(span=12).mean()
                            ema26 = historical_data['Close'].ewm(span=26).mean()
                            macd = ema12 - ema26
                            technical["MACD_Histogram"] = safe_round(macd.iloc[-1])
                        else:
                            technical["MACD_Histogram"] = 0.0
                    except:
                        technical["MACD_Histogram"] = 0.0
                    
                    # Volatility
                    try:
                        returns = historical_data['Close'].pct_change().dropna()
                        if len(returns) > 5:
                            daily_vol = returns.std() * 100
                            technical["Volatility"] = safe_round(daily_vol, 2)
                        else:
                            technical["Volatility"] = 2.0
                    except:
                        technical["Volatility"] = 2.0
                    
                    self.stocks_data[ticker]["Technical"] = technical
                
                except Exception as e:
                    logger.error(f"Technical Analysis Error {ticker}: {e}")
        
        return True
    
    @safe_execute
    def generate_trade_signals(self):
        """Trade Signals (Aggressiver für mehr Trades)"""
        self.update_daily_trade_tracking()
        signals = []
        
        with self.data_lock:
            for ticker, stock_data in self.stocks_data.items():
                try:
                    technical = stock_data.get("Technical", {})
                    stock_info = stock_data.get("Info", {})
                    
                    score = 50.0
                    
                    # RSI Factor (liberaler)
                    rsi = technical.get("RSI", 50.0)
                    if rsi < 35:
                        score += 15
                    elif rsi > 65:
                        score -= 15
                    elif 40 <= rsi <= 60:
                        score += 8
                    
                    # MACD Factor
                    macd_hist = technical.get("MACD_Histogram", 0.0)
                    if macd_hist > 0.5:
                        score += 12
                    elif macd_hist < -0.5:
                        score -= 12
                    
                    # MA Factor
                    price_vs_ma20 = technical.get("Price_vs_MA20", 0.0)
                    if price_vs_ma20 > 1.5:
                        score += 10
                    elif price_vs_ma20 < -1.5:
                        score -= 10
                    
                    # Volatility Bonus
                    volatility = technical.get("Volatility", 2.0)
                    if volatility > 2.0:
                        score += 5
                    
                    score = max(0, min(100, score))
                    
                    # Liberalere Schwellen
                    if score >= 60:  # Gesenkt von 68
                        rating = "Strong Long"
                        action = "BUY"
                    elif score >= 55:  # Gesenkt von 58
                        rating = "Long"
                        action = "BUY"
                    elif score >= 45:
                        rating = "Hold"
                        action = "HOLD"
                    elif score >= 40:  # Erhöht von 32
                        rating = "Short"
                        action = "SELL"
                    else:
                        rating = "Strong Short"
                        action = "SELL"
                    
                    if action in ['BUY', 'SELL']:
                        current_price = stock_info.get("CurrentPrice", 0)
                        confidence = min(100, abs(score - 50) * 2)
                        
                        # Kleinere TP/SL für mehr Trades
                        sl_percent = 1.2
                        tp_percent = 2.0
                        
                        if action == 'BUY':
                            stop_loss = current_price * (1 - sl_percent/100)
                            take_profit = current_price * (1 + tp_percent/100)
                        else:
                            stop_loss = current_price * (1 + sl_percent/100)
                            take_profit = current_price * (1 - tp_percent/100)
                        
                        stop_distance = max(20, int(abs(current_price - stop_loss) * 100))
                        profit_distance = max(20, int(abs(take_profit - current_price) * 100))
                        
                        signal = {
                            'ticker': ticker,
                            'action': action,
                            'rating': rating,
                            'score': safe_round(score, 1),
                            'confidence': confidence,
                            'current_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'stop_distance': stop_distance,
                            'profit_distance': profit_distance,
                            'position_size': 1.0,  # Standard Size
                            'strategy': self.name
                        }
                        
                        signals.append(signal)
                        logger.info(f"{ticker}: {action} Signal (Score: {score:.1f})")
                
                except Exception as e:
                    logger.error(f"Signal Error {ticker}: {e}")
        
        # Minimum Daily Trade Logic
        if len(signals) == 0 and self.daily_trade_count == 0 and self.hours_since_market_open >= 3:
            signals = self._force_minimum_trade()
        
        logger.info(f"Signale generiert: {len(signals)}")
        return signals
    
    def _force_minimum_trade(self):
        """Force minimum daily trade"""
        try:
            with self.data_lock:
                if not self.stocks_data:
                    return []
                
                # Nimm erste verfügbare Aktie
                ticker = list(self.stocks_data.keys())[0]
                stock_data = self.stocks_data[ticker]
                stock_info = stock_data.get("Info", {})
                current_price = stock_info.get("CurrentPrice", 0)
                
                if current_price > 0:
                    # Konservativer Forced Trade
                    signal = {
                        'ticker': ticker,
                        'action': 'BUY',  # Immer BUY für Forced Trade
                        'rating': 'Forced Minimum',
                        'score': 55,
                        'confidence': 40,
                        'current_price': current_price,
                        'stop_loss': current_price * 0.99,  # 1% SL
                        'take_profit': current_price * 1.015,  # 1.5% TP
                        'stop_distance': max(20, int(current_price * 0.01 * 100)),
                        'profit_distance': max(20, int(current_price * 0.015 * 100)),
                        'position_size': 0.5,  # Kleine Size für Forced Trade
                        'strategy': f"{self.name} (Forced)"
                    }
                    
                    logger.info(f"Forced Trade: {ticker} BUY")
                    return [signal]
        
        except Exception as e:
            logger.error(f"Forced Trade Error: {e}")
        
        return []

# GOLD/SILVER STRATEGY (FIXED)
class GoldSilverStrategy:
    """Gold/Silver mit 7.5% Position Size und häufigeren Trades"""
    def __init__(self):
        self.name = "Gold/Silver Strategy"
        self.cache = {}
        self.cache_timeout = 180
        self.trade_count_today = 0
        self.last_trade_time = 0
        self.min_trade_interval = 1800  # 30 min
        logger.info(f"{self.name} initialisiert")
    
    @safe_execute
    def get_simulated_prices(self):
        """Simulierte Preise"""
        now = time.time()
        if 'prices' in self.cache and (now - self.cache.get('timestamp', 0)) < self.cache_timeout:
            return self.cache['prices']
        
        base_gold = 2000.0
        base_silver = 25.0
        
        gold_variation = (np.random.random() - 0.5) * 60
        silver_variation = (np.random.random() - 0.5) * 6
        
        gold_change = (np.random.random() - 0.5) * 6
        silver_change = (np.random.random() - 0.5) * 8
        
        if np.random.random() < 0.4:
            gold_change *= 1.5
            silver_change *= 1.3
        
        prices = {
            'gold': {
                'current': base_gold + gold_variation,
                'previous': base_gold,
                'change_pct': gold_change,
                'source': 'Simulated'
            },
            'silver': {
                'current': base_silver + silver_variation,
                'previous': base_silver,
                'change_pct': silver_change,
                'source': 'Simulated'
            }
        }
        
        self.cache['prices'] = prices
        self.cache['timestamp'] = now
        
        return prices
    
    @safe_execute
    def analyze_silver_trend(self):
        """Silver Trend Analysis"""
        try:
            trend_change = (np.random.random() - 0.5) * 6
            change_24h = (np.random.random() - 0.5) * 3
            
            if np.random.random() < 0.3:
                trend_change *= 1.5
                change_24h *= 1.2
            
            trend_direction = "up" if trend_change > 0 else "down"
            
            return {
                'trend_direction': trend_direction,
                'total_change_pct': trend_change,
                'change_24h_pct': change_24h,
                'is_significant': abs(trend_change) > 0.8,
                'is_strong_trend': abs(trend_change) > 1.8
            }
        except Exception as e:
            logger.error(f"Silver Analysis Error: {e}")
            return None
    
    @safe_execute
    def generate_gold_trade_signal(self):
        """Gold Trade Signal mit 7.5% Position Size"""
        try:
            current_time = time.time()
            if current_time - self.last_trade_time < self.min_trade_interval:
                return None
            
            prices = self.get_simulated_prices()
            if not prices:
                return None
            
            silver_analysis = self.analyze_silver_trend()
            if not silver_analysis:
                return None
            
            signal = None
            
            # SELL GOLD Signals
            if ((silver_analysis['trend_direction'] == 'up' and silver_analysis['total_change_pct'] > 0.8) or
                (silver_analysis['change_24h_pct'] > 1.0)):
                
                confidence = min(100, abs(silver_analysis['total_change_pct']) * 20)
                signal = {
                    'action': 'SELL',
                    'ticker': 'GOLD',
                    'reason': f"Silver up: {silver_analysis['total_change_pct']:.1f}%",
                    'confidence': confidence,
                    'strategy': self.name
                }
            
            # BUY GOLD Signals
            elif ((silver_analysis['change_24h_pct'] < -0.6) or
                  (silver_analysis['trend_direction'] == 'down' and silver_analysis['total_change_pct'] < -0.8)):
                
                confidence = min(100, abs(silver_analysis['total_change_pct']) * 25)
                signal = {
                    'action': 'BUY',
                    'ticker': 'GOLD',
                    'reason': f"Silver down: {silver_analysis['total_change_pct']:.1f}%",
                    'confidence': confidence,
                    'strategy': self.name
                }
            
            # Random Test Trade
            elif self.trade_count_today == 0 and np.random.random() < 0.2:
                signal = {
                    'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
                    'ticker': 'GOLD',
                    'reason': "Random test trade",
                    'confidence': 35,
                    'strategy': f"{self.name} (Random)"
                }
            
            # HOLD
            else:
                return {
                    'action': 'HOLD',
                    'ticker': 'GOLD',
                    'reason': "Silver neutral",
                    'confidence': 0,
                    'strategy': self.name
                }
            
            # Trading Parameters für BUY/SELL
            if signal and signal['action'] in ['BUY', 'SELL']:
                gold_price = prices['gold']['current']
                
                # Kleinere TP/SL für häufigere Trades
                sl_percent = 0.6  # 0.6%
                tp_percent = 1.0  # 1.0%
                
                if signal['confidence'] < 50:
                    sl_percent = 0.4
                    tp_percent = 0.8
                
                if signal['action'] == 'BUY':
                    stop_loss = gold_price * (1 - sl_percent/100)
                    take_profit = gold_price * (1 + tp_percent/100)
                else:
                    stop_loss = gold_price * (1 + sl_percent/100)
                    take_profit = gold_price * (1 - tp_percent/100)
                
                signal.update({
                    'current_price': gold_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'stop_distance': max(15, int(abs(gold_price - stop_loss))),
                    'profit_distance': max(15, int(abs(take_profit - gold_price))),
                    'use_percentage_size': True,
                    'percentage': 7.5,  # 7.5% Position Size
                    'position_size': 1.0  # Fallback
                })
                
                self.last_trade_time = current_time
                self.trade_count_today += 1
                
                logger.info(f"Gold Signal: {signal['action']} (7.5% Size)")
            
            return signal
        
        except Exception as e:
            logger.error(f"Gold Signal Error: {e}")
            return None
    
    def get_current_prices(self):
        """Current prices for dashboard"""
        try:
            prices = self.get_simulated_prices()
            return {
                'gold_price': prices['gold']['current'],
                'silver_price': prices['silver']['current'],
                'gold_change': prices['gold']['change_pct'],
                'silver_change': prices['silver']['change_pct']
            }
        except:
            return {}

# === MAIN TRADING BOT CONTROLLER (MEMORY OPTIMIZED) ===
class TradingBotController:
    """Trading Bot mit Memory Management für Render"""
    def __init__(self):
        self.start_time = datetime.now()
        self.running = False
        self.update_thread = None
        self.data_lock = threading.Lock()
        
        # Memory Management
        self.memory_cleanup_interval = 1800  # 30 Minuten
        self.last_memory_cleanup = time.time()
        self.health_check_count = 0
        self.last_health_check = time.time()
        self.restart_count = 0
        
        # Komponenten
        self.rate_limiter = RateLimiter()
        self.trading_hours = TradingHoursManager()
        self.database = DatabaseManager()
        self.main_strategy = MainTradingStrategy()
        self.gold_silver_strategy = GoldSilverStrategy()
        
        # APIs
        self.main_api = None
        self.gold_api = None
        
        # Status
        self.last_update_time = 0
        self.update_interval = int(os.getenv('UPDATE_INTERVAL_MINUTES', 30)) * 60  # Erhöht auf 30min
        self.analysis_count = 0
        self.trade_count = 0
        self.error_count = 0
        
        self.current_status = {
            'trading_hours': {},
            'last_analysis': None,
            'active_positions': [],
            'recent_trades': []
        }
        
        logger.info("Trading Bot Controller initialisiert")
    
    def perform_memory_cleanup(self):
        """Aggressive Memory Cleanup für Render"""
        try:
            current_time = time.time()
            if current_time - self.last_memory_cleanup > self.memory_cleanup_interval:
                logger.info("Führe Memory Cleanup durch...")
                
                # Standard Cleanup
                cleanup_memory()
                
                # Clear Strategy Data
                with self.main_strategy.data_lock:
                    for ticker in list(self.main_strategy.stocks_data.keys()):
                        if 'HistoricalData' in self.main_strategy.stocks_data[ticker]:
                            # Nur letzte 100 Tage behalten
                            self.main_strategy.stocks_data[ticker]['HistoricalData'] = \
                                self.main_strategy.stocks_data[ticker]['HistoricalData'].tail(100)
                
                # Clear Cache
                self.gold_silver_strategy.cache.clear()
                
                # Database Cleanup
                self.database.cleanup_old_data()
                
                # Rate Limiter Cleanup
                self.rate_limiter._cleanup_old_requests()
                
                # Clear Recent Trades (nur letzte 20 behalten)
                if len(self.current_status['recent_trades']) > 20:
                    self.current_status['recent_trades'] = self.current_status['recent_trades'][-20:]
                
                self.last_memory_cleanup = current_time
                logger.info("Memory Cleanup abgeschlossen")
        
        except Exception as e:
            logger.error(f"Memory Cleanup Error: {e}")
    
    def perform_health_check(self):
        """Health Check mit Memory Management"""
        try:
            self.health_check_count += 1
            current_time = time.time()
            
            # Memory Cleanup
            self.perform_memory_cleanup()
            
            # API Status
            main_api_ok = self.main_api is not None and self.main_api.is_authenticated()
            gold_api_ok = self.gold_api is not None and self.gold_api.is_authenticated()
            
            # Restart bei beiden APIs down
            if self.running and not main_api_ok and not gold_api_ok:
                logger.warning("Beide APIs down - restart...")
                self.restart_apis()
            
            self.last_health_check = current_time
            
            # Log jeden 10. Health Check
            if self.health_check_count % 10 == 0:
                logger.info(f"Health Check #{self.health_check_count}: Main: {'OK' if main_api_ok else 'FAIL'}, Gold: {'OK' if gold_api_ok else 'FAIL'}")
        
        except Exception as e:
            logger.error(f"Health Check Error: {e}")
    
    def restart_apis(self):
        """API Restart"""
        try:
            self.restart_count += 1
            logger.info(f"API Restart #{self.restart_count}")
            time.sleep(5)
            
            success = self.initialize_apis()
            if success:
                logger.info("API Restart erfolgreich")
            else:
                logger.error("API Restart fehlgeschlagen")
        except Exception as e:
            logger.error(f"API Restart Error: {e}")
    
    @safe_execute
    def initialize_apis(self):
        """APIs initialisieren"""
        try:
            # Memory cleanup vor API Init
            cleanup_memory()
            
            self.main_api = CapitalComAPI(self.rate_limiter, account_type="main")
            success_main = self.main_api.authenticate()
            
            if success_main:
                time.sleep(2)
                
                self.gold_api = CapitalComAPI(self.rate_limiter, account_type="demo1")
                success_gold = self.gold_api.authenticate()
                
                if success_gold:
                    self.gold_api.switch_account("demo1")
                    logger.info("APIs initialisiert")
                    return True
        
        except Exception as e:
            logger.error(f"API Init Error: {e}")
        
        return False
    
    def is_trading_allowed(self):
        """Trading erlaubt prüfen"""
        trading_status = self.trading_hours.get_trading_status()
        self.current_status['trading_hours'] = trading_status
        return trading_status['analysis_allowed']
    
    def start_auto_trading(self):
        """Auto-Trading starten"""
        try:
            if self.running:
                logger.info("Auto-Trading läuft bereits")
                return True
            
            logger.info("Auto-Trading wird gestartet...")
            
            if not self.initialize_apis():
                logger.error("API Init fehlgeschlagen")
                return False
            
            self.running = True
            
            if self.update_thread is None or not self.update_thread.is_alive():
                self.update_thread = threading.Thread(target=self._auto_trading_loop, daemon=True)
                self.update_thread.start()
                logger.info("Auto-Trading Thread gestartet")
            
            return True
        
        except Exception as e:
            logger.error(f"Auto-Trading Start Error: {e}")
            self.running = False
            return False
    
    def _auto_trading_loop(self):
        """Main Trading Loop mit Memory Management"""
        logger.info("Auto-Trading Loop gestartet")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Health Check alle 2 Minuten
                if current_time - self.last_health_check > 120:
                    self.perform_health_check()
                
                # Update Check
                if current_time - self.last_update_time >= self.update_interval:
                    if self.is_trading_allowed():
                        logger.info("Führe Trading-Analyse durch...")
                        self.run_analysis_cycle()
                        self.last_update_time = current_time
                    else:
                        logger.info("Außerhalb Handelszeiten")
                
                # 60 Sekunden warten (reduziert CPU Load)
                time.sleep(60)
            
            except Exception as e:
                logger.error(f"Trading Loop Error: {e}")
                self.error_count += 1
                time.sleep(120)
    
    @safe_execute
    def run_analysis_cycle(self):
        """Analysis Cycle"""
        logger.info("Starte Analyse-Zyklus...")
        
        if not self.is_trading_allowed():
            return False
        
        # API Check
        try:
            if self.main_api:
                self.main_api.ensure_authenticated()
            if self.gold_api:
                self.gold_api.ensure_authenticated()
        except Exception as e:
            logger.error(f"API Check Error: {e}")
        
        try:
            analysis_success = False
            
            # Main Strategy
            if self.current_status['trading_hours'].get('main_strategy_trading', False):
                logger.info("Hauptstrategie-Analyse...")
                
                if self.main_strategy.fetch_historical_data():
                    if self.main_strategy.calculate_technical_indicators():
                        main_signals = self.main_strategy.generate_trade_signals()
                        
                        # Save Analysis Data
                        for ticker, stock_data in self.main_strategy.stocks_data.items():
                            technical = stock_data.get("Technical", {})
                            info = stock_data.get("Info", {})
                            
                            self.database.save_analysis(ticker, {
                                'price': info.get('CurrentPrice', 0),
                                'score': 50,
                                'rating': 'Analyzed',
                                'rsi': technical.get('RSI', 0),
                                'macd': technical.get('MACD_Histogram', 0),
                                'volatility': technical.get('Volatility', 0)
                            })
                        
                        # Execute Trades
                        if main_signals and self.main_api:
                            main_trades = self.execute_main_strategy_trades(main_signals)
                            self.current_status['recent_trades'].extend(main_trades)
                        
                        analysis_success = True
                        logger.info(f"Hauptstrategie: {len(main_signals)} Signale")
            
            # Gold/Silver Strategy
            if self.current_status['trading_hours'].get('gold_silver_trading', False):
                logger.info("Gold/Silver-Strategie...")
                gold_signal = self.gold_silver_strategy.generate_gold_trade_signal()
                
                if gold_signal and gold_signal['action'] in ['BUY', 'SELL'] and self.gold_api:
                    gold_trades = self.execute_gold_silver_trades([gold_signal])
                    self.current_status['recent_trades'].extend(gold_trades)
                    analysis_success = True
                    logger.info(f"Gold/Silver: {gold_signal['action']}")
            
            # Update Positions
            self.update_positions()
            
            if analysis_success:
                self.analysis_count += 1
                self.current_status['last_analysis'] = datetime.now().isoformat()
                logger.info(f"✅ Analyse #{self.analysis_count} abgeschlossen")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Analysis Cycle Error: {e}")
            self.error_count += 1
            return False
    
    @safe_execute
    def execute_main_strategy_trades(self, signals):
        """Main Strategy Trades"""
        executed_trades = []
        
        if not self.main_api or not self.main_api.is_authenticated():
            return executed_trades
        
        self.main_api.switch_account("main")
        
        # Max 3 Trades pro Cycle
        for signal in signals[:3]:
            try:
                logger.info(f"Main Trade: {signal['ticker']} {signal['action']}")
                
                result = self.main_api.place_order(
                    ticker=signal['ticker'],
                    direction=signal['action'],
                    size=signal['position_size'],
                    stop_distance=signal['stop_distance'],
                    profit_distance=signal['profit_distance']
                )
                
                if result:
                    trade_data = {
                        'ticker': signal['ticker'],
                        'action': signal['action'],
                        'score': signal['score'],
                        'position_size': signal['position_size'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'status': 'executed',
                        'deal_reference': result.get('dealReference', ''),
                        'account_type': 'demo_main',
                        'strategy': 'main',
                        'result': result
                    }
                    
                    self.database.save_trade(trade_data)
                    executed_trades.append(trade_data)
                    self.trade_count += 1
                    self.main_strategy.daily_trade_count += 1
                    
                    logger.info(f"Main Trade erfolgreich: {signal['ticker']}")
                
                time.sleep(2)  # Rate Limit
            
            except Exception as e:
                logger.error(f"Main Trade Error {signal['ticker']}: {e}")
        
        logger.info(f"Main Strategy: {len(executed_trades)} Trades")
        return executed_trades
    
    @safe_execute
    def execute_gold_silver_trades(self, signals):
        """Gold/Silver Trades mit 7.5% Position Size"""
        executed_trades = []
        
        if not self.gold_api or not self.gold_api.is_authenticated():
            return executed_trades
        
        self.gold_api.switch_account("demo1")
        
        for signal in signals[:1]:
            try:
                if signal['action'] == 'HOLD':
                    logger.info(f"Gold/Silver HOLD: {signal['reason']}")
                    continue
                
                logger.info(f"Gold/Silver: {signal['ticker']} {signal['action']} (7.5% Size)")
                
                result = self.gold_api.place_order(
                    ticker=signal['ticker'],
                    direction=signal['action'],
                    size=None,
                    stop_distance=signal['stop_distance'],
                    profit_distance=signal['profit_distance'],
                    use_percentage_size=True,
                    percentage=7.5
                )
                
                if result:
                    actual_size = signal.get('position_size', 1.0)
                    
                    trade_data = {
                        'ticker': signal['ticker'],
                        'action': signal['action'],
                        'score': signal['confidence'],
                        'position_size': actual_size,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'status': 'executed',
                        'deal_reference': result.get('dealReference', ''),
                        'account_type': 'demo_account1',
                        'strategy': 'gold_silver',
                        'result': result
                    }
                    
                    self.database.save_trade(trade_data)
                    executed_trades.append(trade_data)
                    self.trade_count += 1
                    
                    logger.info(f"Gold/Silver Trade erfolgreich (7.5% Size)")
            
            except Exception as e:
                logger.error(f"Gold/Silver Trade Error: {e}")
        
        return executed_trades
    
    @safe_execute
    def update_positions(self):
        """Positionen aktualisieren"""
        all_positions = []
        
        try:
            if self.main_api and self.main_api.is_authenticated():
                self.main_api.switch_account("main")
                main_pos = self.main_api.get_positions()
                for pos in main_pos:
                    pos['account_type'] = 'demo_main'
                    pos['strategy'] = 'main'
                all_positions.extend(main_pos)
            
            if self.gold_api and self.gold_api.is_authenticated():
                self.gold_api.switch_account("demo1")
                gold_pos = self.gold_api.get_positions()
                for pos in gold_pos:
                    pos['account_type'] = 'demo_account1'
                    pos['strategy'] = 'gold_silver'
                all_positions.extend(gold_pos)
            
            self.current_status['active_positions'] = all_positions
        
        except Exception as e:
            logger.error(f"Update Positions Error: {e}")
    
    def stop_auto_trading(self):
        """Trading stoppen"""
        if self.running:
            logger.info("Stoppe Auto-Trading...")
            self.running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            logger.info("Auto-Trading gestoppt")
    
    def get_comprehensive_status(self):
        """Status für Dashboard"""
        try:
            main_api_status = "Connected" if (self.main_api and self.main_api.is_authenticated()) else "Disconnected"
            gold_api_status = "Connected" if (self.gold_api and self.gold_api.is_authenticated()) else "Disconnected"
            
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600
            
            trading_status = self.trading_hours.get_trading_status()
            
            # Balances
            main_balance = "N/A"
            gold_balance = "N/A"
            
            try:
                if self.main_api and self.main_api.is_authenticated():
                    main_info = self.main_api.get_account_info()
                    if main_info and 'accounts' in main_info:
                        for acc in main_info['accounts']:
                            if acc.get('accountId') == self.main_api.current_account:
                                main_balance = f"{acc.get('balance', {}).get('balance', 0):.2f}"
                                break
                
                if self.gold_api and self.gold_api.is_authenticated():
                    gold_info = self.gold_api.get_account_info()
                    if gold_info and 'accounts' in gold_info:
                        for acc in gold_info['accounts']:
                            if acc.get('accountId') == self.gold_api.current_account:
                                gold_balance = f"{acc.get('balance', {}).get('balance', 0):.2f}"
                                break
            except:
                pass
            
            # Gold/Silver Prices
            try:
                gold_silver_prices = self.gold_silver_strategy.get_current_prices()
            except:
                gold_silver_prices = {}
            
            next_update_seconds = max(0, self.update_interval - (time.time() - self.last_update_time))
            next_update_minutes = next_update_seconds / 60
            
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "running" if self.running else "stopped",
                "uptime_hours": round(uptime_hours, 2),
                "apis": {
                    "main_api": main_api_status,
                    "gold_api": gold_api_status
                },
                "trading_hours": trading_status,
                "statistics": {
                    "analysis_count": self.analysis_count,
                    "trade_count": self.trade_count,
                    "error_count": self.error_count,
                    "restart_count": self.restart_count,
                    "health_checks": self.health_check_count,
                    "last_update": self.current_status.get('last_analysis', 'Never'),
                    "next_update_in_minutes": round(next_update_minutes, 1)
                },
                "strategies": {
                    "main_strategy": {
                        "name": self.main_strategy.name,
                        "status": "active" if main_api_status == "Connected" else "inactive",
                        "daily_trades": self.main_strategy.daily_trade_count,
                        "hours_since_open": self.main_strategy.hours_since_market_open
                    },
                    "gold_silver_strategy": {
                        "name": self.gold_silver_strategy.name,
                        "status": "active" if gold_api_status == "Connected" else "inactive",
                        "current_prices": gold_silver_prices,
                        "daily_trades": self.gold_silver_strategy.trade_count_today,
                        "last_trade_minutes_ago": (time.time() - self.gold_silver_strategy.last_trade_time) / 60 if self.gold_silver_strategy.last_trade_time > 0 else 999
                    }
                },
                "positions": self.current_status.get('active_positions', []),
                "recent_trades": self.current_status.get('recent_trades', []),
                'running': self.running,
                'start_time': self.start_time.isoformat(),
                'runtime_hours': round(uptime_hours, 2),
                'last_update': datetime.fromtimestamp(self.last_update_time).isoformat() if self.last_update_time else None,
                'update_interval_minutes': self.update_interval // 60,
                'analysis_count': self.analysis_count,
                'trade_count': self.trade_count,
                'error_count': self.error_count,
                'success_rate': round((self.analysis_count / max(1, self.analysis_count + self.error_count)) * 100, 1),
                'active_positions': len(self.current_status['active_positions']),
                'recent_trades': self.current_status['recent_trades'][-10:],
                'main_api_connected': self.main_api is not None and self.main_api.is_authenticated(),
                'gold_api_connected': self.gold_api is not None and self.gold_api.is_authenticated(),
                'main_account_balance': main_balance,
                'gold_account_balance': gold_balance,
                'gold_silver_prices': gold_silver_prices,
                'strategies': {
                    'main_strategy_active': self.current_status['trading_hours'].get('main_strategy_trading', False),
                    'gold_silver_active': self.current_status['trading_hours'].get('gold_silver_trading', False)
                }
            }
        
        except Exception as e:
            logger.error(f"Status Error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

# FLASK WEB APPLICATION (MEMORY OPTIMIZED)
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Global Bot Instance
trading_bot = None

def initialize_bot():
    """Bot initialisieren mit Memory Management"""
    global trading_bot
    try:
        if trading_bot is None:
            # Memory cleanup vor Bot Init
            cleanup_memory()
            
            trading_bot = TradingBotController()
            
            if os.getenv('TRADING_ENABLED', 'true').lower() == 'true':
                trading_bot.start_auto_trading()
            else:
                logger.info("Trading deaktiviert")
        
        return trading_bot
    
    except Exception as e:
        logger.error(f"Bot Init Error: {e}")
        return None

# Health Check für jeden Request (Render Stability)
@app.before_request
def ensure_bot_health():
    """Bot Health Check vor jedem Request"""
    global trading_bot
    if os.getenv('RENDER'):
        try:
            cleanup_memory()  # Memory cleanup bei jedem Request
            if not trading_bot or not trading_bot.running:
                logger.info("Bot Health Check: Reinitializing...")
                trading_bot = initialize_bot()
        except Exception as e:
            logger.error(f"Health Check Error: {e}")

# Simplified Dashboard (Memory Optimized)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot v2.0 - Memory Optimized</title>
    <meta http-equiv="refresh" content="120">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
            padding: 10px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 2em; margin-bottom: 5px; color: #2c3e50; }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .card h3 { color: #2c3e50; margin-bottom: 8px; }
        .value { font-size: 1.4em; font-weight: bold; margin: 5px 0; }
        .status-online { color: #27ae60; }
        .status-offline { color: #e74c3c; }
        .small-text { font-size: 0.85em; color: #666; }
        .trades-section {
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .trade-item {
            background: #f8f9fa;
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .trade-buy { border-left-color: #28a745; }
        .trade-sell { border-left-color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Trading Bot v2.0 - Memory Optimized</h1>
            <p>{{ timestamp }} | Update: {{ status.update_interval_minutes }}min</p>
        </div>
        
        <div class="status-grid">
            <div class="card">
                <h3>Bot Status</h3>
                <p class="value {{ 'status-online' if status.running else 'status-offline' }}">
                    {{ 'RUNNING' if status.running else 'STOPPED' }}
                </p>
                <small>Laufzeit: {{ "%.1f"|format(status.runtime_hours) }}h</small>
            </div>
            
            <div class="card">
                <h3>Analysen</h3>
                <p class="value">{{ status.analysis_count }}</p>
                <small>Erfolg: {{ status.success_rate }}%</small>
            </div>
            
            <div class="card">
                <h3>Trades</h3>
                <p class="value">{{ status.trade_count }}</p>
                <small>Positionen: {{ status.active_positions }}</small>
            </div>
            
            <div class="card">
                <h3>APIs</h3>
                <p class="value">
                    <span class="{{ 'status-online' if status.main_api_connected else 'status-offline' }}">Main</span> / 
                    <span class="{{ 'status-online' if status.gold_api_connected else 'status-offline' }}">Gold</span>
                </p>
                <small>Restarts: {{ status.statistics.restart_count }}</small>
            </div>
            
            {% if status.gold_silver_prices %}
            <div class="card">
                <h3>Gold/Silver (7.5%)</h3>
                <p class="value">${{ "%.0f"|format(status.gold_silver_prices.gold_price) }} / ${{ "%.1f"|format(status.gold_silver_prices.silver_price) }}</p>
                <small>Heute: {{ status.strategies.gold_silver_strategy.daily_trades }} Trades</small>
            </div>
            {% endif %}
            
            <div class="card">
                <h3>Balances</h3>
                <p class="value">Main: ${{ status.main_account_balance }}</p>
                <p class="value">Gold: ${{ status.gold_account_balance }}</p>
            </div>
        </div>
        
        <div class="trades-section">
            <h2>Recent Trades (Last 5)</h2>
            {% if status.recent_trades %}
                {% for trade in status.recent_trades[-5:] %}
                <div class="trade-item trade-{{ trade.action.lower() }}">
                    <strong>{{ trade.ticker }} - {{ trade.action }}</strong> 
                    <span class="small-text">({{ trade.strategy }})</span><br>
                    <small>Size: {{ trade.position_size }} | {{ trade.account_type }}</small>
                </div>
                {% endfor %}
            {% else %}
                <p>Keine Trades bisher.</p>
            {% endif %}
        </div>
        
        <div class="card">
            <small class="small-text">
                Memory Optimized for Render | 
                Last Update: {{ status.last_update or 'Never' }} | 
                Next in: {{ "%.1f"|format(status.statistics.next_update_in_minutes) }}min
            </small>
        </div>
    </div>
</body>
</html>
"""

# FLASK ROUTES (SIMPLIFIED)
@app.route("/")
def dashboard():
    """Simplified Dashboard"""
    try:
        bot = initialize_bot()
        
        if bot:
            status = bot.get_comprehensive_status()
        else:
            status = {
                'running': False,
                'runtime_hours': 0.0,
                'analysis_count': 0,
                'trade_count': 0,
                'error_count': 0,
                'success_rate': 0.0,
                'update_interval_minutes': 30,
                'last_update': None,
                'active_positions': 0,
                'recent_trades': [],
                'main_api_connected': False,
                'gold_api_connected': False,
                'main_account_balance': 'N/A',
                'gold_account_balance': 'N/A',
                'gold_silver_prices': {},
                'statistics': {
                    'next_update_in_minutes': 0,
                    'restart_count': 0
                },
                'strategies': {
                    'gold_silver_strategy': {'daily_trades': 0}
                }
            }
        
        now = datetime.now()
        return render_template_string(
            DASHBOARD_HTML,
            status=status,
            timestamp=now.strftime("%H:%M:%S"),
            now=now
        )
    
    except Exception as e:
        logger.error(f"Dashboard Error: {e}")
        return f"""
        <html><head><title>Error</title></head>
        <body style="font-family: Arial; padding: 20px;">
        <h1>Dashboard Error</h1>
        <p><strong>Error:</strong> {str(e)}</p>
        <p><strong>Time:</strong> {datetime.now()}</p>
        <a href="/" style="background: #007bff; color: white; padding: 10px; text-decoration: none; border-radius: 5px;">Reload</a>
        </body></html>
        """, 500

@app.route("/api/status")
def api_status():
    """API Status Endpoint"""
    try:
        bot = initialize_bot()
        if bot:
            status = bot.get_comprehensive_status()
            return jsonify(status)
        else:
            return jsonify({
                "error": "Bot nicht verfügbar",
                "timestamp": datetime.now().isoformat()
            }), 500
    except Exception as e:
        logger.error(f"API Status Error: {e}")
        return jsonify({
            "error": "Status Error",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/api/force-analysis", methods=["POST"])
def force_analysis():
    """Force Analysis"""
    try:
        bot = initialize_bot()
        if not bot:
            return jsonify({"error": "Bot nicht verfügbar"}), 500
        
        if not bot.running:
            return jsonify({"error": "Auto-Trading nicht aktiv"}), 400
        
        logger.info("Force Analysis durch API")
        success = bot.run_analysis_cycle()
        
        return jsonify({
            "success": success,
            "message": "Analyse durchgeführt" if success else "Analyse fehlgeschlagen",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Force Analysis Error: {e}")
        return jsonify({
            "error": "Analysis Error",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/health")
def health_check():
    """Health Check für Render"""
    try:
        # Memory cleanup bei Health Check
        cleanup_memory()
        
        bot = initialize_bot()
        is_healthy = bot is not None
        
        health_data = {
            "status": "healthy" if is_healthy else "unhealthy",
            "bot_running": bot.running if bot else False,
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": (datetime.now() - bot.start_time).total_seconds() / 3600 if bot else 0,
            "memory_optimized": True
        }
        
        if is_healthy:
            health_data.update({
                "api_status": {
                    "main_api": bot.main_api is not None and bot.main_api.is_authenticated(),
                    "gold_api": bot.gold_api is not None and bot.gold_api.is_authenticated()
                },
                "statistics": {
                    "analysis_count": bot.analysis_count,
                    "trade_count": bot.trade_count,
                    "error_count": bot.error_count,
                    "restart_count": bot.restart_count,
                    "health_checks": bot.health_check_count
                }
            })
        
        return jsonify(health_data), 200 if is_healthy else 503
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/api/restart-bot", methods=["POST"])
def restart_bot():
    """Bot Restart"""
    try:
        bot = initialize_bot()
        if bot:
            logger.info("Manual Bot Restart")
            bot.restart_apis()
            
            return jsonify({
                "success": True,
                "message": "Bot restarted",
                "restart_count": bot.restart_count,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Bot nicht verfügbar"}), 500
    
    except Exception as e:
        logger.error(f"Bot Restart Error: {e}")
        return jsonify({
            "error": "Restart failed",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/logs")
def view_logs():
    """Simplified Log View"""
    try:
        if not os.path.exists('trading_bot.log'):
            return "<h1>No Log File</h1>"
        
        with open('trading_bot.log', 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Only last 50 lines for memory optimization
        recent_lines = lines[-50:] if len(lines) > 50 else lines
        
        log_html = f"""
        <html>
        <head><title>Bot Logs</title>
        <style>body{{font-family:monospace;background:#1e1e1e;color:#d4d4d4;padding:20px;}}</style>
        </head>
        <body>
        <h1>Trading Bot Logs (Last {len(recent_lines)} lines)</h1>
        <p>Updated: {datetime.now()}</p>
        <hr>
        """
        
        for line in recent_lines:
            css_class = ""
            if "ERROR" in line:
                css_class = 'style="color:#f44747;"'
            elif "WARNING" in line:
                css_class = 'style="color:#ffcc02;"'
            elif "INFO" in line:
                css_class = 'style="color:#4fc1ff;"'
            
            log_html += f'<div {css_class}>{line.strip()}</div>'
        
        log_html += """
        <hr>
        <a href="/">Back to Dashboard</a> | 
        <a href="/logs">Refresh Logs</a>
        </body></html>
        """
        
        return log_html
    
    except Exception as e:
        return f"<h1>Log Error</h1><p>{str(e)}</p>"

# ERROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not found",
        "available_endpoints": ["/", "/api/status", "/health", "/logs"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal error",
        "timestamp": datetime.now().isoformat()
    }), 500

# STARTUP LOGIC
def startup_checks():
    """Startup Validation"""
    logger.info("Trading Bot v2.0 - Memory Optimized startet...")
    logger.info("="*60)
    
    # Environment Variables Check
    required_vars = ['CAPITAL_API_KEY', 'CAPITAL_PASSWORD', 'CAPITAL_EMAIL']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing Environment Variables: {', '.join(missing_vars)}")
        return False
    
    # Configuration
    logger.info("Configuration:")
    logger.info(f"   Trading Enabled: {os.getenv('TRADING_ENABLED', 'true')}")
    logger.info(f"   Update Interval: {os.getenv('UPDATE_INTERVAL_MINUTES', 30)} minutes")
    logger.info(f"   Email: {os.getenv('CAPITAL_EMAIL', 'NOT SET')}")
    
    # Render Environment
    if os.getenv('RENDER'):
        logger.info("Render Environment detected")
        logger.info(f"   Port: {os.getenv('PORT', '10000')}")
        logger.info("   Memory optimization active")
    
    # Features
    logger.info("v2.0 Features:")
    logger.info("   • Memory optimized for Render stability")
    logger.info("   • Fixed position size calculation (min 0.5)")
    logger.info("   • No endless position sync loop")
    logger.info("   • Gold/Silver 7.5% position sizing")
    logger.info("   • Aggressive stock strategy for more trades")
    logger.info("   • Reduced data retention for memory efficiency")
    
    logger.info("="*60)
    return True

def main():
    """Main function for local development"""
    logger.info("Local development started")
    
    if not startup_checks():
        logger.error("Startup checks failed")
        return 1
    
    # Initialize bot
    bot = initialize_bot()
    if not bot:
        logger.error("Bot initialization failed")
        return 1
    
    try:
        port = int(os.getenv("PORT", 5000))
        logger.info(f"Flask app starting on port {port}")
        
        app.run(
            host="0.0.0.0",
            port=port,
            debug=os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            threaded=True
        )
    
    except KeyboardInterrupt:
        logger.info("Manual stop")
        if bot:
            bot.stop_auto_trading()
        return 0
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

# APP INITIALIZATION FOR RENDER
if __name__ == "__main__":
    # Local development
    exit(main())
else:
    # Render/WSGI - Automatic initialization
    logger.info("WSGI/Render environment detected")
    if startup_checks():
        try:
            # Initialize bot for WSGI
            with app.app_context():
                initialize_bot()
                logger.info("Bot successfully initialized for WSGI/Render")
        except Exception as e:
            logger.error(f"WSGI Bot initialization error: {e}")
        logger.info("Flask app ready for WSGI/Render")
    else:
        logger.error("Startup checks failed for WSGI/Render")

# GRACEFUL SHUTDOWN
import signal
import atexit

def signal_handler(sig, frame):
    logger.info("Shutdown signal received...")
    global trading_bot
    if trading_bot:
        trading_bot.stop_auto_trading()
    logger.info("Graceful shutdown completed")

def cleanup_on_exit():
    logger.info("Cleanup on exit...")
    global trading_bot
    if trading_bot:
        trading_bot.stop_auto_trading()
    cleanup_memory()

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_on_exit)

logger.info("Trading Bot v2.0 Module loaded - Memory Optimized for Render!")
