
---

## 📄 **8. `stock_analysis.py` - Hauptdatei (Teil 1):**

```python
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

warnings.filterwarnings('ignore')
load_dotenv()

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_bot.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# === RATE LIMITER ===
class RateLimiter:
    """Capital.com API Rate Limiter - 10 req/sec, 1 req/0.1sec für Trading"""
    
    def __init__(self):
        self.general_requests = []
        self.trading_requests = []
        self.session_requests = []
        self.lock = threading.Lock()
        logger.info("🚦 Rate Limiter initialisiert")
    
    def can_make_request(self, request_type="general"):
        """Prüft ob Request gemacht werden darf"""
        with self.lock:
            now = time.time()
            
            if request_type == "trading":
                # Trading: Max 1 req/0.1 sec
                self.trading_requests = [t for t in self.trading_requests if now - t < 0.1]
                if len(self.trading_requests) >= 1:
                    return False, 0.1 - (now - self.trading_requests[-1])
                self.trading_requests.append(now)
                
            elif request_type == "session":
                # Session: Max 1 req/sec  
                self.session_requests = [t for t in self.session_requests if now - t < 1.0]
                if len(self.session_requests) >= 1:
                    return False, 1.0 - (now - self.session_requests[-1])
                self.session_requests.append(now)
                
            else:
                # General: Max 10 req/sec
                self.general_requests = [t for t in self.general_requests if now - t < 1.0]
                if len(self.general_requests) >= 10:
                    return False, 1.0 - (now - self.general_requests[0])
                self.general_requests.append(now)
            
            return True, 0

    def wait_if_needed(self, request_type="general"):
        """Wartet falls Rate Limit erreicht"""
        can_proceed, wait_time = self.can_make_request(request_type)
        if not can_proceed:
            logger.info(f"⏳ Rate Limit: Warte {wait_time:.2f}s für {request_type} Request")
            time.sleep(wait_time + 0.01)  # Kleine Buffer

# === TRADING HOURS MANAGER ===  
class TradingHoursManager:
    """Handelszeitenbeschränkungen - KOMPLETT STOPP außerhalb Handelszeiten"""
    
    def __init__(self):
        self.market_hours = {
            'NYSE': {
                'timezone': pytz.timezone('US/Eastern'),
                'open_time': dt_time(9, 30),
                'close_time': dt_time(16, 0),
                'weekdays': [0, 1, 2, 3, 4]  # Mon-Fri
            },
            'XETRA': {
                'timezone': pytz.timezone('Europe/Berlin'),
                'open_time': dt_time(9, 0),
                'close_time': dt_time(17, 30),
                'weekdays': [0, 1, 2, 3, 4]
            },
            'FOREX': {
                'timezone': pytz.timezone('UTC'),
                'open_time': dt_time(21, 0),  # Sunday 9 PM
                'close_time': dt_time(22, 0),  # Friday 10 PM
                'weekdays': [0, 1, 2, 3, 4, 6]  # Mon-Fri + Sun
            }
        }
        logger.info("🕐 Trading Hours Manager initialisiert")
    
    def is_market_open(self, market='NYSE'):
        """Prüft ob Markt geöffnet ist"""
        try:
            market_info = self.market_hours.get(market, self.market_hours['NYSE'])
            market_tz = market_info['timezone']
            
            now_utc = datetime.now(pytz.UTC)
            now_market = now_utc.astimezone(market_tz)
            
            # Wochentag prüfen
            if now_market.weekday() not in market_info['weekdays']:
                return False, f"{market} geschlossen (Wochenende)"
            
            # Uhrzeit prüfen
            current_time = now_market.time()
            open_time = market_info['open_time']
            close_time = market_info['close_time']
            
            if market == 'FOREX':
                # Forex: Sonderbehandlung für 24/5
                if now_market.weekday() == 6:  # Sonntag
                    is_open = current_time >= open_time
                elif now_market.weekday() == 4:  # Freitag
                    is_open = current_time <= close_time
                else:  # Mon-Thu
                    is_open = True
            else:
                is_open = open_time <= current_time <= close_time
            
            status = f"{market} {'OFFEN' if is_open else 'GESCHLOSSEN'}"
            return is_open, status
            
        except Exception as e:
            logger.error(f"Marktzeit-Prüfung Fehler: {e}")
            return False, "Zeitprüfung fehlgeschlagen"
    
    def get_trading_status(self):
        """Umfassender Trading-Status für beide Strategien"""
        nyse_open, nyse_status = self.is_market_open('NYSE')
        xetra_open, xetra_status = self.is_market_open('XETRA') 
        forex_open, forex_status = self.is_market_open('FOREX')
        
        # Hauptstrategie: NYSE oder XETRA offen
        main_strategy_allowed = nyse_open or xetra_open
        
        # Gold/Silver: Forex offen
        gold_silver_allowed = forex_open
        
        # KEINE Analyse außerhalb Handelszeiten
        any_market_open = main_strategy_allowed or gold_silver_allowed
        
        return {
            'analysis_allowed': any_market_open,
            'main_strategy_trading': main_strategy_allowed,
            'gold_silver_trading': gold_silver_allowed,
            'nyse_status': nyse_status,
            'xetra_status': xetra_status,
            'forex_status': forex_status,
            'next_open_time': self.get_next_market_open() if not any_market_open else None
        }
    
    def get_next_market_open(self):
        """Berechnet nächste Marktöffnung"""
        try:
            next_times = []
            for market in ['NYSE', 'XETRA', 'FOREX']:
                next_time = self._get_next_open_time(market)
                if next_time:
                    next_times.append((market, next_time))
            
            if next_times:
                next_times.sort(key=lambda x: x[1])
                market, next_time = next_times[0]
                return {
                    'market': market,
                    'time': next_time.isoformat(),
                    'hours_until': (next_time - datetime.now(pytz.UTC)).total_seconds() / 3600
                }
            return None
        except Exception as e:
            logger.error(f"Nächste Marktöffnung Fehler: {e}")
            return None
    
    def _get_next_open_time(self, market):
        """Berechnet nächste Öffnungszeit für spezifischen Markt"""
        try:
            market_info = self.market_hours[market]
            market_tz = market_info['timezone']
            
            now_utc = datetime.now(pytz.UTC)
            now_market = now_utc.astimezone(market_tz)
            
            # Heute noch nicht geöffnet?
            today_open = now_market.replace(
                hour=market_info['open_time'].hour,
                minute=market_info['open_time'].minute,
                second=0, microsecond=0
            )
            
            if (now_market.weekday() in market_info['weekdays'] and 
                now_market.time() < market_info['open_time']):
                return today_open.astimezone(pytz.UTC)
            
            # Nächster Handelstag
            days_ahead = 1
            while days_ahead < 7:
                next_day = now_market + timedelta(days=days_ahead)
                if next_day.weekday() in market_info['weekdays']:
                    next_open = next_day.replace(
                        hour=market_info['open_time'].hour,
                        minute=market_info['open_time'].minute,
                        second=0, microsecond=0
                    )
                    return next_open.astimezone(pytz.UTC)
                days_ahead += 1
            
            return None
        except Exception as e:
            logger.error(f"Nächste Öffnungszeit für {market}: {e}")
            return None

# === SAFE EXECUTION HELPERS ===
def safe_execute(func):
    """Decorator für sichere Funktionsausführung"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fehler in {func.__name__}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
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
    except (ValueError, TypeError, AttributeError):
        return default

def safe_round(value, decimals=2):
    """Sichere Round-Funktion"""
    try:
        return round(safe_float(value), decimals)
    except:
        return 0.0
# === DATABASE MANAGER ===
class DatabaseManager:
    """SQLite Datenbank für Trade-History und Persistierung"""
    
    def __init__(self, db_path="trading_bot.db"):
        self.db_path = db_path
        self.init_database()
        logger.info("💾 Database Manager initialisiert")
    
    def init_database(self):
        """Datenbank und Tabellen erstellen"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                
                # Analysis History
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
                
                # Trade History
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
                
                # Trading Sessions
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        total_trades INTEGER DEFAULT 0,
                        successful_trades INTEGER DEFAULT 0,
                        main_strategy_trades INTEGER DEFAULT 0,
                        gold_silver_trades INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'active'
                    )
                """)
                
                conn.commit()
                logger.info("✓ Datenbank-Tabellen erstellt/geprüft")
        except Exception as e:
            logger.error(f"Datenbank-Initialisierung Fehler: {e}")
    
    @contextmanager
    def get_connection(self):
        """Sichere Datenbankverbindung"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row  # Dict-like access
            yield conn
        except Exception as e:
            logger.error(f"DB Connection Error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    @safe_execute
    def save_analysis(self, ticker, data, strategy="main", account_type="demo"):
        """Analyse-Daten speichern"""
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
            logger.error(f"Analyse-Speicherung Fehler: {e}")
    
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
            logger.error(f"Trade-Speicherung Fehler: {e}")
    
    @safe_execute
    def get_recent_trades(self, limit=10):
        """Letzte Trades abrufen"""
        try:
            with self.get_connection() as conn:
                if conn:
                    cursor = conn.execute("""
                        SELECT * FROM trades 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Trade-Abruf Fehler: {e}")
        return []

# === CAPITAL.COM API CLIENT ===
class CapitalComAPI:
    """Capital.com API Client mit Dual-Account Support"""
    
    def __init__(self, rate_limiter, account_type="main"):
        self.api_key = os.getenv('CAPITAL_API_KEY')
        self.password = os.getenv('CAPITAL_PASSWORD')
        self.email = os.getenv('CAPITAL_EMAIL')
        self.account_type = account_type
        self.rate_limiter = rate_limiter
        
        # URLs
        self.base_url = "https://demo-api-capital.backend-capital.com"
        
        # Session Tokens
        self.cst_token = None
        self.security_token = None
        self.account_id = None
        self.last_auth_time = 0
        self.auth_timeout = 600  # 10 Minuten
        
        # Account Info
        self.available_accounts = []
        self.current_account = None
        
        # Epic Mapping
        self.epic_mapping = {
            'AAPL': 'AAPL', 'MSFT': 'MSFT', 'AMZN': 'AMZN', 'TSLA': 'TSLA',
            'NVDA': 'NVDA', 'GOOGL': 'GOOGL', 'META': 'META', 'NFLX': 'NFLX',
            'SAP.DE': 'SAP', 'DTE.DE': 'DTE',
            'GOLD': 'GOLD', 'SILVER': 'SILVER', 'GLD': 'GOLD', 'SLV': 'SILVER'
        }
        
        logger.info(f"🏦 Capital.com API initialisiert (Account: {account_type})")
    
    def is_authenticated(self):
        """Prüft Session-Status"""
        if not self.cst_token or not self.security_token:
            return False
        return (time.time() - self.last_auth_time) < self.auth_timeout
    
    @safe_execute
    def authenticate(self):
        """Session erstellen mit Rate Limiting"""
        if not self.api_key or not self.password:
            logger.error("❌ API Credentials fehlen - prüfe Environment Variablen")
            logger.error("Benötigt: CAPITAL_API_KEY, CAPITAL_PASSWORD, CAPITAL_EMAIL")
            return False
        
        if self.is_authenticated():
            return True
        
        try:
            # Rate Limit für Session-Requests
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
            
            logger.info("🔐 Capital.com Session wird erstellt...")
            
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
                    
                    # Account-Informationen laden
                    session_data = response.json()
                    self.available_accounts = session_data.get('accounts', [])
                    self.current_account = session_data.get('currentAccountId')
                    
                    logger.info(f"✅ Capital.com authentifiziert")
                    logger.info(f"📊 {len(self.available_accounts)} Accounts verfügbar")
                    logger.info(f"🎯 Current Account: {self.current_account}")
                    
                    # Account-Details loggen
                    for acc in self.available_accounts:
                        acc_name = acc.get('accountName', 'Unknown')
                        acc_id = acc.get('accountId', 'Unknown')
                        balance = acc.get('balance', {}).get('balance', 0)
                        logger.info(f"   💼 {acc_name} (ID: {acc_id[-8:]}***): {balance}")
                    
                    return True
                else:
                    logger.error("❌ Session-Tokens fehlen in Response")
            else:
                logger.error(f"❌ Authentifizierung fehlgeschlagen: {response.status_code}")
                logger.error(f"Response: {response.text}")
                
                if response.status_code == 401:
                    logger.error("🔑 Ungültige Credentials - prüfe API Key und Password")
                elif response.status_code == 429:
                    logger.error("⏳ Rate Limit erreicht - warte vor nächstem Versuch")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API Verbindungsfehler: {e}")
        except Exception as e:
            logger.error(f"❌ Authentifizierung Fehler: {e}")
        
        return False
    
    @safe_execute
    def switch_account(self, target_account_type="demo1"):
        """Account wechseln (Demo Account 1 vs Standard Demo)"""
        if not self.is_authenticated():
            logger.error("❌ Nicht authentifiziert für Account-Wechsel")
            return False
        
        # Ziel-Account finden
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
            logger.error(f"❌ Ziel-Account '{target_account_type}' nicht gefunden")
            return False
        
        target_id = target_account.get('accountId')
        if target_id == self.current_account:
            logger.info(f"✅ Bereits auf Account '{target_account_type}'")
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
                acc_name = target_account.get('accountName', 'Unknown')
                logger.info(f"✅ Account gewechselt zu: {acc_name}")
                return True
            else:
                logger.error(f"❌ Account-Wechsel fehlgeschlagen: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Account-Wechsel Fehler: {e}")
        
        return False
    
    @safe_execute
    def get_positions(self):
        """Aktuelle Positionen abrufen"""
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
            else:
                logger.warning(f"⚠️ Positionen-Abruf Status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Positionen-Abruf Fehler: {e}")
        
        return []
    
    @safe_execute
    def place_order(self, ticker, direction, size=0.1, stop_distance=None, profit_distance=None):
        """Order platzieren mit Deal Confirmation"""
        if not self.is_authenticated():
            logger.error("❌ Nicht authentifiziert für Trading")
            return None
        
        epic = self.epic_mapping.get(ticker, ticker)
        
        try:
            # Trading Rate Limit (max 1/0.1s)
            self.rate_limiter.wait_if_needed("trading")
            
            headers = self._get_auth_headers()
            
            order_data = {
                'epic': epic,
                'direction': str(direction).upper(),
                'size': float(size),
                'guaranteedStop': False
            }
            
            # Stop Loss & Take Profit hinzufügen
            if stop_distance:
                order_data['stopDistance'] = int(stop_distance)
            if profit_distance:
                order_data['profitDistance'] = int(profit_distance)
            
            logger.info(f"📈 Platziere {direction} Order: {epic} (Size: {size})")
            if stop_distance:
                logger.info(f"   🛑 Stop Loss: {stop_distance}")
            if profit_distance:
                logger.info(f"   🎯 Take Profit: {profit_distance}")
            
            response = requests.post(
                f"{self.base_url}/api/v1/positions",
                headers=headers,
                json=order_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                deal_reference = result.get('dealReference')
                
                logger.info(f"✅ Order erstellt: {deal_reference}")
                
                # Deal Confirmation prüfen
                if deal_reference:
                    time.sleep(1)  # Kurz warten für Verarbeitung
                    confirmation = self.check_deal_confirmation(deal_reference)
                    result['confirmation'] = confirmation
                
                return result
            else:
                logger.error(f"❌ Order fehlgeschlagen ({response.status_code}): {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Order-Fehler für {epic}: {e}")
        
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
                deal_id = confirmation.get('dealId')
                
                logger.info(f"📋 Deal Confirmation: {deal_status}")
                if deal_id:
                    logger.info(f"🆔 Deal ID: {deal_id}")
                
                return confirmation
            else:
                logger.warning(f"⚠️ Confirmation nicht verfügbar: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Deal Confirmation Fehler: {e}")
        
        return None
    
    @safe_execute
    def get_account_info(self):
        """Account-Informationen abrufen"""
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
            logger.error(f"❌ Account-Info Fehler: {e}")
        
        return None
    
    def _get_auth_headers(self):
        """Auth-Header für API Requests"""
        if not self.cst_token or not self.security_token:
            return None
        
        return {
            'X-CAP-API-KEY': self.api_key,
            'CST': self.cst_token,
            'X-SECURITY-TOKEN': self.security_token,
            'Content-Type': 'application/json'
        }
    
    def get_current_account_name(self):
        """Name des aktuellen Accounts"""
        for acc in self.available_accounts:
            if acc.get('accountId') == self.current_account:
                return acc.get('accountName', 'Unknown Account')
        return 'Unknown Account'
# === MAIN TRADING STRATEGY ===
class MainTradingStrategy:
    """Haupt-Trading-Strategie basierend auf deinem ursprünglichen Code"""
    
    def __init__(self):
        self.name = "Main Strategy"
        self.stocks_data = {}
        self.data_lock = threading.Lock()
        logger.info(f"📊 {self.name} initialisiert")
    
    def get_stock_list(self):
        """Standard-Aktienliste"""
        return [
            {"Name": "Apple Inc.", "Ticker": "AAPL", "Currency": "USD"},
            {"Name": "Microsoft Corporation", "Ticker": "MSFT", "Currency": "USD"},
            {"Name": "Amazon.com Inc.", "Ticker": "AMZN", "Currency": "USD"},
            {"Name": "Tesla Inc.", "Ticker": "TSLA", "Currency": "USD"},
            {"Name": "NVIDIA Corporation", "Ticker": "NVDA", "Currency": "USD"},
            {"Name": "Alphabet Inc.", "Ticker": "GOOGL", "Currency": "USD"},
            {"Name": "Meta Platforms Inc.", "Ticker": "META", "Currency": "USD"},
            {"Name": "Netflix Inc.", "Ticker": "NFLX", "Currency": "USD"},
            {"Name": "SAP SE", "Ticker": "SAP.DE", "Currency": "EUR"},
            {"Name": "Deutsche Telekom AG", "Ticker": "DTE.DE", "Currency": "EUR"}
        ]
    
    @safe_execute
    def fetch_historical_data(self, period="1y"):
        """Historische Daten laden"""
        stocks_list = self.get_stock_list()
        logger.info(f"📈 Lade Daten für {len(stocks_list)} Aktien...")
        
        success_count = 0
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
                    # Aktueller Preis extrahieren
                    current_price = 0.0
                    if 'Close' in data.columns and len(data['Close'].dropna()) > 0:
                        current_price = float(data['Close'].dropna().iloc[-1])
                    
                    current_price = max(0.01, current_price)
                    
                    stock_info = stock.copy()
                    stock_info["CurrentPrice"] = current_price
                    stock_info["LastUpdate"] = datetime.now().isoformat()
                    
                    with self.data_lock:
                        self.stocks_data[ticker] = {
                            "HistoricalData": data,
                            "Info": stock_info
                        }
                    
                    success_count += 1
                    logger.info(f"✓ {ticker}: {len(data)} Datenpunkte, Preis: ${current_price:.2f}")
                else:
                    logger.warning(f"⚠️ Ungenügend Daten für {ticker}")
                
            except Exception as e:
                logger.error(f"❌ Daten-Fehler für {ticker}: {e}")
        
        logger.info(f"✅ Daten geladen: {success_count}/{len(stocks_list)} erfolgreich")
        return success_count > 0
    
    @safe_execute
    def calculate_technical_indicators(self):
        """Technische Indikatoren berechnen"""
        logger.info("🔢 Berechne technische Indikatoren...")
        
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
                    historical_data['MA200'] = historical_data['Close'].rolling(window=200, min_periods=50).mean()
                    
                    current_price = safe_float(historical_data['Close'].iloc[-1])
                    ma20_current = safe_float(historical_data['MA20'].iloc[-1])
                    ma50_current = safe_float(historical_data['MA50'].iloc[-1])
                    ma200_current = safe_float(historical_data['MA200'].iloc[-1])
                    
                    # MA Verhältnisse
                    technical["Price_vs_MA20"] = safe_round((current_price / ma20_current - 1) * 100) if ma20_current > 0 else 0.0
                    technical["Price_vs_MA50"] = safe_round((current_price / ma50_current - 1) * 100) if ma50_current > 0 else 0.0
                    technical["Price_vs_MA200"] = safe_round((current_price / ma200_current - 1) * 100) if ma200_current > 0 else 0.0
                    
                    # RSI
                    try:
                        delta = historical_data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=10).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=10).mean()
                        rs = gain / loss.replace(0, 0.0001)
                        rsi_series = 100 - (100 / (1 + rs))
                        technical["RSI"] = safe_round(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
                    except Exception:
                        technical["RSI"] = 50.0
                    
                    # MACD
                    try:
                        if len(historical_data) >= 26:
                            ema12 = historical_data['Close'].ewm(span=12, adjust=False).mean()
                            ema26 = historical_data['Close'].ewm(span=26, adjust=False).mean()
                            macd = ema12 - ema26
                            macd_signal = macd.ewm(span=9, adjust=False).mean()
                            macd_histogram = macd - macd_signal
                            technical["MACD"] = safe_round(macd.iloc[-1])
                            technical["MACD_Signal"] = safe_round(macd_signal.iloc[-1])
                            technical["MACD_Histogram"] = safe_round(macd_histogram.iloc[-1])
                        else:
                            technical["MACD"] = 0.0
                            technical["MACD_Signal"] = 0.0
                            technical["MACD_Histogram"] = 0.0
                    except Exception:
                        technical["MACD"] = 0.0
                        technical["MACD_Signal"] = 0.0
                        technical["MACD_Histogram"] = 0.0
                    
                    # Trend Analysis
                    try:
                        if len(historical_data) >= 20:
                            days = np.arange(1, min(31, len(historical_data) + 1))
                            prices = historical_data['Close'].iloc[-len(days):].values
                            if len(prices) == len(days) and not np.any(np.isnan(prices)):
                                model = LinearRegression()
                                model.fit(days.reshape(-1, 1), prices)
                                technical["Trend_Slope"] = safe_round(model.coef_[0], 4)
                                technical["Trend_Strength"] = safe_round(model.score(days.reshape(-1, 1), prices), 4)
                            else:
                                technical["Trend_Slope"] = 0.0
                                technical["Trend_Strength"] = 0.0
                        else:
                            technical["Trend_Slope"] = 0.0
                            technical["Trend_Strength"] = 0.0
                    except Exception:
                        technical["Trend_Slope"] = 0.0
                        technical["Trend_Strength"] = 0.0
                    
                    # Volatilität
                    try:
                        returns = historical_data['Close'].pct_change().dropna()
                        if len(returns) > 5:
                            returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
                            if len(returns_clean) > 0:
                                daily_vol = returns_clean.std() * 100
                                technical["Volatility"] = safe_round(daily_vol, 2)
                                
                                if daily_vol < 1.0:
                                    technical["Volatility_Rating"] = "Very Low"
                                elif daily_vol < 1.5:
                                    technical["Volatility_Rating"] = "Low"
                                elif daily_vol < 2.5:
                                    technical["Volatility_Rating"] = "Medium"
                                elif daily_vol < 4.0:
                                    technical["Volatility_Rating"] = "High"
                                else:
                                    technical["Volatility_Rating"] = "Very High"
                            else:
                                technical["Volatility"] = 2.0
                                technical["Volatility_Rating"] = "Medium"
                        else:
                            technical["Volatility"] = 2.0
                            technical["Volatility_Rating"] = "Medium"
                    except Exception:
                        technical["Volatility"] = 2.0
                        technical["Volatility_Rating"] = "Medium"
                    
                    self.stocks_data[ticker]["Technical"] = technical
                    
                except Exception as e:
                    logger.error(f"❌ Technische Analyse Fehler für {ticker}: {e}")
        
        logger.info("✅ Technische Indikatoren berechnet")
        return True
    
    @safe_execute
    def generate_trade_signals(self):
        """Trade-Signale generieren basierend auf deinem Scoring-System"""
        signals = []
        
        with self.data_lock:
            for ticker, stock_data in self.stocks_data.items():
                try:
                    technical = stock_data.get("Technical", {})
                    stock_info = stock_data.get("Info", {})
                    
                    # Scoring-System von deinem ursprünglichen Code
                    score = 50.0  # Neutral starting score
                    
                    # RSI Factor
                    rsi = technical.get("RSI", 50.0)
                    if rsi < 30:
                        score += 15  # Oversold, positive for buying
                    elif rsi > 70:
                        score -= 15  # Overbought, negative for buying
                    elif 45 <= rsi <= 55:
                        score += 5   # Neutral RSI is slightly positive
                    
                    # Trend Factor
                    trend_slope = technical.get("Trend_Slope", 0.0)
                    trend_strength = technical.get("Trend_Strength", 0.0)
                    if trend_strength > 0.8:
                        if trend_slope > 0:
                            score += 20  # Strong uptrend
                        else:
                            score -= 20  # Strong downtrend
                    elif trend_strength > 0.6:
                        if trend_slope > 0:
                            score += 15
                        else:
                            score -= 15
                    elif trend_strength > 0.4:
                        if trend_slope > 0:
                            score += 10
                        else:
                            score -= 10
                    
                    # MACD Factor
                    macd_hist = technical.get("MACD_Histogram", 0.0)
                    if macd_hist > 1.0:
                        score += 10
                    elif macd_hist > 0.3:
                        score += 5
                    elif macd_hist > 0:
                        score += 2
                    elif macd_hist < -1.0:
                        score -= 10
                    elif macd_hist < -0.3:
                        score -= 5
                    elif macd_hist < 0:
                        score -= 2
                    
                    # Moving Average Factor
                    price_vs_ma20 = technical.get("Price_vs_MA20", 0.0)
                    price_vs_ma50 = technical.get("Price_vs_MA50", 0.0)
                    if price_vs_ma20 > 3 and price_vs_ma50 > 2:
                        score += 10  # Above moving averages
                    elif price_vs_ma20 < -3 and price_vs_ma50 < -2:
                        score -= 10  # Below moving averages
                    
                    # Volatility Factor
                    volatility = technical.get("Volatility", 2.0)
                    vol_rating = technical.get("Volatility_Rating", "Medium")
                    if vol_rating in ["High", "Very High"]:
                        score += 5  # High volatility can be profitable
                    elif vol_rating == "Medium":
                        score += 3
                    elif vol_rating == "Low":
                        score += 1
                    
                    # Score begrenzen
                    score = max(0, min(100, score))
                    
                    # Rating und Action bestimmen
                    if score >= 75:
                        rating = "Strong Long"
                        action = "BUY"
                    elif score >= 65:
                        rating = "Long"
                        action = "BUY"
                    elif score >= 35:
                        rating = "Hold"
                        action = "HOLD"
                    elif score >= 25:
                        rating = "Short"
                        action = "SELL"
                    else:
                        rating = "Strong Short"
                        action = "SELL"
                    
                    # Nur BUY/SELL Signale weiterverarbeiten
                    if action in ['BUY', 'SELL']:
                        current_price = stock_info.get("CurrentPrice", 0)
                        confidence = min(100, abs(score - 50) * 2)
                        
                        # Trading-Parameter berechnen
                        sl_percent = 2.0 if confidence < 60 else 1.5
                        tp_percent = 3.0 if confidence < 60 else 4.0
                        
                        if action == 'BUY':
                            stop_loss = current_price * (1 - sl_percent/100)
                            take_profit = current_price * (1 + tp_percent/100)
                        else:  # SELL
                            stop_loss = current_price * (1 + sl_percent/100)
                            take_profit = current_price * (1 - tp_percent/100)
                        
                        # Stop/Profit Distance berechnen (für Capital.com API)
                        stop_distance = max(10, int(abs(current_price - stop_loss) * 100))
                        profit_distance = max(10, int(abs(take_profit - current_price) * 100))
                        
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
                            'position_size': min(0.1, confidence / 1000),
                            'strategy': self.name,
                            'reason': f"Score: {score:.1f}, RSI: {rsi:.1f}, Trend: {trend_slope:.4f}"
                        }
                        
                        signals.append(signal)
                        
                        logger.info(f"📊 {ticker}: {action} Signal (Score: {score:.1f}, Confidence: {confidence:.0f}%)")
                
                except Exception as e:
                    logger.error(f"❌ Signal-Generierung Fehler für {ticker}: {e}")
        
        logger.info(f"🎯 {self.name}: {len(signals)} Trade-Signale generiert")
        return signals

# === GOLD/SILVER STRATEGY ===
class GoldSilverStrategy:
    """Gold/Silver Test-Strategie für Demo Account 1"""
    
    def __init__(self):
        self.name = "Gold/Silver Test Strategy"
        self.cache = {}
        self.cache_timeout = 300  # 5 Minuten Cache
        logger.info(f"🥇 {self.name} initialisiert")
    
    @safe_execute
    def get_simulated_prices(self):
        """Simulierte Gold/Silber Preise (ohne externe API-Calls)"""
        now = time.time()
        if 'prices' in self.cache and (now - self.cache.get('timestamp', 0)) < self.cache_timeout:
            return self.cache['prices']
        
        # Basis-Preise mit kleinen Variationen
        base_gold = 2000.0
        base_silver = 25.0
        
        # Zufällige Schwankungen simulieren
        gold_variation = (np.random.random() - 0.5) * 40  # ±20
        silver_variation = (np.random.random() - 0.5) * 4  # ±2
        
        gold_change = (np.random.random() - 0.5) * 4  # ±2%
        silver_change = (np.random.random() - 0.5) * 6  # ±3%
        
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
        
        # Cache aktualisieren
        self.cache['prices'] = prices
        self.cache['timestamp'] = now
        
        return prices
    
    @safe_execute
    def analyze_silver_trend(self):
        """Vereinfachte Silber-Trend-Analyse für Gold-Signale"""
        try:
            # Simulierte Trend-Daten
            trend_change = (np.random.random() - 0.5) * 4  # -2% bis +2%
            change_24h = (np.random.random() - 0.5) * 2    # -1% bis +1%
            
            # Manchmal stärkere Bewegungen für Testing
            if np.random.random() < 0.2:  # 20% Chance
                trend_change *= 2
                change_24h *= 1.5
            
            trend_direction = "up" if trend_change > 0 else "down"
            
            analysis = {
                'trend_direction': trend_direction,
                'total_change_pct': trend_change,
                'change_24h_pct': change_24h,
                'is_significant': abs(trend_change) > 1.0,
                'is_strong_trend': abs(trend_change) > 2.0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Silber-Trend Analyse Fehler: {e}")
            return None
    
    @safe_execute
    def generate_gold_trade_signal(self):
        """Gold Trade-Signal basierend auf Silber-Korrelation"""
        try:
            # Aktuelle Preise abrufen
            prices = self.get_simulated_prices()
            if not prices:
                return None
            
            # Silber-Analyse
            silver_analysis = self.analyze_silver_trend()
            if not silver_analysis:
                return None
            
            signal = None
            
            # SELL GOLD Signal: Silber stark gestiegen (inverse Korrelation)
            if (silver_analysis['trend_direction'] == 'up' and 
                silver_analysis['total_change_pct'] > 1.0):
                
                confidence = min(100, abs(silver_analysis['total_change_pct']) * 25)
                signal = {
                    'action': 'SELL',
                    'ticker': 'GOLD',
                    'instrument': 'GOLD',
                    'reason': f"Silber stieg {silver_analysis['total_change_pct']:.1f}% (inverse Korrelation)",
                    'confidence': confidence,
                    'strategy': self.name
                }
            
            # BUY GOLD Signal: Silber gefallen in 24h
            elif silver_analysis['change_24h_pct'] < -0.5:
                
                confidence = min(100, abs(silver_analysis['change_24h_pct']) * 30)
                signal = {
                    'action': 'BUY',
                    'ticker': 'GOLD',
                    'instrument': 'GOLD',
                    'reason': f"Silber fiel {abs(silver_analysis['change_24h_pct']):.1f}% in 24h",
                    'confidence': confidence,
                    'strategy': self.name
                }
            
            # HOLD: Keine signifikanten Bewegungen
            else:
                signal = {
                    'action': 'HOLD',
                    'ticker': 'GOLD',
                    'instrument': 'GOLD',
                    'reason': f"Silber neutral (3T: {silver_analysis['total_change_pct']:.1f}%, 24h: {silver_analysis['change_24h_pct']:.1f}%)",
                    'confidence': 0,
                    'strategy': self.name
                }
            
            # Trading-Parameter hinzufügen für BUY/SELL
            if signal and signal['action'] in ['BUY', 'SELL']:
                gold_price = prices['gold']['current']
                
                # Konservative Parameter für Test-Trades
                sl_percent = 1.0  # 1% Stop Loss
                tp_percent = 2.0  # 2% Take Profit
                
                if signal['action'] == 'BUY':
                    stop_loss = gold_price * (1 - sl_percent/100)
                    take_profit = gold_price * (1 + tp_percent/100)
                else:  # SELL
                    stop_loss = gold_price * (1 + sl_percent/100)
                    take_profit = gold_price * (1 - tp_percent/100)
                
                # Capital.com API Parameter
                stop_distance = max(10, int(abs(gold_price - stop_loss)))
                profit_distance = max(10, int(abs(take_profit - gold_price)))
                
                signal.update({
                    'current_price': gold_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'stop_distance': stop_distance,
                    'profit_distance': profit_distance,
                    'position_size': 0.05,  # Kleine Test-Position
                    'silver_analysis': silver_analysis
                })
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Gold Signal Generierung Fehler: {e}")
            return None
    
    def get_current_prices(self):
        """Aktuelle Gold/Silber Preise für Dashboard"""
        try:
            prices = self.get_simulated_prices()
            return {
                'gold_price': prices['gold']['current'],
                'silver_price': prices['silver']['current'],
                'gold_change': prices['gold']['change_pct'],
                'silver_change': prices['silver']['change_pct']
            }
        except Exception as e:
            logger.error(f"❌ Preis-Abruf Fehler: {e}")
            return {}
# === MAIN TRADING BOT CONTROLLER ===
class TradingBotController:
    """Haupt-Controller für das Dual-Strategy Trading System"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.running = False
        self.update_thread = None
        self.data_lock = threading.Lock()
        
        # Komponenten initialisieren
        self.rate_limiter = RateLimiter()
        self.trading_hours = TradingHoursManager()
        self.database = DatabaseManager()
        
        # Strategien
        self.main_strategy = MainTradingStrategy()
        self.gold_silver_strategy = GoldSilverStrategy()
        
        # Capital.com APIs (2 separate Instanzen für Account-Switching)
        self.main_api = None
        self.gold_api = None
        
        # Status-Variablen
        self.last_update_time = 0
        self.update_interval = int(os.getenv('UPDATE_INTERVAL_MINUTES', 20)) * 60
        self.analysis_count = 0
        self.trade_count = 0
        self.error_count = 0
        
        # Aktueller Status
        self.current_status = {
            'trading_hours': {},
            'last_analysis': None,
            'active_positions': [],
            'recent_trades': []
        }
        
        logger.info("🤖 Trading Bot Controller initialisiert")
    
    @safe_execute
    def initialize_apis(self):
        """Capital.com APIs initialisieren"""
        try:
            # Haupt-API für Standard Demo Account
            self.main_api = CapitalComAPI(self.rate_limiter, account_type="main")
            success_main = self.main_api.authenticate()
            
            if success_main:
                # Gold/Silver API für Demo Account 1
                self.gold_api = CapitalComAPI(self.rate_limiter, account_type="demo1")
                success_gold = self.gold_api.authenticate()
                
                if success_gold:
                    # Auf Demo Account 1 wechseln
                    self.gold_api.switch_account("demo1")
                    
                    logger.info("✅ Beide Capital.com APIs erfolgreich initialisiert")
                    logger.info(f"📊 Main API: {self.main_api.get_current_account_name()}")
                    logger.info(f"🥇 Gold API: {self.gold_api.get_current_account_name()}")
                    return True
                else:
                    logger.error("❌ Gold/Silver API Initialisierung fehlgeschlagen")
            else:
                logger.error("❌ Haupt-API Initialisierung fehlgeschlagen")
        
        except Exception as e:
            logger.error(f"❌ API Initialisierung Fehler: {e}")
        
        return False
    
    def is_trading_allowed(self):
        """Prüft ob Trading/Analyse erlaubt ist (Handelszeitenbeschränkung)"""
        trading_status = self.trading_hours.get_trading_status()
        self.current_status['trading_hours'] = trading_status
        
        if not trading_status['analysis_allowed']:
            if trading_status.get('next_open_time'):
                next_open = trading_status['next_open_time']
                logger.info(f"🕐 Märkte geschlossen. Nächste Öffnung: {next_open['market']} in {next_open['hours_until']:.1f}h")
            return False
        
        return True
    
    @safe_execute
    def run_analysis_cycle(self):
        """Kompletter Analyse- und Trading-Zyklus"""
        logger.info("🔄 Starte Analyse-Zyklus...")
        
        # Handelszeitenbeschränkung prüfen
        if not self.is_trading_allowed():
            logger.info("⏸️ Außerhalb der Handelszeiten - Analyse übersprungen")
            return False
        
        try:
            analysis_success = False
            
            # Hauptstrategie-Analyse
            if self.current_status['trading_hours'].get('main_strategy_trading', False):
                logger.info("📊 Führe Hauptstrategie-Analyse durch...")
                
                # Daten laden und analysieren
                if self.main_strategy.fetch_historical_data():
                    if self.main_strategy.calculate_technical_indicators():
                        # Trade-Signale generieren
                        main_signals = self.main_strategy.generate_trade_signals()
                        
                        # Analyse-Daten speichern
                        for ticker, stock_data in self.main_strategy.stocks_data.items():
                            technical = stock_data.get("Technical", {})
                            info = stock_data.get("Info", {})
                            
                            self.database.save_analysis(ticker, {
                                'price': info.get('CurrentPrice', 0),
                                'score': 50,  # Placeholder
                                'rating': 'Analyzed',
                                'rsi': technical.get('RSI', 0),
                                'macd': technical.get('MACD_Histogram', 0),
                                'volatility': technical.get('Volatility', 0)
                            }, strategy="main", account_type="demo")
                        
                        # Trading ausführen
                        if main_signals and self.main_api:
                            main_trades = self.execute_main_strategy_trades(main_signals)
                            self.current_status['recent_trades'].extend(main_trades)
                        
                        analysis_success = True
                        logger.info(f"✅ Hauptstrategie-Analyse abgeschlossen: {len(main_signals)} Signale")
            
            # Gold/Silver-Strategie
            if self.current_status['trading_hours'].get('gold_silver_trading', False):
                logger.info("🥇 Führe Gold/Silver-Strategie durch...")
                
                gold_signal = self.gold_silver_strategy.generate_gold_trade_signal()
                if gold_signal:
                    # Gold/Silver Trading ausführen
                    if gold_signal['action'] in ['BUY', 'SELL'] and self.gold_api:
                        gold_trades = self.execute_gold_silver_trades([gold_signal])
                        self.current_status['recent_trades'].extend(gold_trades)
                    
                    analysis_success = True
                    logger.info(f"✅ Gold/Silver-Strategie: {gold_signal['action']} Signal")
            
            # Aktuelle Positionen aktualisieren
            self.update_positions()
            
            if analysis_success:
                self.analysis_count += 1
                self.last_update_time = time.time()
                self.current_status['last_analysis'] = datetime.now().isoformat()
                logger.info(f"🎯 Analyse-Zyklus #{self.analysis_count} erfolgreich abgeschlossen")
                return True
            else:
                logger.warning("⚠️ Keine Analyse durchgeführt (Märkte geschlossen)")
                return False
        
        except Exception as e:
            logger.error(f"❌ Analyse-Zyklus Fehler: {e}")
            self.error_count += 1
            return False
    
    @safe_execute
    def execute_main_strategy_trades(self, signals):
        """Hauptstrategie-Trades ausführen"""
        executed_trades = []
        
        if not self.main_api or not self.main_api.is_authenticated():
            logger.error("❌ Haupt-API nicht verfügbar für Trading")
            return executed_trades
        
        # Auf Standard Demo Account sicherstellen
        self.main_api.switch_account("main")
        
        for signal in signals[:3]:  # Max 3 Trades pro Zyklus
            try:
                logger.info(f"📈 Ausführung Hauptstrategie: {signal['ticker']} {signal['action']}")
                
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
                        'deal_id': result.get('confirmation', {}).get('dealId', ''),
                        'account_type': 'demo_main',
                        'strategy': 'main',
                        'result': result
                    }
                    
                    self.database.save_trade(trade_data)
                    executed_trades.append(trade_data)
                    self.trade_count += 1
                    
                    logger.info(f"✅ Hauptstrategie Trade erfolgreich: {signal['ticker']} {signal['action']}")
                else:
                    logger.error(f"❌ Hauptstrategie Trade fehlgeschlagen: {signal['ticker']}")
                
                # Pause zwischen Trades (Rate Limiting)
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Hauptstrategie Trade Fehler für {signal['ticker']}: {e}")
        
        logger.info(f"📊 Hauptstrategie: {len(executed_trades)} Trades ausgeführt")
        return executed_trades
    
    @safe_execute
    def execute_gold_silver_trades(self, signals):
        """Gold/Silver-Trades ausführen"""
        executed_trades = []
        
        if not self.gold_api or not self.gold_api.is_authenticated():
            logger.error("❌ Gold/Silver-API nicht verfügbar für Trading")
            return executed_trades
        
        # Auf Demo Account 1 sicherstellen
        self.gold_api.switch_account("demo1")
        
        for signal in signals[:1]:  # Max 1 Gold/Silver Trade pro Zyklus
            try:
                if signal['action'] == 'HOLD':
                    logger.info(f"⏸️ Gold/Silver HOLD: {signal['reason']}")
                    continue
                
                logger.info(f"🥇 Ausführung Gold/Silver: {signal['ticker']} {signal['action']}")
                
                result = self.gold_api.place_order(
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
                        'score': signal['confidence'],
                        'position_size': signal['position_size'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'status': 'executed',
                        'deal_reference': result.get('dealReference', ''),
                        'deal_id': result.get('confirmation', {}).get('dealId', ''),
                        'account_type': 'demo_account1',
                        'strategy': 'gold_silver',
                        'result': result
                    }
                    
                    self.database.save_trade(trade_data)
                    executed_trades.append(trade_data)
                    self.trade_count += 1
                    
                    logger.info(f"✅ Gold/Silver Trade erfolgreich: {signal['reason']}")
                    logger.info(f"🎯 Confidence: {signal['confidence']:.0f}%")
                else:
                    logger.error(f"❌ Gold/Silver Trade fehlgeschlagen")
                
            except Exception as e:
                logger.error(f"❌ Gold/Silver Trade Fehler: {e}")
        
        logger.info(f"🥇 Gold/Silver: {len(executed_trades)} Trades ausgeführt")
        return executed_trades
    
    @safe_execute
    def update_positions(self):
        """Aktuelle Positionen von beiden Accounts abrufen"""
        all_positions = []
        
        try:
            # Hauptstrategie Positionen
            if self.main_api and self.main_api.is_authenticated():
                self.main_api.switch_account("main")
                main_positions = self.main_api.get_positions()
                for pos in main_positions:
                    pos['account_type'] = 'demo_main'
                    pos['strategy'] = 'main'
                all_positions.extend(main_positions)
            
            # Gold/Silver Positionen
            if self.gold_api and self.gold_api.is_authenticated():
                self.gold_api.switch_account("demo1")
                gold_positions = self.gold_api.get_positions()
                for pos in gold_positions:
                    pos['account_type'] = 'demo_account1'
                    pos['strategy'] = 'gold_silver'
                all_positions.extend(gold_positions)
            
            self.current_status['active_positions'] = all_positions
            
        except Exception as e:
            logger.error(f"❌ Positionen-Update Fehler: {e}")
    
    def start_auto_trading(self):
        """Automatisches Trading starten"""
        if self.running:
            logger.warning("⚠️ Auto-Trading bereits aktiv")
            return False
        
        # APIs initialisieren
        if not self.initialize_apis():
            logger.error("❌ API-Initialisierung fehlgeschlagen - Auto-Trading nicht gestartet")
            return False
        
        self.running = True
        
        def trading_loop():
            logger.info(f"🚀 Auto-Trading gestartet (Update-Intervall: {self.update_interval//60} Minuten)")
            
            # Erste Analyse nach 30 Sekunden
            time.sleep(30)
            
            while self.running:
                try:
                    success = self.run_analysis_cycle()
                    
                    if success:
                        logger.info(f"⏰ Nächste Analyse in {self.update_interval//60} Minuten")
                    else:
                        logger.info(f"⏰ Nächste Prüfung in {self.update_interval//60} Minuten")
                    
                    # Warten bis zum nächsten Zyklus
                    sleep_start = time.time()
                    while self.running and (time.time() - sleep_start) < self.update_interval:
                        time.sleep(60)  # Jede Minute prüfen
                
                except Exception as e:
                    logger.error(f"❌ Trading-Loop Fehler: {e}")
                    if self.running:
                        time.sleep(300)  # 5 Minuten warten bei Fehlern
            
            logger.info("🛑 Auto-Trading gestoppt")
        
        self.update_thread = threading.Thread(target=trading_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("✅ Auto-Trading Thread gestartet")
        return True
    
    def stop_auto_trading(self):
        """Automatisches Trading stoppen"""
        if self.running:
            logger.info("🛑 Stoppe Auto-Trading...")
            self.running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            logger.info("✅ Auto-Trading gestoppt")
    
    def get_comprehensive_status(self):
        """Umfassender Status für Dashboard"""
        runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        # Account-Balances
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
        except Exception as e:
            logger.error(f"❌ Balance-Abruf Fehler: {e}")
        
        # Gold/Silver Preise
        gold_silver_prices = self.gold_silver_strategy.get_current_prices()
        
        # Recent Trades begrenzen
        self.current_status['recent_trades'] = self.current_status['recent_trades'][-10:]
        
        return {
            'running': self.running,
            'start_time': self.start_time.isoformat(),
            'runtime_hours': round(runtime_hours, 2),
            'last_update': datetime.fromtimestamp(self.last_update_time).isoformat() if self.last_update_time else None,
            'update_interval_minutes': self.update_interval // 60,
            'analysis_count': self.analysis_count,
            'trade_count': self.trade_count,
            'error_count': self.error_count,
            'success_rate': round((self.analysis_count / max(1, self.analysis_count + self.error_count)) * 100, 1),
            
            # Trading Status
            'trading_hours': self.current_status['trading_hours'],
            'active_positions': len(self.current_status['active_positions']),
            'recent_trades': self.current_status['recent_trades'],
            
            # Account Informationen
            'main_api_connected': self.main_api is not None and self.main_api.is_authenticated(),
            'gold_api_connected': self.gold_api is not None and self.gold_api.is_authenticated(),
            'main_account_balance': main_balance,
            'gold_account_balance': gold_balance,
            
            # Market Data
            'gold_silver_prices': gold_silver_prices,
            
            # Strategien
            'strategies': {
                'main_strategy_active': self.current_status['trading_hours'].get('main_strategy_trading', False),
                'gold_silver_active': self.current_status['trading_hours'].get('gold_silver_trading', False)
            }
        }

# === FLASK WEB APPLICATION ===
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global Bot Instance
trading_bot = None

def initialize_bot():
    """Trading Bot initialisieren"""
    global trading_bot
    try:
        if trading_bot is None:
            trading_bot = TradingBotController()
            
            # Auto-Trading starten falls aktiviert
            if os.getenv('TRADING_ENABLED', 'true').lower() == 'true':
                trading_bot.start_auto_trading()
            else:
                logger.info("🔧 Trading deaktiviert (TRADING_ENABLED=false)")
        
        return trading_bot
    except Exception as e:
        logger.error(f"❌ Bot-Initialisierung Fehler: {e}")
        return None

# Erweiterte Dashboard HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Dual-Strategy Trading Bot</title>
    <meta http-equiv="refresh" content="60">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
        
        .header {
            background: rgba(255,255,255,0.95);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .header h1 { font-size: 2.2em; margin-bottom: 8px; color: #2c3e50; }
        .header p { color: #7f8c8d; font-size: 1em; }
        
        .trading-hours {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            color: #2c3e50;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .card {
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }
        .card:hover { transform: translateY(-3px); }
        .card h3 { color: #2c3e50; margin-bottom: 12px; font-size: 1.1em; }
        .value { font-size: 1.6em; font-weight: bold; margin: 8px 0; }
        
        .status-online { color: #27ae60; }
        .status-offline { color: #e74c3c; }
        .status-warning { color: #f39c12; }
        
        .accounts-section {
            background: rgba(255,255,255,0.95);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .accounts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .account-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #2196f3;
        }
        
        .trades-section {
            background: rgba(255,255,255,0.95);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .trade-item {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        
        .trade-buy { border-left-color: #28a745; }
        .trade-sell { border-left-color: #dc3545; }
        
        .gold-silver-panel {
            background: linear-gradient(135deg, #ffd700 0%, #c0c0c0 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            color: #2c3e50;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .price-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .price-card {
            background: rgba(255,255,255,0.9);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .footer { text-align: center; margin-top: 30px; color: rgba(255,255,255,0.8); }
        .timestamp { color: #95a5a6; font-size: 0.85em; }
        .small-text { font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Dual-Strategy Trading Bot</h1>
            <p>Hauptstrategie (Demo Account) + Gold/Silver Test (Demo Account 1)</p>
            <p class="timestamp">{{ timestamp }} | Update alle {{ status.update_interval_minutes }} Minuten</p>
        </div>
        
        <!-- Trading Hours Status -->
        <div class="trading-hours">
            <h2>🕐 Handelszeiten-Status</h2>
            <div class="price-grid">
                <div class="price-card">
                    <h4>📈 NYSE</h4>
                    <div class="{{ 'status-online' if 'OFFEN' in status.trading_hours.nyse_status else 'status-offline' }}">
                        {{ status.trading_hours.nyse_status }}
                    </div>
                </div>
                <div class="price-card">
                    <h4>🇪🇺 XETRA</h4>
                    <div class="{{ 'status-online' if 'OFFEN' in status.trading_hours.xetra_status else 'status-offline' }}">
                        {{ status.trading_hours.xetra_status }}
                    </div>
                </div>
                <div class="price-card">
                    <h4>💱 FOREX</h4>
                    <div class="{{ 'status-online' if 'OFFEN' in status.trading_hours.forex_status else 'status-offline' }}">
                        {{ status.trading_hours.forex_status }}
                    </div>
                </div>
                <div class="price-card">
                    <h4>🎯 Analyse Status</h4>
                    <div class="{{ 'status-online' if status.trading_hours.analysis_allowed else 'status-offline' }}">
                        {{ 'AKTIV' if status.trading_hours.analysis_allowed else 'GESTOPPT' }}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Gold/Silver Prices -->
        {% if status.gold_silver_prices %}
        <div class="gold-silver-panel">
            <h2>🥇 Gold & Silver Live Preise (Simuliert)</h2>
            <div class="price-grid">
                <div class="price-card">
                    <h3>🥇 Gold</h3>
                    <div class="value">${{ "%.2f"|format(status.gold_silver_prices.gold_price) }}</div>
                    <small class="{{ 'status-online' if status.gold_silver_prices.gold_change >= 0 else 'status-offline' }}">
                        {{ "%.2f"|format(status.gold_silver_prices.gold_change) }}% (24h)
                    </small>
                </div>
                <div class="price-card">
                    <h3>🥈 Silver</h3>
                    <div class="value">${{ "%.2f"|format(status.gold_silver_prices.silver_price) }}</div>
                    <small class="{{ 'status-online' if status.gold_silver_prices.silver_change >= 0 else 'status-offline' }}">
                        {{ "%.2f"|format(status.gold_silver_prices.silver_change) }}% (24h)
                    </small>
                </div>
            </div>
        </div>
        {% endif %}
        
        <!-- System Status -->
        <div class="status-grid">
            <div class="card">
                <h3>🤖 Bot Status</h3>
                <p class="value {{ 'status-online' if status.running else 'status-offline' }}">
                    {{ 'RUNNING' if status.running else 'STOPPED' }}
                </p>
                <small>Laufzeit: {{ "%.1f"|format(status.runtime_hours) }}h</small>
            </div>
            <div class="card">
                <h3>📊 Analysen</h3>
                <p class="value">{{ status.analysis_count }}</p>
                <small>Erfolgsrate: {{ status.success_rate }}%</small>
            </div>
            <div class="card">
                <h3>💼 Aktive Positionen</h3>
                <p class="value">{{ status.active_positions }}</p>
                <small>Über beide Accounts</small>
            </div>
            <div class="card">
                <h3>📈 Total Trades</h3>
                <p class="value">{{ status.trade_count }}</p>
                <small>Main + Gold/Silver</small>
            </div>
        </div>
        
        <!-- Account Information -->
        <div class="accounts-section">
            <h2>🏦 Account Status</h2>
            <div class="accounts-grid">
                <div class="account-card">
                    <h3>📊 Main Strategy Account</h3>
                    <p><strong>Status:</strong> 
                        <span class="{{ 'status-online' if status.main_api_connected else 'status-offline' }}">
                            {{ 'CONNECTED' if status.main_api_connected else 'DISCONNECTED' }}
                        </span>
                    </p>
                    <p><strong>Balance:</strong> ${{ status.main_account_balance }}</p>
                    <p><strong>Trading:</strong> 
                        <span class="{{ 'status-online' if status.strategies.main_strategy_active else 'status-offline' }}">
                            {{ 'ACTIVE' if status.strategies.main_strategy_active else 'INACTIVE' }}
                        </span>
                    </p>
                    <p class="small-text">Standard Demo Account</p>
                </div>
                <div class="account-card">
                    <h3>🥇 Gold/Silver Test Account</h3>
                    <p><strong>Status:</strong> 
                        <span class="{{ 'status-online' if status.gold_api_connected else 'status-offline' }}">
                            {{ 'CONNECTED' if status.gold_api_connected else 'DISCONNECTED' }}
                        </span>
                    </p>
                    <p><strong>Balance:</strong> ${{ status.gold_account_balance }}</p>
                    <p><strong>Trading:</strong> 
                        <span class="{{ 'status-online' if status.strategies.gold_silver_active else 'status-offline' }}">
                            {{ 'ACTIVE' if status.strategies.gold_silver_active else 'INACTIVE' }}
                        </span>
                    </p>
                    <p class="small-text">Demo - Account 1</p>
                </div>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="trades-section">
            <h2>📈 Recent Trades</h2>
            {% if status.recent_trades %}
                {% for trade in status.recent_trades[-5:] %}
                <div class="trade-item trade-{{ trade.action.lower() }}">
                    <strong>{{ trade.ticker }} - {{ trade.action }}</strong>
                    <span class="small-text">({{ trade.strategy }})</span><br>
                    <small>
                        Size: {{ trade.position_size }} | 
                        Account: {{ trade.account_type }} | 
                        Status: {{ trade.status }}
                        {% if trade.deal_reference %}
                        | Ref: {{ trade.deal_reference[:20] }}...
                        {% endif %}
                    </small>
                </div>
                {% endfor %}
            {% else %}
                <p>Keine Trades bisher ausgeführt.</p>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>&copy; {{ now.year }} Dual-Strategy Trading Bot - Nur für Demo-Accounts</p>
            <p class="small-text">Letztes Update: {{ status.last_update or 'Noch nicht' }}</p>
        </div>
    </div>
</body>
</html>
# === FLASK ROUTES ===
@app.route("/")
def dashboard():
    """Haupt-Dashboard"""
    try:
        bot = initialize_bot()
        if bot:
            status = bot.get_comprehensive_status()
        else:
            # Fallback-Status wenn Bot nicht initialisiert
            status = {
                'running': False,
                'runtime_hours': 0.0,
                'analysis_count': 0,
                'trade_count': 0,
                'error_count': 0,
                'success_rate': 0.0,
                'update_interval_minutes': 20,
                'last_update': None,
                'active_positions': 0,
                'recent_trades': [],
                'main_api_connected': False,
                'gold_api_connected': False,
                'main_account_balance': 'N/A',
                'gold_account_balance': 'N/A',
                'gold_silver_prices': {},
                'trading_hours': {
                    'analysis_allowed': False,
                    'main_strategy_trading': False,
                    'gold_silver_trading': False,
                    'nyse_status': 'UNBEKANNT',
                    'xetra_status': 'UNBEKANNT',
                    'forex_status': 'UNBEKANNT'
                },
                'strategies': {
                    'main_strategy_active': False,
                    'gold_silver_active': False
                }
            }
        
        now = datetime.now()
        return render_template_string(
            DASHBOARD_HTML,
            status=status,
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
            now=now
        )
        
    except Exception as e:
        logger.error(f"❌ Dashboard Fehler: {e}\n{traceback.format_exc()}")
        return f"""
        <h1>🚨 Dashboard Fehler</h1>
        <p><strong>Fehler:</strong> {str(e)}</p>
        <p><strong>Zeit:</strong> {datetime.now()}</p>
        <hr>
        <p>Prüfe die Logs für weitere Details.</p>
        <a href="/">Dashboard neu laden</a>
        """, 500

@app.route("/api/status")
def api_status():
    """API-Endpoint für Status (JSON)"""
    try:
        bot = initialize_bot()
        if bot:
            status = bot.get_comprehensive_status()
            return jsonify(status)
        else:
            return jsonify({
                "error": "Bot nicht initialisiert",
                "timestamp": datetime.now().isoformat()
            }), 500
    except Exception as e:
        logger.error(f"❌ API Status Fehler: {e}")
        return jsonify({
            "error": "Status konnte nicht geladen werden",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/api/force-analysis", methods=["POST"])
def force_analysis():
    """Manuelle Analyse auslösen"""
    try:
        bot = initialize_bot()
        if not bot:
            return jsonify({"error": "Bot nicht verfügbar"}), 500
        
        if not bot.running:
            return jsonify({"error": "Auto-Trading ist nicht aktiv"}), 400
        
        logger.info("🔄 Manuelle Analyse durch API ausgelöst")
        success = bot.run_analysis_cycle()
        
        return jsonify({
            "success": success,
            "message": "Analyse durchgeführt" if success else "Analyse fehlgeschlagen (möglicherweise außerhalb Handelszeiten)",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Manuelle Analyse Fehler: {e}")
        return jsonify({
            "error": "Analyse-Fehler",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/api/trading-hours")
def api_trading_hours():
    """Aktuelle Handelszeiten-Info"""
    try:
        bot = initialize_bot()
        if bot:
            return jsonify(bot.current_status['trading_hours'])
        else:
            hours_manager = TradingHoursManager()
            return jsonify(hours_manager.get_trading_status())
    except Exception as e:
        logger.error(f"❌ Trading Hours API Fehler: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/positions")
def api_positions():
    """Aktuelle Positionen"""
    try:
        bot = initialize_bot()
        if bot:
            return jsonify({
                "positions": bot.current_status['active_positions'],
                "count": len(bot.current_status['active_positions']),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"positions": [], "count": 0}), 500
    except Exception as e:
        logger.error(f"❌ Positions API Fehler: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/recent-trades")
def api_recent_trades():
    """Letzte Trades"""
    try:
        bot = initialize_bot()
        if bot:
            return jsonify({
                "trades": bot.current_status['recent_trades'][-10:],  # Letzte 10
                "total_trades": bot.trade_count,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"trades": [], "total_trades": 0}), 500
    except Exception as e:
        logger.error(f"❌ Recent Trades API Fehler: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    """Health Check für Render"""
    try:
        bot = initialize_bot()
        is_healthy = bot is not None
        
        return jsonify({
            "status": "healthy" if is_healthy else "unhealthy",
            "bot_running": bot.running if bot else False,
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": (datetime.now() - bot.start_time).total_seconds() / 3600 if bot else 0
        }), 200 if is_healthy else 503
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/logs")
def view_logs():
    """Einfache Log-Ansicht (für Debugging)"""
    try:
        if not os.path.exists('trading_bot.log'):
            return "<h1>Keine Log-Datei gefunden</h1>"
        
        with open('trading_bot.log', 'r') as f:
            lines = f.readlines()
        
        # Nur letzte 100 Zeilen
        recent_lines = lines[-100:] if len(lines) > 100 else lines
        
        log_html = f"""
        <html>
        <head>
            <title>Trading Bot Logs</title>
            <style>
                body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }}
                .log-line {{ margin: 2px 0; }}
                .error {{ color: #f44747; }}
                .warning {{ color: #ffcc02; }}
                .info {{ color: #4fc1ff; }}
            </style>
        </head>
        <body>
            <h1>🤖 Trading Bot Logs (Letzte {len(recent_lines)} Zeilen)</h1>
            <p>Letzte Aktualisierung: {datetime.now()}</p>
            <hr>
            <div>
        """
        
        for line in recent_lines:
            css_class = ""
            if "ERROR" in line:
                css_class = "error"
            elif "WARNING" in line:
                css_class = "warning"
            elif "INFO" in line:
                css_class = "info"
            
            log_html += f'<div class="log-line {css_class}">{line.strip()}</div>'
        
        log_html += """
            </div>
            <hr>
            <p><a href="/">Zurück zum Dashboard</a> | <a href="/logs">Logs aktualisieren</a></p>
        </body>
        </html>
        """
        
        return log_html
        
    except Exception as e:
        return f"<h1>Log-Anzeige Fehler</h1><p>{str(e)}</p>"

# === ERROR HANDLERS ===
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint nicht gefunden",
        "available_endpoints": [
            "/", "/api/status", "/api/trading-hours", 
            "/api/positions", "/health", "/logs"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Interner Server-Fehler",
        "timestamp": datetime.now().isoformat()
    }), 500

# === STARTUP LOGIC ===
def startup_checks():
    """Startup-Validierungen"""
    logger.info("🚀 Trading Bot startet...")
    logger.info("="*60)
    
    # Environment-Variablen prüfen
    required_vars = ['CAPITAL_API_KEY', 'CAPITAL_PASSWORD', 'CAPITAL_EMAIL']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Fehlende Environment-Variablen: {', '.join(missing_vars)}")
        logger.error("🔧 Setze diese in Render Environment Variables:")
        for var in missing_vars:
            logger.error(f"   {var}=your_value_here")
        return False
    
    # Konfiguration loggen
    logger.info("🔧 Konfiguration:")
    logger.info(f"   Trading Enabled: {os.getenv('TRADING_ENABLED', 'true')}")
    logger.info(f"   Update Interval: {os.getenv('UPDATE_INTERVAL_MINUTES', 20)} Minuten")
    logger.info(f"   Debug Mode: {os.getenv('DEBUG_MODE', 'false')}")
    logger.info(f"   Email: {os.getenv('CAPITAL_EMAIL', 'NOT SET')}")
    
    # Render-spezifische Checks
    if os.getenv('RENDER'):
        logger.info(f"🌐 Render Environment erkannt")
        logger.info(f"   Port: {os.getenv('PORT', '10000')}")
        logger.info(f"   Service: {os.getenv('RENDER_SERVICE_NAME', 'Unknown')}")
    
    logger.info("="*60)
    return True

def main():
    """Hauptfunktion für lokale Entwicklung"""
    logger.info("🖥️ Lokale Entwicklung gestartet")
    
    if not startup_checks():
        logger.error("❌ Startup-Checks fehlgeschlagen")
        return 1
    
    # Bot initialisieren
    bot = initialize_bot()
    if not bot:
        logger.error("❌ Bot-Initialisierung fehlgeschlagen")
        return 1
    
    try:
        # Flask App starten
        port = int(os.getenv("PORT", 5000))
        logger.info(f"🌐 Flask App startet auf Port {port}")
        
        app.run(
            host="0.0.0.0", 
            port=port, 
            debug=os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️ Manueller Stopp durch Benutzer")
        if bot:
            bot.stop_auto_trading()
        return 0
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
        return 1

# === APP INITIALISIERUNG FÜR RENDER ===
if __name__ == "__main__":
    # Lokale Entwicklung
    exit(main())
else:
    # Render/WSGI - Automatische Initialisierung
    logger.info("🌐 WSGI/Render Umgebung erkannt")
    
    if startup_checks():
        # Bot im Hintergrund initialisieren
        @app.before_first_request
        def initialize_on_first_request():
            try:
                initialize_bot()
                logger.info("✅ Bot erfolgreich für WSGI/Render initialisiert")
            except Exception as e:
                logger.error(f"❌ WSGI Bot-Initialisierung Fehler: {e}")
        
        logger.info("✅ Flask App bereit für WSGI/Render")
    else:
        logger.error("❌ Startup-Checks für WSGI/Render fehlgeschlagen")

# === GRACEFUL SHUTDOWN ===
import signal
import atexit

def signal_handler(sig, frame):
    logger.info("🛑 Shutdown-Signal erhalten...")
    global trading_bot
    if trading_bot:
        trading_bot.stop_auto_trading()
    logger.info("✅ Graceful Shutdown abgeschlossen")

def cleanup_on_exit():
    logger.info("🧹 Cleanup beim Beenden...")
    global trading_bot
    if trading_bot:
        trading_bot.stop_auto_trading()

# Signal Handler registrieren
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_on_exit)

logger.info("🤖 Trading Bot Module geladen - bereit für Deployment!")
