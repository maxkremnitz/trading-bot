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

# RATE LIMITER
class RateLimiter:
    """Capital.com API Rate Limiter - 10 req/sec, 1 req/0.1sec für Trading"""
    def __init__(self):
        self.general_requests = []
        self.trading_requests = []
        self.session_requests = []
        self.lock = threading.Lock()
        logger.info("Rate Limiter initialisiert")
    
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
            logger.info(f"Rate Limit: Warte {wait_time:.2f}s für {request_type} Request")
            time.sleep(wait_time + 0.01)  # Kleine Buffer

# TRADING HOURS MANAGER
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
        logger.info("Trading Hours Manager initialisiert")
    
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

# SAFE EXECUTION HELPERS
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

# DATABASE MANAGER
class DatabaseManager:
    """SQLite Datenbank für Trade-History und Persistierung"""
    def __init__(self, db_path="trading_bot.db"):
        self.db_path = db_path
        self.init_database()
        logger.info("Database Manager initialisiert")
    
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
                logger.info("Datenbank-Tabellen erstellt/geprüft")
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

# CAPITAL.COM API CLIENT MIT ERROR RECOVERY
class CapitalComAPI:
    """Capital.com API Client mit Dual-Account Support und Error Recovery"""
    def __init__(self, rate_limiter, account_type="main"):
        self.api_key = os.getenv('CAPITAL_API_KEY')
        self.password = os.getenv('CAPITAL_PASSWORD')
        self.email = os.getenv('CAPITAL_EMAIL')
        self.account_type = account_type
        self.rate_limiter = rate_limiter
        
        # URLs gemäß Capital.com API Dokumentation
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
        
        # Position Tracking für Error Recovery
        self.tracked_positions = {}
        self.last_position_sync = 0
        
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
        """Session erstellen mit Rate Limiting"""
        if not self.api_key or not self.password:
            logger.error("API Credentials fehlen - prüfe Environment Variablen")
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
            
            logger.info("Capital.com Session wird erstellt...")
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
                    
                    logger.info("Capital.com authentifiziert")
                    logger.info(f"{len(self.available_accounts)} Accounts verfügbar")
                    logger.info(f"Current Account: {self.current_account}")
                    
                    # Account-Details loggen
                    for acc in self.available_accounts:
                        acc_name = acc.get('accountName', 'Unknown')
                        acc_id = acc.get('accountId', 'Unknown')
                        balance = acc.get('balance', {}).get('balance', 0)
                        logger.info(f"   Account {acc_name} (ID: {acc_id[-8:]}***): {balance}")
                    
                    return True
                else:
                    logger.error("Session-Tokens fehlen in Response")
            else:
                logger.error(f"Authentifizierung fehlgeschlagen: {response.status_code}")
                logger.error(f"Response: {response.text}")
                if response.status_code == 401:
                    logger.error("Ungültige Credentials - prüfe API Key und Password")
                elif response.status_code == 429:
                    logger.error("Rate Limit erreicht - warte vor nächstem Versuch")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API Verbindungsfehler: {e}")
        except Exception as e:
            logger.error(f"Authentifizierung Fehler: {e}")
        
        return False
    
    @safe_execute
    def switch_account(self, target_account_type="demo1"):
        """Account wechseln (Demo Account 1 vs Standard Demo)"""
        if not self.is_authenticated():
            logger.error("Nicht authentifiziert für Account-Wechsel")
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
            logger.error(f"Ziel-Account '{target_account_type}' nicht gefunden")
            return False
        
        target_id = target_account.get('accountId')
        if target_id == self.current_account:
            logger.info(f"Bereits auf Account '{target_account_type}'")
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
                logger.info(f"Account gewechselt zu: {acc_name}")
                
                # Position-Tracking nach Account-Wechsel zurücksetzen
                self.sync_positions_after_manual_changes()
                return True
            else:
                logger.error(f"Account-Wechsel fehlgeschlagen: {response.status_code}")
        except Exception as e:
            logger.error(f"Account-Wechsel Fehler: {e}")
        
        return False
    
    @safe_execute
    def sync_positions_after_manual_changes(self):
        """Synchronisiert Position-Tracking nach manuellen Änderungen"""
        try:
            logger.info("Synchronisiere Positionen nach möglichen manuellen Änderungen...")
            current_positions = self.get_positions()
            
            # Tracking-Daten zurücksetzen
            old_count = len(self.tracked_positions)
            self.tracked_positions.clear()
            
            # Neue Positionen tracken
            for pos in current_positions:
                position_id = pos.get('dealId', pos.get('positionId', ''))
                if position_id:
                    self.tracked_positions[position_id] = {
                        'ticker': pos.get('epic', ''),
                        'direction': pos.get('direction', ''),
                        'size': pos.get('size', 0),
                        'created_time': time.time()
                    }
            
            new_count = len(self.tracked_positions)
            logger.info(f"Position-Sync: {old_count} -> {new_count} Positionen")
            self.last_position_sync = time.time()
            
            return True
        except Exception as e:
            logger.error(f"Position-Sync Fehler: {e}")
            return False
    
    @safe_execute
    def get_positions(self):
        """Aktuelle Positionen abrufen mit Error Recovery"""
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
                positions = data.get('positions', [])
                
                # Regelmäßige Position-Sync (alle 5 Minuten)
                if time.time() - self.last_position_sync > 300:
                    self.sync_positions_after_manual_changes()
                
                return positions
            else:
                logger.warning(f"Positionen-Abruf Status: {response.status_code}")
                # Bei Fehlern Position-Sync versuchen
                if response.status_code in [400, 401]:
                    logger.info("Mögliche Position-Diskrepanz - synchronisiere...")
                    self.sync_positions_after_manual_changes()
        except Exception as e:
            logger.error(f"Positionen-Abruf Fehler: {e}")
        
        return []
    
    @safe_execute
    def get_available_balance(self):
        """Verfügbares Kapital für Position Sizing abrufen"""
        try:
            account_info = self.get_account_info()
            if account_info and 'accounts' in account_info:
                for acc in account_info['accounts']:
                    if acc.get('accountId') == self.current_account:
                        balance_info = acc.get('balance', {})
                        available = balance_info.get('available', 0)
                        total_balance = balance_info.get('balance', 0)
                        logger.info(f"Account Balance - Total: {total_balance}, Available: {available}")
                        return float(available) if available else float(total_balance)
            
            logger.warning("Konnte Balance nicht ermitteln - nutze Fallback")
            return 10000.0  # Fallback für Demo-Account
        except Exception as e:
            logger.error(f"Balance-Abruf Fehler: {e}")
            return 10000.0  # Fallback
    
    @safe_execute
    def calculate_position_size_percentage(self, target_percentage=7.5):
        """Berechnet Position Size basierend auf verfügbarem Kapital"""
        try:
            available_balance = self.get_available_balance()
            target_amount = available_balance * (target_percentage / 100)
            
            # Capital.com Position Size Anpassung
            # Für Demo-Accounts ist oft 0.1 minimum, für größere Beträge entsprechend skalieren
            position_size = max(0.1, target_amount / 2000.0)  # Anpassung basierend auf Asset-Wert
            
            logger.info(f"Position Size Berechnung: {target_percentage}% von {available_balance} = Size {position_size}")
            return round(position_size, 2)
        except Exception as e:
            logger.error(f"Position Size Berechnung Fehler: {e}")
            return 0.1  # Minimum Fallback
    
    @safe_execute
    def place_order(self, ticker, direction, size=None, stop_distance=None, profit_distance=None, use_percentage_size=False, percentage=7.5):
        """Order platzieren mit Deal Confirmation und dynamischer Position Size"""
        if not self.is_authenticated():
            logger.error("Nicht authentifiziert für Trading")
            return None
        
        epic = self.epic_mapping.get(ticker, ticker)
        
        try:
            # Position Size berechnen falls gewünscht
            if use_percentage_size or size is None:
                size = self.calculate_position_size_percentage(percentage)
            
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
            
            logger.info(f"Platziere {direction} Order: {epic} (Size: {size})")
            if stop_distance:
                logger.info(f"   Stop Loss: {stop_distance}")
            if profit_distance:
                logger.info(f"   Take Profit: {profit_distance}")
            
            response = requests.post(
                f"{self.base_url}/api/v1/positions",
                headers=headers,
                json=order_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                deal_reference = result.get('dealReference')
                logger.info(f"Order erstellt: {deal_reference}")
                
                # Deal Confirmation prüfen
                if deal_reference:
                    time.sleep(1)  # Kurz warten für Verarbeitung
                    confirmation = self.check_deal_confirmation(deal_reference)
                    result['confirmation'] = confirmation
                    
                    # Position zu Tracking hinzufügen
                    if confirmation and confirmation.get('dealStatus') == 'ACCEPTED':
                        deal_id = confirmation.get('dealId')
                        if deal_id:
                            self.tracked_positions[deal_id] = {
                                'ticker': ticker,
                                'direction': direction,
                                'size': size,
                                'created_time': time.time()
                            }
                
                return result
            else:
                logger.error(f"Order fehlgeschlagen ({response.status_code}): {response.text}")
                # Bei 400-Fehlern könnte eine Position-Diskrepanz vorliegen
                if response.status_code == 400:
                    self.sync_positions_after_manual_changes()
        except Exception as e:
            logger.error(f"Order-Fehler für {epic}: {e}")
        
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
                logger.info(f"Deal Confirmation: {deal_status}")
                if deal_id:
                    logger.info(f"Deal ID: {deal_id}")
                return confirmation
            else:
                logger.warning(f"Confirmation nicht verfügbar: {response.status_code}")
        except Exception as e:
            logger.error(f"Deal Confirmation Fehler: {e}")
        
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
            logger.error(f"Account-Info Fehler: {e}")
        
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
    
    def ensure_authenticated(self):
        """Stellt sicher, dass Session aktiv ist - Auto-Reconnect"""
        if not self.is_authenticated():
            logger.info(f"[API] {self.account_type} Session abgelaufen - neu authentifizieren...")
            return self.authenticate()
        return True

# === MAIN TRADING STRATEGY (ÜBERARBEITET FÜR MEHR TRADES) ===
class MainTradingStrategy:
    """Haupt-Trading-Strategie mit aggressiveren Parametern für mehr Trades"""
    def __init__(self):
        self.name = "Main Strategy"
        self.stocks_data = {}
        self.data_lock = threading.Lock()
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.hours_since_market_open = 0
        logger.info(f"{self.name} initialisiert")
    
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
    
    def update_daily_trade_tracking(self):
        """Aktualisiert tägliche Trade-Statistiken"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
            logger.info(f"Neuer Handelstag: {today}")
        
        # Geschätzte Stunden seit Marktöffnung (vereinfacht)
        current_hour = datetime.now().hour
        if 15 <= current_hour <= 23:  # NYSE/XETRA etwa geöffnet
            self.hours_since_market_open = current_hour - 15
        else:
            self.hours_since_market_open = 0
    
    @safe_execute
    def fetch_historical_data(self, period="1y"):
        """Historische Daten laden"""
        stocks_list = self.get_stock_list()
        logger.info(f"Lade Daten für {len(stocks_list)} Aktien...")
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
                    logger.info(f"{ticker}: {len(data)} Datenpunkte, Preis: ${current_price:.2f}")
                else:
                    logger.warning(f"Ungenügend Daten für {ticker}")
            
            except Exception as e:
                logger.error(f"Daten-Fehler für {ticker}: {e}")
        
        logger.info(f"Daten geladen: {success_count}/{len(stocks_list)} erfolgreich")
        return success_count > 0
    
    @safe_execute
    def calculate_technical_indicators(self):
        """Technische Indikatoren berechnen"""
        logger.info("Berechne technische Indikatoren...")
        
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
                    logger.error(f"Technische Analyse Fehler für {ticker}: {e}")
        
        logger.info("Technische Indikatoren berechnet")
        return True
    
    @safe_execute
    def generate_trade_signals(self):
        """Trade-Signale mit aggressiveren Parametern für mehr Trades"""
        # Daily Trade Tracking aktualisieren
        self.update_daily_trade_tracking()
        
        signals = []
        with self.data_lock:
            for ticker, stock_data in self.stocks_data.items():
                try:
                    technical = stock_data.get("Technical", {})
                    stock_info = stock_data.get("Info", {})
                    
                    # Aggressiveres Scoring-System
                    score = 50.0  # Neutral starting score
                    
                    # RSI Factor (weniger restriktiv)
                    rsi = technical.get("RSI", 50.0)
                    if rsi < 35:  # Erweitert von 30
                        score += 15  # Oversold, positive for buying
                    elif rsi > 65:  # Erweitert von 70
                        score -= 15  # Overbought, negative for buying
                    elif 40 <= rsi <= 60:  # Erweitert von 45-55
                        score += 8   # Neutral RSI ist mehr positiv
                    
                    # Trend Factor (mehr Gewichtung)
                    trend_slope = technical.get("Trend_Slope", 0.0)
                    trend_strength = technical.get("Trend_Strength", 0.0)
                    
                    if trend_strength > 0.7:  # Gesenkt von 0.8
                        if trend_slope > 0:
                            score += 25  # Erhöht von 20
                        else:
                            score -= 25  # Erhöht von 20
                    elif trend_strength > 0.5:  # Gesenkt von 0.6
                        if trend_slope > 0:
                            score += 18  # Erhöht von 15
                        else:
                            score -= 18
                    elif trend_strength > 0.3:  # Gesenkt von 0.4
                        if trend_slope > 0:
                            score += 12  # Erhöht von 10
                        else:
                            score -= 12
                    
                    # MACD Factor (liberaler)
                    macd_hist = technical.get("MACD_Histogram", 0.0)
                    if macd_hist > 0.5:  # Gesenkt von 1.0
                        score += 12  # Erhöht von 10
                    elif macd_hist > 0.1:  # Gesenkt von 0.3
                        score += 8   # Erhöht von 5
                    elif macd_hist > 0:
                        score += 4   # Erhöht von 2
                    elif macd_hist < -0.5:  # Angepasst
                        score -= 12
                    elif macd_hist < -0.1:
                        score -= 8
                    elif macd_hist < 0:
                        score -= 4
                    
                    # Moving Average Factor (weniger strikt)
                    price_vs_ma20 = technical.get("Price_vs_MA20", 0.0)
                    price_vs_ma50 = technical.get("Price_vs_MA50", 0.0)
                    
                    if price_vs_ma20 > 1.5 and price_vs_ma50 > 1:  # Gesenkt von 3 und 2
                        score += 12  # Erhöht von 10
                    elif price_vs_ma20 < -1.5 and price_vs_ma50 < -1:
                        score -= 12
                    elif price_vs_ma20 > 0.5:  # Neue Regel für kleinere Bewegungen
                        score += 6
                    elif price_vs_ma20 < -0.5:
                        score -= 6
                    
                    # Volatility Factor (mehr Belohnung)
                    volatility = technical.get("Volatility", 2.0)
                    vol_rating = technical.get("Volatility_Rating", "Medium")
                    
                    if vol_rating in ["High", "Very High"]:
                        score += 8  # Erhöht von 5
                    elif vol_rating == "Medium":
                        score += 5  # Erhöht von 3
                    elif vol_rating == "Low":
                        score += 3  # Erhöht von 1
                    
                    # Score begrenzen
                    score = max(0, min(100, score))
                    
                    # Liberalere Rating und Action Schwellen
                    if score >= 68:    # Gesenkt von 75
                        rating = "Strong Long"
                        action = "BUY"
                    elif score >= 58:  # Gesenkt von 65
                        rating = "Long"
                        action = "BUY"
                    elif score >= 42:  # Verengt von 35-65 auf 42-58
                        rating = "Hold"
                        action = "HOLD"
                    elif score >= 32:  # Gesenkt von 25
                        rating = "Short"
                        action = "SELL"
                    else:
                        rating = "Strong Short"
                        action = "SELL"
                    
                    # Nur BUY/SELL Signale weiterverarbeiten
                    if action in ['BUY', 'SELL']:
                        current_price = stock_info.get("CurrentPrice", 0)
                        confidence = min(100, abs(score - 50) * 2)
                        
                        # Aggressivere Trading-Parameter (kleinere TP für mehr Trades)
                        sl_percent = 1.5 if confidence < 60 else 1.2  # Gesenkt von 2.0/1.5
                        tp_percent = 2.0 if confidence < 60 else 2.5  # Gesenkt von 3.0/4.0
                        
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
                            'position_size': min(0.15, confidence / 800),  # Erhöht von 0.1 und 1000
                            'strategy': self.name,
                            'reason': f"Score: {score:.1f}, RSI: {rsi:.1f}, Trend: {trend_slope:.4f}"
                        }
                        
                        signals.append(signal)
                        logger.info(f"{ticker}: {action} Signal (Score: {score:.1f}, Confidence: {confidence:.0f}%)")
                
                except Exception as e:
                    logger.error(f"Signal-Generierung Fehler für {ticker}: {e}")
        
        # Mindestens ein Trade pro Tag forcieren falls nötig
        if len(signals) == 0 and self.daily_trade_count == 0 and self.hours_since_market_open >= 4:
            logger.info("Keine Signale gefunden - versuche Mindest-Trade zu forcieren...")
            signals = self._force_minimum_daily_trade()
        
        logger.info(f"{self.name}: {len(signals)} Trade-Signale generiert (Heute: {self.daily_trade_count} Trades)")
        return signals
    
    def _force_minimum_daily_trade(self):
        """Forciert mindestens einen Trade wenn zu wenig Aktivität"""
        try:
            potential_signals = []
            
            with self.data_lock:
                for ticker, stock_data in self.stocks_data.items():
                    technical = stock_data.get("Technical", {})
                    stock_info = stock_data.get("Info", {})
                    
                    # Vereinfachtes Scoring für Notfall-Trade
                    rsi = technical.get("RSI", 50.0)
                    trend_slope = technical.get("Trend_Slope", 0.0)
                    
                    score_deviation = abs(rsi - 50) + abs(trend_slope * 1000)  # Einfache Abweichung
                    
                    if score_deviation > 5:  # Mindest-Abweichung für Trade
                        action = "BUY" if (rsi < 50 and trend_slope >= 0) or (rsi <= 40) else "SELL"
                        
                        potential_signals.append({
                            'ticker': ticker,
                            'action': action,
                            'rating': 'Forced Minimum',
                            'score': 60 if action == "BUY" else 40,
                            'confidence': min(60, score_deviation * 3),
                            'current_price': stock_info.get("CurrentPrice", 0),
                            'position_size': 0.1,  # Konservativ für Notfall-Trade
                            'strategy': f"{self.name} (Forced)",
                            'reason': f"Minimum Daily Trade - RSI: {rsi:.1f}, Trend: {trend_slope:.4f}",
                            'deviation_score': score_deviation
                        })
            
            if potential_signals:
                # Bestes Signal auswählen (höchste Abweichung)
                best_signal = max(potential_signals, key=lambda x: x['deviation_score'])
                
                # Trading-Parameter hinzufügen
                current_price = best_signal['current_price']
                sl_percent = 1.0  # Konservativ
                tp_percent = 1.5  # Konservativ
                
                if best_signal['action'] == 'BUY':
                    stop_loss = current_price * (1 - sl_percent/100)
                    take_profit = current_price * (1 + tp_percent/100)
                else:
                    stop_loss = current_price * (1 + sl_percent/100)
                    take_profit = current_price * (1 - tp_percent/100)
                
                best_signal.update({
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'stop_distance': max(10, int(abs(current_price - stop_loss) * 100)),
                    'profit_distance': max(10, int(abs(take_profit - current_price) * 100))
                })
                
                logger.info(f"Forciere Minimum-Trade: {best_signal['ticker']} {best_signal['action']} (Abweichung: {best_signal['deviation_score']:.1f})")
                return [best_signal]
        
        except Exception as e:
            logger.error(f"Minimum Trade Force Fehler: {e}")
        
        return []

# GOLD/SILVER STRATEGY (ÜBERARBEITET FÜR 7.5% POSITION SIZE UND HÄUFIGERE TRADES)
class GoldSilverStrategy:
    """Gold/Silver Strategie mit 7.5% Position Size und häufigeren kleinen Trades"""
    def __init__(self):
        self.name = "Gold/Silver Test Strategy"
        self.cache = {}
        self.cache_timeout = 180  # 3 Minuten Cache (häufigere Updates)
        self.trade_count_today = 0
        self.last_trade_time = 0
        self.min_trade_interval = 1800  # 30 Minuten zwischen Trades
        logger.info(f"{self.name} initialisiert")
    
    @safe_execute
    def get_simulated_prices(self):
        """Simulierte Gold/Silber Preise mit mehr Variationen"""
        now = time.time()
        if 'prices' in self.cache and (now - self.cache.get('timestamp', 0)) < self.cache_timeout:
            return self.cache['prices']
        
        # Basis-Preise mit mehr Variationen für häufigere Signale
        base_gold = 2000.0
        base_silver = 25.0
        
        # Größere Schwankungen für mehr Trade-Opportunities
        gold_variation = (np.random.random() - 0.5) * 60  # ±30 (erhöht von ±20)
        silver_variation = (np.random.random() - 0.5) * 6   # ±3 (erhöht von ±2)
        
        gold_change = (np.random.random() - 0.5) * 6    # ±3% (erhöht von ±2%)
        silver_change = (np.random.random() - 0.5) * 8  # ±4% (erhöht von ±3%)
        
        # Gelegentlich stärkere Bewegungen (öfter als vorher)
        if np.random.random() < 0.4:  # 40% statt 20%
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
        
        # Cache aktualisieren
        self.cache['prices'] = prices
        self.cache['timestamp'] = now
        
        return prices
    
    @safe_execute
    def analyze_silver_trend(self):
        """Erweiterte Silber-Trend-Analyse für häufigere Signale"""
        try:
            # Mehr variierende Trend-Daten
            trend_change = (np.random.random() - 0.5) * 6  # ±3% (erhöht von ±2%)
            change_24h = (np.random.random() - 0.5) * 3    # ±1.5% (erhöht von ±1%)
            
            # Häufiger stärkere Bewegungen
            if np.random.random() < 0.3:  # 30% statt 20%
                trend_change *= 1.5
                change_24h *= 1.2
            
            # Zusätzlicher intraday momentum indicator
            intraday_momentum = (np.random.random() - 0.5) * 4  # ±2%
            
            trend_direction = "up" if trend_change > 0 else "down"
            
            analysis = {
                'trend_direction': trend_direction,
                'total_change_pct': trend_change,
                'change_24h_pct': change_24h,
                'intraday_momentum': intraday_momentum,
                'is_significant': abs(trend_change) > 0.8,  # Gesenkt von 1.0
                'is_strong_trend': abs(trend_change) > 1.8,  # Gesenkt von 2.0
                'has_momentum': abs(intraday_momentum) > 1.0
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Silber-Trend Analyse Fehler: {e}")
            return None
    
    @safe_execute
    def generate_gold_trade_signal(self):
        """Gold Trade-Signal mit liberaleren Parametern und 7.5% Position Size"""
        try:
            # Rate Limiting prüfen
            current_time = time.time()
            if current_time - self.last_trade_time < self.min_trade_interval:
                remaining_time = self.min_trade_interval - (current_time - self.last_trade_time)
                logger.info(f"Gold/Silver: Trade-Interval aktiv, noch {remaining_time/60:.1f} Minuten")
                return None
            
            # Aktuelle Preise abrufen
            prices = self.get_simulated_prices()
            if not prices:
                return None
            
            # Silber-Analyse
            silver_analysis = self.analyze_silver_trend()
            if not silver_analysis:
                return None
            
            signal = None
            
            # SELL GOLD Signal: Erweiterte Bedingungen
            if ((silver_analysis['trend_direction'] == 'up' and silver_analysis['total_change_pct'] > 0.8) or  # Gesenkt von 1.0
                (silver_analysis['change_24h_pct'] > 1.0) or  # Neue Bedingung
                (silver_analysis['has_momentum'] and silver_analysis['intraday_momentum'] > 1.2)):  # Neue Bedingung
                
                confidence = min(100, (abs(silver_analysis['total_change_pct']) + abs(silver_analysis['change_24h_pct'])) * 20)
                
                signal = {
                    'action': 'SELL',
                    'ticker': 'GOLD',
                    'instrument': 'GOLD',
                    'reason': f"Silber Signale: 3T: {silver_analysis['total_change_pct']:.1f}%, 24h: {silver_analysis['change_24h_pct']:.1f}%, Mom: {silver_analysis['intraday_momentum']:.1f}%",
                    'confidence': confidence,
                    'strategy': self.name
                }
            
            # BUY GOLD Signal: Erweiterte Bedingungen
            elif ((silver_analysis['change_24h_pct'] < -0.6) or  # Gesenkt von -0.5
                  (silver_analysis['trend_direction'] == 'down' and silver_analysis['total_change_pct'] < -0.8) or  # Neue Bedingung
                  (silver_analysis['has_momentum'] and silver_analysis['intraday_momentum'] < -1.0)):  # Neue Bedingung
                
                confidence = min(100, (abs(silver_analysis['total_change_pct']) + abs(silver_analysis['change_24h_pct'])) * 25)
                
                signal = {
                    'action': 'BUY',
                    'ticker': 'GOLD',
                    'instrument': 'GOLD',
                    'reason': f"Silber Schwäche: 3T: {silver_analysis['total_change_pct']:.1f}%, 24h: {silver_analysis['change_24h_pct']:.1f}%, Mom: {silver_analysis['intraday_momentum']:.1f}%",
                    'confidence': confidence,
                    'strategy': self.name
                }
            
            # Gelegentlicher zufälliger Trade wenn lange nichts (für Testing)
            elif (self.trade_count_today == 0 and 
                  np.random.random() < 0.15):  # 15% Chance für Random Trade
                
                random_action = 'BUY' if np.random.random() > 0.5 else 'SELL'
                signal = {
                    'action': random_action,
                    'ticker': 'GOLD',
                    'instrument': 'GOLD',
                    'reason': f"Random Test Trade - Silber neutral (3T: {silver_analysis['total_change_pct']:.1f}%, 24h: {silver_analysis['change_24h_pct']:.1f}%)",
                    'confidence': 35,
                    'strategy': f"{self.name} (Random)"
                }
            
            # HOLD: Keine signifikanten Bewegungen
            else:
                signal = {
                    'action': 'HOLD',
                    'ticker': 'GOLD',
                    'instrument': 'GOLD',
                    'reason': f"Silber neutral (3T: {silver_analysis['total_change_pct']:.1f}%, 24h: {silver_analysis['change_24h_pct']:.1f}%, Mom: {silver_analysis['intraday_momentum']:.1f}%)",
                    'confidence': 0,
                    'strategy': self.name
                }
            
            # Trading-Parameter für BUY/SELL hinzufügen
            if signal and signal['action'] in ['BUY', 'SELL']:
                gold_price = prices['gold']['current']
                
                # Kleinere TP/SL für häufigere Trades (wie gewünscht)
                sl_percent = 0.8   # Gesenkt von 1.0% 
                tp_percent = 1.2   # Gesenkt von 2.0%
                
                # Manchmal noch kleinere Targets für sehr häufige Trades
                if signal['confidence'] < 50:
                    sl_percent = 0.6
                    tp_percent = 0.9
                
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
                    'use_percentage_size': True,  # 7.5% Position Size verwenden
                    'percentage': 7.5,  # 7.5% vom verfügbaren Kapital
                    'position_size': 0.1,  # Fallback falls Percentage-Berechnung fehlschlägt
                    'silver_analysis': silver_analysis,
                    'sl_percent': sl_percent,
                    'tp_percent': tp_percent
                })
                
                # Trade-Zeit aktualisieren
                self.last_trade_time = current_time
                self.trade_count_today += 1
                
                logger.info(f"Gold Signal generiert: {signal['action']} (7.5% Position Size)")
                logger.info(f"  TP: {tp_percent}%, SL: {sl_percent}%, Confidence: {signal['confidence']}%")
            
            return signal
        
        except Exception as e:
            logger.error(f"Gold Signal Generierung Fehler: {e}")
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
            logger.error(f"Preis-Abruf Fehler: {e}")
            return {}

# === MAIN TRADING BOT CONTROLLER (RENDER STABILITY IMPROVEMENTS) ===
class TradingBotController:
    """Haupt-Controller mit Render Platform Stability Verbesserungen"""
    def __init__(self):
        self.start_time = datetime.now()
        self.running = False
        self.update_thread = None
        self.data_lock = threading.Lock()
        
        # Render-spezifische Stabilität
        self.health_check_count = 0
        self.last_health_check = time.time()
        self.restart_count = 0
        
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
        
        logger.info("Trading Bot Controller initialisiert")
    
    def perform_health_check(self):
        """Regelmäßiger Health Check für Render Stability"""
        try:
            self.health_check_count += 1
            current_time = time.time()
            
            # Memory Management für Render
            if os.getenv('RENDER'):
                import gc
                gc.collect()
            
            # API Status prüfen
            main_api_ok = self.main_api is not None and self.main_api.is_authenticated()
            gold_api_ok = self.gold_api is not None and self.gold_api.is_authenticated()
            
            # Restart falls nötig
            if self.running and not main_api_ok and not gold_api_ok:
                logger.warning("Beide APIs disconnected - versuche Restart...")
                self.restart_apis()
            
            self.last_health_check = current_time
            
            if self.health_check_count % 10 == 0:  # Alle 10 Health Checks loggen
                logger.info(f"Health Check #{self.health_check_count}: Main API: {'OK' if main_api_ok else 'FAIL'}, Gold API: {'OK' if gold_api_ok else 'FAIL'}")
        
        except Exception as e:
            logger.error(f"Health Check Fehler: {e}")
    
    def restart_apis(self):
        """API Restart nach Verbindungsproblemen"""
        try:
            self.restart_count += 1
            logger.info(f"API Restart #{self.restart_count}")
            
            # Kurze Pause
            time.sleep(5)
            
            # APIs neu initialisieren
            success = self.initialize_apis()
            if success:
                logger.info("API Restart erfolgreich")
            else:
                logger.error("API Restart fehlgeschlagen")
        
        except Exception as e:
            logger.error(f"API Restart Fehler: {e}")
    
    @safe_execute
    def initialize_apis(self):
        """Capital.com APIs initialisieren mit Error Recovery"""
        try:
            # Haupt-API für Standard Demo Account
            self.main_api = CapitalComAPI(self.rate_limiter, account_type="main")
            success_main = self.main_api.authenticate()
            
            if success_main:
                # 2 Sekunden warten für Rate Limit
                time.sleep(2)
                
                # Gold/Silver API für Demo Account 1
                self.gold_api = CapitalComAPI(self.rate_limiter, account_type="demo1")
                success_gold = self.gold_api.authenticate()
                
                if success_gold:
                    # Auf Demo Account 1 wechseln
                    self.gold_api.switch_account("demo1")
                    
                    logger.info("Beide Capital.com APIs erfolgreich initialisiert")
                    logger.info(f"Main API: {self.main_api.get_current_account_name()}")
                    logger.info(f"Gold API: {self.gold_api.get_current_account_name()}")
                    return True
                else:
                    logger.error("Gold/Silver API Initialisierung fehlgeschlagen")
            else:
                logger.error("Haupt-API Initialisierung fehlgeschlagen")
        
        except Exception as e:
            logger.error(f"API Initialisierung Fehler: {e}")
        
        return False
    
    def is_trading_allowed(self):
        """Prüft ob Trading/Analyse erlaubt ist (Handelszeitenbeschränkung)"""
        trading_status = self.trading_hours.get_trading_status()
        self.current_status['trading_hours'] = trading_status
        
        if not trading_status['analysis_allowed']:
            if trading_status.get('next_open_time'):
                next_open = trading_status['next_open_time']
                logger.info(f"Märkte geschlossen. Nächste Öffnung: {next_open['market']} in {next_open['hours_until']:.1f}h")
            return False
        
        return True
    
    def start_auto_trading(self):
        """Auto-Trading starten mit Render Stability"""
        try:
            if self.running:
                logger.info("Auto-Trading läuft bereits")
                return True
            
            logger.info("Auto-Trading wird gestartet...")
            
            # APIs initialisieren
            if not self.initialize_apis():
                logger.error("API-Initialisierung fehlgeschlagen - Auto-Trading nicht möglich")
                return False
            
            self.running = True
            
            # Update-Thread starten
            if self.update_thread is None or not self.update_thread.is_alive():
                self.update_thread = threading.Thread(target=self._auto_trading_loop, daemon=True)
                self.Request timed out
                                self.update_thread.start()
                logger.info("Auto-Trading Thread gestartet")
            
            return True
        
        except Exception as e:
            logger.error(f"Auto-Trading Start Fehler: {e}")
            self.running = False
            return False
    
    def _auto_trading_loop(self):
        """Haupt-Trading-Loop mit Health Checks"""
        logger.info("Auto-Trading Loop gestartet")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Regelmäßiger Health Check (alle 60 Sekunden)
                if current_time - self.last_health_check > 60:
                    self.perform_health_check()
                
                # Prüfen ob Update fällig ist
                if current_time - self.last_update_time >= self.update_interval:
                    if self.is_trading_allowed():
                        logger.info("Führe Trading-Analyse durch...")
                        self.run_analysis_cycle()
                        self.last_update_time = current_time
                    else:
                        logger.info("Trading außerhalb der Handelszeiten - warte...")
                
                # 30 Sekunden warten vor nächster Prüfung
                time.sleep(30)
            
            except Exception as e:
                logger.error(f"Auto-Trading Loop Fehler: {e}")
                self.error_count += 1
                time.sleep(60)  # Bei Fehler länger warten
    
    @safe_execute
    def run_analysis_cycle(self):
        """Kompletter Analyse- und Trading-Zyklus mit Error Recovery"""
        logger.info("Starte Analyse-Zyklus...")
        
        # Handelszeitenbeschränkung prüfen
        if not self.is_trading_allowed():
            logger.info("Außerhalb der Handelszeiten - Analyse übersprungen")
            return False
        
        # API Sessions prüfen und bei Bedarf erneuern
        try:
            if self.main_api:
                if not self.main_api.ensure_authenticated():
                    logger.error("Hauptstrategie API Reconnect fehlgeschlagen")
            
            if self.gold_api:
                if not self.gold_api.ensure_authenticated():
                    logger.error("Gold/Silver API Reconnect fehlgeschlagen")
            
            logger.info("API Sessions geprüft/erneuert")
        except Exception as e:
            logger.error(f"API Reconnect Fehler: {e}")
        
        try:
            analysis_success = False
            
            # Hauptstrategie-Analyse
            if self.current_status['trading_hours'].get('main_strategy_trading', False):
                logger.info("Führe Hauptstrategie-Analyse durch...")
                
                # Daten laden und analysieren
                if self.main_strategy.fetch_historical_data():
                    if self.main_strategy.calculate_technical_indicators():
                        # Trade-Signale generieren
                        main_signals = self.main_strategy.generate_trade_signals()
                        
                        # Analyse-Daten speichern
                        for ticker, stock_data in self.main_strategy.stocks_data.items():
                            technical = stock_data.get("Technical", {})
                            info = stock_data.get("Info", {})
                            
                            # Score aus vorhandenem Signal extrahieren oder Fallback
                            signal_score = 50
                            for signal in main_signals:
                                if signal['ticker'] == ticker:
                                    signal_score = signal['score']
                                    break
                            
                            self.database.save_analysis(ticker, {
                                'price': info.get('CurrentPrice', 0),
                                'score': signal_score,
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
                        logger.info(f"Hauptstrategie-Analyse abgeschlossen: {len(main_signals)} Signale")
            
            # Gold/Silver-Strategie
            if self.current_status['trading_hours'].get('gold_silver_trading', False):
                logger.info("Führe Gold/Silver-Strategie durch...")
                gold_signal = self.gold_silver_strategy.generate_gold_trade_signal()
                
                if gold_signal:
                    # Gold/Silver Trading ausführen
                    if gold_signal['action'] in ['BUY', 'SELL'] and self.gold_api:
                        gold_trades = self.execute_gold_silver_trades([gold_signal])
                        self.current_status['recent_trades'].extend(gold_trades)
                    
                    analysis_success = True
                    logger.info(f"Gold/Silver-Strategie: {gold_signal['action']} Signal")
            
            # Aktuelle Positionen aktualisieren
            self.update_positions()
            
            if analysis_success:
                self.analysis_count += 1
                self.last_update_time = time.time()
                self.current_status['last_analysis'] = datetime.now().isoformat()
                
                # Erweiterte Logs
                next_analysis = datetime.now() + timedelta(minutes=self.update_interval//60)
                logger.info(f"✅ Analyse-Zyklus #{self.analysis_count} abgeschlossen")
                logger.info(f"📅 Nächste Analyse geplant: {next_analysis.strftime('%H:%M:%S')}")
                logger.info(f"🔗 API Status - Main: {'OK' if self.main_api and self.main_api.is_authenticated() else 'FAIL'}, Gold: {'OK' if self.gold_api and self.gold_api.is_authenticated() else 'FAIL'}")
                
                return True
            else:
                logger.warning("Keine Analyse durchgeführt (Märkte geschlossen)")
                return False
        
        except Exception as e:
            logger.error(f"Analyse-Zyklus Fehler: {e}")
            self.error_count += 1
            return False
    
    @safe_execute
    def execute_main_strategy_trades(self, signals):
        """Hauptstrategie-Trades ausführen"""
        executed_trades = []
        
        if not self.main_api or not self.main_api.is_authenticated():
            logger.error("Haupt-API nicht verfügbar für Trading")
            return executed_trades
        
        # Auf Standard Demo Account sicherstellen
        self.main_api.switch_account("main")
        
        # Mehr Trades pro Zyklus erlauben (war 3, jetzt 5)
        for signal in signals[:5]:
            try:
                logger.info(f"Ausführung Hauptstrategie: {signal['ticker']} {signal['action']}")
                
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
                    
                    # Daily trade count für Hauptstrategie aktualisieren
                    self.main_strategy.daily_trade_count += 1
                    
                    logger.info(f"Hauptstrategie Trade erfolgreich: {signal['ticker']} {signal['action']}")
                else:
                    logger.error(f"Hauptstrategie Trade fehlgeschlagen: {signal['ticker']}")
                
                # Pause zwischen Trades (Rate Limiting)
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Hauptstrategie Trade Fehler für {signal['ticker']}: {e}")
        
        logger.info(f"Hauptstrategie: {len(executed_trades)} Trades ausgeführt")
        return executed_trades
    
    @safe_execute
    def execute_gold_silver_trades(self, signals):
        """Gold/Silver-Trades mit 7.5% Position Size ausführen"""
        executed_trades = []
        
        if not self.gold_api or not self.gold_api.is_authenticated():
            logger.error("Gold/Silver-API nicht verfügbar für Trading")
            return executed_trades
        
        # Auf Demo Account 1 sicherstellen
        self.gold_api.switch_account("demo1")
        
        for signal in signals[:1]:  # Max 1 Gold/Silver Trade pro Zyklus
            try:
                if signal['action'] == 'HOLD':
                    logger.info(f"Gold/Silver HOLD: {signal['reason']}")
                    continue
                
                logger.info(f"Ausführung Gold/Silver: {signal['ticker']} {signal['action']} (7.5% Position Size)")
                logger.info(f"  TP: {signal.get('tp_percent', 1.2)}%, SL: {signal.get('sl_percent', 0.8)}%")
                
                # 7.5% Position Size verwenden
                result = self.gold_api.place_order(
                    ticker=signal['ticker'],
                    direction=signal['action'],
                    size=None,  # Wird durch percentage berechnet
                    stop_distance=signal['stop_distance'],
                    profit_distance=signal['profit_distance'],
                    use_percentage_size=signal.get('use_percentage_size', True),
                    percentage=signal.get('percentage', 7.5)
                )
                
                if result:
                    # Tatsächliche Position Size aus Result extrahieren
                    actual_size = signal.get('position_size', 0.1)
                    if 'confirmation' in result and result['confirmation']:
                        confirmation = result['confirmation']
                        if 'size' in confirmation:
                            actual_size = confirmation['size']
                    
                    trade_data = {
                        'ticker': signal['ticker'],
                        'action': signal['action'],
                        'score': signal['confidence'],
                        'position_size': actual_size,
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
                    
                    logger.info(f"Gold/Silver Trade erfolgreich: {signal['reason']}")
                    logger.info(f"Position Size: {actual_size} (7.5% vom Kapital)")
                    logger.info(f"Confidence: {signal['confidence']:.0f}%")
                else:
                    logger.error("Gold/Silver Trade fehlgeschlagen")
            
            except Exception as e:
                logger.error(f"Gold/Silver Trade Fehler: {e}")
        
        logger.info(f"Gold/Silver: {len(executed_trades)} Trades ausgeführt")
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
            logger.error(f"Positionen-Update Fehler: {e}")
    
    def stop_auto_trading(self):
        """Automatisches Trading stoppen"""
        if self.running:
            logger.info("Stoppe Auto-Trading...")
            self.running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            logger.info("Auto-Trading gestoppt")
    
    def get_comprehensive_status(self):
        """Umfassender Status für Dashboard"""
        try:
            # APIs Status
            main_api_status = "Connected" if (self.main_api and self.main_api.is_authenticated()) else "Disconnected"
            gold_api_status = "Connected" if (self.gold_api and self.gold_api.is_authenticated()) else "Disconnected"
            
            # Laufzeit berechnen
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600
            
            # Trading Hours Status
            trading_status = self.trading_hours.get_trading_status()
            
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
                logger.error(f"Balance-Abruf Fehler: {e}")
            
            # Gold/Silver Preise
            try:
                gold_silver_prices = self.gold_silver_strategy.get_current_prices()
            except:
                gold_silver_prices = {}
            
            # Nächste Update-Zeit
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
                # Dashboard-spezifische Daten
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
                'recent_trades': self.current_status['recent_trades'][-10:],  # Letzte 10
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
            logger.error(f"Status-Abruf Fehler: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

# FLASK WEB APPLICATION MIT RENDER IMPROVEMENTS
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global Bot Instance
trading_bot = None

def initialize_bot():
    """Trading Bot initialisieren mit Render Stability"""
    global trading_bot
    try:
        if trading_bot is None:
            trading_bot = TradingBotController()
            
            # Auto-Trading starten falls aktiviert
            if os.getenv('TRADING_ENABLED', 'true').lower() == 'true':
                trading_bot.start_auto_trading()
            else:
                logger.info("Trading deaktiviert (TRADING_ENABLED=false)")
        
        return trading_bot
    
    except Exception as e:
        logger.error(f"Bot-Initialisierung Fehler: {e}")
        return None

# Render Health Check vor jedem Request
@app.before_request
def ensure_bot_health():
    """Stellt sicher dass Bot läuft (Render Stability)"""
    global trading_bot
    if os.getenv('RENDER'):
        try:
            if not trading_bot or not trading_bot.running:
                logger.info("Bot Health Check: Reinitializing...")
                trading_bot = initialize_bot()
        except Exception as e:
            logger.error(f"Bot Health Check Fehler: {e}")

# Dashboard HTML Template (ERWEITERT)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dual-Strategy Trading Bot v2.0</title>
    <meta http-equiv="refresh" content="60">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { max-width: 1800px; margin: 0 auto; padding: 20px; }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .header h1 { font-size: 2.4em; margin-bottom: 8px; color: #2c3e50; }
        .header p { color: #7f8c8d; font-size: 1em; }
        .version-badge {
            display: inline-block;
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            margin-left: 10px;
        }
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
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .account-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #2196f3;
        }
        .strategy-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .stat-item {
            background: rgba(255,255,255,0.8);
            padding: 8px 12px;
            border-radius: 8px;
            text-align: center;
            font-size: 0.9em;
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
        .health-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .health-green { background-color: #27ae60; }
        .health-red { background-color: #e74c3c; }
        .health-orange { background-color: #f39c12; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Dual-Strategy Trading Bot<span class="version-badge">v2.0 - Enhanced</span></h1>
            <p>Hauptstrategie (Demo Account) + Gold/Silver 7.5% Strategy (Demo Account 1)</p>
            <p class="timestamp">{{ timestamp }} | Update alle {{ status.update_interval_minutes }} Minuten | Nächste in {{ "%.1f"|format(status.statistics.next_update_in_minutes) }}min</p>
        </div>
        
        <!-- Trading Hours Status -->
        <div class="trading-hours">
            <h2>🕐 Handelszeiten-Status</h2>
            <div class="price-grid">
                <div class="price-card">
                    <h4>NYSE</h4>
                    <div class="{{ 'status-online' if 'OFFEN' in status.trading_hours.nyse_status else 'status-offline' }}">
                        <span class="health-indicator {{ 'health-green' if 'OFFEN' in status.trading_hours.nyse_status else 'health-red' }}"></span>
                        {{ status.trading_hours.nyse_status }}
                    </div>
                </div>
                <div class="price-card">
                    <h4>XETRA</h4>
                    <div class="{{ 'status-online' if 'OFFEN' in status.trading_hours.xetra_status else 'status-offline' }}">
                        <span class="health-indicator {{ 'health-green' if 'OFFEN' in status.trading_hours.xetra_status else 'health-red' }}"></span>
                        {{ status.trading_hours.xetra_status }}
                    </div>
                </div>
                <div class="price-card">
                    <h4>FOREX</h4>
                    <div class="{{ 'status-online' if 'OFFEN' in status.trading_hours.forex_status else 'status-offline' }}">
                        <span class="health-indicator {{ 'health-green' if 'OFFEN' in status.trading_hours.forex_status else 'health-red' }}"></span>
                        {{ status.trading_hours.forex_status }}
                    </div>
                </div>
                <div class="price-card">
                    <h4>Bot Status</h4>
                    <div class="{{ 'status-online' if status.trading_hours.analysis_allowed else 'status-offline' }}">
                        <span class="health-indicator {{ 'health-green' if status.trading_hours.analysis_allowed else 'health-red' }}"></span>
                        {{ 'AKTIV' if status.trading_hours.analysis_allowed else 'GESTOPPT' }}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Gold/Silver Prices -->
        {% if status.gold_silver_prices %}
        <div class="gold-silver-panel">
            <h2>💰 Gold & Silver Live Preise (7.5% Position Size Strategy)</h2>
            <div class="price-grid">
                <div class="price-card">
                    <h3>Gold</h3>
                    <div class="value">${{ "%.2f"|format(status.gold_silver_prices.gold_price) }}</div>
                    <small class="{{ 'status-online' if status.gold_silver_prices.gold_change >= 0 else 'status-offline' }}">
                        {{ "%.2f"|format(status.gold_silver_prices.gold_change) }}% (24h)
                    </small>
                </div>
                <div class="price-card">
                    <h3>Silver</h3>
                    <div class="value">${{ "%.2f"|format(status.gold_silver_prices.silver_price) }}</div>
                    <small class="{{ 'status-online' if status.gold_silver_prices.silver_change >= 0 else 'status-offline' }}">
                        {{ "%.2f"|format(status.gold_silver_prices.silver_change) }}% (24h)
                    </small>
                </div>
                {% if status.strategies.gold_silver_strategy %}
                <div class="price-card">
                    <h3>Heute Trades</h3>
                    <div class="value">{{ status.strategies.gold_silver_strategy.daily_trades }}</div>
                    <small>Letzter vor {{ "%.0f"|format(status.strategies.gold_silver_strategy.last_trade_minutes_ago) }}min</small>
                </div>
                {% endif %}
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
                <small>Laufzeit: {{ "%.1f"|format(status.runtime_hours) }}h | Restarts: {{ status.statistics.restart_count }}</small>
            </div>
            <div class="card">
                <h3>📊 Analysen</h3>
                <p class="value">{{ status.analysis_count }}</p>
                <small>Erfolgsrate: {{ status.success_rate }}% | Health Checks: {{ status.statistics.health_checks }}</small>
            </div>
            <div class="card">
                <h3>📈 Aktive Positionen</h3>
                <p class="value">{{ status.active_positions }}</p>
                <small>Über beide Accounts</small>
            </div>
            <div class="card">
                <h3>🎯 Total Trades</h3>
                <p class="value">{{ status.trade_count }}</p>
                <small>Main + Gold/Silver | Fehler: {{ status.error_count }}</small>
            </div>
        </div>
        
        <!-- Account Information -->
        <div class="accounts-section">
            <h2>💳 Account Status</h2>
            <div class="accounts-grid">
                <div class="account-card">
                    <h3>📈 Main Strategy Account</h3>
                    <p><strong>Status:</strong>
                        <span class="{{ 'status-online' if status.main_api_connected else 'status-offline' }}">
                            <span class="health-indicator {{ 'health-green' if status.main_api_connected else 'health-red' }}"></span>
                            {{ 'CONNECTED' if status.main_api_connected else 'DISCONNECTED' }}
                        </span>
                    </p>
                    <p><strong>Balance:</strong> ${{ status.main_account_balance }}</p>
                    <p><strong>Trading:</strong>
                        <span class="{{ 'status-online' if status.strategies.main_strategy_active else 'status-offline' }}">
                            {{ 'ACTIVE' if status.strategies.main_strategy_active else 'INACTIVE' }}
                        </span>
                    </p>
                    {% if status.strategies.main_strategy %}
                    <div class="strategy-stats">
                        <div class="stat-item">
                            <strong>{{ status.strategies.main_strategy.daily_trades }}</strong><br>
                            <small>Heute Trades</small>
                        </div>
                        <div class="stat-item">
                            <strong>{{ status.strategies.main_strategy.hours_since_open }}h</strong><br>
                            <small>Seit Öffnung</small>
                        </div>
                    </div>
                    {% endif %}
                    <p class="small-text">Standard Demo Account | Aggressivere Parameter</p>
                </div>
                
                <div class="account-card">
                    <h3>🥇 Gold/Silver Test Account</h3>
                    <p><strong>Status:</strong>
                        <span class="{{ 'status-online' if status.gold_api_connected else 'status-offline' }}">
                            <span class="health-indicator {{ 'health-green' if status.gold_api_connected else 'health-red' }}"></span>
                            {{ 'CONNECTED' if status.gold_api_connected else 'DISCONNECTED' }}
                        </span>
                    </p>
                    <p><strong>Balance:</strong> ${{ status.gold_account_balance }}</p>
                    <p><strong>Trading:</strong>
                        <span class="{{ 'status-online' if status.strategies.gold_silver_active else 'status-offline' }}">
                            {{ 'ACTIVE' if status.strategies.gold_silver_active else 'INACTIVE' }}
                        </span>
                    </p>
                    {% if status.strategies.gold_silver_strategy %}
                    <div class="strategy-stats">
                        <div class="stat-item">
                            <strong>{{ status.strategies.gold_silver_strategy.daily_trades }}</strong><br>
                            <small>Heute Trades</small>
                        </div>
                        <div class="stat-item">
                            <strong>7.5%</strong><br>
                            <small>Position Size</small>
                        </div>
                        <div class="stat-item">
                            <strong>{{ "%.0f"|format(status.strategies.gold_silver_strategy.last_trade_minutes_ago) }}min</strong><br>
                            <small>Letzter Trade</small>
                        </div>
                    </div>
                    {% endif %}
                    <p class="small-text">Demo - Account 1 | Kleinere TP für mehr Trades</p>
                </div>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="trades-section">
            <h2>📋 Recent Trades</h2>
            {% if status.recent_trades %}
                {% for trade in status.recent_trades[-8:] %}
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
            <p>&copy; {{ now.year }} Dual-Strategy Trading Bot v2.0 - Nur für Demo-Accounts</p>
            <p class="small-text">Letztes Update: {{ status.last_update or 'Noch nicht' }} | 
            {% if status.statistics.next_update_in_minutes < 5 %}
                <span class="status-warning">Nächstes Update in {{ "%.1f"|format(status.statistics.next_update_in_minutes) }} Minuten</span>
            {% else %}
                Nächstes Update in {{ "%.1f"|format(status.statistics.next_update_in_minutes) }} Minuten
            {% endif %}
            </p>
        </div>
    </div>
</body>
</html>
"""

# FLASK ROUTES
@app.route("/")
def dashboard():
    """Haupt-Dashboard mit Enhanced UI"""
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
                },
                'statistics': {
                    'next_update_in_minutes': 0,
                    'restart_count': 0,
                    'health_checks': 0
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
        logger.error(f"Dashboard Fehler: {e}\n{traceback.format_exc()}")
        return f"""
        <html>
        <head><title>Dashboard Error</title></head>
        <body style="font-family: Arial; padding: 20px; background: #f5f5f5;">
        <h1 style="color: #e74c3c;">Dashboard Fehler</h1>
        <p><strong>Fehler:</strong> {str(e)}</p>
        <p><strong>Zeit:</strong> {datetime.now()}</p>
        <hr>
        <p>Prüfe die Logs für weitere Details.</p>
        <a href="/" style="background: #3498db; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px;">Dashboard neu laden</a>
        <br><br>
        <details>
        <summary>Technische Details</summary>
        <pre style="background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 5px; overflow-x: auto;">{traceback.format_exc()}</pre>
        </details>
        </body>
        </html>
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
        logger.error(f"API Status Fehler: {e}")
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
        
        logger.info("Manuelle Analyse durch API ausgelöst")
        success = bot.run_analysis_cycle()
        
        return jsonify({
            "success": success,
            "message": "Analyse durchgeführt" if success else "Analyse fehlgeschlagen (möglicherweise außerhalb Handelszeiten)",
            "timestamp": datetime.now().isoformat(),
            "next_scheduled": (datetime.now() + timedelta(minutes=bot.update_interval//60)).isoformat()
        })
    
    except Exception as e:
        logger.error(f"Manuelle Analyse Fehler: {e}")
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
        logger.error(f"Trading Hours API Fehler: {e}")
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
                "main_account_positions": [p for p in bot.current_status['active_positions'] if p.get('account_type') == 'demo_main'],
                "gold_account_positions": [p for p in bot.current_status['active_positions'] if p.get('account_type') == 'demo_account1'],
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"positions": [], "count": 0}), 500
    except Exception as e:
        logger.error(f"Positions API Fehler: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/recent-trades")
def api_recent_trades():
    """Letzte Trades"""
    try:
        bot = initialize_bot()
        if bot:
            recent_trades = bot.current_status['recent_trades'][-15:]  # Letzte 15
            
            return jsonify({
                "trades": recent_trades,
                "total_trades": bot.trade_count,
                "main_strategy_trades": len([t for t in recent_trades if t.get('strategy') == 'main']),
                "gold_silver_trades": len([t for t in recent_trades if 'gold_silver' in t.get('strategy', '')]),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"trades": [], "total_trades": 0}), 500
    except Exception as e:
        logger.error(f"Recent Trades API Fehler: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/restart-bot", methods=["POST"])
def restart_bot():
    """Bot-Restart API Endpoint (für Emergency Cases)"""
    try:
        bot = initialize_bot()
        if bot:
            logger.info("Manueller Bot-Restart angefordert")
            bot.restart_apis()
            
            return jsonify({
                "success": True,
                "message": "Bot-Restart durchgeführt",
                "restart_count": bot.restart_count,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Bot nicht verfügbar"}), 500
    
    except Exception as e:
        logger.error(f"Bot Restart Fehler: {e}")
        return jsonify({
            "error": "Restart fehlgeschlagen",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/health")
def health_check():
    """Enhanced Health Check für Render"""
    try:
        bot = initialize_bot()
        is_healthy = bot is not None
        
        # Zusätzliche Health Metrics
        health_data = {
            "status": "healthy" if is_healthy else "unhealthy",
            "bot_running": bot.running if bot else False,
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": (datetime.now() - bot.start_time).total_seconds() / 3600 if bot else 0
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
                },
                "last_update_minutes_ago": (time.time() - bot.last_update_time) / 60 if bot.last_update_time > 0 else 999
            })
        
        return jsonify(health_data), 200 if is_healthy else 503
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/logs")
def view_logs():
    """Enhanced Log-Ansicht"""
    try:
        if not os.path.exists('trading_bot.log'):
            return "<h1>Keine Log-Datei gefunden</h1>"
        
        with open('trading_bot.log', 'r') as f:
            lines = f.readlines()
        
        # Nur letzte 150 Zeilen (mehr für bessere Übersicht)
        recent_lines = lines[-150:] if len(lines) > 150 else lines
        
        log_html = f"""
        <html>
        <head>
            <title>Trading Bot Logs v2.0</title>
            <style>
                body {{ font-family: 'Consolas', 'Monaco', monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; line-height: 1.4; }}
                .log-line {{ margin: 1px 0; padding: 2px 0; }}
                .error {{ color: #f44747; font-weight: bold; }}
                .warning {{ color: #ffcc02; }}
                .info {{ color: #4fc1ff; }}
                .success {{ color: #4CAF50; }}
                .header {{ background: #2d2d30; padding: 15px; margin-bottom: 15px; border-radius: 8px; }}
                .filter-buttons {{ margin: 15px 0; }}
                .filter-btn {{ 
                    background: #007acc; color: white; border: none; padding: 8px 15px; 
                    margin: 0 5px; border-radius: 4px; cursor: pointer; 
                }}
                .filter-btn:hover {{ background: #005999; }}
                .timestamp {{ color: #808080; }}
            </style>
            <script>
            function filterLogs(type) {{
                const lines = document.querySelectorAll('.log-line');
                lines.forEach(line => {{
                    if (type === 'all' || line.classList.contains(type)) {{
                        line.style.display = 'block';
                    }} else {{
                        line.style.display = 'none';
                    }}
                }});
            }}
            </script>
        </head>
        <body>
            <div class="header">
                <h1>🤖 Trading Bot Logs v2.0 (Letzte {len(recent_lines)} Zeilen)</h1>
                <p>Letzte Aktualisierung: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="filter-buttons">
                <button class="filter-btn" onclick="filterLogs('all')">Alle anzeigen</button>
                <button class="filter-btn" onclick="filterLogs('error')">Nur Fehler</button>
                <button class="filter-btn" onclick="filterLogs('warning')">Nur Warnungen</button>
                <button class="filter-btn" onclick="filterLogs('info')">Nur Info</button>
                <button class="filter-btn" onclick="filterLogs('success')">Nur Erfolg</button>
            </div>
            
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
            elif any(keyword in line for keyword in ["erfolgreich", "SUCCESS", "✅", "abgeschlossen"]):
                css_class = "success"
            
            log_html += f'<div class="log-line {css_class}">{line.strip()}</div>'
        
        log_html += """
            </div>
            <hr style="margin: 20px 0;">
            <p style="text-align: center;">
                <a href="/" style="color: #4fc1ff;">Zurück zum Dashboard</a> | 
                <a href="/logs" style="color: #4fc1ff;">Logs aktualisieren</a> |
                <a href="/health" style="color: #4fc1ff;">Health Check</a>
            </p>
        </body>
        </html>
        """
        
        return log_html
    
    except Exception as e:
        return f"""
        <html><head><title>Log Error</title></head><body style="font-family: Arial; padding: 20px;">
        <h1>Log-Anzeige Fehler</h1><p>{str(e)}</p>
        <a href="/">Zurück zum Dashboard</a>
        </body></html>
        """

# ERROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint nicht gefunden",
        "available_endpoints": [
            "/", "/api/status", "/api/trading-hours", "/api/positions", 
            "/api/recent-trades", "/api/force-analysis", "/api/restart-bot",
            "/health", "/logs"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Interner Server-Fehler",
        "timestamp": datetime.now().isoformat()
    }), 500

# STARTUP LOGIC
def startup_checks():
    """Enhanced Startup-Validierungen"""
    logger.info("Trading Bot v2.0 startet...")
    logger.info("="*70)
    
    # Environment-Variablen prüfen
    required_vars = ['CAPITAL_API_KEY', 'CAPITAL_PASSWORD', 'CAPITAL_EMAIL']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Fehlende Environment-Variablen: {', '.join(missing_vars)}")
        logger.error("Setze diese in Render Environment Variables:")
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
        logger.info("🌐 Render Environment erkannt")
        logger.info(f"   Port: {os.getenv('PORT', '10000')}")
        logger.info(f"   Service: {os.getenv('RENDER_SERVICE_NAME', 'Unknown')}")
        logger.info("   Enhanced Stability Features aktiviert")
    
    # Feature-Summary
    logger.info("✨ v2.0 Features:")
    logger.info("   • Aktienstrategie: Aggressivere Parameter für mehr Trades")
    logger.info("   • Gold/Silver: 7.5% Position Size mit kleineren TPs")
    logger.info("   • Error Recovery: Position-Sync nach manuellen Änderungen")
    logger.info("   • Render Stability: Health Checks und Auto-Restart")
    logger.info("   • Enhanced Dashboard mit mehr Statistiken")
    
    logger.info("="*70)
    return True

def main():
    """Hauptfunktion für lokale Entwicklung"""
    logger.info("Lokale Entwicklung gestartet")
    
    if not startup_checks():
        logger.error("Startup-Checks fehlgeschlagen")
        return 1
    
    # Bot initialisieren
    bot = initialize_bot()
    if not bot:
        logger.error("Bot-Initialisierung fehlgeschlagen")
        return 1
    
    try:
        # Flask App starten
        port = int(os.getenv("PORT", 5000))
        logger.info(f"Flask App startet auf Port {port}")
        
        app.run(
            host="0.0.0.0",
            port=port,
            debug=os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            threaded=True
        )
    
    except KeyboardInterrupt:
        logger.info("Manueller Stopp durch Benutzer")
        if bot:
            bot.stop_auto_trading()
        return 0
    
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}")
        return 1

# APP INITIALISIERUNG FÜR RENDER
if __name__ == "__main__":
    # Lokale Entwicklung
    exit(main())
else:
    # Render/WSGI - Automatische Initialisierung
    logger.info("WSGI/Render Umgebung erkannt")
    if startup_checks():
        # Bot im Hintergrund initialisieren
        try:
            # Für Flask 2.x compatibility
            with app.app_context():
                initialize_bot()
                logger.info("Bot erfolgreich für WSGI/Render initialisiert")
        except Exception as e:
            logger.error(f"WSGI Bot-Initialisierung Fehler: {e}")
        logger.info("Flask App bereit für WSGI/Render")
    else:
        logger.error("Startup-Checks für WSGI/Render fehlgeschlagen")

# GRACEFUL SHUTDOWN
import signal
import atexit

def signal_handler(sig, frame):
    logger.info("Shutdown-Signal erhalten...")
    global trading_bot
    if trading_bot:
        trading_bot.stop_auto_trading()
    logger.info("Graceful Shutdown abgeschlossen")

def cleanup_on_exit():
    logger.info("Cleanup beim Beenden...")
    global trading_bot
    if trading_bot:
        trading_bot.stop_auto_trading()

# Signal Handler registrieren
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_on_exit)

logger.info("Trading Bot v2.0 Module geladen - bereit für Deployment!")
