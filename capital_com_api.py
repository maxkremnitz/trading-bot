import os
import requests
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class CapitalComAPI:
    """Capital.com API Wrapper"""
    
    def __init__(self):
        self.api_key = os.getenv('CAPITAL_API_KEY')
        self.password = os.getenv('CAPITAL_PASSWORD')
        self.email = os.getenv('CAPITAL_EMAIL')
        self.base_url = "https://demo-api-capital.backend-capital.com"
        
        self.cst_token = None
        self.security_token = None
        self.last_auth_time = 0
        self.auth_timeout = 600  # 10 Minuten
        
    def authenticate(self):
        """API Authentifizierung"""
        if self.is_authenticated():
            return True
            
        try:
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
                self.last_auth_time = time.time()
                logger.info("Capital.com API authentifiziert")
                return True
            else:
                logger.error(f"Auth fehlgeschlagen: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Auth Fehler: {e}")
            return False
    
    def is_authenticated(self):
        """Prüfe ob Session aktiv"""
        if not self.cst_token or not self.security_token:
            return False
        return (time.time() - self.last_auth_time) < self.auth_timeout
    
    def get_auth_headers(self):
        """Auth Headers für Requests"""
        return {
            'X-CAP-API-KEY': self.api_key,
            'CST': self.cst_token,
            'X-SECURITY-TOKEN': self.security_token,
            'Content-Type': 'application/json'
        }
    
    def place_order(self, signal):
        """Order platzieren"""
        if not self.authenticate():
            logger.error("Authentifizierung fehlgeschlagen")
            return None
            
        try:
            # Position Size berechnen
            position_size = self.calculate_position_size(
                signal['position_size_percent']
            )
            
            # Stop Loss und Take Profit berechnen
            current_price = signal['current_price']
            
            if signal['action'] == 'BUY':
                stop_loss = current_price * (1 - signal['stop_loss_percent']/100)
                take_profit = current_price * (1 + signal['take_profit_percent']/100)
            else:
                stop_loss = current_price * (1 + signal['stop_loss_percent']/100)
                take_profit = current_price * (1 - signal['take_profit_percent']/100)
            
            # Order Data
            order_data = {
                'epic': 'GOLD',
                'direction': signal['action'],
                'size': position_size,
                'guaranteedStop': False,
                'stopDistance': max(50, int(abs(current_price - stop_loss))),
                'profitDistance': max(50, int(abs(take_profit - current_price)))
            }
            
            logger.info(f"Platziere Order: {order_data}")
            
            # Order senden
            response = requests.post(
                f"{self.base_url}/api/v1/positions",
                headers=self.get_auth_headers(),
                json=order_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Order erfolgreich: {result.get('dealReference')}")
                return result
            else:
                logger.error(f"Order fehlgeschlagen: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Order Fehler: {e}")
            return None
    
    def calculate_position_size(self, percentage):
        """Position Size berechnen"""
        # Vereinfachte Berechnung
        # In der Praxis würden Sie hier den Account Balance abrufen
        available_balance = 10000  # Demo Account Standard
        target_amount = available_balance * (percentage / 100)
        
        # Capital.com Mindestgrößen beachten
        if target_amount < 50:
            return 0.5
        elif target_amount < 100:
            return 1.0
        elif target_amount < 500:
            return round(target_amount / 100, 1)
        else:
            return round(target_amount / 200, 1)
    
    def get_positions(self):
        """Aktuelle Positionen abrufen"""
        if not self.authenticate():
            return []
            
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/positions",
                headers=self.get_auth_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get('positions', [])
            else:
                logger.error(f"Get Positions fehlgeschlagen: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Get Positions Fehler: {e}")
            return []
