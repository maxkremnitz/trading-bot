import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import time
import json
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class EnhancedGoldSilverAnalysis:
    """Gold/Silver Analyse mit zusätzlichem Momentum-Indikator"""
    
    def __init__(self):
        self.cache_file = 'price_history.json'
        self.load_price_history()
        
    def load_price_history(self):
        """Lade gespeicherte Preishistorie"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.price_history = json.load(f)
            else:
                self.price_history = {
                    'gold': [],
                    'silver': [],
                    'timestamps': []
                }
        except Exception as e:
            logger.error(f"Fehler beim Laden der Preishistorie: {e}")
            self.price_history = {'gold': [], 'silver': [], 'timestamps': []}
    
    def save_price_history(self):
        """Speichere Preishistorie"""
        try:
            # Nur die letzten 100 Einträge behalten
            for key in ['gold', 'silver', 'timestamps']:
                if len(self.price_history[key]) > 100:
                    self.price_history[key] = self.price_history[key][-100:]
                    
            with open(self.cache_file, 'w') as f:
                json.dump(self.price_history, f)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Preishistorie: {e}")
    
    def get_current_prices(self):
        """Aktuelle Preise abrufen (Simulation)"""
        # In der Realität würden Sie hier echte API-Daten abrufen
        base_gold = 2000.0
        base_silver = 25.0
        
        # Simulierte Preisschwankungen
        gold_variation = (np.random.random() - 0.5) * 40
        silver_variation = (np.random.random() - 0.5) * 2
        
        gold_price = base_gold + gold_variation
        silver_price = base_silver + silver_variation
        
        # Preise zur Historie hinzufügen
        self.price_history['gold'].append(gold_price)
        self.price_history['silver'].append(silver_price)
        self.price_history['timestamps'].append(datetime.now().isoformat())
        
        self.save_price_history()
        
        return {
            'gold': gold_price,
            'silver': silver_price,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_momentum_indicator(self, prices, window=5):
        """NEUER INDIKATOR: Momentum-basierte Analyse"""
        if len(prices) < window:
            return 0
            
        # 5-Perioden Momentum
        momentum = ((prices[-1] - prices[-window]) / prices[-window]) * 100
        return momentum
    
    def calculate_rsi(self, prices, period=14):
        """RSI berechnen"""
        if len(prices) < period + 1:
            return 50  # Neutraler Wert
            
        # Preisänderungen berechnen
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Gewinne und Verluste trennen
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        # Durchschnittliche Gewinne und Verluste
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def analyze(self):
        """Hauptanalyse-Funktion"""
        try:
            # Aktuelle Preise abrufen
            current_prices = self.get_current_prices()
            
            # Momentum berechnen
            gold_momentum = self.calculate_momentum_indicator(
                self.price_history['gold']
            )
            silver_momentum = self.calculate_momentum_indicator(
                self.price_history['silver']
            )
            
            # RSI berechnen
            gold_rsi = self.calculate_rsi(self.price_history['gold'])
            silver_rsi = self.calculate_rsi(self.price_history['silver'])
            
            # Trading-Signal generieren
            signal = self.generate_signal(
                current_prices['gold'],
                current_prices['silver'],
                gold_momentum,
                silver_momentum,
                gold_rsi,
                silver_rsi
            )
            
            # Ergebnis loggen
            logger.info(f"""
            Analyse abgeschlossen:
            Gold: ${current_prices['gold']:.2f} (Mom: {gold_momentum:.2f}%, RSI: {gold_rsi:.1f})
            Silver: ${current_prices['silver']:.2f} (Mom: {silver_momentum:.2f}%, RSI: {silver_rsi:.1f})
            Signal: {signal['action']} - {signal.get('reason', 'N/A')}
            """)
            
            return signal
            
        except Exception as e:
            logger.error(f"Analyse-Fehler: {e}")
            return None
    
    def generate_signal(self, gold_price, silver_price, gold_mom, silver_mom, 
                       gold_rsi, silver_rsi):
        """Signal-Generierung mit mehreren Parametern"""
        score = 0
        reasons = []
        
        # Gold/Silver Ratio
        ratio = gold_price / silver_price
        if ratio > 80:
            score -= 20
            reasons.append(f"G/S Ratio hoch ({ratio:.1f})")
        elif ratio < 70:
            score += 20
            reasons.append(f"G/S Ratio niedrig ({ratio:.1f})")
        
        # Momentum-Analyse
        if gold_mom > 2 and silver_mom < -1:
            score += 25
            reasons.append(f"Gold Momentum +{gold_mom:.1f}%")
        elif gold_mom < -2 and silver_mom > 1:
            score -= 25
            reasons.append(f"Gold Momentum {gold_mom:.1f}%")
        
        # RSI-Analyse
        if gold_rsi < 30:
            score += 15
            reasons.append(f"Gold überverkauft (RSI {gold_rsi:.1f})")
        elif gold_rsi > 70:
            score -= 15
            reasons.append(f"Gold überkauft (RSI {gold_rsi:.1f})")
        
        # Account-spezifische Anpassungen
        account_type = os.getenv('ACCOUNT_TYPE', 'account1')
        
        if account_type == 'account1':
            # Konservativere Parameter für Account 1
            threshold_buy = 50
            threshold_sell = -50
            position_size_percent = 5.0
        else:
            # Aggressivere Parameter für Account 2
            threshold_buy = 40
            threshold_sell = -40
            position_size_percent = 10.0
        
        # Trading-Entscheidung
        if score >= threshold_buy:
            return {
                'action': 'BUY',
                'ticker': 'GOLD',
                'confidence': min(100, abs(score)),
                'reason': ' | '.join(reasons),
                'score': score,
                'position_size_percent': position_size_percent,
                'stop_loss_percent': 1.0,
                'take_profit_percent': 1.5,
                'current_price': gold_price
            }
        elif score <= threshold_sell:
            return {
                'action': 'SELL',
                'ticker': 'GOLD',
                'confidence': min(100, abs(score)),
                'reason': ' | '.join(reasons),
                'score': score,
                'position_size_percent': position_size_percent,
                'stop_loss_percent': 1.0,
                'take_profit_percent': 1.5,
                'current_price': gold_price
            }
        else:
            return {
                'action': 'HOLD',
                'ticker': 'GOLD',
                'reason': f"Neutraler Score: {score}",
                'score': score,
                'current_price': gold_price
            }
