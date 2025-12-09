#!/usr/bin/env python3
import sys
import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from gold_silver_analysis import EnhancedGoldSilverAnalysis
from capital_com_api import CapitalComAPI

# Environment laden
load_dotenv()

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_trading_hours():
    """Prüfe ob Handelszeiten aktiv sind"""
    # Vereinfachte Prüfung - in der Praxis würden Sie Zeitzonen berücksichtigen
    from datetime import datetime
    now = datetime.now()
    
    # Forex Handelszeiten (Sonntag 21:00 UTC bis Freitag 22:00 UTC)
    if now.weekday() == 5:  # Samstag
        return False, "Wochenende - Märkte geschlossen"
    elif now.weekday() == 6:  # Sonntag
        if now.hour < 21:
            return False, "Wochenende - Märkte geschlossen"
    
    return True, "Handelszeiten aktiv"

def execute_trade_if_needed(signal):
    """Führe Trade aus wenn Signal es erfordert"""
    if signal['action'] in ['BUY', 'SELL']:
        logger.info(f"Trade-Signal erkannt: {signal['action']}")
        
        api = CapitalComAPI()
        result = api.place_order(signal)
        
        if result:
            logger.info(f"Trade erfolgreich ausgeführt: {result.get('dealReference')}")
            return True
        else:
            logger.error("Trade-Ausführung fehlgeschlagen")
            return False
    else:
        logger.info(f"Kein Trade erforderlich: {signal['action']}")
        return True

def save_analysis_result(signal):
    """Speichere Analyse-Ergebnis"""
    try:
        # Ergebnis-Datei
        result_file = 'last_analysis.json'
        
        # Ergebnis erweitern
        result = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'account_type': os.getenv('ACCOUNT_TYPE', 'account1')
        }
        
        # Speichern
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Analyse-Ergebnis gespeichert: {result_file}")
        
    except Exception as e:
        logger.error(f"Fehler beim Speichern: {e}")

def main():
    """Hauptfunktion für Cron Job"""
    start_time = datetime.now()
    logger.info(f"=== Gold/Silver Analyse gestartet um {start_time} ===")
    
    try:
        # Handelszeiten prüfen
        trading_allowed, message = check_trading_hours()
        if not trading_allowed:
            logger.info(f"Analyse übersprungen: {message}")
            return 0
        
        # Analyse durchführen
        analyzer = EnhancedGoldSilverAnalysis()
        signal = analyzer.analyze()
        
        if signal:
            # Ergebnis speichern
            save_analysis_result(signal)
            
            # Trade ausführen wenn nötig
            if signal['action'] != 'HOLD':
                execute_trade_if_needed(signal)
            
            logger.info(f"Analyse erfolgreich: {signal}")
        else:
            logger.warning("Keine Analyse-Ergebnisse erhalten")
        
        # Ausführungszeit loggen
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analyse abgeschlossen in {duration:.1f} Sekunden")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fehler in Hauptfunktion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        logger.info("=== Analyse beendet ===\n")

if __name__ == "__main__":
    sys.exit(main())
