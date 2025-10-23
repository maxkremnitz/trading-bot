import os
import time
import requests
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RenderKeepAlive:
    """Render-spezifischer Keep-Alive Service"""
    
    def __init__(self, service_url=None):
        self.service_url = service_url or f"https://{os.getenv('RENDER_SERVICE_NAME', 'your-service')}.onrender.com"
        self.running = False
        self.ping_interval = 840  # 14 Minuten (vor 15min Sleep)
        self.thread = None
        
    def start(self):
        """Keep-Alive starten"""
        if not self.running and os.getenv('RENDER'):
            self.running = True
            self.thread = threading.Thread(target=self._ping_loop, daemon=True)
            self.thread.start()
            logger.info(f"Render Keep-Alive gestartet für {self.service_url}")
    
    def _ping_loop(self):
        """Ping-Loop für Render"""
        while self.running:
            try:
                response = requests.get(
                    f"{self.service_url}/health",
                    timeout=10
                )
                if response.status_code == 200:
                    logger.info("Keep-Alive Ping erfolgreich")
                else:
                    logger.warning(f"Keep-Alive Ping: {response.status_code}")
            except Exception as e:
                logger.error(f"Keep-Alive Error: {e}")
            
            time.sleep(self.ping_interval)
    
    def stop(self):
        """Keep-Alive stoppen"""
        self.running = False
