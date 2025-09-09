# 🚂 Trading Bot - Dual Strategy System

Automatischer Trading Bot mit zwei separaten Strategien und Handelszeitenbeschränkung.

## 🎯 **Strategien:**
1. **Hauptstrategie** (Technical Analysis) → **Standard Demo Account**  
2. **Gold/Silver Test** (Correlation Strategy) → **Demo - Account 1**

## ⚙️ **Features:**
- ✅ **Handelszeitenbeschränkung** (nur während NYSE/XETRA/FOREX Zeiten)
- ✅ **Dual-Account System** mit automatischem Account-Switching
- ✅ **Capital.com API Integration** (vollständig nach Dokumentation)
- ✅ **20-Minuten Update-Intervall** 
- ✅ **Flask Web Dashboard** mit Live-Status
- ✅ **SQLite Persistierung** für Trade-History
- ✅ **Rate Limiting** und Error Handling

## 🔧 **Setup:**

### **1. Capital.com API Keys erstellen:**
1. Einloggen bei Capital.com (Demo Account)
2. Settings → API Integrations
3. Generate API Key erstellen
4. API Key + Custom Password notieren

### **2. Environment Variablen (in Render setzen):**
```bash
CAPITAL_API_KEY=dein_api_key_hier
CAPITAL_PASSWORD=dein_custom_password
CAPITAL_EMAIL=deine_email@example.com
TRADING_ENABLED=true
UPDATE_INTERVAL_MINUTES=20
