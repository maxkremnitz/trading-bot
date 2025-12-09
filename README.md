# Gold/Silver Trading Bot - Render Optimized

Ein automatisierter Trading Bot für Gold-Trading basierend auf Silver-Preisbewegungen, optimiert für Render Free Tier mit Cron Jobs.

## 🚀 Features

- **Dual-Account System**: Konservative Strategie (Account 1) und aggressive Strategie (Account 2)
- **Render-optimiert**: Läuft als Cron Job statt kontinuierlicher Service
- **Erweiterte Analyse**: Momentum-Indikator + RSI + Gold/Silver-Ratio
- **Memory-effizient**: Keine Memory-Leaks oder Timeouts
- **Automatische Trades**: Via Capital.com Demo API

## 📊 Trading-Strategie

### Account 1 (Konservativ)
- Position Size: 5% des Kapitals
- Höhere Schwellenwerte für Signale
- Geeignet für risikoaverse Ansätze

### Account 2 (Aggressiv)  
- Position Size: 10% des Kapitals
- Niedrigere Schwellenwerte für Signale
- Mehr Trading-Opportunitäten

## 🛠️ Setup

### 1. GitHub Repository
```bash
git clone https://github.com/YOUR_USERNAME/gold-silver-bot.git
cd gold-silver-bot
