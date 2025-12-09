from flask import Flask, render_template_string, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Gold/Silver Trading Bot</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .status-card {
            margin: 20px 0;
            padding: 15px;
            border-left: 4px solid #007bff;
            background: #f8f9fa;
        }
        .buy { border-left-color: #28a745; }
        .sell { border-left-color: #dc3545; }
        .hold { border-left-color: #6c757d; }
        .timestamp {
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gold/Silver Trading Bot Status</h1>
        
        <div class="status-card">
            <h3>Account Type: {{ account_type }}</h3>
            <p class="timestamp">Server Time: {{ timestamp }}</p>
        </div>
        
        {% if last_analysis %}
        <div class="status-card {{ last_analysis.signal.action.lower() }}">
            <h3>Last Signal: {{ last_analysis.signal.action }}</h3>
            <p><strong>Ticker:</strong> {{ last_analysis.signal.ticker }}</p>
            <p><strong>Price:</strong> ${{ "%.2f"|format(last_analysis.signal.current_price) }}</p>
            <p><strong>Score:</strong> {{ last_analysis.signal.score }}</p>
            <p><strong>Reason:</strong> {{ last_analysis.signal.reason }}</p>
            <p class="timestamp">{{ last_analysis.timestamp }}</p>
        </div>
        {% else %}
        <div class="status-card">
            <p>No analysis data available yet.</p>
        </div>
        {% endif %}
        
        {% if price_history %}
        <div class="status-card">
            <h3>Price History (Last 5)</h3>
            <ul>
            {% for i in range(5) %}
                {% if i < price_history.gold|length %}
                <li>Gold: ${{ "%.2f"|format(price_history.gold[-(i+1)]) }} | 
                    Silver: ${{ "%.2f"|format(price_history.silver[-(i+1)]) }}</li>
                {% endif %}
            {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Dashboard anzeigen"""
    try:
        # Letzte Analyse laden
        last_analysis = None
        if os.path.exists('last_analysis.json'):
            with open('last_analysis.json', 'r') as f:
                last_analysis = json.load(f)
        
        # Preis-Historie laden
        price_history = None
        if os.path.exists('price_history.json'):
            with open('price_history.json', 'r') as f:
                price_history = json.load(f)
        
        return render_template_string(
            DASHBOARD_HTML,
            account_type=os.getenv('ACCOUNT_TYPE', 'unknown'),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            last_analysis=last_analysis,
            price_history=price_history
        )
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/health')
def health():
    """Health Check Endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/status')
def api_status():
    """API Status"""
    try:
        result = {
            'status': 'running',
            'account_type': os.getenv('ACCOUNT_TYPE', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Letzte Analyse hinzufügen
        if os.path.exists('last_analysis.json'):
            with open('last_analysis.json', 'r') as f:
                result['last_analysis'] = json.load(f)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
