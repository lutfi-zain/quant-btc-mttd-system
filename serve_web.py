#!/usr/bin/env python3
"""
MTTD System Web Dashboard
==========================

Runs generate_charts.py first, then creates a simple HTTP server
on port 8080 to serve the MTTD dashboard.
"""

import os
import sys
import json
import http.server
import socketserver
import threading
import time
from pathlib import Path

# Project root
project_root = os.path.dirname(os.path.abspath(__file__))
mttd_dir = os.path.join(project_root, 'mttd')

def run_generate_charts():
    """Run generate_charts.py to create charts and data."""
    print("Running generate_charts.py...")
    import subprocess
    result = subprocess.run([sys.executable, os.path.join(project_root, 'generate_charts.py')], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running generate_charts.py: {result.stderr}")
        return False
    print("Charts generated successfully")
    return True

def create_index_html():
    """Create index.html dashboard in mttd directory."""
    metrics_path = os.path.join(mttd_dir, 'metrics.json')
    chart_path = os.path.join(mttd_dir, 'system_performance.png')
    
    # Load metrics
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    # Extract metrics for display
    if 'performance' in metrics:
        perf = metrics['performance']
        sharpe = perf.get('sharpe', 'N/A')
        cagr = perf.get('cagr', 'N/A')
        max_dd = perf.get('max_dd', 'N/A')
        win_rate = perf.get('win_rate', 'N/A')
        n_trades = perf.get('n_trades', 'N/A')
        avg_hold = perf.get('avg_hold', 'N/A')
    else:
        sharpe = metrics.get('sharpe', 'N/A')
        cagr = metrics.get('cagr', 'N/A')
        max_dd = metrics.get('max_drawdown', 'N/A')
        win_rate = metrics.get('win_rate', 'N/A')
        n_trades = metrics.get('total_trades', 'N/A')
        avg_hold = 'N/A'
    
    # Create HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">  <!-- Auto-refresh every 30 seconds -->
    <title>MTTD Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .card h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .chart-container {{
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card h3 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }}
        .status {{
            text-align: center;
            color: #4CAF50;
            font-weight: bold;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MTTD Dashboard</h1>
            <p>Medium-Term Trend following Consensus System</p>
            <div class="status">● System Active</div>
        </div>
        
        <div class="card">
            <h2>System Performance Chart</h2>
            <div class="chart-container">
                <img src="system_performance.png" alt="MTTD System Performance" 
                     onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22800%22 height=%22400%22><rect width=%22800%22 height=%22400%22 fill=%22%23f0f0f0%22/><text x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 font-family=%22Arial%22 font-size=%2220%22 fill=%22%23666%22>Chart not available. Run generate_charts.py first.</text></svg>'">
            </div>
        </div>
        
        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Sharpe Ratio</h3>
                    <div class="value">{sharpe:.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>CAGR</h3>
                    <div class="value">{cagr:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Max Drawdown</h3>
                    <div class="value">{max_dd:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Win Rate</h3>
                    <div class="value">{win_rate:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Total Trades</h3>
                    <div class="value">{n_trades}</div>
                </div>
                <div class="metric-card">
                    <h3>Avg Hold Period</h3>
                    <div class="value">{avg_hold:.0f} days</div>
                </div>
                <div class="metric-card">
                    <h3>System Status</h3>
                    <div class="value" style="color: #4CAF50;">● Active</div>
                </div>
                <div class="metric-card">
                    <h3>Configuration</h3>
                    <div class="value" style="font-size: 1em;">T75/250 BB25 MH45</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>MTTD System v1.0 | Auto-refreshes every 30 seconds</p>
            <p>Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
    
    # Write HTML file
    html_path = os.path.join(mttd_dir, 'index.html')
    with open(html_path, 'w') as f:
        f.write(html)
    
    print(f"Created dashboard: {html_path}")
    return html_path

class MTTPDHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for MTTD dashboard."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=mttd_dir, **kwargs)
    
    def end_headers(self):
        # Add CORS headers for local development only
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:8080')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Custom log format
        print(f"[{time.strftime('%H:%M:%S')}] {format % args}")

def start_server(port=8080):
    """Start HTTP server on specified port."""
    with socketserver.TCPServer(("", port), MTTPDHandler) as httpd:
        print(f"\n{'='*50}")
        print(f"MTTD Dashboard Server")
        print(f"{'='*50}")
        print(f"Serving at: http://localhost:{port}")
        print(f"Directory: {mttd_dir}")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*50}\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    print("MTTD System Web Dashboard")
    print("="*50)
    
    # Step 1: Run generate_charts.py
    if not run_generate_charts():
        print("Failed to generate charts")
        sys.exit(1)
    
    # Step 2: Create index.html
    create_index_html()
    
    # Step 3: Start web server
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_server(port)