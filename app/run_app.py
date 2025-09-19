#!/usr/bin/env python3
"""
Script pour lancer l'application Streamlit
"""

import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    
    print("🚀 Lancement de l'interface web...")
    print(f"📍 URL: http://localhost:8501")
    print("🛑 Arrêt: Ctrl+C")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée")

if __name__ == "__main__":
    main()