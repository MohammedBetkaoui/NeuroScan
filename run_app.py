#!/usr/bin/env python3
"""
NeuroScan AI - Desktop Application
Lance l'application Flask dans une fenêtre PyWebView
"""

import sys
import os
import threading
import time

# Ajouter les packages système Python au path pour webview
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import webview
from app import app

# Configuration
HOST = '127.0.0.1'
PORT = 5000
WINDOW_TITLE = "NeuroScan AI"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

def start_flask():
    """Démarre le serveur Flask dans un thread séparé"""
    try:
        # Désactiver le mode debug en production
        app.run(
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"Erreur lors du démarrage de Flask: {e}")
        sys.exit(1)

def main():
    """Point d'entrée principal de l'application"""
    print("=" * 60)
    print("🧠 NeuroScan AI - Application Desktop")
    print("=" * 60)
    print(f"📍 Démarrage du serveur Flask sur {HOST}:{PORT}...")
    
    # Démarrer Flask dans un thread séparé
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # Attendre que Flask soit prêt
    print("⏳ Attente du démarrage du serveur...")
    time.sleep(3)
    
    # URL de l'application
    app_url = f'http://{HOST}:{PORT}'
    print(f"✅ Serveur Flask démarré!")
    print(f"🌐 URL: {app_url}")
    print(f"🪟 Ouverture de la fenêtre PyWebView...")
    print("=" * 60)
    
    # Créer et afficher la fenêtre PyWebView
    try:
        window = webview.create_window(
            title=WINDOW_TITLE,
            url=app_url,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            resizable=True,
            fullscreen=False,
            min_size=(800, 600),
            background_color='#FFFFFF',
            text_select=True,
            # Activer toutes les fonctionnalités JavaScript
            js_api=None,
            # Permettre l'accès localStorage et sessionStorage
            easy_drag=False
        )
        
        # Démarrer l'interface graphique (bloquant)
        # Debug=True pour voir les erreurs JavaScript dans la console
        webview.start(debug=True)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'ouverture de PyWebView: {e}")
        print("💡 Assurez-vous que PyWebView et ses dépendances sont installées:")
        print("   sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0")
        sys.exit(1)
    
    print("\n👋 Fermeture de l'application...")

if __name__ == '__main__':
    main()
