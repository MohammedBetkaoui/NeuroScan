#!/usr/bin/env python3
"""
Script de test pour les nouveaux endpoints du tableau de bord avancé
"""

import requests
import json
import sys

def test_endpoint(url, endpoint_name):
    """Test un endpoint et affiche le résultat"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ {endpoint_name}: OK")
                return True
            else:
                print(f"❌ {endpoint_name}: Erreur dans les données - {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ {endpoint_name}: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {endpoint_name}: Erreur de connexion - {e}")
        return False
    except Exception as e:
        print(f"❌ {endpoint_name}: Erreur - {e}")
        return False

def main():
    """Test tous les nouveaux endpoints"""
    base_url = "http://127.0.0.1:5000"
    
    endpoints = [
        ("/api/analytics/diagnostic-distribution", "Distribution des diagnostics"),
        ("/api/analytics/hourly-activity", "Activité par heure"),
        ("/api/analytics/confidence-distribution", "Distribution de confiance"),
        ("/api/analytics/processing-time-analysis", "Analyse temps de traitement"),
        ("/api/analytics/monthly-trends", "Tendances mensuelles"),
        ("/api/analytics/ai-insights", "Insights IA"),
        ("/api/analytics/advanced-metrics", "Métriques avancées"),
        ("/api/analytics/overview", "Vue d'ensemble"),
        ("/api/analytics/alerts", "Alertes"),
        ("/api/analytics/performance", "Performance"),
        ("/api/analytics/comparison", "Comparaison")
    ]
    
    print("🚀 Test des endpoints du tableau de bord avancé")
    print("=" * 60)
    
    success_count = 0
    total_count = len(endpoints)
    
    for endpoint, name in endpoints:
        url = base_url + endpoint
        if test_endpoint(url, name):
            success_count += 1
    
    print("=" * 60)
    print(f"📊 Résultats: {success_count}/{total_count} endpoints fonctionnels")
    
    if success_count == total_count:
        print("🎉 Tous les endpoints fonctionnent parfaitement !")
        return 0
    else:
        print("⚠️  Certains endpoints nécessitent une attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
