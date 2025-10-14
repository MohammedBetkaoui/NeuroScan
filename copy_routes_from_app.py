#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour copier les routes essentielles de app.py vers app_web_minimal.py
en les convertissant de SQLite vers MongoDB
"""

# Routes essentielles à copier
ESSENTIAL_ROUTES = [
    # Pages principales
    '/nouvelle-analyse',
    '/manage-patients',
    '/medical-alerts',
    '/alerts',
    '/patients',
    '/patients/new',
    '/tumor-tracking',
    '/platform-stats',
    '/pro-dashboard',
    '/pro-dashboard-advanced',
    
    # Upload et analyse
    '/upload',
    '/uploads/<path:filename>',
    '/analysis/<int:analysis_id>',
    
    # API Patients
    '/api/patients',
    '/api/my-patients',
    '/api/patients/list',
    '/api/patients/next-id',
    '/api/patients/<patient_id>',
    '/api/patients/<patient_id>/details',
    '/api/patients/<patient_id>/evolution',
    
    # API Alerts
    '/api/alerts',
    '/api/notifications',
    
    # API Analytics
    '/api/analytics/overview',
    '/api/analytics/platform-overview',
    '/api/analytics/period/<period>',
    '/api/analytics/recent',
    
    # API Doctor
    '/api/doctor/stats',
    '/api/doctor/update-profile',
    
    # Health
    '/health',
    '/api/health',
]

print("Routes essentielles identifiées:")
for route in ESSENTIAL_ROUTES:
    print(f"  - {route}")

print(f"\nTotal: {len(ESSENTIAL_ROUTES)} routes à copier")
print("\n⚠️  NOTE: La copie manuelle est recommandée pour assurer la qualité")
print("Utilisez ce script comme référence pour les routes à implémenter.")
