#!/usr/bin/env python3
"""
Script pour commenter toutes les fonctions SQLite restantes dans app_web.py
"""

import re

def fix_remaining_sqlite():
    file_path = '/home/mohammed/Bureau/ai scan/app_web.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fonctions SQLite à commenter/remplacer
    sqlite_functions = [
        'manage_patient_record',
        'analyze_tumor_evolution',
        'create_medical_alerts',
        'get_patient_history',
        'calculate_risk_level',
        'update_patient_info',
        'delete_analysis',
        'get_analyses_by_patient',
        'get_all_patients_for_doctor'
    ]
    
    in_sqlite_function = False
    function_name = None
    indent_level = 0
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Détecter le début d'une fonction SQLite
        for func in sqlite_functions:
            if f'def {func}(' in line:
                in_sqlite_function = True
                function_name = func
                # Ajouter un commentaire
                new_lines.append(f'# {line.strip()} # TODO: MongoDB implementation needed\n')
                new_lines.append(f'def {func}(*args, **kwargs):\n')
                new_lines.append(f'    """Fonction désactivée - nécessite implémentation MongoDB"""\n')
                new_lines.append(f'    pass\n\n')
                
                # Trouver la fin de la fonction (prochain def au même niveau)
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    # Si c'est une nouvelle fonction au même niveau ou une ligne non indentée
                    if (next_line.startswith('def ') or 
                        next_line.startswith('@app.') or 
                        (next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'))):
                        i -= 1  # Revenir en arrière pour traiter cette ligne normalement
                        break
                    i += 1
                
                in_sqlite_function = False
                break
        else:
            # Ligne normale, la garder
            new_lines.append(line)
        
        i += 1
    
    # Sauvegarder
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("✅ Fonctions SQLite commentées")

if __name__ == '__main__':
    fix_remaining_sqlite()
