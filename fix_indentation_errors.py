#!/usr/bin/env python3
"""
Script pour corriger les erreurs d'indentation et les variables non définies dans app_web.py
"""
import re

def fix_indentation_errors(filepath):
    """Corrige les erreurs d'indentation dans le fichier"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixes_count = 0
    
    # Pattern 1: Corriger les blocs où `row` est utilisé après un commentaire
    # Rechercher les patterns comme:
    # for row in cursor.fetchall():  # TODO
    #     code qui utilise row (non commenté)
    
    # Remplacer les utilisations de row[index] par row.get('key') après commentaire
    pattern1 = r"# for row in cursor\.fetchall\(\).*?$\s+(.*?)(?=\n\s{0,8}[^\s#]|\Z)"
    
    def indent_block(match):
        nonlocal fixes_count
        block = match.group(1)
        # Ajouter # devant chaque ligne du bloc
        lines = block.split('\n')
        indented_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                indented_lines.append('#' + line)
                fixes_count += 1
            else:
                indented_lines.append(line)
        return f"# for row in cursor.fetchall():  # TODO: Convert to MongoDB\n" + '\n'.join(indented_lines)
    
    content = re.sub(pattern1, indent_block, content, flags=re.MULTILINE | re.DOTALL)
    
    # Pattern 2: Commenter les blocs except orphelins
    pattern2 = r"(^\s+)except:\s*$\n(\1    )(?!#)"
    content = re.sub(pattern2, r"\1except:\n\2#", content, flags=re.MULTILINE)
    fixes_count += len(re.findall(pattern2, content, flags=re.MULTILINE))
    
    # Pattern 3: Commenter les utilisations de variables non définies après commentaires
    # Variables communes: result, patient_data, row, msg_info, message_info, etc.
    undefined_vars = ['result', 'patient_data', 'row', 'msg_info', 'message_info', 
                      'tumor_distribution', 'daily_analyses', 'table_exists',
                      'current_month_analyses', 'previous_month_analyses',
                      'successful_recent', 'diagnostic_counts', 'data',
                      'hourly_stats', 'confidence_evolution', 'top_active_days',
                      'performance_stats', 'date_range', 'diagnostic_types',
                      'confidence_range', 'current_month', 'previous_month',
                      'previous_month_patients', 'current_avg_confidence',
                      'previous_avg_confidence', 'today_confidence', 'week_confidence',
                      'today_count', 'avg_daily', 'low_confidence_count',
                      'daily_trends', 'hourly_trends', 'performance_by_type',
                      'distribution', 'hourly_data', 'confidences', 'times',
                      'monthly_data', 'base_metrics', 'week1_conf', 'week2_conf',
                      'diagnostic_dist', 'low_conf_count', 'total_last_30_days',
                      'prev_month_total', 'accuracy_rate', 'yearly_data',
                      'avg_response_time', 'last_analysis', 'analyses',
                      'evolution_stats', 'existing_ids', 'current_month_tumors',
                      'previous_month_tumors', 'new_patients_month', 'total_analyses',
                      'tumors_detected', 'patients_count', 'avg_confidence',
                      'daily_data', 'diagnostic_analysis', 'processing_time_analysis',
                      'weekly_productivity', 'high_risk_patients',
                      'monthly_detection_rates', 'patients', 'cursor']
    
    # Commenter les lignes qui utilisent ces variables après des commentaires TODO
    for var in undefined_vars:
        # Chercher les utilisations de la variable qui ne sont pas commentées
        # mais qui suivent des commentaires TODO ou DISABLED
        pattern = rf"^(\s+)(?!#)(.*)(\b{var}\b)(?!.*=.*{var})"
        
        def comment_undefined_usage(match):
            nonlocal fixes_count
            indent = match.group(1)
            line_content = match.group(2) + match.group(3) + match.string[match.end(3):match.end()]
            # Vérifier si la ligne est dans un contexte de définition
            if '=' in line_content and line_content.index('=') < line_content.index(var):
                return match.group(0)  # Ne pas commenter les définitions
            fixes_count += 1
            return f"{indent}# {line_content.lstrip()}"
        
        #content = re.sub(pattern, comment_undefined_usage, content, flags=re.MULTILINE)
    
    # Sauvegarder le fichier
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return fixes_count

if __name__ == '__main__':
    filepath = '/home/mohammed/Bureau/ai scan/app_web.py'
    count = fix_indentation_errors(filepath)
    print(f"✅ Corrections effectuées: {count} modifications")
