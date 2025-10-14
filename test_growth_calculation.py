#!/usr/bin/env python3
"""
Test du calcul de la croissance mensuelle
"""

def calculate_growth_rate(counts):
    """
    Calculer le taux de croissance entre les deux derniers mois
    Avec plafonnement pour les petites valeurs
    """
    if len(counts) >= 2:
        previous_month_count = counts[-2]
        current_month_count = counts[-1]
        
        if previous_month_count >= 10:
            # Si on a au moins 10 analyses le mois précédent, 
            # le pourcentage est significatif
            growth_rate = round(((current_month_count - previous_month_count) / previous_month_count) * 100, 1)
        elif previous_month_count > 0:
            # Pour de petits nombres, plafonner à +200% max
            raw_growth = ((current_month_count - previous_month_count) / previous_month_count) * 100
            growth_rate = round(min(raw_growth, 200.0), 1)
        elif current_month_count > 0:
            # Si le mois précédent était à 0, afficher +100%
            growth_rate = 100.0
        else:
            # Les deux sont à 0
            growth_rate = 0.0
    else:
        growth_rate = 0.0
    
    return growth_rate

# Tests
test_cases = [
    ([1, 11], "De 1 à 11 analyses"),
    ([10, 21], "De 10 à 21 analyses"),
    ([13, 21], "De 13 à 21 analyses"),
    ([0, 10], "De 0 à 10 analyses"),
    ([10, 10], "Stable à 10 analyses"),
    ([20, 15], "Baisse de 20 à 15 analyses"),
    ([5, 10, 20], "Croissance progressive: 5 → 10 → 20"),
    ([1, 2, 3, 4, 5], "Croissance régulière"),
]

print("=" * 70)
print("TEST DU CALCUL DE LA CROISSANCE MENSUELLE")
print("=" * 70)

for counts, description in test_cases:
    growth = calculate_growth_rate(counts)
    print(f"\n📊 {description}")
    print(f"   Données: {counts}")
    print(f"   Croissance: {'+' if growth >= 0 else ''}{growth}%")
    
    # Interprétation
    if growth > 100:
        print(f"   ✅ Forte croissance (plus que doublé)")
    elif growth > 50:
        print(f"   ✅ Bonne croissance")
    elif growth > 0:
        print(f"   ✅ Croissance positive")
    elif growth == 0:
        print(f"   ➡️  Stable")
    else:
        print(f"   ⚠️  Baisse")

print("\n" + "=" * 70)
print("FIN DES TESTS")
print("=" * 70)
