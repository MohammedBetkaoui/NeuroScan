"""
Test pour vérifier la gestion des IDs MongoDB
"""
from bson import ObjectId

# Test 1: Conversion d'une string en ObjectId
test_id_str = "68ef0067f2e0be8d5cfb3e5a"
try:
    obj_id = ObjectId(test_id_str)
    print(f"✅ Test 1 réussi: String '{test_id_str}' → ObjectId({obj_id})")
except Exception as e:
    print(f"❌ Test 1 échoué: {e}")

# Test 2: Conversion d'un int en ObjectId (doit échouer)
test_id_int = 68
try:
    obj_id = ObjectId(test_id_int)
    print(f"❌ Test 2 inattendu: Int {test_id_int} converti en ObjectId")
except Exception as e:
    print(f"✅ Test 2 réussi: Int {test_id_int} ne peut pas être converti → {type(e).__name__}")

# Test 3: Vérifier qu'on peut reconvertir en string
try:
    obj_id = ObjectId(test_id_str)
    str_id = str(obj_id)
    print(f"✅ Test 3 réussi: ObjectId → String '{str_id}'")
    assert str_id == test_id_str, "La string ne correspond pas"
except Exception as e:
    print(f"❌ Test 3 échoué: {e}")

# Test 4: Fonction helper pour conversion sécurisée
def safe_to_objectid(value):
    """Convertir une valeur en ObjectId de manière sécurisée"""
    try:
        if isinstance(value, ObjectId):
            return value
        elif isinstance(value, str):
            return ObjectId(value)
        else:
            raise ValueError(f"Type non supporté: {type(value)}")
    except Exception as e:
        print(f"Erreur conversion: {e}")
        return None

# Test de la fonction helper
print("\n--- Tests de la fonction helper ---")
print(f"String → ObjectId: {safe_to_objectid(test_id_str)}")
print(f"ObjectId → ObjectId: {safe_to_objectid(ObjectId(test_id_str))}")
print(f"Int → ObjectId: {safe_to_objectid(68)}")

print("\n✅ Tous les tests terminés")
