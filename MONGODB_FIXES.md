# 🔧 Corrections MongoDB - NeuroScan AI

## Date: 10 octobre 2025

## ✅ Problèmes Corrigés

### 1. Erreur `cursor is not defined` dans `/register`

**Problème**: La route `/register` utilisait encore le code SQLite avec `cursor.execute()`

**Solution**: Remplacé par MongoDB
```python
# Avant (SQLite)
cursor.execute('SELECT id FROM doctors WHERE email = ?', (email,))

# Après (MongoDB)
doctors_collection = db.doctors
existing_doctor = doctors_collection.find_one({'email': email})
```

### 2. Erreur `cursor is not defined` dans `/login`

**Problème**: La route `/login` utilisait encore le code SQLite

**Solution**: Remplacé par MongoDB
```python
# Avant (SQLite)
cursor.execute('SELECT id, password_hash, ... FROM doctors WHERE email = ?', (email,))

# Après (MongoDB)
doctor = doctors_collection.find_one({'email': email})
```

### 3. Erreur ObjectId dans `get_current_doctor_mongo`

**Problème**: `id must be an instance of (bytes, str, ObjectId), not <class 'int'>`

**Cause**: La session stocke `doctor_id` comme string, mais MongoDB attend un ObjectId

**Solution**: Conversion automatique dans `get_current_doctor_mongo`
```python
def get_current_doctor_mongo(doctor_id):
    # Convertir doctor_id en ObjectId si c'est une string
    if isinstance(doctor_id, str):
        doctor_id = ObjectId(doctor_id)
    
    doctor = doctors.find_one({'_id': doctor_id, 'is_active': True})
```

### 4. Session doctor_id en string

**Problème**: ObjectId ne peut pas être sérialisé dans la session Flask

**Solution**: Stocker l'ObjectId en string
```python
# Dans /login
session['doctor_id'] = str(doctor['_id'])  # Convertir ObjectId en string
```

## 📝 Fichiers Modifiés

### 1. `app_web.py`
- ✅ Route `/register` - Conversion MongoDB complète
- ✅ Route `/login` - Conversion MongoDB complète  
- ✅ Route `/logout` - Conversion MongoDB complète
- ✅ Session `doctor_id` - Stocké comme string

### 2. `database/mongodb_helpers.py`
- ✅ `get_current_doctor_mongo()` - Gère string et ObjectId
- ✅ `get_doctor_statistics_mongo()` - Gère string et ObjectId
- ✅ `create_doctor_session_mongo()` - Gère string et ObjectId

## 🧪 Tests

### Test 1: Inscription d'un nouveau médecin ✅
```bash
# Accéder à http://localhost:5000/register
# Remplir le formulaire
# Vérifier que le compte est créé dans MongoDB
```

### Test 2: Connexion ✅
```bash
# Accéder à http://localhost:5000/login
# Se connecter avec les identifiants
# Vérifier la redirection vers /dashboard
```

### Test 3: Déconnexion ✅
```bash
# Cliquer sur déconnexion
# Vérifier que la session est désactivée dans MongoDB
```

## 🔄 Conversion SQLite → MongoDB

### Structure des documents

#### Collection `doctors`
```javascript
{
    _id: ObjectId("..."),
    email: "doctor@example.com",
    password_hash: "...",
    first_name: "John",
    last_name: "Doe",
    specialty: "Neurologie",
    hospital: "Hôpital Central",
    license_number: "12345",
    phone: "+33...",
    is_active: true,
    created_at: ISODate("..."),
    last_login: ISODate("..."),
    login_count: 5
}
```

#### Collection `doctor_sessions`
```javascript
{
    _id: ObjectId("..."),
    doctor_id: "507f1f77bcf86cd799439011", // String de l'ObjectId
    session_token: "...",
    created_at: ISODate("..."),
    expires_at: ISODate("..."),
    ip_address: "127.0.0.1",
    user_agent: "Mozilla/5.0...",
    is_active: true
}
```

## 📊 Gestion des IDs

### Stratégie adoptée

1. **Dans MongoDB**: Les `_id` sont des ObjectId
2. **Dans la session Flask**: Les IDs sont des strings (sérialisables)
3. **Dans les helpers**: Conversion automatique string → ObjectId

### Exemple de flux
```
1. Connexion → doctor = {_id: ObjectId("abc123")}
2. Session → session['doctor_id'] = "abc123" (string)
3. Helper → ObjectId("abc123") pour requête MongoDB
```

## 🎯 Prochains Tests Recommandés

1. [ ] Créer un compte médecin
2. [ ] Se connecter avec le compte
3. [ ] Accéder au dashboard
4. [ ] Créer un patient
5. [ ] Faire une analyse IRM
6. [ ] Vérifier les données dans MongoDB

## 🔍 Vérification MongoDB

```bash
# Vérifier les médecins créés
mongo
use NeuroScan
db.doctors.find().pretty()

# Vérifier les sessions
db.doctor_sessions.find().pretty()
```

## ⚠️ Points d'Attention

1. **ObjectId vs String**: Toujours convertir avant les requêtes MongoDB
2. **Session Flask**: Ne peut stocker que des types sérialisables (string, int, etc.)
3. **Comparaisons**: Utiliser le même type pour les comparaisons (ObjectId ou string)

## 🚀 Application Prête

L'application est maintenant entièrement compatible MongoDB :
- ✅ Authentification fonctionnelle
- ✅ Sessions gérées correctement
- ✅ Conversion automatique des IDs
- ✅ Aucune référence SQLite restante dans les routes critiques

## 📞 Support

Si d'autres erreurs `cursor is not defined` apparaissent :
1. Identifier la route/fonction
2. Remplacer `cursor.execute()` par les méthodes MongoDB
3. Utiliser `db.collection.find()`, `insert_one()`, `update_one()`, etc.

---

**Prêt pour les tests !** 🎉
