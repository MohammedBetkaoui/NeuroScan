# 📊 Limites et Quotas de l'API Gemini

## 🎯 Quota API Gemini Gratuit

### Limites actuelles (Tier Gratuit)
- **Requêtes par jour** : 250 requêtes/jour
- **Modèle utilisé** : gemini-2.5-flash
- **Reset du quota** : Tous les jours à minuit (heure UTC)

### Message d'erreur typique
```json
{
  "error": {
    "code": 429,
    "message": "You exceeded your current quota...",
    "status": "RESOURCE_EXHAUSTED"
  }
}
```

---

## ⚠️ Que faire quand le quota est atteint?

### Option 1 : Attendre la réinitialisation
- Le quota se réinitialise automatiquement après 24h
- Un délai de retry est indiqué dans l'erreur (généralement 15 secondes)
- Les utilisateurs verront un message explicite

### Option 2 : Utiliser une autre clé API
```bash
# Créer une nouvelle clé API sur:
# https://makersuite.google.com/app/apikey

# Mettre à jour le fichier .env
nano .env
# Modifier: GEMINI_API_KEY=nouvelle_clé_ici

# Redémarrer l'application
python3 app.py
```

### Option 3 : Passer au plan payant
- **Gemini Pro** : Plus de quotas, meilleure performance
- **Tarification** : Pay-as-you-go après le quota gratuit
- **Lien** : https://ai.google.dev/pricing

---

## 🛠️ Gestion des erreurs implémentée

### Dans `app.py` - Fonction `call_gemini_api()`
```python
# Détection du code 429
elif response.status_code == 429:
    error_data = response.json()
    retry_delay = extract_retry_delay(error_data)
    return f"QUOTA_EXCEEDED:{retry_delay}"
```

### Dans `visitor_chatbot.js`
```javascript
if (response.status === 429) {
    // Message convivial pour l'utilisateur
    return "⚠️ Quota dépassé. Réessayez dans X secondes."
}
```

---

## 📈 Optimisation de l'utilisation

### Bonnes pratiques

1. **Limiter les appels API**
   - Mettre en cache les réponses fréquentes
   - Grouper les requêtes similaires
   - Ajouter un délai entre les requêtes

2. **Réduire la taille des prompts**
   - Contexte plus concis
   - Éviter les répétitions
   - Limiter l'historique des conversations

3. **Implémenter du rate limiting côté serveur**
   ```python
   # Exemple avec Flask-Limiter
   @limiter.limit("10 per minute")
   @app.route('/api/visitor-chat')
   def chatbot():
       # ...
   ```

4. **Utiliser plusieurs clés API en rotation**
   ```python
   GEMINI_API_KEYS = [
       os.getenv('GEMINI_API_KEY_1'),
       os.getenv('GEMINI_API_KEY_2'),
       os.getenv('GEMINI_API_KEY_3')
   ]
   # Rotation entre les clés
   ```

---

## 🔄 Solutions alternatives

### Option A : Mode dégradé
```python
# Si quota dépassé, utiliser des réponses pré-définies
FALLBACK_RESPONSES = {
    "fonctionnalités": "NeuroScan propose...",
    "précision": "Notre modèle atteint 99.7%...",
    # ...
}
```

### Option B : File d'attente
```python
# Mettre en queue les requêtes
from queue import Queue
chatbot_queue = Queue()
# Traiter progressivement
```

### Option C : Chatbot local (sans API)
- Utiliser un modèle local (Llama, Mistral)
- Plus complexe mais pas de limite
- Nécessite plus de ressources serveur

---

## 📊 Monitoring du quota

### Script de vérification
```bash
#!/bin/bash
# check_gemini_quota.sh

API_KEY=$(grep GEMINI_API_KEY .env | cut -d '=' -f2)

echo "🔍 Vérification du quota Gemini..."

# Faire un appel test
python3 << EOF
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

response = requests.post(URL, json={
    "contents": [{"parts": [{"text": "Hi"}]}]
})

if response.status_code == 200:
    print("✅ API fonctionnelle - Quota OK")
elif response.status_code == 429:
    print("❌ Quota dépassé - Attendez 24h")
else:
    print(f"⚠️  Erreur {response.status_code}")
EOF
```

---

## 📚 Ressources

- **Documentation officielle** : https://ai.google.dev/gemini-api/docs/rate-limits
- **Pricing** : https://ai.google.dev/pricing
- **Support** : https://support.google.com/googleai

---

## ✅ Résumé

| Aspect | Valeur |
|--------|--------|
| **Quota gratuit** | 250 req/jour |
| **Reset** | Tous les jours à minuit UTC |
| **Gestion d'erreur** | ✅ Implémentée |
| **Message utilisateur** | ✅ Convivial |
| **Retry automatique** | ❌ Non (manuel) |
| **Solutions alternatives** | ✅ Documentées |

---

## 🎯 Prochaines étapes recommandées

1. ✅ **Implémenter le rate limiting côté serveur**
2. ✅ **Ajouter un cache pour les questions fréquentes**
3. ✅ **Créer des réponses fallback pré-définies**
4. ⏳ **Considérer le passage au plan payant pour production**
5. ⏳ **Implémenter la rotation de clés API**

---

**Dernière mise à jour** : 5 octobre 2025
