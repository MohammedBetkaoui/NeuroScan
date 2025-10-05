# 🔧 Guide de Dépannage - Chat NeuroScan AI

## 📋 Problème : Le chat ne fonctionne pas dans l'application desktop

### ✅ Ce qui fonctionne
- ✅ Configuration API Gemini OK
- ✅ Routes API du chat OK
- ✅ Le chat fonctionne sur le site web

### 🔍 Causes possibles

#### 1. **Problème de permissions JavaScript dans PyWebView**
PyWebView peut bloquer certaines APIs JavaScript comme `localStorage`, `sessionStorage`, ou `fetch`.

#### 2. **Erreurs dans la console JavaScript**
Les erreurs JavaScript ne sont pas visibles par défaut dans PyWebView.

#### 3. **CORS / Content Security Policy**
PyWebView peut avoir des restrictions CORS différentes du navigateur.

---

## 🛠️ Solutions

### Solution 1: Activer le mode debug de PyWebView

Le fichier `run_app.py` a été modifié pour activer `debug=True` :

```python
webview.start(debug=True)
```

**Pour tester:**
```bash
./launch_neuroscan.sh
```

Les erreurs JavaScript seront maintenant affichées dans le terminal.

---

### Solution 2: Vérifier localStorage/sessionStorage

Le chat utilise probablement localStorage pour stocker les conversations.

**Test dans la console du navigateur (si accessible) :**
```javascript
// Vérifier si localStorage fonctionne
try {
    localStorage.setItem('test', 'value');
    localStorage.getItem('test');
    console.log('localStorage: OK');
} catch(e) {
    console.error('localStorage: ERREUR', e);
}
```

---

### Solution 3: Utiliser la version web temporairement

Si le chat ne fonctionne toujours pas dans PyWebView :

1. **Lancer le serveur Flask seul:**
   ```bash
   source venv/bin/activate
   python3 app.py
   ```

2. **Ouvrir dans un navigateur:**
   ```
   http://127.0.0.1:5000
   ```

Le chat fonctionnera parfaitement dans un navigateur standard.

---

### Solution 4: Vérifier les erreurs dans le terminal

Quand vous lancez l'application avec `./launch_neuroscan.sh`, regardez le terminal pour:

- ❌ Erreurs JavaScript (maintenant visibles avec `debug=True`)
- ❌ Erreurs 403/404/500 lors des appels API
- ❌ Messages d'erreur Gemini API

**Exemples d'erreurs courantes:**

```bash
# Quota API dépassé
⚠️  Quota API Gemini dépassé: 429
   Limite gratuite: 250 requêtes/jour atteinte

# Erreur de permission
SecurityError: Failed to read the 'localStorage' property

# Erreur CORS
Access to fetch blocked by CORS policy
```

---

### Solution 5: Test direct de l'API

Testez l'API du chat directement:

```bash
# Lancer l'application
./launch_neuroscan.sh

# Dans un autre terminal
curl -X POST http://127.0.0.1:5000/api/visitor-chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Bonjour"}'
```

Si ça fonctionne, le problème est dans PyWebView, pas dans Flask.

---

## 🎯 Actions à faire MAINTENANT

### 1. Relancer l'application en mode debug

```bash
./launch_neuroscan.sh
```

### 2. Aller sur la page de chat

Cliquez sur "Chat Médical IA" dans le dashboard.

### 3. Vérifier les logs du terminal

Cherchez les erreurs rouges dans le terminal. Elles vous diront exactement quel est le problème.

### 4. Tester l'envoi d'un message

Tapez un message dans le chat et envoyez-le. Observez le terminal.

---

## 📊 Messages d'erreur typiques et solutions

### Erreur: "Failed to fetch"
**Cause:** PyWebView bloque les requêtes fetch  
**Solution:** Utiliser XMLHttpRequest au lieu de fetch (modification du code JS)

### Erreur: "localStorage is not defined"
**Cause:** PyWebView n'a pas accès à localStorage  
**Solution:** Utiliser des cookies ou sessionStorage côté serveur

### Erreur: 429 Too Many Requests
**Cause:** Quota API Gemini dépassé  
**Solution:** Attendre 24h ou utiliser une autre clé API

### Erreur: "CORS policy"
**Cause:** Restriction CORS dans PyWebView  
**Solution:** Ajouter les headers CORS dans Flask

---

## 🔧 Modification du code si nécessaire

Si le problème persiste, nous pouvons modifier le code JavaScript du chat pour:

1. **Remplacer fetch par XMLHttpRequest**
2. **Utiliser des cookies au lieu de localStorage**
3. **Ajouter des fallbacks pour PyWebView**

---

## 📞 Prochaine étape

**Lancez l'application et envoyez-moi les erreurs du terminal !**

```bash
./launch_neuroscan.sh
# Allez sur le chat
# Envoyez un message
# Copiez les erreurs du terminal ici
```

Je pourrai alors diagnostiquer précisément le problème et le corriger ! 🚀
