# Débogage - Message Vide lors de l'envoi

## Problème Observé
```
Données manquantes - conversation_id: 68f0321443ff09d2a31ff485, message: vide
POST http://127.0.0.1:5000/api/chat/send 400 (BAD REQUEST)
```

L'ID de conversation est maintenant valide, mais le message arrive vide au serveur.

## Logs de Débogage Ajoutés

### Backend (app_web.py)

Dans la route `/api/chat/send` :

```python
# Log des données brutes reçues
print(f"DEBUG - Données reçues: {data}")

# Log du type et de la valeur du message
print(f"DEBUG - message_raw type: {type(message_raw)}, value: {repr(message_raw)}")

# Log du message final après traitement
print(f"DEBUG - message final: {repr(message)}, length: {len(message)}")
```

### Frontend (chat.js)

Dans la fonction `handleMessageSubmit` :

```javascript
// Log du message avant traitement
console.log('handleMessageSubmit - message:', message, 'length:', message.length);

// Avertissement si le message est vide
if (!message) {
    console.warn('Message vide, arrêt de la soumission');
    return;
}

// Log de la requête complète avant envoi
const requestBody = {
    conversation_id: window.currentConversationId,
    message: message
};
console.log('Envoi de la requête avec:', requestBody);

// Log de la réponse
console.log('Réponse reçue:', data);
```

## Tests à Effectuer

### 1. Tester l'envoi depuis le formulaire principal
1. Ouvrir la console du navigateur (F12)
2. Sélectionner une conversation existante
3. Taper un message dans le champ de texte
4. Appuyer sur Entrée ou cliquer sur le bouton Envoyer
5. Observer les logs dans la console du navigateur ET dans le terminal du serveur

**Logs attendus dans la console :**
```javascript
handleMessageSubmit - message: "Votre message ici" length: 18
Envoi de la requête avec: {conversation_id: "68f0...", message: "Votre message ici"}
```

**Logs attendus dans le terminal :**
```
DEBUG - Données reçues: {'conversation_id': '68f0...', 'message': 'Votre message ici'}
DEBUG - message_raw type: <class 'str'>, value: 'Votre message ici'
DEBUG - message final: 'Votre message ici', length: 18
```

### 2. Tester l'envoi depuis l'écran de bienvenue
1. Créer une nouvelle conversation avec un message initial
2. Observer les mêmes logs

### 3. Cas problématiques à vérifier
- Message avec uniquement des espaces : ` ` → doit être rejeté
- Message avec des retours à la ligne : `Ligne 1\nLigne 2` → doit être traité
- Message très long (> 2000 caractères) → doit être bloqué côté client
- Message vide : `` → doit être rejeté

## Causes Possibles du Problème

### 1. Vidage prématuré du textarea
❌ **Improbable** - Le code sauvegarde le message dans une variable avant de vider le textarea

### 2. Problème de timing/asynchrone
⚠️ **Possible** - Si plusieurs événements se déclenchent en même temps

### 3. Interception par un autre gestionnaire d'événements
⚠️ **Possible** - Vérifier s'il n'y a pas de double liaison d'événements

### 4. Problème avec un cache ou un intercepteur HTTP
⚠️ **Possible** - Vérifier les extensions du navigateur (bloqueurs de pubs, etc.)

### 5. Problème de sérialisation JSON
⚠️ **Possible** - Vérifier si le message contient des caractères spéciaux

## Actions de Correction Suggérées

Si les logs montrent que :

### Le message est vide côté client
→ Problème avec la récupération de la valeur du textarea
- Vérifier que `messageInput.value` contient bien le texte
- Vérifier qu'il n'y a pas de conflit d'ID dans le HTML

### Le message est non-vide côté client mais vide côté serveur
→ Problème de transmission réseau ou de sérialisation
- Vérifier les headers HTTP
- Vérifier s'il n'y a pas un proxy qui modifie les requêtes
- Essayer avec un autre navigateur

### Le message contient des caractères étranges
→ Problème d'encodage
- Vérifier l'encodage UTF-8 côté serveur et client

## Commandes Utiles

### Relancer le serveur avec les logs de débogage
```bash
cd "/home/mohammed/Bureau/ai scan"
source venv/bin/activate
python app_web.py
```

### Vider le cache du navigateur
1. Ouvrir DevTools (F12)
2. Onglet "Application" ou "Stockage"
3. Cliquer sur "Clear storage" / "Effacer le stockage"
4. Recharger la page avec Ctrl+Shift+R

### Vider le localStorage
```javascript
// Dans la console du navigateur
localStorage.clear();
location.reload();
```

## Date
16 octobre 2025 - 00:41
