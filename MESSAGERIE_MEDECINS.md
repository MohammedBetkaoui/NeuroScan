# 📧 Système de Messagerie entre Médecins - NeuroScan AI

## Vue d'ensemble

Un système complet de messagerie professionnelle permettant aux médecins de communiquer entre eux et de partager des analyses médicales en temps réel.

---

## ✨ Fonctionnalités principales

### 1. 💬 Messages Texte entre Médecins
- **Conversations privées** : Échange de messages texte en temps réel
- **Liste des médecins** : Accès à tous les médecins inscrits sur la plateforme
- **Recherche** : Filtrage des conversations et médecins par nom, spécialité ou hôpital
- **Historique** : Conservation complète de l'historique des conversations
- **Status de lecture** : Indicateurs de messages lus/non lus
- **Notifications** : Badge de messages non lus

### 2. 🔄 Partage d'Analyses
- **Partage direct** : Partager une analyse médicale avec un collègue
- **Message d'accompagnement** : Ajouter un commentaire au partage
- **Aperçu visuel** : Miniature de l'image IRM/scanner
- **Détails complets** : Diagnostic, confiance, probabilités, recommandations
- **Traçabilité** : Information sur le médecin qui a partagé l'analyse

### 3. 👁️ Visualisation des Analyses Partagées
- **Liste dédiée** : Vue de toutes les analyses partagées reçues
- **Détails complets** : Accès aux informations complètes de l'analyse
- **Informations du patient** : Nom, date d'examen, diagnostic
- **Données médicales** : Graphiques de probabilités, recommandations
- **Médecin émetteur** : Nom, spécialité, hôpital du médecin qui a partagé

---

## 🗄️ Architecture Base de Données (MongoDB)

### Collections créées

#### `doctor_conversations`
Gestion des conversations entre médecins
```javascript
{
  _id: ObjectId,
  participants: [ObjectId, ObjectId], // IDs des 2 médecins
  created_at: DateTime,
  updated_at: DateTime
}
```

**Index** :
- `participants` (ASCENDING)
- `updated_at` (DESCENDING)

#### `doctor_messages`
Stockage des messages échangés
```javascript
{
  _id: ObjectId,
  conversation_id: ObjectId,
  sender_id: ObjectId,
  content: String,
  message_type: String, // "text" ou "analysis_share"
  
  // Pour les partages d'analyses
  analysis_id: ObjectId, // (optionnel)
  analysis_data: {
    patient_name: String,
    exam_date: String,
    predicted_label: String,
    confidence: Number,
    probabilities: Object,
    recommendations: Array,
    image_filename: String
  },
  
  is_read: Boolean,
  read_at: DateTime,
  created_at: DateTime
}
```

**Index** :
- `conversation_id` (ASCENDING)
- `sender_id` (ASCENDING)
- `created_at` (DESCENDING)
- `message_type` (ASCENDING)
- `is_read` (ASCENDING)
- `analysis_id` (ASCENDING)

---

## 🛣️ Routes API

### Gestion des médecins

#### `GET /api/messages/doctors`
Récupère la liste de tous les médecins (sauf l'utilisateur connecté)

**Réponse** :
```json
{
  "success": true,
  "doctors": [
    {
      "id": "string",
      "first_name": "string",
      "last_name": "string",
      "full_name": "string",
      "email": "string",
      "specialty": "string",
      "hospital": "string",
      "is_online": false
    }
  ]
}
```

### Gestion des conversations

#### `GET /api/messages/conversations`
Récupère toutes les conversations du médecin connecté

**Réponse** :
```json
{
  "success": true,
  "conversations": [
    {
      "id": "string",
      "other_doctor": {
        "id": "string",
        "first_name": "string",
        "last_name": "string",
        "full_name": "string",
        "specialty": "string",
        "hospital": "string"
      },
      "last_message": {
        "content": "string",
        "created_at": "ISO8601",
        "is_from_me": false
      },
      "unread_count": 0,
      "created_at": "ISO8601"
    }
  ]
}
```

#### `POST /api/messages/conversations`
Créer une nouvelle conversation avec un médecin

**Body** :
```json
{
  "recipient_id": "string"
}
```

**Réponse** :
```json
{
  "success": true,
  "conversation_id": "string",
  "exists": false
}
```

### Gestion des messages

#### `GET /api/messages/conversations/<conversation_id>/messages`
Récupère tous les messages d'une conversation

**Réponse** :
```json
{
  "success": true,
  "messages": [
    {
      "id": "string",
      "content": "string",
      "message_type": "text",
      "is_from_me": false,
      "sender": {
        "id": "string",
        "first_name": "string",
        "last_name": "string",
        "full_name": "string",
        "specialty": "string"
      },
      "is_read": false,
      "created_at": "ISO8601"
    }
  ]
}
```

#### `POST /api/messages/send`
Envoyer un message texte

**Body** :
```json
{
  "conversation_id": "string",
  "content": "string"
}
```

**Réponse** :
```json
{
  "success": true,
  "message_id": "string",
  "created_at": "ISO8601"
}
```

### Partage d'analyses

#### `POST /api/messages/share-analysis`
Partager une analyse avec un médecin

**Body** :
```json
{
  "conversation_id": "string",
  "analysis_id": "string",
  "message": "string (optionnel)"
}
```

**Réponse** :
```json
{
  "success": true,
  "message_id": "string",
  "created_at": "ISO8601"
}
```

#### `GET /api/messages/shared-analyses`
Récupère toutes les analyses partagées avec le médecin

**Réponse** :
```json
{
  "success": true,
  "shared_analyses": [
    {
      "message_id": "string",
      "analysis_id": "string",
      "analysis_data": {
        "patient_name": "string",
        "exam_date": "string",
        "predicted_label": "string",
        "confidence": 0.95,
        "image_filename": "string"
      },
      "message": "string",
      "sender": {
        "id": "string",
        "first_name": "string",
        "last_name": "string",
        "full_name": "string",
        "specialty": "string",
        "hospital": "string"
      },
      "is_read": false,
      "shared_at": "ISO8601"
    }
  ]
}
```

#### `GET /api/messages/analysis/<analysis_id>`
Obtenir les détails complets d'une analyse partagée

**Réponse** :
```json
{
  "success": true,
  "analysis": {
    "id": "string",
    "patient_name": "string",
    "patient_id": "string",
    "exam_date": "string",
    "predicted_label": "string",
    "confidence": 0.95,
    "probabilities": {
      "Normal": 0.05,
      "Gliome": 0.87,
      "Méningiome": 0.06,
      "Tumeur pituitaire": 0.02
    },
    "recommendations": ["array"],
    "image_filename": "string",
    "owner_doctor": {
      "id": "string",
      "first_name": "string",
      "last_name": "string",
      "full_name": "string",
      "specialty": "string",
      "hospital": "string"
    }
  }
}
```

---

## 📁 Structure des Fichiers

```
NeuroScan-main/
├── app_web.py                          # Routes API principales (lignes 460-1020)
├── database/
│   └── mongodb_connector.py            # Configuration MongoDB + index
├── templates/
│   └── messages.html                   # Interface de messagerie
├── static/
│   ├── css/
│   │   └── messages_pro.css           # Styles de la messagerie
│   └── js/
│       └── messages_pro.js            # Logique frontend complète
└── MESSAGERIE_MEDECINS.md            # Cette documentation
```

---

## 🎨 Interface Utilisateur

### Layout Principal
- **Sidebar gauche** : Liste des conversations avec recherche et filtres
- **Zone centrale** : Messages de la conversation active
- **Input en bas** : Zone de saisie avec boutons d'action

### Fonctionnalités UI
1. **Recherche de conversations** : Filtrage en temps réel
2. **Avatars avec initiales** : Générés automatiquement (prénom + nom)
3. **Status de lecture** : Icônes de confirmation (simple/double check)
4. **Badge de messages non lus** : Compteur visible
5. **Timestamps intelligents** : Affichage relatif (à l'instant, 5min, hier, etc.)
6. **Modals** :
   - Nouveau message : Sélection d'un médecin
   - Partage d'analyse : Sélection d'une analyse à partager
   - Analyses partagées : Liste des analyses reçues
   - Détails analyse : Vue complète avec graphiques

### Composants Visuels
- **Message texte** : Bulles de message avec alignement (envoyé/reçu)
- **Carte de partage d'analyse** : Card avec miniature, diagnostic, confiance
- **Graphiques de probabilités** : Barres horizontales colorées
- **Badges de confiance** : Vert (>90%), Orange (70-90%), Rouge (<70%)

---

## 🔄 Flux de Données

### Envoi d'un Message
1. Utilisateur saisit le message
2. Click sur bouton d'envoi ou Enter
3. Requête POST `/api/messages/send`
4. Message enregistré en base
5. Conversation mise à jour (updated_at)
6. Rechargement des messages
7. Scroll automatique vers le bas

### Partage d'Analyse
1. Click sur bouton "Partager une analyse"
2. Modal avec liste des analyses du médecin
3. Sélection d'une analyse
4. Saisie d'un message optionnel
5. Requête POST `/api/messages/share-analysis`
6. Message spécial créé avec type "analysis_share"
7. Notification créée pour le destinataire
8. Affichage carte d'analyse dans la conversation

### Visualisation d'Analyse Partagée
1. Click sur carte d'analyse dans la conversation
2. Requête GET `/api/messages/analysis/<id>`
3. Modal avec détails complets
4. Affichage image, graphiques, recommandations
5. Option d'ouvrir dans l'analyseur complet

---

## ⚡ Fonctionnalités Temps Réel

### Polling des Messages
- Intervalle : 10 secondes
- Rafraîchissement automatique des conversations
- Mise à jour du compteur de messages non lus
- Rechargement des messages de la conversation active

### Marquage Automatique comme Lu
- Messages marqués comme lus lors de l'ouverture de la conversation
- Champ `is_read` mis à `true`
- Champ `read_at` enregistré avec timestamp
- Mise à jour immédiate du compteur

---

## 🔒 Sécurité

### Authentification
- Décorateur `@login_required` sur toutes les routes
- Vérification du médecin connecté via session
- Exclusion du mot de passe dans toutes les requêtes

### Autorisations
- Un médecin ne peut voir que ses propres conversations
- Vérification de participation avant accès aux messages
- Seules les analyses du médecin peuvent être partagées
- Validation de l'ownership avant partage

### Validation des Données
- Vérification de l'existence des conversations
- Validation des IDs ObjectId
- Protection contre les requêtes malformées
- Gestion complète des erreurs

---

## 🎯 Cas d'Utilisation

### Scénario 1 : Demande d'Avis Médical
1. Dr. Ahmed ouvre une conversation avec Dr. Sarah
2. Il envoie : "Bonjour, j'ai un cas complexe de gliome"
3. Il partage l'analyse IRM du patient
4. Dr. Sarah reçoit la notification
5. Elle consulte l'analyse partagée
6. Elle répond avec son avis médical

### Scénario 2 : Suivi de Patient
1. Dr. Martin partage une analyse de suivi
2. Message : "Évolution du patient après 3 mois"
3. Dr. Benali compare avec l'analyse précédente
4. Discussion sur l'efficacité du traitement
5. Décision commune sur les prochaines étapes

### Scénario 3 : Consultation d'Expert
1. Médecin généraliste identifie une anomalie
2. Partage avec neurochirurgien spécialisé
3. Expert analyse les données partagées
4. Recommandations pour examens complémentaires
5. Référence du patient si nécessaire

---

## 📊 Métriques et Analytics

### Données Trackées
- Nombre total de conversations par médecin
- Nombre de messages échangés
- Nombre d'analyses partagées
- Temps de réponse moyen
- Taux de lecture des messages
- Analyses partagées par type de diagnostic

### Statistiques Potentielles
- Médecins les plus actifs
- Spécialités les plus consultées
- Types d'analyses les plus partagés
- Heures de pointe de messagerie
- Taux d'engagement

---

## 🚀 Améliorations Futures

### Court Terme
- [ ] WebSocket pour temps réel sans polling
- [ ] Notifications push navigateur
- [ ] Pièces jointes (PDF, images supplémentaires)
- [ ] Émojis et réactions rapides
- [ ] Edition/suppression de messages

### Moyen Terme
- [ ] Appels vidéo intégrés
- [ ] Groupes de discussion entre médecins
- [ ] Partage de rapports PDF générés
- [ ] Historique de recherche dans messages
- [ ] Favoris et messages épinglés

### Long Terme
- [ ] Intelligence artificielle pour suggestions
- [ ] Transcription vocale des messages
- [ ] Traduction automatique multilingue
- [ ] Intégration calendrier pour rendez-vous
- [ ] Statistiques d'utilisation avancées

---

## 🐛 Débogage

### Logs Importants
```python
# Dans app_web.py
print(f"✅ {allDoctors.length} médecins chargés")
print(f"❌ Erreur récupération conversations: {e}")
print(f"✅ Collections MongoDB initialisées avec index")
```

### Console JavaScript
```javascript
console.log('📧 Initialisation de la messagerie professionnelle...');
console.log(`✅ ${allDoctors.length} médecins chargés`);
console.log('❌ Erreur chargement médecins:', error);
```

### Vérifications MongoDB
```javascript
// Vérifier les conversations
db.doctor_conversations.find().pretty()

// Vérifier les messages
db.doctor_messages.find().sort({created_at: -1}).pretty()

// Compter les non lus
db.doctor_messages.countDocuments({is_read: false})
```

---

## 📝 Notes Importantes

1. **Champs nom** : Le système utilise `first_name` et `last_name` avec fallback sur `full_name`
2. **Timestamps** : Tous les timestamps sont en ISO8601
3. **ObjectId** : Conversion systématique en string pour le frontend
4. **Images** : Les images d'analyses sont servies via `/uploads/<filename>`
5. **Polling** : Intervalle de 10 secondes, peut être ajusté selon la charge

---

## ✅ Tests Recommandés

### Tests Fonctionnels
- [ ] Créer une nouvelle conversation
- [ ] Envoyer un message texte
- [ ] Recevoir et lire un message
- [ ] Partager une analyse
- [ ] Consulter une analyse partagée
- [ ] Rechercher une conversation
- [ ] Vérifier les compteurs non lus
- [ ] Tester le polling des messages

### Tests d'Intégration
- [ ] Conversation entre 2 médecins
- [ ] Partage multiple d'analyses
- [ ] Navigation entre conversations
- [ ] Scroll et chargement des messages
- [ ] Responsive sur mobile

---

## 📞 Support

Pour toute question ou problème :
- Email : mohammed.betkaoui@neuroscan.ai
- Documentation : Ce fichier MESSAGERIE_MEDECINS.md
- Code source : `app_web.py`, `messages_pro.js`, `messages_pro.css`

---

**Version** : 1.0  
**Date** : 15 octobre 2025  
**Auteur** : NeuroScan AI Team  
**Licence** : Propriétaire
