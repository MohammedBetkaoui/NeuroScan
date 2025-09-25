# Chat Médical NeuroScan - Assistant IA Gemini

## 🧠 Vue d'ensemble

Le Chat Médical NeuroScan est un assistant intelligent spécialisé en neurologie et imagerie médicale, alimenté par Google Gemini. Il offre aux médecins un outil conversationnel avancé pour l'aide à la décision clinique, l'interprétation d'examens et les recommandations thérapeutiques.

## ✨ Fonctionnalités Principales

### 🔬 Intelligence Artificielle Spécialisée
- **Domaine d'expertise** : Neurologie, neurochirurgie, imagerie cérébrale
- **Modèle IA** : Google Gemini 2.0 Flash
- **Spécialisation** : Tumeurs cérébrales, pathologies neurologiques
- **Filtrage intelligent** : Réponses strictement limitées au domaine médical

### 💬 Système de Chat Avancé
- **Conversations persistantes** : Historique complet sauvegardé
- **Contexte enrichi** : L'IA utilise l'historique pour des réponses personnalisées
- **Interface responsive** : Optimisée pour desktop et mobile
- **Temps de réponse** : < 3 secondes en moyenne

### 👥 Gestion des Patients
- **Assignation de patients** : Lien conversations ↔ patients
- **Contexte patient automatique** : Antécédents, analyses récentes
- **Historique médical** : Accès aux données patients pertinentes
- **Suivi personnalisé** : Recommandations basées sur le profil patient

### 🔒 Sécurité & Confidentialité
- **Authentification médecin** : Accès restreint aux professionnels
- **Chiffrement des données** : Conversations sécurisées
- **Isolation par médecin** : Chaque médecin accède uniquement à ses données
- **Conformité RGPD** : Respect des normes de confidentialité médicale

## 🚀 Guide d'Utilisation

### Première Connexion
1. **Connexion** : Utilisez vos identifiants médicaux
2. **Accès au chat** : Dashboard → "Chat Médical IA"
3. **Nouvelle consultation** : Cliquez sur "Nouvelle consultation"
4. **Premier message** : Posez votre première question médicale

### Fonctionnalités Avancées

#### 🆕 Créer une Conversation
```
Bouton "Nouvelle consultation" → Saisir le titre → Commencer à discuter
```

#### 👤 Assigner un Patient
```
Sélectionner conversation → "Assigner patient" → Choisir dans la liste
```

#### 📱 Interface Mobile
- **Menu hamburger** : Accès au sidebar sur mobile
- **Gestes tactiles** : Navigation intuitive
- **Adaptation automatique** : Interface optimisée par taille d'écran

## 💡 Exemples d'Utilisation

### 🔬 Interprétation d'Imagerie
```
"Comment interpréter cette lésion hyperintense en T2 temporal droit chez un patient de 45 ans ?"
```
**Réponse type :**
- Diagnostic différentiel détaillé
- Critères d'imagerie spécifiques
- Recommandations d'examens complémentaires

### 🩺 Questions Cliniques
```
"Protocole de suivi pour un méningiome grade I de 2cm ?"
```
**Réponse type :**
- Fréquence de surveillance
- Examens recommandés
- Critères de progression

### 💊 Recommandations Thérapeutiques
```
"Prise en charge d'un glioblastome nouvellement diagnostiqué ?"
```
**Réponse type :**
- Protocole standard de traitement
- Options thérapeutiques
- Référencement spécialisé

## 🛠️ Architecture Technique

### Stack Technologique
- **Backend** : Flask (Python)
- **Base de données** : SQLite avec chiffrement
- **IA** : Google Gemini API
- **Frontend** : HTML5, CSS3, JavaScript ES6
- **UI Framework** : Tailwind CSS
- **Icons** : Font Awesome

### Structure de Base de Données

#### Table `chat_conversations`
```sql
CREATE TABLE chat_conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER NOT NULL,
    patient_id TEXT,
    title TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);
```

#### Table `chat_messages`
```sql
CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_medical_query BOOLEAN DEFAULT 1,
    confidence_score REAL,
    gemini_model TEXT DEFAULT 'gemini-2.0-flash'
);
```

### APIs Principales

#### Conversations
- `GET /api/chat/conversations` - Liste des conversations
- `POST /api/chat/conversations` - Créer une conversation
- `PUT /api/chat/conversations/{id}/update` - Modifier une conversation

#### Messages
- `GET /api/chat/conversations/{id}/messages` - Messages d'une conversation
- `POST /api/chat/send` - Envoyer un message et obtenir une réponse IA

#### Patients
- `GET /api/patients/list` - Liste des patients pour assignation

## 🔧 Configuration

### Variables d'Environnement
```bash
GEMINI_API_KEY=votre_cle_api_gemini
DATABASE_PATH=neuroscan_analytics.db
FLASK_ENV=development
```

### Configuration Gemini
```python
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_CONFIG = {
    "temperature": 0.7,
    "topK": 40,
    "topP": 0.95,
    "maxOutputTokens": 1024
}
```

## 📊 Métriques et Analytics

### Suivi des Conversations
- **Nombre total de conversations**
- **Messages par conversation**
- **Score de confiance moyen des réponses**
- **Temps de réponse moyen**

### Analyse de Qualité
- **Questions médicales vs non-médicales**
- **Satisfaction utilisateur (implicite)**
- **Utilisation par médecin**
- **Contexte patient utilisé**

## 🚨 Limitations et Avertissements

### Limitations Techniques
- **Domaine strict** : Neurologie et imagerie uniquement
- **Pas de diagnostic final** : Outil d'aide à la décision uniquement
- **Dépendance internet** : Nécessite une connexion pour Gemini API
- **Limite de tokens** : Conversations très longues peuvent être tronquées

### Avertissements Médicaux
- ⚠️ **Ne remplace pas le jugement clinique**
- ⚠️ **Toujours vérifier les recommandations**
- ⚠️ **En cas d'urgence, contacter directement les services d'urgence**
- ⚠️ **Les réponses sont basées sur des connaissances générales**

## 🧪 Tests et Validation

### Tests Automatisés
```bash
# Données de test
python create_chat_test_data.py

# Tests unitaires (à implémenter)
python -m pytest tests/test_chat_medical.py
```

### Comptes de Test
```
Email: test@neuroscan.com
Mot de passe: test123
Patients de test: PAT001, PAT002, PAT003
```

## 🔄 Roadmap

### Version 1.1 (Court terme)
- [ ] Support audio (speech-to-text)
- [ ] Export conversations PDF
- [ ] Recherche dans l'historique
- [ ] Notifications temps réel

### Version 1.2 (Moyen terme)
- [ ] Intégration DICOM viewer
- [ ] Analyse d'images dans le chat
- [ ] Multi-langues (anglais, espagnol)
- [ ] API REST publique

### Version 2.0 (Long terme)
- [ ] Modèle IA personnalisé spécialisé
- [ ] Intégration dossier médical électronique
- [ ] Collaboration multi-médecins
- [ ] Mobile app native

## 🐛 Dépannage

### Problèmes Courants

#### "Erreur API Gemini"
- Vérifier la clé API
- Contrôler la connexion internet
- Vérifier les quotas API

#### "Conversation ne se charge pas"
- Rafraîchir la page
- Vérifier la session utilisateur
- Contrôler la base de données

#### "Patient non trouvé pour assignation"
- Vérifier que le patient existe
- Contrôler les permissions médecin
- Synchroniser les données

### Support Technique
- **Email** : support@neuroscan.medical
- **Documentation** : [URL vers la doc complète]
- **Issues GitHub** : [URL du repository]

## 📝 Changelog

### Version 1.0.0 (2024-09-24)
- ✅ Chat médical avec Gemini AI
- ✅ Gestion des conversations persistantes
- ✅ Assignation de patients
- ✅ Interface responsive
- ✅ Authentification sécurisée
- ✅ Base de données structurée
- ✅ Limitation domaine médical
- ✅ Score de confiance des réponses

---

## 📄 Licence

Ce projet est sous licence propriétaire NeuroScan Medical Technologies.
© 2024 NeuroScan - Tous droits réservés.

## 👥 Crédits

- **Développement** : Équipe NeuroScan
- **IA** : Google Gemini
- **Interface** : Tailwind CSS + Font Awesome
- **Tests** : Données synthétiques médicales

---

*Pour toute question technique ou suggestion d'amélioration, n'hésitez pas à contacter l'équipe de développement.*
