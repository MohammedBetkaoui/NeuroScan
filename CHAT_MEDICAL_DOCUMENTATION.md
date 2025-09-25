# Chat Médical NeuroScan - Documentation Complète

## 📋 Vue d'ensemble

Le Chat Médical NeuroScan est un système de consultation IA avancé intégré avec Google Gemini, spécialement conçu pour les professionnels de santé dans le domaine de la neurologie et de l'imagerie médicale.

## 🏥 Fonctionnalités Principales

### 1. Interface Professionnelle
- **Design responsive** : Adapté aux ordinateurs de bureau, tablettes et mobiles
- **Interface intuitive** : Navigation simple pour médecins, patients et professionnels de santé
- **Thème médical** : Couleurs sobres et icônes spécialisées dans le domaine de la santé

### 2. Gestion des Conversations
- **Historique complet** : Toutes les conversations sont sauvegardées et horodatées
- **Sessions persistantes** : Les utilisateurs peuvent reprendre leurs conversations
- **Organisation par patient** : Chaque conversation peut être liée à un patient spécifique
- **Barre latérale** : Liste des conversations récentes avec navigation rapide

### 3. IA Médicale Spécialisée
- **Domaine strictement médical** : L'IA ne répond qu'aux questions de santé
- **Expertise neurologique** : Spécialisée en neurologie et imagerie cérébrale
- **Contexte conversationnel** : L'IA utilise l'historique pour des réponses précises
- **Score de confiance** : Chaque réponse inclut un niveau de confiance

### 4. Sécurité et Authentification
- **Connexion obligatoire** : Seuls les médecins authentifiés peuvent accéder
- **Données sécurisées** : Stockage crypté des conversations
- **Sessions médecin** : Chaque conversation est liée à l'ID du médecin
- **Confidentialité** : Respect des règles de confidentialité médicale

## 🛠 Architecture Technique

### Base de Données
```sql
-- Conversations de chat
CREATE TABLE chat_conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER NOT NULL,
    patient_id TEXT,
    title TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Messages de chat
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

### API Endpoints
- `GET /chat` - Page principale du chat médical
- `GET /api/chat/conversations` - Récupérer les conversations du médecin
- `POST /api/chat/conversations` - Créer une nouvelle conversation
- `GET /api/chat/conversations/{id}/messages` - Récupérer les messages d'une conversation
- `POST /api/chat/send` - Envoyer un message et obtenir une réponse IA
- `PUT /api/chat/conversations/{id}/update` - Mettre à jour une conversation
- `GET /api/patients/list` - Liste des patients pour assignation

### Intégration Gemini
- **Modèle** : gemini-2.0-flash
- **Prompt système** : Restriction au domaine médical neurologique
- **Contexte** : Injection de l'historique et des informations patient
- **Sécurité** : Filtrage des requêtes non-médicales

## 🎯 Utilisation

### Pour les Médecins

1. **Accès** : Se connecter au dashboard NeuroScan
2. **Navigation** : Cliquer sur "Chat Médical IA" dans le dashboard
3. **Nouvelle consultation** : Cliquer sur "Nouvelle consultation"
4. **Questions** : Poser des questions médicales spécialisées
5. **Historique** : Accéder aux conversations précédentes via la barre latérale

### Exemples de Questions Médicales
- "Quels sont les signes d'alarme dans une IRM cérébrale ?"
- "Comment interpréter une lésion hypointense en T1 ?"
- "Quelles sont les recommandations pour le suivi d'un gliome ?"
- "Différences entre méningiome et schwannome à l'IRM ?"

## 🔒 Sécurité et Confidentialité

### Mesures de Protection
- **Authentification** : Connexion médecin obligatoire
- **Chiffrement** : Communications sécurisées HTTPS
- **Isolation** : Chaque médecin n'accède qu'à ses propres conversations
- **Audit** : Traçabilité complète des interactions
- **Disclaimer** : Rappel que l'IA ne remplace pas une consultation

### Conformité
- **RGPD** : Respect des règles de protection des données
- **Secret médical** : Confidentialité des échanges
- **Données sensibles** : Aucune donnée patient identifiable transmise à l'IA

## 📱 Interface Utilisateur

### Desktop
- **Barre latérale** : 320px de largeur, liste des conversations
- **Zone principale** : Chat en temps réel avec historique
- **En-tête** : Informations de conversation et actions
- **Zone de saisie** : Textarea expansible avec compteur de caractères

### Mobile
- **Menu hamburger** : Barre latérale escamotable
- **Interface adaptative** : Optimisée pour écrans tactiles
- **Gestes** : Navigation intuitive
- **Responsive** : S'adapte à toutes les tailles d'écran

## 🚀 Fonctionnalités Avancées

### Contexte Patient
- **Association** : Lier une conversation à un patient
- **Historique médical** : Injection des antécédents dans le contexte IA
- **Analyses récentes** : Référence aux examens précédents
- **Recommandations personnalisées** : Basées sur le profil patient

### Métriques et Analytics
- **Score de confiance** : Évaluation de la qualité des réponses
- **Détection médicale** : Classification automatique des questions
- **Temps de réponse** : Optimisation des performances
- **Statistiques d'usage** : Suivi de l'utilisation par médecin

## 🔧 Configuration et Déploiement

### Variables d'Environnement
```bash
GEMINI_API_KEY="votre_clé_api_gemini"
DATABASE_PATH="neuroscan_analytics.db"
SECRET_KEY="neuroscan_secret_key_2024_medical_auth"
```

### Installation
```bash
# Installation des dépendances
pip install google-generativeai flask sqlite3

# Initialisation de la base de données
python -c "from app import init_database; init_database()"

# Démarrage de l'application
python app.py
```

## 📊 Monitoring et Maintenance

### Métriques à Surveiller
- **Taux de réponse IA** : Pourcentage de réponses réussies
- **Temps de réponse** : Latence des appels Gemini
- **Utilisation** : Nombre de conversations par médecin
- **Erreurs** : Monitoring des échecs d'API

### Maintenance Préventive
- **Sauvegarde BD** : Backup quotidien des conversations
- **Rotation logs** : Gestion des fichiers de log
- **Mise à jour modèle** : Suivi des versions Gemini
- **Tests de sécurité** : Audit régulier des vulnérabilités

## 🎓 Formation et Support

### Guide Utilisateur
1. **Formation initiale** : Session de présentation des fonctionnalités
2. **Documentation** : Guide d'utilisation disponible dans l'interface
3. **Support technique** : Assistance pour les problèmes techniques
4. **Mise à jour** : Communication des nouvelles fonctionnalités

### Bonnes Pratiques
- **Questions précises** : Formuler des questions claires et spécifiques
- **Contexte médical** : Inclure les informations cliniques pertinentes
- **Vérification** : Toujours vérifier les recommandations de l'IA
- **Confidentialité** : Ne pas inclure d'informations patient identifiables

## 📞 Contact et Support

Pour toute question technique ou médicale concernant le Chat Médical NeuroScan :

- **Support Technique** : support@neuroscan.fr
- **Questions Médicales** : medical@neuroscan.fr
- **Documentation** : https://docs.neuroscan.fr/chat-medical

---

*Dernière mise à jour : 24 septembre 2025*
*Version : 1.0.0*
*Développé avec ❤️ pour la communauté médicale*
