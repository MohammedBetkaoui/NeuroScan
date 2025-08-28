# 📊 Espace Professionnel NeuroScan

## 🎯 Vue d'ensemble

L'Espace Professionnel NeuroScan est un tableau de bord avancé conçu pour les professionnels de santé qui souhaitent analyser leurs données d'utilisation, suivre les performances et obtenir des insights détaillés sur leurs analyses IRM.

## ✨ Fonctionnalités principales

### 📈 **Statistiques en temps réel**
- **Total des analyses** effectuées
- **Confiance moyenne** des diagnostics
- **Jours actifs** d'utilisation
- **Temps de traitement moyen** par analyse

### 📊 **Analyses par période**
- **Vue par jour** : Analyses par heure (aujourd'hui)
- **Vue par mois** : Analyses par jour (ce mois)
- **Vue par année** : Analyses par mois (cette année)
- **Graphiques interactifs** avec Chart.js

### 🧠 **Répartition des diagnostics**
- **Graphique en donut** des types de tumeurs
- **Pourcentages précis** pour chaque catégorie
- **Couleurs distinctives** pour chaque type
- **Légende interactive**

### 📋 **Analyses récentes**
- **Tableau détaillé** des 10 dernières analyses
- **Informations complètes** : date, fichier, diagnostic, confiance, temps
- **Badges colorés** pour les types de diagnostic
- **Tri chronologique** automatique

## 🗄️ Base de données

### **Tables créées automatiquement**

#### **1. Table `analyses`**
```sql
CREATE TABLE analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    filename TEXT,
    predicted_class INTEGER,
    predicted_label TEXT,
    confidence REAL,
    probabilities TEXT,
    description TEXT,
    recommendations TEXT,
    processing_time REAL,
    user_session TEXT,
    ip_address TEXT
);
```

#### **2. Table `daily_stats`**
```sql
CREATE TABLE daily_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE UNIQUE,
    total_analyses INTEGER DEFAULT 0,
    normal_count INTEGER DEFAULT 0,
    gliome_count INTEGER DEFAULT 0,
    meningiome_count INTEGER DEFAULT 0,
    pituitary_count INTEGER DEFAULT 0,
    avg_confidence REAL DEFAULT 0,
    avg_processing_time REAL DEFAULT 0
);
```

#### **3. Table `user_sessions`**
```sql
CREATE TABLE user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
    analyses_count INTEGER DEFAULT 0,
    ip_address TEXT,
    user_agent TEXT
);
```

## 🔌 API Endpoints

### **1. `/pro-dashboard`**
- **Méthode** : GET
- **Description** : Page du tableau de bord professionnel
- **Retour** : Template HTML avec interface complète

### **2. `/api/analytics/overview`**
- **Méthode** : GET
- **Description** : Statistiques générales
- **Retour** : JSON avec métriques globales

```json
{
    "success": true,
    "data": {
        "total_analyses": 268,
        "active_days": 30,
        "avg_confidence": 85.2,
        "avg_processing_time": 5.4,
        "tumor_distribution": {
            "Normal": 162,
            "Gliome": 44,
            "Méningiome": 37,
            "Tumeur pituitaire": 25
        },
        "daily_analyses": [...]
    }
}
```

### **3. `/api/analytics/period/<period>`**
- **Méthode** : GET
- **Paramètres** : period (day/month/year)
- **Description** : Analyses par période
- **Retour** : JSON avec données temporelles

```json
{
    "success": true,
    "data": {
        "labels": ["08h", "09h", "10h", ...],
        "values": [5, 8, 12, ...],
        "period": "day"
    }
}
```

### **4. `/api/analytics/recent`**
- **Méthode** : GET
- **Description** : 10 analyses les plus récentes
- **Retour** : JSON avec liste des analyses

```json
{
    "success": true,
    "data": [
        {
            "timestamp": "2025-07-26T15:30:00",
            "filename": "brain_scan_001.jpg",
            "predicted_label": "Normal",
            "confidence": 92.5,
            "processing_time": 4.2
        },
        ...
    ]
}
```

## 🎨 Interface utilisateur

### **Design moderne**
- **Cartes statistiques** avec gradients colorés
- **Graphiques interactifs** Chart.js
- **Tableau responsive** avec badges
- **Animations** et effets hover
- **Couleurs médicales** professionnelles

### **Responsive Design**
- **Mobile** : Cartes empilées, tableau scrollable
- **Tablet** : Grille 2x2 pour les stats
- **Desktop** : Layout complet 4 colonnes

### **Couleurs et thèmes**
- **Bleu** : Total analyses
- **Vert** : Confiance moyenne
- **Violet** : Jours actifs
- **Rose** : Temps de traitement

## 📊 Métriques calculées

### **Statistiques globales**
- **Total analyses** : COUNT(*) FROM analyses
- **Confiance moyenne** : AVG(confidence) * 100
- **Jours actifs** : COUNT(DISTINCT DATE(timestamp))
- **Temps moyen** : AVG(processing_time)

### **Répartition diagnostics**
- **Par type** : GROUP BY predicted_label
- **Pourcentages** : (count / total) * 100
- **Graphique donut** avec couleurs distinctes

### **Analyses temporelles**
- **Par heure** : strftime('%H', timestamp)
- **Par jour** : strftime('%d', timestamp)
- **Par mois** : strftime('%m', timestamp)

## 🔄 Mise à jour automatique

### **Sauvegarde automatique**
- **Chaque analyse** est automatiquement sauvegardée
- **Statistiques quotidiennes** mises à jour en temps réel
- **Calculs** de moyennes et totaux automatiques

### **Rafraîchissement**
- **Auto-refresh** toutes les 30 secondes
- **Données en temps réel** sans rechargement
- **Graphiques** mis à jour dynamiquement

## 🛠️ Utilisation

### **Accès au tableau de bord**
1. **Cliquer** sur "Espace Pro" dans la navigation
2. **Accéder** à `/pro-dashboard`
3. **Visualiser** les statistiques en temps réel

### **Navigation des périodes**
1. **Cliquer** sur les boutons Jour/Mois/Année
2. **Voir** les graphiques se mettre à jour
3. **Analyser** les tendances temporelles

### **Analyse des données**
1. **Examiner** les cartes de statistiques
2. **Étudier** la répartition des diagnostics
3. **Consulter** les analyses récentes
4. **Identifier** les patterns et tendances

## 📈 Données de test

### **Script de génération**
- **`generate_test_data.py`** : Génère 30 jours de données
- **268 analyses** réparties sur la période
- **Distribution réaliste** : 60% Normal, 40% Tumeurs
- **Timestamps** aléatoires pendant les heures de travail

### **Exécution**
```bash
python3 generate_test_data.py
```

### **Résultat**
- ✅ 268 analyses générées
- 📊 60.4% Normal, 16.4% Gliome, 13.8% Méningiome, 9.3% Pituitaire
- 📅 Répartition sur 30 jours
- ⏱️ Temps de traitement réalistes (2.5-8.5s)

## 🔮 Améliorations futures

### **Fonctionnalités avancées**
- **Filtres** par date, type, confiance
- **Export** des données (CSV, PDF)
- **Alertes** pour anomalies
- **Comparaisons** périodiques

### **Analyses approfondies**
- **Tendances** de performance
- **Corrélations** entre métriques
- **Prédictions** de charge
- **Benchmarking** avec moyennes

### **Intégrations**
- **PACS** pour données patients
- **HIS/RIS** pour workflow
- **APIs** externes pour enrichissement
- **Notifications** push en temps réel

## 🎯 Avantages pour les professionnels

### **📊 Suivi de performance**
- **Monitoring** de l'utilisation quotidienne
- **Évaluation** de la précision diagnostique
- **Optimisation** des workflows
- **Amélioration** continue

### **📈 Insights métier**
- **Patterns** d'utilisation
- **Répartition** des cas cliniques
- **Efficacité** temporelle
- **Qualité** des diagnostics

### **🏥 Gestion clinique**
- **Planification** des ressources
- **Formation** du personnel
- **Audit** qualité
- **Reporting** institutionnel

L'Espace Professionnel NeuroScan transforme les données d'analyse en insights actionables pour améliorer la pratique médicale et optimiser l'utilisation de l'IA diagnostique.
