# NeuroScan - Analyse de Tumeurs Cérébrales

Application Flask utilisant l'intelligence artificielle pour l'analyse et la détection de tumeurs cérébrales à partir d'images IRM.

Nouveautés (09/2025)
- Nouvelle interface PRO du profil patient (vue moderne, responsive) avec KPIs, courbe d'évolution de la confiance, tableau des analyses, notes et export JSON.
- Unification des dashboards et cohérence UI via `base_dashboard.html` et `dashboard-unified.css`.
- Endpoints APIs étendus pour analytics et historique détaillé du patient.

## Fonctionnalités

- **Interface web moderne** : Interface utilisateur intuitive avec design responsive
- **Analyse IA** : Utilisation d'un modèle CNN entraîné pour la classification de tumeurs cérébrales
- **Support multi-formats** : Compatible avec les formats DICOM, NIfTI, JPEG, PNG
- **Résultats détaillés** : Probabilités pour chaque type de tumeur et recommandations cliniques
- **Visualisation** : Affichage des zones suspectes sur l'image analysée
- **Profil Patient PRO** : Nouvelle page `patient_profile_pro.html` avec:
   - Carte patient (infos clés, badge de risque, premières/dernières analyses)
   - KPIs (analyses totales, normal/anormal, confiance moyenne)
   - Graphique d'évolution (confiance) avec Chart.js
   - Tableau des analyses (date, diagnostic, confiance, lien image)
   - Notes/recommandations et export JSON

## Types de tumeurs détectées

1. **Normal** - Aucune anomalie détectée
2. **Gliome** - Tumeur des cellules gliales
3. **Méningiome** - Tumeur des méninges
4. **Tumeur pituitaire** - Tumeur de l'hypophyse

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Démarrage rapide (Mode démo)

Pour tester l'application rapidement avec des prédictions simulées :

```bash
# Rendre le script exécutable et le lancer
chmod +x start_demo.sh
./start_demo.sh
```

### Installation complète avec PyTorch

Pour utiliser le vrai modèle d'IA :

```bash
# 1. Démarrer d'abord en mode démo pour créer l'environnement
./start_demo.sh

# 2. Dans un autre terminal, installer PyTorch
./install_pytorch.sh

# 3. Utiliser l'application complète
source venv/bin/activate
python3 app.py
```

### Installation manuelle

```bash
# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances de base
pip install Flask Pillow numpy Werkzeug

# Pour le modèle complet, installer PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python
```

### Fichiers requis

Assurez-vous que les fichiers suivants sont présents :
- `best_brain_tumor_model.pth` - Le modèle entraîné
- `app.py` - Application Flask avec IA
- `app_demo.py` - Application Flask en mode démo
- `templates/index.html` - Interface utilisateur
- `start_demo.sh` - Script de démarrage rapide
- `install_pytorch.sh` - Script d'installation PyTorch

## Utilisation

### Authentification et accès
- Créez un compte médecin via `/register` ou connectez-vous via `/login`.
- Les sessions sont persistées et les routes privées nécessitent une connexion.

### Mode démo (recommandé pour tester)

```bash
./start_demo.sh
```

### Mode complet avec IA

```bash
source venv/bin/activate
python3 app.py
```

L'application sera accessible à l'adresse : `http://localhost:5000`

### Parcours d'utilisation
1. Connectez-vous (ou enregistrez-vous) en tant que médecin.
2. Depuis le dashboard, uploadez une image avec les informations patient (ID, nom, date d'examen) via `/upload`.
3. L'analyse est enregistrée et attribuée au patient et au médecin connecté.
4. Ouvrez la liste des patients via `/patients` et cliquez sur un patient pour voir son profil PRO `/patient/<patient_id>`.

### Page profil patient (PRO)
- Route: `/patient/<patient_id>`
- Template: `templates/patient_profile_pro.html`
- Sections:
   - Carte patient (avatar, infos, risque, dates clés)
   - KPIs (totaux, normal/anormal, confiance moyenne)
   - Onglets: Evolution (graph), Analyses (table), Notes (desc + reco)
   - Actions: Exporter (JSON), Nouvelle analyse

Conseils d'utilisation:
- Renseignez l'ID patient et la date d'examen lors de l'upload pour assurer le suivi.
- Le graphique d'évolution utilise la confiance (0..1) transformée en % côté front.

## Architecture du modèle

Le modèle utilise une architecture CNN (Convolutional Neural Network) avec :
- 5 couches de convolution avec pooling
- 2 couches fully connected
- Dropout pour la régularisation
- 4 classes de sortie (Normal, Gliome, Méningiome, Tumeur pituitaire)

## API Endpoints

### `GET /`
Page d'accueil avec l'interface utilisateur

### `POST /upload`
Upload et analyse d'une image IRM

**Paramètres :**
- `file` : Fichier image (multipart/form-data)

**Réponse :**
```json
{
  "success": true,
  "image_url": "data:image/jpeg;base64,...",
  "prediction": "Gliome",
  "confidence": 0.89,
  "probabilities": {
    "Normal": 0.05,
    "Gliome": 0.89,
    "Méningiome": 0.04,
    "Tumeur pituitaire": 0.02
  },
  "is_tumor": true,
  "recommendations": [...]
}
```

### `GET /health`
Vérification de l'état de l'application

### Routes principales (UI)
- `GET /dashboard` — Tableau de bord médecin
- `GET /patients` — Liste des patients du médecin
- `GET /patient/<patient_id>` — Profil patient (interface PRO)
- `GET /alerts` — Alertes médicales
- `GET /pro-dashboard` — Statistiques pro
- `GET /pro-dashboard-advanced` — Statistiques avancées
- `GET /platform-stats` — Stats globales plateforme

### API Patients et Analytics
- `GET /api/my-patients` — Patients du médecin connecté
- `GET /api/patients/<patient_id>/detailed-history` — Historique détaillé + métriques
- `GET /api/patients/<patient_id>/comparison` — Comparaison 2 dernières analyses
- `GET /api/evolution/summary` — Résumé global des évolutions
- `GET /api/analytics/overview` — Stat perso du médecin
- `GET /api/analytics/platform-overview` — Stat globales
- `GET /api/analytics/filter-counts` — Compteurs filtres
- `POST /api/analytics/filter-preview` — Prévisualisation filtres

## Sécurité et limitations

⚠️ **IMPORTANT** : Cette application est destinée à des fins éducatives et de recherche uniquement. Elle ne doit pas être utilisée pour des diagnostics médicaux réels sans validation par des professionnels de santé qualifiés.

### Limitations :
- Taille maximale des fichiers : 16MB
- Formats d'image limités
- Modèle entraîné sur un dataset spécifique
- Pas de validation clinique complète

## Structure du projet

```
neuroscan-project/
├── app.py                          # Application Flask avec IA (PyTorch requis)
├── app_demo.py                     # Application Flask en mode démo
├── best_brain_tumor_model.pth      # Modèle entraîné PyTorch
├── requirements.txt                # Dépendances Python complètes
├── requirements_basic.txt          # Dépendances de base
├── README.md                       # Documentation
├── start_demo.sh                   # Script de démarrage rapide
├── install_pytorch.sh              # Script d'installation PyTorch
├── create_test_image.py            # Générateur d'images de test
├── templates/
│   ├── base_dashboard.html         # Layout unifié
│   ├── dashboard.html              # Dashboard
│   ├── patients_list.html          # Liste des patients
│   ├── patient_profile_pro.html    # Nouveau profil patient (PRO)
│   └── index.html                  # Interface d'accueil
├── test_images/                    # Images de test générées
│   ├── brain_normal.jpg            # Image de cerveau normal
│   └── brain_with_tumor.jpg        # Image avec tumeur simulée
├── venv/                           # Environnement virtuel Python
└── uploads/                        # Dossier temporaire (créé automatiquement)
```

## Test de l'application

### Images de test

Le projet inclut un générateur d'images de test. Pour créer des images d'exemple :

```bash
source venv/bin/activate
python3 create_test_image.py
```

Cela créera deux images dans le dossier `test_images/` :
- `brain_normal.jpg` : Image de cerveau normal
- `brain_with_tumor.jpg` : Image avec anomalie simulée

### Test de l'interface

1. Démarrez l'application (mode démo ou complet)
2. Ouvrez http://localhost:5000 dans votre navigateur
3. Uploadez une des images de test
4. Observez les résultats de l'analyse

## Base de données
- SQLite: `neuroscan_analytics.db` (créée automatiquement)
- Tables principales: `analyses`, `patients`, `medical_alerts`, `daily_stats`, `doctors` (+ sessions)
- Migrations légères: Ajouts de colonnes conditionnels au démarrage dans `app.py`

## Développement

### Variables d'environnement

- `FLASK_ENV=development` : Mode développement
- `FLASK_DEBUG=1` : Activation du debug

### Personnalisation

Pour adapter le modèle ou l'interface :
1. Modifiez la classe `BrainTumorCNN` dans `app.py` selon votre architecture
2. Ajustez les transformations d'image si nécessaire
3. Personnalisez l'interface dans `templates/*` (notamment `patient_profile_pro.html`)

## Support

Pour toute question ou problème :
1. Vérifiez que toutes les dépendances sont installées
2. Assurez-vous que le modèle `best_brain_tumor_model.pth` est présent (sinon le mode démo s'active)
3. Consultez les logs de l'application pour les erreurs

## Licence

Ce projet est fourni à des fins éducatives. Veuillez respecter les conditions d'utilisation des datasets et modèles utilisés.
