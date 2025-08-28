# NeuroScan - Analyse de Tumeurs Cérébrales

Application Flask utilisant l'intelligence artificielle pour l'analyse et la détection de tumeurs cérébrales à partir d'images IRM.

## Fonctionnalités

- **Interface web moderne** : Interface utilisateur intuitive avec design responsive
- **Analyse IA** : Utilisation d'un modèle CNN entraîné pour la classification de tumeurs cérébrales
- **Support multi-formats** : Compatible avec les formats DICOM, NIfTI, JPEG, PNG
- **Résultats détaillés** : Probabilités pour chaque type de tumeur et recommandations cliniques
- **Visualisation** : Affichage des zones suspectes sur l'image analysée

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

### Utilisation de l'interface

1. **Upload d'image** : 
   - Glissez-déposez une image IRM ou cliquez sur "Sélectionner un fichier"
   - Formats supportés : .dcm, .nii, .jpg, .png

2. **Analyse** :
   - Cliquez sur "Lancer l'analyse"
   - Attendez que l'analyse se termine (quelques secondes)

3. **Résultats** :
   - Visualisez le diagnostic principal
   - Consultez les probabilités pour chaque type de tumeur
   - Lisez les recommandations cliniques

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
│   └── index.html                  # Interface utilisateur
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

## Développement

### Variables d'environnement

- `FLASK_ENV=development` : Mode développement
- `FLASK_DEBUG=1` : Activation du debug

### Personnalisation

Pour adapter le modèle ou l'interface :
1. Modifiez la classe `BrainTumorCNN` dans `app.py` selon votre architecture
2. Ajustez les transformations d'image si nécessaire
3. Personnalisez l'interface dans `templates/index.html`

## Support

Pour toute question ou problème :
1. Vérifiez que toutes les dépendances sont installées
2. Assurez-vous que le modèle `best_brain_tumor_model.pth` est présent
3. Consultez les logs de l'application pour les erreurs

## Licence

Ce projet est fourni à des fins éducatives. Veuillez respecter les conditions d'utilisation des datasets et modèles utilisés.
