# NeuroScan - Projet Flask d'Analyse de Tumeurs Cérébrales

## 🎯 Objectif du projet

Créer une application web Flask qui utilise l'interface HTML fournie et le modèle d'IA `best_brain_tumor_model.pth` pour analyser des images IRM et détecter des tumeurs cérébrales.

## ✅ Réalisations

### 1. Application Flask complète
- **`app.py`** : Application principale avec intégration PyTorch
- **`app_demo.py`** : Version démo avec prédictions simulées
- Support des formats d'images : DICOM, NIfTI, JPEG, PNG
- API REST pour l'upload et l'analyse d'images

### 2. Interface utilisateur moderne
- **`templates/index.html`** : Interface web responsive basée sur le fichier fourni
- Design moderne avec Tailwind CSS
- Fonctionnalités drag & drop pour l'upload
- Affichage des résultats avec visualisations
- Recommandations cliniques automatiques

### 3. Modèle d'IA intégré
- Architecture CNN personnalisée compatible avec `best_brain_tumor_model.pth`
- 4 classes de classification :
  - Normal (aucune anomalie)
  - Gliome
  - Méningiome
  - Tumeur pituitaire
- Préprocessing automatique des images
- Calcul des probabilités et confiance

### 4. Scripts d'automatisation
- **`start_demo.sh`** : Démarrage rapide en mode démo
- **`install_pytorch.sh`** : Installation automatique de PyTorch
- **`create_test_image.py`** : Générateur d'images de test

### 5. Documentation complète
- **`README.md`** : Guide d'installation et d'utilisation détaillé
- **`PROJET_RESUME.md`** : Ce résumé du projet
- Instructions pour les deux modes (démo et complet)

## 🚀 Démarrage rapide

### Mode démo (recommandé pour tester)
```bash
chmod +x start_demo.sh
./start_demo.sh
```

### Mode complet avec IA
```bash
./install_pytorch.sh
source venv/bin/activate
python3 app.py
```

## 🧪 Test de l'application

1. **Créer des images de test :**
   ```bash
   source venv/bin/activate
   python3 create_test_image.py
   ```

2. **Accéder à l'interface :** http://localhost:5000

3. **Tester l'upload :** Utiliser les images dans `test_images/`

## 📁 Structure des fichiers

```
├── app.py                      # Application Flask avec IA
├── app_demo.py                 # Application Flask démo
├── best_brain_tumor_model.pth  # Modèle PyTorch fourni
├── templates/index.html        # Interface web (basée sur le fichier fourni)
├── start_demo.sh              # Script de démarrage rapide
├── install_pytorch.sh         # Installation PyTorch
├── create_test_image.py       # Générateur d'images test
├── test_images/               # Images de test générées
├── requirements*.txt          # Dépendances Python
└── README.md                  # Documentation complète
```

## 🔧 Fonctionnalités techniques

### Backend Flask
- Upload sécurisé de fichiers (16MB max)
- Validation des formats d'images
- Préprocessing automatique (redimensionnement, normalisation)
- API REST avec réponses JSON
- Gestion d'erreurs robuste

### Frontend
- Interface responsive (mobile-friendly)
- Drag & drop pour l'upload
- Barre de progression animée
- Affichage des résultats avec graphiques
- Recommandations cliniques contextuelles

### IA et Machine Learning
- Modèle CNN avec 5 couches de convolution
- Preprocessing compatible avec le modèle fourni
- Calcul des probabilités avec softmax
- Support GPU/CPU automatique

## 🎨 Interface utilisateur

L'interface reprend fidèlement le design du fichier HTML fourni avec :
- Header avec navigation
- Section d'upload avec drag & drop
- Carte de progression de l'analyse
- Affichage des résultats avec :
  - Image analysée avec zones surlignées
  - Probabilités pour chaque type de tumeur
  - Recommandations cliniques
  - Boutons d'export et de partage

## 🔒 Sécurité et limitations

### Sécurité
- Validation stricte des types de fichiers
- Noms de fichiers sécurisés
- Nettoyage automatique des fichiers temporaires
- Limitation de taille des uploads

### Limitations importantes
⚠️ **ATTENTION** : Cette application est destinée à des fins éducatives et de recherche uniquement. Elle ne doit pas être utilisée pour des diagnostics médicaux réels sans validation par des professionnels de santé qualifiés.

## 🛠️ Technologies utilisées

- **Backend :** Flask, PyTorch, NumPy, OpenCV
- **Frontend :** HTML5, CSS3 (Tailwind), JavaScript
- **IA :** PyTorch, torchvision, PIL
- **Outils :** Bash scripts, Python venv

## 📊 Modes de fonctionnement

### Mode démo (`app_demo.py`)
- Prédictions simulées aléatoirement
- Pas besoin de PyTorch
- Installation rapide
- Idéal pour tester l'interface

### Mode complet (`app.py`)
- Utilise le vrai modèle `best_brain_tumor_model.pth`
- Nécessite PyTorch
- Prédictions réelles basées sur l'IA
- Installation plus longue

## 🎯 Objectifs atteints

✅ Intégration complète de l'interface HTML fournie  
✅ Utilisation du modèle `best_brain_tumor_model.pth`  
✅ Application Flask fonctionnelle  
✅ Support des formats médicaux (DICOM, NIfTI)  
✅ Interface moderne et responsive  
✅ Scripts d'automatisation  
✅ Documentation complète  
✅ Images de test générées  
✅ Mode démo pour tests rapides  
✅ Gestion d'erreurs robuste  

## 🚀 Prochaines étapes possibles

- Intégration de vrais datasets médicaux
- Amélioration de l'architecture du modèle
- Ajout de fonctionnalités d'export (PDF, DICOM)
- Interface d'administration
- Base de données pour historique
- Authentification utilisateur
- Déploiement en production

---

**Projet créé avec succès !** 🎉  
L'application NeuroScan est maintenant prête à être utilisée pour l'analyse de tumeurs cérébrales.
