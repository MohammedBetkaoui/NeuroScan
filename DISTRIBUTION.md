# 🧠 NeuroScan AI - Application Desktop Linux

## 📦 Distribution

Votre application Flask a été transformée en une application desktop Linux autonome !

### ✅ Fichiers créés

- **`dist/NeuroScan_AI`** : L'exécutable principal (909 Mo)
- **`launch_neuroscan.sh`** : Script de lancement pratique
- **`build_app.sh`** : Script de compilation (pour recompiler si nécessaire)
- **`run_app.py`** : Source Python utilisant PyWebView

### 🚀 Lancement de l'application

#### Méthode 1 : Via le script de lancement
```bash
./launch_neuroscan.sh
```

#### Méthode 2 : Directement
```bash
./dist/NeuroScan_AI
```

### 📋 Caractéristiques

- **Fenêtre native** : 1200x800 pixels (redimensionnable)
- **Taille minimale** : 800x600 pixels
- **Mode standalone** : Toutes les dépendances incluses
- **Base de données** : neuroscan_analytics.db incluse
- **Modèle IA** : best_brain_tumor_model.pth inclus

### 🔧 Recompilation

Si vous modifiez le code source, recompilez avec :
```bash
./build_app.sh
```

Cette commande va :
1. Activer l'environnement virtuel
2. Lancer PyInstaller avec tous les fichiers nécessaires
3. Créer un nouvel exécutable dans `dist/`

### 📦 Dépendances incluses

L'exécutable contient :
- Flask + toutes ses dépendances
- PyTorch + TorchVision
- PyWebView
- Pillow (PIL)
- Google Generative AI (Gemini)
- ReportLab
- SQLite3
- Toutes les bibliothèques Python nécessaires

### 🖥️ Configuration système requise

**Dépendances système Linux** (pour l'exécution) :
```bash
sudo apt-get install \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    gir1.2-webkit2-4.1
```

**Ces paquets sont nécessaires pour PyWebView sur Linux.**

### 📁 Structure des fichiers embarqués

```
NeuroScan_AI (exécutable)
├── templates/          # Templates HTML Flask
├── static/            # CSS, JS, images
├── best_brain_tumor_model.pth  # Modèle PyTorch
├── neuroscan_analytics.db      # Base de données
└── .env               # Configuration (variables d'environnement)
```

### 🔐 Sécurité

- Le fichier `.env` est inclus dans l'exécutable
- **Important** : Ne distribuez pas publiquement l'exécutable si `.env` contient des clés API sensibles
- Pour la distribution publique, créez un mécanisme de configuration externe

### 🐛 Dépannage

#### L'application ne se lance pas
```bash
# Vérifier les permissions
chmod +x dist/NeuroScan_AI

# Vérifier les dépendances système
sudo apt-get install gir1.2-webkit2-4.1
```

#### Erreur "Cannot find templates"
- Les templates sont normalement embarqués
- Vérifiez que la compilation s'est bien passée
- Recompilez avec `./build_app.sh`

#### Erreur de base de données
- La base de données est copiée dans `dist/`
- Vérifiez les permissions en écriture sur le dossier `dist/`

### 📊 Taille de l'exécutable

L'exécutable fait ~909 Mo car il contient :
- PyTorch (~600 Mo)
- TorchVision
- Toutes les dépendances Python
- Vos templates, static, et modèles

C'est normal pour une application avec Deep Learning embarqué !

### 🎯 Utilisation en production

Pour distribuer l'application :

1. **Créez un package** :
```bash
mkdir -p NeuroScan_AI_Release
cp dist/NeuroScan_AI NeuroScan_AI_Release/
cp launch_neuroscan.sh NeuroScan_AI_Release/
cp DISTRIBUTION.md NeuroScan_AI_Release/README.md
```

2. **Archivez** :
```bash
tar -czf NeuroScan_AI_v1.0_Linux_x64.tar.gz NeuroScan_AI_Release/
```

3. **Distribuez** le fichier .tar.gz

### 🔄 Mise à jour

Pour mettre à jour l'application :
1. Modifiez le code source
2. Recompilez avec `./build_app.sh`
3. L'exécutable dans `dist/` sera mis à jour

### 📝 Notes

- L'application utilise Flask en mode production (debug=False)
- Le serveur Flask tourne en local sur 127.0.0.1:5000
- PyWebView crée une fenêtre native qui affiche l'interface web
- L'application est 100% autonome (pas besoin de Python installé)

### 🎉 Succès !

Votre application Flask est maintenant une application desktop Linux complète et autonome !

Pour toute question ou problème, consultez la documentation de :
- PyWebView : https://pywebview.flowrl.com/
- PyInstaller : https://pyinstaller.org/
