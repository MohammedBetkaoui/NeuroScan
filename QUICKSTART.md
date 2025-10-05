# 🚀 Guide de Démarrage Rapide - NeuroScan AI

## Lancement Immédiat

```bash
# Méthode la plus simple
./launch_neuroscan.sh
```

## Toutes les Commandes Disponibles

### 📱 Utilisation

```bash
# Lancer l'application (recommandé)
./launch_neuroscan.sh

# Lancer directement l'exécutable
./dist/NeuroScan_AI

# Afficher le résumé
./show_summary.sh

# Vérifier l'installation
./verify_installation.sh
```

### 🔧 Développement

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Tester en mode développement
python3 app.py

# Tester run_app.py (PyWebView)
python3 run_app.py
```

### 📦 Compilation

```bash
# Recompiler l'application
./build_app.sh

# Nettoyer les fichiers de build
rm -rf build/ dist/ *.spec
```

### 💿 Installation Système

```bash
# Installer dans /opt/ (nécessite sudo)
sudo ./install_system.sh

# Désinstaller
sudo ./uninstall_system.sh

# Après installation, lancer avec
neuroscan-ai
```

### 📤 Distribution

```bash
# Créer un package complet
tar -czf NeuroScan_AI_v1.0.tar.gz \
  dist/NeuroScan_AI \
  launch_neuroscan.sh \
  DISTRIBUTION.md

# Créer un package minimal (exécutable seul)
tar -czf NeuroScan_AI_v1.0_minimal.tar.gz \
  dist/NeuroScan_AI

# Vérifier le package
tar -tzf NeuroScan_AI_v1.0.tar.gz
```

### 🔍 Débogage

```bash
# Lancer avec logs
./dist/NeuroScan_AI 2>&1 | tee neuroscan.log

# Vérifier les dépendances système
dpkg -l | grep -E 'python3-gi|webkit2|gtk-3'

# Vérifier la taille de l'exécutable
ls -lh dist/NeuroScan_AI

# Tester le modèle PyTorch
python3 -c "import torch; print(torch.__version__)"

# Tester PyWebView
python3 -c "import webview; print(webview.__version__)"
```

### 🗄️ Base de Données

```bash
# Backup de la base de données
cp neuroscan_analytics.db neuroscan_analytics.db.backup

# Restaurer une sauvegarde
cp neuroscan_analytics.db.backup neuroscan_analytics.db

# Inspecter la base
sqlite3 neuroscan_analytics.db ".tables"
```

### 🔐 Configuration

```bash
# Éditer les variables d'environnement
nano .env

# Vérifier la configuration
cat .env

# Exemple de .env
# GEMINI_API_KEY=votre_clé_ici
# FLASK_SECRET_KEY=votre_secret_ici
```

### 📊 Monitoring

```bash
# Surveiller les ressources
htop

# Voir les processus Flask
ps aux | grep python

# Voir les connexions réseau
netstat -tulpn | grep 5000
```

## 🎯 Cas d'Usage Courants

### Premier lancement
```bash
./verify_installation.sh
./launch_neuroscan.sh
```

### Après modification du code
```bash
./build_app.sh
./launch_neuroscan.sh
```

### Préparation pour distribution
```bash
./verify_installation.sh
tar -czf NeuroScan_AI_v1.0.tar.gz dist/NeuroScan_AI launch_neuroscan.sh DISTRIBUTION.md
```

### Installation pour un utilisateur final
```bash
# Extraire le package
tar -xzf NeuroScan_AI_v1.0.tar.gz

# Installer les dépendances système (une fois)
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1

# Lancer
./launch_neuroscan.sh
```

## 🆘 Dépannage Rapide

### Problème : "Permission denied"
```bash
chmod +x dist/NeuroScan_AI
chmod +x launch_neuroscan.sh
```

### Problème : "Cannot find webview"
```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1
```

### Problème : "Port 5000 already in use"
```bash
# Tuer le processus qui utilise le port
sudo lsof -ti:5000 | xargs kill -9
```

### Problème : L'application ne démarre pas
```bash
# Vérifier l'installation
./verify_installation.sh

# Recompiler
./build_app.sh

# Tester en mode développement
python3 run_app.py
```

## 📖 Documentation Complète

- **TRANSFORMATION_COMPLETE.md** : Guide complet
- **DISTRIBUTION.md** : Guide de distribution
- **README.md** : Vue d'ensemble du projet

## ✨ Félicitations !

Votre application est prête à l'emploi ! 🎉

Pour démarrer maintenant :
```bash
./launch_neuroscan.sh
```
