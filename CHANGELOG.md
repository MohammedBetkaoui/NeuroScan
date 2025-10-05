# 📝 Changelog - NeuroScan AI

Tous les changements notables de ce projet seront documentés dans ce fichier.

## [Desktop v1.0.0] - 2025-10-05

### 🎉 Transformation en Application Desktop

#### ✨ Ajouté
- **Application Desktop Native**
  - Fenêtre PyWebView native (GTK/WebKit)
  - Taille : 1200x800 pixels (redimensionnable, min 800x600)
  - Intégration complète avec le bureau Linux

- **Exécutable Autonome**
  - Compilation avec PyInstaller
  - Exécutable unique de 909 Mo
  - Toutes les dépendances embarquées (Flask, PyTorch, etc.)
  - Mode standalone (pas besoin de Python installé)

- **Scripts Utilitaires**
  - `run_app.py` : Point d'entrée avec PyWebView + Flask en thread
  - `build_app.sh` : Script de compilation automatique
  - `launch_neuroscan.sh` : Script de lancement simple
  - `install_system.sh` : Installation système dans /opt/
  - `uninstall_system.sh` : Désinstallation propre
  - `verify_installation.sh` : Vérification complète de l'installation
  - `show_summary.sh` : Affichage du résumé de transformation

- **Intégration Système**
  - Fichier `.desktop` pour menu des applications
  - Lien symbolique `/usr/local/bin/neuroscan-ai`
  - Support des catégories d'applications (Medical, Science)

- **Documentation Complète**
  - `TRANSFORMATION_COMPLETE.md` : Guide complet (8.5 KB)
  - `DISTRIBUTION.md` : Guide de distribution (4.2 KB)
  - `QUICKSTART.md` : Guide de démarrage rapide
  - `CHANGELOG.md` : Ce fichier

#### 🔧 Modifié
- **requirements.txt** : Ajout de PyWebView, PyInstaller, PyGObject
- **Architecture** : Flask s'exécute maintenant en thread daemon
- **Configuration** : Mode production (debug=False) pour l'exécutable

#### 📦 Fichiers Embarqués
L'exécutable contient maintenant :
- Runtime Python complet
- Flask + toutes ses extensions
- PyTorch + TorchVision (~600 Mo)
- Modèle IA : `best_brain_tumor_model.pth`
- Base de données : `neuroscan_analytics.db`
- Templates HTML complets
- Assets statiques (CSS, JS, images)
- Configuration `.env`

#### 🎯 Fonctionnalités Préservées
- ✅ Analyse d'IRM cérébrales avec PyTorch
- ✅ Chat IA avec Google Gemini
- ✅ Gestion complète des patients
- ✅ Historique des analyses
- ✅ Génération de rapports PDF
- ✅ Authentification sécurisée
- ✅ Dashboard statistiques

#### 🚀 Performance
- Démarrage : ~3 secondes
- Utilisation RAM : 500-800 Mo
- Taille exécutable : 909 Mo
- Interface fluide et responsive

#### 🔐 Sécurité
- Flask en mode production
- Sessions sécurisées
- Hachage des mots de passe
- Variables d'environnement protégées

#### 📊 Technologies Utilisées
- **Frontend** : HTML5, CSS3, JavaScript
- **Backend** : Flask 2.0+, Python 3.12
- **Desktop** : PyWebView 4.0+, GTK 3.0, WebKit2
- **IA** : PyTorch 2.0+, Google Gemini API
- **Base de données** : SQLite 3
- **Packaging** : PyInstaller 6.0+

#### 🐛 Corrections
- Fix : Threading pour éviter le blocage de l'interface
- Fix : Gestion propre de la fermeture de l'application
- Fix : Chemins relatifs pour les ressources embarquées

#### 🗑️ Supprimé
- Dépendance : Serveur web externe (maintenant intégré)
- Dépendance : Installation Python requise pour utilisateurs finaux

---

## [Web v0.9.0] - 2025-10-04

### Application Flask Web Initiale

#### ✨ Fonctionnalités
- Application web Flask classique
- Analyse d'IRM cérébrales
- Chat IA Gemini
- Gestion des patients
- Authentification
- Dashboard

#### 🔧 Technologies
- Flask
- PyTorch
- SQLite
- Google Gemini API

---

## Notes de Version

### Version Desktop vs Web

| Caractéristique | Web v0.9 | Desktop v1.0 |
|----------------|----------|--------------|
| Type | Application web | Application native |
| Installation Python | Requise | Non requise |
| Démarrage | `python3 app.py` | `./launch_neuroscan.sh` |
| Distribution | Code source | Exécutable unique |
| Taille | ~50 Mo (code) | 909 Mo (tout inclus) |
| Dépendances | Multiples | Aucune (embarquées) |
| Interface | Navigateur web | Fenêtre native |
| Déploiement | Serveur requis | Standalone |

### Migration Web → Desktop

Pour les utilisateurs existants :

1. **Données préservées** : La base de données SQLite est compatible
2. **Configuration** : Le fichier `.env` reste inchangé
3. **Fonctionnalités** : 100% identiques
4. **Interface** : Identique (même HTML/CSS/JS)

### Prochaines Étapes (Roadmap)

#### v1.1.0 (Prévu)
- [ ] Icône personnalisée haute résolution
- [ ] Système de mise à jour automatique
- [ ] Mode sombre/clair
- [ ] Page "À propos" avec version

#### v1.2.0 (Prévu)
- [ ] Support multi-langues (Français, Anglais, Arabe)
- [ ] Export CSV/Excel des données
- [ ] Notifications desktop
- [ ] Graphiques interactifs avancés

#### v2.0.0 (Futur)
- [ ] Version Windows (PyInstaller cross-platform)
- [ ] Version macOS
- [ ] Mode serveur optionnel (multi-utilisateurs)
- [ ] API REST complète
- [ ] Application mobile (React Native)

---

**Légende** :
- ✨ Ajouté : Nouvelles fonctionnalités
- 🔧 Modifié : Changements de fonctionnalités existantes
- 🐛 Corrigé : Corrections de bugs
- 🗑️ Supprimé : Fonctionnalités retirées
- 🔐 Sécurité : Correctifs de sécurité
- 📚 Documentation : Changements de documentation

---

*Pour toute question sur les versions, consultez la documentation complète.*
