from flask import Flask, render_template, request, jsonify
import base64
import os
from werkzeug.utils import secure_filename
import random
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Classes de tumeurs (simulation)
TUMOR_CLASSES = {
    0: 'Normal',
    1: 'Gliome',
    2: 'Méningiome', 
    3: 'Tumeur pituitaire'
}

def allowed_file(filename):
    """Vérifier si le fichier est autorisé"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'nii'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def simulate_prediction():
    """Simuler une prédiction de tumeur (pour la démo)"""
    # Simulation aléatoire pour la démo
    predicted_class = random.randint(0, 3)
    
    # Générer des probabilités aléatoires qui somment à 1
    probs = [random.random() for _ in range(4)]
    total = sum(probs)
    probs = [p/total for p in probs]
    
    # Ajuster pour que la classe prédite ait la plus haute probabilité
    max_prob_idx = probs.index(max(probs))
    if max_prob_idx != predicted_class:
        probs[predicted_class], probs[max_prob_idx] = probs[max_prob_idx], probs[predicted_class]
    
    results = {
        'predicted_class': predicted_class,
        'predicted_label': TUMOR_CLASSES[predicted_class],
        'confidence': probs[predicted_class],
        'probabilities': {
            'Normal': probs[0],
            'Gliome': probs[1],
            'Méningiome': probs[2],
            'Tumeur pituitaire': probs[3]
        }
    }
    
    return results

def get_recommendations(results):
    """Générer des recommandations basées sur les résultats"""
    recommendations = []
    
    if results['predicted_class'] == 0:  # Normal
        recommendations = [
            "Aucune anomalie détectée dans cette analyse",
            "Suivi de routine recommandé selon les protocoles standards",
            "Consultation avec un radiologue pour confirmation"
        ]
    else:  # Tumeur détectée
        recommendations = [
            "Biopsie recommandée pour confirmation histologique",
            "IRM de suivi dans 3 mois pour évaluation de la croissance", 
            "Consultation avec un neuro-oncologue spécialisé"
        ]
        
        if results['confidence'] < 0.7:
            recommendations.append("Analyse complémentaire recommandée en raison de la faible confiance")
    
    return recommendations

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Gérer l'upload et l'analyse d'image (version démo)"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Sauvegarder le fichier
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Simuler un délai de traitement
            time.sleep(2)
            
            # Faire une prédiction simulée
            results = simulate_prediction()
            
            # Convertir l'image en base64 pour l'affichage
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
                img_url = f"data:image/jpeg;base64,{img_data}"
            
            # Nettoyer le fichier temporaire
            os.remove(filepath)
            
            # Préparer la réponse
            response = {
                'success': True,
                'image_url': img_url,
                'prediction': results['predicted_label'],
                'confidence': results['confidence'],
                'probabilities': results['probabilities'],
                'is_tumor': results['predicted_class'] != 0,  # 0 = Normal
                'recommendations': get_recommendations(results)
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Erreur lors du traitement: {e}")
            return jsonify({'error': 'Erreur lors du traitement de l\'image'}), 500
    
    return jsonify({'error': 'Type de fichier non autorisé'}), 400

@app.route('/health')
def health_check():
    """Vérification de l'état de l'application"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': 'demo_mode',
        'note': 'Application en mode démo - prédictions simulées'
    })

if __name__ == '__main__':
    print("Démarrage de l'application en mode DÉMO")
    print("Les prédictions sont simulées à des fins de démonstration")
    print("Pour utiliser un vrai modèle, utilisez app.py avec PyTorch installé")
    app.run(debug=True, host='0.0.0.0', port=5000)
