"""
Fonctions helper MongoDB pour NeuroScan AI
Remplacement de toutes les fonctions SQLite par MongoDB
"""

from database.mongodb_connector import get_mongodb, get_collection
from datetime import datetime, timedelta, date
from bson.objectid import ObjectId
import json
from collections import Counter

def save_analysis_to_db_mongo(results, filename, processing_time, session_id=None, ip_address=None, 
                              patient_id=None, patient_name=None, exam_date=None, doctor_id=None):
    """Sauvegarder une analyse dans MongoDB"""
    try:
        print(f"DEBUG save_analysis: Starting save with patient_id={patient_id}, doctor_id={doctor_id}")
        
        db = get_mongodb()
        analyses = db.analyses
        
        # Utiliser la date actuelle si exam_date n'est pas fournie
        if exam_date is None:
            exam_date = datetime.now()
        elif isinstance(exam_date, str):
            exam_date = datetime.strptime(exam_date, '%Y-%m-%d')
        elif isinstance(exam_date, date) and not isinstance(exam_date, datetime):
            # Convertir date en datetime (MongoDB ne supporte que datetime)
            exam_date = datetime.combine(exam_date, datetime.min.time())
        
        # Vérifier que doctor_id est fourni
        if not doctor_id:
            print("Erreur: doctor_id requis pour sauvegarder l'analyse")
            return None
        
        print(f"DEBUG save_analysis: doctor_id OK, searching for previous analysis...")
        
        # Trouver l'analyse précédente pour ce patient et ce médecin
        previous_analysis_id = None
        if patient_id:
            prev_analysis = analyses.find_one(
                {'patient_id': patient_id, 'doctor_id': doctor_id},
                sort=[('exam_date', -1), ('timestamp', -1)]
            )
            if prev_analysis:
                previous_analysis_id = prev_analysis['_id']
        
        # Préparer le document d'analyse
        analysis_doc = {
            'timestamp': datetime.now(),
            'filename': filename,
            'patient_id': patient_id,
            'patient_name': patient_name,
            'exam_date': exam_date,
            # image filename stored for consistency
            'image_filename': filename,
            'predicted_class': results['predicted_class'],
            'predicted_label': results['predicted_label'],
            'confidence': results['confidence'],
            'probabilities': results['probabilities'],
            'description': results.get('description', ''),
            'recommendations': results.get('recommendations', []),
            'processing_time': processing_time,
            'user_session': session_id,
            'ip_address': ip_address,
            'tumor_size_estimate': None,  # Désactivé
            'previous_analysis_id': previous_analysis_id,
            'doctor_id': doctor_id
        }

        # Si on a un patient_id, essayer d'enrichir le document avec âge/genre depuis la fiche patient
        if patient_id:
            try:
                db = get_mongodb()
                patients = db.patients
                # patient_id peut être string non-ObjectId; on essaie de trouver par patient_id champ
                patient_doc = None
                # Première tentative: chercher par champ 'patient_id'
                patient_doc = patients.find_one({'patient_id': patient_id})
                if not patient_doc:
                    # Si patient_id ressemble à un ObjectId, tenter la recherche par _id
                    try:
                        if isinstance(patient_id, str) and len(patient_id) == 24 and all(c in '0123456789abcdefABCDEF' for c in patient_id):
                            patient_doc = patients.find_one({'_id': ObjectId(patient_id)})
                    except Exception:
                        patient_doc = None

                if patient_doc:
                    # Computation of age from date_of_birth if present
                    dob = patient_doc.get('date_of_birth')
                    computed_age = None
                    try:
                        if dob:
                            if isinstance(dob, str):
                                try:
                                    dob_dt = datetime.fromisoformat(dob)
                                except Exception:
                                    dob_dt = datetime.strptime(dob, '%Y-%m-%d')
                            elif isinstance(dob, date) and not isinstance(dob, datetime):
                                dob_dt = datetime.combine(dob, datetime.min.time())
                            else:
                                dob_dt = dob
                            today = datetime.now().date()
                            birth = dob_dt.date()
                            computed_age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
                    except Exception:
                        computed_age = None

                    # Mettre à jour le document d'analyse
                    if computed_age is not None:
                        analysis_doc['patient_age'] = computed_age
                    elif patient_doc.get('age'):
                        analysis_doc['patient_age'] = patient_doc.get('age')

                    if patient_doc.get('gender'):
                        analysis_doc['patient_gender'] = patient_doc.get('gender')
            except Exception as e:
                print(f"[WARN] enrich analysis with patient info failed: {e}")
        
        # Insérer l'analyse
        print(f"DEBUG save_analysis: Inserting analysis document...")
        result = analyses.insert_one(analysis_doc)
        analysis_id = result.inserted_id
        print(f"DEBUG save_analysis: Analysis inserted successfully! _id={analysis_id}")
        
        # Gérer le patient (créer ou mettre à jour)
        if patient_id and doctor_id:
            print(f"DEBUG save_analysis: Managing patient record...")
            manage_patient_record_mongo(patient_id, patient_name, exam_date, doctor_id)
        
        # Analyser l'évolution si il y a une analyse précédente
        if patient_id and previous_analysis_id:
            print(f"DEBUG save_analysis: Found previous analysis, creating alerts...")
            print(f"DEBUG save_analysis: previous_analysis_id={previous_analysis_id}")
            analyze_tumor_evolution_mongo(patient_id, analysis_id, previous_analysis_id, results, exam_date)
            # Créer des alertes si nécessaire
            create_medical_alerts_mongo(patient_id, analysis_id, results, doctor_id, previous_analysis_id)
        else:
            print(f"DEBUG save_analysis: No previous analysis found (patient_id={patient_id}, previous_analysis_id={previous_analysis_id})")
        
        # Mettre à jour les statistiques quotidiennes
        update_daily_stats_mongo(results, processing_time)
        
        print(f"DEBUG save_analysis: Returning analysis_id={str(analysis_id)}")
        return str(analysis_id)
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return None

def manage_patient_record_mongo(patient_id, patient_name, exam_date, doctor_id):
    """Gérer l'enregistrement du patient dans MongoDB"""
    try:
        db = get_mongodb()
        patients = db.patients
        
        # Convertir exam_date en datetime si nécessaire
        if isinstance(exam_date, str):
            exam_date = datetime.strptime(exam_date, '%Y-%m-%d')
        
        # Vérifier si le patient existe déjà
        existing_patient = patients.find_one({
            'patient_id': patient_id,
            'doctor_id': doctor_id
        })
        
        if existing_patient:
            # Mettre à jour le patient existant
            patients.update_one(
                {'patient_id': patient_id, 'doctor_id': doctor_id},
                {
                    '$set': {
                        'patient_name': patient_name,
                        'last_analysis_date': exam_date,
                        'updated_at': datetime.now()
                    },
                    '$inc': {'total_analyses': 1}
                }
            )
        else:
            # Créer un nouveau patient
            patients.insert_one({
                'patient_id': patient_id,
                'patient_name': patient_name,
                'date_of_birth': None,
                'gender': None,
                'first_analysis_date': exam_date,
                'last_analysis_date': exam_date,
                'total_analyses': 1,
                'doctor_id': doctor_id,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            })
    
    except Exception as e:
        print(f"Erreur lors de la gestion du patient MongoDB: {e}")

def analyze_tumor_evolution_mongo(patient_id, current_analysis_id, previous_analysis_id, current_results, exam_date):
    """Analyser l'évolution d'une tumeur dans MongoDB"""
    try:
        db = get_mongodb()
        analyses = db.analyses
        tumor_evolution = db.tumor_evolution
        
        # Récupérer l'analyse précédente
        prev_analysis = analyses.find_one({'_id': previous_analysis_id})
        
        if not prev_analysis:
            return
        
        prev_label = prev_analysis['predicted_label']
        prev_confidence = prev_analysis['confidence']
        prev_size = prev_analysis.get('tumor_size_estimate')
        
        # Analyser les changements
        diagnosis_change = None
        if prev_label != current_results['predicted_label']:
            diagnosis_change = f"{prev_label} → {current_results['predicted_label']}"
        
        confidence_change = current_results['confidence'] - prev_confidence
        
        size_change = None
        # Désactivé : estimation de taille
        
        # Déterminer le type d'évolution
        evolution_type = "stable"
        if diagnosis_change:
            if "Normal" in diagnosis_change:
                evolution_type = "amélioration" if current_results['predicted_label'] == "Normal" else "dégradation"
            else:
                evolution_type = "changement_type"
        elif abs(confidence_change) > 0.1:
            evolution_type = "confiance_modifiée"
        
        # Générer des notes
        notes = []
        if diagnosis_change:
            notes.append(f"Changement de diagnostic: {diagnosis_change}")
        if abs(confidence_change) > 0.05:
            direction = "augmentation" if confidence_change > 0 else "diminution"
            notes.append(f"{direction.capitalize()} de confiance: {confidence_change*100:+.1f}%")
        
        notes_text = "; ".join(notes) if notes else "Évolution stable"
        
        # Convertir exam_date en datetime si nécessaire
        if isinstance(exam_date, str):
            exam_date = datetime.strptime(exam_date, '%Y-%m-%d')
        
        # Enregistrer l'évolution
        tumor_evolution.insert_one({
            'patient_id': patient_id,
            'analysis_id': current_analysis_id,
            'exam_date': exam_date,
            'diagnosis_change': diagnosis_change,
            'confidence_change': confidence_change,
            'size_change': size_change,
            'evolution_type': evolution_type,
            'notes': notes_text,
            'created_at': datetime.now()
        })
    
    except Exception as e:
        print(f"Erreur lors de l'analyse d'évolution MongoDB: {e}")

def create_medical_alerts_mongo(patient_id, analysis_id, current_results, doctor_id, previous_analysis_id):
    """Créer des alertes médicales dans MongoDB"""
    try:
        print(f"DEBUG create_alerts: Starting alert creation...")
        print(f"DEBUG create_alerts: patient_id={patient_id}, analysis_id={analysis_id}")
        print(f"DEBUG create_alerts: current_results={current_results.get('predicted_label')}, doctor_id={doctor_id}")
        
        db = get_mongodb()
        analyses = db.analyses
        medical_alerts = db.medical_alerts
        notifications = db.notifications
        
        # Récupérer l'analyse précédente
        prev_analysis = analyses.find_one({'_id': previous_analysis_id})
        
        if not prev_analysis:
            print(f"DEBUG create_alerts: Previous analysis not found!")
            return
        
        prev_label = prev_analysis['predicted_label']
        prev_confidence = prev_analysis['confidence']
        
        print(f"DEBUG create_alerts: Previous: {prev_label} (conf={prev_confidence})")
        print(f"DEBUG create_alerts: Current: {current_results['predicted_label']} (conf={current_results['confidence']})")
        
        alerts_to_create = []
        
        # Alerte changement de diagnostic critique
        if prev_label != current_results['predicted_label']:
            if prev_label == 'Normal' and current_results['predicted_label'] != 'Normal':
                alerts_to_create.append({
                    'type': 'new_tumor_detected',
                    'severity': 'high',
                    'title': 'Nouvelle tumeur détectée',
                    'message': f'Changement de diagnostic: {prev_label} → {current_results["predicted_label"]}. Consultation urgente recommandée.'
                })
            elif prev_label != 'Normal' and current_results['predicted_label'] == 'Normal':
                alerts_to_create.append({
                    'type': 'tumor_resolved',
                    'severity': 'medium',
                    'title': 'Amélioration significative',
                    'message': f'Changement positif: {prev_label} → {current_results["predicted_label"]}. Suivi recommandé.'
                })
            else:
                alerts_to_create.append({
                    'type': 'diagnosis_change',
                    'severity': 'high',
                    'title': 'Changement de type de tumeur',
                    'message': f'Changement de diagnostic: {prev_label} → {current_results["predicted_label"]}. Réévaluation nécessaire.'
                })
        
        # Alerte baisse significative de confiance
        confidence_change = current_results['confidence'] - prev_confidence
        if confidence_change < -0.2:
            alerts_to_create.append({
                'type': 'confidence_drop',
                'severity': 'medium',
                'title': 'Baisse de confiance diagnostique',
                'message': f'Baisse significative de confiance: {confidence_change*100:+.1f}%. Analyse complémentaire recommandée.'
            })
        
        # Alerte tumeur de haut grade
        if current_results['predicted_label'] == 'Gliome' and current_results['confidence'] > 0.9:
            alerts_to_create.append({
                'type': 'high_grade_tumor',
                'severity': 'high',
                'title': 'Tumeur de haut grade suspectée',
                'message': f'Gliome détecté avec haute confiance ({current_results["confidence"]*100:.1f}%). Prise en charge oncologique urgente.'
            })
        
        # Créer les alertes
        print(f"DEBUG create_alerts: Creating {len(alerts_to_create)} alerts...")
        for alert in alerts_to_create:
            print(f"DEBUG create_alerts: - {alert['title']} ({alert['severity']})")
            # Insérer l'alerte
            medical_alerts.insert_one({
                'patient_id': patient_id,
                'doctor_id': doctor_id,
                'analysis_id': analysis_id,
                'alert_type': alert['type'],
                'severity': alert['severity'],
                'title': alert['title'],
                'message': alert['message'],
                'is_read': False,
                'is_resolved': False,
                'created_at': datetime.now(),
                'resolved_at': None,
                'resolved_by': None
            })
            
            # Créer une notification
            notifications.insert_one({
                'doctor_id': doctor_id,
                'type': 'medical_alert',
                'title': alert['title'],
                'message': alert['message'],
                'data': json.dumps({
                    'patient_id': patient_id,
                    'analysis_id': str(analysis_id)
                }),
                'is_read': False,
                'created_at': datetime.now()
            })
    
    except Exception as e:
        print(f"Erreur lors de la création des alertes MongoDB: {e}")

def update_daily_stats_mongo(results, processing_time):
    """Mettre à jour les statistiques quotidiennes dans MongoDB"""
    try:
        db = get_mongodb()
        daily_stats = db.daily_stats
        
        today = datetime.now().date()
        today_datetime = datetime.combine(today, datetime.min.time())
        
        # Déterminer le compteur à incrémenter
        label_field = {
            'Normal': 'normal_count',
            'Gliome': 'gliome_count',
            'Méningiome': 'meningiome_count',
            'Tumeur pituitaire': 'pituitary_count'
        }.get(results['predicted_label'], 'normal_count')
        
        # Récupérer les stats actuelles
        current_stats = daily_stats.find_one({'date': today_datetime})
        
        if current_stats:
            # Calculer les nouvelles moyennes
            total = current_stats['total_analyses']
            new_total = total + 1
            new_avg_confidence = ((current_stats['avg_confidence'] * total) + results['confidence']) / new_total
            new_avg_processing_time = ((current_stats['avg_processing_time'] * total) + processing_time) / new_total
            
            # Mettre à jour
            daily_stats.update_one(
                {'date': today_datetime},
                {
                    '$inc': {
                        'total_analyses': 1,
                        label_field: 1
                    },
                    '$set': {
                        'avg_confidence': new_avg_confidence,
                        'avg_processing_time': new_avg_processing_time
                    }
                }
            )
        else:
            # Créer une nouvelle entrée
            new_stats = {
                'date': today_datetime,
                'total_analyses': 1,
                'normal_count': 1 if label_field == 'normal_count' else 0,
                'gliome_count': 1 if label_field == 'gliome_count' else 0,
                'meningiome_count': 1 if label_field == 'meningiome_count' else 0,
                'pituitary_count': 1 if label_field == 'pituitary_count' else 0,
                'avg_confidence': results['confidence'],
                'avg_processing_time': processing_time
            }
            daily_stats.insert_one(new_stats)
    
    except Exception as e:
        print(f"Erreur lors de la mise à jour des statistiques MongoDB: {e}")

def get_patient_alerts_mongo(patient_id, doctor_id, limit=None):
    """Récupérer les alertes d'un patient depuis MongoDB"""
    try:
        db = get_mongodb()
        medical_alerts = db.medical_alerts
        
        query = {'patient_id': patient_id, 'doctor_id': doctor_id}
        cursor = medical_alerts.find(query).sort('created_at', -1)
        
        if limit:
            cursor = cursor.limit(limit)
        
        alerts = []
        for alert in cursor:
            alerts.append({
                'id': str(alert['_id']),
                'alert_type': alert['alert_type'],
                'severity': alert['severity'],
                'title': alert['title'],
                'message': alert['message'],
                'is_read': alert['is_read'],
                'is_resolved': alert['is_resolved'],
                'created_at': alert['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return alerts
    
    except Exception as e:
        print(f"Erreur lors de la récupération des alertes MongoDB: {e}")
        return []

def get_doctor_statistics_mongo(doctor_id):
    """Calculer les statistiques d'un médecin depuis MongoDB"""
    try:
        db = get_mongodb()
        patients = db.patients
        analyses = db.analyses
        chat_conversations = db.chat_conversations
        doctors = db.doctors
        
        # Convertir doctor_id en ObjectId si nécessaire pour la requête doctors
        doctor_object_id = ObjectId(doctor_id) if isinstance(doctor_id, str) else doctor_id
        
        # Nombre total de patients (doctor_id est stocké comme string ou int dans patients)
        total_patients = patients.count_documents({'doctor_id': doctor_id})
        
        # Nombre total d'analyses
        pipeline = [
            {'$match': {'doctor_id': doctor_id}},
            {'$group': {'_id': None, 'total': {'$sum': '$total_analyses'}}}
        ]
        result = list(patients.aggregate(pipeline))
        total_analyses = result[0]['total'] if result else 0
        
        # Nombre total de conversations de chat
        total_conversations = chat_conversations.count_documents({
            'doctor_id': doctor_id,
            'is_active': True
        })
        
        # Nombre de jours depuis la création du compte
        doctor = doctors.find_one({'_id': doctor_object_id})
        if doctor and 'created_at' in doctor:
            account_age = (datetime.now() - doctor['created_at']).days
        else:
            account_age = 0
        
        # Analyses récentes (derniers 30 jours)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_analyses = analyses.count_documents({
            'doctor_id': doctor_id,
            'timestamp': {'$gte': thirty_days_ago}
        })
        
        # Patient le plus récent
        latest_patient = patients.find_one(
            {'doctor_id': doctor_id},
            sort=[('created_at', -1)]
        )
        
        return {
            'total_patients': total_patients,
            'total_analyses': total_analyses,
            'total_conversations': total_conversations,
            'account_age': account_age,
            'recent_analyses': recent_analyses,
            'latest_patient_name': latest_patient['patient_name'] if latest_patient else None,
            'latest_patient_date': latest_patient['created_at'].strftime('%Y-%m-%d %H:%M:%S') if latest_patient else None
        }
    
    except Exception as e:
        print(f"Erreur calcul statistiques médecin MongoDB: {e}")
        return {
            'total_patients': 0,
            'total_analyses': 0,
            'total_conversations': 0,
            'account_age': 0,
            'recent_analyses': 0,
            'latest_patient_name': None,
            'latest_patient_date': None
        }

def get_current_doctor_mongo(doctor_id):
    """Obtenir les informations du médecin depuis MongoDB"""
    try:
        db = get_mongodb()
        doctors = db.doctors
        
        # Convertir doctor_id en ObjectId si c'est une string
        if isinstance(doctor_id, str):
            doctor_id = ObjectId(doctor_id)
        
        doctor = doctors.find_one({
            '_id': doctor_id,
            'is_active': True
        })
        
        if doctor:
            return {
                'id': str(doctor['_id']),
                'email': doctor['email'],
                'first_name': doctor['first_name'],
                'last_name': doctor['last_name'],
                'specialty': doctor.get('specialty'),
                'hospital': doctor.get('hospital'),
                'license_number': doctor.get('license_number'),
                'phone': doctor.get('phone'),
                'created_at': doctor.get('created_at'),
                'full_name': f"{doctor['first_name']} {doctor['last_name']}"
            }
        
        return None
    
    except Exception as e:
        print(f"Erreur lors de la récupération du médecin MongoDB: {e}")
        return None

def create_doctor_session_mongo(doctor_id, ip_address, user_agent):
    """Créer une session pour un médecin dans MongoDB"""
    try:
        import secrets
        
        db = get_mongodb()
        doctor_sessions = db.doctor_sessions
        
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)
        
        # Désactiver les anciennes sessions (doctor_id peut être string ou ObjectId)
        doctor_sessions.update_many(
            {'doctor_id': doctor_id},
            {'$set': {'is_active': False}}
        )
        
        # Créer la nouvelle session
        doctor_sessions.insert_one({
            'doctor_id': doctor_id,
            'session_token': session_token,
            'created_at': datetime.now(),
            'expires_at': expires_at,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'is_active': True
        })
        
        return session_token
    
    except Exception as e:
        print(f"Erreur lors de la création de session MongoDB: {e}")
        return None

def register_doctor_mongo(email, password_hash, first_name, last_name, specialty=None, 
                         hospital=None, license_number=None, phone=None):
    """
    Inscrire un nouveau médecin dans MongoDB
    
    Args:
        email: Email du médecin (sera converti en minuscules)
        password_hash: Hash du mot de passe (utiliser generate_password_hash)
        first_name: Prénom
        last_name: Nom
        specialty: Spécialité médicale (optionnel)
        hospital: Hôpital (optionnel)
        license_number: Numéro de licence (optionnel)
        phone: Téléphone (optionnel)
    
    Returns:
        str: ID du médecin créé ou None en cas d'erreur
    """
    try:
        db = get_mongodb()
        doctors = db.doctors
        
        # Normaliser l'email
        email = email.strip().lower()
        
        # Vérifier si l'email existe déjà
        existing_doctor = doctors.find_one({'email': email})
        if existing_doctor:
            return None  # Email déjà utilisé
        
        # Créer le document du médecin
        doctor_doc = {
            'email': email,
            'password_hash': password_hash,
            'first_name': first_name.strip(),
            'last_name': last_name.strip(),
            'specialty': specialty.strip() if specialty else None,
            'hospital': hospital.strip() if hospital else None,
            'license_number': license_number.strip() if license_number else None,
            'phone': phone.strip() if phone else None,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'is_active': True,
            'last_login': None,
            'login_count': 0
        }
        
        # Insérer le médecin
        result = doctors.insert_one(doctor_doc)
        
        return str(result.inserted_id)
    
    except Exception as e:
        print(f"Erreur lors de l'inscription MongoDB: {e}")
        return None

def verify_doctor_credentials_mongo(email, password_check_func):
    """
    Vérifier les identifiants d'un médecin et renvoyer ses informations
    
    Args:
        email: Email du médecin
        password_check_func: Fonction pour vérifier le mot de passe (ex: check_password_hash(hash, password))
    
    Returns:
        dict: Informations du médecin si authentification réussie, None sinon
    """
    try:
        db = get_mongodb()
        doctors = db.doctors
        
        # Normaliser l'email
        email = email.strip().lower()
        
        # Chercher le médecin
        doctor = doctors.find_one({'email': email})
        
        if not doctor:
            return None
        
        # Vérifier que le compte est actif
        if not doctor.get('is_active', True):
            return {'error': 'account_disabled'}
        
        # Vérifier le mot de passe
        if not password_check_func(doctor['password_hash']):
            return None
        
        # Mettre à jour les statistiques de connexion
        doctors.update_one(
            {'_id': doctor['_id']},
            {
                '$set': {'last_login': datetime.now()},
                '$inc': {'login_count': 1}
            }
        )
        
        # Retourner les informations du médecin
        return {
            'id': str(doctor['_id']),
            'email': doctor['email'],
            'first_name': doctor['first_name'],
            'last_name': doctor['last_name'],
            'specialty': doctor.get('specialty'),
            'hospital': doctor.get('hospital'),
            'license_number': doctor.get('license_number'),
            'phone': doctor.get('phone'),
            'created_at': doctor.get('created_at'),
            'last_login': doctor.get('last_login'),
            'login_count': doctor.get('login_count', 0),
            'full_name': f"{doctor['first_name']} {doctor['last_name']}"
        }
    
    except Exception as e:
        print(f"Erreur lors de la vérification des identifiants MongoDB: {e}")
        return None
