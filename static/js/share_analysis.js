/**
 * NeuroScan AI - Partage d'Analyses entre Médecins
 * Module pour partager des analyses depuis n'importe quelle page
 */

// Variables globales
let availableDoctors = [];
let currentAnalysisToShare = null;

// ========================================
// INITIALISATION
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('📤 Module de partage d\'analyses chargé');
    
    // Charger la liste des médecins pour le partage
    loadDoctorsForSharing();
    
    // Ajouter les boutons de partage aux analyses existantes
    addShareButtonsToAnalyses();
});

// ========================================
// CHARGEMENT DES MÉDECINS
// ========================================

async function loadDoctorsForSharing() {
    try {
        const response = await fetch('/api/messages/doctors');
        const data = await response.json();
        
        if (data.success) {
            availableDoctors = data.doctors;
            console.log(`✅ ${availableDoctors.length} médecins disponibles pour le partage`);
        }
    } catch (error) {
        console.error('❌ Erreur chargement médecins:', error);
    }
}

// ========================================
// INTERFACE DE PARTAGE
// ========================================

function openShareAnalysisModal(analysisId, patientName = 'Patient') {
    currentAnalysisToShare = analysisId;
    
    if (availableDoctors.length === 0) {
        showShareNotification('Chargement des médecins...', 'info');
        loadDoctorsForSharing().then(() => {
            if (availableDoctors.length > 0) {
                showShareModal(analysisId, patientName);
            } else {
                showShareNotification('Aucun médecin disponible pour le partage', 'warning');
            }
        });
        return;
    }
    
    showShareModal(analysisId, patientName);
}

function showShareModal(analysisId, patientName) {
    const modalHTML = `
        <div class="modal-overlay active" id="shareAnalysisOverlay">
            <div class="share-modal">
                <div class="share-modal-header">
                    <h3>
                        <i class="fas fa-share-alt"></i>
                        Partager l'analyse
                    </h3>
                    <button class="btn-close-share-modal" onclick="closeShareAnalysisModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <div class="share-modal-body">
                    <div class="analysis-info-share">
                        <i class="fas fa-brain"></i>
                        <div>
                            <strong>Analyse à partager</strong>
                            <p>${patientName}</p>
                        </div>
                    </div>
                    
                    <div class="form-group-share">
                        <label for="doctorSearchShare">
                            <i class="fas fa-user-md"></i>
                            Sélectionner un médecin
                        </label>
                        <input 
                            type="text" 
                            id="doctorSearchShare" 
                            class="form-control-share"
                            placeholder="Rechercher par nom, spécialité, hôpital..."
                            oninput="filterDoctorsForShare(this.value)">
                    </div>
                    
                    <div class="doctors-list-share" id="doctorsListShare">
                        ${renderDoctorsForShare(availableDoctors)}
                    </div>
                    
                    <div class="form-group-share">
                        <label for="shareMessage">
                            <i class="fas fa-comment"></i>
                            Message d'accompagnement (optionnel)
                        </label>
                        <textarea 
                            id="shareMessage" 
                            class="form-control-share"
                            placeholder="Ajouter un commentaire ou une note pour votre collègue..."
                            rows="3"></textarea>
                    </div>
                </div>
                
                <div class="share-modal-footer">
                    <button class="btn-cancel-share" onclick="closeShareAnalysisModal()">
                        Annuler
                    </button>
                    <button class="btn-share-submit" id="btnShareSubmit" disabled>
                        <i class="fas fa-paper-plane"></i>
                        Partager
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Focus sur le champ de recherche
    setTimeout(() => {
        document.getElementById('doctorSearchShare')?.focus();
    }, 100);
}

function renderDoctorsForShare(doctors) {
    if (doctors.length === 0) {
        return `
            <div class="empty-doctors-list">
                <i class="fas fa-user-md fa-2x"></i>
                <p>Aucun médecin trouvé</p>
            </div>
        `;
    }
    
    return doctors.map(doctor => `
        <div class="doctor-item-share" onclick="selectDoctorForShare('${doctor.id}', '${doctor.full_name}')">
            <div class="doctor-avatar-share">
                ${getInitialsShare(doctor.full_name)}
            </div>
            <div class="doctor-info-share">
                <h4>${doctor.full_name}</h4>
                <p>
                    <i class="fas fa-stethoscope"></i>
                    ${doctor.specialty}
                    ${doctor.hospital ? `<span class="separator">•</span> ${doctor.hospital}` : ''}
                </p>
            </div>
            <div class="doctor-select-icon">
                <i class="fas fa-chevron-right"></i>
            </div>
        </div>
    `).join('');
}

function filterDoctorsForShare(query) {
    const filtered = availableDoctors.filter(doctor => {
        const searchText = query.toLowerCase();
        return doctor.full_name.toLowerCase().includes(searchText) ||
               doctor.specialty.toLowerCase().includes(searchText) ||
               (doctor.hospital && doctor.hospital.toLowerCase().includes(searchText));
    });
    
    document.getElementById('doctorsListShare').innerHTML = renderDoctorsForShare(filtered);
}

let selectedDoctorId = null;
let selectedDoctorName = null;

function selectDoctorForShare(doctorId, doctorName) {
    selectedDoctorId = doctorId;
    selectedDoctorName = doctorName;
    
    // Mettre en surbrillance le médecin sélectionné
    document.querySelectorAll('.doctor-item-share').forEach(item => {
        item.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');
    
    // Activer le bouton de partage
    const btnShare = document.getElementById('btnShareSubmit');
    if (btnShare) {
        btnShare.disabled = false;
        btnShare.onclick = () => shareAnalysisWithDoctor();
    }
    
    // Mise à jour visuelle
    event.currentTarget.style.background = 'linear-gradient(135deg, #E0E7FF, #F3F4F6)';
    event.currentTarget.style.borderColor = '#4F46E5';
}

// ========================================
// PARTAGE DE L'ANALYSE
// ========================================

async function shareAnalysisWithDoctor() {
    if (!selectedDoctorId || !currentAnalysisToShare) {
        showShareNotification('Veuillez sélectionner un médecin', 'warning');
        return;
    }
    
    const message = document.getElementById('shareMessage')?.value || '';
    const btnShare = document.getElementById('btnShareSubmit');
    
    // Désactiver le bouton pendant l'envoi
    if (btnShare) {
        btnShare.disabled = true;
        btnShare.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Partage en cours...';
    }
    
    try {
        const response = await fetch('/api/messages/quick-share-analysis', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                recipient_id: selectedDoctorId,
                analysis_id: currentAnalysisToShare,
                message: message
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showShareNotification(
                `Analyse partagée avec succès à ${selectedDoctorName || 'le médecin'}`, 
                'success'
            );
            closeShareAnalysisModal();
            
            // Option : Rediriger vers la messagerie
            const goToMessages = confirm('Souhaitez-vous ouvrir la conversation ?');
            if (goToMessages) {
                window.location.href = '/messages';
            }
        } else {
            showShareNotification(data.error || 'Erreur lors du partage', 'error');
            if (btnShare) {
                btnShare.disabled = false;
                btnShare.innerHTML = '<i class="fas fa-paper-plane"></i> Partager';
            }
        }
    } catch (error) {
        console.error('❌ Erreur partage analyse:', error);
        showShareNotification('Erreur de connexion', 'error');
        if (btnShare) {
            btnShare.disabled = false;
            btnShare.innerHTML = '<i class="fas fa-paper-plane"></i> Partager';
        }
    }
}

function closeShareAnalysisModal() {
    const overlay = document.getElementById('shareAnalysisOverlay');
    if (overlay) {
        overlay.classList.remove('active');
        setTimeout(() => overlay.remove(), 300);
    }
    
    // Réinitialiser les variables
    currentAnalysisToShare = null;
    selectedDoctorId = null;
    selectedDoctorName = null;
}

// ========================================
// AJOUT DES BOUTONS DE PARTAGE
// ========================================

function addShareButtonsToAnalyses() {
    // Attendre que le DOM soit complètement chargé
    setTimeout(() => {
        // Chercher tous les conteneurs d'analyses
        const analysisContainers = document.querySelectorAll('[data-analysis-id]');
        
        analysisContainers.forEach(container => {
            const analysisId = container.getAttribute('data-analysis-id');
            const patientName = container.getAttribute('data-patient-name') || 'Patient';
            
            // Vérifier si le bouton n'existe pas déjà
            if (!container.querySelector('.btn-share-analysis')) {
                const shareButton = createShareButton(analysisId, patientName);
                
                // Trouver où insérer le bouton (dans les actions)
                const actionsContainer = container.querySelector('.analysis-actions, .card-actions, .action-buttons');
                if (actionsContainer) {
                    actionsContainer.insertAdjacentHTML('beforeend', shareButton);
                }
            }
        });
    }, 500);
}

function createShareButton(analysisId, patientName) {
    return `
        <button 
            class="btn-share-analysis" 
            onclick="openShareAnalysisModal('${analysisId}', '${patientName}')"
            title="Partager cette analyse avec un collègue">
            <i class="fas fa-share-alt"></i>
            <span>Partager</span>
        </button>
    `;
}

// ========================================
// UTILITAIRES
// ========================================

function getInitialsShare(name) {
    if (!name) return '?';
    const parts = name.trim().split(' ');
    if (parts.length >= 2) {
        return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }
    return name.substring(0, 2).toUpperCase();
}

function showShareNotification(message, type = 'info') {
    // Créer une notification toast
    const toast = document.createElement('div');
    toast.className = `share-toast share-toast-${type}`;
    
    const icon = type === 'success' ? 'check-circle' : 
                 type === 'error' ? 'exclamation-circle' : 
                 type === 'warning' ? 'exclamation-triangle' : 'info-circle';
    
    toast.innerHTML = `
        <i class="fas fa-${icon}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
    }, 100);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Fermer le modal avec la touche Escape
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeShareAnalysisModal();
    }
});

// Fonction globale pour partager depuis n'importe où
window.shareAnalysis = openShareAnalysisModal;

console.log('✅ Module de partage d\'analyses initialisé');
