/**
 * Système de confirmation de suppression moderne et sécurisé
 * NeuroScan - Gestion des Patients
 */

// Générer un code de confirmation aléatoire
function generateConfirmationCode() {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let code = '';
    for (let i = 0; i < 6; i++) {
        code += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return code;
}

// Afficher la modal de confirmation de suppression
function showDeleteConfirmation(patient) {
    // Générer le code de confirmation
    const confirmationCode = generateConfirmationCode();
    
    // Calculer l'âge si disponible
    let age = null;
    if (patient.date_of_birth) {
        const birth = new Date(patient.date_of_birth);
        const today = new Date();
        age = today.getFullYear() - birth.getFullYear();
        const monthDiff = today.getMonth() - birth.getMonth();
        if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
            age--;
        }
    }
    
    // Obtenir les initiales
    const initials = (patient.patient_name || patient.patient_id || 'P')
        .split(' ')
        .map(n => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2);
    
    // Créer la modal simplifiée
    const modalHTML = `
        <div class="delete-modal-overlay" id="deleteModal" role="dialog" aria-modal="true" aria-labelledby="deleteModalTitle">
            <div class="delete-modal-container">
                
                <!-- Corps simplifié -->
                <div class="delete-modal-body">
                    
                    <!-- En-tête avec icône et titre -->
                    <div class="simple-delete-header">
                        <div class="simple-delete-icon">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                        <h2 class="simple-delete-title" id="deleteModalTitle">
                            Supprimer le patient ?
                        </h2>
                    </div>
                    
                    <!-- Informations du patient - Compact -->
                    <div class="simple-patient-card">
                        <div class="simple-patient-avatar">
                            ${initials}
                        </div>
                        <div class="simple-patient-info">
                            <div class="simple-patient-name">
                                ${patient.patient_name || `Patient ${patient.patient_id}`}
                            </div>
                            <div class="simple-patient-meta">
                                <span><i class="fas fa-id-card"></i> ${patient.patient_id}</span>
                                ${age ? `<span><i class="fas fa-birthday-cake"></i> ${age} ans</span>` : ''}
                                <span><i class="fas fa-brain"></i> ${patient.total_analyses || 0} analyses</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Avertissement simple -->
                    <div class="simple-warning">
                        <i class="fas fa-info-circle"></i>
                        <span>Cette action est <strong>irréversible</strong>. Toutes les données du patient seront définitivement supprimées.</span>
                    </div>
                    
                    <!-- Code de confirmation - Focus principal -->
                    <div class="simple-code-section">
                        <div class="simple-code-label">
                            Pour confirmer la suppression, tapez ce code :
                        </div>
                        <div class="simple-code-display">
                            ${confirmationCode}
                        </div>
                        <input 
                            type="text" 
                            id="deleteConfirmationInput" 
                            class="simple-code-input" 
                            placeholder="Tapez le code ici"
                            autocomplete="off"
                            maxlength="6"
                            aria-label="Code de confirmation"
                        >
                        <div class="simple-code-hint">
                            <i class="fas fa-keyboard"></i> Tapez exactement le code affiché ci-dessus
                        </div>
                    </div>
                    
                </div>
                
                <!-- Pied avec boutons simples -->
                <div class="simple-modal-footer">
                    <button 
                        type="button" 
                        class="simple-btn simple-btn-cancel" 
                        onclick="closeDeleteModal()"
                    >
                        <i class="fas fa-times"></i>
                        Annuler
                    </button>
                    <button 
                        type="button" 
                        class="simple-btn simple-btn-delete" 
                        id="confirmDeleteButton"
                        disabled
                    >
                        <span class="simple-btn-spinner"></span>
                        <span class="simple-btn-text">
                            <i class="fas fa-trash-alt"></i>
                            Supprimer
                        </span>
                    </button>
                </div>
            </div>
        </div>
    `;
    
    // Ajouter la modal au DOM
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    document.body.style.overflow = 'hidden';
    
    // Référence à la modal et aux éléments
    const modal = document.getElementById('deleteModal');
    const input = document.getElementById('deleteConfirmationInput');
    const confirmButton = document.getElementById('confirmDeleteButton');
    
    // Focus sur l'input
    setTimeout(() => input.focus(), 300);
    
    // Validation en temps réel du code
    input.addEventListener('input', function(e) {
        const value = e.target.value.toUpperCase();
        e.target.value = value;
        
        if (value.length === 0) {
            input.classList.remove('valid', 'invalid');
            confirmButton.disabled = true;
        } else if (value === confirmationCode) {
            input.classList.remove('invalid');
            input.classList.add('valid');
            confirmButton.disabled = false;
        } else {
            input.classList.remove('valid');
            input.classList.add('invalid');
            confirmButton.disabled = true;
        }
    });
    
    // Gérer la suppression
    confirmButton.addEventListener('click', async function() {
        if (input.value.toUpperCase() !== confirmationCode) {
            showNotification('❌ Code de confirmation incorrect', 'error');
            input.classList.add('invalid');
            // Animation shake
            input.style.animation = 'none';
            setTimeout(() => {
                input.style.animation = '';
            }, 10);
            return;
        }
        
        // Désactiver le bouton et afficher le spinner
        confirmButton.disabled = true;
        confirmButton.classList.add('loading');
        input.disabled = true;
        
        try {
            // Appeler l'API de suppression
            const response = await fetch(`/api/patients/${patient.patient_id}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Animation de succès visuelle
                const iconElement = modal.querySelector('.simple-delete-icon i');
                const iconContainer = modal.querySelector('.simple-delete-icon');
                
                if (iconElement && iconContainer) {
                    iconElement.className = 'fas fa-check-circle';
                    iconContainer.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                    iconContainer.style.animation = 'successPulse 0.5s ease';
                }
                
                // Afficher notification de succès
                showNotification('✅ Patient supprimé avec succès', 'success');
                
                // Fermer immédiatement la modal et actualiser la liste
                closeDeleteModal();
                
                // Actualiser la liste des patients
                if (typeof loadPatients === 'function') {
                    // Recharger la liste avec animation
                    setTimeout(() => {
                        loadPatients();
                        updateSystemStatus();
                    }, 300);
                } else {
                    // Recharger la page si la fonction n'existe pas
                    setTimeout(() => {
                        window.location.reload();
                    }, 300);
                }
            } else {
                throw new Error(result.error || 'Erreur lors de la suppression');
            }
        } catch (error) {
            console.error('Erreur suppression:', error);
            showNotification(`❌ Erreur: ${error.message}`, 'error');
            
            // Réactiver le bouton
            confirmButton.disabled = false;
            confirmButton.classList.remove('loading');
            input.disabled = false;
            input.focus();
        }
    });
    
    // Fermer avec Échap
    document.addEventListener('keydown', handleEscapeKey);
    
    // Fermer en cliquant sur l'overlay
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeDeleteModal();
        }
    });
}

// Fonction pour gérer la touche Échap
function handleEscapeKey(e) {
    if (e.key === 'Escape') {
        closeDeleteModal();
    }
}

// Fermer la modal de suppression
function closeDeleteModal() {
    const modal = document.getElementById('deleteModal');
    if (!modal) return;
    
    // Animation de fermeture
    modal.classList.add('closing');
    
    // Retirer l'écouteur d'événement Échap
    document.removeEventListener('keydown', handleEscapeKey);
    
    // Supprimer la modal après l'animation
    setTimeout(() => {
        if (modal.parentNode) {
            modal.parentNode.removeChild(modal);
        }
        document.body.style.overflow = '';
    }, 300);
}

// Fonction principale pour initier la suppression
function deletePatient(patientId) {
    // Trouver le patient dans la liste
    const patient = allPatients.find(p => p.patient_id === patientId);
    
    if (!patient) {
        showNotification('❌ Patient non trouvé', 'error');
        return;
    }
    
    // Afficher la modal de confirmation
    showDeleteConfirmation(patient);
}

// Fonction de notification (utilise la fonction existante ou en crée une)
function showNotification(message, type = 'info', duration = 4000) {
    // Vérifier si la fonction existe déjà
    if (typeof window.showNotification === 'function') {
        return window.showNotification(message, type, duration);
    }
    
    // Sinon, créer une notification simple
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        padding: 16px 24px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        animation: slideInRight 0.3s ease;
        max-width: 400px;
    `;
    
    // Couleurs selon le type
    const colors = {
        success: 'linear-gradient(135deg, #10b981, #059669)',
        error: 'linear-gradient(135deg, #ef4444, #dc2626)',
        warning: 'linear-gradient(135deg, #f59e0b, #d97706)',
        info: 'linear-gradient(135deg, #3b82f6, #2563eb)'
    };
    
    notification.style.background = colors[type] || colors.info;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, duration);
}

// Ajouter les animations CSS si elles n'existent pas
if (!document.getElementById('deleteModalAnimations')) {
    const style = document.createElement('style');
    style.id = 'deleteModalAnimations';
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
}

console.log('✅ Système de confirmation de suppression chargé');
