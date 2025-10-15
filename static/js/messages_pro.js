/**
 * NeuroScan AI - Messagerie Professionnelle entre Médecins
 * Gestion des messages texte et partage d'analyses
 */

let currentConversation = null;
let currentRecipient = null;
let allDoctors = [];
let conversations = [];
let messagePollingInterval = null;

// ========================================
// INITIALISATION
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('📧 Initialisation de la messagerie professionnelle...');
    
    // Charger les données initiales
    loadDoctorsList();
    loadConversations();
    
    // Configurer les événements
    setupEventListeners();
    
    // Démarrer le polling des messages
    startMessagePolling();
});

// ========================================
// CHARGEMENT DES DONNÉES
// ========================================

async function loadDoctorsList() {
    try {
        const response = await fetch('/api/messages/doctors');
        const data = await response.json();
        
        if (data.success) {
            allDoctors = data.doctors;
            console.log(`✅ ${allDoctors.length} médecins chargés`);
        }
    } catch (error) {
        console.error('❌ Erreur chargement médecins:', error);
        showNotification('Erreur de chargement des médecins', 'error');
    }
}

async function loadConversations() {
    try {
        const response = await fetch('/api/messages/conversations');
        const data = await response.json();
        
        if (data.success) {
            conversations = data.conversations;
            displayConversations(conversations);
            
            // Charger la première conversation si elle existe
            if (conversations.length > 0 && !currentConversation) {
                selectConversation(conversations[0].id);
            }
        }
    } catch (error) {
        console.error('❌ Erreur chargement conversations:', error);
        showNotification('Erreur de chargement des conversations', 'error');
    }
}

async function loadMessages(conversationId) {
    try {
        const response = await fetch(`/api/messages/conversations/${conversationId}/messages`);
        const data = await response.json();
        
        if (data.success) {
            displayMessages(data.messages);
            scrollToBottom();
        }
    } catch (error) {
        console.error('❌ Erreur chargement messages:', error);
        showNotification('Erreur de chargement des messages', 'error');
    }
}

// ========================================
// AFFICHAGE DES CONVERSATIONS
// ========================================

function displayConversations(conversations) {
    const listContainer = document.getElementById('conversationsList');
    
    if (conversations.length === 0) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-comments fa-3x"></i>
                <h3>Aucune conversation</h3>
                <p>Commencez une nouvelle conversation avec un collègue</p>
                <button class="btn-primary" onclick="openNewMessageModal()">
                    <i class="fas fa-plus"></i> Nouveau message
                </button>
            </div>
        `;
        return;
    }
    
    listContainer.innerHTML = conversations.map(conv => {
        const otherDoctor = conv.other_doctor;
        const lastMessage = conv.last_message;
        const isActive = currentConversation === conv.id;
        
        return `
            <div class="conversation-item ${isActive ? 'active' : ''}" 
                 data-id="${conv.id}"
                 onclick="selectConversation('${conv.id}')">
                <div class="conversation-avatar">
                    <div class="avatar-circle">
                        ${getInitials(otherDoctor.full_name)}
                    </div>
                    ${otherDoctor.is_online ? '<span class="status-indicator online"></span>' : ''}
                </div>
                <div class="conversation-content">
                    <div class="conversation-header">
                        <h4>${otherDoctor.full_name}</h4>
                        <span class="conversation-time">
                            ${lastMessage ? formatMessageTime(lastMessage.created_at) : ''}
                        </span>
                    </div>
                    <div class="conversation-preview">
                        ${lastMessage ? `
                            <p class="last-message">
                                ${lastMessage.is_from_me ? '<i class="fas fa-reply"></i> ' : ''}
                                ${truncate(lastMessage.content, 50)}
                            </p>
                        ` : `
                            <p class="specialty-tag">${otherDoctor.specialty}</p>
                        `}
                        ${conv.unread_count > 0 ? `
                            <span class="unread-badge">${conv.unread_count}</span>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ========================================
// AFFICHAGE DES MESSAGES
// ========================================

function displayMessages(messages) {
    const container = document.getElementById('messagesArea');
    
    if (messages.length === 0) {
        container.innerHTML = `
            <div class="empty-messages">
                <i class="fas fa-comment-dots fa-3x"></i>
                <h3>Aucun message</h3>
                <p>Commencez la conversation</p>
            </div>
        `;
        return;
    }
    
    let currentDate = '';
    let html = '';
    
    messages.forEach(msg => {
        const msgDate = new Date(msg.created_at).toLocaleDateString('fr-FR');
        
        // Ajouter un séparateur de date si nécessaire
        if (msgDate !== currentDate) {
            html += `
                <div class="date-divider">
                    <span>${formatMessageDate(msg.created_at)}</span>
                </div>
            `;
            currentDate = msgDate;
        }
        
        // Message selon le type
        if (msg.message_type === 'analysis_share') {
            html += renderAnalysisMessage(msg);
        } else {
            html += renderTextMessage(msg);
        }
    });
    
    container.innerHTML = html;
}

function renderTextMessage(msg) {
    const isFromMe = msg.is_from_me;
    const wrapperClass = isFromMe ? 'sent' : 'received';
    
    return `
        <div class="message-wrapper ${wrapperClass}">
            ${!isFromMe ? `
                <div class="message-avatar">
                    <div class="avatar-circle-sm">
                        ${getInitials(msg.sender.full_name)}
                    </div>
                </div>
            ` : ''}
            <div class="message-content">
                <div class="message-bubble">
                    <p>${escapeHtml(msg.content)}</p>
                </div>
                <span class="message-time">
                    ${formatTime(msg.created_at)}
                    ${isFromMe ? `<i class="fas fa-check-double ${msg.is_read ? 'read' : ''}"></i>` : ''}
                </span>
            </div>
        </div>
    `;
}

function renderAnalysisMessage(msg) {
    const isFromMe = msg.is_from_me;
    const wrapperClass = isFromMe ? 'sent' : 'received';
    const analysisData = msg.analysis_data || {};
    
    return `
        <div class="message-wrapper ${wrapperClass}">
            ${!isFromMe ? `
                <div class="message-avatar">
                    <div class="avatar-circle-sm">
                        ${getInitials(msg.sender.full_name)}
                    </div>
                </div>
            ` : ''}
            <div class="message-content">
                ${msg.content ? `
                    <div class="message-bubble">
                        <p>${escapeHtml(msg.content)}</p>
                    </div>
                ` : ''}
                <div class="analysis-share-card" onclick="viewSharedAnalysis('${msg.analysis_id}')">
                    <div class="analysis-share-header">
                        <i class="fas fa-brain"></i>
                        <span>Analyse partagée</span>
                    </div>
                    <div class="analysis-share-body">
                        <div class="analysis-info">
                            <p class="patient-name">
                                <i class="fas fa-user"></i>
                                ${analysisData.patient_name || 'Patient'}
                            </p>
                            <p class="analysis-result">
                                <strong>${analysisData.predicted_label || 'N/A'}</strong>
                                <span class="confidence-badge ${getConfidenceClass(analysisData.confidence)}">
                                    ${(analysisData.confidence * 100).toFixed(1)}%
                                </span>
                            </p>
                            ${analysisData.exam_date ? `
                                <p class="exam-date">
                                    <i class="fas fa-calendar"></i>
                                    ${formatDate(analysisData.exam_date)}
                                </p>
                            ` : ''}
                        </div>
                        ${analysisData.image_filename ? `
                            <div class="analysis-thumbnail">
                                <img src="/uploads/${analysisData.image_filename}" 
                                     alt="Analyse" 
                                     onerror="this.style.display='none'">
                            </div>
                        ` : ''}
                    </div>
                    <div class="analysis-share-footer">
                        <button class="btn-view-analysis">
                            <i class="fas fa-eye"></i> Voir l'analyse complète
                        </button>
                    </div>
                </div>
                <span class="message-time">
                    ${formatTime(msg.created_at)}
                    ${isFromMe ? `<i class="fas fa-check-double ${msg.is_read ? 'read' : ''}"></i>` : ''}
                </span>
            </div>
        </div>
    `;
}

// ========================================
// GESTION DES CONVERSATIONS
// ========================================

async function selectConversation(conversationId) {
    currentConversation = conversationId;
    
    // Mettre à jour l'interface
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('active');
    });
    
    const selectedItem = document.querySelector(`[data-id="${conversationId}"]`);
    if (selectedItem) {
        selectedItem.classList.add('active');
    }
    
    // Trouver les infos du destinataire
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
        currentRecipient = conversation.other_doctor;
        updateChatHeader(currentRecipient);
    }
    
    // Charger les messages
    await loadMessages(conversationId);
    
    // Activer l'input
    document.getElementById('messageInput').disabled = false;
}

function updateChatHeader(doctor) {
    const headerHTML = `
        <div class="chat-user-info">
            <div class="chat-avatar">
                <div class="avatar-circle">
                    ${getInitials(doctor.full_name)}
                </div>
                ${doctor.is_online ? '<span class="status-indicator online"></span>' : ''}
            </div>
            <div class="chat-user-details">
                <h3>${doctor.full_name}</h3>
                <p class="user-status">
                    ${doctor.specialty}${doctor.hospital ? ` • ${doctor.hospital}` : ''}
                </p>
            </div>
        </div>
        <div class="chat-actions">
            <button class="btn-icon" onclick="shareAnalysisModal()" title="Partager une analyse">
                <i class="fas fa-share-alt"></i>
            </button>
            <button class="btn-icon" onclick="viewSharedAnalyses()" title="Analyses partagées">
                <i class="fas fa-folder-open"></i>
            </button>
            <button class="btn-icon" title="Plus d'options">
                <i class="fas fa-ellipsis-v"></i>
            </button>
        </div>
    `;
    
    document.querySelector('.chat-header').innerHTML = headerHTML;
}

// ========================================
// ENVOI DE MESSAGES
// ========================================

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const content = input.value.trim();
    
    if (!content || !currentConversation) return;
    
    try {
        const response = await fetch('/api/messages/send', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                conversation_id: currentConversation,
                content: content
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            input.value = '';
            input.style.height = 'auto';
            
            // Recharger les messages
            await loadMessages(currentConversation);
            
            // Mettre à jour la liste des conversations
            await loadConversations();
        } else {
            showNotification(data.error || 'Erreur d\'envoi', 'error');
        }
    } catch (error) {
        console.error('❌ Erreur envoi message:', error);
        showNotification('Erreur d\'envoi du message', 'error');
    }
}

// ========================================
// PARTAGE D'ANALYSES
// ========================================

async function shareAnalysisModal() {
    if (!currentConversation) {
        showNotification('Sélectionnez une conversation d\'abord', 'warning');
        return;
    }
    
    // Récupérer les analyses du médecin
    try {
        const response = await fetch('/api/my-patients');
        const data = await response.json();
        
        if (data.success && data.patients) {
            // Extraire toutes les analyses
            let allAnalyses = [];
            data.patients.forEach(patient => {
                if (patient.analyses && patient.analyses.length > 0) {
                    patient.analyses.forEach(analysis => {
                        allAnalyses.push({
                            ...analysis,
                            patient_name: patient.patient_name
                        });
                    });
                }
            });
            
            showAnalysisSelectionModal(allAnalyses);
        }
    } catch (error) {
        console.error('❌ Erreur chargement analyses:', error);
        showNotification('Erreur de chargement des analyses', 'error');
    }
}

function showAnalysisSelectionModal(analyses) {
    if (analyses.length === 0) {
        showNotification('Aucune analyse disponible pour le partage', 'info');
        return;
    }
    
    const modalHTML = `
        <div class="modal active" id="shareAnalysisModal">
            <div class="modal-content modal-large">
                <div class="modal-header">
                    <h3><i class="fas fa-share-alt"></i> Partager une analyse</h3>
                    <button class="btn-close-modal" onclick="closeShareModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="analyses-grid">
                        ${analyses.map(analysis => `
                            <div class="analysis-card" onclick="selectAnalysisToShare('${analysis.id}')">
                                ${analysis.image_filename ? `
                                    <div class="analysis-card-image">
                                        <img src="/uploads/${analysis.image_filename}" alt="Analyse">
                                    </div>
                                ` : ''}
                                <div class="analysis-card-body">
                                    <h4>${analysis.patient_name}</h4>
                                    <p class="analysis-result">
                                        <strong>${analysis.predicted_label}</strong>
                                        <span class="confidence-badge ${getConfidenceClass(analysis.confidence)}">
                                            ${(analysis.confidence * 100).toFixed(1)}%
                                        </span>
                                    </p>
                                    <p class="analysis-date">
                                        <i class="fas fa-calendar"></i>
                                        ${formatDate(analysis.exam_date || analysis.timestamp)}
                                    </p>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

async function selectAnalysisToShare(analysisId) {
    const message = prompt('Message d\'accompagnement (optionnel):');
    
    try {
        const response = await fetch('/api/messages/share-analysis', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                conversation_id: currentConversation,
                analysis_id: analysisId,
                message: message || ''
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('Analyse partagée avec succès', 'success');
            closeShareModal();
            
            // Recharger les messages
            await loadMessages(currentConversation);
            await loadConversations();
        } else {
            showNotification(data.error || 'Erreur de partage', 'error');
        }
    } catch (error) {
        console.error('❌ Erreur partage analyse:', error);
        showNotification('Erreur de partage de l\'analyse', 'error');
    }
}

function closeShareModal() {
    const modal = document.getElementById('shareAnalysisModal');
    if (modal) {
        modal.remove();
    }
}

// ========================================
// VISUALISATION DES ANALYSES PARTAGÉES
// ========================================

async function viewSharedAnalyses() {
    try {
        const response = await fetch('/api/messages/shared-analyses');
        const data = await response.json();
        
        if (data.success) {
            showSharedAnalysesModal(data.shared_analyses);
        }
    } catch (error) {
        console.error('❌ Erreur chargement analyses partagées:', error);
        showNotification('Erreur de chargement', 'error');
    }
}

function showSharedAnalysesModal(sharedAnalyses) {
    if (sharedAnalyses.length === 0) {
        showNotification('Aucune analyse partagée avec vous', 'info');
        return;
    }
    
    const modalHTML = `
        <div class="modal active" id="sharedAnalysesModal">
            <div class="modal-content modal-large">
                <div class="modal-header">
                    <h3><i class="fas fa-folder-open"></i> Analyses partagées avec moi</h3>
                    <button class="btn-close-modal" onclick="closeSharedAnalysesModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="shared-analyses-list">
                        ${sharedAnalyses.map(item => {
                            const analysisData = item.analysis_data;
                            return `
                                <div class="shared-analysis-item" onclick="viewSharedAnalysis('${item.analysis_id}')">
                                    <div class="shared-analysis-header">
                                        <div class="sender-info">
                                            <div class="avatar-circle-sm">
                                                ${getInitials(item.sender.full_name)}
                                            </div>
                                            <div>
                                                <h4>${item.sender.full_name}</h4>
                                                <p>${item.sender.specialty}</p>
                                            </div>
                                        </div>
                                        <span class="shared-date">${formatMessageDate(item.shared_at)}</span>
                                    </div>
                                    <div class="shared-analysis-body">
                                        <div class="analysis-details">
                                            <h5>${analysisData.patient_name}</h5>
                                            <p class="analysis-result">
                                                <strong>${analysisData.predicted_label}</strong>
                                                <span class="confidence-badge ${getConfidenceClass(analysisData.confidence)}">
                                                    ${(analysisData.confidence * 100).toFixed(1)}%
                                                </span>
                                            </p>
                                            ${item.message ? `<p class="share-message">"${item.message}"</p>` : ''}
                                        </div>
                                        ${analysisData.image_filename ? `
                                            <div class="analysis-thumbnail">
                                                <img src="/uploads/${analysisData.image_filename}" alt="Analyse">
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

function closeSharedAnalysesModal() {
    const modal = document.getElementById('sharedAnalysesModal');
    if (modal) {
        modal.remove();
    }
}

async function viewSharedAnalysis(analysisId) {
    try {
        const response = await fetch(`/api/messages/analysis/${analysisId}`);
        const data = await response.json();
        
        if (data.success) {
            showAnalysisDetailsModal(data.analysis);
        }
    } catch (error) {
        console.error('❌ Erreur chargement détails analyse:', error);
        showNotification('Erreur de chargement', 'error');
    }
}

function showAnalysisDetailsModal(analysis) {
    const modalHTML = `
        <div class="modal active" id="analysisDetailsModal">
            <div class="modal-content modal-large">
                <div class="modal-header">
                    <h3><i class="fas fa-brain"></i> Détails de l'analyse</h3>
                    <button class="btn-close-modal" onclick="closeAnalysisDetailsModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="analysis-details-container">
                        <div class="analysis-main">
                            ${analysis.image_filename ? `
                                <div class="analysis-image">
                                    <img src="/uploads/${analysis.image_filename}" alt="Analyse">
                                </div>
                            ` : ''}
                            
                            <div class="analysis-info-grid">
                                <div class="info-card">
                                    <h4><i class="fas fa-user"></i> Patient</h4>
                                    <p>${analysis.patient_name}</p>
                                    ${analysis.patient_id ? `<small>ID: ${analysis.patient_id}</small>` : ''}
                                </div>
                                
                                <div class="info-card">
                                    <h4><i class="fas fa-calendar"></i> Date</h4>
                                    <p>${formatDate(analysis.exam_date || analysis.created_at)}</p>
                                </div>
                                
                                <div class="info-card">
                                    <h4><i class="fas fa-diagnoses"></i> Diagnostic</h4>
                                    <p class="diagnosis-result">${analysis.predicted_label}</p>
                                    <span class="confidence-badge ${getConfidenceClass(analysis.confidence)}">
                                        Confiance: ${(analysis.confidence * 100).toFixed(1)}%
                                    </span>
                                </div>
                                
                                ${analysis.owner_doctor ? `
                                    <div class="info-card">
                                        <h4><i class="fas fa-user-md"></i> Médecin</h4>
                                        <p>${analysis.owner_doctor.full_name}</p>
                                        <small>${analysis.owner_doctor.specialty}</small>
                                    </div>
                                ` : ''}
                            </div>
                            
                            ${analysis.probabilities ? `
                                <div class="probabilities-section">
                                    <h4>Probabilités détaillées</h4>
                                    <div class="probabilities-bars">
                                        ${Object.entries(analysis.probabilities).map(([label, prob]) => `
                                            <div class="probability-item">
                                                <span class="prob-label">${label}</span>
                                                <div class="prob-bar">
                                                    <div class="prob-fill" style="width: ${prob * 100}%"></div>
                                                </div>
                                                <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${analysis.recommendations && analysis.recommendations.length > 0 ? `
                                <div class="recommendations-section">
                                    <h4>Recommandations</h4>
                                    <ul class="recommendations-list">
                                        ${analysis.recommendations.map(rec => `
                                            <li><i class="fas fa-check"></i> ${rec}</li>
                                        `).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-secondary" onclick="closeAnalysisDetailsModal()">
                        Fermer
                    </button>
                    <a href="/analysis/${analysis.id}" class="btn-primary" target="_blank">
                        <i class="fas fa-external-link-alt"></i> Ouvrir dans l'analyseur
                    </a>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

function closeAnalysisDetailsModal() {
    const modal = document.getElementById('analysisDetailsModal');
    if (modal) {
        modal.remove();
    }
}

// ========================================
// NOUVELLE CONVERSATION
// ========================================

function openNewMessageModal() {
    if (allDoctors.length === 0) {
        showNotification('Chargement des médecins...', 'info');
        loadDoctorsList().then(() => {
            if (allDoctors.length > 0) {
                showNewMessageModal();
            } else {
                showNotification('Aucun médecin disponible', 'warning');
            }
        });
    } else {
        showNewMessageModal();
    }
}

function showNewMessageModal() {
    const modalHTML = `
        <div class="modal active" id="newMessageModal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-edit"></i> Nouveau message</h3>
                    <button class="btn-close-modal" onclick="closeNewMessageModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="form-group">
                        <label>Rechercher un médecin</label>
                        <input type="text" 
                               id="doctorSearch" 
                               class="form-control"
                               placeholder="Nom, spécialité, hôpital..."
                               oninput="filterDoctors(this.value)">
                    </div>
                    <div class="doctors-list" id="doctorsListModal">
                        ${renderDoctorsList(allDoctors)}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

function renderDoctorsList(doctors) {
    if (doctors.length === 0) {
        return '<p class="text-center text-muted">Aucun médecin trouvé</p>';
    }
    
    return doctors.map(doctor => `
        <div class="doctor-item" onclick="startConversationWith('${doctor.id}')">
            <div class="avatar-circle">
                ${getInitials(doctor.full_name)}
            </div>
            <div class="doctor-info">
                <h4>${doctor.full_name}</h4>
                <p>${doctor.specialty}${doctor.hospital ? ` • ${doctor.hospital}` : ''}</p>
            </div>
        </div>
    `).join('');
}

function filterDoctors(query) {
    const filtered = allDoctors.filter(doctor => {
        const searchText = query.toLowerCase();
        return doctor.full_name.toLowerCase().includes(searchText) ||
               doctor.specialty.toLowerCase().includes(searchText) ||
               (doctor.hospital && doctor.hospital.toLowerCase().includes(searchText));
    });
    
    document.getElementById('doctorsListModal').innerHTML = renderDoctorsList(filtered);
}

async function startConversationWith(doctorId) {
    try {
        const response = await fetch('/api/messages/conversations', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({recipient_id: doctorId})
        });
        
        const data = await response.json();
        
        if (data.success) {
            closeNewMessageModal();
            
            // Recharger les conversations
            await loadConversations();
            
            // Sélectionner la nouvelle conversation
            await selectConversation(data.conversation_id);
            
            // Focus sur l'input
            document.getElementById('messageInput').focus();
        }
    } catch (error) {
        console.error('❌ Erreur création conversation:', error);
        showNotification('Erreur de création de conversation', 'error');
    }
}

function closeNewMessageModal() {
    const modal = document.getElementById('newMessageModal');
    if (modal) {
        modal.remove();
    }
}

// ========================================
// EVENT LISTENERS
// ========================================

function setupEventListeners() {
    // Bouton nouveau message
    const btnNewMessage = document.getElementById('btnNewMessage');
    if (btnNewMessage) {
        btnNewMessage.addEventListener('click', openNewMessageModal);
    }
    
    // Input message
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        // Auto-resize
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Envoyer avec Enter
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Bouton envoyer
    const btnSend = document.getElementById('btnSend');
    if (btnSend) {
        btnSend.addEventListener('click', sendMessage);
    }
    
    // Recherche conversations
    const searchInput = document.getElementById('searchConversations');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            const filtered = conversations.filter(conv => {
                const doctor = conv.other_doctor;
                return doctor.full_name.toLowerCase().includes(query) ||
                       doctor.specialty.toLowerCase().includes(query);
            });
            displayConversations(filtered);
        });
    }
}

// ========================================
// POLLING DES MESSAGES
// ========================================

function startMessagePolling() {
    // Rafraîchir toutes les 10 secondes
    messagePollingInterval = setInterval(() => {
        if (currentConversation) {
            loadMessages(currentConversation);
        }
        loadConversations();
    }, 10000);
}

// ========================================
// UTILITAIRES
// ========================================

function getInitials(name) {
    if (!name) return '?';
    const parts = name.split(' ');
    if (parts.length >= 2) {
        return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }
    return name.substring(0, 2).toUpperCase();
}

function truncate(text, length) {
    if (!text) return '';
    return text.length > length ? text.substring(0, length) + '...' : text;
}

function formatMessageTime(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;
    const hours = Math.floor(diff / (1000 * 60 * 60));
    
    if (hours < 1) {
        const minutes = Math.floor(diff / (1000 * 60));
        return minutes < 1 ? 'À l\'instant' : `${minutes}min`;
    } else if (hours < 24) {
        return `${hours}h`;
    } else {
        const days = Math.floor(hours / 24);
        return days === 1 ? 'Hier' : `${days}j`;
    }
}

function formatMessageDate(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return 'Aujourd\'hui';
    if (days === 1) return 'Hier';
    if (days < 7) return date.toLocaleDateString('fr-FR', { weekday: 'long' });
    return date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'long', year: 'numeric' });
}

function formatTime(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
}

function formatDate(dateStr) {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return date.toLocaleDateString('fr-FR', { 
        day: 'numeric', 
        month: 'short', 
        year: 'numeric' 
    });
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.9) return 'confidence-high';
    if (confidence >= 0.7) return 'confidence-medium';
    return 'confidence-low';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    const container = document.getElementById('messagesArea');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

function showNotification(message, type = 'info') {
    // Créer une notification toast
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
    }, 100);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Nettoyer au déchargement de la page
window.addEventListener('beforeunload', () => {
    if (messagePollingInterval) {
        clearInterval(messagePollingInterval);
    }
});

console.log('✅ Module messagerie professionnelle chargé');
