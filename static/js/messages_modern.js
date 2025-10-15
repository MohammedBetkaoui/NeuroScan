/**
 * NeuroScan AI - Interface de Messagerie Moderne
 * Gestion complète des messages entre médecins
 */

// ========================================
// VARIABLES GLOBALES
// ========================================
let currentConversation = null;
let currentRecipient = null;
let allDoctors = [];
let conversations = [];
let messagePollingInterval = null;
let currentMessages = []; // Cache des messages actuels
let lastMessageId = null; // ID du dernier message chargé
let useWebSocket = false; // TEMPORAIRE: Désactiver WebSocket pour tester le polling d'abord
let fallbackPolling = false; // Mode fallback si WebSocket échoue

// ========================================
// INITIALISATION
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('📧 Initialisation de la messagerie moderne...');
    
    // Charger les données initiales
    initializeMessaging();
    
    // Configurer les événements
    setupEventListeners();
    
    // Initialiser WebSocket ou polling selon la disponibilité
    if (useWebSocket && window.wsManager) {
        initializeWebSocket();
    } else {
        // Fallback au polling classique
        startMessagePolling();
    }
});

async function initializeMessaging() {
    try {
        // Afficher un état de chargement
        showLoadingState();
        
        // Charger les médecins et conversations en parallèle
        await Promise.all([
            loadDoctorsList(),
            loadConversations()
        ]);
        
        console.log('✅ Messagerie initialisée avec succès');
    } catch (error) {
        console.error('❌ Erreur initialisation messagerie:', error);
        showErrorState('Impossible de charger la messagerie');
    }
}

// ========================================
// WEBSOCKET - TEMPS RÉEL
// ========================================

/**
 * Initialiser la connexion WebSocket
 */
function initializeWebSocket() {
    console.log('🔌 Initialisation WebSocket...');
    
    // Récupérer l'ID du médecin connecté
    const doctorId = getCurrentDoctorId();
    console.log('👤 Doctor ID:', doctorId);
    
    if (!doctorId) {
        console.error('❌ ID médecin non trouvé, fallback au polling');
        fallbackPolling = true;
        startMessagePolling();
        return;
    }
    
    // Vérifier que wsManager existe
    if (!window.wsManager) {
        console.error('❌ wsManager non trouvé, fallback au polling');
        fallbackPolling = true;
        startMessagePolling();
        return;
    }
    
    // Connecter le WebSocket
    window.wsManager.connect(doctorId);
    
    // Enregistrer les callbacks pour les événements
    window.wsManager.onNewMessage((message) => {
        handleWebSocketNewMessage(message);
    });
    
    window.wsManager.onTyping((data) => {
        handleWebSocketTyping(data);
    });
    
    window.wsManager.onMessagesRead((data) => {
        handleWebSocketMessagesRead(data);
    });
    
    window.wsManager.onDoctorOnline((data) => {
        handleWebSocketDoctorOnline(data);
    });
    
    window.wsManager.onDoctorOffline((data) => {
        handleWebSocketDoctorOffline(data);
    });
    
    console.log('✅ WebSocket initialisé');
}

/**
 * Gérer la réception d'un nouveau message via WebSocket
 */
function handleWebSocketNewMessage(message) {
    console.log('📨 Nouveau message WebSocket:', message);
    
    // Vérifier si le message appartient à la conversation actuelle
    if (currentConversation && message.conversation_id === currentConversation) {
        // Vérifier que le message n'existe pas déjà
        const exists = currentMessages.find(m => 
            (m._id || m.id) === (message._id || message.id)
        );
        
        if (!exists) {
            // Déterminer si c'est notre message
            const doctorId = getCurrentDoctorId();
            message.is_from_me = message.sender_id === doctorId;
            
            // Ajouter au cache
            currentMessages.push(message);
            
            // Afficher le message
            appendMessages([message]);
            
            // Marquer comme lu si ce n'est pas notre message
            if (!message.is_from_me) {
                window.wsManager.markAsRead(currentConversation);
            }
        }
    }
    
    // Mettre à jour la liste des conversations (afficher la preview du dernier message)
    updateConversationPreview(message.conversation_id, message.content, message.created_at);
}

/**
 * Gérer l'indicateur "en train d'écrire"
 */
function handleWebSocketTyping(data) {
    if (currentConversation && data.doctor_id !== getCurrentDoctorId()) {
        showTypingIndicator(data.is_typing);
    }
}

/**
 * Gérer les messages marqués comme lus
 */
function handleWebSocketMessagesRead(data) {
    if (currentConversation === data.conversation_id) {
        // Mettre à jour les indicateurs de lecture
        updateReadIndicators();
    }
}

/**
 * Gérer un médecin qui vient en ligne
 */
function handleWebSocketDoctorOnline(data) {
    console.log('🟢 Médecin en ligne:', data.doctor_id);
    updateDoctorOnlineStatus(data.doctor_id, true);
}

/**
 * Gérer un médecin qui passe hors ligne
 */
function handleWebSocketDoctorOffline(data) {
    console.log('🔴 Médecin hors ligne:', data.doctor_id);
    updateDoctorOnlineStatus(data.doctor_id, false);
}

/**
 * Fonction de fallback activée si WebSocket échoue
 */
window.activateFallbackPolling = function() {
    console.warn('⚠️ Activation du polling HTTP en fallback');
    fallbackPolling = true;
    useWebSocket = false;
    startMessagePolling();
};

// ========================================
// CHARGEMENT DES DONNÉES
// ========================================
async function loadDoctorsList() {
    try {
        const response = await fetch('/api/messages/doctors');
        const data = await response.json();
        
        if (data.success) {
            allDoctors = data.doctors || [];
            console.log(`✅ ${allDoctors.length} médecins chargés`);
            return allDoctors;
        } else {
            throw new Error(data.message || 'Erreur de chargement');
        }
    } catch (error) {
        console.error('❌ Erreur chargement médecins:', error);
        showNotification('Erreur de chargement des médecins', 'error');
        return [];
    }
}

async function loadConversations() {
    try {
        const response = await fetch('/api/messages/conversations');
        const data = await response.json();
        
        if (data.success) {
            conversations = data.conversations || [];
            console.log(`✅ ${conversations.length} conversations chargées`);
            displayConversations(conversations);
            
            // Sélectionner la première conversation si disponible
            if (conversations.length > 0 && !currentConversation) {
                selectConversation(conversations[0].id);
            }
            
            return conversations;
        } else {
            throw new Error(data.message || 'Erreur de chargement');
        }
    } catch (error) {
        console.error('❌ Erreur chargement conversations:', error);
        displayConversations([]);
        return [];
    }
}

async function loadMessages(conversationId, append = false) {
    try {
        const response = await fetch(`/api/messages/conversations/${conversationId}/messages`);
        const data = await response.json();
        
        if (data.success) {
            const messages = data.messages || [];
            
            if (append && currentMessages.length > 0) {
                // Mode incrémental : ajouter seulement les nouveaux messages
                const newMessages = messages.filter(msg => {
                    const msgId = msg._id || msg.id;
                    return !currentMessages.some(existingMsg => {
                        const existingId = existingMsg._id || existingMsg.id;
                        return existingId === msgId;
                    });
                });
                
                if (newMessages.length > 0) {
                    console.log(`➕ ${newMessages.length} nouveau(x) message(s)`);
                    currentMessages = [...currentMessages, ...newMessages];
                    appendMessages(newMessages);
                    scrollToBottom();
                }
            } else {
                // Mode complet : charger tous les messages
                currentMessages = messages;
                displayMessages(messages);
                scrollToBottom();
            }
            
            // Mettre à jour le dernier message ID
            if (currentMessages.length > 0) {
                const lastMsg = currentMessages[currentMessages.length - 1];
                lastMessageId = lastMsg._id || lastMsg.id;
            }
            
            // Note: Les messages sont automatiquement marqués comme lus par l'API
        } else {
            throw new Error(data.message || 'Erreur de chargement');
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
    
    if (!listContainer) return;
    
    if (conversations.length === 0) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-comments fa-3x"></i>
                <h3>Aucune conversation</h3>
                <p>Commencez une nouvelle conversation avec un collègue</p>
            </div>
        `;
        return;
    }
    
    listContainer.innerHTML = conversations.map(conv => {
        const unreadBadge = conv.unread_count > 0 
            ? `<span class="unread-badge">${conv.unread_count}</span>` 
            : '';
        
        // L'API retourne other_doctor avec les infos du destinataire
        const otherDoctor = conv.other_doctor || {};
        const doctorName = otherDoctor.full_name || `${otherDoctor.first_name || ''} ${otherDoctor.last_name || ''}`.trim() || 'Médecin';
        const doctorSpecialty = otherDoctor.specialty || '';
        
        // Status (TODO: implémenter le système de présence en ligne)
        const statusClass = 'offline'; // Par défaut offline
        const isUnread = conv.unread_count > 0 ? 'unread' : '';
        
        // Dernier message
        const lastMessage = conv.last_message || {};
        const lastMessageText = lastMessage.content || 'Aucun message';
        const lastMessageTime = lastMessage.created_at;
        
        return `
            <div class="conversation-item ${isUnread}" data-id="${conv.id}" data-doctor-id="${otherDoctor.id}" onclick="selectConversation('${conv.id}')">
                <div class="conversation-avatar">
                    <img src="/static/images/avatar-default.svg" alt="${doctorName}" onerror="this.src='/static/images/avatar-default.svg'">
                    <span class="status-indicator ${statusClass}"></span>
                </div>
                <div class="conversation-content">
                    <div class="conversation-header">
                        <h4>${doctorName}</h4>
                        <span class="conversation-time">${formatTime(lastMessageTime)}</span>
                    </div>
                    <div class="conversation-preview">
                        <p>${escapeHtml(lastMessageText)}</p>
                        ${unreadBadge}
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
    const messagesArea = document.getElementById('messagesArea');
    
    if (!messagesArea) return;
    
    if (messages.length === 0) {
        messagesArea.innerHTML = `
            <div class="empty-messages">
                <i class="fas fa-comment-dots fa-3x"></i>
                <h3>Aucun message</h3>
                <p>Commencez la conversation en envoyant un message</p>
            </div>
        `;
        return;
    }
    
    // Grouper les messages par date
    const groupedMessages = groupMessagesByDate(messages);
    
    let html = '';
    for (const [date, msgs] of Object.entries(groupedMessages)) {
        html += `<div class="date-divider"><span>${date}</span></div>`;
        
        msgs.forEach(msg => {
            // L'API retourne is_from_me pour savoir si c'est nous qui avons envoyé
            const messageClass = msg.is_from_me ? 'sent' : 'received';
            const readIndicator = msg.is_read ? '<i class="fas fa-check-double read"></i>' : '<i class="fas fa-check"></i>';
            
            // Nom de l'expéditeur
            const senderName = msg.sender ? msg.sender.full_name : 'Médecin';
            
            // Contenu du message (l'API utilise 'content' pas 'message_text')
            const messageContent = msg.content || msg.message_text || '';
            
            // Construire le HTML des fichiers attachés
            let filesHTML = '';
            if (msg.files && msg.files.length > 0) {
                filesHTML = '<div class="message-files">';
                msg.files.forEach(file => {
                    const icon = getFileIcon(file.mime_type || file.file_extension || '');
                    const fileId = file._id || file.id;
                    filesHTML += `
                        <a href="/api/messages/files/${fileId}" 
                           class="message-file-item" 
                           download="${escapeHtml(file.original_filename)}"
                           target="_blank"
                           title="Télécharger ${escapeHtml(file.original_filename)}">
                            <div class="message-file-icon">
                                <i class="${icon}"></i>
                            </div>
                            <div class="message-file-info">
                                <div class="message-file-name">${escapeHtml(file.original_filename)}</div>
                                <div class="message-file-size">${file.file_size_formatted || formatFileSize(file.file_size)}</div>
                            </div>
                            <div class="message-file-download">
                                <i class="fas fa-download"></i>
                            </div>
                        </a>
                    `;
                });
                filesHTML += '</div>';
            }
            
            html += `
                <div class="message-wrapper ${messageClass}">
                    <div class="message-avatar">
                        <img src="/static/images/avatar-default.svg" alt="${senderName}" onerror="this.src='/static/images/avatar-default.svg'">
                    </div>
                    <div class="message-content">
                        <div class="message-bubble">
                            ${messageContent ? `<p>${escapeHtml(messageContent)}</p>` : ''}
                            ${filesHTML}
                        </div>
                        <span class="message-time">
                            ${formatTime(msg.created_at)} ${messageClass === 'sent' ? readIndicator : ''}
                        </span>
                    </div>
                </div>
            `;
        });
    }
    
    messagesArea.innerHTML = html;
}

/**
 * Ajoute de nouveaux messages à la fin (temps réel)
 */
function appendMessages(newMessages) {
    const messagesArea = document.getElementById('messagesArea');
    if (!messagesArea || newMessages.length === 0) return;
    
    // Vérifier si la zone de messages est vide
    const isEmpty = messagesArea.querySelector('.empty-messages');
    if (isEmpty) {
        // Si vide, afficher normalement
        displayMessages(currentMessages);
        return;
    }
    
    // Sauvegarder la position de scroll actuelle
    const wasScrolledToBottom = isScrolledToBottom();
    
    // Créer un fragment pour les nouveaux messages
    const fragment = document.createDocumentFragment();
    const tempDiv = document.createElement('div');
    const messageElements = []; // Stocker les éléments pour animation après ajout
    
    newMessages.forEach(msg => {
        const messageClass = msg.is_from_me ? 'sent' : 'received';
        const readIndicator = msg.is_read ? '<i class="fas fa-check-double read"></i>' : '<i class="fas fa-check"></i>';
        const senderName = msg.sender ? msg.sender.full_name : 'Médecin';
        const messageContent = msg.content || msg.message_text || '';
        
        // Construire le HTML des fichiers attachés
        let filesHTML = '';
        if (msg.files && msg.files.length > 0) {
            filesHTML = '<div class="message-files">';
            msg.files.forEach(file => {
                const icon = getFileIcon(file.mime_type || file.file_extension || '');
                const fileId = file._id || file.id;
                filesHTML += `
                    <a href="/api/messages/files/${fileId}" 
                       class="message-file-item" 
                       download="${escapeHtml(file.original_filename)}"
                       target="_blank"
                       title="Télécharger ${escapeHtml(file.original_filename)}">
                        <div class="message-file-icon">
                            <i class="${icon}"></i>
                        </div>
                        <div class="message-file-info">
                            <div class="message-file-name">${escapeHtml(file.original_filename)}</div>
                            <div class="message-file-size">${file.file_size_formatted || formatFileSize(file.file_size)}</div>
                        </div>
                        <div class="message-file-download">
                            <i class="fas fa-download"></i>
                        </div>
                    </a>
                `;
            });
            filesHTML += '</div>';
        }
        
        // Vérifier si on doit ajouter un séparateur de date
        const messageDate = formatDateDivider(msg.created_at);
        const lastDivider = messagesArea.querySelector('.date-divider:last-of-type span');
        const needsDateDivider = !lastDivider || lastDivider.textContent !== messageDate;
        
        if (needsDateDivider) {
            tempDiv.innerHTML = `<div class="date-divider"><span>${messageDate}</span></div>`;
            const dividerElement = tempDiv.firstChild;
            if (dividerElement) {
                fragment.appendChild(dividerElement);
            }
        }
        
        tempDiv.innerHTML = `
            <div class="message-wrapper ${messageClass}" data-message-id="${msg._id || msg.id}">
                <div class="message-avatar">
                    <img src="/static/images/avatar-default.svg" alt="${senderName}" onerror="this.src='/static/images/avatar-default.svg'">
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        ${messageContent ? `<p>${escapeHtml(messageContent)}</p>` : ''}
                        ${filesHTML}
                    </div>
                    <span class="message-time">
                        ${formatTime(msg.created_at)} ${messageClass === 'sent' ? readIndicator : ''}
                    </span>
                </div>
            </div>
        `;
        
        // Récupérer l'élément et le préparer pour animation
        const messageElement = tempDiv.firstChild;
        if (messageElement && messageElement.nodeType === 1) {
            // Styles initiaux pour animation
            messageElement.style.opacity = '0';
            messageElement.style.transform = 'translateY(10px)';
            messageElement.style.transition = 'all 0.3s ease';
            
            fragment.appendChild(messageElement);
            messageElements.push(messageElement);
        }
    });
    
    // Ajouter les nouveaux messages à la zone
    messagesArea.appendChild(fragment);
    
    // Animer les éléments maintenant qu'ils sont dans le DOM
    requestAnimationFrame(() => {
        messageElements.forEach(el => {
            if (el && el.style) {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }
        });
    });
    
    // Scroller vers le bas si l'utilisateur était déjà en bas
    if (wasScrolledToBottom) {
        scrollToBottom();
    }
}

/**
 * Récupère le timestamp du dernier message dans le cache
 */
function getLastMessageTimestamp() {
    if (!currentMessages || currentMessages.length === 0) {
        return null;
    }
    
    // Le dernier message du cache
    const lastMessage = currentMessages[currentMessages.length - 1];
    return lastMessage.created_at || lastMessage.sent_at;
}

/**
 * Vérifie si l'utilisateur a scrollé en bas de la zone de messages
 */
function isScrolledToBottom() {
    const messagesArea = document.getElementById('messagesArea');
    if (!messagesArea) return false;
    
    const threshold = 100; // 100px de marge
    return messagesArea.scrollHeight - messagesArea.scrollTop - messagesArea.clientHeight < threshold;
}

// ========================================
// SÉLECTION DE CONVERSATION
// ========================================
async function selectConversation(conversationId) {
    try {
        // Quitter l'ancienne conversation si on utilise WebSocket
        if (useWebSocket && window.wsManager && currentConversation && currentConversation !== conversationId) {
            window.wsManager.leaveConversation(currentConversation);
        }
        
        // Réinitialiser le cache des messages quand on change de conversation
        if (currentConversation !== conversationId) {
            currentMessages = [];
            lastMessageId = null;
        }
        
        currentConversation = conversationId;
        
        // Rejoindre la nouvelle conversation si on utilise WebSocket
        if (useWebSocket && window.wsManager && window.wsManager.isConnected()) {
            window.wsManager.joinConversation(conversationId);
        }
        
        // Mettre à jour l'UI
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const selectedItem = document.querySelector(`[data-id="${conversationId}"]`);
        if (selectedItem) {
            selectedItem.classList.add('active');
            selectedItem.classList.remove('unread');
            
            // Supprimer le badge non lu
            const badge = selectedItem.querySelector('.unread-badge');
            if (badge) badge.remove();
        }
        
        // Trouver la conversation
        const conversation = conversations.find(c => c.id === conversationId);
        if (conversation && conversation.other_doctor) {
            const otherDoctor = conversation.other_doctor;
            currentRecipient = otherDoctor.id;
            
            // Nom complet du médecin
            const doctorName = otherDoctor.full_name || 
                              `${otherDoctor.first_name || ''} ${otherDoctor.last_name || ''}`.trim() || 
                              'Médecin';
            
            // Spécialité
            const specialty = otherDoctor.specialty || '';
            
            // Status en ligne (maintenant géré par WebSocket)
            const isOnline = false; // Sera mis à jour par les événements WebSocket
            
            updateChatHeader(doctorName, specialty, isOnline);
        }
        
        // Charger les messages
        await loadMessages(conversationId);
        
        // Activer l'input
        const messageInput = document.getElementById('messageInput');
        const btnSend = document.getElementById('btnSend');
        if (messageInput) {
            messageInput.disabled = false;
            messageInput.placeholder = 'Écrivez votre message...';
        }
        if (btnSend) {
            btnSend.disabled = false;
        }
        
        // Fermer la sidebar sur mobile
        if (window.innerWidth <= 768) {
            const sidebar = document.getElementById('conversationsSidebar');
            if (sidebar) sidebar.classList.remove('show');
        }
        
    } catch (error) {
        console.error('❌ Erreur sélection conversation:', error);
        showNotification('Erreur lors de la sélection', 'error');
    }
}

// ========================================
// MISE À JOUR DU HEADER
// ========================================
function updateChatHeader(name, specialty = '', isOnline = false) {
    const chatHeader = document.querySelector('.chat-header');
    if (!chatHeader) return;
    
    const statusClass = isOnline ? 'online' : 'offline';
    const statusText = isOnline ? 'En ligne' : specialty || 'Hors ligne';
    
    chatHeader.innerHTML = `
        <div class="chat-user-info">
            <div class="chat-avatar">
                <img src="/static/images/avatar-default.svg" alt="${escapeHtml(name)}" onerror="this.src='/static/images/avatar-default.svg'">
            </div>
            <div class="chat-user-details">
                <h3>${escapeHtml(name)}</h3>
                <span class="user-status ${statusClass}">${escapeHtml(statusText)}</span>
            </div>
        </div>
        <div class="chat-actions">
            <button class="btn-icon" title="Rechercher dans la conversation">
                <i class="fas fa-search"></i>
            </button>
            <button class="btn-icon" title="Informations du contact">
                <i class="fas fa-info-circle"></i>
            </button>
            <button class="btn-icon" title="Plus d'options">
                <i class="fas fa-ellipsis-v"></i>
            </button>
        </div>
    `;
}

// ========================================
// ENVOI DE MESSAGE
// ========================================
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    if (!messageInput) return;
    
    const messageText = messageInput.value.trim();
    
    if (!messageText || !currentConversation) return;
    
    try {
        // Désactiver l'input temporairement
        messageInput.disabled = true;
        const btnSend = document.getElementById('btnSend');
        if (btnSend) btnSend.disabled = true;
        
        // Si WebSocket est disponible, l'utiliser
        if (useWebSocket && window.wsManager && window.wsManager.isConnected()) {
            const sent = window.wsManager.sendMessage(currentConversation, messageText);
            
            if (sent) {
                // Le message sera reçu via l'événement 'new_message' et ajouté automatiquement
                // Réinitialiser l'input
                messageInput.value = '';
                messageInput.style.height = 'auto';
                if (btnSend) btnSend.disabled = false;
                messageInput.disabled = false;
                messageInput.focus();
                
                // Arrêter l'indicateur "en train d'écrire"
                if (window.wsManager) {
                    window.wsManager.sendTyping(currentConversation, false);
                }
                return;
            }
        }
        
        // Fallback à l'API REST si WebSocket n'est pas disponible
        const response = await fetch('/api/messages/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: currentConversation,
                content: messageText
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Ajouter le message à l'UI immédiatement
            addMessageToUI({
                is_from_me: true,
                content: messageText,
                created_at: data.created_at || new Date().toISOString(),
                is_read: false
            });
            
            // Réinitialiser l'input
            messageInput.value = '';
            messageInput.style.height = 'auto';
            if (btnSend) btnSend.disabled = false;
            
            // Recharger les conversations pour mettre à jour la prévisualisation
            loadConversations();
        } else {
            throw new Error(data.error || 'Erreur d\'envoi');
        }
    } catch (error) {
        console.error('❌ Erreur envoi message:', error);
        showNotification('Erreur d\'envoi du message', 'error');
        
        const btnSend = document.getElementById('btnSend');
        if (btnSend) btnSend.disabled = false;
    } finally {
        messageInput.disabled = false;
        messageInput.focus();
    }
}

// ========================================
// AJOUT DE MESSAGE À L'UI
// ========================================
function addMessageToUI(message) {
    const messagesArea = document.getElementById('messagesArea');
    if (!messagesArea) return;
    
    // Ajouter au cache des messages
    currentMessages.push(message);
    if (message._id || message.id) {
        lastMessageId = message._id || message.id;
    }
    
    // Supprimer l'état vide si présent
    const emptyState = messagesArea.querySelector('.empty-messages');
    if (emptyState) emptyState.remove();
    
    // Utiliser is_from_me si disponible, sinon déduire de sender_id
    const messageClass = message.is_from_me || message.sender_id === getCurrentUserId() ? 'sent' : 'received';
    
    // Utiliser content ou message_text
    const messageContent = message.content || message.message_text || '';
    
    // Utiliser created_at ou sent_at
    const timestamp = message.created_at || message.sent_at;
    
    // Construire le HTML des fichiers attachés
    let filesHTML = '';
    if (message.files && message.files.length > 0) {
        filesHTML = '<div class="message-files">';
        message.files.forEach(file => {
            const icon = getFileIcon(file.mime_type || file.file_extension || '');
            const fileId = file._id || file.id;
            filesHTML += `
                <a href="/api/messages/files/${fileId}" 
                   class="message-file-item" 
                   download="${escapeHtml(file.original_filename)}"
                   target="_blank"
                   title="Télécharger ${escapeHtml(file.original_filename)}">
                    <div class="message-file-icon">
                        <i class="${icon}"></i>
                    </div>
                    <div class="message-file-info">
                        <div class="message-file-name">${escapeHtml(file.original_filename)}</div>
                        <div class="message-file-size">${file.file_size_formatted || formatFileSize(file.file_size)}</div>
                    </div>
                    <div class="message-file-download">
                        <i class="fas fa-download"></i>
                    </div>
                </a>
            `;
        });
        filesHTML += '</div>';
    }
    
    const messageHTML = `
        <div class="message-wrapper ${messageClass}">
            <div class="message-avatar">
                <img src="/static/images/avatar-default.svg" alt="Avatar" onerror="this.src='/static/images/avatar-default.svg'">
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    ${messageContent ? `<p>${escapeHtml(messageContent)}</p>` : ''}
                    ${filesHTML}
                </div>
                <span class="message-time">
                    ${formatTime(timestamp)} 
                    ${messageClass === 'sent' ? '<i class="fas fa-check"></i>' : ''}
                </span>
            </div>
        </div>
    `;
    
    messagesArea.insertAdjacentHTML('beforeend', messageHTML);
    scrollToBottom();
}

// ========================================
// CONFIGURATION DES ÉVÉNEMENTS
// ========================================
function setupEventListeners() {
    // Input de message
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        // Auto-resize
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            
            // Activer/désactiver le bouton envoyer
            const btnSend = document.getElementById('btnSend');
            if (btnSend) {
                btnSend.disabled = this.value.trim() === '';
            }
        });
        
        // Envoyer avec Entrée
        messageInput.addEventListener('keydown', function(e) {
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
    
    // Bouton nouveau message
    const btnNewMessage = document.getElementById('btnNewMessage');
    if (btnNewMessage) {
        btnNewMessage.addEventListener('click', openNewMessageModal);
    }
    
    // Recherche de conversations
    const searchInput = document.getElementById('searchConversations');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            filterConversations(this.value);
        });
    }
    
    // Filtres
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            filterBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            applyFilter(this.dataset.filter);
        });
    });
}

// ========================================
// POLLING DES MESSAGES
// ========================================
function startMessagePolling() {
    console.log('🔄 Démarrage du polling HTTP (intervalle: 5s)');
    
    // Rafraîchir toutes les 5 secondes
    messagePollingInterval = setInterval(async () => {
        try {
            if (currentConversation) {
                // Mode incrémental : récupérer seulement les nouveaux messages
                await loadMessages(currentConversation, false);
            }
            // Rafraîchir la liste des conversations pour les compteurs
            await loadConversations();
        } catch (error) {
            console.error('Erreur polling:', error);
            // Ne pas arrêter le polling en cas d'erreur
        }
    }, 5000); // 5 secondes
}

function stopMessagePolling() {
    if (messagePollingInterval) {
        console.log('🛑 Arrêt du polling');
        clearInterval(messagePollingInterval);
        messagePollingInterval = null;
    }
}

// ========================================
// UTILITAIRES
// ========================================
function formatTime(timestamp) {
    if (!timestamp) return '';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    // Moins d'une minute
    if (diff < 60000) {
        return 'À l\'instant';
    }
    
    // Moins d'une heure
    if (diff < 3600000) {
        const minutes = Math.floor(diff / 60000);
        return `Il y a ${minutes} min`;
    }
    
    // Aujourd'hui
    if (date.toDateString() === now.toDateString()) {
        return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
    }
    
    // Hier
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (date.toDateString() === yesterday.toDateString()) {
        return 'Hier';
    }
    
    // Cette semaine
    if (diff < 604800000) {
        const days = Math.floor(diff / 86400000);
        return `Il y a ${days} jour${days > 1 ? 's' : ''}`;
    }
    
    // Date complète
    return date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' });
}

function groupMessagesByDate(messages) {
    const groups = {};
    const now = new Date();
    
    messages.forEach(msg => {
        // L'API utilise created_at
        const timestamp = msg.created_at || msg.sent_at;
        if (!timestamp) return;
        
        const date = new Date(timestamp);
        let label;
        
        if (date.toDateString() === now.toDateString()) {
            label = 'Aujourd\'hui';
        } else {
            const yesterday = new Date(now);
            yesterday.setDate(yesterday.getDate() - 1);
            if (date.toDateString() === yesterday.toDateString()) {
                label = 'Hier';
            } else {
                label = date.toLocaleDateString('fr-FR', { 
                    day: 'numeric', 
                    month: 'long', 
                    year: 'numeric' 
                });
            }
        }
        
        if (!groups[label]) {
            groups[label] = [];
        }
        groups[label].push(msg);
    });
    
    return groups;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    const messagesArea = document.getElementById('messagesArea');
    if (messagesArea) {
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }
}

function isScrolledToBottom() {
    const messagesArea = document.getElementById('messagesArea');
    if (!messagesArea) return true;
    
    const threshold = 100; // pixels de tolérance
    return messagesArea.scrollHeight - messagesArea.scrollTop - messagesArea.clientHeight < threshold;
}

function formatDateDivider(timestamp) {
    if (!timestamp) return 'Aujourd\'hui';
    
    const date = new Date(timestamp);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    // Réinitialiser les heures pour la comparaison
    today.setHours(0, 0, 0, 0);
    yesterday.setHours(0, 0, 0, 0);
    date.setHours(0, 0, 0, 0);
    
    if (date.getTime() === today.getTime()) {
        return 'Aujourd\'hui';
    } else if (date.getTime() === yesterday.getTime()) {
        return 'Hier';
    } else {
        return date.toLocaleDateString('fr-FR', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
    }
}

function getCurrentUserId() {
    // Récupérer l'ID depuis la session ou le DOM
    const userAvatar = document.querySelector('.nav-user');
    return userAvatar?.dataset?.userId || 'current_user';
}

function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // TODO: Implémenter un système de notification visuelle
}

function showLoadingState() {
    const listContainer = document.getElementById('conversationsList');
    if (listContainer) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-spinner fa-spin fa-3x"></i>
                <h3>Chargement...</h3>
                <p>Récupération des conversations</p>
            </div>
        `;
    }
}

function showErrorState(message) {
    const listContainer = document.getElementById('conversationsList');
    if (listContainer) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle fa-3x"></i>
                <h3>Erreur</h3>
                <p>${message}</p>
                <button class="btn-primary" onclick="initializeMessaging()">
                    <i class="fas fa-redo"></i> Réessayer
                </button>
            </div>
        `;
    }
}

function filterConversations(searchTerm) {
    const items = document.querySelectorAll('.conversation-item');
    const term = searchTerm.toLowerCase();
    
    items.forEach(item => {
        const name = item.querySelector('h4').textContent.toLowerCase();
        const preview = item.querySelector('.conversation-preview p').textContent.toLowerCase();
        
        if (name.includes(term) || preview.includes(term)) {
            item.style.display = 'flex';
        } else {
            item.style.display = 'none';
        }
    });
}

function applyFilter(filter) {
    const items = document.querySelectorAll('.conversation-item');
    
    items.forEach(item => {
        switch(filter) {
            case 'all':
                item.style.display = 'flex';
                break;
            case 'unread':
                item.style.display = item.classList.contains('unread') ? 'flex' : 'none';
                break;
            case 'important':
                // TODO: Implémenter la logique des messages importants
                item.style.display = 'flex';
                break;
        }
    });
}

function openNewMessageModal() {
    console.log('Ouvrir modal nouvelle conversation');
    // TODO: Implémenter le modal de nouvelle conversation
    showNotification('Fonctionnalité en cours de développement', 'info');
}

// ========================================
// NETTOYAGE
// ========================================
window.addEventListener('beforeunload', function() {
    stopMessagePolling();
});

console.log('✅ Script de messagerie moderne chargé');

// ========================================
// MODAL: NOUVELLE CONVERSATION
// ========================================

let filteredDoctors = [];

/**
 * Ouvre le modal de nouvelle conversation et charge les médecins
 */
function openNewConversationModal() {
    console.log('📬 Ouvrir modal nouvelle conversation');
    const modal = document.getElementById('newConversationModal');
    if (!modal) return;
    
    modal.classList.add('show');
    document.body.style.overflow = 'hidden'; // Désactiver le scroll du body
    
    // Charger la liste des médecins
    loadDoctorsList();
    
    // Focus sur le champ de recherche
    setTimeout(() => {
        const searchInput = document.getElementById('searchDoctors');
        if (searchInput) searchInput.focus();
    }, 300);
}

/**
 * Ferme le modal de nouvelle conversation
 */
function closeNewConversationModal() {
    const modal = document.getElementById('newConversationModal');
    if (!modal) return;
    
    modal.classList.remove('show');
    document.body.style.overflow = ''; // Réactiver le scroll du body
    
    // Réinitialiser la recherche
    const searchInput = document.getElementById('searchDoctors');
    if (searchInput) searchInput.value = '';
    allDoctors = [];
    filteredDoctors = [];
}

/**
 * Charge la liste de tous les médecins disponibles
 */
async function loadDoctorsList() {
    const doctorsList = document.getElementById('doctorsList');
    if (!doctorsList) return;
    
    try {
        // Afficher le loading
        doctorsList.innerHTML = `
            <div class="loading-doctors">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Chargement des médecins...</p>
            </div>
        `;
        
        const response = await fetch('/api/messages/doctors');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Erreur lors du chargement des médecins');
        }
        
        if (data.success && data.doctors) {
            allDoctors = data.doctors;
            filteredDoctors = [...allDoctors];
            displayDoctorsList(filteredDoctors);
        } else {
            throw new Error('Format de réponse invalide');
        }
        
    } catch (error) {
        console.error('❌ Erreur lors du chargement des médecins:', error);
        doctorsList.innerHTML = `
            <div class="empty-doctors">
                <i class="fas fa-exclamation-circle"></i>
                <h4>Erreur de chargement</h4>
                <p>${error.message}</p>
            </div>
        `;
    }
}

/**
 * Affiche la liste des médecins
 */
function displayDoctorsList(doctors) {
    const doctorsList = document.getElementById('doctorsList');
    if (!doctorsList) return;
    
    if (doctors.length === 0) {
        doctorsList.innerHTML = `
            <div class="empty-doctors">
                <i class="fas fa-user-md"></i>
                <h4>Aucun médecin trouvé</h4>
                <p>Aucun médecin ne correspond à votre recherche</p>
            </div>
        `;
        return;
    }
    
    doctorsList.innerHTML = doctors.map(doctor => {
        const avatarUrl = doctor.avatar || '/static/images/avatar-default.svg';
        const specialty = doctor.specialty || doctor.specialite || 'Médecin généraliste';
        const fullName = doctor.full_name || doctor.name || 'Médecin';
        
        return `
            <div class="doctor-item" onclick="startConversationWithDoctor('${doctor._id || doctor.id}')">
                <img src="${avatarUrl}" alt="${fullName}" class="avatar" onerror="this.src='/static/images/avatar-default.svg'">
                <div class="doctor-info">
                    <div class="doctor-name">${fullName}</div>
                    <div class="doctor-specialty">${specialty}</div>
                </div>
                <button class="btn-start-chat" onclick="event.stopPropagation(); startConversationWithDoctor('${doctor._id || doctor.id}')">
                    <i class="fas fa-comment-medical"></i>
                </button>
            </div>
        `;
    }).join('');
}

/**
 * Filtre les médecins selon la recherche
 */
function filterDoctors(searchTerm) {
    const term = searchTerm.toLowerCase().trim();
    
    if (!term) {
        filteredDoctors = [...allDoctors];
    } else {
        filteredDoctors = allDoctors.filter(doctor => {
            const fullName = (doctor.full_name || doctor.name || '').toLowerCase();
            const specialty = (doctor.specialty || doctor.specialite || '').toLowerCase();
            return fullName.includes(term) || specialty.includes(term);
        });
    }
    
    displayDoctorsList(filteredDoctors);
}

/**
 * Démarre une conversation avec un médecin
 */
async function startConversationWithDoctor(doctorId) {
    console.log('💬 Démarrer conversation avec médecin:', doctorId);
    
    try {
        // Fermer le modal
        closeNewConversationModal();
        
        // Afficher un état de chargement
        showLoadingState();
        
        // Créer ou récupérer la conversation
        const response = await fetch('/api/messages/conversations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ recipient_id: doctorId })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Erreur lors de la création de la conversation');
        }
        
        if (data.success && data.conversation_id) {
            // Recharger les conversations
            await loadConversations();
            
            // Ouvrir la conversation
            await loadConversation(data.conversation_id);
            
            showNotification('Conversation ouverte avec succès', 'success');
        } else {
            throw new Error('Format de réponse invalide');
        }
        
    } catch (error) {
        console.error('❌ Erreur lors de la création de la conversation:', error);
        showNotification('Erreur lors de la création de la conversation: ' + error.message, 'error');
    }
}

// Event listener pour la recherche de médecins
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchDoctors');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            filterDoctors(e.target.value);
        });
    }
    
    // Event listener pour le bouton nouvelle conversation
    const btnNewMessage = document.getElementById('btnNewMessage');
    if (btnNewMessage) {
        btnNewMessage.addEventListener('click', openNewConversationModal);
    }
    
    // Fermer le modal avec Escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const modal = document.getElementById('newConversationModal');
            if (modal && modal.classList.contains('show')) {
                closeNewConversationModal();
            }
        }
    });
    
    // Fermer le modal en cliquant sur l'overlay
    const modalOverlay = document.getElementById('newConversationModal');
    if (modalOverlay) {
        modalOverlay.addEventListener('click', function(e) {
            if (e.target === modalOverlay) {
                closeNewConversationModal();
            }
        });
    }
});

// ========================================
// FONCTIONS UTILITAIRES WEBSOCKET
// ========================================

/**
 * Afficher l'indicateur "en train d'écrire"
 */
function showTypingIndicator(isTyping) {
    const messagesArea = document.getElementById('messagesArea');
    if (!messagesArea) return;
    
    let typingIndicator = document.getElementById('typingIndicator');
    
    if (isTyping) {
        if (!typingIndicator) {
            typingIndicator = document.createElement('div');
            typingIndicator.id = 'typingIndicator';
            typingIndicator.className = 'typing-indicator';
            typingIndicator.innerHTML = `
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span class="typing-text">En train d'écrire...</span>
            `;
            messagesArea.appendChild(typingIndicator);
            scrollToBottom();
        }
    } else {
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
}

/**
 * Mettre à jour les indicateurs de lecture
 */
function updateReadIndicators() {
    const messages = document.querySelectorAll('.message-wrapper.sent');
    messages.forEach(msg => {
        const readIcon = msg.querySelector('.fa-check-double');
        if (readIcon) {
            readIcon.classList.add('read');
        }
    });
}

/**
 * Mettre à jour le statut en ligne d'un médecin
 */
function updateDoctorOnlineStatus(doctorId, isOnline) {
    // Mettre à jour dans l'en-tête du chat si c'est le médecin actuel
    if (currentRecipient === doctorId) {
        const statusIndicator = document.querySelector('.chat-header .status-indicator');
        if (statusIndicator) {
            if (isOnline) {
                statusIndicator.classList.add('online');
                statusIndicator.classList.remove('offline');
            } else {
                statusIndicator.classList.remove('online');
                statusIndicator.classList.add('offline');
            }
        }
        
        const statusText = document.querySelector('.chat-header .status-text');
        if (statusText) {
            statusText.textContent = isOnline ? 'En ligne' : 'Hors ligne';
        }
    }
    
    // Mettre à jour dans la liste des conversations
    const conversationItem = document.querySelector(`.conversation-item[data-doctor-id="${doctorId}"]`);
    if (conversationItem) {
        const avatar = conversationItem.querySelector('.conversation-avatar');
        if (avatar) {
            const badge = avatar.querySelector('.online-badge');
            if (isOnline) {
                if (!badge) {
                    const newBadge = document.createElement('span');
                    newBadge.className = 'online-badge';
                    avatar.appendChild(newBadge);
                }
            } else {
                if (badge) {
                    badge.remove();
                }
            }
        }
    }
}

/**
 * Mettre à jour la preview d'une conversation
 */
function updateConversationPreview(conversationId, content, timestamp) {
    const conversationItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
    if (!conversationItem) return;
    
    // Mettre à jour le dernier message
    const messagePreview = conversationItem.querySelector('.message-preview');
    if (messagePreview) {
        messagePreview.textContent = content.substring(0, 50) + (content.length > 50 ? '...' : '');
    }
    
    // Mettre à jour l'heure
    const timeElement = conversationItem.querySelector('.message-time');
    if (timeElement) {
        timeElement.textContent = formatTime(timestamp);
    }
    
    // Déplacer la conversation en haut de la liste
    const conversationsList = document.getElementById('conversationsList');
    if (conversationsList) {
        conversationsList.insertBefore(conversationItem, conversationsList.firstChild);
    }
}

/**
 * Récupérer l'ID du médecin connecté
 */
function getCurrentDoctorId() {
    // Essayer de récupérer depuis l'attribut data
    const doctorElement = document.querySelector('[data-doctor-id]');
    if (doctorElement) {
        return doctorElement.dataset.doctorId;
    }
    
    // Fallback: récupérer depuis la session ou autre
    return window.currentDoctorId || null;
}

// ========================================
// NAVIGATION MOBILE & RESPONSIVE
// ========================================

/**
 * Toggle sidebar sur mobile
 */
function toggleSidebar() {
    const sidebar = document.getElementById('conversationsSidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    
    if (sidebar && overlay) {
        sidebar.classList.toggle('show');
        overlay.classList.toggle('show');
    }
}

/**
 * Fermer la sidebar (overlay click)
 */
function closeSidebar() {
    const sidebar = document.getElementById('conversationsSidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    
    if (sidebar && overlay) {
        sidebar.classList.remove('show');
        overlay.classList.remove('show');
    }
}

/**
 * Ouvrir le chat sur mobile (cache la sidebar)
 */
function openChatMobile() {
    const chatArea = document.querySelector('.chat-area');
    const sidebar = document.getElementById('conversationsSidebar');
    
    if (window.innerWidth <= 968) {
        if (chatArea) {
            chatArea.classList.add('active');
        }
        if (sidebar) {
            sidebar.classList.add('hidden-mobile');
        }
    }
}

/**
 * Retourner à la liste des conversations (mobile)
 */
function backToConversations() {
    const chatArea = document.querySelector('.chat-area');
    const sidebar = document.getElementById('conversationsSidebar');
    
    if (chatArea) {
        chatArea.classList.remove('active');
    }
    if (sidebar) {
        sidebar.classList.remove('hidden-mobile');
    }
    
    // Optionnel: désélectionner la conversation
    currentConversation = null;
    currentRecipient = null;
}

/**
 * Gérer le redimensionnement de la fenêtre
 */
function handleWindowResize() {
    const chatArea = document.querySelector('.chat-area');
    const sidebar = document.getElementById('conversationsSidebar');
    
    // Réinitialiser les classes mobiles si on repasse en desktop
    if (window.innerWidth > 968) {
        if (chatArea) {
            chatArea.classList.remove('active');
        }
        if (sidebar) {
            sidebar.classList.remove('show', 'hidden-mobile');
        }
        closeSidebar();
    }
}

// Écouter le redimensionnement
let resizeTimeout;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(handleWindowResize, 150);
});

/**
 * Ajouter le bouton retour dans le header du chat (mobile)
 */
function addMobileBackButton() {
    const chatHeader = document.querySelector('.chat-header');
    if (!chatHeader) return;
    
    // Vérifier si le bouton existe déjà
    if (chatHeader.querySelector('.btn-back-mobile')) return;
    
    // Créer le bouton
    const backButton = document.createElement('button');
    backButton.className = 'btn-back-mobile';
    backButton.innerHTML = '<i class="fas fa-arrow-left"></i>';
    backButton.onclick = backToConversations;
    backButton.title = 'Retour aux conversations';
    
    // Insérer au début du header
    const userInfo = chatHeader.querySelector('.chat-user-info');
    if (userInfo) {
        chatHeader.insertBefore(backButton, userInfo);
    }
}

/**
 * Initialiser les événements responsive
 */
function initializeResponsiveEvents() {
    // Overlay click pour fermer sidebar
    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) {
        overlay.addEventListener('click', closeSidebar);
    }
    
    // FAB pour nouvelle conversation
    const fab = document.querySelector('.fab');
    if (fab) {
        fab.addEventListener('click', function() {
            openNewConversationModal();
            closeSidebar();
        });
    }
    
    // Ajouter le bouton retour mobile
    addMobileBackButton();
    
    // Gérer les clics sur les conversations pour mobile
    document.addEventListener('click', function(e) {
        const conversationItem = e.target.closest('.conversation-item');
        if (conversationItem && window.innerWidth <= 968) {
            // Sur mobile, ouvrir le chat et cacher la sidebar
            setTimeout(() => {
                openChatMobile();
            }, 100);
        }
    });
}

// Initialiser au chargement
document.addEventListener('DOMContentLoaded', function() {
    initializeResponsiveEvents();
});

// ========================================
// GESTION UPLOAD DE FICHIERS
// ========================================

/**
 * Variables globales pour l'upload
 */
let selectedFiles = [];
let uploadedFiles = [];

/**
 * Ouvrir le sélecteur de fichiers
 */
function attachFile() {
    console.log('📎 Joindre fichier');
    
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*,.pdf,.doc,.docx,.xls,.xlsx,.txt,.zip,.rar';
    input.multiple = true;
    input.style.display = 'none';
    
    input.onchange = function(e) {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            handleFileSelection(files);
        }
    };
    
    document.body.appendChild(input);
    input.click();
    document.body.removeChild(input);
}

/**
 * Gérer la sélection de fichiers
 */
function handleFileSelection(files) {
    // Limites de validation
    const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB
    const MAX_FILES = 10;
    
    // Filtrer et valider les fichiers
    const validFiles = [];
    const errors = [];
    
    for (const file of files) {
        // Vérifier la taille
        if (file.size > MAX_FILE_SIZE) {
            errors.push(`${file.name}: Taille maximale 50MB`);
            continue;
        }
        
        // Vérifier le nombre total
        if (selectedFiles.length + validFiles.length >= MAX_FILES) {
            errors.push(`Maximum ${MAX_FILES} fichiers autorisés`);
            break;
        }
        
        validFiles.push(file);
    }
    
    // Afficher les erreurs
    if (errors.length > 0) {
        showNotification(errors.join('\n'), 'error');
    }
    
    // Ajouter les fichiers valides
    if (validFiles.length > 0) {
        selectedFiles.push(...validFiles);
        displayFilePreview();
        showNotification(`${validFiles.length} fichier(s) sélectionné(s)`, 'success');
    }
}

/**
 * Afficher la preview des fichiers sélectionnés
 */
function displayFilePreview() {
    // Chercher ou créer le conteneur de preview
    let previewContainer = document.getElementById('filePreviewContainer');
    
    if (!previewContainer) {
        const inputContainer = document.querySelector('.message-input-container');
        if (!inputContainer) return;
        
        previewContainer = document.createElement('div');
        previewContainer.id = 'filePreviewContainer';
        previewContainer.className = 'file-preview-container';
        
        // Insérer AVANT le conteneur input (pas dedans)
        inputContainer.parentElement.insertBefore(previewContainer, inputContainer);
    }
    
    // Vider et remplir
    previewContainer.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const fileItem = createFilePreviewItem(file, index);
        previewContainer.appendChild(fileItem);
    });
    
    // Afficher/cacher le conteneur
    previewContainer.style.display = selectedFiles.length > 0 ? 'flex' : 'none';
}

/**
 * Créer un élément de preview de fichier
 */
function createFilePreviewItem(file, index) {
    const item = document.createElement('div');
    item.className = 'file-preview-item';
    
    // Icône selon le type
    const icon = getFileIcon(file.type);
    
    // Taille formatée
    const size = formatFileSize(file.size);
    
    item.innerHTML = `
        <div class="file-preview-icon">
            <i class="${icon}"></i>
        </div>
        <div class="file-preview-info">
            <div class="file-preview-name">${truncateFileName(file.name, 20)}</div>
            <div class="file-preview-size">${size}</div>
        </div>
        <button class="file-preview-remove" onclick="removeFile(${index})" title="Supprimer">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    return item;
}

/**
 * Obtenir l'icône selon le type de fichier
 */
function getFileIcon(mimeType) {
    if (mimeType.startsWith('image/')) return 'fas fa-image';
    if (mimeType.includes('pdf')) return 'fas fa-file-pdf';
    if (mimeType.includes('word') || mimeType.includes('document')) return 'fas fa-file-word';
    if (mimeType.includes('excel') || mimeType.includes('spreadsheet')) return 'fas fa-file-excel';
    if (mimeType.includes('zip') || mimeType.includes('rar')) return 'fas fa-file-archive';
    return 'fas fa-file';
}

/**
 * Formater la taille de fichier
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Tronquer un nom de fichier
 */
function truncateFileName(name, maxLength) {
    if (name.length <= maxLength) return name;
    const ext = name.split('.').pop();
    const nameWithoutExt = name.substring(0, name.lastIndexOf('.'));
    const truncated = nameWithoutExt.substring(0, maxLength - ext.length - 4) + '...';
    return truncated + '.' + ext;
}

/**
 * Retirer un fichier de la sélection
 */
function removeFile(index) {
    selectedFiles.splice(index, 1);
    displayFilePreview();
    
    if (selectedFiles.length === 0) {
        const previewContainer = document.getElementById('filePreviewContainer');
        if (previewContainer) {
            previewContainer.style.display = 'none';
        }
    }
}

/**
 * Upload les fichiers sélectionnés
 */
async function uploadSelectedFiles() {
    if (selectedFiles.length === 0 || !currentConversation) {
        return [];
    }
    
    const uploadedFileIds = [];
    
    for (const file of selectedFiles) {
        try {
            const fileData = await uploadFile(file, currentConversation);
            if (fileData && fileData._id) {
                uploadedFileIds.push(fileData);
            }
        } catch (error) {
            console.error('Erreur upload fichier:', error);
            showNotification(`Erreur upload ${file.name}`, 'error');
        }
    }
    
    return uploadedFileIds;
}

/**
 * Upload un seul fichier
 */
async function uploadFile(file, conversationId) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conversation_id', conversationId);
    
    try {
        const response = await fetch('/api/messages/upload-file', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            return result.file;
        } else {
            throw new Error(result.error || 'Erreur upload');
        }
    } catch (error) {
        console.error('Erreur upload:', error);
        throw error;
    }
}

/**
 * Modifier la fonction sendMessage pour inclure les fichiers
 */
const originalSendMessage = window.sendMessage;
window.sendMessage = async function() {
    const messageInput = document.getElementById('messageInput');
    const content = messageInput.value.trim();
    
    // Si pas de message et pas de fichiers, ne rien faire
    if (!content && selectedFiles.length === 0) {
        return;
    }
    
    if (!currentConversation || !currentRecipient) {
        showNotification('Veuillez sélectionner une conversation', 'error');
        return;
    }
    
    try {
        // Désactiver le bouton d'envoi
        const sendButton = document.getElementById('btnSend');
        if (sendButton) {
            sendButton.disabled = true;
            sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        }
        
        // Upload les fichiers d'abord
        let uploadedFileIds = [];
        if (selectedFiles.length > 0) {
            uploadedFileIds = await uploadSelectedFiles();
        }
        
        // Préparer les données du message
        const messageData = {
            conversation_id: currentConversation,
            recipient_id: currentRecipient,
            content: content || '📎 Fichier(s) joint(s)',
            file_ids: uploadedFileIds.map(f => f._id || f.id)
        };
        
        // Envoyer le message via WebSocket ou HTTP
        if (useWebSocket && window.wsManager && window.wsManager.isConnected) {
            window.wsManager.sendMessage(messageData);
        } else {
            const response = await fetch('/api/messages/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(messageData)
            });
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || 'Erreur envoi message');
            }
            
            // Ajouter le message localement
            if (result.message) {
                appendMessages([result.message]);
            }
        }
        
        // Nettoyer
        messageInput.value = '';
        selectedFiles = [];
        displayFilePreview();
        
        // Réactiver le bouton
        if (sendButton) {
            sendButton.disabled = false;
            sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
        
    } catch (error) {
        console.error('Erreur envoi message:', error);
        showNotification('Erreur lors de l\'envoi', 'error');
        
        // Réactiver le bouton
        const sendButton = document.getElementById('btnSend');
        if (sendButton) {
            sendButton.disabled = false;
            sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }
};

/**
 * Afficher les fichiers joints dans un message
 */
function displayMessageFiles(message, messageElement) {
    if (!message.files || message.files.length === 0) return;
    
    const filesContainer = document.createElement('div');
    filesContainer.className = 'message-files';
    
    message.files.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'message-file-item';
        
        const icon = getFileIcon(file.mime_type);
        const fileSize = file.file_size_formatted || formatFileSize(file.file_size);
        
        fileItem.innerHTML = `
            <div class="message-file-icon">
                <i class="${icon}"></i>
            </div>
            <div class="message-file-info">
                <div class="message-file-name">${file.original_filename}</div>
                <div class="message-file-size">${fileSize}</div>
            </div>
            <button class="message-file-download" onclick="downloadFile('${file._id || file.id}', '${file.original_filename}')" title="Télécharger">
                <i class="fas fa-download"></i>
            </button>
        `;
        
        filesContainer.appendChild(fileItem);
    });
    
    messageElement.appendChild(filesContainer);
}

/**
 * Télécharger un fichier
 */
async function downloadFile(fileId, filename) {
    try {
        console.log(`📥 Téléchargement du fichier: ${filename}`);
        
        // Afficher un indicateur de chargement
        showNotification('Téléchargement en cours...', 'info');
        
        // Récupérer le fichier
        const response = await fetch(`/api/messages/files/${fileId}`);
        
        if (!response.ok) {
            throw new Error('Erreur lors du téléchargement');
        }
        
        // Convertir en blob
        const blob = await response.blob();
        
        // Créer un lien de téléchargement temporaire
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        
        // Ajouter au DOM, cliquer, et retirer
        document.body.appendChild(a);
        a.click();
        
        // Nettoyer
        setTimeout(() => {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 100);
        
        showNotification(`Fichier téléchargé: ${filename}`, 'success');
        
    } catch (error) {
        console.error('Erreur téléchargement:', error);
        showNotification('Erreur lors du téléchargement du fichier', 'error');
    }
}

/**
 * Prévisualiser un fichier (images uniquement)
 */
function previewFile(fileId, filename, mimeType) {
    if (!mimeType.startsWith('image/')) {
        downloadFile(fileId, filename);
        return;
    }
    
    // Créer une modal de prévisualisation pour les images
    const modal = document.createElement('div');
    modal.className = 'file-preview-modal';
    modal.innerHTML = `
        <div class="file-preview-modal-content">
            <div class="file-preview-modal-header">
                <h3>${filename}</h3>
                <button onclick="this.closest('.file-preview-modal').remove()" class="btn-close-modal">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="file-preview-modal-body">
                <img src="/api/messages/files/${fileId}" alt="${filename}">
            </div>
            <div class="file-preview-modal-footer">
                <button onclick="downloadFile('${fileId}', '${filename}')" class="btn-download">
                    <i class="fas fa-download"></i> Télécharger
                </button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Fermer en cliquant sur le fond
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// ========================================
// EMOJI PICKER
// ========================================

// Liste des emojis par catégorie
const emojiData = {
    frequent: ['😊', '😂', '❤️', '👍', '🙏', '👏', '🎉', '💯'],
    smileys: [
        '😀', '😃', '😄', '😁', '😆', '😅', '😂', '🤣', '😊', '😇',
        '🙂', '🙃', '😉', '😌', '😍', '🥰', '😘', '😗', '😙', '😚',
        '😋', '😛', '😝', '😜', '🤪', '🤨', '🧐', '🤓', '😎', '🤩',
        '🥳', '😏', '😒', '😞', '😔', '😟', '😕', '🙁', '☹️', '😣',
        '😖', '😫', '😩', '🥺', '😢', '😭', '😤', '😠', '😡', '🤬',
        '🤯', '😳', '🥵', '🥶', '😱', '😨', '😰', '😥', '😓', '🤗'
    ],
    gestures: [
        '👋', '🤚', '🖐️', '✋', '🖖', '👌', '🤌', '🤏', '✌️', '🤞',
        '🤟', '🤘', '🤙', '👈', '👉', '👆', '🖕', '👇', '☝️', '👍',
        '👎', '✊', '👊', '🤛', '🤜', '👏', '🙌', '👐', '🤲', '🤝',
        '🙏', '✍️', '💪', '🦾', '🦿', '🦵', '🦶', '👂', '🦻', '👃'
    ],
    medical: [
        '⚕️', '🏥', '🩺', '💉', '💊', '🩹', '🩼', '🧬', '🔬', '🧪',
        '🧫', '🦠', '🧠', '🫀', '🫁', '🦷', '🦴', '👁️', '👀', '🧑‍⚕️',
        '👨‍⚕️', '👩‍⚕️', '🚑', '🏨', '🔬', '🩺', '💉'
    ],
    hearts: [
        '❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '🤎', '💔',
        '❤️‍🔥', '❤️‍🩹', '💕', '💞', '💓', '💗', '💖', '💘', '💝', '💟'
    ],
    objects: [
        '📱', '💻', '⌨️', '🖥️', '🖨️', '🖱️', '📞', '📟', '📠', '📺',
        '📷', '📸', '📹', '🎥', '📽️', '📝', '📄', '📃', '📑', '📊',
        '📈', '📉', '🗒️', '📅', '📆', '🗓️', '📇', '🗃️', '📋', '📁',
        '📂', '🗂️', '📌', '📍', '📎', '🖇️', '📏', '📐', '✂️', '🔒'
    ],
    symbols: [
        '✅', '❌', '⭕', '✔️', '✖️', '➕', '➖', '➗', '✳️', '✴️',
        '❇️', '‼️', '⁉️', '❓', '❔', '❗', '〰️', '⚠️', '🚫', '🔞',
        '📵', '🚭', '❎', '✴️', '🆚', '📶', '🎦', '🔅', '🔆', '📳'
    ]
};

const emojiCategories = {
    frequent: { icon: '🕐', label: 'Récents' },
    smileys: { icon: '😊', label: 'Smileys' },
    gestures: { icon: '👋', label: 'Gestes' },
    medical: { icon: '⚕️', label: 'Médical' },
    hearts: { icon: '❤️', label: 'Coeurs' },
    objects: { icon: '📁', label: 'Objets' },
    symbols: { icon: '✅', label: 'Symboles' }
};

let currentEmojiCategory = 'frequent';
let emojiPickerElement = null;
let frequentEmojis = [];

/**
 * Initialiser le picker d'emoji
 */
function initializeEmojiPicker() {
    // Charger les emojis fréquents depuis localStorage
    const stored = localStorage.getItem('frequentEmojis');
    if (stored) {
        try {
            frequentEmojis = JSON.parse(stored);
        } catch (e) {
            frequentEmojis = [...emojiData.frequent];
        }
    } else {
        frequentEmojis = [...emojiData.frequent];
    }
    
    // Créer l'élément picker s'il n'existe pas
    if (!emojiPickerElement) {
        createEmojiPicker();
    }
}

/**
 * Créer l'élément HTML du picker
 */
function createEmojiPicker() {
    const picker = document.createElement('div');
    picker.className = 'emoji-picker';
    picker.id = 'emojiPicker';
    
    picker.innerHTML = `
        <div class="emoji-picker-header">
            <h4>Emojis</h4>
            <button class="btn-close-emoji" onclick="closeEmojiPicker()">
                <i class="fas fa-times"></i>
            </button>
        </div>
        
        <div class="emoji-categories" id="emojiCategories">
            ${Object.entries(emojiCategories).map(([key, cat]) => `
                <button class="emoji-category-btn ${key === 'frequent' ? 'active' : ''}" 
                        data-category="${key}" 
                        onclick="selectEmojiCategory('${key}')"
                        title="${cat.label}">
                    ${cat.icon}
                </button>
            `).join('')}
        </div>
        
        <div class="emoji-search">
            <input type="text" 
                   id="emojiSearchInput" 
                   placeholder="Rechercher un emoji..." 
                   oninput="searchEmojis(this.value)">
        </div>
        
        <div class="emoji-grid" id="emojiGrid">
            <!-- Les emojis seront insérés ici -->
        </div>
    `;
    
    // Ajouter au wrapper de l'input
    const inputWrapper = document.querySelector('.message-input-wrapper');
    if (inputWrapper) {
        inputWrapper.appendChild(picker);
        emojiPickerElement = picker;
        displayEmojis(currentEmojiCategory);
    }
}

/**
 * Toggle le picker d'emoji
 */
function toggleEmojiPicker() {
    if (!emojiPickerElement) {
        initializeEmojiPicker();
    }
    
    const picker = document.getElementById('emojiPicker');
    if (picker) {
        picker.classList.toggle('show');
        
        // Focus sur la recherche si ouvert
        if (picker.classList.contains('show')) {
            setTimeout(() => {
                const searchInput = document.getElementById('emojiSearchInput');
                if (searchInput) searchInput.focus();
            }, 100);
        }
    }
}

/**
 * Fermer le picker
 */
function closeEmojiPicker() {
    const picker = document.getElementById('emojiPicker');
    if (picker) {
        picker.classList.remove('show');
    }
}

/**
 * Sélectionner une catégorie
 */
function selectEmojiCategory(category) {
    currentEmojiCategory = category;
    
    // Mettre à jour les boutons actifs
    document.querySelectorAll('.emoji-category-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.category === category);
    });
    
    // Afficher les emojis de cette catégorie
    displayEmojis(category);
}

/**
 * Afficher les emojis d'une catégorie
 */
function displayEmojis(category) {
    const grid = document.getElementById('emojiGrid');
    if (!grid) return;
    
    let emojis = [];
    
    if (category === 'frequent') {
        emojis = frequentEmojis.length > 0 ? frequentEmojis : emojiData.frequent;
    } else {
        emojis = emojiData[category] || [];
    }
    
    if (emojis.length === 0) {
        grid.innerHTML = `
            <div class="emoji-no-results">
                <i class="fas fa-search"></i>
                <p>Aucun emoji trouvé</p>
            </div>
        `;
        return;
    }
    
    grid.innerHTML = emojis.map(emoji => `
        <button class="emoji-item" onclick="insertEmoji('${emoji}')" title="${emoji}">
            ${emoji}
        </button>
    `).join('');
}

/**
 * Rechercher des emojis
 */
function searchEmojis(query) {
    const grid = document.getElementById('emojiGrid');
    if (!grid) return;
    
    if (!query.trim()) {
        displayEmojis(currentEmojiCategory);
        return;
    }
    
    // Rechercher dans toutes les catégories
    const allEmojis = Object.values(emojiData).flat();
    const results = allEmojis.filter(emoji => {
        // Recherche simple (on pourrait améliorer avec des mots-clés)
        return emoji.includes(query.toLowerCase());
    });
    
    if (results.length === 0) {
        grid.innerHTML = `
            <div class="emoji-no-results">
                <i class="fas fa-search"></i>
                <p>Aucun emoji trouvé pour "${query}"</p>
            </div>
        `;
        return;
    }
    
    grid.innerHTML = results.slice(0, 64).map(emoji => `
        <button class="emoji-item" onclick="insertEmoji('${emoji}')" title="${emoji}">
            ${emoji}
        </button>
    `).join('');
}

/**
 * Insérer un emoji dans l'input
 */
function insertEmoji(emoji) {
    const messageInput = document.getElementById('messageInput');
    if (!messageInput) return;
    
    // Insérer à la position du curseur
    const start = messageInput.selectionStart;
    const end = messageInput.selectionEnd;
    const text = messageInput.value;
    
    messageInput.value = text.substring(0, start) + emoji + text.substring(end);
    
    // Replacer le curseur après l'emoji
    const newPosition = start + emoji.length;
    messageInput.setSelectionRange(newPosition, newPosition);
    messageInput.focus();
    
    // Ajouter aux emojis fréquents
    addToFrequentEmojis(emoji);
    
    // Optionnel: fermer le picker après insertion
    // closeEmojiPicker();
}

/**
 * Ajouter un emoji aux fréquents
 */
function addToFrequentEmojis(emoji) {
    // Retirer l'emoji s'il existe déjà
    frequentEmojis = frequentEmojis.filter(e => e !== emoji);
    
    // Ajouter au début
    frequentEmojis.unshift(emoji);
    
    // Limiter à 20 emojis
    frequentEmojis = frequentEmojis.slice(0, 20);
    
    // Sauvegarder dans localStorage
    try {
        localStorage.setItem('frequentEmojis', JSON.stringify(frequentEmojis));
    } catch (e) {
        console.error('Erreur sauvegarde emojis fréquents:', e);
    }
    
    // Mettre à jour l'affichage si on est dans la catégorie frequent
    if (currentEmojiCategory === 'frequent') {
        displayEmojis('frequent');
    }
}

/**
 * Fermer le picker en cliquant en dehors
 */
document.addEventListener('click', function(e) {
    const picker = document.getElementById('emojiPicker');
    const emojiBtn = document.querySelector('.btn-emoji');
    
    if (picker && picker.classList.contains('show')) {
        if (!picker.contains(e.target) && e.target !== emojiBtn && !emojiBtn?.contains(e.target)) {
            closeEmojiPicker();
        }
    }
});

// Initialiser le picker au chargement
document.addEventListener('DOMContentLoaded', function() {
    initializeEmojiPicker();
});

console.log('✅ Fonctions utilitaires WebSocket, Responsive, Upload et Emoji chargées');
