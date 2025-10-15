/**
 * NeuroScan AI - WebSocket Manager
 * Gestion des connexions WebSocket en temps réel pour la messagerie
 */

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.currentDoctorId = null;
        this.currentConversationId = null;
        this.messageCallbacks = [];
        this.typingCallbacks = [];
        this.readCallbacks = [];
        this.onlineCallbacks = [];
        this.offlineCallbacks = [];
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    /**
     * Initialiser la connexion WebSocket
     */
    connect(doctorId) {
        console.log('🔌 Initialisation de la connexion WebSocket...');
        
        this.currentDoctorId = doctorId;
        
        // Créer la connexion Socket.IO
        this.socket = io({
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: this.maxReconnectAttempts
        });

        // Événements de connexion
        this.socket.on('connect', () => {
            console.log('✅ WebSocket connecté:', this.socket.id);
            this.connected = true;
            this.reconnectAttempts = 0;
            
            // Si on était dans une conversation, la rejoindre
            if (this.currentConversationId) {
                this.joinConversation(this.currentConversationId);
            }
        });

        this.socket.on('disconnect', (reason) => {
            console.log('❌ WebSocket déconnecté:', reason);
            this.connected = false;
        });

        this.socket.on('connect_error', (error) => {
            console.error('❌ Erreur de connexion WebSocket:', error);
            this.reconnectAttempts++;
            
            if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                console.error('❌ Nombre max de tentatives de reconnexion atteint');
                // Fallback au polling HTTP
                this.handleFallbackToPolling();
            }
        });

        // Événements personnalisés
        this.socket.on('connected', (data) => {
            console.log('🎉 Connexion établie:', data);
        });

        this.socket.on('new_message', (message) => {
            console.log('📨 Nouveau message reçu:', message);
            this.handleNewMessage(message);
        });

        this.socket.on('user_typing', (data) => {
            console.log('⌨️ Utilisateur en train d\'écrire:', data);
            this.handleTyping(data);
        });

        this.socket.on('messages_read', (data) => {
            console.log('✅ Messages lus:', data);
            this.handleMessagesRead(data);
        });

        this.socket.on('doctor_online', (data) => {
            console.log('🟢 Médecin en ligne:', data);
            this.handleDoctorOnline(data);
        });

        this.socket.on('doctor_offline', (data) => {
            console.log('🔴 Médecin hors ligne:', data);
            this.handleDoctorOffline(data);
        });

        this.socket.on('error', (error) => {
            console.error('❌ Erreur WebSocket:', error);
        });
    }

    /**
     * Rejoindre une conversation
     */
    joinConversation(conversationId) {
        if (!this.socket || !this.connected) {
            console.error('❌ WebSocket non connecté');
            return;
        }

        this.currentConversationId = conversationId;
        
        console.log(`🚪 Rejoindre la conversation ${conversationId}`);
        this.socket.emit('join', {
            doctor_id: this.currentDoctorId,
            conversation_id: conversationId
        });

        this.socket.once('joined', (data) => {
            console.log('✅ Conversation rejointe:', data);
        });
    }

    /**
     * Quitter une conversation
     */
    leaveConversation(conversationId) {
        if (!this.socket || !this.connected) {
            return;
        }

        console.log(`👋 Quitter la conversation ${conversationId}`);
        this.socket.emit('leave', {
            doctor_id: this.currentDoctorId,
            conversation_id: conversationId
        });

        if (this.currentConversationId === conversationId) {
            this.currentConversationId = null;
        }
    }

    /**
     * Envoyer un message
     */
    sendMessage(conversationId, content) {
        if (!this.socket || !this.connected) {
            console.error('❌ WebSocket non connecté, impossible d\'envoyer le message');
            return false;
        }

        console.log('📤 Envoi du message...');
        this.socket.emit('send_message', {
            conversation_id: conversationId,
            sender_id: this.currentDoctorId,
            content: content
        });

        return true;
    }

    /**
     * Notifier qu'on est en train d'écrire
     */
    sendTyping(conversationId, isTyping = true) {
        if (!this.socket || !this.connected) {
            return;
        }

        this.socket.emit('typing', {
            conversation_id: conversationId,
            doctor_id: this.currentDoctorId,
            is_typing: isTyping
        });
    }

    /**
     * Marquer les messages comme lus
     */
    markAsRead(conversationId) {
        if (!this.socket || !this.connected) {
            return;
        }

        this.socket.emit('mark_read', {
            conversation_id: conversationId,
            doctor_id: this.currentDoctorId
        });
    }

    /**
     * Gestionnaires d'événements
     */
    handleNewMessage(message) {
        this.messageCallbacks.forEach(callback => {
            try {
                callback(message);
            } catch (error) {
                console.error('Erreur callback message:', error);
            }
        });
    }

    handleTyping(data) {
        this.typingCallbacks.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error('Erreur callback typing:', error);
            }
        });
    }

    handleMessagesRead(data) {
        this.readCallbacks.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error('Erreur callback read:', error);
            }
        });
    }

    handleDoctorOnline(data) {
        this.onlineCallbacks.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error('Erreur callback online:', error);
            }
        });
    }

    handleDoctorOffline(data) {
        this.offlineCallbacks.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error('Erreur callback offline:', error);
            }
        });
    }

    /**
     * Enregistrer des callbacks
     */
    onNewMessage(callback) {
        this.messageCallbacks.push(callback);
    }

    onTyping(callback) {
        this.typingCallbacks.push(callback);
    }

    onMessagesRead(callback) {
        this.readCallbacks.push(callback);
    }

    onDoctorOnline(callback) {
        this.onlineCallbacks.push(callback);
    }

    onDoctorOffline(callback) {
        this.offlineCallbacks.push(callback);
    }

    /**
     * Fallback au polling HTTP en cas d'échec WebSocket
     */
    handleFallbackToPolling() {
        console.warn('⚠️ Passage en mode polling HTTP');
        // Cette fonction sera appelée depuis messages_modern.js pour activer le polling
        if (window.activateFallbackPolling) {
            window.activateFallbackPolling();
        }
    }

    /**
     * Déconnecter le WebSocket
     */
    disconnect() {
        if (this.socket) {
            if (this.currentConversationId) {
                this.leaveConversation(this.currentConversationId);
            }
            this.socket.disconnect();
            this.connected = false;
            console.log('🔌 WebSocket déconnecté');
        }
    }

    /**
     * Vérifier si connecté
     */
    isConnected() {
        return this.connected && this.socket && this.socket.connected;
    }
}

// Créer une instance globale
window.wsManager = new WebSocketManager();

console.log('✅ WebSocket Manager chargé');
