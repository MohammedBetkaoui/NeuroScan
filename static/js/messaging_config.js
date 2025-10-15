/**
 * Configuration de la messagerie NeuroScan
 */

const MESSAGING_CONFIG = {
    // WebSocket
    useWebSocket: false,  // Mettre à true quand WebSocket est prêt
    websocketReconnectAttempts: 5,
    websocketReconnectDelay: 1000,
    
    // Polling (fallback)
    pollingInterval: 5000,  // 5 secondes
    pollingEnabled: true,
    
    // Messages
    messageLoadLimit: 50,
    messageCacheSize: 100,
    
    // UI
    showTypingIndicator: true,
    autoScrollToBottom: true,
    markAsReadDelay: 1000,  // 1 seconde
    
    // Notifications
    enableNotifications: true,
    notificationSound: true,
    
    // Debug
    debugMode: true,
    logWebSocketEvents: true
};

// Export pour utilisation globale
window.MESSAGING_CONFIG = MESSAGING_CONFIG;

console.log('⚙️ Configuration messagerie chargée:', MESSAGING_CONFIG);
