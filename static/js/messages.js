// ===========================
// Messages Page JavaScript
// Modern UI Interactions & Animations
// ===========================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Messages interface initialized');
    
    // ===========================
    // Elements Selection
    // ===========================
    const conversationItems = document.querySelectorAll('.conversation-item');
    const messageInput = document.getElementById('messageInput');
    const btnSend = document.getElementById('btnSend');
    const messagesArea = document.getElementById('messagesArea');
    const typingIndicator = document.getElementById('typingIndicator');
    const btnNewMessage = document.getElementById('btnNewMessage');
    const newMessageModal = document.getElementById('newMessageModal');
    const btnCloseModal = document.getElementById('btnCloseModal');
    const searchConversations = document.getElementById('searchConversations');
    const filterBtns = document.querySelectorAll('.filter-btn');
    const infoPanel = document.getElementById('infoPanel');
    const conversationsSidebar = document.getElementById('conversationsSidebar');
    
    // ===========================
    // Auto-resize Message Input
    // ===========================
    if (messageInput) {
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            
            // Enable/disable send button based on input
            if (btnSend) {
                btnSend.disabled = this.value.trim() === '';
            }
        });
        
        // Handle Enter key (send message) and Shift+Enter (new line)
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Focus management
        messageInput.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        messageInput.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    }
    
    // ===========================
    // Send Message Function
    // ===========================
    function sendMessage() {
        const messageText = messageInput.value.trim();
        
        if (messageText === '') return;
        
        // Create message element
        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'message-wrapper sent';
        
        const currentTime = new Date().toLocaleTimeString('fr-FR', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        messageWrapper.innerHTML = `
            <div class="message-content">
                <div class="message-bubble">
                    <p>${escapeHtml(messageText)}</p>
                </div>
                <span class="message-time">${currentTime} <i class="fas fa-check"></i></span>
            </div>
        `;
        
        // Add message to chat
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator && typingIndicator.style.display !== 'none') {
            messagesArea.insertBefore(messageWrapper, typingIndicator);
        } else {
            messagesArea.appendChild(messageWrapper);
        }
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        
        // Scroll to bottom
        scrollToBottom();
        
        // Simulate typing indicator (demo purposes)
        setTimeout(() => {
            showTypingIndicator();
            setTimeout(() => {
                hideTypingIndicator();
                simulateReceivedMessage();
            }, 2000);
        }, 500);
    }
    
    // ===========================
    // Send Button Click
    // ===========================
    if (btnSend) {
        btnSend.addEventListener('click', sendMessage);
    }
    
    // ===========================
    // Conversation Selection
    // ===========================
    conversationItems.forEach(item => {
        item.addEventListener('click', function() {
            // Remove active class from all
            conversationItems.forEach(i => i.classList.remove('active'));
            
            // Add active class to clicked
            this.classList.add('active');
            
            // Remove unread badge
            const unreadBadge = this.querySelector('.unread-badge');
            if (unreadBadge) {
                unreadBadge.remove();
            }
            
            // Update chat header with selected user info
            const avatar = this.querySelector('.conversation-avatar img').src;
            const name = this.querySelector('.conversation-header h4').textContent;
            const statusIndicator = this.querySelector('.status-indicator');
            let status = 'Hors ligne';
            
            if (statusIndicator.classList.contains('online')) {
                status = 'En ligne';
            } else if (statusIndicator.classList.contains('away')) {
                status = 'Absent';
            }
            
            updateChatHeader(avatar, name, status);
            
            // Load conversation messages (demo)
            loadConversationMessages(this.dataset.id);
        });
    });
    
    // ===========================
    // Update Chat Header
    // ===========================
    function updateChatHeader(avatar, name, status) {
        const chatAvatar = document.querySelector('.chat-avatar img');
        const chatName = document.querySelector('.chat-user-details h3');
        const chatStatus = document.querySelector('.user-status');
        
        if (chatAvatar) chatAvatar.src = avatar;
        if (chatName) chatName.textContent = name;
        if (chatStatus) chatStatus.textContent = status;
        
        // Update info panel
        const profileAvatar = document.querySelector('.profile-avatar img');
        const profileName = document.querySelector('.profile-card h3');
        
        if (profileAvatar) profileAvatar.src = avatar;
        if (profileName) profileName.textContent = name;
    }
    
    // ===========================
    // Load Conversation Messages (Demo)
    // ===========================
    function loadConversationMessages(conversationId) {
        // Clear current messages
        messagesArea.innerHTML = `
            <div class="date-divider">
                <span>Aujourd'hui</span>
            </div>
        `;
        
        // Demo messages based on conversation ID
        const demoMessages = {
            '1': [
                {
                    type: 'received',
                    text: 'Bonjour, j\'ai examiné le scanner du patient référé.',
                    time: '10:25'
                },
                {
                    type: 'sent',
                    text: 'Merci pour votre retour rapide.',
                    time: '10:27'
                }
            ],
            '2': [
                {
                    type: 'received',
                    text: 'Le patient présente des symptômes intéressants.',
                    time: '14:30'
                },
                {
                    type: 'sent',
                    text: 'J\'ai vu le dossier. Nous devrions discuter en équipe.',
                    time: '14:35'
                }
            ]
        };
        
        const messages = demoMessages[conversationId] || [];
        
        messages.forEach(msg => {
            addMessageToChat(msg.type, msg.text, msg.time);
        });
        
        scrollToBottom();
    }
    
    // ===========================
    // Add Message to Chat
    // ===========================
    function addMessageToChat(type, text, time) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message-wrapper ${type}`;
        
        if (type === 'received') {
            const currentAvatar = document.querySelector('.chat-avatar img').src;
            messageWrapper.innerHTML = `
                <div class="message-avatar">
                    <img src="${currentAvatar}" alt="User">
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <p>${escapeHtml(text)}</p>
                    </div>
                    <span class="message-time">${time}</span>
                </div>
            `;
        } else {
            messageWrapper.innerHTML = `
                <div class="message-content">
                    <div class="message-bubble">
                        <p>${escapeHtml(text)}</p>
                    </div>
                    <span class="message-time">${time} <i class="fas fa-check-double read"></i></span>
                </div>
            `;
        }
        
        messagesArea.appendChild(messageWrapper);
    }
    
    // ===========================
    // Typing Indicator
    // ===========================
    function showTypingIndicator() {
        if (typingIndicator) {
            typingIndicator.style.display = 'flex';
            scrollToBottom();
        }
    }
    
    function hideTypingIndicator() {
        if (typingIndicator) {
            typingIndicator.style.display = 'none';
        }
    }
    
    // ===========================
    // Simulate Received Message (Demo)
    // ===========================
    function simulateReceivedMessage() {
        const responses = [
            'D\'accord, je vais vérifier cela.',
            'Merci pour l\'information.',
            'C\'est noté, je m\'en occupe.',
            'Parfait, on en reparle bientôt.',
            'Je vous tiens au courant.'
        ];
        
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        const currentTime = new Date().toLocaleTimeString('fr-FR', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        addMessageToChat('received', randomResponse, currentTime);
        scrollToBottom();
        
        // Play notification sound (optional)
        playNotificationSound();
    }
    
    // ===========================
    // Scroll to Bottom
    // ===========================
    function scrollToBottom() {
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }
    
    // ===========================
    // Search Conversations
    // ===========================
    if (searchConversations) {
        searchConversations.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            
            conversationItems.forEach(item => {
                const name = item.querySelector('.conversation-header h4').textContent.toLowerCase();
                const preview = item.querySelector('.conversation-preview p').textContent.toLowerCase();
                
                if (name.includes(searchTerm) || preview.includes(searchTerm)) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }
    
    // ===========================
    // Filter Messages
    // ===========================
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Remove active from all
            filterBtns.forEach(b => b.classList.remove('active'));
            
            // Add active to clicked
            this.classList.add('active');
            
            const filter = this.dataset.filter;
            
            conversationItems.forEach(item => {
                switch(filter) {
                    case 'all':
                        item.style.display = 'flex';
                        break;
                    case 'unread':
                        if (item.querySelector('.unread-badge')) {
                            item.style.display = 'flex';
                        } else {
                            item.style.display = 'none';
                        }
                        break;
                    case 'important':
                        // Demo: show random items
                        item.style.display = Math.random() > 0.5 ? 'flex' : 'none';
                        break;
                }
            });
        });
    });
    
    // ===========================
    // New Message Modal
    // ===========================
    if (btnNewMessage) {
        btnNewMessage.addEventListener('click', function() {
            newMessageModal.classList.add('active');
        });
    }
    
    if (btnCloseModal) {
        btnCloseModal.addEventListener('click', function() {
            newMessageModal.classList.remove('active');
        });
    }
    
    // Close modal when clicking outside
    if (newMessageModal) {
        newMessageModal.addEventListener('click', function(e) {
            if (e.target === this) {
                this.classList.remove('active');
            }
        });
    }
    
    // ===========================
    // File Attachment (Demo)
    // ===========================
    const btnAttach = document.querySelector('.btn-icon[title="Joindre un fichier"]');
    if (btnAttach) {
        btnAttach.addEventListener('click', function() {
            alert('Fonctionnalité de pièce jointe - À implémenter avec la logique backend');
        });
    }
    
    const btnImage = document.querySelector('.btn-icon[title="Insérer une image"]');
    if (btnImage) {
        btnImage.addEventListener('click', function() {
            alert('Fonctionnalité d\'image - À implémenter avec la logique backend');
        });
    }
    
    // ===========================
    // Video Call (Demo)
    // ===========================
    const btnVideoCall = document.querySelector('.btn-icon[title="Appel vidéo"]');
    if (btnVideoCall) {
        btnVideoCall.addEventListener('click', function() {
            alert('Fonctionnalité d\'appel vidéo - À implémenter avec la logique backend');
        });
    }
    
    // ===========================
    // Info Panel Toggle (Mobile)
    // ===========================
    const btnMoreOptions = document.querySelector('.chat-actions .btn-icon[title="Plus d\'options"]');
    if (btnMoreOptions && infoPanel) {
        btnMoreOptions.addEventListener('click', function() {
            infoPanel.classList.toggle('active');
        });
    }
    
    const btnClosePanel = document.getElementById('btnClosePanel');
    if (btnClosePanel && infoPanel) {
        btnClosePanel.addEventListener('click', function() {
            infoPanel.classList.remove('active');
        });
    }
    
    // ===========================
    // Emoji Button (Demo)
    // ===========================
    const btnEmoji = document.querySelector('.btn-emoji');
    if (btnEmoji) {
        btnEmoji.addEventListener('click', function() {
            alert('Sélecteur d\'émojis - À implémenter');
        });
    }
    
    // ===========================
    // Quick Actions (Demo)
    // ===========================
    const actionBtns = document.querySelectorAll('.action-btn');
    actionBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const action = this.querySelector('span').textContent;
            alert(`Action: ${action} - À implémenter avec la logique backend`);
        });
    });
    
    // ===========================
    // Shared Files Click (Demo)
    // ===========================
    const fileItems = document.querySelectorAll('.file-item');
    fileItems.forEach(item => {
        item.addEventListener('click', function() {
            const fileName = this.querySelector('.file-name').textContent;
            alert(`Ouverture du fichier: ${fileName} - À implémenter`);
        });
    });
    
    // ===========================
    // Notification Sound
    // ===========================
    function playNotificationSound() {
        // Check if notification sound is available
        const audio = new Audio('/static/shop-notification-355746.mp3');
        audio.volume = 0.3;
        audio.play().catch(e => {
            console.log('Notification sound not available:', e);
        });
    }
    
    // ===========================
    // Utility Functions
    // ===========================
    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
    
    // ===========================
    // Initialize - Scroll to bottom on load
    // ===========================
    scrollToBottom();
    
    // ===========================
    // Demo: Simulate online status changes
    // ===========================
    setInterval(() => {
        const statusIndicators = document.querySelectorAll('.status-indicator');
        statusIndicators.forEach(indicator => {
            if (Math.random() > 0.9) {
                const statuses = ['online', 'away', 'offline'];
                const currentStatus = statuses.find(s => indicator.classList.contains(s));
                const newStatus = statuses[Math.floor(Math.random() * statuses.length)];
                
                if (currentStatus !== newStatus) {
                    indicator.classList.remove('online', 'away', 'offline');
                    indicator.classList.add(newStatus);
                }
            }
        });
    }, 10000);
    
    // ===========================
    // Responsive - Mobile Menu Toggle
    // ===========================
    function createMobileMenu() {
        if (window.innerWidth <= 768) {
            const chatHeader = document.querySelector('.chat-header');
            if (chatHeader && !document.querySelector('.btn-menu-mobile')) {
                const btnMenu = document.createElement('button');
                btnMenu.className = 'btn-icon btn-menu-mobile';
                btnMenu.innerHTML = '<i class="fas fa-bars"></i>';
                btnMenu.style.marginRight = 'auto';
                
                btnMenu.addEventListener('click', function() {
                    const sidebar = document.querySelector('.conversations-sidebar');
                    if (sidebar) {
                        sidebar.classList.toggle('active');
                    }
                });
                
                const userInfo = chatHeader.querySelector('.chat-user-info');
                chatHeader.insertBefore(btnMenu, userInfo);
            }
        }
    }
    
    createMobileMenu();
    window.addEventListener('resize', createMobileMenu);
    
    // ===========================
    // Demo Welcome Message
    // ===========================
    console.log('%c NeuroScan AI - Messagerie Professionnelle ', 'background: #4F46E5; color: white; padding: 10px 20px; border-radius: 5px; font-size: 14px; font-weight: bold;');
    console.log('Interface de messagerie chargée avec succès!');
    console.log('Note: Cette interface est une démonstration UI. La logique backend doit être implémentée.');
    
});
