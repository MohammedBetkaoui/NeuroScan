// ==========================================
// Chatbot Visiteur - Version Moderne & Pro
// ==========================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🤖 Initialisation du chatbot visiteurs (Version Pro)...');
    
    // Éléments DOM
    const chatbotButton = document.getElementById('visitor-chatbot-button');
    const chatbotWindow = document.getElementById('visitor-chatbot-window');
    const chatbotClose = document.getElementById('chatbot-close');
    const chatbotMessages = document.getElementById('chatbot-messages');
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotSend = document.getElementById('chatbot-send');
    
    // État du chatbot
    let isOpen = false;
    let conversationHistory = [];
    let isTyping = false;
    
    // Messages de bienvenue et contexte
    const SYSTEM_CONTEXT = `Tu es l'assistant virtuel de NeuroScan, une plateforme médicale d'analyse IA pour les tumeurs cérébrales.

RÈGLES STRICTES:
1. Tu DOIS répondre UNIQUEMENT aux questions concernant le projet NeuroScan et ses fonctionnalités
2. Tu NE DOIS PAS répondre aux questions médicales générales, diagnostics ou conseils de santé
3. Si on te pose une question médicale, redirige poliment vers un professionnel de santé

INFORMATIONS SUR NEUROSCAN:

Présentation:
- Plateforme d'intelligence artificielle pour l'analyse d'images IRM cérébrales
- Détection automatique de tumeurs avec précision de 99.7%
- Résultats en moins de 10 secondes
- Interface moderne et intuitive

Fonctionnalités principales:
1. Analyse IA CNN (Réseau de Neurones Convolutionnel)
   - Détection de 4 types: Normal, Gliome, Méningiome, Tumeur pituitaire
   - Précision: 99.7% validée cliniquement
   - Temps d'analyse: < 10 secondes
   - Rapports détaillés avec probabilités

2. Chat Médical IA
   - Assistant conversationnel spécialisé en neurologie
   - Questions médicales et interprétation d'examens
   - Historique des conversations
   - Interface moderne

3. Gestion des Patients
   - Dossiers médicaux complets
   - Suivi longitudinal et évolution
   - Historique des analyses
   - Export de rapports PDF

4. Tableau de Bord Professionnel
   - Statistiques en temps réel
   - Graphiques interactifs
   - Métriques de performance
   - Alertes médicales

5. Sécurité et Conformité
   - Chiffrement AES-256 de bout en bout
   - Conformité RGPD
   - Certification ISO 27001
   - CE Médical
   - Infrastructure cloud sécurisée

Statistiques de performance:
- Précision: 99.7%
- Temps d'analyse: < 10 secondes
- 50,000+ analyses effectuées
- 500+ médecins utilisateurs
- 98.7% de satisfaction médecins

Processus d'utilisation:
1. Upload sécurisé de l'image IRM (DICOM, NIfTI, JPEG, PNG)
2. Analyse IA automatique en moins de 10s
3. Résultats détaillés avec classification et probabilités
4. Génération rapport PDF professionnel

Technologies:
- Deep Learning CNN (PyTorch)
- Modèle entraîné sur 100,000+ images validées
- API Gemini pour le chat médical
- Base de données SQLite pour le suivi
- Interface Flask avec design moderne

Accès:
- Inscription gratuite pour les médecins
- Authentification sécurisée
- Dashboard personnel
- Support 24/7

Contact:
- Email: mohammed.betkaoui@neuroscan.ai
- Téléphone: +123783962348
- Adresse: Bordj Bou Arréridj, Algérie

Réponds de manière concise, professionnelle et amicale. Si la question sort du contexte de NeuroScan, explique poliment que tu ne peux répondre qu'aux questions sur la plateforme.`;

    const WELCOME_MESSAGE = "👋 **Bienvenue sur NeuroScan!**\n\nJe suis votre assistant virtuel, prêt à vous aider à découvrir notre plateforme d'analyse IA pour les tumeurs cérébrales.\n\n💡 *Astuce: Cliquez sur une suggestion ci-dessous ou posez votre question!*";
    
    // Suggestions rapides
    const QUICK_SUGGESTIONS = [
        "Comment fonctionne l'analyse IA? 🧠",
        "Quelles fonctionnalités proposez-vous? ⚡",
        "Quelle est la précision du système? 🎯",
        "Comment créer un compte? 👤",
        "Quels types de tumeurs détectez-vous? 🔬"
    ];
    
    // Vérification des éléments
    if (!chatbotButton || !chatbotWindow) {
        console.error('❌ Éléments du chatbot non trouvés');
        return;
    }
    
    console.log('✅ Éléments du chatbot trouvés');
    
    // ==========================================
    // Fonctions utilitaires
    // ==========================================
    
    // Formater le markdown avec support étendu
    function formatMarkdown(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>')
            // Support des listes
            .replace(/^• (.+)$/gm, '<li>$1</li>')
            .replace(/^- (.+)$/gm, '<li>$1</li>')
            // Support des liens
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    }
    
    // Obtenir l'heure actuelle
    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
    }
    
    // Scroll fluide vers le bas
    function scrollToBottom(smooth = true) {
        if (smooth) {
            chatbotMessages.scrollTo({
                top: chatbotMessages.scrollHeight,
                behavior: 'smooth'
            });
        } else {
            chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        }
    }
    
    // Auto-resize du textarea
    function autoResizeTextarea() {
        chatbotInput.style.height = 'auto';
        chatbotInput.style.height = Math.min(chatbotInput.scrollHeight, 120) + 'px';
    }
    
    // ==========================================
    // Fonctions de gestion des messages
    // ==========================================
    
    // Ajouter un message
    function addMessage(content, isBot = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isBot ? 'bot' : 'user'}`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = `<i class="fas fa-${isBot ? 'robot' : 'user'}"></i>`;
        
        const contentDiv = document.createElement('div');
        contentDiv.style.display = 'flex';
        contentDiv.style.flexDirection = 'column';
        contentDiv.style.gap = '4px';
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-content';
        textDiv.innerHTML = formatMarkdown(content);
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = getCurrentTime();
        
        contentDiv.appendChild(textDiv);
        contentDiv.appendChild(timeDiv);
        
        if (isBot) {
            messageDiv.appendChild(avatarDiv);
            messageDiv.appendChild(contentDiv);
        } else {
            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(avatarDiv);
        }
        
        chatbotMessages.appendChild(messageDiv);
        scrollToBottom();
        
        // Ajouter effet sonore léger (optionnel)
        playMessageSound(isBot);
    }
    
    // Effet sonore léger pour les messages (optionnel)
    function playMessageSound(isBot) {
        // Peut être activé pour ajouter un feedback audio
        // const audio = new Audio(isBot ? '/static/sounds/bot.mp3' : '/static/sounds/user.mp3');
        // audio.volume = 0.2;
        // audio.play().catch(e => console.log('Son désactivé'));
    }
    
    // Afficher l'indicateur de frappe
    function showTypingIndicator() {
        if (document.getElementById('typing-indicator')) return;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typing-indicator';
        
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        chatbotMessages.appendChild(typingDiv);
        scrollToBottom();
        isTyping = true;
    }
    
    // Masquer l'indicateur de frappe
    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.style.animation = 'messageSlideIn 0.3s ease reverse';
            setTimeout(() => typingIndicator.remove(), 300);
        }
        isTyping = false;
    }
    
    // Afficher les suggestions rapides
    function showQuickSuggestions() {
        const suggestionsContainer = document.querySelector('.quick-suggestions');
        if (!suggestionsContainer) return;
        
        suggestionsContainer.innerHTML = '';
        suggestionsContainer.style.display = 'flex';
        
        QUICK_SUGGESTIONS.forEach((suggestion, index) => {
            const chip = document.createElement('div');
            chip.className = 'suggestion-chip';
            chip.innerHTML = `<i class="fas fa-lightbulb"></i>${suggestion}`;
            chip.style.animationDelay = `${index * 0.1}s`;
            chip.onclick = () => {
                chatbotInput.value = suggestion.replace(/[🧠⚡🎯👤🔬]/g, '').trim();
                sendMessage();
            };
            suggestionsContainer.appendChild(chip);
        });
    }
    
    // Masquer les suggestions
    function hideSuggestions() {
        const suggestionsContainer = document.querySelector('.quick-suggestions');
        if (suggestionsContainer) {
            suggestionsContainer.style.display = 'none';
        }
    }
    
    // ==========================================
    // Fonctions API
    // ==========================================
    
    // Appeler l'API Gemini
    async function callGeminiAPI(userMessage) {
        try {
            const response = await fetch('/api/visitor-chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userMessage,
                    history: conversationHistory
                })
            });
            
            const data = await response.json();
            
            if (response.status === 429) {
                const retryAfter = data.retry_after || 15;
                return `⚠️ **Quota d'utilisation dépassé**\n\nLe service de chat a atteint sa limite gratuite de 250 requêtes par jour.\n\n⏱️ Veuillez réessayer dans **${retryAfter} secondes** ou contactez-nous pour plus d'informations.\n\n📧 Email: mohammed.betkaoui@neuroscan.ai\n📞 Téléphone: +123783962348`;
            }
            
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            
            if (data.success) {
                conversationHistory = data.history;
                return data.response;
            } else {
                if (data.error === 'quota_exceeded') {
                    const retryAfter = data.retry_after || 15;
                    return `⚠️ **Quota d'utilisation dépassé**\n\nLe service de chat a atteint sa limite gratuite de 250 requêtes par jour.\n\n⏱️ Veuillez réessayer dans **${retryAfter} secondes**.\n\n💡 **Astuce**: Vous pouvez toujours créer un compte et explorer les fonctionnalités d'analyse IA sans limitation!`;
                } else {
                    throw new Error(data.message || data.error || 'Erreur inconnue');
                }
            }
        } catch (error) {
            console.error('Erreur API:', error);
            return `😔 **Service temporairement indisponible**\n\nDésolé, je rencontre un problème technique.\n\n🔄 Veuillez réessayer dans quelques instants ou contactez notre support:\n📧 mohammed.betkaoui@neuroscan.ai\n📞 +123783962348`;
        }
    }
    
    // ==========================================
    // Fonctions principales
    // ==========================================
    
    // Envoyer un message
    async function sendMessage() {
        const message = chatbotInput.value.trim();
        
        if (!message || isTyping) return;
        
        // Ajouter le message utilisateur
        addMessage(message, false);
        chatbotInput.value = '';
        chatbotInput.style.height = 'auto';
        chatbotSend.disabled = true;
        
        // Masquer les suggestions
        hideSuggestions();
        
        // Afficher l'indicateur de frappe
        showTypingIndicator();
        
        // Appeler l'API avec délai réaliste
        setTimeout(async () => {
            const response = await callGeminiAPI(message);
            
            // Masquer l'indicateur de frappe
            hideTypingIndicator();
            
            // Ajouter la réponse du bot avec délai pour effet naturel
            setTimeout(() => {
                addMessage(response, true);
                chatbotSend.disabled = false;
                chatbotInput.focus();
            }, 300);
        }, 800);
    }
    
    // Ouvrir/fermer le chatbot
    function toggleChatbot() {
        isOpen = !isOpen;
        chatbotWindow.classList.toggle('show', isOpen);
        
        if (isOpen) {
            chatbotButton.classList.remove('attention');
            chatbotInput.focus();
            
            // Afficher le message de bienvenue si c'est la première ouverture
            if (conversationHistory.length === 0) {
                setTimeout(() => {
                    addMessage(WELCOME_MESSAGE, true);
                    showQuickSuggestions();
                }, 400);
            }
            
            // Marquer comme vu dans le localStorage
            localStorage.setItem('chatbot_opened', 'true');
        } else {
            // Sauvegarder l'état
            saveChatState();
        }
    }
    
    // Sauvegarder l'état du chat
    function saveChatState() {
        try {
            localStorage.setItem('chatbot_history', JSON.stringify(conversationHistory));
        } catch (e) {
            console.log('Impossible de sauvegarder l\'historique');
        }
    }
    
    // Restaurer l'état du chat
    function restoreChatState() {
        try {
            const savedHistory = localStorage.getItem('chatbot_history');
            if (savedHistory) {
                conversationHistory = JSON.parse(savedHistory);
            }
        } catch (e) {
            console.log('Impossible de restaurer l\'historique');
        }
    }
    
    // ==========================================
    // Event Listeners
    // ==========================================
    
    // Bouton chatbot - Toggle
    chatbotButton.addEventListener('click', toggleChatbot);
    
    // Bouton fermeture
    chatbotClose.addEventListener('click', toggleChatbot);
    
    // Bouton envoi
    chatbotSend.addEventListener('click', sendMessage);
    
    // Input - Entrée pour envoyer
    chatbotInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Input - Auto-resize et activation bouton
    chatbotInput.addEventListener('input', () => {
        autoResizeTextarea();
        chatbotSend.disabled = !chatbotInput.value.trim();
    });
    
    // Input - Shift+Enter pour nouvelle ligne
    chatbotInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.shiftKey) {
            e.stopPropagation();
        }
    });
    
    // Fermer avec Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isOpen) {
            toggleChatbot();
        }
    });
    
    // Fermer en cliquant à l'extérieur (optionnel)
    document.addEventListener('click', (e) => {
        if (isOpen && 
            !chatbotWindow.contains(e.target) && 
            !chatbotButton.contains(e.target)) {
            // Optionnel: décommenter pour fermer en cliquant dehors
            // toggleChatbot();
        }
    });
    
    // ==========================================
    // Initialisation
    // ==========================================
    
    // Restaurer l'état sauvegardé
    restoreChatState();
    
    // Animation d'attention au démarrage (si pas déjà ouvert)
    const hasOpened = localStorage.getItem('chatbot_opened');
    if (!hasOpened) {
        setTimeout(() => {
            chatbotButton.classList.add('attention');
        }, 3000);
        
        // Retirer l'animation après 15 secondes
        setTimeout(() => {
            chatbotButton.classList.remove('attention');
        }, 18000);
    }
    
    // Afficher tooltip au survol (optionnel)
    let tooltipTimeout;
    chatbotButton.addEventListener('mouseenter', () => {
        if (!isOpen) {
            tooltipTimeout = setTimeout(() => {
                // Créer tooltip
                const tooltip = document.createElement('div');
                tooltip.id = 'chatbot-tooltip';
                tooltip.style.cssText = `
                    position: fixed;
                    bottom: 105px;
                    right: 24px;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 12px 18px;
                    border-radius: 16px;
                    font-size: 14px;
                    font-weight: 600;
                    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
                    z-index: 999;
                    animation: slideInUp 0.3s ease;
                    white-space: nowrap;
                `;
                tooltip.textContent = '💬 Besoin d\'aide? Cliquez ici!';
                document.body.appendChild(tooltip);
            }, 1000);
        }
    });
    
    chatbotButton.addEventListener('mouseleave', () => {
        clearTimeout(tooltipTimeout);
        const tooltip = document.getElementById('chatbot-tooltip');
        if (tooltip) {
            tooltip.style.animation = 'slideInUp 0.3s ease reverse';
            setTimeout(() => tooltip.remove(), 300);
        }
    });
    
    // Support du copier-coller d'images (fonctionnalité avancée - optionnel)
    chatbotInput.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (items) {
            for (let item of items) {
                if (item.type.indexOf('image') !== -1) {
                    e.preventDefault();
                    addMessage('ℹ️ Les images ne sont pas supportées dans ce chat. Pour analyser une image médicale, veuillez vous connecter à votre compte.', true);
                    break;
                }
            }
        }
    });
    
    // Détection de l'inactivité (optionnel)
    let inactivityTimer;
    function resetInactivityTimer() {
        clearTimeout(inactivityTimer);
        if (isOpen && conversationHistory.length > 0) {
            inactivityTimer = setTimeout(() => {
                if (!isTyping) {
                    addMessage('👋 Vous avez d\'autres questions? Je suis toujours là pour vous aider!', true);
                }
            }, 120000); // 2 minutes
        }
    }
    
    // Reset timer lors d'une interaction
    chatbotInput.addEventListener('focus', resetInactivityTimer);
    chatbotMessages.addEventListener('scroll', resetInactivityTimer);
    
    // Log d'initialisation réussie
    console.log('✅ Chatbot visiteurs initialisé avec succès (Version Pro)');
    console.log('📊 Fonctionnalités activées:');
    console.log('   - Messages formatés (Markdown)');
    console.log('   - Suggestions rapides');
    console.log('   - Auto-resize textarea');
    console.log('   - Sauvegarde état');
    console.log('   - Animations fluides');
    console.log('   - Responsive design');
    console.log('   - Raccourcis clavier (Enter, Shift+Enter, Esc)');
    console.log('   - Détection inactivité');
    
    // Analytics (optionnel - pour suivre l'utilisation)
    if (window.gtag) {
        chatbotButton.addEventListener('click', () => {
            gtag('event', 'chatbot_opened', {
                event_category: 'engagement',
                event_label: 'visitor_chatbot'
            });
        });
    }
});

// ==========================================
// Fin du fichier JavaScript
// ==========================================
