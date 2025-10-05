// Chatbot pour visiteurs - JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('🤖 Initialisation du chatbot visiteurs...');
    
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

    const WELCOME_MESSAGE = "👋 Bonjour! Je suis l'assistant virtuel de **NeuroScan**.\n\nJe peux vous aider à comprendre notre plateforme d'analyse IA pour les tumeurs cérébrales.\n\nQue souhaitez-vous savoir?";
    
    // Suggestions rapides
    const QUICK_SUGGESTIONS = [
        "Comment fonctionne l'analyse IA?",
        "Quelles fonctionnalités proposez-vous?",
        "Quelle est la précision du système?",
        "Comment créer un compte?",
        "Quels types de tumeurs détectez-vous?"
    ];
    
    // Vérification des éléments
    if (!chatbotButton || !chatbotWindow) {
        console.error('❌ Éléments du chatbot non trouvés');
        return;
    }
    
    console.log('✅ Éléments du chatbot trouvés');
    
    // Fonction pour formater le markdown simple
    function formatMarkdown(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }
    
    // Fonction pour obtenir l'heure actuelle
    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
    }
    
    // Fonction pour ajouter un message
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
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }
    
    // Fonction pour afficher l'indicateur de frappe
    function showTypingIndicator() {
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
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }
    
    // Fonction pour masquer l'indicateur de frappe
    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    // Fonction pour afficher les suggestions rapides
    function showQuickSuggestions() {
        const suggestionsContainer = document.querySelector('.quick-suggestions');
        if (!suggestionsContainer) return;
        
        suggestionsContainer.innerHTML = '';
        QUICK_SUGGESTIONS.forEach(suggestion => {
            const chip = document.createElement('div');
            chip.className = 'suggestion-chip';
            chip.innerHTML = `<i class="fas fa-lightbulb"></i>${suggestion}`;
            chip.onclick = () => {
                chatbotInput.value = suggestion;
                sendMessage();
            };
            suggestionsContainer.appendChild(chip);
        });
    }
    
    // Fonction pour appeler l'API Gemini
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
            
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                conversationHistory = data.history;
                return data.response;
            } else {
                throw new Error(data.error || 'Erreur inconnue');
            }
        } catch (error) {
            console.error('Erreur API:', error);
            return "Désolé, je rencontre un problème technique. Veuillez réessayer dans quelques instants.";
        }
    }
    
    // Fonction pour envoyer un message
    async function sendMessage() {
        const message = chatbotInput.value.trim();
        
        if (!message) return;
        
        // Ajouter le message utilisateur
        addMessage(message, false);
        chatbotInput.value = '';
        chatbotSend.disabled = true;
        
        // Masquer les suggestions
        const suggestionsContainer = document.querySelector('.quick-suggestions');
        if (suggestionsContainer) {
            suggestionsContainer.style.display = 'none';
        }
        
        // Afficher l'indicateur de frappe
        showTypingIndicator();
        
        // Appeler l'API
        const response = await callGeminiAPI(message);
        
        // Masquer l'indicateur de frappe
        hideTypingIndicator();
        
        // Ajouter la réponse du bot
        addMessage(response, true);
        
        chatbotSend.disabled = false;
        chatbotInput.focus();
    }
    
    // Fonction pour ouvrir/fermer le chatbot
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
                }, 300);
            }
        }
    }
    
    // Event listeners
    chatbotButton.addEventListener('click', toggleChatbot);
    chatbotClose.addEventListener('click', toggleChatbot);
    
    chatbotSend.addEventListener('click', sendMessage);
    
    chatbotInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    chatbotInput.addEventListener('input', () => {
        chatbotSend.disabled = !chatbotInput.value.trim();
    });
    
    // Animation d'attention au démarrage
    setTimeout(() => {
        chatbotButton.classList.add('attention');
    }, 2000);
    
    // Retirer l'animation après 10 secondes
    setTimeout(() => {
        chatbotButton.classList.remove('attention');
    }, 12000);
    
    console.log('✅ Chatbot visiteurs initialisé avec succès');
});
