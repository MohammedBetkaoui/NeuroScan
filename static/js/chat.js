// Variables globales - accessibles dans window
window.currentConversationId = null;
window.conversations = [];
window.isLoading = false;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize feather icons
    feather.replace();
    
    // Initialiser le mode sombre
    initializeTheme();
    
    // Initialiser le système responsive mobile
    initializeMobileResponsive();
            
            // Sidebar toggle for mobile
            const sidebar = document.getElementById('sidebar');
            const openSidebar = document.getElementById('openSidebar');
            const closeSidebar = document.getElementById('closeSidebar');
            const overlay = document.getElementById('sidebarOverlay');
            
            if (openSidebar) {
                openSidebar.addEventListener('click', () => {
                    sidebar.classList.remove('sidebar-hidden');
                    if (overlay) overlay.classList.add('show');
                });
            }
            
            if (closeSidebar) {
                closeSidebar.addEventListener('click', () => {
                    sidebar.classList.add('sidebar-hidden');
                    if (overlay) overlay.classList.remove('show');
                });
            }
            
            if (overlay) {
                overlay.addEventListener('click', () => {
                    sidebar.classList.add('sidebar-hidden');
                    overlay.classList.remove('show');
                });
            }
            
        // ===== GESTION RESPONSIVE MOBILE AVANCÉE =====
        function initializeMobileResponsive() {
            const inputSection = document.getElementById('inputSection');
            const chatContainer = document.getElementById('chatContainer');
            const messageInput = document.getElementById('messageInput');
            const welcomeInput = document.getElementById('welcomeMessageInput');
            
            let isInputFocused = false;
            let lastScrollTop = 0;
            let scrollTimeout;
            
            // Gestion du focus sur les champs de saisie mobile
            [messageInput, welcomeInput].forEach(input => {
                if (!input) return;
                
                input.addEventListener('focus', () => {
                    if (window.innerWidth <= 768) {
                        isInputFocused = true;
                        inputSection.classList.add('show');
                        chatContainer.classList.add('input-active');
                        
                        // Scroll vers le bas pour voir le message en cours
                        setTimeout(() => {
                            const messagesContainer = document.getElementById('messagesContainer');
                            messagesContainer.scrollTop = messagesContainer.scrollHeight;
                        }, 300);
                    }
                });
                
                input.addEventListener('blur', () => {
                    setTimeout(() => {
                        if (window.innerWidth <= 768 && !document.activeElement.closest('#inputSection')) {
                            isInputFocused = false;
                            // Auto-masquage après 3 secondes si pas d'activité
                            setTimeout(() => {
                                if (!isInputFocused && window.innerWidth <= 768) {
                                    inputSection.classList.remove('show');
                                    chatContainer.classList.remove('input-active');
                                }
                            }, 3000);
                        }
                    }, 100);
                });
            });
            
            // Gestion du scroll intelligent sur mobile
            const messagesContainer = document.getElementById('messagesContainer');
            if (messagesContainer) {
                messagesContainer.addEventListener('scroll', () => {
                    if (window.innerWidth > 768) return;
                    
                    const scrollTop = messagesContainer.scrollTop;
                    const isScrollingDown = scrollTop > lastScrollTop;
                    
                    // Clearner le timeout précédent
                    clearTimeout(scrollTimeout);
                    
                    // Si on scroll vers le bas et qu'on n'est pas en train de taper
                    if (isScrollingDown && !isInputFocused) {
                        inputSection.classList.add('mobile-scroll-hide');
                        inputSection.classList.remove('show');
                        chatContainer.classList.remove('input-active');
                    }
                    // Si on scroll vers le haut, montrer temporairement
                    else if (!isScrollingDown) {
                        inputSection.classList.remove('mobile-scroll-hide');
                        inputSection.classList.add('show');
                        chatContainer.classList.add('input-active');
                        
                        // Auto-masquage après 2 secondes si pas d'interaction
                        scrollTimeout = setTimeout(() => {
                            if (!isInputFocused && window.innerWidth <= 768) {
                                inputSection.classList.add('mobile-scroll-hide');
                                inputSection.classList.remove('show');
                                chatContainer.classList.remove('input-active');
                            }
                        }, 2000);
                    }
                    
                    lastScrollTop = scrollTop;
                });
            }
            
            // Gestion du redimensionnement
            window.addEventListener('resize', () => {
                if (window.innerWidth > 768) {
                    // Mode desktop : toujours montrer l'input
                    inputSection.classList.remove('mobile-scroll-hide', 'show');
                    chatContainer.classList.remove('input-active');
                    isInputFocused = false;
                } else {
                    // Mode mobile : appliquer la logique mobile
                    if (!isInputFocused) {
                        inputSection.classList.add('mobile-scroll-hide');
                        inputSection.classList.remove('show');
                        chatContainer.classList.remove('input-active');
                    }
                }
            });
            
            // Initialisation de l'état selon la taille d'écran
            if (window.innerWidth <= 768) {
                inputSection.classList.add('mobile-scroll-hide');
            } else {
                inputSection.classList.add('show');
                chatContainer.classList.add('input-active');
            }
        }

        // Initialisation
        function initializePage() {
            // Configuration des inputs
            const messageInput = document.getElementById('messageInput');
            const welcomeMessageInput = document.getElementById('welcomeMessageInput');
            const sendButton = document.getElementById('sendButton');
            
            // Auto-resize textarea pour messageInput
            function setupTextarea(textarea, sendBtn = null) {
                if (!textarea) return;
                
                textarea.addEventListener('input', function() {
                    // Auto-resize
                    this.style.height = 'auto';
                    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
                    
                    if (sendBtn) {
                        // Mise à jour du compteur de caractères
                        const charCount = document.getElementById('charCount');
                        const currentLength = this.value.length;
                        const maxLength = 2000; // Réduit à 2000 caractères
                        
                        if (charCount) {
                            charCount.textContent = `${currentLength}/2000`;
                            
                            // Couleur progressive du compteur
                            if (currentLength > maxLength * 0.9) {
                                charCount.className = 'text-xs text-red-500 font-medium';
                            } else if (currentLength > maxLength * 0.7) {
                                charCount.className = 'text-xs text-orange-500';
                            } else {
                                charCount.className = 'text-xs text-gray-400';
                            }
                        }
                        
                        // État du bouton d'envoi
                        const hasText = this.value.trim().length > 0;
                        sendBtn.disabled = !(hasText && currentLength <= maxLength && !window.isLoading);
                    }
                });
                
                // Gestion Enter key
                textarea.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const form = this.closest('form');
                        if (form) {
                            form.dispatchEvent(new Event('submit', { cancelable: true }));
                        }
                    }
                });
            }
            
            // Configuration des champs de saisie
            setupTextarea(messageInput, sendButton);
            
            // Configuration simple pour welcomeMessageInput
            if (welcomeMessageInput) {
                welcomeMessageInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const form = this.closest('form');
                        if (form) {
                            form.dispatchEvent(new Event('submit', { cancelable: true }));
                        }
                    }
                });
            }
        }

        function setupEventListeners() {
            // Bouton nouvelle conversation
            const newChatBtn = document.getElementById('newChatBtn');
            if (newChatBtn) {
                newChatBtn.addEventListener('click', window.createNewConversation);
            }

            // Barre de recherche des conversations
            const searchInput = document.getElementById('conversationSearch');
            const clearSearch = document.getElementById('clearSearch');
            
            if (searchInput) {
                searchInput.addEventListener('input', function() {
                    const query = this.value.toLowerCase().trim();
                    window.filterConversations(query);
                    
                    // Afficher/masquer le bouton clear
                    if (clearSearch) {
                        clearSearch.classList.toggle('hidden', !query);
                    }
                });
            }
            
            if (clearSearch) {
                clearSearch.addEventListener('click', function() {
                    if (searchInput) {
                        searchInput.value = '';
                        searchInput.dispatchEvent(new Event('input'));
                        searchInput.focus();
                    }
                });
            }

            // Boutons d'actions rapides
            const templateBtn = document.getElementById('templateBtn');
            const emojiBtn = document.getElementById('emojiBtn');
            const attachmentBtn = document.getElementById('attachmentBtn');
            const voiceBtn = document.getElementById('voiceBtn');
            const closeTemplates = document.getElementById('closeTemplates');
            
            if (templateBtn) {
                templateBtn.addEventListener('click', window.toggleTemplates);
            }
            
            if (closeTemplates) {
                closeTemplates.addEventListener('click', window.hideTemplates);
            }
            
            // Templates de messages
            document.addEventListener('click', function(e) {
                if (e.target.classList.contains('template-btn')) {
                    const template = e.target.dataset.template;
                    const messageInput = document.getElementById('messageInput');
                    if (messageInput && template) {
                        messageInput.value = template;
                        messageInput.dispatchEvent(new Event('input'));
                        messageInput.focus();
                        window.hideTemplates();
                    }
                }
            });
            
            // Raccourcis clavier globaux
            document.addEventListener('keydown', function(e) {
                // Ctrl/Cmd + K : Focus sur la recherche
                if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                    e.preventDefault();
                    if (searchInput) {
                        searchInput.focus();
                        searchInput.select();
                    }
                }
                
                // Ctrl/Cmd + / : Nouvelle conversation
                if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                    e.preventDefault();
                    window.createNewConversation();
                }
                
                // Ctrl/Cmd + Enter : Envoyer le message (dans le champ de saisie)
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    const activeElement = document.activeElement;
                    if (activeElement && (activeElement.id === 'messageInput' || activeElement.id === 'welcomeMessageInput')) {
                        e.preventDefault();
                        const form = activeElement.closest('form');
                        if (form) {
                            form.dispatchEvent(new Event('submit', { cancelable: true }));
                        }
                    }
                }
                
                // Échap : Fermer les modales, templates et focus
                if (e.key === 'Escape') {
                    window.hideTemplates();
                    // Le reste est géré plus haut pour les modales
                }
            });

            // Amélioration de l'auto-scroll intelligent
            const messagesContainer = document.getElementById('messagesContainer');
            if (messagesContainer) {
                let userScrolledUp = false;
                let lastScrollTop = 0;
                
                messagesContainer.addEventListener('scroll', function() {
                    const scrollTop = this.scrollTop;
                    const scrollHeight = this.scrollHeight;
                    const clientHeight = this.clientHeight;
                    const isAtBottom = scrollTop + clientHeight >= scrollHeight - 100; // 100px de tolérance
                    
                    // Détecter si l'utilisateur scroll vers le haut
                    if (scrollTop < lastScrollTop && scrollTop > 100) {
                        userScrolledUp = true;
                    } else if (isAtBottom) {
                        userScrolledUp = false;
                    }
                    
                    lastScrollTop = scrollTop;
                    
                    // Afficher/masquer le bouton "scroll to bottom" si nécessaire
                    window.toggleScrollToBottomButton(!isAtBottom && userScrolledUp);
                });
            }

            // Formulaires de messages
            const messageForm = document.getElementById('messageForm');
            const welcomeForm = document.getElementById('welcomeForm');
            
            if (messageForm) {
                messageForm.addEventListener('submit', window.handleMessageSubmit);
            }
            
            if (welcomeForm) {
                welcomeForm.addEventListener('submit', window.handleWelcomeSubmit);
            }

            // Modale patient
            const assignPatientBtn = document.getElementById('assignPatientBtn');
            const assignPatientCancel = document.getElementById('assignPatientCancel');
            const assignPatientConfirm = document.getElementById('assignPatientConfirm');
            const closePatientModalBtn = document.getElementById('closePatientModal');
            const patientSelect = document.getElementById('patientSelect');
            
            if (assignPatientBtn) assignPatientBtn.addEventListener('click', window.openPatientModal);
            if (assignPatientCancel) assignPatientCancel.addEventListener('click', window.closePatientModal);
            if (assignPatientConfirm) assignPatientConfirm.addEventListener('click', window.assignPatient);
            if (closePatientModalBtn) closePatientModalBtn.addEventListener('click', window.closePatientModal);
            
            // Modale suppression conversation
            const closeDeleteModal = document.getElementById('closeDeleteModal');
            const cancelDeleteConversation = document.getElementById('cancelDeleteConversation');
            const confirmDeleteConversation = document.getElementById('confirmDeleteConversation');
            
            if (closeDeleteModal) closeDeleteModal.addEventListener('click', window.closeDeleteConversationModal);
            if (cancelDeleteConversation) cancelDeleteConversation.addEventListener('click', window.closeDeleteConversationModal);
            if (confirmDeleteConversation) confirmDeleteConversation.addEventListener('click', window.confirmDeleteConversation);
            
            // Gestion de la sélection de patient
            if (patientSelect) {
                patientSelect.addEventListener('change', function() {
                    const selectedValue = this.value;
                    const patientInfo = document.getElementById('patientInfo');
                    const selectedName = document.getElementById('selectedPatientName');
                    const selectedInfo = document.getElementById('selectedPatientInfo');
                    
                    if (selectedValue && this.selectedOptions[0]) {
                        const option = this.selectedOptions[0];
                        const patientName = option.textContent;
                        
                        if (selectedName) selectedName.textContent = patientName;
                        if (selectedInfo) selectedInfo.textContent = `ID: ${selectedValue}`;
                        if (patientInfo) patientInfo.classList.remove('hidden');
                    } else {
                        if (patientInfo) patientInfo.classList.add('hidden');
                    }
                });
            }
            
            // Fermer la modal en cliquant en dehors
            document.addEventListener('click', function(e) {
                const assignModal = document.getElementById('assignPatientModal');
                const deleteModal = document.getElementById('deleteConversationModal');
                const assignModalContent = assignModal?.querySelector('.bg-white');
                const deleteModalContent = deleteModal?.querySelector('.bg-white');
                
                // Fermer modale d'assignation
                if (assignModal && assignModal.style.display === 'flex' && 
                    !assignModalContent?.contains(e.target) && 
                    e.target !== document.getElementById('assignPatientBtn')) {
                    window.closePatientModal();
                }
                
                // Fermer modale de suppression
                if (deleteModal && deleteModal.style.display === 'flex' && 
                    !deleteModalContent?.contains(e.target)) {
                    window.closeDeleteConversationModal();
                }
            });
            
            // Fermer la modal avec la touche Escape
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    const assignModal = document.getElementById('assignPatientModal');
                    const deleteModal = document.getElementById('deleteConversationModal');
                    
                    if (assignModal && assignModal.style.display === 'flex') {
                        window.closePatientModal();
                    }
                    
                    if (deleteModal && deleteModal.style.display === 'flex') {
                        window.closeDeleteConversationModal();
                    }
                }
            });
        }

        // Fonction pour restaurer la conversation sauvegardée
        window.restoreSavedConversation = function() {
            const savedConversationId = localStorage.getItem('neuroscan-current-conversation');
            if (savedConversationId && window.conversations.length > 0) {
                const conversationExists = window.conversations.find(conv => conv.id.toString() === savedConversationId);
                if (conversationExists) {
                    // Restaurer la conversation sauvegardée (pas de parseInt pour supporter MongoDB ObjectId)
                    window.selectConversation(savedConversationId);
                } else {
                    // La conversation n'existe plus, nettoyer le localStorage
                    localStorage.removeItem('neuroscan-current-conversation');
                }
            }
        }

        // Analyse intelligente des titres de conversation
        window.analyzeConversationTitle = function(conversation) {
            const title = (conversation.title || '').toLowerCase();
            const lastMessage = (conversation.last_message || '').toLowerCase();

            // Mots-clés d'urgence médicale
            const urgentKeywords = [
                'urgence', 'urgent', 'douleur', 'chute', 'accident', 'hémorragie',
                'infarctus', 'avc', 'coma', 'crise', 'convulsion', 'étouffement',
                'brûlure', 'traumatisme', 'fracture', 'saignement'
            ];

            // Spécialités médicales
            const specialties = {
                cardiology: ['coeur', 'cardiaque', 'infarctus', 'angine', 'hypertension', 'arythmie'],
                neurology: ['cerveau', 'neurologique', 'céphalée', 'migraine', 'avc', 'épilepsie', 'parkinson'],
                pneumology: ['poumon', 'respiratoire', 'asthme', 'bronchite', 'pneumonie', 'dyspnée'],
                gastroenterology: ['digestif', 'estomac', 'foie', 'intestin', 'nausée', 'vomissement'],
                endocrinology: ['diabète', 'thyroïde', 'hormone', 'métabolique'],
                rheumatology: ['articulaire', 'rhumatisme', 'arthrose', 'arthrite'],
                dermatology: ['peau', 'dermatologique', 'eczéma', 'psoriasis'],
                psychiatry: ['psychologique', 'dépression', 'anxiété', 'stress', 'sommeil'],
                pediatry: ['enfant', 'pédiatrique', 'bébé', 'nourrisson'],
                gynecology: ['gynécologique', 'menstruation', 'grossesse', 'ménopause']
            };

            // Symptômes courants
            const symptoms = [
                'douleur', 'fièvre', 'fatigue', 'nausée', 'vomissement', 'diarrhée',
                'constipation', 'toux', 'rhume', 'mal de tête', 'migraine', 'vertige',
                'insomnie', 'anxiété', 'dépression', 'douleur thoracique', 'dyspnée'
            ];

            // Détection d'urgence
            const isUrgent = urgentKeywords.some(keyword =>
                title.includes(keyword) || lastMessage.includes(keyword)
            );

            // Détection de spécialité
            let detectedSpecialty = null;
            for (const [specialty, keywords] of Object.entries(specialties)) {
                if (keywords.some(keyword => title.includes(keyword) || lastMessage.includes(keyword))) {
                    detectedSpecialty = specialty;
                    break;
                }
            }

            // Détection de symptômes
            const detectedSymptoms = symptoms.filter(symptom =>
                title.includes(symptom) || lastMessage.includes(symptom)
            );

            return {
                isUrgent,
                specialty: detectedSpecialty ? window.formatSpecialtyName(detectedSpecialty) : null,
                symptoms: detectedSymptoms.slice(0, 3) // Maximum 3 symptômes
            };
        }

        // Formatage des noms de spécialités
        window.formatSpecialtyName = function(specialty) {
            const names = {
                cardiology: 'Cardio',
                neurology: 'Neuro',
                pneumology: 'Pneumo',
                gastroenterology: 'Digestif',
                endocrinology: 'Endocrino',
                rheumatology: 'Rhumato',
                dermatology: 'Dermato',
                psychiatry: 'Psych',
                pediatry: 'Pédiatrie',
                gynecology: 'Gynéco'
            };
            return names[specialty] || specialty;
        }

        // Génération d'icônes spécialisées selon le type de consultation
        window.getConversationIcon = function(conversation, analysis) {
            const title = (conversation.title || '').toLowerCase();
            const lastMessage = (conversation.last_message || '').toLowerCase();

            // Icônes selon la spécialité détectée
            if (analysis.specialty) {
                const specialtyIcons = {
                    'Cardio': '<i data-feather="heart" class="w-4 h-4 text-red-500"></i>',
                    'Neuro': '<i data-feather="cpu" class="w-4 h-4 text-purple-500"></i>',
                    'Pneumo': '<i data-feather="wind" class="w-4 h-4 text-blue-500"></i>',
                    'Digestif': '<i data-feather="activity" class="w-4 h-4 text-green-500"></i>',
                    'Endocrino': '<i data-feather="zap" class="w-4 h-4 text-yellow-500"></i>',
                    'Rhumatologie': '<i data-feather="bone" class="w-4 h-4 text-orange-500"></i>',
                    'Dermato': '<i data-feather="layers" class="w-4 h-4 text-pink-500"></i>',
                    'Psych': '<i data-feather="user" class="w-4 h-4 text-indigo-500"></i>',
                    'Pédiatrie': '<i data-feather="baby" class="w-4 h-4 text-cyan-500"></i>',
                    'Gynéco': '<i data-feather="flower" class="w-4 h-4 text-rose-500"></i>'
                };
                return specialtyIcons[analysis.specialty] || '<i data-feather="activity" class="w-4 h-4 text-medical-500"></i>';
            }

            // Icône d'urgence pour les cas urgents
            if (analysis.isUrgent) {
                return '<i data-feather="alert-triangle" class="w-4 h-4 text-red-500 animate-pulse"></i>';
            }

            // Icônes selon le contenu du message
            if (lastMessage.includes('imagerie') || lastMessage.includes('scanner') || lastMessage.includes('irm')) {
                return '<i data-feather="image" class="w-4 h-4 text-blue-500"></i>';
            }

            if (lastMessage.includes('diagnostic') || lastMessage.includes('analyse')) {
                return '<i data-feather="search" class="w-4 h-4 text-green-500"></i>';
            }

            if (lastMessage.includes('traitement') || lastMessage.includes('prescription')) {
                return '<i data-feather="pill" class="w-4 h-4 text-purple-500"></i>';
            }

            // Icône par défaut pour les consultations médicales
            return '<i data-feather="activity" class="w-4 h-4 text-primary-600"></i>';
        }

        // Indicateur médical intelligent
        window.getMedicalIndicator = function(message) {
            if (!message || message === 'Nouvelle consultation') {
                return '<i data-feather="plus-circle" class="w-3 h-3 text-gray-400 mr-1"></i>';
            }

            const lowerMessage = message.toLowerCase();

            // Indicateurs selon le type de contenu médical
            if (lowerMessage.includes('diagnostic') || lowerMessage.includes('analyse')) {
                return '<i data-feather="search" class="w-3 h-3 text-blue-500 mr-1"></i>';
            }

            if (lowerMessage.includes('traitement') || lowerMessage.includes('prescription')) {
                return '<i data-feather="pill" class="w-3 h-3 text-green-500 mr-1"></i>';
            }

            if (lowerMessage.includes('urgence') || lowerMessage.includes('urgent')) {
                return '<i data-feather="alert-circle" class="w-3 h-3 text-red-500 mr-1"></i>';
            }

            if (lowerMessage.includes('question') || lowerMessage.includes('?')) {
                return '<i data-feather="help-circle" class="w-3 h-3 text-orange-500 mr-1"></i>';
            }

            // Indicateur par défaut
            return '<i data-feather="message-circle" class="w-3 h-3 text-gray-500 mr-1"></i>';
        }

        // Résumé intelligent des messages
        window.summarizeMessage = function(message) {
            if (!message || message === 'Nouvelle consultation') {
                return 'Nouvelle consultation';
            }

            // Si le message est court, le retourner tel quel
            if (message.length <= 60) {
                return message;
            }

            // Chercher des mots-clés médicaux importants
            const medicalKeywords = [
                'diagnostic', 'traitement', 'symptôme', 'douleur', 'fièvre', 'analyse',
                'prescription', 'consultation', 'examen', 'résultat', 'urgence'
            ];

            const lowerMessage = message.toLowerCase();
            const foundKeywords = medicalKeywords.filter(keyword =>
                lowerMessage.includes(keyword)
            );

            // Si on trouve des mots-clés, créer un résumé intelligent
            if (foundKeywords.length > 0) {
                const firstKeyword = foundKeywords[0];
                const keywordIndex = lowerMessage.indexOf(firstKeyword);
                const contextStart = Math.max(0, keywordIndex - 20);
                const contextEnd = Math.min(message.length, keywordIndex + firstKeyword.length + 30);

                let summary = message.substring(contextStart, contextEnd);
                if (contextStart > 0) summary = '...' + summary;
                if (contextEnd < message.length) summary = summary + '...';

                return summary;
            }

            // Résumé par défaut : début du message
            return message.substring(0, 57) + '...';
        }

        // Fonction de renommage des conversations
        window.renameConversation = async function(conversationId, currentTitle) {
            const newTitle = prompt('Renommer la consultation:', currentTitle);
            if (!newTitle || newTitle.trim() === currentTitle) return;

            try {
                const response = await fetch(`/api/chat/conversations/${conversationId}/rename`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        title: newTitle.trim()
                    })
                });

                const data = await response.json();
                if (data.success) {
                    await window.loadConversations();
                    window.showNotification('Consultation renommée avec succès', 'success');
                } else {
                    window.showNotification('Erreur lors du renommage', 'error');
                }
            } catch (error) {
                console.error('Erreur renommage:', error);
                window.showNotification('Erreur réseau', 'error');
            }
        }

        // Analyse d'un message pour générer un titre intelligent
        window.analyzeMessageForTitle = function(message) {
            if (!message || typeof message !== 'string' || message.length < 10) {
                return { suggestedTitle: null };
            }

            const lowerMessage = message.toLowerCase();

            // Patterns de titres médicaux courants
            const titlePatterns = [
                {
                    keywords: ['mal de tête', 'céphalée', 'migraine', 'maux de tête'],
                    title: 'Céphalées et migraines'
                },
                {
                    keywords: ['douleur thoracique', 'douleur poitrine', 'infarctus', 'angine'],
                    title: 'Douleurs thoraciques'
                },
                {
                    keywords: ['douleur abdominale', 'douleur ventre', 'nausée', 'vomissement'],
                    title: 'Douleurs abdominales'
                },
                {
                    keywords: ['fièvre', 'température', 'infection'],
                    title: 'Fièvre et infection'
                },
                {
                    keywords: ['toux', 'respiratoire', 'poumon', 'asthme', 'bronchite'],
                    title: 'Problèmes respiratoires'
                },
                {
                    keywords: ['articulaire', 'rhumatisme', 'arthrose', 'douleur articulaire'],
                    title: 'Douleurs articulaires'
                },
                {
                    keywords: ['peau', 'dermatologique', 'eczéma', 'allergie'],
                    title: 'Problèmes dermatologiques'
                },
                {
                    keywords: ['diabète', 'glycémie', 'insuline'],
                    title: 'Suivi diabète'
                },
                {
                    keywords: ['hypertension', 'tension artérielle'],
                    title: 'Hypertension artérielle'
                },
                {
                    keywords: ['dépression', 'anxiété', 'stress', 'sommeil'],
                    title: 'Santé mentale'
                },
                {
                    keywords: ['enfant', 'pédiatrique', 'bébé', 'nourrisson'],
                    title: 'Consultation pédiatrique'
                },
                {
                    keywords: ['gynécologique', 'menstruation', 'grossesse'],
                    title: 'Consultation gynécologique'
                },
                {
                    keywords: ['imagerie', 'scanner', 'irm', 'radio'],
                    title: 'Analyse d\'imagerie'
                },
                {
                    keywords: ['diagnostic', 'analyse', 'résultat'],
                    title: 'Analyse diagnostique'
                },
                {
                    keywords: ['traitement', 'prescription', 'médicament'],
                    title: 'Prescription et traitement'
                }
            ];

            // Chercher le pattern le plus pertinent
            for (const pattern of titlePatterns) {
                if (pattern.keywords.some(keyword => lowerMessage.includes(keyword))) {
                    return { suggestedTitle: pattern.title };
                }
            }

            // Si aucun pattern spécifique, créer un titre basé sur les premiers mots significatifs
            const words = message.split(' ').filter(word => word.length > 3);
            if (words.length >= 2) {
                const titleWords = words.slice(0, 4).join(' ');
                const suggestedTitle = titleWords.charAt(0).toUpperCase() + titleWords.slice(1).toLowerCase();
                return { suggestedTitle: suggestedTitle.length > 50 ? suggestedTitle.substring(0, 47) + '...' : suggestedTitle };
            }

            // Titre par défaut
            return { suggestedTitle: null };
        }

        // ===== FIN DES NOUVELLES FONCTIONS =====
        
        // Fonction de recherche dans les conversations
        window.filterConversations = function(query) {
            const conversationItems = document.querySelectorAll('.conversation-item');
            let hasResults = false;
            
            conversationItems.forEach(item => {
                const title = item.querySelector('h4')?.textContent.toLowerCase() || '';
                const subtitle = item.querySelector('p')?.textContent.toLowerCase() || '';
                const patientName = item.querySelector('.patient-name')?.textContent.toLowerCase() || '';
                
                const matches = title.includes(query) || 
                               subtitle.includes(query) || 
                               patientName.includes(query);
                
                item.style.display = matches ? 'block' : 'none';
                if (matches) hasResults = true;
            });
            
            // Afficher un message si aucune conversation ne correspond
            const container = document.getElementById('conversationsList');
            const existingNoResults = container.querySelector('.no-results');
            
            if (!hasResults && query) {
                if (!existingNoResults) {
                    const noResultsDiv = document.createElement('div');
                    noResultsDiv.className = 'no-results text-center text-gray-500 py-8';
                    noResultsDiv.innerHTML = `
                        <i data-feather="search" class="w-8 h-8 mx-auto mb-3 text-gray-400"></i>
                        <p class="font-medium">Aucune conversation trouvée</p>
                        <p class="text-sm">Essayez avec des termes différents</p>
                    `;
                    container.appendChild(noResultsDiv);
                    feather.replace();
                }
            } else if (existingNoResults) {
                existingNoResults.remove();
            }
        }
        
        // Gestion des templates de messages
        window.toggleTemplates = function() {
            const templates = document.getElementById('messageTemplates');
            if (templates) {
                const isVisible = !templates.classList.contains('hidden');
                if (isVisible) {
                    hideTemplates();
                } else {
                    showTemplates();
                }
            }
        }
        
        function showTemplates() {
            const templates = document.getElementById('messageTemplates');
            if (templates) {
                templates.classList.remove('hidden');
                templates.style.animation = 'slideDown 0.2s ease-out';
            }
        }
        
        window.hideTemplates = function() {
            const templates = document.getElementById('messageTemplates');
            if (templates) {
                templates.classList.add('hidden');
            }
        }
        
        // Bouton "scroll to bottom" intelligent
        window.toggleScrollToBottomButton = function(show) {
            let scrollBtn = document.getElementById('scrollToBottomBtn');
            
            if (show) {
                if (!scrollBtn) {
                    scrollBtn = document.createElement('button');
                    scrollBtn.id = 'scrollToBottomBtn';
                    scrollBtn.className = 'fixed bottom-20 right-4 bg-primary-600 hover:bg-primary-700 text-white p-3 rounded-full shadow-lg transition-all duration-300 z-40 opacity-0 transform translate-y-4';
                    scrollBtn.innerHTML = '<i data-feather="chevron-down" class="w-5 h-5"></i>';
                    scrollBtn.title = 'Aller au bas de la conversation';
                    scrollBtn.addEventListener('click', scrollToBottom);
                    document.body.appendChild(scrollBtn);
                    feather.replace();
                    
                    // Animation d'apparition
                    setTimeout(() => {
                        scrollBtn.classList.remove('opacity-0', 'translate-y-4');
                    }, 10);
                }
            } else if (scrollBtn) {
                scrollBtn.classList.add('opacity-0', 'translate-y-4');
                setTimeout(() => {
                    if (scrollBtn.parentNode) {
                        scrollBtn.parentNode.removeChild(scrollBtn);
                    }
                }, 300);
            }
        }
        
        function scrollToBottom() {
            const container = document.getElementById('messagesContainer');
            if (container) {
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });
            }
        }
        
        // Amélioration de l'auto-scroll avec détection intelligente
        function smartScrollToBottom(force = false) {
            const container = document.getElementById('messagesContainer');
            if (!container) return;
            
            const scrollTop = container.scrollTop;
            const scrollHeight = container.scrollHeight;
            const clientHeight = container.clientHeight;
            const isAtBottom = scrollTop + clientHeight >= scrollHeight - 100;
            
            // Scroll automatique seulement si l'utilisateur n'a pas scrollé vers le haut
            // ou si c'est forcé (nouveau message)
            if (isAtBottom || force) {
                setTimeout(() => {
                    container.scrollTo({
                        top: container.scrollHeight,
                        behavior: 'smooth'
                    });
                }, 100);
            }
        }
        
        // Notifications améliorées avec actions
        function showEnhancedNotification(message, type = 'info', actions = null) {
            const notification = document.createElement('div');
            notification.className = `fixed top-4 right-4 px-6 py-4 rounded-lg shadow-lg z-50 max-w-sm notification-enhanced ${
                type === 'success' ? 'bg-medical-500 text-white' : 
                type === 'error' ? 'bg-red-500 text-white' : 
                type === 'warning' ? 'bg-orange-500 text-white' :
                'bg-primary-500 text-white'
            }`;
            
            const icon = type === 'success' ? 'check-circle' : 
                        type === 'error' ? 'alert-circle' : 
                        type === 'warning' ? 'alert-triangle' :
                        'info';
            
            let actionsHTML = '';
            if (actions && actions.length > 0) {
                actionsHTML = '<div class="flex gap-2 mt-3">' + 
                    actions.map(action => 
                        `<button class="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-sm transition-colors" onclick="${action.callback}">${action.label}</button>`
                    ).join('') + '</div>';
            }
            
            notification.innerHTML = `
                <div class="flex items-start gap-3">
                    <i data-feather="${icon}" class="w-5 h-5 flex-shrink-0 mt-0.5"></i>
                    <div class="flex-1">
                        <span class="text-sm font-medium block">${message}</span>
                        ${actionsHTML}
                    </div>
                    <button class="text-white/70 hover:text-white transition-colors" onclick="this.parentElement.parentElement.remove()">
                        <i data-feather="x" class="w-4 h-4"></i>
                    </button>
                </div>
            `;
            
            document.body.appendChild(notification);
            feather.replace();
            
            // Animation d'entrée
            setTimeout(() => notification.classList.add('fade-in'), 10);
            
            // Supprimer automatiquement après 5 secondes (sauf si warning/error)
            const autoRemoveDelay = type === 'error' || type === 'warning' ? 8000 : 5000;
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    notification.style.transform = 'translateX(100%)';
                    notification.style.opacity = '0';
                    setTimeout(() => {
                        if (document.body.contains(notification)) {
                            document.body.removeChild(notification);
                        }
                    }, 300);
                }
            }, autoRemoveDelay);
        }

        window.handleWelcomeSubmit = async function(e) {
            e.preventDefault();
            
            const welcomeInput = document.getElementById('welcomeMessageInput');
            const message = welcomeInput.value.trim();
            
            if (!message) return;
            
            // Créer une nouvelle conversation avec titre intelligent
            await window.createNewConversation(message);
            
            // Vider le champ et masquer l'input d'accueil
            welcomeInput.value = '';
        }

        // Fonctions principales
        window.loadConversations = async function() {
            try {
                const response = await fetch('/api/chat/conversations');
                const data = await response.json();

                if (data.success) {
                    window.conversations = data.conversations;
                    window.renderConversations();
                    
                    // Restaurer la conversation sauvegardée après le chargement
                    window.restoreSavedConversation();
                } else {
                    console.error('Erreur chargement conversations:', data.error);
                }
            } catch (error) {
                console.error('Erreur réseau:', error);
            }
        }

        window.renderConversations = function() {
            const container = document.getElementById('conversationsList');

            if (window.conversations.length === 0) {
                container.innerHTML = `
                    <div class="text-center text-gray-500 py-8">
                        <i data-feather="message-circle" class="w-8 h-8 mx-auto mb-3 text-gray-400"></i>
                        <p class="font-medium">Aucune consultation</p>
                        <p class="text-sm">Commencez une nouvelle consultation</p>
                    </div>
                `;
                feather.replace();
                return;
            }

            container.innerHTML = window.conversations.map(conv => {
                // Analyse intelligente du titre et génération d'informations médicales
                const titleAnalysis = window.analyzeConversationTitle(conv);
                const isUrgent = titleAnalysis.isUrgent;
                const specialty = titleAnalysis.specialty;
                const symptoms = titleAnalysis.symptoms;

                return `
                <div class="conversation-item group cursor-pointer p-3 rounded-xl border border-transparent hover:border-gray-200 transition-all duration-200 ${String(conv.id) === String(window.currentConversationId) ? 'active bg-primary-100 border-primary-200 shadow-sm' : 'hover:bg-gray-50'}"
                     data-id="${conv.id}">
                    <div class="flex justify-between items-start">
                        <div class="flex-1 min-w-0">
                            <!-- En-tête avec icône spécialisée et indicateurs -->
                            <div class="flex items-center gap-2 mb-2">
                                <div class="flex-shrink-0">
                                    ${window.getConversationIcon(conv, titleAnalysis)}
                                </div>
                                <div class="flex-1 min-w-0">
                                    <div class="flex items-center gap-2 mb-1">
                                        <h4 class="font-semibold text-gray-900 leading-tight break-words line-clamp-2 flex-1"
                                            title="${escapeHtml(conv.title)}"
                                            ondblclick="renameConversation(${conv.id}, '${escapeHtml(conv.title)}')">
                                            ${escapeHtml(conv.title)}
                                        </h4>
                                        ${isUrgent ? '<span class="px-1.5 py-0.5 bg-red-100 text-red-700 text-xs font-medium rounded-full animate-pulse">URGENT</span>' : ''}
                                        ${specialty ? `<span class="px-1.5 py-0.5 bg-medical-100 text-medical-700 text-xs font-medium rounded-full">${specialty}</span>` : ''}
                                    </div>
                                </div>
                            </div>

                            <!-- Informations patient et symptômes -->
                            ${conv.patient_name ? `
                                <div class="flex items-center gap-2 mb-1">
                                    <p class="text-sm text-medical-700 font-medium flex items-center gap-1 truncate">
                                        <i data-feather="user" class="w-3 h-3 flex-shrink-0"></i>
                                        ${escapeHtml(conv.patient_name)}
                                    </p>
                                    ${symptoms.length > 0 ? `
                                        <div class="flex gap-1">
                                            ${symptoms.slice(0, 2).map(symptom =>
                                                `<span class="px-1.5 py-0.5 bg-orange-100 text-orange-700 text-xs rounded-full">${symptom}</span>`
                                            ).join('')}
                                            ${symptoms.length > 2 ? `<span class="px-1.5 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">+${symptoms.length - 2}</span>` : ''}
                                        </div>
                                    ` : ''}
                                </div>
                            ` : ''}

                            <!-- Aperçu du dernier message avec indicateur médical -->
                            <p class="text-sm text-gray-600 truncate mb-2 flex items-center gap-1">
                                ${window.getMedicalIndicator(conv.last_message || 'Nouvelle consultation')}
                                <span class="truncate">${escapeHtml(window.summarizeMessage(conv.last_message || 'Nouvelle consultation'))}</span>
                            </p>

                            <!-- Métadonnées améliorées -->
                            <div class="flex items-center justify-between">
                                <div class="flex items-center space-x-2 text-xs text-gray-500">
                                    <i data-feather="clock" class="w-3 h-3"></i>
                                    <span>${formatDate(conv.updated_at)}</span>
                                    ${conv.patient_id ? '<span class="w-1.5 h-1.5 bg-medical-500 rounded-full" title="Patient assigné"></span>' : ''}
                                    ${conv.message_count > 5 ? '<span class="px-1.5 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">Active</span>' : ''}
                                </div>
                                <div class="flex items-center gap-2">
                                    <span class="text-xs text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded">
                                        ${conv.message_count || 0}
                                    </span>
                                    <button onclick="deleteConversation(event, ${conv.id})"
                                            class="delete-conversation-btn opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-600 p-1.5 rounded-lg transition-all hover:bg-red-50 hover:shadow-sm"
                                            title="Supprimer la conversation">
                                        <i data-feather="trash-2" class="w-3.5 h-3.5"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `}).join('');

            // Remplacer les icônes feather
            feather.replace();

            // Ajouter les event listeners
            container.querySelectorAll('.conversation-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    if (e.target.closest('.delete-conversation-btn')) {
                        return;
                    }
                    const id = item.dataset.id; // Garder comme string pour supporter MongoDB ObjectId
                    window.selectConversation(id);
                });

                // Afficher le bouton de suppression au survol
                item.addEventListener('mouseenter', () => {
                    const deleteBtn = item.querySelector('.delete-conversation-btn');
                    if (deleteBtn) deleteBtn.classList.remove('opacity-0');
                });

                item.addEventListener('mouseleave', () => {
                    const deleteBtn = item.querySelector('.delete-conversation-btn');
                    if (deleteBtn) deleteBtn.classList.add('opacity-0');
                });
            });
        }

        window.selectConversation = async function(conversationId) {
            // Convertir en string pour supporter MongoDB ObjectId
            conversationId = String(conversationId);
            
            if (String(window.currentConversationId) === conversationId) return;

            window.currentConversationId = conversationId;
            
            // Sauvegarder la conversation actuelle dans le localStorage
            localStorage.setItem('neuroscan-current-conversation', conversationId.toString());
            
            // Fermer la sidebar mobile après sélection
            if (window.innerWidth <= 768) {
                const sidebar = document.getElementById('sidebar');
                const overlay = document.getElementById('sidebarOverlay');
                if (sidebar) sidebar.classList.add('sidebar-hidden');
                if (overlay) overlay.classList.remove('show');
            }
            
            // Mettre à jour l'interface
            window.updateConversationHeader();
            window.renderConversations();
            
            // Charger les messages
            await window.loadMessages();
            
            // Basculer vers l'interface de chat
            document.getElementById('welcomeMessages').style.display = 'none';
            document.getElementById('welcomeInput').style.display = 'none';
            document.getElementById('chatInputArea').classList.remove('hidden');
            document.getElementById('conversationHeader').classList.remove('hidden');
            
            // Afficher le bouton d'assignation de patient
            const assignBtn = document.getElementById('assignPatientBtn');
            if (assignBtn) {
                assignBtn.classList.remove('hidden');
            }

            // Focus sur le champ de saisie
            setTimeout(() => {
                const messageInput = document.getElementById('messageInput');
                if (messageInput) messageInput.focus();
            }, 100);
        }

        window.updateConversationHeader = function() {
            const conversation = window.conversations.find(c => c.id === window.currentConversationId);
            if (!conversation) return;

            document.getElementById('conversationTitle').textContent = conversation.title;
            
            let subtitleText = 'Aucun patient assigné';
            if (conversation.patient_id && conversation.patient_name) {
                subtitleText = `👤 Patient: ${conversation.patient_name}`;
            } else if (conversation.patient_id) {
                subtitleText = `Patient ID: ${conversation.patient_id}`;
            }
            
            document.getElementById('conversationSubtitle').textContent = subtitleText;
        }

        window.loadMessages = async function() {
            if (!window.currentConversationId) return;

            try {
                const response = await fetch(`/api/chat/conversations/${window.currentConversationId}/messages`);
                const data = await response.json();

                if (data.success) {
                    window.renderMessages(data.messages);
                } else {
                    console.error('Erreur chargement messages:', data.error);
                }
            } catch (error) {
                console.error('Erreur réseau:', error);
            }
        }

        window.renderMessages = function(messages) {
            const container = document.getElementById('messagesContainer');
            
            // Supprimer les anciens messages
            container.innerHTML = '';

            // Ajouter les nouveaux messages
            messages.forEach(message => {
                window.appendMessage(message, false); // false = ne pas scroller pendant le chargement
            });

            // Scroll intelligent vers le bas après le chargement
            smartScrollToBottom(true);
        }

        window.appendMessage = function(message, shouldScroll = true) {
            const container = document.getElementById('messagesContainer');
            const isUser = message.role === 'user';
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message message-bubble ${isUser ? 'user' : 'assistant'}`;
            messageDiv.setAttribute('data-message-id', message.id || 'temp-' + Date.now());
            
            // Créer l'avatar
            const avatarContent = isUser ? 
                '<i data-feather="user" class="w-3 h-3"></i>' : 
                '<i data-feather="cpu" class="w-3 h-3"></i>';
            
            // Créer les actions selon le type de message
            const messageActions = isUser && message.id ? `
                <div class="message-actions">
                    <button onclick="editMessage(${message.id})" title="Éditer ce message">
                        <i data-feather="edit-2" class="w-3 h-3"></i>
                    </button>
                    <button onclick="regenerateResponse(${message.id})" title="Régénérer la réponse">
                        <i data-feather="refresh-cw" class="w-3 h-3"></i>
                    </button>
                </div>
            ` : '';
            
            // Créer les indicateurs spéciaux
            const indicators = [];
            if (!isUser && message.is_medical_query) {
                indicators.push('<div class="message-indicator medical-indicator"><i data-feather="heart-pulse" class="w-3 h-3"></i><span>Analyse médicale</span></div>');
            }
            if (!isUser && message.confidence_score) {
                const confidencePercent = Math.round(message.confidence_score * 100);
                indicators.push(`
                    <div class="message-indicator confidence-indicator">
                        <i data-feather="bar-chart-2" class="w-3 h-3"></i>
                        <span>Confiance: ${confidencePercent}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                    </div>
                `);
            }
            if (message.is_edited) {
                indicators.push('<div class="message-indicator"><i data-feather="edit-3" class="w-3 h-3"></i><span>Édité</span></div>');
            }
            
            messageDiv.innerHTML = `
                <div class="message-content ${message.is_edited ? 'message-edited' : ''}">
                    <div class="message-header">
                        <div class="avatar">${avatarContent}</div>
                        <span class="sender-name">${isUser ? 'Vous' : 'NeuroScan IA'}</span>
                        <span class="timestamp">${window.formatTime(message.timestamp)}</span>
                    </div>
                    <div class="prose prose-sm max-w-none" id="message-content-${message.id || 'temp'}">
                        ${window.formatMessageContent(message.content)}
                    </div>
                    ${indicators.length > 0 ? `<div class="mt-2">${indicators.join('')}</div>` : ''}
                </div>
                ${messageActions}
            `;
            
            container.appendChild(messageDiv);
            
            // Remplacer les icônes feather
            feather.replace();
            
            // Animer l'apparition
            setTimeout(() => {
                messageDiv.classList.add('visible');
            }, 50);
            
            // Utiliser le scroll intelligent
            if (shouldScroll) {
                smartScrollToBottom(true); // true = forcer le scroll pour les nouveaux messages
            }
        }

        window.handleMessageSubmit = async function(e) {
            e.preventDefault();
            
            if (window.isLoading || !window.currentConversationId) return;
            
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            
            if (!message) return;
            
            // Désactiver l'interface
            window.setLoading(true);
            messageInput.value = '';
            messageInput.style.height = 'auto'; // Réinitialiser la hauteur
            const charCount = document.getElementById('charCount');
            if (charCount) charCount.textContent = '0/2000';
            
            // Afficher le message utilisateur
            window.appendMessage({
                role: 'user',
                content: message,
                timestamp: new Date().toISOString()
            });
            
            // Afficher l'indicateur de frappe moderne
            const typingIndicator = document.getElementById('typingIndicatorModern');
            if (typingIndicator) typingIndicator.classList.remove('hidden');
            
            try {
                console.log('📤 Envoi message - conversation_id:', window.currentConversationId, 'message:', message.substring(0, 50));
                
                const response = await fetch('/api/chat/send', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        conversation_id: window.currentConversationId,
                        message: message
                    })
                });
                
                const data = await response.json();
                
                // Masquer l'indicateur de frappe moderne
                const typingIndicator = document.getElementById('typingIndicatorModern');
                if (typingIndicator) typingIndicator.classList.add('hidden');
                
                if (data.success || data.response) {
                    // Afficher la réponse de l'assistant
                    window.appendMessage({
                        role: 'assistant',
                        content: data.response,
                        timestamp: new Date().toISOString(),
                        is_medical_query: data.is_medical,
                        confidence_score: data.confidence_score
                    });
                } else {
                    throw new Error(data.error || 'Erreur inconnue');
                }
                
            } catch (error) {
                console.error('Erreur envoi message:', error);
                const typingIndicator = document.getElementById('typingIndicatorModern');
                if (typingIndicator) typingIndicator.classList.add('hidden');
                
                // Afficher un message d'erreur amélioré avec suggestions
                const errorMessage = error.message || 'Erreur inconnue';
                let userFriendlyMessage = 'Désolé, une erreur est survenue.';
                let suggestions = [];
                
                if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
                    userFriendlyMessage = 'Problème de connexion réseau.';
                    suggestions = [
                        { label: 'Réessayer', callback: 'retryLastMessage()' },
                        { label: 'Vérifier connexion', callback: 'checkConnection()' }
                    ];
                } else if (errorMessage.includes('timeout')) {
                    userFriendlyMessage = 'La requête a pris trop de temps.';
                    suggestions = [
                        { label: 'Réessayer', callback: 'retryLastMessage()' }
                    ];
                } else if (errorMessage.includes('rate limit')) {
                    userFriendlyMessage = 'Trop de requêtes. Veuillez patienter.';
                    suggestions = [
                        { label: 'Attendre 30s', callback: 'scheduleRetry(30000)' }
                    ];
                }
                
                showEnhancedNotification(userFriendlyMessage, 'error', suggestions);
                
                // Afficher le message d'erreur dans le chat
                window.appendMessage({
                    role: 'assistant',
                    content: `${userFriendlyMessage} ${suggestions.length > 0 ? 'Essayez les actions suggérées ci-dessus.' : 'Veuillez réessayer dans quelques instants.'}`,
                    timestamp: new Date().toISOString(),
                    is_medical_query: false,
                    confidence_score: 0
                });
            }
            
            // Réactiver l'interface
            window.setLoading(false);
        }

        window.setLoading = function(loading) {
            window.isLoading = loading;
            const sendButton = document.getElementById('sendButton');
            const sendIcon = document.getElementById('sendIcon');
            const messageInput = document.getElementById('messageInput');
            
            if (loading) {
                sendButton.disabled = true;
                if (sendIcon) {
                    sendIcon.setAttribute('data-feather', 'loader');
                    sendIcon.classList.add('animate-spin');
                    feather.replace();
                }
                if (messageInput) {
                    messageInput.disabled = true;
                    messageInput.classList.add('opacity-50');
                }
            } else {
                if (sendIcon) {
                    sendIcon.setAttribute('data-feather', 'send');
                    sendIcon.classList.remove('animate-spin');
                    feather.replace();
                }
                if (messageInput) {
                    messageInput.disabled = false;
                    messageInput.classList.remove('opacity-50');
                    
                    // Réactiver le bouton selon le contenu
                    const hasText = messageInput.value.trim().length > 0;
                    sendButton.disabled = !(hasText && messageInput.value.length <= 2000);
                    
                    messageInput.focus();
                }
            }
        }

        window.createNewConversation = async function(initialMessage = null) {
            try {
                // Générer un titre intelligent basé sur le message initial
                let title = `Consultation ${new Date().toLocaleDateString('fr-FR')}`;

                if (initialMessage) {
                    // Analyser le message pour créer un titre descriptif
                    const analysis = window.analyzeMessageForTitle(initialMessage);
                    if (analysis.suggestedTitle) {
                        title = analysis.suggestedTitle;
                    }
                }

                const response = await fetch('/api/chat/conversations', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        title: title
                    })
                });

                const data = await response.json();
                if (data.success) {
                    await window.loadConversations();
                    window.selectConversation(data.conversation_id);

                    // Si on a un message initial, l'envoyer automatiquement
                    if (initialMessage) {
                        setTimeout(async () => {
                            try {
                                const sendResponse = await fetch('/api/chat/send', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json'
                                    },
                                    body: JSON.stringify({
                                        conversation_id: data.conversation_id,
                                        message: initialMessage
                                    })
                                });

                                const sendData = await sendResponse.json();
                                if (sendData.success) {
                                    await window.loadMessages();
                                }
                            } catch (error) {
                                console.error('Erreur envoi message initial:', error);
                            }
                        }, 500);
                    }

                    window.showNotification('Nouvelle consultation créée', 'success');
                } else {
                    window.showNotification('Erreur lors de la création de la consultation', 'error');
                }
            } catch (error) {
                console.error('Erreur création conversation:', error);
                window.showNotification('Erreur réseau', 'error');
            }
        }

        window.showNotification = function(message, type = 'info') {
            // Créer une notification temporaire style moderne
            const notification = document.createElement('div');
            notification.className = `fixed top-4 right-4 px-6 py-4 rounded-lg shadow-lg z-50 flex items-center gap-3 max-w-sm ${
                type === 'success' ? 'bg-medical-500 text-white' : 
                type === 'error' ? 'bg-red-500 text-white' : 
                'bg-primary-500 text-white'
            } notification`;
            
            const icon = type === 'success' ? 'check-circle' : 
                        type === 'error' ? 'alert-circle' : 'info';
            
            notification.innerHTML = `
                <i data-feather="${icon}" class="w-5 h-5 flex-shrink-0"></i>
                <span class="text-sm font-medium">${message}</span>
            `;
            
            document.body.appendChild(notification);
            feather.replace();
            
            // Animation d'entrée
            setTimeout(() => notification.classList.add('fade-in'), 10);
            
            // Supprimer après 4 secondes avec animation de sortie
            setTimeout(() => {
                notification.style.transform = 'translateX(100%)';
                notification.style.opacity = '0';
                setTimeout(() => {
                    if (document.body.contains(notification)) {
                        document.body.removeChild(notification);
                    }
                }, 300);
            }, 4000);
        }

        // Initialiser l'application
        initializePage();
        setupEventListeners();
        
        // Initialiser les animations du profil médecin
        initializeDoctorProfile();

        // ===== GESTION DU MODE SOMBRE =====
        
        function initializeTheme() {
            // Récupérer la préférence sauvegardée ou détecter la préférence système
            const savedTheme = localStorage.getItem('neuroscan-theme');
            const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            
            // Définir le thème initial
            const initialTheme = savedTheme || (systemPrefersDark ? 'dark' : 'light');
            setTheme(initialTheme, true); // true = mode silencieux (pas de notification)
            
            // Event listeners pour les toggles
            const themeToggle = document.getElementById('themeToggle');
            const themeToggleMobile = document.getElementById('themeToggleMobile');
            
            if (themeToggle) {
                themeToggle.addEventListener('click', toggleTheme);
            }
            
            if (themeToggleMobile) {
                themeToggleMobile.addEventListener('click', toggleTheme);
            }
            
            // Écouter les changements de préférence système
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
                if (!localStorage.getItem('neuroscan-theme')) {
                    setTheme(e.matches ? 'dark' : 'light', true);
                }
            });
        }
        
        function toggleTheme() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            setTheme(newTheme);
        }
        
        function setTheme(theme, silent = false) {
            // Appliquer le thème
            document.documentElement.setAttribute('data-theme', theme);
            
            // Sauvegarder la préférence
            localStorage.setItem('neuroscan-theme', theme);
            
            // Mettre à jour le body class pour la compatibilité
            if (theme === 'dark') {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
            
            // Mettre à jour l'indicateur de statut
            const themeStatus = document.getElementById('themeStatus');
            if (themeStatus) {
                themeStatus.textContent = theme === 'dark' ? 'Mode sombre' : 'Mode clair';
            }
            
            // Animation de transition douce
            document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
            
            // Notification subtile du changement (seulement si pas en mode silencieux)
            if (!silent && document.readyState === 'complete') {
                const themeText = theme === 'dark' ? 'Mode sombre activé 🌙' : 'Mode clair activé ☀️';
                showNotification(themeText, 'info');
            }
        }
        
        // Fonction pour obtenir le thème actuel
        function getCurrentTheme() {
            return document.documentElement.getAttribute('data-theme') || 'light';
        }

        // ===== ANIMATION DU PROFIL MÉDECIN =====
        
        function initializeDoctorProfile() {
            // Animation des statistiques au chargement
            animateProfileStats();
            
            // Mettre à jour les statistiques périodiquement
            setInterval(updateProfileStats, 30000); // Toutes les 30 secondes
            
            // Effet hover sur la carte profil
            const profileCard = document.querySelector('.sidebar .bg-white\\/90');
            if (profileCard) {
                profileCard.addEventListener('mouseenter', () => {
                    profileCard.style.transform = 'translateY(-2px) scale(1.02)';
                    profileCard.style.boxShadow = '0 20px 40px rgba(0,0,0,0.15)';
                });
                
                profileCard.addEventListener('mouseleave', () => {
                    profileCard.style.transform = 'translateY(0) scale(1)';
                    profileCard.style.boxShadow = '';
                });
            }
        }
        
        function animateProfileStats() {
            // Animation du compteur de consultations
            const consultationsElement = document.getElementById('dailyConsultations');
            if (consultationsElement) {
                let count = 0;
                const target = 12;
                const increment = target / 30;
                
                const countInterval = setInterval(() => {
                    count += increment;
                    if (count >= target) {
                        count = target;
                        clearInterval(countInterval);
                    }
                    consultationsElement.textContent = Math.floor(count);
                }, 50);
            }
        }
        
        function updateProfileStats() {
            // Simuler une mise à jour des statistiques
            const consultationsElement = document.getElementById('dailyConsultations');
            if (consultationsElement) {
                const currentCount = parseInt(consultationsElement.textContent);
                const newCount = currentCount + Math.floor(Math.random() * 3); // +0 à +2
                
                // Animation de changement
                consultationsElement.style.transform = 'scale(1.2)';
                consultationsElement.style.color = '#10b981';
                
                setTimeout(() => {
                    consultationsElement.textContent = newCount;
                    consultationsElement.style.transform = 'scale(1)';
                    consultationsElement.style.color = '';
                }, 150);
            }
        }

        // ===== FONCTIONS UTILITAIRES POUR LES ERREURS =====
        
        // Réessayer le dernier message
        window.retryLastMessage = function() {
            const messageInput = document.getElementById('messageInput');
            if (messageInput && messageInput.value.trim()) {
                const form = messageInput.closest('form');
                if (form) {
                    form.dispatchEvent(new Event('submit', { cancelable: true }));
                }
            } else {
                showEnhancedNotification('Aucun message à réessayer', 'warning');
            }
        }
        
        // Vérifier la connexion
        window.checkConnection = function() {
            fetch('/api/health', { method: 'HEAD' })
                .then(() => {
                    showEnhancedNotification('Connexion rétablie', 'success');
                })
                .catch(() => {
                    showEnhancedNotification('Problème de connexion persistant', 'error');
                });
        }
        
        // Programmer une nouvelle tentative
        window.scheduleRetry = function(delay) {
            showEnhancedNotification(`Nouvelle tentative dans ${delay/1000}s...`, 'info');
            setTimeout(() => {
                window.retryLastMessage();
            }, delay);
        }
        
        // Améliorer la gestion des erreurs réseau globales
        window.addEventListener('online', function() {
            updateConnectionStatus(true);
            showEnhancedNotification('Connexion internet rétablie', 'success');
        });
        
        window.addEventListener('offline', function() {
            updateConnectionStatus(false);
            showEnhancedNotification('Connexion internet perdue', 'warning');
        });
        
        // Fonction pour mettre à jour l'indicateur de statut de connexion
        function updateConnectionStatus(isOnline = navigator.onLine) {
            const statusElement = document.getElementById('connectionStatus');
            if (!statusElement) return;
            
            const statusDot = statusElement.querySelector('.w-2');
            const statusText = statusElement.querySelector('span');
            
            if (isOnline) {
                statusElement.className = 'flex items-center gap-1.5 px-2.5 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium';
                if (statusDot) statusDot.className = 'w-2 h-2 bg-green-500 rounded-full animate-pulse';
                if (statusText) statusText.textContent = 'En ligne';
            } else {
                statusElement.className = 'flex items-center gap-1.5 px-2.5 py-1 bg-red-100 text-red-700 rounded-full text-xs font-medium';
                if (statusDot) statusDot.className = 'w-2 h-2 bg-red-500 rounded-full';
                if (statusText) statusText.textContent = 'Hors ligne';
            }
        }
        
        // Vérifier périodiquement la connexion au serveur
        function checkServerConnection() {
            fetch('/api/health', { method: 'HEAD', timeout: 5000 })
                .then(response => {
                    if (response.ok) {
                        updateConnectionStatus(true);
                    } else {
                        updateConnectionStatus(false);
                    }
                })
                .catch(() => {
                    updateConnectionStatus(false);
                });
        }
        
        // Initialiser la vérification de connexion
        setInterval(checkServerConnection, 30000); // Toutes les 30 secondes
        checkServerConnection(); // Vérification initiale
        
        // Fonction utilitaire pour échapper le HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        window.formatDate = function(dateString) {
            const date = new Date(dateString);
            const now = new Date();
            const diffTime = Math.abs(now - date);
            const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

            if (diffDays === 1) return "Aujourd'hui";
            if (diffDays === 2) return "Hier";
            if (diffDays <= 7) return `Il y a ${diffDays - 1} jours`;
            
            return date.toLocaleDateString('fr-FR');
        }

        window.formatTime = function(dateString) {
            return new Date(dateString).toLocaleTimeString('fr-FR', {
                hour: '2-digit',
                minute: '2-digit'
            });
        }

        window.formatMessageContent = function(content) {
            // Conversion basique markdown vers HTML
            return content
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code class="bg-gray-200 px-1 rounded">$1</code>')
                .replace(/\n/g, '<br>');
        }

        // Gestion des patients
        window.openPatientModal = async function() {
            try {
                // Charger la liste des patients
                const response = await fetch('/api/patients/list');
                const data = await response.json();

                if (data.success) {
                    const select = document.getElementById('patientSelect');
                    select.innerHTML = '<option value="">Choisir un patient...</option>';
                    
                    data.patients.forEach(patient => {
                        const option = document.createElement('option');
                        option.value = patient.patient_id;
                        option.textContent = patient.display_name;
                        select.appendChild(option);
                    });
                    
                    // Afficher la modal avec animation
                    const modal = document.getElementById('assignPatientModal');
                    modal.style.display = 'flex';
                    
                    // Animation d'entrée
                    setTimeout(() => {
                        modal.classList.add('fade-in');
                        const modalContent = modal.querySelector('.bg-white');
                        if (modalContent) {
                            modalContent.style.transform = 'scale(1)';
                            modalContent.style.opacity = '1';
                        }
                    }, 10);
                    
                    // Replacer les icônes feather
                    feather.replace();
                    
                } else {
                    window.showNotification('Erreur lors du chargement des patients', 'error');
                }
            } catch (error) {
                console.error('Erreur chargement patients:', error);
                window.showNotification('Erreur réseau', 'error');
            }
        }

        window.closePatientModal = function() {
            const modal = document.getElementById('assignPatientModal');
            const modalContent = modal.querySelector('.bg-white');
            
            // Animation de sortie
            if (modalContent) {
                modalContent.style.transform = 'scale(0.95)';
                modalContent.style.opacity = '0';
            }
            
            setTimeout(() => {
                modal.style.display = 'none';
                modal.classList.remove('fade-in');
                
                // Réinitialiser le formulaire
                const select = document.getElementById('patientSelect');
                const patientInfo = document.getElementById('patientInfo');
                
                if (select) select.value = '';
                if (patientInfo) patientInfo.classList.add('hidden');
                
            }, 200);
        }

        window.assignPatient = async function() {
            if (!window.currentConversationId) {
                window.showNotification('Aucune conversation sélectionnée', 'error');
                return;
            }
            
            const patientId = document.getElementById('patientSelect').value;
            
            if (!patientId) {
                window.showNotification('Veuillez sélectionner un patient', 'error');
                return;
            }
            
            try {
                // Désactiver le bouton pendant la requête
                const confirmBtn = document.getElementById('assignPatientConfirm');
                const originalText = confirmBtn.innerHTML;
                
                confirmBtn.disabled = true;
                confirmBtn.innerHTML = '<i data-feather="loader" class="w-4 h-4 inline mr-2 animate-spin"></i>Attribution...';
                feather.replace();
                
                const response = await fetch(`/api/chat/conversations/${window.currentConversationId}/update`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        patient_id: patientId || null
                    })
                });

                const data = await response.json();
                if (data.success) {
                    // Recharger les conversations pour mettre à jour l'affichage
                    await window.loadConversations();
                    window.updateConversationHeader();
                    window.closePatientModal();
                    
                    // Obtenir le nom du patient sélectionné pour la notification
                    const select = document.getElementById('patientSelect');
                    const patientName = select.selectedOptions[0]?.textContent || 'Patient';
                    
                    window.showNotification(`${patientName} assigné avec succès`, 'success');
                } else {
                    window.showNotification('Erreur lors de l\'assignation: ' + data.error, 'error');
                }
                
                // Réactiver le bouton
                confirmBtn.disabled = false;
                confirmBtn.innerHTML = originalText;
                feather.replace();
                
            } catch (error) {
                console.error('Erreur assignation patient:', error);
                window.showNotification('Erreur réseau lors de l\'assignation', 'error');
                
                // Réactiver le bouton en cas d'erreur
                const confirmBtn = document.getElementById('assignPatientConfirm');
                confirmBtn.disabled = false;
                confirmBtn.innerHTML = '<i data-feather="check" class="w-4 h-4 inline mr-2"></i>Assigner';
                feather.replace();
            }
        }

        // Fonctions pour la modale de suppression de conversation
        window.openDeleteConversationModal = function(conversationId) {
            const modal = document.getElementById('deleteConversationModal');
            if (!modal) return;
            
            // Stocker l'ID de la conversation à supprimer
            modal.dataset.conversationId = conversationId;
            
            // Récupérer les détails de la conversation
            const conversation = window.conversations.find(conv => conv.id === conversationId);
            if (conversation) {
                document.getElementById('conversationTitle').textContent = conversation.title || 'Sans titre';
                document.getElementById('conversationDate').textContent = `Créée le ${window.formatDate(conversation.created_at)}`;
                document.getElementById('conversationMessages').textContent = `${conversation.message_count || 0} message(s)`;
            }
            
            modal.style.display = 'flex';
            requestAnimationFrame(() => {
                modal.classList.add('fade-in');
            });
        }

        window.closeDeleteConversationModal = function() {
            const modal = document.getElementById('deleteConversationModal');
            if (!modal) return;
            
            modal.classList.remove('fade-in');
            
            setTimeout(() => {
                modal.style.display = 'none';
                delete modal.dataset.conversationId;
            }, 200);
        }

        window.confirmDeleteConversation = async function() {
            const modal = document.getElementById('deleteConversationModal');
            const conversationId = modal?.dataset.conversationId;
            
            if (!conversationId) {
                window.showNotification('Erreur: conversation non trouvée', 'error');
                return;
            }
            
            const confirmBtn = document.getElementById('confirmDeleteConversation');
            const originalText = confirmBtn.innerHTML;
            
            try {
                // Désactiver le bouton et montrer le loading
                confirmBtn.disabled = true;
                confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin w-4 h-4 inline mr-2"></i>Suppression...';
                
                const response = await fetch(`/api/chat/conversations/${conversationId}/delete`, {
                    method: 'DELETE'
                });

                const data = await response.json();
                if (data.success) {
                    // Si c'est la conversation actuelle, revenir à l'état d'accueil
                    if (conversationId == window.currentConversationId) {
                        // Nettoyer le localStorage
                        localStorage.removeItem('neuroscan-current-conversation');
                        
                        // Fermer la sidebar mobile si elle est ouverte
                        const sidebar = document.getElementById('sidebar');
                        const overlay = document.getElementById('sidebarOverlay');
                        if (sidebar) sidebar.classList.add('sidebar-hidden');
                        if (overlay) overlay.classList.remove('show');
                        
                        // Remettre l'interface à l'état d'accueil
                        window.currentConversationId = null;
                        document.getElementById('chatInputArea').style.display = 'none';
                        document.getElementById('welcomeMessages').style.display = 'block';
                        document.getElementById('welcomeInput').style.display = 'block';
                        document.getElementById('conversationHeader').classList.add('hidden');
                        document.getElementById('assignPatientBtn').classList.add('hidden');
                        
                        // Vider les messages et réinitialiser le conteneur
                        window.renderMessages([]);
                        
                        // Remettre à zéro le titre de la conversation dans le header
                        const conversationTitle = document.getElementById('conversationTitle');
                        const conversationSubtitle = document.getElementById('conversationSubtitle');
                        if (conversationTitle) conversationTitle.textContent = 'Sélectionnez une conversation';
                        if (conversationSubtitle) conversationSubtitle.textContent = 'Commencez une nouvelle consultation';
                        
                        // Fermer la modale de suppression
                        window.closeDeleteConversationModal();
                        
                        // Recharger la liste des conversations
                        await window.loadConversations();
                        
                        // Notification de succès
                        window.showNotification('💫 Conversation supprimée - Retour à l\'accueil', 'success');
                        
                        return; // Sortir de la fonction après avoir géré le cas spécial
                    }
                    
                    // Fermer la modale
                    window.closeDeleteConversationModal();
                    
                    // Recharger la liste des conversations
                    await window.loadConversations();
                    
                    // Afficher un message de confirmation
                    window.showNotification('💫 Conversation supprimée avec succès', 'success');
                } else {
                    window.showNotification('Erreur lors de la suppression: ' + data.error, 'error');
                }
                
            } catch (error) {
                console.error('Erreur suppression conversation:', error);
                window.showNotification('Erreur réseau lors de la suppression', 'error');
            } finally {
                // Réactiver le bouton
                confirmBtn.disabled = false;
                confirmBtn.innerHTML = originalText;
            }
        }

        window.deleteConversation = async function(event, conversationId) {
            event.stopPropagation(); // Empêcher la sélection de la conversation
            
            // Ouvrir la modale de confirmation au lieu d'utiliser confirm()
            window.openDeleteConversationModal(conversationId);
        }

        function showNotification(message, type = 'info') {
            // Créer une notification temporaire
            const notification = document.createElement('div');
            notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg z-50 ${
                type === 'success' ? 'bg-green-500 text-white' : 
                type === 'error' ? 'bg-red-500 text-white' : 
                'bg-blue-500 text-white'
            }`;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            // Supprimer après 3 secondes
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 3000);
        }

        // ===== Fonctions pour l'édition et le branchement de messages =====

        function editMessage(messageId) {
            // Activer le mode édition pour un message
            const contentDiv = document.getElementById(`message-content-${messageId}`);
            const editArea = document.getElementById(`edit-area-${messageId}`);
            
            if (contentDiv && editArea) {
                contentDiv.style.display = 'none';
                editArea.classList.remove('hidden');
                
                // Focus sur le textarea
                const textarea = document.getElementById(`edit-input-${messageId}`);
                if (textarea) {
                    textarea.focus();
                    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
                }
            }
        }

        function cancelEdit(messageId) {
            // Annuler l'édition d'un message
            const contentDiv = document.getElementById(`message-content-${messageId}`);
            const editArea = document.getElementById(`edit-area-${messageId}`);
            
            if (contentDiv && editArea) {
                contentDiv.style.display = 'block';
                editArea.classList.add('hidden');
            }
        }

        async function saveEdit(messageId) {
            // Sauvegarder l'édition d'un message et créer une branche
            const textarea = document.getElementById(`edit-input-${messageId}`);
            const newContent = textarea.value.trim();
            
            if (!newContent) {
                showNotification('Le message ne peut pas être vide', 'error');
                return;
            }
            
            // Récupérer le contenu original pour comparer
            const originalContent = textarea.getAttribute('data-original') || textarea.defaultValue;
            
            if (newContent === originalContent.trim()) {
                showNotification('Aucune modification détectée', 'info');
                cancelEdit(messageId);
                return;
            }
            
            try {
                // Désactiver l'interface pendant la sauvegarde
                textarea.disabled = true;
                const saveBtn = document.querySelector(`#edit-area-${messageId} .bg-green-600`);
                if (saveBtn) saveBtn.textContent = 'Sauvegarde...';
                
                const response = await fetch(`/api/chat/messages/${messageId}/edit`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        content: newContent
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Masquer l'interface d'édition
                    cancelEdit(messageId);
                    
                    // Recharger les messages pour afficher la nouvelle branche
                    await window.loadMessages();
                    
                    // Si une nouvelle réponse a été générée, la montrer
                    if (data.assistant_response) {
                        window.showNotification('✅ Message édité et nouvelle réponse générée!', 'success');
                        
                        // Scroller vers le bas pour voir la nouvelle réponse
                        setTimeout(() => {
                            const container = document.getElementById('messagesContainer');
                            container.scrollTop = container.scrollHeight;
                        }, 100);
                    } else {
                        window.showNotification('✅ Message édité avec succès', 'success');
                    }
                } else {
                    showNotification('❌ Erreur lors de l\'édition: ' + (data.error || 'Erreur inconnue'), 'error');
                    textarea.disabled = false;
                    if (saveBtn) saveBtn.innerHTML = '<i class="fas fa-check mr-1"></i>Sauvegarder';
                }
                
            } catch (error) {
                console.error('Erreur édition:', error);
                showNotification('❌ Erreur réseau lors de l\'édition', 'error');
                textarea.disabled = false;
                const saveBtn = document.querySelector(`#edit-area-${messageId} .bg-green-600`);
                if (saveBtn) saveBtn.innerHTML = '<i class="fas fa-check mr-1"></i>Sauvegarder';
            }
        }

        async function regenerateResponse(messageId) {
            // Régénérer la réponse de l'assistant pour un message utilisateur
            if (!confirm('Voulez-vous régénérer une nouvelle réponse pour ce message ?')) {
                return;
            }
            
            try {
                // Afficher un indicateur de chargement
                showNotification('Génération d\'une nouvelle réponse...', 'info');
                
                const response = await fetch(`/api/chat/messages/${messageId}/regenerate`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Recharger les messages pour afficher la nouvelle réponse
                    await loadMessages();
                    showNotification('Nouvelle réponse générée avec succès', 'success');
                } else {
                    showNotification('Erreur lors de la régénération: ' + data.error, 'error');
                }
                
            } catch (error) {
                console.error('Erreur régénération:', error);
                showNotification('Erreur réseau lors de la régénération', 'error');
            }
        }

        function toggleBranches(messageId) {
            // Basculer l'affichage des branches d'un message
            const branchesContainer = document.getElementById(`branches-${messageId}`);
            
            if (branchesContainer.style.display === 'none' || !branchesContainer.innerHTML) {
                loadMessageBranches(messageId);
            } else {
                branchesContainer.style.display = 'none';
            }
        }

        async function loadMessageBranches(messageId) {
            // Charger et afficher les branches d'un message
            try {
                const response = await fetch(`/api/chat/messages/${messageId}/branches`);
                const data = await response.json();
                
                if (data.success && data.branches.length > 1) {
                    const branchesContainer = document.getElementById(`branches-${messageId}`);
                    
                    let branchesHTML = '<div class="mt-3 pt-3 border-t border-gray-200">';
                    branchesHTML += '<p class="text-xs text-gray-500 mb-2">Versions alternatives:</p>';
                    
                    data.branches.forEach((branch, index) => {
                        const isOriginal = branch.is_original;
                        const isCurrentLevel = branch.branch_level === 0;
                        
                        branchesHTML += `
                            <div class="mb-2 p-2 bg-gray-50 rounded-lg text-sm">
                                <div class="flex items-center justify-between mb-1">
                                    <span class="text-xs font-medium">
                                        ${isOriginal ? 'Original' : `Version ${index}`}
                                        ${isCurrentLevel ? '' : ` (Branche ${branch.branch_level})`}
                                    </span>
                                    <span class="text-xs text-gray-400">${formatTime(branch.timestamp)}</span>
                                </div>
                                <div class="text-gray-700">${formatMessageContent(branch.content)}</div>
                            </div>
                        `;
                    });
                    
                    branchesHTML += '</div>';
                    branchesContainer.innerHTML = branchesHTML;
                    branchesContainer.style.display = 'block';
                } else if (data.branches.length <= 1) {
                    showNotification('Aucune branche alternative trouvée', 'info');
                }
                
            } catch (error) {
                console.error('Erreur chargement branches:', error);
                showNotification('Erreur lors du chargement des branches', 'error');
            }
        }

        // Améliorer la fonction de chargement des messages pour supporter les branches
        async function loadMessagesWithBranches() {
            if (!currentConversationId) return;

            try {
                const response = await fetch(`/api/chat/conversations/${currentConversationId}/messages-with-branches`);
                const data = await response.json();

                if (data.success) {
                    renderMessagesWithBranches(data.messages);
                } else {
                    console.error('Erreur chargement messages avec branches:', data.error);
                    // Fallback vers l'ancienne méthode
                    await loadMessages();
                }
            } catch (error) {
                console.error('Erreur réseau:', error);
                // Fallback vers l'ancienne méthode
                await loadMessages();
            }
        }

        function renderMessagesWithBranches(messages) {
            const container = document.getElementById('messagesContainer');
            
            // Garder les messages d'accueil masqués si on a des messages
            if (messages.length > 0) {
                document.getElementById('welcomeMessages').style.display = 'none';
            }
            
            // Supprimer les anciens messages (garder seulement welcomeMessages)
            const oldMessages = container.querySelectorAll('.message-bubble');
            oldMessages.forEach(msg => msg.remove());

            // Ajouter les nouveaux messages avec leurs branches
            messages.forEach(message => {
                appendMessage(message, false);
                
                // Si le message a des branches, ajouter un indicateur
                if (message.branches && message.branches.length > 0) {
                    addBranchIndicator(message.id, message.branches.length);
                }
            });

            // Scroller vers le bas
            container.scrollTop = container.scrollHeight;
        }

        function addBranchIndicator(messageId, branchCount) {
            // Ajouter un indicateur de branches à un message
            const messageDiv = document.querySelector(`[data-message-id="${messageId}"]`);
            if (messageDiv) {
                const actionsDiv = messageDiv.querySelector('.message-actions');
                if (actionsDiv) {
                    const branchBtn = document.createElement('button');
                    branchBtn.className = 'text-xs px-2 py-1 bg-white bg-opacity-20 hover:bg-opacity-30 rounded text-current transition-colors ml-1';
                    branchBtn.title = `${branchCount} branche(s) alternative(s)`;
                    branchBtn.innerHTML = `<i class="fas fa-code-branch"></i> ${branchCount}`;
                    branchBtn.onclick = () => toggleBranches(messageId);
                    
                    actionsDiv.appendChild(branchBtn);
                }
            }
        }
    
    // Charger les conversations au démarrage
    window.loadConversations();
});
