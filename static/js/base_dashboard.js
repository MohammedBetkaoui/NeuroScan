        // Variables globales
        let alertsData = [];
        let lastAlertCount = 0; // Pour détecter les nouvelles alertes
        let currentDoctor = {};
        try {
            const el = document.getElementById('doctor-data');
            if (el) currentDoctor = JSON.parse(el.textContent || '{}') || {};
        } catch (e) { currentDoctor = {}; }

        // Initialisation
        document.addEventListener('DOMContentLoaded', function() {
            initializePage();
            loadAlerts();
            setupEventListeners();
            
            // Actualiser les alertes toutes les 30 secondes
            setInterval(loadAlerts, 30000);
        });

        // Initialisation de la page
        function initializePage() {
            // Animation d'entrée des éléments
            animateElements();
            
            // Configuration des tooltips et autres éléments interactifs
            setupInteractiveElements();
        }

        // Animations d'entrée
        function animateElements() {
            const cards = document.querySelectorAll('.dashboard-card');
            cards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';

                setTimeout(() => {
                    card.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 100);
            });
        }

        // Configuration des event listeners
        function setupEventListeners() {
            // Menu utilisateur
            setupUserMenu();
            
            // Alertes dropdown
            setupAlertsDropdown();
            
            // Fermeture des modals avec Escape
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    closeAllModals();
                }
            });
        }

        // Menu utilisateur
        function setupUserMenu() {
            const button = document.getElementById('userMenuButton');
            const dropdown = document.getElementById('userMenuDropdown');

            if (button && dropdown) {
                button.addEventListener('click', function(e) {
                    e.stopPropagation();
                    dropdown.classList.toggle('hidden');
                });

                document.addEventListener('click', function() {
                    dropdown.classList.add('hidden');
                });

                dropdown.addEventListener('click', function(e) {
                    e.stopPropagation();
                });
            }
        }

        // Système d'alertes
        function setupAlertsDropdown() {
            const button = document.getElementById('alertsButton');
            const dropdown = document.getElementById('alertsDropdown');

            if (button && dropdown) {
                button.addEventListener('click', function(e) {
                    e.stopPropagation();
                    dropdown.classList.toggle('hidden');
                });

                document.addEventListener('click', function() {
                    dropdown.classList.add('hidden');
                });

                dropdown.addEventListener('click', function(e) {
                    e.stopPropagation();
                });
            }
        }

        // Chargement des alertes
        async function loadAlerts() {
            if (!currentDoctor.id) return;
            
            try {
                const response = await fetch('/api/alerts');
                const data = await response.json();

                if (data.success) {
                    const newAlertCount = data.data.length;
                    
                    // Détecter nouvelles alertes et jouer le son
                    if (lastAlertCount > 0 && newAlertCount > lastAlertCount) {
                        const newAlertsNumber = newAlertCount - lastAlertCount;
                        console.log(`🔔 ${newAlertsNumber} nouvelle(s) alerte(s) détectée(s)`);
                        
                        // Jouer le son de notification
                        if (notificationAudio) {
                            try {
                                const audio = notificationAudio.cloneNode();
                                audio.volume = 0.5;
                                audio.play().catch(error => {
                                    console.log('Impossible de jouer le son:', error);
                                });
                            } catch (error) {
                                console.log('Erreur lecture son:', error);
                            }
                        }
                        
                        // Afficher notification visuelle
                        showNotification(
                            `🔔 ${newAlertsNumber} nouvelle${newAlertsNumber > 1 ? 's' : ''} alerte${newAlertsNumber > 1 ? 's' : ''} médicale${newAlertsNumber > 1 ? 's' : ''}`,
                            'info',
                            5000,
                            false // Ne pas jouer le son deux fois
                        );
                    }
                    
                    lastAlertCount = newAlertCount;
                    alertsData = data.data;
                    updateAlertsUI();
                }
            } catch (error) {
                console.error('Erreur lors du chargement des alertes:', error);
            }
        }

        // Mise à jour de l'interface des alertes
        function updateAlertsUI() {
            const badge = document.getElementById('alertsBadge');
            const alertsList = document.getElementById('alertsList');

            if (!badge || !alertsList) return;

            // Mettre à jour le badge
            const unreadCount = alertsData.filter(alert => !alert.is_read).length;
            if (unreadCount > 0) {
                badge.textContent = unreadCount;
                badge.classList.remove('hidden');
            } else {
                badge.classList.add('hidden');
            }

            // Contenu des alertes
            let alertsContent = '';
            if (alertsData.length === 0) {
                alertsContent = `
                    <div class="p-4 text-center text-gray-500">
                        <i class="fas fa-check-circle text-green-500 text-2xl mb-2"></i>
                        <p class="font-medium">Aucune alerte active</p>
                        <p class="text-sm">Tous vos patients sont sous surveillance normale</p>
                    </div>
                `;
            } else {
                alertsContent = alertsData.slice(0, 5).map(alert => `
                    <div class="p-3 border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors duration-200 ${!alert.is_read ? 'bg-neuroscan-50 border-l-4 border-l-neuroscan-400' : ''}"
                         onclick="handleAlertClick(${alert.id})">
                        <div class="flex items-start space-x-3">
                            <div class="flex-shrink-0 mt-1">
                                <i class="fas fa-${getSeverityIcon(alert.severity)} text-${getSeverityColor(alert.severity)}"></i>
                            </div>
                            <div class="flex-1 min-w-0">
                                <p class="text-sm font-medium text-gray-900 truncate">${alert.title}</p>
                                <p class="text-xs text-gray-600 mt-1">${alert.patient_name} - ID: ${alert.patient_id}</p>
                                <p class="text-xs text-gray-500 mt-1 flex items-center">
                                    <i class="fas fa-clock mr-1"></i>
                                    ${formatDate(alert.created_at)}
                                </p>
                            </div>
                            ${!alert.is_read ? '<div class="w-2 h-2 bg-neuroscan-500 rounded-full mt-2"></div>' : ''}
                        </div>
                    </div>
                `).join('');
            }

            // Mettre à jour la liste
            alertsList.innerHTML = alertsContent;
        }

        // Gestion des clics sur les alertes
        async function handleAlertClick(alertId) {
            try {
                await fetch(`/api/alerts/${alertId}/mark-read`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                const alert = alertsData.find(a => a.id === alertId);
                if (alert) {
                    window.location.href = `/patient/${alert.patient_id}`;
                }
            } catch (error) {
                console.error('Erreur lors du traitement de l\'alerte:', error);
            }
        }

        // Utilitaires
        function getSeverityIcon(severity) {
            switch (severity) {
                case 'high': return 'exclamation-triangle';
                case 'medium': return 'exclamation-circle';
                case 'low': return 'info-circle';
                default: return 'bell';
            }
        }

        function getSeverityColor(severity) {
            switch (severity) {
                case 'high': return 'red-500';
                case 'medium': return 'amber-500';
                case 'low': return 'blue-500';
                default: return 'gray-500';
            }
        }

        function formatDate(dateString) {
            if (!dateString) return '-';
            const date = new Date(dateString);
            if (isNaN(date.getTime())) return '-';
            const now = new Date();
            const diffMs = now.getTime() - date.getTime();
            const diffMins = Math.floor(diffMs / 60000);
            const diffHours = Math.floor(diffMins / 60);
            const diffDays = Math.floor(diffHours / 24);

            if (diffMins < 1) return 'À l\'instant';
            if (diffMins < 60) return `Il y a ${diffMins} min`;
            if (diffHours < 24) return `Il y a ${diffHours}h`;
            if (diffDays < 7) return `Il y a ${diffDays}j`;
            try { return date.toLocaleDateString('fr-FR'); } catch { return '-'; }
        }

        // Gestion des modales
        function openProfileModal() {
            document.getElementById('profileModal').classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }

        function closeProfileModal() {
            document.getElementById('profileModal').classList.add('hidden');
            document.body.style.overflow = 'auto';
        }

        function openSettingsModal() {
            document.getElementById('settingsModal').classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }

        function closeSettingsModal() {
            document.getElementById('settingsModal').classList.add('hidden');
            document.body.style.overflow = 'auto';
        }

        function closeAllModals() {
            const modals = document.querySelectorAll('.modal-overlay');
            modals.forEach(modal => {
                modal.classList.add('hidden');
            });
            document.body.style.overflow = 'auto';
        }

        // Éléments interactifs
        function setupInteractiveElements() {
            // Fermeture des modals en cliquant sur l'arrière-plan
            document.querySelectorAll('.modal-overlay').forEach(modal => {
                modal.addEventListener('click', function(e) {
                    if (e.target === this) {
                        closeAllModals();
                    }
                });
            });
        }

        // Notifications système
        // Audio de notification préchargé
        let notificationAudio = null;
        try {
            notificationAudio = new Audio('/static/shop-notification-355746.mp3');
            notificationAudio.volume = 0.5;
            notificationAudio.preload = 'auto';
        } catch (error) {
            console.log('Audio de notification non disponible');
        }

        function showNotification(message, type = 'info', duration = 5000, playSound = true) {
            const notification = document.createElement('div');
            notification.className = `notification ${type} show animate-slide-in-right`;
            notification.innerHTML = `
                <div class="flex items-center">
                    <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'} mr-3"></i>
                    <span>${message}</span>
                    <button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-current hover:opacity-70">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            
            document.body.appendChild(notification);
            
            // Jouer le son de notification
            if (playSound && notificationAudio) {
                try {
                    const audio = notificationAudio.cloneNode();
                    audio.volume = 0.5;
                    audio.play().catch(error => {
                        console.log('Impossible de jouer le son:', error);
                    });
                } catch (error) {
                    console.log('Erreur lecture son:', error);
                }
            }
            
            setTimeout(() => {
                notification.remove();
            }, duration);
        }

        // Gestionnaires pour les paramètres
        function showPasswordChangeForm() {
            showNotification('Fonctionnalité en développement - Contactez l\'administrateur', 'info');
        }

        function showSessionManagement() {
            showNotification('Fonctionnalité en développement - Contactez l\'administrateur', 'info');
        }
    