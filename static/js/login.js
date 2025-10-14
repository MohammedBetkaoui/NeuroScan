
        let currentStep = 1;
        const totalSteps = 1;

        // Animation d'entrée simple
        function initializeAnimations() {
            const card = document.querySelector('.login-card');
            const formFields = document.querySelectorAll('.form-field');

            // Animation de la carte principale
            card.style.animation = 'cardEntrance 0.8s ease-out forwards';

            // Animation échelonnée des champs
            formFields.forEach((field, index) => {
                field.style.opacity = '0';
                field.style.transform = 'translateY(10px)';
                field.style.animation = `staggerEntrance 0.5s ease-out ${0.2 + index * 0.1}s forwards`;
            });

            // Focus automatique avec délai
            setTimeout(() => {
                document.getElementById('email').focus();
            }, 1000);
        }

        // Gestionnaire d'erreurs simple
        function showError(message, type = 'error', duration = 5000) {
            const notification = document.createElement('div');
            notification.className = `error-notification fixed top-4 right-4 z-50 min-w-80 max-w-96 rounded-xl p-4 shadow-lg text-white
                ${type === 'success' ? 'bg-green-500' :
                  type === 'warning' ? 'bg-yellow-500' :
                  type === 'info' ? 'bg-blue-500' : 'bg-red-500'}`;

            notification.innerHTML = `
                <div class="flex items-start space-x-3">
                    <div class="flex-shrink-0 mt-0.5">
                        <i class="fas ${getIconForType(type)} text-xl"></i>
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="font-medium text-sm leading-relaxed">${ message }</p>
                    </div>
                    <button onclick="dismissNotification(this)" class="flex-shrink-0 ml-3 hover:bg-white/20 rounded-full p-1 transition-colors duration-200">
                        <i class="fas fa-times text-sm"></i>
                    </button>
                </div>
            `;

            document.body.appendChild(notification);

            // Auto-dismiss simple
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, duration);
        }

        function getIconForType(type) {
            switch (type) {
                case 'success': return 'fa-check-circle';
                case 'warning': return 'fa-exclamation-triangle';
                case 'info': return 'fa-info-circle';
                case 'error':
                default: return 'fa-exclamation-circle';
            }
        }

        function dismissNotification(button) {
            const notification = button.closest('.error-notification');
            notification.style.opacity = '0';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }

        // Gestion des états de validation
        function markFieldAsError(fieldId, errorMessage) {
            const field = document.getElementById(fieldId);
            if (!field) return;

            const formField = field.closest('.form-field');
            if (!formField) return;

            formField.classList.remove('field-valid');
            formField.classList.add('field-error');

            // Animation de shake simple
            field.style.animation = 'errorShake 0.4s ease-in-out';
            setTimeout(() => {
                field.style.animation = '';
            }, 400);
        }

        function markFieldAsValid(fieldId) {
            const field = document.getElementById(fieldId);
            if (!field) return;

            const formField = field.closest('.form-field');
            if (!formField) return;

            formField.classList.remove('field-error');
            formField.classList.add('field-valid');
        }

        function clearAllFieldErrors() {
            document.querySelectorAll('.field-error').forEach(field => {
                field.classList.remove('field-error');
            });

            document.querySelectorAll('.field-valid').forEach(field => {
                field.classList.remove('field-valid');
            });
        }

        // Gestion du formulaire avec indicateur de chargement
        function handleFormSubmission(e) {
            const form = e.target;
            const submitBtn = form.querySelector('button[type="submit"]');

            // Validation
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value;

            clearAllFieldErrors();
            let isValid = true;

            if (!email) {
                markFieldAsError('email', 'L\'email est obligatoire');
                isValid = false;
            } else if (!/^[^\^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
                markFieldAsError('email', 'Format d\'email invalide');
                isValid = false;
            } else {
                markFieldAsValid('email');
            }

            if (!password) {
                markFieldAsError('password', 'Le mot de passe est obligatoire');
                isValid = false;
            } else {
                markFieldAsValid('password');
            }

            if (!isValid) {
                e.preventDefault();
                showError('Veuillez corriger les erreurs dans le formulaire avant de continuer.', 'error', 6000);
                return;
            }

            // Afficher l'indicateur de chargement
            submitBtn.classList.add('loading');
            submitBtn.innerHTML = '<i class="fas fa-spinner mr-2"></i>Connexion en cours...';
            submitBtn.disabled = true;

            // Le formulaire sera soumis normalement
        }

        // Gestionnaire pour le bouton d'œil du mot de passe
        function togglePasswordVisibility() {
            const input = document.getElementById('password');
            const button = document.getElementById('password-toggle');
            const icon = button.querySelector('i');

            if (input.type === 'password') {
                input.type = 'text';
                icon.className = 'fas fa-eye-slash';
                button.setAttribute('aria-label', 'Masquer le mot de passe');
            } else {
                input.type = 'password';
                icon.className = 'fas fa-eye';
                button.setAttribute('aria-label', 'Afficher le mot de passe');
            }
        }

        // Validation en temps réel
        function setupRealTimeValidation() {
            document.getElementById('email').addEventListener('blur', function() {
                const value = this.value.trim();
                if (value.length === 0) {
                    markFieldAsError('email', 'L\'email est obligatoire');
                } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
                    markFieldAsError('email', 'Format d\'email invalide');
                } else {
                    markFieldAsValid('email');
                }
            });

            document.getElementById('password').addEventListener('blur', function() {
                const value = this.value;
                if (value.length === 0) {
                    markFieldAsError('password', 'Le mot de passe est obligatoire');
                } else {
                    markFieldAsValid('password');
                }
            });
        }

        // Gestion de l'accessibilité
        function setupAccessibility() {
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    document.querySelectorAll('.help-tooltip .help-content').forEach(tooltip => {
                        tooltip.style.opacity = '0';
                        tooltip.style.visibility = 'hidden';
                        tooltip.style.transform = 'translateX(-50%)';
                    });
                }
            });

            document.querySelectorAll('.help-tooltip .help-icon').forEach(icon => {
                icon.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        const tooltip = this.nextElementSibling;
                        const isVisible = tooltip.style.opacity === '1';

                        document.querySelectorAll('.help-tooltip .help-content').forEach(t => {
                            t.style.opacity = '0';
                            t.style.visibility = 'hidden';
                            t.style.transform = 'translateX(-50%)';
                        });

                        if (!isVisible) {
                            tooltip.style.opacity = '1';
                            tooltip.style.visibility = 'visible';
                            tooltip.style.transform = 'translateX(-50%) translateY(-5px)';
                        }
                    }
                });
            });
        }

        // Initialisation
        document.addEventListener('DOMContentLoaded', function() {
            initializeAnimations();
            setupRealTimeValidation();
            setupAccessibility();

            // Gestionnaires d'événements
            document.getElementById('password-toggle').addEventListener('click', togglePasswordVisibility);
            document.getElementById('loginForm').addEventListener('submit', handleFormSubmission);
        });
    