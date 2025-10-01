
        let currentStep = 1;
        const totalSteps = 3;

        function updateStepIndicators() {
            const indicators = document.querySelectorAll('.step-indicator');
            const connectors = document.querySelectorAll('.connector-fill');
            const glows = document.querySelectorAll('.connector-glow');

            indicators.forEach((indicator, index) => {
                const stepNum = index + 1;
                indicator.classList.remove('active', 'completed', 'bg-gradient-to-br', 'from-green-500', 'to-green-600', 'from-blue-500', 'to-blue-600', 'bg-gray-200', 'text-white', 'text-gray-400', 'shadow-lg', 'shadow-xl', 'scale-110', 'scale-125', 'ring-4', 'ring-blue-300/50');

                if (stepNum < currentStep) {
                    // Étape complétée - Animation de succès
                    indicator.className += ' bg-gradient-to-br from-green-500 to-green-600 text-white shadow-lg scale-110';
                    indicator.style.animation = 'bounce 0.6s ease-out';
                    setTimeout(() => {
                        indicator.style.animation = '';
                    }, 600);
                } else if (stepNum === currentStep) {
                    // Étape actuelle - Animation de focus
                    indicator.className += ' bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-xl scale-125 ring-4 ring-blue-300/50';
                    indicator.style.animation = 'pulse 2s infinite';
                } else {
                    // Étape future
                    indicator.className += ' bg-gray-200 text-gray-400';
                    indicator.style.animation = '';
                }
            });

            // Animation des connecteurs avec effet de vague
            connectors.forEach((connector, index) => {
                if (index < currentStep - 1) {
                    connector.style.width = '100%';
                    connector.style.animation = 'slideIn 0.8s ease-out forwards';
                    setTimeout(() => {
                        connector.style.animation = 'glowPulse 2s infinite';
                    }, 800);
                } else {
                    connector.style.width = '0%';
                    connector.style.animation = '';
                }
            });

            // Effet de lueur sur les connecteurs actifs
            glows.forEach((glow, index) => {
                if (index < currentStep - 1) {
                    glow.style.opacity = '1';
                    glow.style.animation = 'glowPulse 2s infinite';
                } else {
                    glow.style.opacity = '0';
                    glow.style.animation = '';
                }
            });
        }

        function updateProgressBar() {
            const progress = ((currentStep - 1) / (totalSteps - 1)) * 100;
            const progressBar = document.querySelector('.progress-bar');

            // Animation fluide de la barre de progression
            progressBar.style.width = `${progress}%`;

            // Mettre à jour le tooltip de progression
            const progressHint = document.querySelector('.progress-hint');
            const stepNames = ['', 'Informations personnelles', 'Sécurité du compte', 'Informations professionnelles'];
            progressHint.textContent = `Étape ${currentStep} sur ${totalSteps} - ${stepNames[currentStep]}`;

            // Ajouter un effet de lueur temporaire lors de la progression
            if (progress > 0) {
                progressBar.style.boxShadow = '0 0 20px rgba(59, 130, 246, 0.5)';
                setTimeout(() => {
                    progressBar.style.boxShadow = '';
                }, 1000);
            }
        }

        function showStep(stepNum) {
            // Effacer toutes les erreurs de validation avant de changer d'étape
            clearAllFieldErrors();

            // Masquer toutes les étapes avec animation
            document.querySelectorAll('.step').forEach(step => {
                step.classList.remove('active');
                step.style.opacity = '0';
                setTimeout(() => {
                    step.style.display = 'none';
                }, 150);
            });

            // Afficher l'étape actuelle avec animation
            setTimeout(() => {
                const currentStepElement = document.getElementById(`step${stepNum}`);
                currentStepElement.style.display = 'block';
                currentStepElement.classList.add('active');

                setTimeout(() => {
                    currentStepElement.style.opacity = '1';
                }, 50);
            }, 200);

            // Mettre à jour les boutons
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            const submitBtn = document.getElementById('submitBtn');

            // Réinitialiser le bouton Suivant à son état par défaut
            nextBtn.innerHTML = 'Suivant<i class="fas fa-arrow-right ml-2"></i>';
            nextBtn.classList.remove('btn-success', 'btn-secondary');
            nextBtn.classList.add('btn-primary');
            nextBtn.disabled = false;

            prevBtn.style.display = stepNum > 1 ? 'block' : 'none';
            nextBtn.style.display = stepNum < totalSteps ? 'block' : 'none';
            submitBtn.style.display = stepNum === totalSteps ? 'block' : 'none';

            updateStepIndicators();
            updateProgressBar();

            // Si on affiche l'étape 2, vérifier la force du mot de passe actuel
            if (stepNum === 2) {
                const password = document.getElementById('password').value;
                if (password) {
                    checkPasswordStrength(password);
                }
                updateNextButtonState(); // Mettre à jour l'état du bouton
            }
        }

        function validateStep(stepNum) {
            let isValid = true;
            clearAllFieldErrors();

            if (stepNum === 1) {
                const firstName = document.getElementById('first_name').value.trim();
                const lastName = document.getElementById('last_name').value.trim();
                const email = document.getElementById('email').value.trim();

                if (!firstName) {
                    markFieldAsError('first_name', 'Le prénom est obligatoire');
                    isValid = false;
                } else {
                    markFieldAsValid('first_name');
                }

                if (!lastName) {
                    markFieldAsError('last_name', 'Le nom est obligatoire');
                    isValid = false;
                } else {
                    markFieldAsValid('last_name');
                }

                if (!email) {
                    markFieldAsError('email', 'L\'email est obligatoire');
                    isValid = false;
                } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
                    markFieldAsError('email', 'Veuillez entrer une adresse email valide');
                    isValid = false;
                } else {
                    markFieldAsValid('email');
                }

                if (!isValid) {
                    showError('Veuillez corriger les erreurs dans le formulaire avant de continuer.', 'error', 6000);
                }
            } else if (stepNum === 2) {
                const password = document.getElementById('password').value;
                const confirmPassword = document.getElementById('confirm_password').value;
                const criteria = checkPasswordStrength(password);

                let passwordValid = true;
                let errorMessages = [];

                // Vérifier que le mot de passe respecte TOUS les critères de sécurité
                if (!password) {
                    markFieldAsError('password', 'Le mot de passe est obligatoire');
                    errorMessages.push('Mot de passe manquant');
                    passwordValid = false;
                } else if (!criteria.length) {
                    markFieldAsError('password', 'Au moins 8 caractères requis');
                    errorMessages.push('Longueur insuffisante (8 caractères minimum)');
                    passwordValid = false;
                } else if (!criteria.uppercase || !criteria.lowercase) {
                    markFieldAsError('password', 'Doit contenir majuscules et minuscules');
                    errorMessages.push('Majuscules et minuscules requises');
                    passwordValid = false;
                } else if (!criteria.number) {
                    markFieldAsError('password', 'Au moins un chiffre requis');
                    errorMessages.push('Au moins un chiffre requis');
                    passwordValid = false;
                } else if (!criteria.symbol) {
                    markFieldAsError('password', 'Au moins un symbole spécial requis');
                    errorMessages.push('Au moins un symbole spécial requis (!@#$%^&*...)');
                    passwordValid = false;
                } else {
                    markFieldAsValid('password');
                }

                // Vérifier la confirmation du mot de passe
                if (!confirmPassword) {
                    markFieldAsError('confirm_password', 'La confirmation du mot de passe est obligatoire');
                    errorMessages.push('Confirmation du mot de passe manquante');
                    passwordValid = false;
                } else if (password !== confirmPassword) {
                    markFieldAsError('confirm_password', 'Les mots de passe ne correspondent pas');
                    errorMessages.push('Les mots de passe ne correspondent pas');
                    passwordValid = false;
                } else if (passwordValid) {
                    markFieldAsValid('confirm_password');
                }

                if (!passwordValid) {
                    // Afficher un message d'erreur détaillé
                    let errorMessage = 'Votre mot de passe ne respecte pas les critères de sécurité :\n';
                    errorMessages.forEach(msg => {
                        errorMessage += '• ' + msg + '\n';
                    });
                    errorMessage += '\nVeuillez corriger ces problèmes avant de continuer.';

                    showError(errorMessage, 'error', 8000);
                    isValid = false;
                }
            } else if (stepNum === 3) {
                // Validation de l'étape professionnelle
                const specialty = document.getElementById('specialty').value;
                const hospital = document.getElementById('hospital').value.trim();
                const licenseNumber = document.getElementById('license_number').value.trim();
                const phone = document.getElementById('phone').value.trim();

                if (!specialty) {
                    markFieldAsError('specialty', 'Veuillez sélectionner une spécialité');
                    isValid = false;
                } else {
                    markFieldAsValid('specialty');
                }

                if (!hospital) {
                    markFieldAsError('hospital', 'L\'établissement est obligatoire');
                    isValid = false;
                } else {
                    markFieldAsValid('hospital');
                }

                if (!licenseNumber) {
                    markFieldAsError('license_number', 'Le numéro RPPS/ADELI est obligatoire');
                    isValid = false;
                } else if (!/^\d{11}$/.test(licenseNumber)) {
                    markFieldAsError('license_number', 'Le numéro doit contenir 11 chiffres');
                    isValid = false;
                } else {
                    markFieldAsValid('license_number');
                }

                if (!phone) {
                    markFieldAsError('phone', 'Le numéro de téléphone est obligatoire');
                    isValid = false;
                } else if (!/^(\+33|0)[1-9](\d{2}){4}$/.test(phone.replace(/\s/g, ''))) {
                    markFieldAsError('phone', 'Format de téléphone invalide');
                    isValid = false;
                } else {
                    markFieldAsValid('phone');
                }

                if (!isValid) {
                    showError('Veuillez corriger les erreurs dans les informations professionnelles.', 'error', 6000);
                } else {
                    // Animation de succès finale
                    showFinalSuccessAnimation();
                }
            }

            return isValid;
        }

        function checkPasswordStrength(password) {
            const criteria = {
                length: password.length >= 8,
                uppercase: /[A-Z]/.test(password),
                lowercase: /[a-z]/.test(password),
                number: /\d/.test(password),
                symbol: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)
            };

            // Mettre à jour les indicateurs visuels
            updatePasswordCriteria(criteria);

            // Calculer le score de force
            const score = Object.values(criteria).filter(Boolean).length;

            // Mettre à jour la barre de force et le texte
            updatePasswordStrengthIndicator(score);

            return criteria;
        }

        function updatePasswordCriteria(criteria) {
            // Longueur
            const lengthIcon = document.querySelector('#length-criterion i');
            const lengthText = document.querySelector('#length-criterion span');
            if (criteria.length) {
                lengthIcon.className = 'fas fa-check-circle text-green-500 transition-all duration-300 transform scale-110';
                lengthText.className = 'text-green-700 font-medium transition-all duration-300';
            } else {
                lengthIcon.className = 'fas fa-times-circle text-red-400 transition-all duration-300';
                lengthText.className = 'text-gray-600 transition-all duration-300';
            }

            // Majuscules et minuscules
            const caseIcon = document.querySelector('#case-criterion i');
            const caseText = document.querySelector('#case-criterion span');
            if (criteria.uppercase && criteria.lowercase) {
                caseIcon.className = 'fas fa-check-circle text-green-500 transition-all duration-300 transform scale-110';
                caseText.className = 'text-green-700 font-medium transition-all duration-300';
            } else {
                caseIcon.className = 'fas fa-times-circle text-red-400 transition-all duration-300';
                caseText.className = 'text-gray-600 transition-all duration-300';
            }

            // Chiffres
            const numberIcon = document.querySelector('#number-criterion i');
            const numberText = document.querySelector('#number-criterion span');
            if (criteria.number) {
                numberIcon.className = 'fas fa-check-circle text-green-500 transition-all duration-300 transform scale-110';
                numberText.className = 'text-green-700 font-medium transition-all duration-300';
            } else {
                numberIcon.className = 'fas fa-times-circle text-red-400 transition-all duration-300';
                numberText.className = 'text-gray-600 transition-all duration-300';
            }

            // Symboles
            const symbolIcon = document.querySelector('#symbol-criterion i');
            const symbolText = document.querySelector('#symbol-criterion span');
            if (criteria.symbol) {
                symbolIcon.className = 'fas fa-check-circle text-green-500 transition-all duration-300 transform scale-110';
                symbolText.className = 'text-green-700 font-medium transition-all duration-300';
            } else {
                symbolIcon.className = 'fas fa-times-circle text-red-400 transition-all duration-300';
                symbolText.className = 'text-gray-600 transition-all duration-300';
            }
        }

        function updatePasswordStrengthIndicator(score) {
            const strengthBar = document.getElementById('strength-bar');
            const strengthText = document.getElementById('password-strength');

            let width, color, text, bgColor, textColor;

            switch (score) {
                case 0:
                case 1:
                    width = '20%';
                    color = '#ef4444'; // red
                    bgColor = 'bg-red-100';
                    textColor = 'text-red-700';
                    text = 'Très faible';
                    break;
                case 2:
                    width = '40%';
                    color = '#f97316'; // orange
                    bgColor = 'bg-orange-100';
                    textColor = 'text-orange-700';
                    text = 'Faible';
                    break;
                case 3:
                    width = '60%';
                    color = '#eab308'; // yellow
                    bgColor = 'bg-yellow-100';
                    textColor = 'text-yellow-700';
                    text = 'Moyen';
                    break;
                case 4:
                    width = '80%';
                    color = '#22c55e'; // green
                    bgColor = 'bg-green-100';
                    textColor = 'text-green-700';
                    text = 'Fort';
                    break;
                case 5:
                    width = '100%';
                    color = '#16a34a'; // dark green
                    bgColor = 'bg-green-100';
                    textColor = 'text-green-700';
                    text = 'Très fort';
                    break;
                default:
                    width = '0%';
                    color = '#ef4444';
                    bgColor = 'bg-gray-200';
                    textColor = 'text-gray-600';
                    text = 'Faible';
            }

            strengthBar.style.width = width;
            strengthBar.style.backgroundColor = color;
            strengthText.textContent = text;
            strengthText.className = `text-xs font-medium px-2 py-1 rounded-full ${bgColor} ${textColor} transition-all duration-300`;
        }

        function showError(message, type = 'error', duration = 5000) {
            // Créer une notification d'erreur améliorée
            const notification = document.createElement('div');
            notification.className = `error-notification ${type} rounded-xl p-4 shadow-xl text-white max-w-sm`;
            notification.innerHTML = `
                <div class="flex items-start space-x-3">
                    <div class="flex-shrink-0 mt-0.5">
                        <i class="fas ${getIconForType(type)} text-xl"></i>
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="font-medium text-sm leading-relaxed">${message}</p>
                    </div>
                    <button onclick="dismissNotification(this)" class="flex-shrink-0 ml-3 hover:bg-white/20 rounded-full p-1 transition-colors duration-200">
                        <i class="fas fa-times text-sm"></i>
                    </button>
                </div>
                <div class="mt-3 bg-white/20 rounded-full h-1">
                    <div class="bg-white/60 h-1 rounded-full transition-all duration-100 ease-linear" style="width: 100%; animation: countdown ${duration}ms linear forwards;"></div>
                </div>
            `;

            document.body.appendChild(notification);

            // Animation d'entrée
            setTimeout(() => {
                notification.style.transform = 'translateX(0) scale(1)';
            }, 10);

            // Auto-dismiss avec animation
            setTimeout(() => {
                dismissNotification(notification.querySelector('button'));
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
            notification.classList.add('fade-out');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 500);
        }

        // Fonction pour marquer un champ comme ayant une erreur
        function markFieldAsError(fieldId, errorMessage) {
            const field = document.getElementById(fieldId);
            if (!field) return;

            const formGroup = field.closest('.form-group');
            if (!formGroup) return;

            // Supprimer les erreurs précédentes
            formGroup.classList.remove('field-valid');
            formGroup.classList.add('field-error');

            // Animation de shake
            field.style.animation = 'shake 0.5s ease-in-out';
            setTimeout(() => {
                field.style.animation = '';
            }, 500);
        }

        // Fonction pour marquer un champ comme valide
        function markFieldAsValid(fieldId) {
            const field = document.getElementById(fieldId);
            if (!field) return;

            const formGroup = field.closest('.form-group');
            if (!formGroup) return;

            // Supprimer les erreurs
            formGroup.classList.remove('field-error');
            formGroup.classList.add('field-valid');

            // Supprimer l'icône d'erreur existante
            const existingErrorIcon = formGroup.querySelector('.error-icon');
            if (existingErrorIcon) {
                existingErrorIcon.remove();
            }

            // Supprimer le tooltip d'erreur
            const tooltip = formGroup.querySelector('.error-tooltip');
            if (tooltip) {
                tooltip.remove();
            }
        }

        // Fonction pour effacer toutes les erreurs de validation
        function clearAllFieldErrors() {
            document.querySelectorAll('.field-error').forEach(field => {
                field.classList.remove('field-error');
            });

            document.querySelectorAll('.field-valid').forEach(field => {
                field.classList.remove('field-valid');
            });
        }

        // Animation de succès finale
        function showFinalSuccessAnimation() {
            const form = document.getElementById('registerForm');
            const submitBtn = document.getElementById('submitBtn');

            // Ajouter la classe de succès au formulaire
            form.classList.add('form-success');

            // Créer et ajouter le badge de succès
            const successBadge = document.createElement('div');
            successBadge.className = 'success-badge';
            successBadge.innerHTML = '<i class="fas fa-check text-lg"></i>';
            form.appendChild(successBadge);

            // Animation du bouton de soumission
            submitBtn.style.animation = 'pulse 1s infinite';
            submitBtn.innerHTML = '<i class="fas fa-rocket mr-2"></i>Prêt à créer mon compte !';

            // Message de félicitations
            setTimeout(() => {
                showError('🎉 Toutes les informations sont validées ! Vous pouvez maintenant créer votre compte NeuroScan Pro.', 'success', 5000);
            }, 1000);

            // Supprimer l'animation après un délai
            setTimeout(() => {
                form.classList.remove('form-success');
                if (successBadge.parentNode) {
                    successBadge.parentNode.removeChild(successBadge);
                }
                submitBtn.style.animation = '';
            }, 5000);
        }

        // Fonction pour mettre à jour l'état du bouton Suivant selon la validation des mots de passe
        function updateNextButtonState() {
            if (currentStep !== 2) return; // Ne s'applique qu'à l'étape 2

            const nextBtn = document.getElementById('nextBtn');
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm_password').value;
            const criteria = checkPasswordStrength(password);

            // Vérifier si tous les critères sont respectés
            const passwordIsStrong = criteria.length && criteria.uppercase && criteria.lowercase && criteria.number && criteria.symbol;
            const passwordsMatch = password === confirmPassword && password.length > 0 && confirmPassword.length > 0;

            if (passwordIsStrong && passwordsMatch) {
                // Activer le bouton avec style de succès
                nextBtn.classList.remove('btn-secondary');
                nextBtn.classList.add('btn-success');
                nextBtn.innerHTML = '<i class="fas fa-shield-check mr-2"></i>Sécurité validée - Continuer';
                nextBtn.disabled = false;
            } else {
                // Désactiver le bouton avec style d'avertissement
                nextBtn.classList.remove('btn-success');
                nextBtn.classList.add('btn-secondary');
                nextBtn.innerHTML = '<i class="fas fa-lock mr-2"></i>Complétez la sécurité';
                nextBtn.disabled = true;
            }
        }

        document.addEventListener('DOMContentLoaded', function() {
            // Animation d'entrée améliorée
            const card = document.querySelector('.register-card');
            card.style.opacity = '0';
            card.style.transform = 'scale(0.95) translateY(20px)';

            setTimeout(() => {
                card.style.transition = 'all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)';
                card.style.opacity = '1';
                card.style.transform = 'scale(1) translateY(0)';
            }, 200);

            // Gestionnaire pour le bouton Suivant
            document.getElementById('nextBtn').addEventListener('click', function() {
                if (validateStep(currentStep)) {
                    // Vérification supplémentaire pour l'étape 2 : s'assurer que le mot de passe est vraiment sécurisé
                    if (currentStep === 2) {
                        const password = document.getElementById('password').value;
                        const confirmPassword = document.getElementById('confirm_password').value;
                        const criteria = checkPasswordStrength(password);

                        // Vérifier TOUS les critères de sécurité
                        const passwordIsStrong = criteria.length && criteria.uppercase && criteria.lowercase && criteria.number && criteria.symbol;
                        const passwordsMatch = password === confirmPassword && password.length > 0 && confirmPassword.length > 0;

                        if (!passwordIsStrong || !passwordsMatch) {
                            showError('❌ Sécurité insuffisante ! Votre mot de passe doit respecter tous les critères avant de continuer :\n• 8 caractères minimum\n• Majuscules et minuscules\n• Au moins un chiffre\n• Au moins un symbole spécial\n• Les mots de passe doivent correspondre', 'error', 10000);
                            return; // Ne pas continuer
                        }
                    }

                    // Afficher un message de succès pour l'étape validée
                    const stepNames = ['', 'Informations personnelles', 'Sécurité du compte', 'Informations professionnelles'];
                    showError(`✅ Étape "${stepNames[currentStep]}" validée avec succès !`, 'success', 3000);
                    currentStep++;
                    showStep(currentStep);
                }
            });

            // Gestionnaire pour le bouton Précédent
            document.getElementById('prevBtn').addEventListener('click', function() {
                currentStep--;
                showStep(currentStep);
            });

            // Validation en temps réel des champs individuels
            document.getElementById('first_name').addEventListener('blur', function() {
                const value = this.value.trim();
                if (value.length > 0) {
                    markFieldAsValid('first_name');
                } else {
                    markFieldAsError('first_name', 'Le prénom est obligatoire');
                }
            });

            document.getElementById('last_name').addEventListener('blur', function() {
                const value = this.value.trim();
                if (value.length > 0) {
                    markFieldAsValid('last_name');
                } else {
                    markFieldAsError('last_name', 'Le nom est obligatoire');
                }
            });

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

            // Validation en temps réel de la force du mot de passe
            document.getElementById('password').addEventListener('input', function() {
                const password = this.value;
                checkPasswordStrength(password);
                updateNextButtonState();

                // Validation stricte : tous les critères doivent être respectés
                if (password.length > 0) {
                    const criteria = checkPasswordStrength(password);
                    const allValid = criteria.length && criteria.uppercase && criteria.lowercase && criteria.number && criteria.symbol;
                    if (allValid) {
                        markFieldAsValid('password');
                    } else {
                        // Marquer comme erreur si les critères ne sont pas respectés
                        const missingCriteria = [];
                        if (!criteria.length) missingCriteria.push('8 caractères minimum');
                        if (!criteria.uppercase || !criteria.lowercase) missingCriteria.push('majuscules et minuscules');
                        if (!criteria.number) missingCriteria.push('au moins un chiffre');
                        if (!criteria.symbol) missingCriteria.push('au moins un symbole');

                        const errorMsg = 'Critères manquants : ' + missingCriteria.join(', ');
                        markFieldAsError('password', errorMsg);
                    }
                } else {
                    // Champ vide - enlever les indicateurs
                    this.closest('.form-group').classList.remove('field-valid', 'field-error');
                    const validIcon = this.closest('.form-group').querySelector('.valid-icon');
                    const errorIcon = this.closest('.form-group').querySelector('.error-icon');
                    const tooltip = this.closest('.form-group').querySelector('.error-tooltip');
                    if (validIcon) validIcon.remove();
                    if (errorIcon) errorIcon.remove();
                    if (tooltip) tooltip.remove();
                }
            });

            // Validation en temps réel des mots de passe
            document.getElementById('confirm_password').addEventListener('input', function() {
                const password = document.getElementById('password').value;
                const confirmPassword = this.value;
                const matchDiv = document.getElementById('password-match');

                // Vérifier d'abord si le mot de passe principal respecte les critères
                const criteria = checkPasswordStrength(password);
                const passwordIsStrong = criteria.length && criteria.uppercase && criteria.lowercase && criteria.number && criteria.symbol;

                if (confirmPassword.length === 0) {
                    matchDiv.classList.add('hidden');
                    this.closest('.form-group').classList.remove('field-valid', 'field-error');
                    const validIcon = this.closest('.form-group').querySelector('.valid-icon');
                    const errorIcon = this.closest('.form-group').querySelector('.error-icon');
                    const tooltip = this.closest('.form-group').querySelector('.error-tooltip');
                    if (validIcon) validIcon.remove();
                    if (errorIcon) errorIcon.remove();
                    if (tooltip) tooltip.remove();
                } else if (!passwordIsStrong) {
                    // Si le mot de passe principal n'est pas assez fort, ne pas valider la confirmation
                    markFieldAsError('confirm_password', 'Le mot de passe principal doit d\'abord respecter tous les critères de sécurité');
                    matchDiv.classList.remove('hidden');
                    matchDiv.classList.add('text-orange-600');
                    matchDiv.innerHTML = '<i class="fas fa-exclamation-triangle mr-1"></i>Améliorez d\'abord le mot de passe principal';
                } else if (password !== confirmPassword) {
                    markFieldAsError('confirm_password', 'Les mots de passe ne correspondent pas');
                    matchDiv.classList.remove('hidden');
                    matchDiv.classList.add('text-red-600');
                    matchDiv.innerHTML = '<i class="fas fa-exclamation-triangle mr-1"></i>Les mots de passe ne correspondent pas';
                } else {
                    markFieldAsValid('confirm_password');
                    matchDiv.classList.remove('hidden', 'text-red-600', 'text-orange-600');
                    matchDiv.classList.add('text-green-600');
                    matchDiv.innerHTML = '<i class="fas fa-check-circle mr-1"></i>Les mots de passe correspondent';
                }
                updateNextButtonState();
            });

            // Fonction pour mettre à jour l'état du bouton Suivant selon la validation des mots de passe
            function updateNextButtonState() {
                if (currentStep !== 2) return; // Ne s'applique qu'à l'étape 2

                const nextBtn = document.getElementById('nextBtn');
                const password = document.getElementById('password').value;
                const confirmPassword = document.getElementById('confirm_password').value;
                const criteria = checkPasswordStrength(password);

                // Vérifier si tous les critères sont respectés
                const passwordIsStrong = criteria.length && criteria.uppercase && criteria.lowercase && criteria.number && criteria.symbol;
                const passwordsMatch = password === confirmPassword && password.length > 0 && confirmPassword.length > 0;

                if (passwordIsStrong && passwordsMatch) {
                    // Activer le bouton avec style de succès
                    nextBtn.classList.remove('btn-secondary');
                    nextBtn.classList.add('btn-success');
                    nextBtn.innerHTML = '<i class="fas fa-shield-check mr-2"></i>Sécurité validée - Continuer';
                    nextBtn.disabled = false;
                } else {
                    // Désactiver le bouton avec style d'avertissement
                    nextBtn.classList.remove('btn-success');
                    nextBtn.classList.add('btn-secondary');
                    nextBtn.innerHTML = '<i class="fas fa-lock mr-2"></i>Complétez la sécurité';
                    nextBtn.disabled = true;
                }
            }

            // Gestion des icônes pour voir/masquer les mots de passe
            function togglePasswordVisibility(inputId, buttonId) {
                const input = document.getElementById(inputId);
                const button = document.getElementById(buttonId);
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

            // Gestionnaire pour le bouton d'œil du mot de passe principal
            document.getElementById('password-toggle').addEventListener('click', function() {
                togglePasswordVisibility('password', 'password-toggle');
            });

            // Gestionnaire pour le bouton d'œil de la confirmation
            document.getElementById('confirm-password-toggle').addEventListener('click', function() {
                togglePasswordVisibility('confirm_password', 'confirm-password-toggle');
            });

            // Focus automatique sur le premier champ
            setTimeout(() => {
                document.getElementById('first_name').focus();
            }, 800);

            // Gestion de l'accessibilité des tooltips
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    // Fermer tous les tooltips ouverts
                    document.querySelectorAll('.help-tooltip .help-content').forEach(tooltip => {
                        tooltip.style.opacity = '0';
                        tooltip.style.visibility = 'hidden';
                        tooltip.style.transform = 'translateX(-50%)';
                    });
                }
            });

            // Améliorer l'accessibilité des tooltips
            document.querySelectorAll('.help-tooltip .help-icon').forEach(icon => {
                icon.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        const tooltip = this.nextElementSibling;
                        const isVisible = tooltip.style.opacity === '1';

                        // Fermer tous les autres tooltips
                        document.querySelectorAll('.help-tooltip .help-content').forEach(t => {
                            t.style.opacity = '0';
                            t.style.visibility = 'hidden';
                            t.style.transform = 'translateX(-50%)';
                        });

                        // Ouvrir ou fermer le tooltip actuel
                        if (!isVisible) {
                            tooltip.style.opacity = '1';
                            tooltip.style.visibility = 'visible';
                            tooltip.style.transform = 'translateX(-50%) translateY(-5px)';
                        }
                    }
                });
            });
        });
    