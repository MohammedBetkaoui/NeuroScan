/**
 * NeuroScan Modern JavaScript - Ultra Professional Interactions
 * Advanced animations, interactions, and user experience enhancements
 */

class NeuroScanModern {
    constructor() {
        this.init();
    }

    init() {
        this.initializeAnimations();
        this.setupInteractiveElements();
        this.initializeParallax();
        this.setupNotificationSystem();
        this.initializePerformanceOptimizations();
    }

    // Initialize advanced animations
    initializeAnimations() {
        // Stagger animations for cards
        const cards = document.querySelectorAll('.neuroscan-card');
        cards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
        });

        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in');
                }
            });
        }, observerOptions);

        // Observe all animatable elements
        document.querySelectorAll('[data-animate]').forEach(el => {
            observer.observe(el);
        });
    }

    // Setup interactive elements
    setupInteractiveElements() {
        // Enhanced button interactions
        document.querySelectorAll('.btn-neuroscan').forEach(btn => {
            btn.addEventListener('mouseenter', this.createRippleEffect);
            btn.addEventListener('mouseleave', this.removeRippleEffect);
        });

        // Interactive icons
        document.querySelectorAll('.interactive-icon').forEach(icon => {
            icon.addEventListener('click', this.iconClickEffect);
        });

        // Magnetic effect for buttons
        this.setupMagneticEffect();
    }

    // Create ripple effect on button hover
    createRippleEffect(e) {
        const button = e.currentTarget;
        const ripple = document.createElement('span');
        const rect = button.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
        ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
        ripple.classList.add('ripple-effect');
        
        button.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    }

    // Remove ripple effect
    removeRippleEffect(e) {
        const ripples = e.currentTarget.querySelectorAll('.ripple-effect');
        ripples.forEach(ripple => ripple.remove());
    }

    // Icon click effect
    iconClickEffect(e) {
        const icon = e.currentTarget;
        icon.style.transform = 'scale(0.9)';
        setTimeout(() => {
            icon.style.transform = 'scale(1.1)';
            setTimeout(() => {
                icon.style.transform = 'scale(1)';
            }, 150);
        }, 100);
    }

    // Setup magnetic effect for interactive elements
    setupMagneticEffect() {
        document.querySelectorAll('[data-magnetic]').forEach(element => {
            element.addEventListener('mousemove', (e) => {
                const rect = element.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;
                
                element.style.transform = `translate(${x * 0.1}px, ${y * 0.1}px)`;
            });
            
            element.addEventListener('mouseleave', () => {
                element.style.transform = 'translate(0px, 0px)';
            });
        });
    }

    // Initialize parallax effects
    initializeParallax() {
        const parallaxElements = document.querySelectorAll('[data-parallax]');
        
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            
            parallaxElements.forEach(element => {
                const rate = scrolled * -0.5;
                element.style.transform = `translateY(${rate}px)`;
            });
        });
    }

    // Notification system
    setupNotificationSystem() {
        this.notificationContainer = document.createElement('div');
        this.notificationContainer.className = 'notification-container';
        document.body.appendChild(this.notificationContainer);
    }

    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icon = this.getNotificationIcon(type);
        notification.innerHTML = `
            <div class="flex items-center space-x-3">
                <i class="${icon} text-lg"></i>
                <span>${message}</span>
                <button class="ml-auto text-gray-400 hover:text-gray-600" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        this.notificationContainer.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto remove
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 400);
        }, duration);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'fas fa-check-circle text-green-500',
            error: 'fas fa-exclamation-circle text-red-500',
            warning: 'fas fa-exclamation-triangle text-yellow-500',
            info: 'fas fa-info-circle text-blue-500'
        };
        return icons[type] || icons.info;
    }

    // Performance optimizations
    initializePerformanceOptimizations() {
        // Lazy loading for images
        this.setupLazyLoading();
        
        // Debounced scroll handler
        this.setupDebouncedScroll();
        
        // Preload critical resources
        this.preloadCriticalResources();
    }

    setupLazyLoading() {
        const images = document.querySelectorAll('img[data-src]');
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('loading');
                    imageObserver.unobserve(img);
                }
            });
        });

        images.forEach(img => imageObserver.observe(img));
    }

    setupDebouncedScroll() {
        let ticking = false;
        
        const updateScrollEffects = () => {
            // Update scroll-based effects here
            this.updateScrollProgress();
            this.updateParallax();
            ticking = false;
        };

        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(updateScrollEffects);
                ticking = true;
            }
        });
    }

    updateScrollProgress() {
        const scrollProgress = document.querySelector('.scroll-indicator');
        if (scrollProgress) {
            const scrollTop = window.pageYOffset;
            const docHeight = document.body.offsetHeight - window.innerHeight;
            const scrollPercent = scrollTop / docHeight;
            scrollProgress.style.transform = `scaleX(${scrollPercent})`;
        }
    }

    updateParallax() {
        const scrolled = window.pageYOffset;
        const parallaxElements = document.querySelectorAll('[data-parallax]');
        
        parallaxElements.forEach(element => {
            const speed = element.dataset.parallax || 0.5;
            const yPos = -(scrolled * speed);
            element.style.transform = `translateY(${yPos}px)`;
        });
    }

    preloadCriticalResources() {
        const criticalImages = [
            '/static/images/hero-bg.jpg',
            '/static/images/logo.png'
        ];

        criticalImages.forEach(src => {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.as = 'image';
            link.href = src;
            document.head.appendChild(link);
        });
    }

    // Utility methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
    // Enhanced analysis simulation
    simulateAnalysis() {
        const progressCard = document.getElementById('progressCard');
        const progressBar = document.getElementById('analysisProgress');
        const progressPercent = document.getElementById('progressPercent');
        const currentStatus = document.getElementById('currentStatus');

        if (!progressCard || !progressBar) return;

        progressCard.classList.remove('hidden');

        const steps = [
            { progress: 25, status: 'Pré-traitement de l\'image...', duration: 3000 },
            { progress: 50, status: 'Segmentation des tissus...', duration: 8000 },
            { progress: 75, status: 'Classification par IA...', duration: 6000 },
            { progress: 100, status: 'Génération du rapport...', duration: 3000 }
        ];

        let currentStep = 0;

        const updateProgress = () => {
            if (currentStep >= steps.length) {
                this.showResults();
                return;
            }

            const step = steps[currentStep];

            // Update progress bar with animation
            progressBar.style.width = step.progress + '%';
            progressPercent.textContent = step.progress + '%';

            if (currentStatus) {
                currentStatus.textContent = step.status;
            }

            // Update step indicators
            this.updateStepIndicators(currentStep + 1);

            currentStep++;
            setTimeout(updateProgress, step.duration);
        };

        updateProgress();
    }

    updateStepIndicators(currentStep) {
        const steps = ['step1', 'step2', 'step3', 'step4'];

        steps.forEach((stepId, index) => {
            const stepElement = document.getElementById(stepId);
            if (!stepElement) return;

            const icon = stepElement.querySelector('i');
            const badge = stepElement.querySelector('span');
            const progressBar = stepElement.querySelector('.bg-gray-400, .bg-blue-500, .bg-emerald-500');

            if (index < currentStep) {
                // Completed step
                stepElement.className = stepElement.className.replace(/from-gray-\d+/g, 'from-emerald-50').replace(/to-slate-\d+/g, 'to-green-50');
                stepElement.className = stepElement.className.replace(/border-gray-\d+/g, 'border-emerald-200');

                if (icon) {
                    icon.className = 'fas fa-check text-white text-xl';
                }
                if (badge) {
                    badge.className = 'bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full text-sm font-semibold';
                    badge.textContent = 'Terminé';
                }
                if (progressBar) {
                    progressBar.className = progressBar.className.replace(/bg-gray-\d+/g, 'bg-emerald-500');
                    progressBar.style.width = '100%';
                }
            } else if (index === currentStep) {
                // Current step
                stepElement.className = stepElement.className.replace(/from-gray-\d+/g, 'from-blue-50').replace(/to-slate-\d+/g, 'to-cyan-50');
                stepElement.className = stepElement.className.replace(/border-gray-\d+/g, 'border-blue-200');

                if (icon) {
                    icon.className = 'fas fa-spinner fa-spin text-white text-xl';
                }
                if (badge) {
                    badge.className = 'bg-blue-100 text-blue-700 px-3 py-1 rounded-full text-sm font-semibold';
                    badge.textContent = 'En cours';
                }
            }
        });
    }

    showResults() {
        const progressCard = document.getElementById('progressCard');
        const resultsCard = document.getElementById('resultsCard');

        if (progressCard) {
            progressCard.style.opacity = '0';
            setTimeout(() => {
                progressCard.classList.add('hidden');
            }, 500);
        }

        if (resultsCard) {
            setTimeout(() => {
                resultsCard.classList.remove('hidden');
                resultsCard.style.opacity = '0';
                setTimeout(() => {
                    resultsCard.style.opacity = '1';
                }, 100);

                // Animate result values
                this.animateResultValues();
            }, 600);
        }
    }

    animateResultValues() {
        // Animate probability percentages
        this.animateCounter('tumorProbability', 89, '%');
        this.animateCounter('meningiomaProb', 12, '%');
        this.animateCounter('gliomaProb', 34, '%');
        this.animateCounter('metastasisProb', 43, '%');
        this.animateCounter('normalProb', 11, '%');

        // Update completion time
        const now = new Date();
        const timeString = now.toLocaleTimeString('fr-FR', {
            hour: '2-digit',
            minute: '2-digit'
        });

        const completionTimeElement = document.getElementById('completionTime');
        if (completionTimeElement) {
            completionTimeElement.textContent = `Terminé à ${timeString}`;
        }

        // Set random analysis time
        const analysisTimeElement = document.getElementById('analysisTime');
        if (analysisTimeElement) {
            const randomTime = (15 + Math.random() * 10).toFixed(1);
            analysisTimeElement.textContent = `${randomTime}s`;
        }
    }

    animateCounter(elementId, targetValue, suffix = '') {
        const element = document.getElementById(elementId);
        if (!element) return;

        let currentValue = 0;
        const increment = targetValue / 50;
        const duration = 2000;
        const stepTime = duration / 50;

        const timer = setInterval(() => {
            currentValue += increment;
            if (currentValue >= targetValue) {
                currentValue = targetValue;
                clearInterval(timer);
            }

            element.textContent = Math.round(currentValue) + suffix;
        }, stepTime);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.neuroScanModern = new NeuroScanModern();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NeuroScanModern;
}
