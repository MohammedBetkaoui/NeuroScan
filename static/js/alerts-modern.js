/**
 * NeuroScan - Système d'Alertes Moderne
 * Gestion avancée des notifications et alertes médicales
 */

class AlertsManager {
    constructor() {
        this.allAlerts = [];
        this.filteredAlerts = [];
        this.currentPage = 1;
        this.alertsPerPage = 10;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadAlerts();
        this.initializeAnimations();
    }

    setupEventListeners() {
        // Filtres en temps réel
        document.getElementById('severityFilter')?.addEventListener('change', () => this.applyFilters());
        document.getElementById('typeFilter')?.addEventListener('change', () => this.applyFilters());
        document.getElementById('statusFilter')?.addEventListener('change', () => this.applyFilters());
        document.getElementById('searchInput')?.addEventListener('input', this.debounce(() => this.applyFilters(), 300));

        // Raccourcis clavier
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'f':
                        e.preventDefault();
                        document.getElementById('searchInput')?.focus();
                        break;
                    case 'r':
                        e.preventDefault();
                        this.refreshAlerts();
                        break;
                }
            }
        });
    }

    async loadAlerts() {
        try {
            const response = await fetch('/api/alerts');
            if (!response.ok) throw new Error('Erreur lors du chargement des alertes');
            
            this.allAlerts = await response.json();
            this.filteredAlerts = [...this.allAlerts];
            this.displayAlerts();
            this.updateStats();
            this.animateCards();
        } catch (error) {
            console.error('Erreur:', error);
            this.showToast('Erreur lors du chargement des alertes', 'error');
        }
    }

    displayAlerts() {
        const alertsList = document.getElementById('alertsList');
        const startIndex = (this.currentPage - 1) * this.alertsPerPage;
        const endIndex = startIndex + this.alertsPerPage;
        const alertsToShow = this.filteredAlerts.slice(startIndex, endIndex);

        if (alertsToShow.length === 0) {
            alertsList.innerHTML = this.getEmptyStateHTML();
            return;
        }

        alertsList.innerHTML = alertsToShow.map((alert, index) => 
            this.createAlertCardHTML(alert, index)
        ).join('');

        this.updatePagination();
        this.animateCards();
    }

    createAlertCardHTML(alert, index) {
        const severityConfig = this.getSeverityConfig(alert.severity);
        const isUnread = !alert.is_read;
        
        return `
            <div class="alert-card alert-${alert.severity} ${isUnread ? 'alert-unread' : ''}" 
                 onclick="alertsManager.showAlertDetails(${alert.id})"
                 style="animation-delay: ${index * 0.1}s">
                <div class="d-flex align-items-start">
                    <div class="flex-shrink-0 me-3">
                        <i class="fas fa-${severityConfig.icon} alert-icon text-${severityConfig.color}"></i>
                    </div>
                    <div class="flex-grow-1">
                        <div class="d-flex justify-content-between align-items-start mb-3">
                            <h6 class="mb-0 fw-bold">${alert.title}</h6>
                            <div class="d-flex align-items-center gap-2">
                                <span class="severity-badge badge-${severityConfig.color}">
                                    ${severityConfig.text}
                                </span>
                                ${isUnread ? '<div class="bg-primary rounded-circle pulse-dot" style="width: 10px; height: 10px;"></div>' : ''}
                            </div>
                        </div>
                        <p class="text-muted mb-3" style="line-height: 1.6;">${alert.message}</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <div class="d-flex align-items-center text-muted">
                                <i class="fas fa-user-circle me-2"></i>
                                <span class="fw-medium">${alert.patient_name}</span>
                                <span class="mx-2">•</span>
                                <span class="text-primary">#${alert.patient_id}</span>
                            </div>
                            <div class="d-flex align-items-center text-muted">
                                <i class="fas fa-clock me-1"></i>
                                <span>${this.formatDate(alert.created_at)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getEmptyStateHTML() {
        return `
            <div class="text-center py-5">
                <div class="mb-4">
                    <i class="fas fa-check-circle text-success" style="font-size: 4rem; opacity: 0.7;"></i>
                </div>
                <h5 class="fw-bold text-muted mb-2">Aucune alerte trouvée</h5>
                <p class="text-muted">Toutes vos alertes ont été traitées ou aucune alerte ne correspond aux filtres appliqués.</p>
                <button class="btn btn-outline-primary mt-3" onclick="alertsManager.clearFilters()">
                    <i class="fas fa-times me-1"></i>Effacer les filtres
                </button>
            </div>
        `;
    }

    getSeverityConfig(severity) {
        const configs = {
            high: { icon: 'exclamation-triangle', color: 'danger', text: 'Critique' },
            medium: { icon: 'exclamation-circle', color: 'warning', text: 'Moyenne' },
            low: { icon: 'info-circle', color: 'info', text: 'Faible' }
        };
        return configs[severity] || configs.low;
    }

    applyFilters() {
        const severityFilter = document.getElementById('severityFilter')?.value || '';
        const typeFilter = document.getElementById('typeFilter')?.value || '';
        const statusFilter = document.getElementById('statusFilter')?.value || '';
        const searchInput = document.getElementById('searchInput')?.value.toLowerCase() || '';

        this.filteredAlerts = this.allAlerts.filter(alert => {
            const matchesSeverity = !severityFilter || alert.severity === severityFilter;
            const matchesType = !typeFilter || alert.alert_type === typeFilter;
            const matchesStatus = !statusFilter ||
                (statusFilter === 'unread' && !alert.is_read) ||
                (statusFilter === 'read' && alert.is_read);
            
            const matchesSearch = !searchInput || 
                alert.title.toLowerCase().includes(searchInput) ||
                alert.message.toLowerCase().includes(searchInput) ||
                alert.patient_name.toLowerCase().includes(searchInput);

            return matchesSeverity && matchesType && matchesStatus && matchesSearch;
        });

        this.currentPage = 1;
        this.displayAlerts();
        this.updateStats();
    }

    updateStats() {
        const stats = this.calculateStats();
        
        document.getElementById('criticalCount').textContent = stats.critical;
        document.getElementById('mediumCount').textContent = stats.medium;
        document.getElementById('lowCount').textContent = stats.low;
        document.getElementById('unreadCount').textContent = stats.unread;
        document.getElementById('totalAlertsCount').textContent = this.filteredAlerts.length;
    }

    calculateStats() {
        return {
            critical: this.filteredAlerts.filter(a => a.severity === 'high').length,
            medium: this.filteredAlerts.filter(a => a.severity === 'medium').length,
            low: this.filteredAlerts.filter(a => a.severity === 'low').length,
            unread: this.filteredAlerts.filter(a => !a.is_read).length
        };
    }

    async refreshAlerts() {
        const refreshBtn = document.querySelector('[onclick*="refreshAlerts"]');
        const icon = refreshBtn?.querySelector('i');
        
        if (icon) {
            icon.classList.add('fa-spin');
            refreshBtn.disabled = true;
        }
        
        try {
            await this.loadAlerts();
            this.showToast('Alertes actualisées avec succès', 'success');
        } catch (error) {
            this.showToast('Erreur lors de l\'actualisation', 'error');
        } finally {
            if (icon) {
                icon.classList.remove('fa-spin');
                refreshBtn.disabled = false;
            }
        }
    }

    exportAlerts() {
        const csvContent = "data:text/csv;charset=utf-8," + 
            "ID,Titre,Message,Sévérité,Patient,Date,Statut\n" +
            this.filteredAlerts.map(alert => 
                `${alert.id},"${alert.title}","${alert.message}",${alert.severity},"${alert.patient_name}",${alert.created_at},${alert.is_read ? 'Lu' : 'Non lu'}`
            ).join("\n");

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `alertes_${new Date().toISOString().split('T')[0]}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        this.showToast('Export terminé avec succès', 'success');
    }

    clearFilters() {
        document.getElementById('severityFilter').value = '';
        document.getElementById('typeFilter').value = '';
        document.getElementById('statusFilter').value = '';
        document.getElementById('searchInput').value = '';
        this.applyFilters();
        this.showToast('Filtres effacés', 'info');
    }

    showToast(message, type = 'info') {
        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        toastContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            animation: slideInRight 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        `;

        const iconMap = {
            success: 'check-circle text-success',
            error: 'exclamation-circle text-danger',
            warning: 'exclamation-triangle text-warning',
            info: 'info-circle text-info'
        };

        toastContainer.innerHTML = `
            <div class="toast show" role="alert" style="
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                min-width: 300px;
            ">
                <div class="toast-header" style="background: transparent; border-bottom: 1px solid rgba(0, 0, 0, 0.1);">
                    <i class="fas fa-${iconMap[type]} me-2"></i>
                    <strong class="me-auto">NeuroScan</strong>
                    <button type="button" class="btn-close" onclick="this.closest('.toast-container').remove()"></button>
                </div>
                <div class="toast-body" style="font-weight: 500;">
                    ${message}
                </div>
            </div>
        `;

        document.body.appendChild(toastContainer);

        setTimeout(() => {
            if (toastContainer.parentNode) {
                toastContainer.style.animation = 'slideOutRight 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
                setTimeout(() => toastContainer.remove(), 400);
            }
        }, 4000);
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        if (diffDays === 1) return 'Aujourd\'hui';
        if (diffDays === 2) return 'Hier';
        if (diffDays <= 7) return `Il y a ${diffDays} jours`;
        
        return date.toLocaleDateString('fr-FR', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric'
        });
    }

    animateCards() {
        const cards = document.querySelectorAll('.alert-card');
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

    initializeAnimations() {
        // Ajouter les styles d'animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }

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
}

// Initialiser le gestionnaire d'alertes
let alertsManager;
document.addEventListener('DOMContentLoaded', () => {
    alertsManager = new AlertsManager();
});

// Fonctions globales pour la compatibilité
function refreshAlerts() {
    alertsManager?.refreshAlerts();
}

function exportAlerts() {
    alertsManager?.exportAlerts();
}

function clearFilters() {
    alertsManager?.clearFilters();
}
