/**
 * NeuroScan - Gestionnaire d'Alertes Moderne
 * Version: 2.0
 * Fonctionnalités: Real-time updates, Smart filtering, Audio notifications, Animations
 */

class AlertsManager {
    constructor() {
        this.alerts = [];
        this.filtered = [];
        this.currentPage = 1;
        this.pageSize = 10;
        this.currentAlert = null;
        this.autoRefreshInterval = null;
        this.filters = {
            severity: '',
            type: '',
            status: '',
            search: ''
        };
        this.sortBy = 'date';
        this.sortOrder = 'desc';
        this.soundEnabled = localStorage.getItem('alertsSoundEnabled') !== 'false';
        this.lastAlertCount = 0;
        
        // Précharger le son de notification pour de meilleures performances
        this.notificationSound = new Audio('/static/shop-notification-355746.mp3');
        this.notificationSound.volume = 0.5;
        this.notificationSound.preload = 'auto';
        
        this.init();
    }

    init() {
        console.log('🎯 Initialisation du gestionnaire d\'alertes...');
        this.setupEventListeners();
        this.loadAlerts();
        this.startAutoRefresh();
        this.setupKeyboardShortcuts();
        this.loadUserPreferences();
    }

    setupEventListeners() {
        // Filtres
        const severityFilter = document.getElementById('severityFilter');
        const typeFilter = document.getElementById('typeFilter');
        const statusFilter = document.getElementById('statusFilter');
        const searchInput = document.getElementById('searchInput');

        if (severityFilter) severityFilter.addEventListener('change', () => this.applyFilters());
        if (typeFilter) typeFilter.addEventListener('change', () => this.applyFilters());
        if (statusFilter) statusFilter.addEventListener('change', () => this.applyFilters());
        if (searchInput) searchInput.addEventListener('input', this.debounce(() => this.applyFilters(), 300));

        // Boutons d'action
        const resolveBtn = document.getElementById('alertDetailsResolveBtn');
        const viewBtn = document.getElementById('alertDetailsViewBtn');

        if (resolveBtn) resolveBtn.addEventListener('click', () => {
            if (this.currentAlert) this.resolveAlert(this.currentAlert.id);
        });

        if (viewBtn) viewBtn.addEventListener('click', () => {
            if (this.currentAlert) window.location.href = `/patient/${this.currentAlert.patient_id}`;
        });

        // Fermeture du modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.closeDetails();
        });

        // Click en dehors du modal
        const modal = document.getElementById('alertDetailsModal');
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) this.closeDetails();
            });
        }

        // Scroll trap sur la liste
        const listElement = document.getElementById('alertsPageList');
        if (listElement) this.trapScroll(listElement);
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + R : Rafraîchir
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                this.refresh();
            }
            // Ctrl/Cmd + M : Marquer tout comme lu
            if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
                e.preventDefault();
                this.markAllAsRead();
            }
            // Ctrl/Cmd + E : Exporter
            if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
                e.preventDefault();
                this.exportAlerts();
            }
        });
    }

    loadUserPreferences() {
        const savedPageSize = localStorage.getItem('alertsPageSize');
        if (savedPageSize) this.pageSize = parseInt(savedPageSize);

        const savedSort = localStorage.getItem('alertsSort');
        if (savedSort) {
            const [sortBy, sortOrder] = savedSort.split(':');
            this.sortBy = sortBy;
            this.sortOrder = sortOrder;
        }
    }

    async loadAlerts() {
        try {
            this.showLoadingState();
            
            const response = await fetch('/api/alerts');
            const data = await response.json();

            if (data && data.success) {
                const newAlertCount = data.data.length;
                
                // Détecter nouvelles alertes
                if (this.lastAlertCount > 0 && newAlertCount > this.lastAlertCount) {
                    const newAlerts = newAlertCount - this.lastAlertCount;
                    this.showNotification(
                        `🔔 ${newAlerts} nouvelle${newAlerts > 1 ? 's' : ''} alerte${newAlerts > 1 ? 's' : ''}`,
                        'info'
                    );
                    if (this.soundEnabled) this.playNotificationSound();
                }

                this.lastAlertCount = newAlertCount;
                this.alerts = data.data;
                this.sortAlerts();
                this.filtered = [...this.alerts];
                this.applyFilters();
                this.updateStatistics();
                this.hideLoadingState();
            } else {
                throw new Error('Échec du chargement des alertes');
            }
        } catch (error) {
            console.error('❌ Erreur chargement alertes:', error);
            this.showErrorState();
            this.showNotification('Erreur lors du chargement des alertes', 'error');
        }
    }

    sortAlerts() {
        this.alerts.sort((a, b) => {
            let comparison = 0;

            switch (this.sortBy) {
                case 'date':
                    comparison = new Date(b.created_at) - new Date(a.created_at);
                    break;
                case 'severity':
                    const severityOrder = { high: 3, medium: 2, low: 1 };
                    comparison = severityOrder[b.severity] - severityOrder[a.severity];
                    break;
                case 'patient':
                    comparison = (a.patient_name || '').localeCompare(b.patient_name || '');
                    break;
                case 'type':
                    comparison = (a.alert_type || '').localeCompare(b.alert_type || '');
                    break;
            }

            return this.sortOrder === 'desc' ? comparison : -comparison;
        });
    }

    applyFilters() {
        const severityValue = document.getElementById('severityFilter')?.value || '';
        const typeValue = document.getElementById('typeFilter')?.value || '';
        const statusValue = document.getElementById('statusFilter')?.value || '';
        const searchValue = (document.getElementById('searchInput')?.value || '').toLowerCase();

        this.filters = {
            severity: severityValue,
            type: typeValue,
            status: statusValue,
            search: searchValue
        };

        this.filtered = this.alerts.filter(alert => {
            // Filtre sévérité
            if (this.filters.severity && alert.severity !== this.filters.severity) {
                return false;
            }

            // Filtre type
            if (this.filters.type && alert.alert_type !== this.filters.type) {
                return false;
            }

            // Filtre statut
            if (this.filters.status) {
                if (this.filters.status === 'unread' && alert.is_read) return false;
                if (this.filters.status === 'read' && !alert.is_read) return false;
            }

            // Recherche
            if (this.filters.search) {
                const searchFields = [
                    alert.title,
                    alert.message,
                    alert.patient_name,
                    String(alert.patient_id)
                ].join(' ').toLowerCase();

                if (!searchFields.includes(this.filters.search)) {
                    return false;
                }
            }

            return true;
        });

        this.currentPage = 1;
        this.displayAlerts();
        this.updateStatistics();
        this.animateFilterChange();
    }

    displayAlerts() {
        const listElement = document.getElementById('alertsPageList');
        if (!listElement) return;

        const startIndex = (this.currentPage - 1) * this.pageSize;
        const endIndex = startIndex + this.pageSize;
        const pageAlerts = this.filtered.slice(startIndex, endIndex);

        if (pageAlerts.length === 0) {
            this.showEmptyState();
            this.renderPagination(0);
            return;
        }

        listElement.innerHTML = pageAlerts.map(alert => this.createAlertCard(alert)).join('');
        
        // Animation d'entrée
        setTimeout(() => {
            const cards = listElement.querySelectorAll('.alert-card');
            cards.forEach((card, index) => {
                card.style.animation = `slideInUp 0.4s ease-out ${index * 0.05}s both`;
            });
        }, 10);

        const totalPages = Math.ceil(this.filtered.length / this.pageSize);
        this.renderPagination(totalPages);
    }

    createAlertCard(alert) {
        const severityConfig = this.getSeverityConfig(alert.severity);
        const typeText = this.getAlertTypeText(alert.alert_type);
        const formattedDate = this.formatDate(alert.created_at);
        const isUnread = !alert.is_read;

        return `
            <div class="alert-card patient-card ${isUnread ? 'alert-unread' : ''} alert-${alert.severity}" 
                 onclick="alertsManager.showDetails(${alert.id})"
                 data-alert-id="${alert.id}">
                <div class="flex items-start gap-4">
                    <!-- Indicateur de sévérité -->
                    <div class="flex-shrink-0 flex items-center">
                        <span class="lamp-dot ${severityConfig.lampClass}" 
                              title="${severityConfig.text}"></span>
                        <div class="w-12 h-12 rounded-xl flex items-center justify-center ${severityConfig.bgClass} transition-transform hover:scale-110">
                            <i class="fas fa-${severityConfig.icon} ${severityConfig.textClass} text-lg"></i>
                        </div>
                    </div>

                    <!-- Contenu principal -->
                    <div class="flex-1 min-w-0">
                        <!-- En-tête -->
                        <div class="flex items-start justify-between gap-3 mb-2">
                            <h4 class="font-bold text-gray-900 text-lg flex items-center gap-2">
                                ${this.escapeHTML(alert.title)}
                                ${isUnread ? '<span class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>' : ''}
                            </h4>
                            <div class="flex items-center gap-2 flex-shrink-0">
                                <span class="severity-badge badge-${severityConfig.badgeClass}">
                                    ${severityConfig.text}
                                </span>
                                <span class="status-chip ${isUnread ? 'status-unread' : 'status-read'}">
                                    <i class="fas fa-${isUnread ? 'envelope' : 'envelope-open'}"></i>
                                    ${isUnread ? 'Non lue' : 'Lue'}
                                </span>
                            </div>
                        </div>

                        <!-- Message -->
                        <p class="text-gray-600 mb-3 leading-relaxed">
                            ${this.escapeHTML(alert.message)}
                        </p>

                        <!-- Métadonnées -->
                        <div class="flex flex-wrap items-center gap-2 mb-3">
                            <span class="meta-chip">
                                <i class="fas fa-tag"></i>
                                ${typeText}
                            </span>
                            ${alert.patient_name ? `
                                <span class="meta-chip">
                                    <i class="fas fa-user"></i>
                                    ${this.escapeHTML(alert.patient_name)}
                                </span>
                            ` : ''}
                            ${alert.patient_id ? `
                                <span class="meta-chip">
                                    <i class="fas fa-hashtag"></i>
                                    ${this.escapeHTML(String(alert.patient_id))}
                                </span>
                            ` : ''}
                            <span class="meta-chip">
                                <i class="fas fa-clock"></i>
                                ${formattedDate}
                            </span>
                        </div>

                        <!-- Actions -->
                        <div class="flex items-center gap-2">
                            ${isUnread ? `
                                <button class="btn-dashboard secondary text-sm" 
                                        onclick="event.stopPropagation(); alertsManager.markAsRead(${alert.id})"
                                        title="Marquer comme lu">
                                    <i class="fas fa-check"></i>
                                    Marquer lu
                                </button>
                            ` : ''}
                            <button class="btn-dashboard success text-sm" 
                                    onclick="event.stopPropagation(); alertsManager.resolveAlert(${alert.id})"
                                    title="Résoudre l'alerte">
                                <i class="fas fa-check-circle"></i>
                                Résoudre
                            </button>
                            <button class="btn-dashboard primary text-sm" 
                                    onclick="event.stopPropagation(); alertsManager.showDetails(${alert.id})"
                                    title="Voir les détails">
                                <i class="fas fa-eye"></i>
                                Détails
                            </button>
                            ${alert.patient_id ? `
                                <button class="btn-dashboard info text-sm" 
                                        onclick="event.stopPropagation(); window.location.href='/patient/${alert.patient_id}'"
                                        title="Voir le dossier patient">
                                    <i class="fas fa-user-circle"></i>
                                    Patient
                                </button>
                            ` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    showDetails(alertId) {
        const alert = this.alerts.find(a => a.id === alertId);
        if (!alert) return;

        this.currentAlert = alert;
        const severityConfig = this.getSeverityConfig(alert.severity);
        const typeText = this.getAlertTypeText(alert.alert_type);

        const modalBody = document.getElementById('alertDetailsBody');
        if (modalBody) {
            modalBody.innerHTML = `
                <div class="space-y-6">
                    <!-- Carte d'information principale -->
                    <div class="bg-gradient-to-br from-${severityConfig.colorName}-50 to-white rounded-2xl p-6 border-2 border-${severityConfig.colorName}-200">
                        <div class="flex items-start gap-4">
                            <div class="w-16 h-16 ${severityConfig.bgClass} rounded-2xl flex items-center justify-center flex-shrink-0">
                                <i class="fas fa-${severityConfig.icon} ${severityConfig.textClass} text-2xl"></i>
                            </div>
                            <div class="flex-1">
                                <h3 class="text-2xl font-bold text-gray-900 mb-2">
                                    ${this.escapeHTML(alert.title)}
                                </h3>
                                <p class="text-gray-700 leading-relaxed">
                                    ${this.escapeHTML(alert.message)}
                                </p>
                            </div>
                        </div>
                    </div>

                    <!-- Grille d'informations -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <!-- Informations de l'alerte -->
                        <div class="dashboard-card p-5">
                            <h4 class="font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <i class="fas fa-info-circle text-blue-600"></i>
                                Informations de l'alerte
                            </h4>
                            <div class="space-y-3 text-sm">
                                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                                    <span class="text-gray-600 font-medium">Type d'alerte</span>
                                    <span class="font-semibold text-gray-900">${typeText}</span>
                                </div>
                                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                                    <span class="text-gray-600 font-medium">Sévérité</span>
                                    <span class="severity-badge badge-${severityConfig.badgeClass}">
                                        ${severityConfig.text}
                                    </span>
                                </div>
                                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                                    <span class="text-gray-600 font-medium">Date de création</span>
                                    <span class="font-semibold text-gray-900">${this.formatDate(alert.created_at, true)}</span>
                                </div>
                                <div class="flex justify-between items-center py-2">
                                    <span class="text-gray-600 font-medium">Statut</span>
                                    <span class="status-chip ${alert.is_read ? 'status-read' : 'status-unread'}">
                                        <i class="fas fa-${alert.is_read ? 'envelope-open' : 'envelope'}"></i>
                                        ${alert.is_read ? 'Lue' : 'Non lue'}
                                    </span>
                                </div>
                            </div>
                        </div>

                        <!-- Informations du patient -->
                        <div class="dashboard-card p-5">
                            <h4 class="font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <i class="fas fa-user-circle text-green-600"></i>
                                Informations patient
                            </h4>
                            <div class="space-y-3 text-sm">
                                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                                    <span class="text-gray-600 font-medium">Nom complet</span>
                                    <span class="font-semibold text-gray-900">
                                        ${this.escapeHTML(alert.patient_name || 'N/A')}
                                    </span>
                                </div>
                                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                                    <span class="text-gray-600 font-medium">ID Patient</span>
                                    <span class="font-mono font-semibold text-gray-900">
                                        ${this.escapeHTML(String(alert.patient_id || 'N/A'))}
                                    </span>
                                </div>
                                <div class="flex justify-between items-center py-2">
                                    <span class="text-gray-600 font-medium">Dossier</span>
                                    ${alert.patient_id ? `
                                        <a href="/patient/${alert.patient_id}" 
                                           class="text-blue-600 hover:text-blue-800 font-semibold flex items-center gap-1">
                                            Consulter
                                            <i class="fas fa-external-link-alt text-xs"></i>
                                        </a>
                                    ` : '<span class="text-gray-400">Non disponible</span>'}
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Actions recommandées -->
                    <div class="bg-blue-50 border-2 border-blue-200 rounded-xl p-5">
                        <h4 class="font-bold text-blue-900 mb-3 flex items-center gap-2">
                            <i class="fas fa-lightbulb"></i>
                            Actions recommandées
                        </h4>
                        <ul class="space-y-2 text-sm text-blue-800">
                            ${this.getRecommendedActions(alert).map(action => `
                                <li class="flex items-start gap-2">
                                    <i class="fas fa-check-circle mt-0.5 flex-shrink-0"></i>
                                    <span>${action}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
            `;
        }

        const modal = document.getElementById('alertDetailsModal');
        if (modal) {
            modal.classList.remove('hidden');
            modal.style.animation = 'fadeIn 0.3s ease-out';
            document.body.style.overflow = 'hidden';
        }

        // Marquer comme lu automatiquement
        if (!alert.is_read) {
            setTimeout(() => this.markAsRead(alertId), 1000);
        }
    }

    closeDetails() {
        const modal = document.getElementById('alertDetailsModal');
        if (modal) {
            modal.style.animation = 'fadeOut 0.3s ease-out';
            setTimeout(() => {
                modal.classList.add('hidden');
                document.body.style.overflow = 'auto';
            }, 300);
        }
        this.currentAlert = null;
    }

    async markAsRead(alertId) {
        try {
            const response = await fetch(`/api/alerts/${alertId}/mark-read`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (response.ok) {
                // Mettre à jour localement
                const alert = this.alerts.find(a => a.id === alertId);
                if (alert) alert.is_read = true;

                const filteredAlert = this.filtered.find(a => a.id === alertId);
                if (filteredAlert) filteredAlert.is_read = true;

                this.displayAlerts();
                this.updateStatistics();
                this.showNotification('Alerte marquée comme lue', 'success');
            }
        } catch (error) {
            console.error('Erreur marquage lecture:', error);
            this.showNotification('Erreur lors du marquage', 'error');
        }
    }

    async resolveAlert(alertId) {
        if (!confirm('Êtes-vous sûr de vouloir résoudre cette alerte ?')) return;

        try {
            const response = await fetch(`/api/alerts/${alertId}/resolve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (response.ok) {
                // Animer la suppression
                const card = document.querySelector(`[data-alert-id="${alertId}"]`);
                if (card) {
                    card.style.animation = 'slideOutRight 0.4s ease-out';
                    setTimeout(() => {
                        this.alerts = this.alerts.filter(a => a.id !== alertId);
                        this.filtered = this.filtered.filter(a => a.id !== alertId);
                        this.displayAlerts();
                        this.updateStatistics();
                    }, 400);
                } else {
                    this.alerts = this.alerts.filter(a => a.id !== alertId);
                    this.filtered = this.filtered.filter(a => a.id !== alertId);
                    this.displayAlerts();
                    this.updateStatistics();
                }

                this.closeDetails();
                this.showNotification('✅ Alerte résolue avec succès', 'success');
            }
        } catch (error) {
            console.error('Erreur résolution:', error);
            this.showNotification('Erreur lors de la résolution', 'error');
        }
    }

    async markAllAsRead() {
        const unreadAlerts = this.alerts.filter(a => !a.is_read);
        if (unreadAlerts.length === 0) {
            this.showNotification('Aucune alerte non lue', 'info');
            return;
        }

        if (!confirm(`Marquer ${unreadAlerts.length} alerte(s) comme lue(s) ?`)) return;

        try {
            const promises = unreadAlerts.map(alert => this.markAsRead(alert.id));
            await Promise.all(promises);
            this.showNotification(`${unreadAlerts.length} alerte(s) marquée(s) comme lue(s)`, 'success');
        } catch (error) {
            console.error('Erreur marquage multiple:', error);
        }
    }

    exportAlerts() {
        const data = this.filtered.map(alert => ({
            'ID': alert.id,
            'Titre': alert.title,
            'Message': alert.message,
            'Sévérité': this.getSeverityConfig(alert.severity).text,
            'Type': this.getAlertTypeText(alert.alert_type),
            'Patient': alert.patient_name || '',
            'ID Patient': alert.patient_id || '',
            'Date': this.formatDate(alert.created_at, true),
            'Statut': alert.is_read ? 'Lue' : 'Non lue'
        }));

        const csv = this.convertToCSV(data);
        const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `alertes_neuroscan_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        this.showNotification('📥 Export terminé avec succès', 'success');
    }

    convertToCSV(data) {
        if (data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const rows = data.map(row => 
            headers.map(header => {
                const value = row[header] || '';
                return `"${String(value).replace(/"/g, '""')}"`;
            }).join(',')
        );

        return [headers.join(','), ...rows].join('\n');
    }

    clearFilters() {
        document.getElementById('severityFilter').value = '';
        document.getElementById('typeFilter').value = '';
        document.getElementById('statusFilter').value = '';
        document.getElementById('searchInput').value = '';
        this.applyFilters();
        this.showNotification('Filtres effacés', 'info');
    }

    refresh() {
        this.showNotification('🔄 Actualisation...', 'info');
        this.loadAlerts();
    }

    updateStatistics() {
        const criticalCount = this.alerts.filter(a => a.severity === 'high').length;
        const mediumCount = this.alerts.filter(a => a.severity === 'medium').length;
        const lowCount = this.alerts.filter(a => a.severity === 'low').length;
        const unreadCount = this.alerts.filter(a => !a.is_read).length;

        this.animateCounter('criticalCount', criticalCount);
        this.animateCounter('mediumCount', mediumCount);
        this.animateCounter('lowCount', lowCount);
        this.animateCounter('unreadCount', unreadCount);

        // Mettre à jour les nouveaux compteurs de la section Alertes Actives
        this.animateCounter('unreadBadgeMini', unreadCount);
        this.animateCounter('unreadCountStatus', unreadCount);

        const totalElement = document.getElementById('totalAlertsCount');
        if (totalElement) totalElement.textContent = this.filtered.length;

        // Compter les filtres actifs
        const activeFilters = [
            this.filters.severity,
            this.filters.type,
            this.filters.status,
            this.filters.search
        ].filter(f => f).length;

        const activeFiltersElement = document.getElementById('activeFiltersCount');
        if (activeFiltersElement) {
            activeFiltersElement.textContent = activeFilters;
            activeFiltersElement.className = activeFilters > 0 
                ? 'font-semibold text-blue-600' 
                : 'font-semibold text-gray-900';
        }

        // Mettre à jour les informations de pagination
        const paginationInfo = document.getElementById('paginationInfo');
        if (paginationInfo) {
            const totalPages = Math.ceil(this.filtered.length / this.pageSize);
            paginationInfo.textContent = `Page ${this.currentPage} / ${totalPages}`;
        }
    }

    animateCounter(elementId, targetValue) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const currentValue = parseInt(element.textContent) || 0;
        if (currentValue === targetValue) return;

        const duration = 500;
        const steps = 20;
        const increment = (targetValue - currentValue) / steps;
        let current = currentValue;
        let step = 0;

        const timer = setInterval(() => {
            step++;
            current += increment;
            
            if (step >= steps) {
                element.textContent = targetValue;
                clearInterval(timer);
            } else {
                element.textContent = Math.round(current);
            }
        }, duration / steps);
    }

    renderPagination(totalPages) {
        const container = document.getElementById('paginationContainer');
        if (!container) return;

        if (totalPages <= 1) {
            container.innerHTML = '';
            return;
        }

        let html = `
            <div class="flex items-center gap-2">
                <button class="btn-dashboard secondary text-sm" 
                        ${this.currentPage === 1 ? 'disabled' : ''}
                        onclick="alertsManager.changePage(${this.currentPage - 1})">
                    <i class="fas fa-chevron-left"></i>
                </button>
        `;

        for (let i = 1; i <= totalPages; i++) {
            if (i === this.currentPage) {
                html += `<button class="btn-dashboard primary text-sm">${i}</button>`;
            } else if (i === 1 || i === totalPages || Math.abs(i - this.currentPage) <= 1) {
                html += `<button class="btn-dashboard secondary text-sm" 
                                onclick="alertsManager.changePage(${i})">${i}</button>`;
            } else if (i === this.currentPage - 2 || i === this.currentPage + 2) {
                html += `<span class="text-gray-400 px-2">...</span>`;
            }
        }

        html += `
                <button class="btn-dashboard secondary text-sm" 
                        ${this.currentPage === totalPages ? 'disabled' : ''}
                        onclick="alertsManager.changePage(${this.currentPage + 1})">
                    <i class="fas fa-chevron-right"></i>
                </button>
            </div>
        `;

        container.innerHTML = html;
    }

    changePage(page) {
        const totalPages = Math.ceil(this.filtered.length / this.pageSize);
        if (page >= 1 && page <= totalPages) {
            this.currentPage = page;
            this.displayAlerts();
            
            // Scroll vers le haut de la liste
            const listElement = document.getElementById('alertsPageList');
            if (listElement) {
                listElement.scrollTo({ top: 0, behavior: 'smooth' });
            }
        }
    }

    startAutoRefresh() {
        // Rafraîchir toutes les 2 minutes
        this.autoRefreshInterval = setInterval(() => {
            this.loadAlerts();
        }, 120000);
    }

    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    }

    // Utilitaires
    getSeverityConfig(severity) {
        const configs = {
            high: {
                text: 'Critique',
                icon: 'exclamation-triangle',
                lampClass: 'lamp-danger',
                badgeClass: 'danger',
                bgClass: 'bg-red-100',
                textClass: 'text-red-600',
                colorName: 'red'
            },
            medium: {
                text: 'Moyenne',
                icon: 'exclamation-circle',
                lampClass: 'lamp-warning',
                badgeClass: 'warning',
                bgClass: 'bg-amber-100',
                textClass: 'text-amber-600',
                colorName: 'amber'
            },
            low: {
                text: 'Faible',
                icon: 'info-circle',
                lampClass: 'lamp-info',
                badgeClass: 'info',
                bgClass: 'bg-blue-100',
                textClass: 'text-blue-600',
                colorName: 'blue'
            }
        };
        return configs[severity] || configs.low;
    }

    getAlertTypeText(type) {
        const types = {
            'new_tumor_detected': 'Nouvelle tumeur détectée',
            'diagnosis_change': 'Changement de diagnostic',
            'rapid_growth': 'Croissance rapide',
            'confidence_drop': 'Baisse de confiance',
            'tumor_resolved': 'Amélioration significative',
            'high_grade_tumor': 'Tumeur de haut grade',
            'follow_up_required': 'Suivi requis',
            'urgent_review': 'Révision urgente'
        };
        return types[type] || type || 'Type inconnu';
    }

    getRecommendedActions(alert) {
        const actions = {
            'new_tumor_detected': [
                'Planifier une consultation avec le patient',
                'Effectuer des examens complémentaires',
                'Évaluer les options de traitement',
                'Informer le patient des résultats'
            ],
            'diagnosis_change': [
                'Revoir le dossier médical complet',
                'Consulter un spécialiste si nécessaire',
                'Mettre à jour le plan de traitement',
                'Communiquer les changements au patient'
            ],
            'rapid_growth': [
                'Évaluation urgente recommandée',
                'Envisager une intervention rapide',
                'Surveillance accrue du patient',
                'Consultation multidisciplinaire'
            ],
            'confidence_drop': [
                'Vérifier la qualité des images',
                'Demander des examens supplémentaires',
                'Faire une double lecture',
                'Documenter les incertitudes'
            ]
        };
        return actions[alert.alert_type] || [
            'Examiner l\'alerte en détail',
            'Consulter le dossier patient',
            'Prendre les mesures appropriées',
            'Documenter les actions effectuées'
        ];
    }

    formatDate(dateString, detailed = false) {
        const date = new Date(dateString);
        if (isNaN(date)) return 'Date invalide';

        if (detailed) {
            return date.toLocaleString('fr-FR', {
                day: '2-digit',
                month: 'long',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        }

        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 1) return "À l'instant";
        if (diffMins < 60) return `Il y a ${diffMins} min`;
        if (diffHours < 24) return `Il y a ${diffHours}h`;
        if (diffDays < 7) return `Il y a ${diffDays}j`;

        return date.toLocaleDateString('fr-FR');
    }

    showNotification(message, type = 'info') {
        // Réutiliser la fonction globale si elle existe
        if (typeof showNotification === 'function') {
            showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }

    showLoadingState() {
        const listElement = document.getElementById('alertsPageList');
        if (listElement) {
            listElement.innerHTML = `
                <div class="flex flex-col items-center justify-center py-20">
                    <div class="relative">
                        <div class="w-16 h-16 border-4 border-blue-200 rounded-full animate-spin"></div>
                        <div class="absolute top-0 left-0 w-16 h-16 border-4 border-blue-600 rounded-full animate-spin border-t-transparent"></div>
                    </div>
                    <p class="text-gray-600 mt-6 text-lg font-medium">Chargement des alertes...</p>
                    <div class="flex gap-2 mt-4">
                        <div class="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                        <div class="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style="animation-delay: 0.1s;"></div>
                        <div class="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style="animation-delay: 0.2s;"></div>
                    </div>
                </div>
            `;
        }
    }

    hideLoadingState() {
        // La méthode displayAlerts() remplacera le contenu
    }

    showErrorState() {
        const listElement = document.getElementById('alertsPageList');
        if (listElement) {
            listElement.innerHTML = `
                <div class="text-center py-20">
                    <div class="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-6">
                        <i class="fas fa-exclamation-triangle text-red-600 text-3xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-2">Erreur de chargement</h3>
                    <p class="text-gray-600 mb-6">Impossible de charger les alertes</p>
                    <button onclick="alertsManager.refresh()" class="btn-dashboard primary">
                        <i class="fas fa-sync-alt mr-2"></i>
                        Réessayer
                    </button>
                </div>
            `;
        }
    }

    showEmptyState() {
        const listElement = document.getElementById('alertsPageList');
        if (listElement) {
            const hasFilters = this.filters.severity || this.filters.type || 
                             this.filters.status || this.filters.search;

            listElement.innerHTML = `
                <div class="text-center py-20">
                    <div class="w-24 h-24 bg-gradient-to-br from-green-100 to-emerald-100 rounded-full flex items-center justify-center mx-auto mb-6">
                        <i class="fas fa-${hasFilters ? 'filter' : 'check-circle'} text-green-600 text-4xl"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-gray-900 mb-3">
                        ${hasFilters ? 'Aucune alerte trouvée' : 'Aucune alerte active'}
                    </h3>
                    <p class="text-gray-600 mb-6 max-w-md mx-auto">
                        ${hasFilters 
                            ? 'Essayez d\'élargir vos critères de recherche ou effacez les filtres' 
                            : 'Excellent ! Vous n\'avez aucune alerte en attente'}
                    </p>
                    ${hasFilters ? `
                        <button onclick="alertsManager.clearFilters()" class="btn-dashboard secondary">
                            <i class="fas fa-times mr-2"></i>
                            Effacer les filtres
                        </button>
                    ` : ''}
                </div>
            `;
        }
    }

    animateFilterChange() {
        const listElement = document.getElementById('alertsPageList');
        if (listElement) {
            listElement.style.opacity = '0.5';
            setTimeout(() => {
                listElement.style.transition = 'opacity 0.3s ease';
                listElement.style.opacity = '1';
            }, 50);
        }
    }

    playNotificationSound() {
        if (!this.soundEnabled) return;
        
        // Utiliser le son préchargé pour de meilleures performances
        try {
            // Cloner l'audio pour permettre plusieurs lectures simultanées
            const audio = this.notificationSound.cloneNode();
            audio.volume = 1; // Volume à 50%
            
            // Jouer le son
            audio.play().catch(error => {
                console.log('Impossible de jouer le son:', error);
                // Fallback vers le son synthétique si le fichier ne peut pas être lu
                this.playSyntheticSound();
            });
        } catch (error) {
            console.error('Erreur lors de la lecture du son:', error);
            // Fallback vers le son synthétique
            this.playSyntheticSound();
        }
    }

    playSyntheticSound() {
        // Son de notification synthétique (fallback)
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.frequency.value = 800;
            oscillator.type = 'sine';

            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);

            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.3);
        } catch (error) {
            console.error('Impossible de jouer le son synthétique:', error);
        }
    }

    escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
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

    trapScroll(element) {
        element.addEventListener('wheel', (e) => {
            const delta = e.deltaY;
            const atTop = element.scrollTop === 0;
            const atBottom = Math.ceil(element.scrollTop + element.clientHeight) >= element.scrollHeight;

            if ((delta < 0 && atTop) || (delta > 0 && atBottom)) {
                e.preventDefault();
            }
            e.stopPropagation();
        }, { passive: false });
    }
}

// Initialiser au chargement
let alertsManager;
document.addEventListener('DOMContentLoaded', () => {
    alertsManager = new AlertsManager();
    
    // Exposer globalement pour les onclick
    window.alertsManager = alertsManager;
});

// Nettoyer au déchargement
window.addEventListener('beforeunload', () => {
    if (alertsManager) {
        alertsManager.stopAutoRefresh();
    }
});

// Fonctions globales supplémentaires pour la nouvelle interface

/**
 * Changer la taille de page
 */
function changePageSize(size) {
    if (alertsManager) {
        alertsManager.pageSize = parseInt(size);
        localStorage.setItem('alertsPageSize', size);
        alertsManager.currentPage = 1;
        alertsManager.displayAlerts();
        showNotification(`Affichage mis à jour: ${size} alertes par page`, 'info');
    }
}

/**
 * Toggle des paramètres d'alertes
 */
function toggleAlertsSettings() {
    const settingsPanel = document.getElementById('alertsSettingsPanel');
    
    if (!settingsPanel) {
        // Créer le panneau de paramètres
        const panel = document.createElement('div');
        panel.id = 'alertsSettingsPanel';
        panel.className = 'fixed top-20 right-4 w-80 bg-white rounded-2xl shadow-2xl border border-gray-200 p-6 z-50 transform transition-all duration-300';
        panel.style.animation = 'slideInRight 0.3s ease-out';
        
        panel.innerHTML = `
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-bold text-gray-900 flex items-center gap-2">
                    <i class="fas fa-cog text-blue-600"></i>
                    Paramètres
                </h3>
                <button onclick="toggleAlertsSettings()" class="text-gray-400 hover:text-gray-600">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            
            <div class="space-y-4">
                <!-- Son des notifications -->
                <div class="flex items-center justify-between">
                    <div>
                        <label class="font-semibold text-gray-900">Son des notifications</label>
                        <p class="text-xs text-gray-600">Alerte sonore pour nouvelles alertes</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" id="soundToggle" class="sr-only peer" 
                               ${alertsManager.soundEnabled ? 'checked' : ''}
                               onchange="toggleSound(this.checked)">
                        <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </label>
                </div>

                <!-- Taille de page -->
                <div>
                    <label class="font-semibold text-gray-900 block mb-2">Alertes par page</label>
                    <select class="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            onchange="changePageSize(this.value)">
                        <option value="10" ${alertsManager.pageSize === 10 ? 'selected' : ''}>10 alertes</option>
                        <option value="20" ${alertsManager.pageSize === 20 ? 'selected' : ''}>20 alertes</option>
                        <option value="50" ${alertsManager.pageSize === 50 ? 'selected' : ''}>50 alertes</option>
                        <option value="100" ${alertsManager.pageSize === 100 ? 'selected' : ''}>100 alertes</option>
                    </select>
                </div>

                <!-- Intervalle de rafraîchissement -->
                <div>
                    <label class="font-semibold text-gray-900 block mb-2">Auto-refresh</label>
                    <select class="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            onchange="changeRefreshInterval(this.value)">
                        <option value="60">Chaque minute</option>
                        <option value="120" selected>Toutes les 2 minutes</option>
                        <option value="300">Toutes les 5 minutes</option>
                        <option value="0">Désactivé</option>
                    </select>
                </div>

                <!-- Boutons d'action -->
                <div class="pt-4 border-t border-gray-200 space-y-2">
                    <button onclick="resetAlertsSettings()" 
                            class="w-full px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors font-medium">
                        <i class="fas fa-undo mr-2"></i>
                        Réinitialiser
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(panel);
    } else {
        // Fermer le panneau
        settingsPanel.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => {
            settingsPanel.remove();
        }, 300);
    }
}

/**
 * Toggle du son
 */
function toggleSound(enabled) {
    if (alertsManager) {
        alertsManager.soundEnabled = enabled;
        localStorage.setItem('alertsSoundEnabled', enabled);
        showNotification(
            enabled ? '🔊 Sons activés' : '🔇 Sons désactivés',
            'info'
        );
    }
}

/**
 * Changer l'intervalle de rafraîchissement
 */
function changeRefreshInterval(seconds) {
    if (alertsManager) {
        alertsManager.stopAutoRefresh();
        
        if (seconds > 0) {
            alertsManager.autoRefreshInterval = setInterval(() => {
                alertsManager.loadAlerts();
            }, seconds * 1000);
            showNotification(`Auto-refresh: toutes les ${seconds / 60} minute(s)`, 'success');
        } else {
            showNotification('Auto-refresh désactivé', 'info');
        }
    }
}

/**
 * Réinitialiser les paramètres
 */
function resetAlertsSettings() {
    if (confirm('Réinitialiser tous les paramètres aux valeurs par défaut ?')) {
        localStorage.removeItem('alertsPageSize');
        localStorage.removeItem('alertsSoundEnabled');
        localStorage.removeItem('alertsSort');
        
        if (alertsManager) {
            alertsManager.pageSize = 10;
            alertsManager.soundEnabled = true;
            alertsManager.sortBy = 'date';
            alertsManager.sortOrder = 'desc';
            
            changeRefreshInterval(120);
            alertsManager.displayAlerts();
        }
        
        toggleAlertsSettings();
        showNotification('Paramètres réinitialisés', 'success');
    }
}

// Ajouter les animations CSS si elles n'existent pas
if (!document.getElementById('alerts-animations-style')) {
    const style = document.createElement('style');
    style.id = 'alerts-animations-style';
    style.textContent = `
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideOutRight {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(100%);
            }
        }
    `;
    document.head.appendChild(style);
}
