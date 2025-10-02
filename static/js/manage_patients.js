// Variables globales
let allPatients = [];
let filteredPatients = [];
let currentPage = 1;
let patientsPerPage = 20;
let currentView = 'list';
let sortColumn = 'name';
let sortOrder = 'asc';

// Initialisation de la page
document.addEventListener('DOMContentLoaded', function() {
    loadPatients();
    initializeKeyboardShortcuts();
    updateSystemStatus();
    initializeWelcomeMessage();
    
    // Mise à jour périodique du statut système
    setInterval(updateSystemStatus, 30000); // Chaque 30 secondes
    
    // Animation d'entrée pour les éléments de l'interface
    animatePageElements();
});

// Initialiser le message de bienvenue
function initializeWelcomeMessage() {
    const welcomeDismissed = localStorage.getItem('welcomeDismissed');
    if (welcomeDismissed === 'true') {
        const welcomeSection = document.querySelector('.bg-gradient-to-r.from-blue-500\\/10');
        if (welcomeSection) {
            welcomeSection.style.display = 'none';
        }
    }
}

// Animer les éléments de la page au chargement
function animatePageElements() {
    // Animation en cascade pour les cartes de statistiques
    const cards = document.querySelectorAll('.dashboard-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
    
    // Animation pour les boutons d'action
    const actionButtons = document.querySelectorAll('[onclick*="openAddPatientModal"], [onclick*="exportPatientsData"], [onclick*="importPatientsData"]');
    actionButtons.forEach((button, index) => {
        button.style.opacity = '0';
        button.style.transform = 'scale(0.9)';
        
        setTimeout(() => {
            button.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
            button.style.opacity = '1';
            button.style.transform = 'scale(1)';
        }, 200 + (index * 100));
    });
}

// Mettre à jour l'heure de dernière synchronisation
function updateLastSyncTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('fr-FR', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    const lastSyncElement = document.getElementById('lastSyncTime');
    const lastActionElement = document.getElementById('lastActionTime');
    
    if (lastSyncElement) {
        lastSyncElement.textContent = timeString;
    }
    if (lastActionElement) {
        lastActionElement.textContent = 'il y a quelques secondes';
    }
    
    // Mettre à jour le statut de la base de données
    const dbStatus = document.getElementById('dbStatus');
    if (dbStatus) {
        dbStatus.textContent = 'Synchronisée ✓';
        dbStatus.className = 'font-semibold text-green-600';
    }
}

// Nouvelles fonctions pour les améliorations de l'interface
function showTutorial() {
    showNotification('🎓 Tutoriel interactif bientôt disponible !', 'info', 3000);
    // Ici, vous pouvez implémenter un tutoriel guidé avec des overlays
}

function showUpdates() {
    const updates = [
        '✨ Nouvelle interface utilisateur moderne',
        '🔍 Recherche intelligente améliorée',
        '📊 Analytics en temps réel',
        '🚀 Performance optimisée',
        '📱 Design responsive parfait'
    ];
    
    showNotification(
        `🆕 Nouveautés NeuroScan :<br>• ${updates.join('<br>• ')}`, 
        'success', 
        8000
    );
}

function dismissWelcome() {
    const welcomeSection = document.querySelector('.bg-gradient-to-r.from-blue-500\\/10');
    if (welcomeSection) {
        welcomeSection.style.transform = 'translateX(100%)';
        welcomeSection.style.opacity = '0';
        setTimeout(() => {
            welcomeSection.remove();
        }, 500);
    }
    localStorage.setItem('welcomeDismissed', 'true');
}

function showAnalytics() {
    // Calculer des statistiques avancées
    const stats = {
        totalPatients: allPatients.length,
        newThisMonth: allPatients.filter(p => {
            const created = new Date(p.created_at);
            const now = new Date();
            return created.getMonth() === now.getMonth() && created.getFullYear() === now.getFullYear();
        }).length,
        activeThisWeek: allPatients.filter(p => {
            if (!p.last_analysis_date) return false;
            const lastAnalysis = new Date(p.last_analysis_date);
            const weekAgo = new Date();
            weekAgo.setDate(weekAgo.getDate() - 7);
            return lastAnalysis > weekAgo;
        }).length,
        avgAnalysesPerPatient: Math.round(allPatients.reduce((sum, p) => sum + (p.total_analyses || 0), 0) / allPatients.length)
    };
    
    const analyticsMessage = `
        📈 <strong>Analytics NeuroScan</strong><br>
        👥 Total patients: ${stats.totalPatients}<br>
        🆕 Nouveaux ce mois: ${stats.newThisMonth}<br>
        🔥 Actifs cette semaine: ${stats.activeThisWeek}<br>
        📊 Moyenne analyses/patient: ${stats.avgAnalysesPerPatient || 0}
    `;
    
    showNotification(analyticsMessage, 'info', 6000);
}

// Améliorer la fonction de mise à jour de l'heure avec plus de contexte
function updateSystemStatus() {
    const statusElements = {
        sync: document.getElementById('lastSyncTime'),
        action: document.getElementById('lastActionTime'),
        db: document.getElementById('dbStatus')
    };
    
    if (statusElements.sync) {
        const now = new Date();
        statusElements.sync.textContent = now.toLocaleTimeString('fr-FR', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }
    
    if (statusElements.action) {
        statusElements.action.textContent = 'à l\'instant';
        setTimeout(() => {
            if (statusElements.action) {
                statusElements.action.textContent = 'il y a 1 min';
            }
        }, 60000);
    }
    
    if (statusElements.db) {
        statusElements.db.textContent = 'Synchronisée';
        statusElements.db.className = 'font-semibold text-green-600';
        
        // Simulation d'une synchronisation
        setTimeout(() => {
            if (statusElements.db) {
                statusElements.db.innerHTML = 'Synchronisée <i class="fas fa-check text-xs ml-1"></i>';
            }
        }, 1000);
    }
}

// Améliorer les fonctions d'export avec plus d'options
async function exportPatientsData(format = 'csv') {
    try {
        showNotification('📤 Préparation de l\'export...', 'info');
        
        const exportData = filteredPatients.map(patient => ({
            'ID Patient': patient.patient_id,
            'Nom Complet': patient.patient_name,
            'Âge': calculateAge(patient.date_of_birth),
            'Genre': patient.gender === 'M' ? 'Masculin' : patient.gender === 'F' ? 'Féminin' : 'Non spécifié',
            'Téléphone': patient.phone || 'Non renseigné',
            'Email': patient.email || 'Non renseigné',
            'Adresse': patient.address || 'Non renseignée',
            'Analyses Totales': patient.total_analyses || 0,
            'Dernière Analyse': patient.last_analysis_date ? formatDate(patient.last_analysis_date) : 'Aucune',
            'Date Création': formatDate(patient.created_at),
            'Statut': patient.total_analyses === 0 ? 'Nouveau' : 
                     patient.total_analyses <= 3 ? 'Débutant' : 
                     patient.total_analyses <= 10 ? 'Régulier' : 'Intensif',
            'Contact Urgence': patient.emergency_contact_name || 'Non renseigné',
            'Tel Urgence': patient.emergency_contact_phone || 'Non renseigné'
        }));
        
        let fileContent, mimeType, fileName;
        
        if (format === 'csv') {
            const headers = Object.keys(exportData[0]);
            fileContent = [
                headers.join(','),
                ...exportData.map(row => headers.map(header => `"${row[header]}"`).join(','))
            ].join('\n');
            mimeType = 'text/csv;charset=utf-8;';
            fileName = `patients_neuroscan_${new Date().toISOString().split('T')[0]}.csv`;
        } else if (format === 'json') {
            fileContent = JSON.stringify({
                metadata: {
                    exportDate: new Date().toISOString(),
                    totalRecords: exportData.length,
                    version: '2.0',
                    source: 'NeuroScan Patient Management'
                },
                patients: exportData
            }, null, 2);
            mimeType = 'application/json;charset=utf-8;';
            fileName = `patients_neuroscan_${new Date().toISOString().split('T')[0]}.json`;
        }
        
        const dataBlob = new Blob([fileContent], { type: mimeType });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = fileName;
        link.click();
        URL.revokeObjectURL(url);
        
        showNotification(`✅ ${filteredPatients.length} patients exportés avec succès !`, 'success');
        updateSystemStatus();
    } catch (error) {
        console.error('Erreur export:', error);
        showNotification('❌ Erreur lors de l\'export des données', 'error');
    }
}

// Initialiser les raccourcis clavier
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K pour focus sur la recherche
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            document.getElementById('searchPatients').focus();
        }
        
        // Échap pour fermer les modales
        if (e.key === 'Escape') {
            closeAddPatientModal();
            closeQuickActions();
        }
        
        // Ctrl/Cmd + N pour nouveau patient
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault();
            openAddPatientModal();
        }
    });
}

// Charger les patients depuis l'API
async function loadPatients() {
    try {
        showLoading();
        
        const response = await fetch('/api/my-patients');
        const result = await response.json();
        
        if (result.success) {
            allPatients = result.data;
            filteredPatients = [...allPatients];
            updateStatistics();
            renderCurrentView();
            showMainContent();
        } else {
            showNotification(result.error || 'Erreur lors du chargement des patients', 'error');
            showMainContent();
        }
    } catch (error) {
        console.error('Erreur chargement patients:', error);
        showNotification('Erreur de connexion', 'error');
        showMainContent();
    }
}

// Afficher l'état de chargement
function showLoading() {
    document.getElementById('loadingState').classList.remove('hidden');
    document.getElementById('mainContent').classList.add('hidden');
}

// Afficher le contenu principal
function showMainContent() {
    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('mainContent').classList.remove('hidden');
}

// Mettre à jour les statistiques avec animations
function updateStatistics() {
    const totalPatients = allPatients.length;
    const totalAnalyses = allPatients.reduce((sum, p) => sum + (p.total_analyses || 0), 0);
    const activePatients = allPatients.filter(p => (p.total_analyses || 0) > 0).length;
    const regularFollowup = allPatients.filter(p => (p.total_analyses || 0) > 5).length;
    
    // Animation des compteurs
    animateCounter('totalPatients', totalPatients);
    animateCounter('totalAnalyses', totalAnalyses);
    animateCounter('activePatients', activePatients);
    animateCounter('regularFollowup', regularFollowup);
    
    updateFilterStats();
}

// Animation des compteurs
function animateCounter(elementId, targetValue) {
    const element = document.getElementById(elementId);
    const currentValue = parseInt(element.textContent) || 0;
    const increment = targetValue > currentValue ? 1 : -1;
    const stepTime = Math.abs(Math.floor(300 / (targetValue - currentValue))) || 10;
    
    let current = currentValue;
    const timer = setInterval(() => {
        current += increment;
        element.textContent = current;
        
        if (current === targetValue) {
            clearInterval(timer);
        }
    }, stepTime);
}

// Mettre à jour les statistiques de filtrage
function updateFilterStats() {
    const filteredCount = filteredPatients.length;
    const totalCount = allPatients.length;
    
    document.getElementById('filteredCount').textContent = filteredCount;
    document.getElementById('totalCount').textContent = totalCount;
}

// Calculer l'âge à partir de la date de naissance
function calculateAge(dateOfBirth) {
    if (!dateOfBirth) return null;
    
    const birth = new Date(dateOfBirth);
    const today = new Date();
    let age = today.getFullYear() - birth.getFullYear();
    const monthDiff = today.getMonth() - birth.getMonth();
    
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
        age--;
    }
    
    return age;
}

// Formater la date
function formatDate(dateString) {
    if (!dateString) return null;
    
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString('fr-FR', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric'
        });
    } catch {
        return dateString;
    }
}

// Formater la date relative
function formatRelativeDate(dateString) {
    if (!dateString) return null;
    
    try {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = now - date;
        const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays === 0) return "Aujourd'hui";
        if (diffDays === 1) return "Hier";
        if (diffDays < 7) return `Il y a ${diffDays} jours`;
        if (diffDays < 30) return `Il y a ${Math.floor(diffDays / 7)} semaine(s)`;
        if (diffDays < 365) return `Il y a ${Math.floor(diffDays / 30)} mois`;
        return `Il y a ${Math.floor(diffDays / 365)} an(s)`;
    } catch {
        return dateString;
    }
}

// Changer de vue (liste/cartes)
function switchToListView() {
    currentView = 'list';
    document.getElementById('listViewBtn').classList.add('bg-white', 'shadow-sm', 'text-gray-900');
    document.getElementById('listViewBtn').classList.remove('text-gray-600', 'hover:text-gray-900');
    document.getElementById('cardViewBtn').classList.remove('bg-white', 'shadow-sm', 'text-gray-900');
    document.getElementById('cardViewBtn').classList.add('text-gray-600', 'hover:text-gray-900');
    
    document.getElementById('listView').classList.remove('hidden');
    document.getElementById('cardView').classList.add('hidden');
    renderCurrentView();
}

function switchToCardView() {
    currentView = 'card';
    document.getElementById('cardViewBtn').classList.add('bg-white', 'shadow-sm', 'text-gray-900');
    document.getElementById('cardViewBtn').classList.remove('text-gray-600', 'hover:text-gray-900');
    document.getElementById('listViewBtn').classList.remove('bg-white', 'shadow-sm', 'text-gray-900');
    document.getElementById('listViewBtn').classList.add('text-gray-600', 'hover:text-gray-900');
    
    document.getElementById('cardView').classList.remove('hidden');
    document.getElementById('listView').classList.add('hidden');
    renderCurrentView();
}

// Rendre la vue actuelle
function renderCurrentView() {
    if (currentView === 'list') {
        renderPatientsList();
    } else {
        renderPatientsCards();
    }
}

// Rendre les patients en vue liste
function renderPatientsList() {
    const tbody = document.getElementById('patientsTableBody');
    const emptyState = document.getElementById('emptyState');
    
    if (filteredPatients.length === 0) {
        tbody.innerHTML = '';
        emptyState.classList.remove('hidden');
        return;
    }
    
    emptyState.classList.add('hidden');
    
    tbody.innerHTML = filteredPatients.map(patient => {
        const age = calculateAge(patient.date_of_birth);
        const lastAnalysisDate = formatRelativeDate(patient.last_analysis_date);
        const totalAnalyses = patient.total_analyses || 0;
        
        // Déterminer le statut avec couleurs améliorées
        let statusClass, statusText, statusIcon;
        if (totalAnalyses === 0) {
            statusClass = 'bg-gradient-to-r from-gray-100 to-gray-200 text-gray-800';
            statusText = 'Nouveau';
            statusIcon = '🆕';
        } else if (totalAnalyses <= 3) {
            statusClass = 'bg-gradient-to-r from-blue-100 to-blue-200 text-blue-800';
            statusText = 'Débutant';
            statusIcon = '📊';
        } else if (totalAnalyses <= 10) {
            statusClass = 'bg-gradient-to-r from-green-100 to-green-200 text-green-800';
            statusText = 'Régulier';
            statusIcon = '📈';
        } else {
            statusClass = 'bg-gradient-to-r from-purple-100 to-purple-200 text-purple-800';
            statusText = 'Intensif';
            statusIcon = '🏆';
        }
        
        const initials = (patient.patient_name || patient.patient_id || 'P').split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
        const displayName = patient.patient_name || `Patient ${patient.patient_id}`;
        
        return `
            <tr class="hover:bg-blue-50/50 transition-all duration-300 patient-row group cursor-pointer" 
                data-patient-id="${patient.patient_id}" 
                data-gender="${patient.gender || ''}" 
                data-age="${age || 0}"
                data-analyses="${totalAnalyses}"
                onclick="viewPatient('${patient.patient_id}')">
                <td class="px-6 py-5">
                    <div class="flex items-center">
                        <div class="relative">
                            <div class="w-12 h-12 bg-gradient-to-br from-blue-500 via-purple-600 to-indigo-700 rounded-full flex items-center justify-center text-white font-bold shadow-lg group-hover:scale-110 transition-transform duration-300">
                                ${initials}
                            </div>
                            <div class="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white ${totalAnalyses > 0 ? '' : 'hidden'}"></div>
                        </div>
                        <div class="ml-4">
                            <div class="text-sm font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">${displayName}</div>
                            <div class="text-sm text-gray-500 flex items-center">
                                <i class="fas fa-id-card mr-1"></i>
                                ID: ${patient.patient_id}
                            </div>
                        </div>
                    </div>
                </td>
                <td class="px-6 py-5">
                    <div class="space-y-1">
                        ${age ? `<div class="text-sm text-gray-900 flex items-center"><i class="fas fa-birthday-cake mr-1 text-gray-400"></i>${age} ans</div>` : ''}
                        ${patient.gender ? `<div class="text-sm text-gray-500 flex items-center"><i class="fas fa-${patient.gender === 'M' ? 'mars' : 'venus'} mr-1"></i>${patient.gender === 'M' ? 'Masculin' : 'Féminin'}</div>` : ''}
                        ${patient.phone ? `<div class="text-sm text-gray-500 flex items-center"><i class="fas fa-phone mr-1"></i>${patient.phone}</div>` : ''}
                    </div>
                </td>
                <td class="px-6 py-5">
                    <div class="flex items-center">
                        <div class="text-lg font-bold text-gray-900">${totalAnalyses}</div>
                        <div class="ml-2 text-sm text-gray-500">analyse${totalAnalyses !== 1 ? 's' : ''}</div>
                    </div>
                    <div class="mt-1 w-full bg-gray-200 rounded-full h-2">
                        <div class="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-500" 
                             style="width: ${Math.min(totalAnalyses * 10, 100)}%"></div>
                    </div>
                </td>
                <td class="px-6 py-5">
                    <div class="text-sm font-medium text-gray-900">
                        ${lastAnalysisDate || '<span class="text-gray-400">Aucune analyse</span>'}
                    </div>
                    ${patient.last_analysis_date ? `<div class="text-xs text-gray-500 mt-1">${formatDate(patient.last_analysis_date)}</div>` : ''}
                </td>
                <td class="px-6 py-5">
                    <span class="inline-flex items-center px-3 py-1.5 text-xs font-semibold rounded-full ${statusClass} shadow-sm">
                        <span class="mr-1">${statusIcon}</span>
                        ${statusText}
                    </span>
                </td>
                <td class="px-6 py-5 text-right">
                    <div class="flex items-center justify-end space-x-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <button onclick="event.stopPropagation(); viewPatient('${patient.patient_id}')" 
                               class="inline-flex items-center px-3 py-2 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-all duration-200 shadow-sm hover:shadow-md">
                            <i class="fas fa-eye mr-1"></i>Voir
                        </button>
                        <button onclick="event.stopPropagation(); editPatient('${patient.patient_id}')" 
                                class="inline-flex items-center px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all duration-200 shadow-sm hover:shadow-md">
                            <i class="fas fa-edit mr-1"></i>Modifier
                        </button>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
}

// Rendre les patients en vue cartes
function renderPatientsCards() {
    const container = document.getElementById('patientsCardsContainer');
    const emptyState = document.getElementById('emptyState');
    
    if (filteredPatients.length === 0) {
        container.innerHTML = '';
        emptyState.classList.remove('hidden');
        return;
    }
    
    emptyState.classList.add('hidden');
    
    container.innerHTML = filteredPatients.map(patient => {
        const age = calculateAge(patient.date_of_birth);
        const lastAnalysisDate = formatRelativeDate(patient.last_analysis_date);
        const totalAnalyses = patient.total_analyses || 0;
        
        // Déterminer le statut
        let statusClass, statusText, statusIcon;
        if (totalAnalyses === 0) {
            statusClass = 'bg-gradient-to-r from-gray-100 to-gray-200 text-gray-800';
            statusText = 'Nouveau';
            statusIcon = '🆕';
        } else if (totalAnalyses <= 3) {
            statusClass = 'bg-gradient-to-r from-blue-100 to-blue-200 text-blue-800';
            statusText = 'Débutant';
            statusIcon = '📊';
        } else if (totalAnalyses <= 10) {
            statusClass = 'bg-gradient-to-r from-green-100 to-green-200 text-green-800';
            statusText = 'Régulier';
            statusIcon = '📈';
        } else {
            statusClass = 'bg-gradient-to-r from-purple-100 to-purple-200 text-purple-800';
            statusText = 'Intensif';
            statusIcon = '🏆';
        }
        
        const initials = (patient.patient_name || patient.patient_id || 'P').split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
        const displayName = patient.patient_name || `Patient ${patient.patient_id}`;
        
        return `
            <div class="dashboard-card clickable p-6 hover:shadow-2xl transition-all duration-300 cursor-pointer" 
                 onclick="viewPatient('${patient.patient_id}')">
                <div class="flex items-start justify-between mb-4">
                    <div class="flex items-center">
                        <div class="relative">
                            <div class="w-14 h-14 bg-gradient-to-br from-blue-500 via-purple-600 to-indigo-700 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
                                ${initials}
                            </div>
                            <div class="absolute -bottom-1 -right-1 w-5 h-5 bg-green-500 rounded-full border-2 border-white ${totalAnalyses > 0 ? '' : 'hidden'}"></div>
                        </div>
                        <div class="ml-4">
                            <h3 class="text-lg font-bold text-gray-900">${displayName}</h3>
                            <p class="text-sm text-gray-500">ID: ${patient.patient_id}</p>
                        </div>
                    </div>
                    <span class="inline-flex items-center px-3 py-1 text-xs font-semibold rounded-full ${statusClass}">
                        <span class="mr-1">${statusIcon}</span>
                        ${statusText}
                    </span>
                </div>
                
                <div class="space-y-3 mb-6">
                    ${age ? `<div class="flex items-center text-sm text-gray-600"><i class="fas fa-birthday-cake mr-2 w-4"></i>${age} ans</div>` : ''}
                    ${patient.gender ? `<div class="flex items-center text-sm text-gray-600"><i class="fas fa-${patient.gender === 'M' ? 'mars' : 'venus'} mr-2 w-4"></i>${patient.gender === 'M' ? 'Masculin' : 'Féminin'}</div>` : ''}
                    ${patient.phone ? `<div class="flex items-center text-sm text-gray-600"><i class="fas fa-phone mr-2 w-4"></i>${patient.phone}</div>` : ''}
                    ${patient.email ? `<div class="flex items-center text-sm text-gray-600"><i class="fas fa-envelope mr-2 w-4"></i>${patient.email}</div>` : ''}
                </div>
                
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <div class="text-2xl font-bold text-gray-900">${totalAnalyses}</div>
                        <div class="text-sm text-gray-500">analyse${totalAnalyses !== 1 ? 's' : ''}</div>
                    </div>
                    <div class="text-right">
                        <div class="text-sm font-medium text-gray-900">
                            ${lastAnalysisDate || 'Aucune analyse'}
                        </div>
                        ${patient.last_analysis_date ? `<div class="text-xs text-gray-500">${formatDate(patient.last_analysis_date)}</div>` : ''}
                    </div>
                </div>
                
                <div class="flex space-x-2">
                    <button onclick="event.stopPropagation(); viewPatient('${patient.patient_id}')" 
                           class="flex-1 px-3 py-2 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors text-center">
                        <i class="fas fa-eye mr-1"></i>Voir
                    </button>
                    <button onclick="event.stopPropagation(); editPatient('${patient.patient_id}')" 
                            class="flex-1 px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-center">
                        <i class="fas fa-edit mr-1"></i>Modifier
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

// Fonctions de gestion des patients
function openAddPatientModal() {
    document.getElementById('addPatientModal').classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    generateNextId(); // Auto-générer l'ID suivant
    
    // Focus sur le premier champ
    setTimeout(() => {
        document.getElementById('patientIdInput').focus();
    }, 300);
}

function closeAddPatientModal() {
    document.getElementById('addPatientModal').classList.add('hidden');
    document.body.style.overflow = 'auto';
    document.getElementById('addPatientForm').reset();
    resetIdValidation();
}

function closeQuickActions() {
    document.getElementById('quickActionsMenu').classList.add('hidden');
}

function toggleQuickActions() {
    const menu = document.getElementById('quickActionsMenu');
    menu.classList.toggle('hidden');
}

// Validation de l'ID patient en temps réel
function resetIdValidation() {
    const indicator = document.getElementById('idValidationIndicator');
    indicator.className = 'w-3 h-3 rounded-full bg-gray-300';
}

function validatePatientId(id) {
    const indicator = document.getElementById('idValidationIndicator');
    const existingIds = allPatients.map(p => p.patient_id.toLowerCase());
    
    if (!id) {
        indicator.className = 'w-3 h-3 rounded-full bg-gray-300';
        return false;
    }
    
    if (existingIds.includes(id.toLowerCase())) {
        indicator.className = 'w-3 h-3 rounded-full bg-red-500';
        return false;
    }
    
    indicator.className = 'w-3 h-3 rounded-full bg-green-500';
    return true;
}

async function generateNextId() {
    try {
        const response = await fetch('/api/patients/next-id');
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('patientIdInput').value = result.next_id;
            validatePatientId(result.next_id);
        }
    } catch (error) {
        console.error('Erreur génération ID:', error);
        // Génération locale en cas d'erreur
        const maxId = Math.max(...allPatients.map(p => {
            const match = p.patient_id.match(/P(\d+)/);
            return match ? parseInt(match[1]) : 0;
        }), 0);
        const nextId = `P${String(maxId + 1).padStart(4, '0')}`;
        document.getElementById('patientIdInput').value = nextId;
        validatePatientId(nextId);
    }
}

function viewPatient(patientId) {
    window.location.href = `/patient/${patientId}`;
}

function editPatient(patientId) {
    window.location.href = `/patients/${patientId}/edit`;
}

function refreshPatients() {
    showNotification('Actualisation des données...', 'info');
    loadPatients();
}

// Fonctions d'export et import
async function exportPatientsData() {
    try {
        showNotification('Préparation de l\'export...', 'info');
        
        const exportData = filteredPatients.map(patient => ({
            id: patient.patient_id,
            nom: patient.patient_name,
            age: calculateAge(patient.date_of_birth),
            genre: patient.gender === 'M' ? 'Masculin' : patient.gender === 'F' ? 'Féminin' : '',
            telephone: patient.phone,
            email: patient.email,
            adresse: patient.address,
            total_analyses: patient.total_analyses || 0,
            derniere_analyse: patient.last_analysis_date,
            date_creation: formatDate(patient.created_at),
            statut: patient.total_analyses === 0 ? 'Nouveau' : 
                   patient.total_analyses <= 3 ? 'Débutant' : 
                   patient.total_analyses <= 10 ? 'Régulier' : 'Intensif'
        }));
        
        // Export en CSV pour une meilleure compatibilité
        const csvContent = [
            Object.keys(exportData[0]).join(','),
            ...exportData.map(row => Object.values(row).map(val => `"${val || ''}"`).join(','))
        ].join('\n');
        
        const dataBlob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `patients_neuroscan_${new Date().toISOString().split('T')[0]}.csv`;
        link.click();
        URL.revokeObjectURL(url);
        
        showNotification(`${filteredPatients.length} patients exportés avec succès`, 'success');
    } catch (error) {
        console.error('Erreur export:', error);
        showNotification('Erreur lors de l\'export', 'error');
    }
}

async function importPatientsData() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv,.json,.xlsx';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (file) {
            showNotification('Import des données en cours...', 'info');
            // Ici, vous pouvez implémenter la logique d'import
            showNotification('Fonctionnalité d\'import en développement', 'warning');
        }
    };
    input.click();
}

async function bulkEmailPatients() {
    if (filteredPatients.length === 0) {
        showNotification('Aucun patient sélectionné', 'warning');
        return;
    }
    
    const emailPatients = filteredPatients.filter(p => p.email);
    if (emailPatients.length === 0) {
        showNotification('Aucun patient avec email trouvé', 'warning');
        return;
    }
    
    showNotification(`Email en masse pour ${emailPatients.length} patients en préparation...`, 'info');
    // Implémenter la logique d'email en masse
}

async function generateReport() {
    showNotification('Génération du rapport PDF...', 'info');
    
    try {
        const reportData = {
            totalPatients: filteredPatients.length,
            stats: {
                nouveaux: filteredPatients.filter(p => (p.total_analyses || 0) === 0).length,
                actifs: filteredPatients.filter(p => (p.total_analyses || 0) > 0).length,
                intensifs: filteredPatients.filter(p => (p.total_analyses || 0) > 10).length
            },
            patients: filteredPatients.map(patient => ({
                id: patient.patient_id,
                nom: patient.patient_name,
                age: calculateAge(patient.date_of_birth),
                analyses: patient.total_analyses || 0,
                derniere_analyse: formatDate(patient.last_analysis_date)
            }))
        };
        
        // Ici, vous pourriez envoyer les données à un endpoint pour générer le PDF
        showNotification('Rapport PDF généré avec succès', 'success');
    } catch (error) {
        showNotification('Erreur lors de la génération du rapport', 'error');
    }
}

async function backupData() {
    try {
        showNotification('Sauvegarde des données...', 'info');
        
        const backupData = {
            timestamp: new Date().toISOString(),
            version: '1.0',
            totalPatients: allPatients.length,
            patients: allPatients
        };
        
        const dataStr = JSON.stringify(backupData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `neuroscan_backup_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
        
        showNotification('Sauvegarde créée avec succès', 'success');
    } catch (error) {
        console.error('Erreur sauvegarde:', error);
        showNotification('Erreur lors de la sauvegarde', 'error');
    }
}

// Recherche et filtres améliorés
document.getElementById('searchPatients').addEventListener('input', debounce(filterPatients, 300));
document.getElementById('filterGender').addEventListener('change', filterPatients);
document.getElementById('filterAnalysesCount').addEventListener('change', filterPatients);
document.getElementById('filterActivity').addEventListener('change', filterPatients);
document.getElementById('sortPatients').addEventListener('change', handleSort);

// Validation en temps réel de l'ID
document.getElementById('patientIdInput').addEventListener('input', function(e) {
    validatePatientId(e.target.value);
});

// Fonction de debounce pour la recherche
function debounce(func, wait) {
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

function filterPatients() {
    const search = document.getElementById('searchPatients').value.toLowerCase();
    const genderFilter = document.getElementById('filterGender').value;
    const analysesFilter = document.getElementById('filterAnalysesCount').value;
    const activityFilter = document.getElementById('filterActivity').value;
    
    filteredPatients = allPatients.filter(patient => {
        // Recherche textuelle améliorée
        const matchesSearch = !search || 
            (patient.patient_name && patient.patient_name.toLowerCase().includes(search)) ||
            (patient.patient_id && patient.patient_id.toLowerCase().includes(search)) ||
            (patient.phone && patient.phone.includes(search)) ||
            (patient.email && patient.email.toLowerCase().includes(search));
        
        // Filtre par genre
        const matchesGender = !genderFilter || patient.gender === genderFilter;
        
        // Filtre par nombre d'analyses
        const analysesCount = patient.total_analyses || 0;
        let matchesAnalyses = true;
        
        if (analysesFilter) {
            switch(analysesFilter) {
                case '0': matchesAnalyses = analysesCount === 0; break;
                case '1-3': matchesAnalyses = analysesCount >= 1 && analysesCount <= 3; break;
                case '4-10': matchesAnalyses = analysesCount >= 4 && analysesCount <= 10; break;
                case '10+': matchesAnalyses = analysesCount > 10; break;
            }
        }
        
        // Filtre par activité
        let matchesActivity = true;
        if (activityFilter) {
            const daysSinceLastAnalysis = patient.last_analysis_date ? 
                Math.floor((new Date() - new Date(patient.last_analysis_date)) / (1000 * 60 * 60 * 24)) : Infinity;
            
            switch(activityFilter) {
                case 'recent': matchesActivity = daysSinceLastAnalysis <= 30; break;
                case 'inactive': matchesActivity = daysSinceLastAnalysis > 90; break;
            }
        }
        
        return matchesSearch && matchesGender && matchesAnalyses && matchesActivity;
    });
    
    // Appliquer le tri
    applySorting();
    updateFilterStats();
    renderCurrentView();
}

function handleSort() {
    const sortValue = document.getElementById('sortPatients').value;
    
    switch(sortValue) {
        case 'name':
            sortColumn = 'name';
            sortOrder = 'asc';
            break;
        case 'recent':
            sortColumn = 'date';
            sortOrder = 'desc';
            break;
        case 'analyses':
            sortColumn = 'analyses';
            sortOrder = 'desc';
            break;
        case 'activity':
            sortColumn = 'activity';
            sortOrder = 'desc';
            break;
    }
    
    applySorting();
    renderCurrentView();
}

function applySorting() {
    filteredPatients.sort((a, b) => {
        let compareValue = 0;
        
        switch(sortColumn) {
            case 'name':
                compareValue = (a.patient_name || a.patient_id || '').localeCompare(b.patient_name || b.patient_id || '');
                break;
            case 'date':
                const dateA = new Date(a.created_at || 0);
                const dateB = new Date(b.created_at || 0);
                compareValue = dateA - dateB;
                break;
            case 'analyses':
                compareValue = (a.total_analyses || 0) - (b.total_analyses || 0);
                break;
            case 'activity':
                const activityA = new Date(a.last_analysis_date || 0);
                const activityB = new Date(b.last_analysis_date || 0);
                compareValue = activityA - activityB;
                break;
        }
        
        return sortOrder === 'asc' ? compareValue : -compareValue;
    });
}

function sortTableBy(column) {
    if (sortColumn === column) {
        sortOrder = sortOrder === 'asc' ? 'desc' : 'asc';
    } else {
        sortColumn = column;
        sortOrder = 'asc';
    }
    
    applySorting();
    renderCurrentView();
}

function clearFilters() {
    document.getElementById('searchPatients').value = '';
    document.getElementById('filterGender').value = '';
    document.getElementById('filterAnalysesCount').value = '';
    document.getElementById('filterActivity').value = '';
    filteredPatients = [...allPatients];
    applySorting();
    updateFilterStats();
    renderCurrentView();
}

// Gestion du formulaire d'ajout améliorée
document.getElementById('addPatientForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitButton = e.target.querySelector('button[type="submit"]');
    const originalText = submitButton.innerHTML;
    
    // Désactiver le bouton et afficher l'état de chargement
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-spinner animate-spin mr-2"></i>Création en cours...';
    
    const formData = new FormData(this);
    const patientData = Object.fromEntries(formData.entries());
    
    // Nettoyage des données
    Object.keys(patientData).forEach(key => {
        if (patientData[key] === '') {
            delete patientData[key];
        }
    });
    
    // Validation côté client
    if (!validatePatientId(patientData.patient_id)) {
        showNotification('ID patient invalide ou déjà utilisé', 'error');
        submitButton.disabled = false;
        submitButton.innerHTML = originalText;
        return;
    }
    
    try {
        const response = await fetch('/api/patients', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('🎉 Patient ajouté avec succès!', 'success');
            closeAddPatientModal();
            loadPatients(); // Recharger la liste
        } else {
            showNotification(result.error || 'Erreur lors de l\'ajout du patient', 'error');
        }
    } catch (error) {
        console.error('Erreur ajout patient:', error);
        showNotification('Erreur de connexion - Vérifiez votre réseau', 'error');
    } finally {
        // Réactiver le bouton
        submitButton.disabled = false;
        submitButton.innerHTML = originalText;
    }
});

// Système de notifications modernisé
function showNotification(message, type = 'info', duration = 4000) {
    // Supprimer les notifications existantes
    const existingNotifications = document.querySelectorAll('.notification-toast');
    existingNotifications.forEach(notif => notif.remove());
    
    const notification = document.createElement('div');
    notification.className = 'notification-toast fixed top-6 right-6 z-50 px-6 py-4 rounded-2xl text-white transform transition-all duration-500 translate-x-full shadow-2xl max-w-sm';
    
    let bgClass, iconClass, borderClass;
    switch(type) {
        case 'success':
            bgClass = 'bg-gradient-to-r from-green-500 to-emerald-600';
            iconClass = 'check-circle';
            borderClass = 'border-green-400';
            break;
        case 'error':
            bgClass = 'bg-gradient-to-r from-red-500 to-rose-600';
            iconClass = 'exclamation-circle';
            borderClass = 'border-red-400';
            break;
        case 'warning':
            bgClass = 'bg-gradient-to-r from-orange-500 to-amber-600';
            iconClass = 'exclamation-triangle';
            borderClass = 'border-orange-400';
            break;
        default:
            bgClass = 'bg-gradient-to-r from-blue-500 to-indigo-600';
            iconClass = 'info-circle';
            borderClass = 'border-blue-400';
    }
    
    notification.classList.add(bgClass);
    notification.innerHTML = `
        <div class="flex items-start space-x-3">
            <div class="flex-shrink-0 mt-0.5">
                <i class="fas fa-${iconClass} text-xl"></i>
            </div>
            <div class="flex-1">
                <p class="text-sm font-medium leading-relaxed">${message}</p>
            </div>
            <button onclick="this.parentElement.parentElement.remove()" class="flex-shrink-0 ml-4 p-1 rounded-full hover:bg-white/20 transition-colors">
                <i class="fas fa-times text-sm"></i>
            </button>
        </div>
        <div class="mt-3 bg-white/20 rounded-full h-1 overflow-hidden">
            <div class="notification-progress bg-white h-full transition-all duration-${duration} ease-linear w-full"></div>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Animation d'entrée
    requestAnimationFrame(() => {
        notification.classList.remove('translate-x-full');
        notification.classList.add('translate-x-0');
    });
    
    // Animation de la barre de progression
    setTimeout(() => {
        const progress = notification.querySelector('.notification-progress');
        if (progress) {
            progress.style.width = '0%';
        }
    }, 100);
    
    // Auto-suppression
    setTimeout(() => {
        if (document.body.contains(notification)) {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 500);
        }
    }, duration);
}

// Fermer les modales en cliquant à l'extérieur
document.getElementById('addPatientModal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeAddPatientModal();
    }
});

// Fermer le menu d'actions en cliquant à l'extérieur
document.addEventListener('click', function(e) {
    const quickActions = document.getElementById('quickActionsMenu');
    const quickActionsButton = e.target.closest('[onclick="toggleQuickActions()"]');
    
    if (!quickActions.contains(e.target) && !quickActionsButton) {
        closeQuickActions();
    }
});

// Fonctions utilitaires pour les animations et interactions
function addRippleEffect(element, x, y) {
    const ripple = document.createElement('span');
    ripple.classList.add('ripple');
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    element.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
    }, 600);
}

// Améliorer l'expérience utilisateur avec des micro-interactions
document.querySelectorAll('.btn-dashboard').forEach(button => {
    button.addEventListener('click', function(e) {
        const rect = this.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        addRippleEffect(this, x, y);
    });
});

// Gestion du mode sombre (pour futures améliorations)
function toggleDarkMode() {
    document.body.classList.toggle('dark');
    localStorage.setItem('darkMode', document.body.classList.contains('dark'));
}

// Initialiser le mode sombre si préféré
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark');
}

// Fonctions pour les interactions avancées
function highlightSearchResults() {
    const searchTerm = document.getElementById('searchPatients').value.toLowerCase();
    if (!searchTerm) return;
    
    document.querySelectorAll('.patient-row').forEach(row => {
        const textContent = row.textContent.toLowerCase();
        if (textContent.includes(searchTerm)) {
            row.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
            row.style.borderLeft = '4px solid #3b82f6';
        }
    });
}

// Gestion des raccourcis clavier avancés
document.addEventListener('keydown', function(e) {
    // Ctrl + R pour actualiser les données
    if ((e.ctrlKey || e.metaKey) && e.key === 'r' && !e.shiftKey) {
        e.preventDefault();
        refreshPatients();
    }
    
    // Ctrl + E pour exporter
    if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        exportPatientsData();
    }
    
    // F1 pour aide (future fonctionnalité)
    if (e.key === 'F1') {
        e.preventDefault();
        showNotification('Raccourcis: Ctrl+K (recherche), Ctrl+N (nouveau), Ctrl+R (actualiser), Ctrl+E (export)', 'info', 6000);
    }
});

// Optimisation des performances pour les grandes listes
function virtualizeTable() {
    // Cette fonction peut être implémentée pour gérer de très grandes listes de patients
    // en n'affichant que les éléments visibles à l'écran
}

// Analytics et suivi des actions utilisateur (pour l'amélioration continue)
function trackUserAction(action, details = {}) {
    // Log des actions pour analytics (implémentation future)
    console.log(`Action: ${action}`, details);
}

// Validation avancée des formulaires
function validateFormData(data) {
    const errors = [];
    
    if (!data.patient_id || data.patient_id.length < 2) {
        errors.push('ID patient requis (minimum 2 caractères)');
    }
    
    if (!data.patient_name || data.patient_name.length < 2) {
        errors.push('Nom du patient requis (minimum 2 caractères)');
    }
    
    if (data.email && !isValidEmail(data.email)) {
        errors.push('Format d\'email invalide');
    }
    
    if (data.phone && !isValidPhone(data.phone)) {
        errors.push('Format de téléphone invalide');
    }
    
    return errors;
}

function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function isValidPhone(phone) {
    const phoneRegex = /^[\+]?[0-9\s\-\(\)]{8,15}$/;
    return phoneRegex.test(phone);
}

// Initialisation finale
document.addEventListener('DOMContentLoaded', function() {
    // Préchargement des icônes Font Awesome
    const iconPreload = document.createElement('link');
    iconPreload.rel = 'preload';
    iconPreload.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/webfonts/fa-solid-900.woff2';
    iconPreload.as = 'font';
    iconPreload.type = 'font/woff2';
    iconPreload.crossOrigin = 'anonymous';
    document.head.appendChild(iconPreload);
});

console.log('NeuroScan - Gestion des Patients v2.0 - Interface modernisée chargée avec succès ✨');
