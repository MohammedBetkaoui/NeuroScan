/**
 * NeuroScan Pro - Advanced Dashboard JavaScript
 * Gestion complète du tableau de bord avancé avec toutes les fonctionnalités dynamiques
 */

// Variables globales pour les graphiques
let comparisonChart = null;
let performanceChart = null;
let diagnosticDistributionChart = null;
let yearComparisonChart = null;
let hourlyHeatmapChart = null;
let confidenceHistogramChart = null;
let processingTimeChart = null;
let monthlyTrendsChart = null;

// Variables globales pour les données
let currentFilters = {};
let dashboardData = null;
let autoRefreshInterval = null;

// Configuration Chart.js par défaut
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = '#6b7280';
Chart.defaults.plugins.legend.display = true;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(17, 24, 39, 0.95)';
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;

/**
 * ========================================
 * INITIALISATION
 * ========================================
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initialisation du dashboard avancé...');
    
    initializeUI();
    initializeFilters();
    initializeChartControls();
    loadAllData();
    
    // Auto-refresh toutes les 30 secondes
    autoRefreshInterval = setInterval(() => {
        console.log('🔄 Actualisation automatique...');
        loadAllData();
    }, 30000);
    
    // Animation d'entrée
    animateElements();
    
    console.log('✅ Dashboard initialisé avec succès');
    showNotification('Tableau de bord chargé avec succès', 'success');
});

/**
 * ========================================
 * CHARGEMENT DES DONNÉES
 * ========================================
 */
async function loadAllData() {
    try {
        console.log('📊 Chargement de toutes les données...');
        
        // Charger toutes les données en parallèle
        await Promise.all([
            loadDashboardOverview(),
            loadAlerts(),
            loadComparisons(),
            loadPerformanceTrends(),
            loadAdvancedAnalytics(),
            loadAIInsights(),
            loadAdvancedMetrics()
        ]);
        
        console.log('✅ Toutes les données chargées');
    } catch (error) {
        console.error('❌ Erreur lors du chargement des données:', error);
        showNotification('Erreur de chargement des données', 'error');
    }
}

async function loadDashboardOverview() {
    try {
        const response = await fetch('/api/analytics/overview');
        const data = await response.json();
        
        console.log('📊 Données Overview reçues:', data);
        
        if (data.success) {
            dashboardData = data.data;
            updateOverviewStats(data.data);
            updateFilterCounts(data.data);
        }
    } catch (error) {
        console.error('Erreur chargement overview:', error);
    }
}

async function loadAlerts() {
    try {
        const response = await fetch('/api/analytics/alerts');
        const data = await response.json();
        
        if (data.success) {
            updateAlertsSection(data.data.alerts);
            updateNavbarAlerts(data.data.alerts);
        }
    } catch (error) {
        console.error('Erreur chargement alertes:', error);
    }
}

async function loadComparisons() {
    try {
        const response = await fetch('/api/analytics/comparison');
        const data = await response.json();

        console.log('📊 Données Comparison reçues:', data);

        if (data.success) {
            updateComparisonChart(data.data);
            // Mettre à jour aussi le graphique de Comparaison Annuelle
            updateYearComparison(data.data);
        }
    } catch (error) {
        console.error('Erreur chargement comparaisons:', error);
    }
}

async function loadPerformanceTrends() {
    try {
        const response = await fetch('/api/analytics/performance');
        const data = await response.json();

        console.log('📊 Données Performance reçues:', data);

        if (data && data.success && data.data) {
            updatePerformanceChart(data.data);
            updatePerformanceMetrics(data.data);
        } else {
            console.warn('⚠️ Données de performance non disponibles');
            updatePerformanceMetrics({}); // Utiliser les valeurs par défaut
        }
    } catch (error) {
        console.error('Erreur chargement performance:', error);
        updatePerformanceMetrics({}); // Utiliser les valeurs par défaut en cas d'erreur
    }
}

async function loadAdvancedAnalytics() {
    try {
        const [diagnostic, hourly, confidence, processing, monthly] = await Promise.all([
            fetch('/api/analytics/diagnostic-distribution'),
            fetch('/api/analytics/hourly-activity'),
            fetch('/api/analytics/confidence-distribution'),
            fetch('/api/analytics/processing-time-analysis'),
            fetch('/api/analytics/monthly-trends')
        ]);

        const diagnosticData = await diagnostic.json();
        const hourlyData = await hourly.json();
        const confidenceData = await confidence.json();
        const processingData = await processing.json();
        const monthlyData = await monthly.json();

        console.log('📊 Données Diagnostic reçues:', diagnosticData);
        console.log('📊 Données Hourly reçues:', hourlyData);
        console.log('📊 Données Confidence reçues:', confidenceData);
        console.log('📊 Données Processing reçues:', processingData);
        console.log('📊 Données Monthly reçues:', monthlyData);

        if (diagnosticData && diagnosticData.success && diagnosticData.data) {
            updateDiagnosticDistribution(diagnosticData.data);
        } else {
            console.warn('⚠️ Données de distribution diagnostique non disponibles');
        }
        
        if (hourlyData && hourlyData.success && hourlyData.data) {
            updateHourlyActivity(hourlyData.data);
        } else {
            console.warn('⚠️ Données d\'activité horaire non disponibles');
        }
        
        if (confidenceData && confidenceData.success && confidenceData.data) {
            updateConfidenceDistribution(confidenceData.data);
        } else {
            console.warn('⚠️ Données de confiance non disponibles');
        }
        
        if (processingData && processingData.success && processingData.data) {
            updateProcessingTimeAnalysis(processingData.data);
        } else {
            console.warn('⚠️ Données de temps de traitement non disponibles');
        }
        
        if (monthlyData && monthlyData.success && monthlyData.data) {
            updateMonthlyTrends(monthlyData.data);
        } else {
            console.warn('⚠️ Données de tendances mensuelles non disponibles');
        }

    } catch (error) {
        console.error('Erreur chargement analytics avancées:', error);
    }
}

async function loadAIInsights() {
    try {
        const response = await fetch('/api/analytics/ai-insights');
        const data = await response.json();

        if (data.success) {
            updateAIInsights(data.data);
        }
    } catch (error) {
        console.error('Erreur chargement AI insights:', error);
    }
}

async function loadAdvancedMetrics() {
    try {
        const response = await fetch('/api/analytics/advanced-metrics');
        const data = await response.json();

        if (data.success) {
            updateAdvancedMetrics(data.data);
        }
    } catch (error) {
        console.error('Erreur chargement métriques avancées:', error);
    }
}

/**
 * ========================================
 * MISE À JOUR DE L'INTERFACE
 * ========================================
 */
function updateOverviewStats(data) {
    // Total Analyses
    document.getElementById('totalAnalyses').textContent = formatNumber(data.total_analyses || 0);
    
    // Confiance Moyenne
    const avgConf = (data.avg_confidence || 0).toFixed(1);
    document.getElementById('avgConfidence').textContent = avgConf + '%';
    
    // Analyses Aujourd'hui
    const today = new Date().toISOString().split('T')[0];
    const todayData = data.daily_analyses?.find(d => d[0] === today);
    document.getElementById('todayAnalyses').textContent = formatNumber(todayData ? todayData[1] : 0);
    
    // Temps Moyen
    const avgTime = (data.avg_processing_time || 0).toFixed(2);
    document.getElementById('avgProcessingTime').textContent = avgTime + 's';
    
    // Changements (calcul des tendances)
    if (data.daily_analyses && data.daily_analyses.length > 1) {
        const lastTwo = data.daily_analyses.slice(-2);
        if (lastTwo.length === 2) {
            const change = ((lastTwo[1][1] - lastTwo[0][1]) / lastTwo[0][1] * 100).toFixed(1);
            updateChangeIndicators(change);
        }
    }
}

function updateChangeIndicators(change) {
    const changeValue = parseFloat(change);
    const icon = changeValue >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
    const color = changeValue >= 0 ? 'text-green-200' : 'text-red-200';
    
    // Mise à jour des indicateurs de changement
    ['confidenceChange', 'todayChange', 'timeChange'].forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = `<i class="fas ${icon} mr-1"></i>${Math.abs(changeValue)}%`;
            element.className = `flex items-center ${color} text-xs`;
        }
    });
}

function updateAlertsSection(alerts) {
    const alertsSection = document.getElementById('alertsSection');
    
    if (!alerts || alerts.length === 0) {
        alertsSection.innerHTML = '';
        return;
    }

    const alertsHTML = alerts.map(alert => `
        <div class="alert-${alert.type} text-white p-4 rounded-xl mb-4 flex items-center shadow-lg fade-in-up">
            <div class="w-12 h-12 bg-white bg-opacity-20 rounded-full flex items-center justify-center mr-4">
                <i class="fas ${getAlertIcon(alert.type)} text-2xl"></i>
            </div>
            <div class="flex-1">
                <h4 class="font-bold text-lg mb-1">${alert.title}</h4>
                <p class="text-sm opacity-90">${alert.message}</p>
            </div>
            <button onclick="dismissAlert(this)" class="ml-4 hover:bg-white hover:bg-opacity-20 p-2 rounded-full transition-colors">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `).join('');

    alertsSection.innerHTML = alertsHTML;
}

function updateNavbarAlerts(alerts) {
    const alertsList = document.getElementById('alertsList');
    const alertsBadge = document.getElementById('alertsBadge');
    
    // Mettre à jour le badge
    if (alerts && alerts.length > 0) {
        alertsBadge.classList.remove('hidden');
        alertsBadge.textContent = alerts.length;
    } else {
        alertsBadge.classList.add('hidden');
    }

    // Mettre à jour la liste
    if (!alertsList) return;
    
    alertsList.innerHTML = '';
    
    if (!alerts || alerts.length === 0) {
        alertsList.innerHTML = `
            <div class="p-8 text-center text-gray-500">
                <i class="fas fa-check-circle text-4xl text-green-500 mb-3"></i>
                <p class="font-medium">Aucune alerte</p>
                <p class="text-sm">Tout fonctionne normalement</p>
            </div>
        `;
        return;
    }

    alerts.forEach(alert => {
        const alertElement = document.createElement('div');
        alertElement.className = 'p-4 border-b border-gray-100 hover:bg-gray-50 transition-colors cursor-pointer';
        
        alertElement.innerHTML = `
            <div class="flex items-start">
                <div class="w-10 h-10 bg-${getAlertColor(alert.type)}-100 rounded-full flex items-center justify-center mr-3">
                    <i class="fas ${getAlertIcon(alert.type)} text-${getAlertColor(alert.type)}-600"></i>
                </div>
                <div class="flex-1">
                    <h4 class="font-semibold text-gray-900 text-sm">${alert.title}</h4>
                    <p class="text-xs text-gray-600 mt-1">${alert.message}</p>
                    <p class="text-xs text-gray-400 mt-1"><i class="fas fa-clock mr-1"></i>${formatDate(alert.timestamp)}</p>
                </div>
            </div>
        `;
        
        alertsList.appendChild(alertElement);
    });
}

function updatePerformanceMetrics(data) {
    // Vérification robuste des données avec plusieurs niveaux de sécurité
    if (data && data.daily_trends && data.daily_trends.data && Array.isArray(data.daily_trends.data) && data.daily_trends.data.length > 0) {
        const values = data.daily_trends.data;
        const avg = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);
        const peak = Math.max(...values).toFixed(2);
        
        const avgElement = document.getElementById('avgTime');
        const peakElement = document.getElementById('peakTime');
        
        if (avgElement) avgElement.textContent = avg + 's';
        if (peakElement) peakElement.textContent = peak + 's';
    } else {
        // Valeurs par défaut si les données sont absentes
        const avgElement = document.getElementById('avgTime');
        const peakElement = document.getElementById('peakTime');
        
        if (avgElement) avgElement.textContent = '--';
        if (peakElement) peakElement.textContent = '--';
    }
}

function updateFilterCounts(data) {
    if (!data.diagnostic_distribution) return;
    
    const counts = data.diagnostic_distribution;
    ['Normal', 'Gliome', 'Méningiome', 'Tumeur pituitaire'].forEach(type => {
        const element = document.getElementById(`count-${type}`);
        if (element && counts[type]) {
            element.textContent = counts[type];
        }
    });
}

/**
 * ========================================
 * GRAPHIQUES
 * ========================================
 */
function updateComparisonChart(data) {
    const ctx = document.getElementById('comparisonChart')?.getContext('2d');
    if (!ctx) return;
    
    console.log('🔍 updateComparisonChart - données reçues:', data);
    console.log('🔍 data.monthly:', data.monthly);
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }

    const labels = ['Ce mois', 'Mois dernier'];
    
    // L'API peut renvoyer différents formats
    let thisMonth = 0;
    let lastMonth = 0;
    
    if (data.monthly) {
        console.log('📊 Structure monthly:', Object.keys(data.monthly));
        // Format: {monthly: {'Ce mois': {count: N}, 'Mois dernier': {count: N}}}
        thisMonth = data.monthly['Ce mois']?.count || data.monthly['Ce mois'] || 0;
        lastMonth = data.monthly['Mois dernier']?.count || data.monthly['Mois dernier'] || 0;
        
        console.log('📊 Extraction depuis monthly:', {
            'Ce mois': data.monthly['Ce mois'],
            'Mois dernier': data.monthly['Mois dernier'],
            thisMonth,
            lastMonth
        });
    } else if (data.thisMonth !== undefined && data.lastMonth !== undefined) {
        // Format direct: {thisMonth: N, lastMonth: N}
        thisMonth = data.thisMonth || 0;
        lastMonth = data.lastMonth || 0;
    }

    console.log('📊 Valeurs FINALES pour comparisonChart:', { thisMonth, lastMonth });

    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Analyses',
                data: [thisMonth, lastMonth],
                backgroundColor: [
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(139, 92, 246, 0.8)'
                ],
                borderColor: [
                    'rgba(99, 102, 241, 1)',
                    'rgba(139, 92, 246, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            // Protection contre context.parsed undefined
                            if (!context || !context.parsed || context.parsed.y === undefined) {
                                return context.dataset?.label || 'Données';
                            }
                            return context.dataset.label + ': ' + formatNumber(context.parsed.y) + ' analyses';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function updatePerformanceChart(data) {
    const ctx = document.getElementById('performanceChart')?.getContext('2d');
    if (!ctx) return;

    console.log('🔍 updatePerformanceChart - données reçues:', data);

    if (performanceChart) {
        performanceChart.destroy();
    }

    // L'API renvoie {daily_trends: {labels, data, count, confidence, processing_time}, hourly_trends: {}, performance_by_type: {}}
    const trendsData = data.daily_trends || data.hourly_trends || {};
    const chartLabels = trendsData.labels || [];
    const chartData = trendsData.count || trendsData.data || [];
    
    console.log('📊 Données pour performanceChart:', { chartLabels, chartData, trendsData });

    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Analyses',
                data: chartData,
                borderColor: 'rgba(99, 102, 241, 1)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 4,
                pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function updateDiagnosticDistribution(data) {
    const ctx = document.getElementById('diagnosticDistributionChart')?.getContext('2d');
    if (!ctx) return;
    
    console.log('🔍 updateDiagnosticDistribution - données reçues:', data);
    
    if (diagnosticDistributionChart) {
        diagnosticDistributionChart.destroy();
    }
    
    // Vérification et normalisation des données
    // L'API renvoie {labels: [], counts: [], percentages: []}
    const labels = (data && data.labels && Array.isArray(data.labels)) ? data.labels : [];
    const dataValues = (data && data.counts && Array.isArray(data.counts)) ? data.counts : 
                       (data && data.data && Array.isArray(data.data)) ? data.data : [];
    
    console.log('📊 Labels et valeurs pour diagnosticDistribution:', { labels, dataValues });
    
    const colors = {
        'Normal': { bg: 'rgba(16, 185, 129, 0.8)', border: 'rgba(16, 185, 129, 1)' },
        'Gliome': { bg: 'rgba(239, 68, 68, 0.8)', border: 'rgba(239, 68, 68, 1)' },
        'Méningiome': { bg: 'rgba(251, 191, 36, 0.8)', border: 'rgba(251, 191, 36, 1)' },
        'Tumeur pituitaire': { bg: 'rgba(168, 85, 247, 0.8)', border: 'rgba(168, 85, 247, 1)' }
    };
    
    diagnosticDistributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: dataValues,
                backgroundColor: labels.map(label => colors[label]?.bg || 'rgba(156, 163, 175, 0.8)'),
                borderColor: labels.map(label => colors[label]?.border || 'rgba(156, 163, 175, 1)'),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                        font: {
                            size: 12,
                            weight: 500
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            // Protection contre données invalides
                            if (!context || !context.dataset || !context.dataset.data || context.parsed === undefined) {
                                return context?.label || 'Données';
                            }
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const value = context.parsed;
                            const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : '0.0';
                            return context.label + ': ' + value + ' (' + percentage + '%)';
                        }
                    }
                }
            }
        }
    });

    // Mettre à jour les compteurs avec vérification de sécurité
    const counts = ['normalCount', 'gliomeCount', 'meningiomeCount', 'pituitaireCount'];
    // Les données peuvent avoir soit 'counts' soit 'data'
    const countsData = (data && data.counts && Array.isArray(data.counts)) ? data.counts :
                       (data && data.data && Array.isArray(data.data)) ? data.data : [];
    
    if (data && data.labels && Array.isArray(data.labels) && countsData.length > 0) {
        data.labels.forEach((label, index) => {
            const elementId = counts[index];
            const element = document.getElementById(elementId);
            if (element && countsData[index] !== undefined) {
                element.textContent = formatNumber(countsData[index]);
            }
        });
    }
}

function updateHourlyActivity(data) {
    const ctx = document.getElementById('hourlyHeatmapChart')?.getContext('2d');
    if (!ctx) return;
    
    console.log('🔍 updateHourlyActivity - données reçues:', data);
    
    if (hourlyHeatmapChart) {
        hourlyHeatmapChart.destroy();
    }
    
    // Normalisation des données avec valeurs par défaut
    // L'API renvoie {hourly_activity: [], peak_hour: N, quiet_hour: N, max_hourly_analyses: N}
    const chartData = (data && data.hourly_activity && Array.isArray(data.hourly_activity)) ? data.hourly_activity :
                      (data && data.data && Array.isArray(data.data)) ? data.data : [];
    const chartLabels = chartData.length > 0 ? chartData.map((_, i) => i + 'h') : [];
    const maxValue = chartData.length > 0 ? Math.max(...chartData) : 1;
    
    console.log('📊 Données pour hourlyActivity:', { chartData, chartLabels, maxValue });
    
    hourlyHeatmapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Activité',
                data: chartData,
                backgroundColor: function(context) {
                    // Vérification de sécurité pour context.parsed
                    if (!context || !context.parsed || context.parsed.y === undefined) {
                        return 'rgba(99, 102, 241, 0.5)'; // Couleur par défaut
                    }
                    const value = context.parsed.y;
                    const ratio = maxValue > 0 ? value / maxValue : 0;
                    return `rgba(99, 102, 241, ${0.3 + ratio * 0.7})`;
                },
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    // Mettre à jour les statistiques
    if (data.peak_hour !== undefined) {
        document.getElementById('peakHour').textContent = `${data.peak_hour.toString().padStart(2, '0')}h`;
    }
    if (data.max_hourly_analyses !== undefined) {
        document.getElementById('maxHourlyAnalyses').textContent = formatNumber(data.max_hourly_analyses);
    }
    if (data.quiet_hour !== undefined) {
        document.getElementById('quietHour').textContent = `${data.quiet_hour.toString().padStart(2, '0')}h`;
    }
}

function updateConfidenceDistribution(data) {
    const ctx = document.getElementById('confidenceHistogramChart')?.getContext('2d');
    if (!ctx) return;
    
    console.log('🔍 updateConfidenceDistribution - données reçues:', data);
    
    if (confidenceHistogramChart) {
        confidenceHistogramChart.destroy();
    }
    
    // L'API renvoie {histogram: [], intervals: {}}
    const chartData = (data && data.histogram && Array.isArray(data.histogram)) ? data.histogram :
                      (data && data.data && Array.isArray(data.data)) ? data.data : [];
    const chartLabels = chartData.length > 0 ? chartData.map((_, i) => `${(i * 10)}%-${(i + 1) * 10}%`) : [];
    
    console.log('📊 Données pour confidenceDistribution:', { chartData, chartLabels });
    
    confidenceHistogramChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Analyses',
                data: chartData,
                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    // Mettre à jour les statistiques
    if (data.intervals) {
        document.getElementById('veryHighConf').textContent = formatNumber(data.intervals.very_high || 0);
        document.getElementById('highConf').textContent = formatNumber(data.intervals.high || 0);
        document.getElementById('mediumConf').textContent = formatNumber(data.intervals.medium || 0);
        document.getElementById('lowConf').textContent = formatNumber(data.intervals.low || 0);
    }
}

function updateProcessingTimeAnalysis(data) {
    const ctx = document.getElementById('processingTimeChart')?.getContext('2d');
    if (!ctx) return;
    
    console.log('🔍 updateProcessingTimeAnalysis - données reçues:', data);
    
    if (processingTimeChart) {
        processingTimeChart.destroy();
    }
    
    // L'API renvoie {histogram: [], stats: {}}
    const chartData = (data && data.histogram && Array.isArray(data.histogram)) ? data.histogram :
                      (data && data.data && Array.isArray(data.data)) ? data.data : [];
    const chartLabels = chartData.length > 0 ? chartData.map((_, i) => `${(i * 0.5).toFixed(1)}s`) : [];
    
    console.log('📊 Données pour processingTime:', { chartData, chartLabels });
    
    processingTimeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Temps (s)',
                data: chartData,
                borderColor: 'rgba(251, 191, 36, 1)',
                backgroundColor: 'rgba(251, 191, 36, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 4,
                pointBackgroundColor: 'rgba(251, 191, 36, 1)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    // Mettre à jour les statistiques
    if (data.stats) {
        document.getElementById('fastProcessing').textContent = formatNumber(data.stats.fast || 0);
        document.getElementById('normalProcessing').textContent = formatNumber(data.stats.normal || 0);
        document.getElementById('slowProcessing').textContent = formatNumber(data.stats.slow || 0);
        document.getElementById('medianTime').textContent = (data.stats.median || 0).toFixed(2) + 's';
    }
}

function updateMonthlyTrends(data) {
    const ctx = document.getElementById('monthlyTrendsChart')?.getContext('2d');
    if (!ctx) return;
    
    console.log('🔍 updateMonthlyTrends - données reçues:', data);
    
    if (monthlyTrendsChart) {
        monthlyTrendsChart.destroy();
    }
    
    // L'API renvoie {labels: [], counts: [], confidences: [], growth_rate: N, most_active_month: ''}
    const chartData = (data && data.counts && Array.isArray(data.counts)) ? data.counts :
                      (data && data.data && Array.isArray(data.data)) ? data.data : [];
    const chartLabels = (data && data.labels && Array.isArray(data.labels)) ? data.labels : [];
    
    console.log('📊 Données pour monthlyTrends:', { chartData, chartLabels });
    
    monthlyTrendsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Analyses',
                data: chartData,
                borderColor: 'rgba(139, 92, 246, 1)',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointBackgroundColor: 'rgba(139, 92, 246, 1)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    // Mettre à jour les insights
    if (data.growth_rate !== undefined) {
        const growthElement = document.getElementById('monthlyGrowth');
        if (growthElement) {
            const growth = data.growth_rate;
            growthElement.textContent = `${growth > 0 ? '+' : ''}${growth.toFixed(1)}%`;
            growthElement.className = `text-2xl font-bold ${growth >= 0 ? 'text-green-600' : 'text-red-600'}`;
        }
    }
    if (data.most_active_month) {
        document.getElementById('mostActiveMonth').textContent = data.most_active_month;
    }
}

function updateYearComparison(data) {
    const ctx = document.getElementById('yearComparisonChart')?.getContext('2d');
    if (!ctx) return;
    
    console.log('🔍 updateYearComparison - données reçues:', data);
    
    if (yearComparisonChart) {
        yearComparisonChart.destroy();
    }
    
    // Transformer les données monthly en format pour le graphique
    const labels = ['Ce mois', 'Mois dernier'];
    let thisMonthCount = 0;
    let lastMonthCount = 0;
    
    if (data.monthly) {
        thisMonthCount = data.monthly['Ce mois']?.count || 0;
        lastMonthCount = data.monthly['Mois dernier']?.count || 0;
    }
    
    console.log('📊 Données transformées pour yearComparison:', {
        labels,
        data: [thisMonthCount, lastMonthCount]
    });
    
    yearComparisonChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Nombre d\'analyses',
                data: [thisMonthCount, lastMonthCount],
                borderColor: 'rgba(99, 102, 241, 1)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 6,
                pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                pointBorderColor: '#fff',
                pointBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
    
    // Mettre à jour les statistiques
    const growth = thisMonthCount > 0 && lastMonthCount > 0 
        ? (((thisMonthCount - lastMonthCount) / lastMonthCount) * 100).toFixed(1) 
        : '0.0';
    const growthElem = document.getElementById('yearGrowth');
    if (growthElem) {
        growthElem.textContent = growth > 0 ? `+${growth}%` : `${growth}%`;
        growthElem.className = growth > 0 
            ? 'text-lg font-bold text-green-600' 
            : 'text-lg font-bold text-red-600';
    }
    
    // Prédiction simple basée sur la croissance
    const prediction = thisMonthCount > 0 
        ? Math.round(thisMonthCount * (1 + parseFloat(growth) / 100))
        : 0;
    const predictionElem = document.getElementById('nextMonthPrediction');
    if (predictionElem) {
        predictionElem.textContent = prediction;
    }
    
    // Tendance
    const trendElem = document.getElementById('trendDirection');
    if (trendElem) {
        if (growth > 5) {
            trendElem.innerHTML = '<i class="fas fa-arrow-up mr-1"></i>Hausse';
            trendElem.className = 'text-lg font-bold text-green-600';
        } else if (growth < -5) {
            trendElem.innerHTML = '<i class="fas fa-arrow-down mr-1"></i>Baisse';
            trendElem.className = 'text-lg font-bold text-red-600';
        } else {
            trendElem.innerHTML = '<i class="fas fa-minus mr-1"></i>Stable';
            trendElem.className = 'text-lg font-bold text-blue-600';
        }
    }
}

function updateAIInsights(data) {
    // Performance Insights
    const perfContainer = document.getElementById('performanceInsights');
    if (perfContainer && data.performance_insights) {
        perfContainer.innerHTML = data.performance_insights.length === 0 ? 
            '<p class="text-sm text-gray-600">Aucune anomalie détectée</p>' :
            data.performance_insights.map(insight => `
                <div class="flex items-start p-3 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow">
                    <i class="fas fa-lightbulb text-blue-600 mr-3 mt-1"></i>
                    <p class="text-sm text-gray-700">${insight}</p>
                </div>
            `).join('');
    }

    // Quality Insights
    const qualityContainer = document.getElementById('qualityInsights');
    if (qualityContainer && data.quality_insights) {
        qualityContainer.innerHTML = data.quality_insights.length === 0 ?
            '<p class="text-sm text-gray-600">Qualité optimale</p>' :
            data.quality_insights.map(insight => `
                <div class="flex items-start p-3 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow">
                    <i class="fas fa-check-circle text-green-600 mr-3 mt-1"></i>
                    <p class="text-sm text-gray-700">${insight}</p>
                </div>
            `).join('');
    }

    // Recommendations
    const recoContainer = document.getElementById('recommendationsInsights');
    if (recoContainer && data.recommendations) {
        recoContainer.innerHTML = data.recommendations.length === 0 ?
            '<p class="text-sm text-gray-600">Aucune recommandation</p>' :
            data.recommendations.map(reco => `
                <div class="flex items-start p-3 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow">
                    <i class="fas fa-star text-purple-600 mr-3 mt-1"></i>
                    <p class="text-sm text-gray-700">${reco}</p>
                </div>
            `).join('');
    }

    // Update scores
    if (data.scores) {
        document.getElementById('accuracyScore').textContent = (data.scores.accuracy || 0) + '%';
        document.getElementById('efficiencyScore').textContent = (data.scores.efficiency || 0) + '%';
        document.getElementById('reliabilityScore').textContent = (data.scores.reliability || 0) + '%';
        document.getElementById('overallScore').textContent = (data.scores.overall || 0) + '%';
    }
}

function updateAdvancedMetrics(data) {
    if (!data) return;
    
    // KPIs
    document.getElementById('throughputRate').textContent = data.throughput_rate || '0';
    document.getElementById('throughputChange').textContent = `${data.throughput_change > 0 ? '+' : ''}${data.throughput_change || 0}%`;
    document.getElementById('accuracyRate').textContent = (data.accuracy_rate || 0) + '%';
    document.getElementById('avgResponseTime').textContent = (data.avg_response_time || 0).toFixed(2) + 's';
    document.getElementById('systemUptime').textContent = (data.system_uptime || 0) + '%';

    // Year comparison
    document.getElementById('yearGrowth').textContent = `${data.year_growth > 0 ? '+' : ''}${data.year_growth || 0}%`;
    document.getElementById('nextMonthPrediction').textContent = data.next_month_prediction || '0';
    document.getElementById('trendDirection').textContent = data.trend_direction || 'Stable';

    // Response time status
    const responseTime = data.avg_response_time || 0;
    const statusElement = document.getElementById('responseTimeStatus');
    if (statusElement) {
        if (responseTime < 2) {
            statusElement.textContent = 'Excellent';
            statusElement.className = 'text-green-600 font-medium';
        } else if (responseTime < 4) {
            statusElement.textContent = 'Bon';
            statusElement.className = 'text-blue-600 font-medium';
        } else {
            statusElement.textContent = 'À améliorer';
            statusElement.className = 'text-orange-600 font-medium';
        }
    }

    // Progress bars
    const systemLoad = Math.min(Math.max(responseTime * 10, 20), 80);
    document.getElementById('systemLoad').textContent = systemLoad.toFixed(0) + '%';
    const loadBar = document.getElementById('systemLoadBar');
    if (loadBar) {
        loadBar.style.width = systemLoad + '%';
        loadBar.className = systemLoad < 50 ? 'bg-green-500 h-2 rounded-full' :
                           systemLoad < 75 ? 'bg-yellow-500 h-2 rounded-full' :
                           'bg-red-500 h-2 rounded-full';
    }

    const modelUsage = Math.min(Math.max((data.throughput_rate || 0) * 5, 40), 90);
    document.getElementById('modelUsage').textContent = modelUsage.toFixed(0) + '%';
    const usageBar = document.getElementById('modelUsageBar');
    if (usageBar) {
        usageBar.style.width = modelUsage + '%';
    }
}

/**
 * ========================================
 * FILTRES
 * ========================================
 */
function initializeFilters() {
    const toggleFilters = document.getElementById('toggleFilters');
    const filtersPanel = document.getElementById('filtersPanel');
    
    toggleFilters?.addEventListener('click', () => {
        filtersPanel.classList.toggle('hidden');
        if (!filtersPanel.classList.contains('hidden')) {
            loadFilterCounts();
        }
    });

    // Range sliders
    ['minConfidence', 'maxConfidence', 'maxProcessingTime'].forEach(id => {
        const slider = document.getElementById(id);
        slider?.addEventListener('input', (e) => {
            const valueId = id + 'Value';
            const suffix = id.includes('Time') ? 's' : '%';
            document.getElementById(valueId).textContent = e.target.value + suffix;
            updateActiveFilters();
        });
    });

    // Quick date buttons
    document.querySelectorAll('.quick-date-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const days = parseInt(e.target.dataset.days);
            const endDate = new Date();
            const startDate = new Date();
            startDate.setDate(startDate.getDate() - days);
            
            document.getElementById('startDate').value = startDate.toISOString().split('T')[0];
            document.getElementById('endDate').value = endDate.toISOString().split('T')[0];
            updateActiveFilters();
        });
    });

    // Quick time buttons
    document.querySelectorAll('.quick-time-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const time = parseFloat(e.target.dataset.time);
            document.getElementById('maxProcessingTime').value = time;
            document.getElementById('maxTimeValue').textContent = time + 's';
            updateActiveFilters();
        });
    });

    // Checkboxes
    document.querySelectorAll('.diagnostic-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', updateActiveFilters);
    });

    // Date inputs
    document.getElementById('startDate')?.addEventListener('change', updateActiveFilters);
    document.getElementById('endDate')?.addEventListener('change', updateActiveFilters);

    // Action buttons
    document.getElementById('applyFilters')?.addEventListener('click', applyFiltersFunction);
    document.getElementById('previewFilters')?.addEventListener('click', previewFiltersFunction);
    document.getElementById('resetFilters')?.addEventListener('click', resetFiltersFunction);
}

function updateActiveFilters() {
    const activeFilters = [];
    const container = document.getElementById('activeFilters');
    const list = document.getElementById('activeFiltersList');
    const badge = document.getElementById('activeFiltersCount');

    // Date filters
    const startDate = document.getElementById('startDate')?.value;
    const endDate = document.getElementById('endDate')?.value;
    if (startDate || endDate) {
        activeFilters.push({
            type: 'date',
            label: `📅 ${startDate || '...'} → ${endDate || '...'}`,
            value: `${startDate}-${endDate}`
        });
    }

    // Diagnostic filters
    const selected = Array.from(document.querySelectorAll('.diagnostic-checkbox:checked')).map(cb => cb.value);
    if (selected.length > 0) {
        activeFilters.push({
            type: 'diagnostic',
            label: `🧠 ${selected.join(', ')}`,
            value: selected
        });
    }

    // Confidence range
    const minConf = document.getElementById('minConfidence')?.value;
    const maxConf = document.getElementById('maxConfidence')?.value;
    if (minConf > 0 || maxConf < 100) {
        activeFilters.push({
            type: 'confidence',
            label: `📊 Confiance: ${minConf}% - ${maxConf}%`,
            value: `${minConf}-${maxConf}`
        });
    }

    // Processing time
    const maxTime = document.getElementById('maxProcessingTime')?.value;
    if (maxTime < 10) {
        activeFilters.push({
            type: 'time',
            label: `⏱️ Temps max: ${maxTime}s`,
            value: maxTime
        });
    }

    // Update display
    if (activeFilters.length > 0) {
        container?.classList.remove('hidden');
        badge?.classList.remove('hidden');
        if (badge) badge.textContent = activeFilters.length;

        if (list) {
            list.innerHTML = activeFilters.map(filter => `
                <span class="filter-badge px-3 py-1.5 rounded-full text-xs font-medium flex items-center">
                    ${filter.label}
                    <button onclick="removeFilter('${filter.type}')" class="ml-2 hover:text-red-200">
                        <i class="fas fa-times"></i>
                    </button>
                </span>
            `).join('');
        }
    } else {
        container?.classList.add('hidden');
        badge?.classList.add('hidden');
        if (list) list.innerHTML = '';
    }
}

async function loadFilterCounts() {
    try {
        const response = await fetch('/api/analytics/filter-counts');
        const data = await response.json();
        
        if (data.success && data.counts) {
            Object.keys(data.counts).forEach(key => {
                const element = document.getElementById(`count-${key}`);
                if (element) {
                    element.textContent = data.counts[key];
                }
            });
        }
    } catch (error) {
        console.error('Erreur chargement compteurs:', error);
    }
}

async function applyFiltersFunction() {
    const filters = getFilterValues();
    showNotification('Application des filtres...', 'info');

    try {
        const response = await fetch('/api/analytics/filtered', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(filters)
        });

        const data = await response.json();

        if (data.success) {
            updateFilteredResults(data.data);
            showNotification(`${data.data.analyses.length} résultats trouvés`, 'success');
        } else {
            showNotification('Erreur lors de l\'application des filtres', 'error');
        }
    } catch (error) {
        console.error('Erreur application filtres:', error);
        showNotification('Erreur lors de l\'application des filtres', 'error');
    }
}

async function previewFiltersFunction() {
    const filters = getFilterValues();
    
    try {
        const response = await fetch('/api/analytics/filter-preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(filters)
        });

        const data = await response.json();

        if (data.success) {
            const preview = document.getElementById('filterResultsPreview');
            if (preview) {
                preview.textContent = `Aperçu: ${data.count} résultat(s) correspondant aux critères`;
            }
            showNotification(`${data.count} résultats trouvés`, 'info');
        }
    } catch (error) {
        console.error('Erreur aperçu filtres:', error);
    }
}

function resetFiltersFunction() {
    document.getElementById('startDate').value = '';
    document.getElementById('endDate').value = '';
    document.querySelectorAll('.diagnostic-checkbox').forEach(cb => cb.checked = false);
    document.getElementById('minConfidence').value = 0;
    document.getElementById('maxConfidence').value = 100;
    document.getElementById('maxProcessingTime').value = 10;
    document.getElementById('minConfidenceValue').textContent = '0%';
    document.getElementById('maxConfidenceValue').textContent = '100%';
    document.getElementById('maxTimeValue').textContent = '10s';
    
    const preview = document.getElementById('filterResultsPreview');
    if (preview) {
        preview.textContent = 'Filtres réinitialisés';
    }
    
    updateActiveFilters();
    showNotification('Filtres réinitialisés', 'success');
}

function getFilterValues() {
    return {
        start_date: document.getElementById('startDate')?.value || null,
        end_date: document.getElementById('endDate')?.value || null,
        diagnostic_types: Array.from(document.querySelectorAll('.diagnostic-checkbox:checked')).map(cb => cb.value),
        min_confidence: parseInt(document.getElementById('minConfidence')?.value || 0),
        max_confidence: parseInt(document.getElementById('maxConfidence')?.value || 100),
        max_processing_time: parseFloat(document.getElementById('maxProcessingTime')?.value || 10)
    };
}

function updateFilteredResults(data) {
    const tbody = document.getElementById('filteredAnalysesTable');
    const countSpan = document.getElementById('filteredCount');
    
    if (countSpan) countSpan.textContent = `${data.analyses.length} résultats`;
    if (!tbody) return;
    
    tbody.innerHTML = '';

    if (data.analyses.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="px-6 py-12 text-center text-gray-500">
                    <i class="fas fa-search text-4xl mb-3 opacity-50"></i>
                    <p class="font-medium">Aucun résultat trouvé</p>
                    <p class="text-sm">Essayez de modifier vos critères de recherche</p>
                </td>
            </tr>
        `;
        return;
    }

    data.analyses.forEach(analysis => {
        const row = document.createElement('tr');
        row.className = 'hover:bg-gray-50 transition-colors';
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${formatDate(analysis.created_at)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                ${analysis.filename}
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                <span class="px-3 py-1 rounded-full text-xs font-medium ${getBadgeClass(analysis.predicted_label)}">
                    ${analysis.predicted_label}
                </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-semibold">
                ${analysis.confidence.toFixed(2)}%
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                ${analysis.processing_time.toFixed(2)}s
            </td>
            <td class="px-6 py-4 text-sm text-gray-500">
                ${analysis.description || '-'}
            </td>
        `;
        tbody.appendChild(row);
    });
}

function removeFilter(type) {
    switch(type) {
        case 'date':
            document.getElementById('startDate').value = '';
            document.getElementById('endDate').value = '';
            break;
        case 'diagnostic':
            document.querySelectorAll('.diagnostic-checkbox').forEach(cb => cb.checked = false);
            break;
        case 'confidence':
            document.getElementById('minConfidence').value = 0;
            document.getElementById('maxConfidence').value = 100;
            document.getElementById('minConfidenceValue').textContent = '0%';
            document.getElementById('maxConfidenceValue').textContent = '100%';
            break;
        case 'time':
            document.getElementById('maxProcessingTime').value = 10;
            document.getElementById('maxTimeValue').textContent = '10s';
            break;
    }
    updateActiveFilters();
}

/**
 * ========================================
 * CONTRÔLES DES GRAPHIQUES
 * ========================================
 */
function initializeChartControls() {
    // Boutons de période pour le graphique de comparaison
    document.querySelectorAll('.comparison-period-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.comparison-period-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            const period = this.dataset.period;
            loadComparisonByPeriod(period);
        });
    });

    // Boutons de période pour les tendances
    document.querySelectorAll('.trend-period-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.trend-period-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            const period = this.dataset.period;
            loadMonthlyTrendsByPeriod(period);
        });
    });
}

async function loadComparisonByPeriod(period) {
    showNotification(`Chargement de la période: ${period}`, 'info');
    // Cette fonction peut être étendue pour charger des données spécifiques selon la période
    loadComparisons();
}

async function loadMonthlyTrendsByPeriod(period) {
    showNotification(`Chargement des tendances: ${period}`, 'info');
    loadAdvancedAnalytics();
}

/**
 * ========================================
 * INTERFACE UTILISATEUR
 * ========================================
 */
function initializeUI() {
    // Export menu
    const exportBtn = document.getElementById('exportBtn');
    const exportMenu = document.getElementById('exportMenu');
    
    exportBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        exportMenu?.classList.toggle('hidden');
        document.getElementById('userMenuDropdown')?.classList.add('hidden');
        document.getElementById('alertsDropdown')?.classList.add('hidden');
    });

    // User menu
    const userMenuButton = document.getElementById('userMenuButton');
    const userMenuDropdown = document.getElementById('userMenuDropdown');
    
    userMenuButton?.addEventListener('click', (e) => {
        e.stopPropagation();
        userMenuDropdown?.classList.toggle('hidden');
        exportMenu?.classList.add('hidden');
        document.getElementById('alertsDropdown')?.classList.add('hidden');
    });

    // Alerts menu
    const alertsButton = document.getElementById('alertsButton');
    const alertsDropdown = document.getElementById('alertsDropdown');
    
    alertsButton?.addEventListener('click', (e) => {
        e.stopPropagation();
        alertsDropdown?.classList.toggle('hidden');
        exportMenu?.classList.add('hidden');
        userMenuDropdown?.classList.add('hidden');
        
        if (!alertsDropdown?.classList.contains('hidden')) {
            loadAlerts();
        }
    });

    // Fermer tous les menus en cliquant ailleurs
    document.addEventListener('click', () => {
        exportMenu?.classList.add('hidden');
        userMenuDropdown?.classList.add('hidden');
        alertsDropdown?.classList.add('hidden');
    });
}

function animateElements() {
    const elements = document.querySelectorAll('.fade-in-up, .slide-in-right');
    elements.forEach((el, index) => {
        el.style.opacity = '0';
        setTimeout(() => {
            el.style.opacity = '1';
        }, index * 50);
    });
}

/**
 * ========================================
 * FONCTIONS UTILITAIRES
 * ========================================
 */
function formatNumber(num) {
    return new Intl.NumberFormat('fr-FR').format(num);
}

function formatDate(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

function getBadgeClass(label) {
    const classes = {
        'Normal': 'bg-green-100 text-green-800',
        'Gliome': 'bg-red-100 text-red-800',
        'Méningiome': 'bg-yellow-100 text-yellow-800',
        'Tumeur pituitaire': 'bg-purple-100 text-purple-800'
    };
    return classes[label] || 'bg-gray-100 text-gray-800';
}

function getAlertIcon(type) {
    const icons = {
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle',
        'success': 'fa-check-circle',
        'error': 'fa-times-circle'
    };
    return icons[type] || 'fa-bell';
}

function getAlertColor(type) {
    const colors = {
        'warning': 'amber',
        'info': 'blue',
        'success': 'green',
        'error': 'red'
    };
    return colors[type] || 'gray';
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg transition-all duration-300 transform translate-x-full max-w-md';

    const colors = {
        'success': 'bg-green-500',
        'error': 'bg-red-500',
        'warning': 'bg-yellow-500',
        'info': 'bg-blue-500'
    };
    
    const icons = {
        'success': 'fa-check-circle',
        'error': 'fa-times-circle',
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle'
    };

    notification.className += ` ${colors[type]} text-white`;

    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas ${icons[type]} mr-3 text-xl"></i>
            <div class="flex-1">
                <p class="font-semibold text-sm">${message}</p>
            </div>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-4 hover:bg-white hover:bg-opacity-20 p-2 rounded-full transition-colors">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

    document.body.appendChild(notification);

    // Animation d'entrée
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);

    // Auto-suppression après 5 secondes
    setTimeout(() => {
        notification.style.transform = 'translateX(120%)';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function dismissAlert(button) {
    const alert = button.closest('.alert-warning, .alert-info, .alert-success, .alert-error');
    if (alert) {
        alert.style.opacity = '0';
        alert.style.transform = 'translateX(100%)';
        setTimeout(() => alert.remove(), 300);
    }
}

function clearAllAlerts() {
    document.getElementById('alertsBadge')?.classList.add('hidden');
    const alertsList = document.getElementById('alertsList');
    if (alertsList) {
        alertsList.innerHTML = `
            <div class="p-8 text-center text-gray-500">
                <i class="fas fa-check-circle text-4xl text-green-500 mb-3"></i>
                <p class="font-medium">Toutes les alertes ont été marquées comme lues</p>
            </div>
        `;
    }
    showNotification('Alertes marquées comme lues', 'success');
}

/**
 * ========================================
 * FONCTIONS D'EXPORT
 * ========================================
 */
function exportToPDF() {
    window.location.href = '/api/analytics/export/pdf';
    showNotification('Génération du rapport PDF...', 'info');
}

function exportDiagnosticData() {
    window.location.href = '/api/analytics/export/diagnostic-csv';
    showNotification('Export diagnostic en cours...', 'info');
}

function exportInsights() {
    window.location.href = '/api/analytics/export/insights-pdf';
    showNotification('Export du rapport d\'insights...', 'info');
}

function exportFilteredResults() {
    const filteredData = getCurrentFilteredData();
    if (filteredData.length === 0) {
        showNotification('Aucune donnée à exporter', 'warning');
        return;
    }

    const csv = convertToCSV(filteredData);
    downloadCSV(csv, 'analyses_filtrees.csv');
    showNotification('Export réussi', 'success');
}

function refreshInsights() {
    showNotification('Actualisation des insights...', 'info');
    loadAIInsights();
}

function refreshFilteredResults() {
    showNotification('Actualisation en cours...', 'info');
    applyFiltersFunction();
}

function toggleYearComparison() {
    showNotification('Actualisation de la comparaison annuelle...', 'info');
    loadAdvancedMetrics();
}

function changeHeatmapView(view) {
    showNotification(`Vue ${view === 'today' ? 'aujourd\'hui' : '7 jours'} sélectionnée`, 'info');
    loadAdvancedAnalytics();
}

function convertToCSV(data) {
    const headers = ['Date/Heure', 'Fichier', 'Diagnostic', 'Confiance', 'Temps', 'Description'];
    const csvContent = [
        headers.join(','),
        ...data.map(row => [
            row.created_at,
            row.filename,
            row.predicted_label,
            row.confidence,
            row.processing_time,
            row.description || ''
        ].join(','))
    ].join('\n');

    return csvContent;
}

function downloadCSV(csv, filename) {
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function getCurrentFilteredData() {
    // Cette fonction retourne les données actuellement filtrées
    // Elle devrait être implémentée selon votre logique de stockage des données
    return [];
}

/**
 * ========================================
 * MODAL FUNCTIONS
 * ========================================
 */
function openProfileModal() {
    showNotification('Modal profil - À implémenter', 'info');
    // TODO: Implémenter la modal du profil utilisateur
}

function openSettingsModal() {
    showNotification('Modal paramètres - À implémenter', 'info');
    // TODO: Implémenter la modal des paramètres
}

// Cleanup au démontage
window.addEventListener('beforeunload', () => {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    
    // Détruire tous les graphiques
    [comparisonChart, performanceChart, diagnosticDistributionChart, 
     yearComparisonChart, hourlyHeatmapChart, confidenceHistogramChart,
     processingTimeChart, monthlyTrendsChart].forEach(chart => {
        if (chart) chart.destroy();
    });
});

console.log('✅ Script pro_dashboard_advanced.js chargé');
