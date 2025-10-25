// ====================================================================
// NEUROSCAN ANALYTICS - PAGE RÉSULTAT D'ANALYSE
// Design System Pro avec Animations & Interactions Avancées
// ====================================================================

// Données d'analyse pour JavaScript
window.__analysisData = JSON.parse(document.getElementById('analysis-json')?.textContent || '{}');

// ====================================================================
// ANIMATIONS D'ENTRÉE - Système d'animation séquentielle
// ====================================================================
document.addEventListener('DOMContentLoaded', function() {
  // Animation des cartes avec délai progressif
  const cards = document.querySelectorAll('.fade-in');
  cards.forEach((card, index) => {
    card.style.animationDelay = `${index * 0.12}s`;
    card.style.opacity = '0';
  });
  
  // Ajouter des effets de parallaxe légers au scroll
  window.addEventListener('scroll', handleParallaxScroll, { passive: true });
  
  // Initialiser tous les composants interactifs
  initializeInteractiveElements();
  
  // Effet de particules subtil sur l'en-tête
  createHeaderParticles();
});

// ====================================================================
// FONCTION D'INITIALISATION DES ÉLÉMENTS INTERACTIFS
// ====================================================================
function initializeInteractiveElements() {
  // Ajouter des effets de hover avancés sur les cartes
  document.querySelectorAll('.analysis-card').forEach(card => {
    card.addEventListener('mouseenter', handleCardHover);
    card.addEventListener('mouseleave', handleCardLeave);
  });
  
  // Ajouter des tooltips personnalisés
  initializeTooltips();
  
  // Initialiser les compteurs animés
  animateNumbers();
  
  // Animer les cartes de métriques
  animateMetricCards();

  // Animer les cartes d'informations patient
  animatePatientInfoCards();
}// Effet de parallaxe léger au scroll
function handleParallaxScroll() {
  const scrolled = window.pageYOffset;
  const parallaxElements = document.querySelectorAll('.analysis-card');
  
  parallaxElements.forEach((el, index) => {
    const speed = 0.05 + (index * 0.01);
    const yPos = -(scrolled * speed);
    el.style.transform = `translateY(${yPos}px)`;
  });
}

// Effet de tilt sur les cartes au hover
function handleCardHover(e) {
  const card = e.currentTarget;
  const rect = card.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  
  const centerX = rect.width / 2;
  const centerY = rect.height / 2;
  
  const rotateX = (y - centerY) / 20;
  const rotateY = (centerX - x) / 20;
  
  card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-6px) scale(1.01)`;
}

function handleCardLeave(e) {
  const card = e.currentTarget;
  card.style.transform = '';
}

// Créer des particules subtiles dans l'en-tête
function createHeaderParticles() {
  const header = document.querySelector('.analysis-header');
  if (!header) return;
  
  const particlesContainer = document.createElement('div');
  particlesContainer.className = 'header-particles';
  particlesContainer.style.cssText = `
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    pointer-events: none;
    z-index: 1;
  `;
  
  // Créer 15 particules
  for (let i = 0; i < 15; i++) {
    const particle = document.createElement('div');
    particle.style.cssText = `
      position: absolute;
      width: ${Math.random() * 4 + 2}px;
      height: ${Math.random() * 4 + 2}px;
      background: rgba(255, 255, 255, ${Math.random() * 0.5 + 0.2});
      border-radius: 50%;
      left: ${Math.random() * 100}%;
      top: ${Math.random() * 100}%;
      animation: float ${Math.random() * 10 + 10}s linear infinite;
      animation-delay: ${Math.random() * 5}s;
    `;
    particlesContainer.appendChild(particle);
  }
  
  header.insertBefore(particlesContainer, header.firstChild);
  
  // Ajouter l'animation CSS
  const style = document.createElement('style');
  style.textContent = `
    @keyframes float {
      0%, 100% { transform: translateY(0) translateX(0); }
      25% { transform: translateY(-20px) translateX(10px); }
      50% { transform: translateY(-40px) translateX(-10px); }
      75% { transform: translateY(-20px) translateX(5px); }
    }
  `;
  document.head.appendChild(style);
}

// Animation des nombres (compteur)
function animateNumbers() {
  const confidence = parseFloat(window.__analysisData.confidence || 0);
  const confidenceElement = document.querySelector('.text-4xl.font-black.text-gray-900');
  const confidenceFill = document.getElementById('confFill');

  if (confidenceElement && confidence > 0) {
    let current = 0;
    const increment = confidence / 60; // Animation plus lente et fluide
    const timer = setInterval(() => {
      current += increment;
      if (current >= confidence) {
        current = confidence;
        clearInterval(timer);
      }
      confidenceElement.textContent = current.toFixed(1) + '%';
    }, 25);
  }

  // Animation de la barre de confiance avec délai
  if (confidenceFill) {
    setTimeout(() => {
      confidenceFill.style.width = confidence + '%';
    }, 800); // Délai pour laisser le temps aux autres animations
  }
}

// Animation des cartes de métriques avec effet de révélation
function animateMetricCards() {
  const metricCards = document.querySelectorAll('.metric-card-advanced');
  metricCards.forEach((card, index) => {
    // Réinitialiser l'état initial
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px) scale(0.95)';

    // Animer avec délai progressif (2 cartes seulement maintenant)
    setTimeout(() => {
      card.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
      card.style.opacity = '1';
      card.style.transform = 'translateY(0) scale(1)';
    }, 1200 + (index * 200)); // Délai plus espacé pour 2 cartes
  });
}

// Animation des cartes d'informations patient
function animatePatientInfoCards() {
  const patientCards = document.querySelectorAll('.patient-info-item-pro');
  patientCards.forEach((card, index) => {
    // Les cartes ont déjà leurs animations CSS définies, mais on peut ajouter un délai initial
    card.style.animationDelay = `${0.1 + (index * 0.1)}s`;
  });
}

// Initialiser les tooltips
function initializeTooltips() {
  document.querySelectorAll('[title]').forEach(el => {
    el.addEventListener('mouseenter', showTooltip);
    el.addEventListener('mouseleave', hideTooltip);
  });
}

function showTooltip(e) {
  const title = e.currentTarget.getAttribute('title');
  if (!title) return;
  
  const tooltip = document.createElement('div');
  tooltip.className = 'custom-tooltip';
  tooltip.textContent = title;
  tooltip.style.cssText = `
    position: fixed;
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 12px;
    z-index: 10000;
    pointer-events: none;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    animation: tooltipFadeIn 0.2s ease;
  `;
  
  document.body.appendChild(tooltip);
  
  const rect = e.currentTarget.getBoundingClientRect();
  tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
  tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
  
  e.currentTarget._tooltip = tooltip;
}

function hideTooltip(e) {
  const tooltip = e.currentTarget._tooltip;
  if (tooltip) {
    tooltip.style.animation = 'tooltipFadeOut 0.2s ease';
    setTimeout(() => tooltip.remove(), 200);
  }
}

// ====================================================================
// ANIMATION DE LA BARRE DE CONFIANCE
// ====================================================================
(function(){
  const confFill = document.getElementById('confFill');
  if (confFill) {
    const width = parseFloat(confFill.dataset.width || '0');
    setTimeout(() => {
      confFill.style.width = Math.max(0, Math.min(100, width)) + '%';
    }, 500);
  }
})();

// ====================================================================
// ANIMATION DES BARRES DE PROBABILITÉ
// ====================================================================
(function(){
  const colors = ['#3b82f6','#7c3aed','#10b981','#f59e0b','#ef4444'];
  const probBars = document.querySelectorAll('.prob-bar-fill');
  
  probBars.forEach((bar, index) => {
    const width = parseFloat(bar.dataset.width || '0');
    const color = colors[index % colors.length];
    
    // Appliquer la couleur avec dégradé amélioré
    bar.style.background = `linear-gradient(135deg, ${color}, ${adjustBrightness(color, -25)})`;
    bar.style.boxShadow = `0 2px 8px ${color}40`;
    
    // Animation de largeur avec effet élastique amélioré
    setTimeout(() => {
      bar.style.width = Math.max(0, Math.min(100, width)) + '%';
      
      // Ajouter des effets de hover avancés
      const probItem = bar.closest('.prob-item');
      if (probItem) {
        probItem.addEventListener('mouseenter', () => {
          bar.style.transform = 'scaleY(1.15) scaleX(1.02)';
          bar.style.boxShadow = `0 4px 12px ${color}60`;
          bar.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
        });
        
        probItem.addEventListener('mouseleave', () => {
          bar.style.transform = 'scaleY(1) scaleX(1)';
          bar.style.boxShadow = `0 2px 8px ${color}40`;
        });
      }
    }, 1000 + (index * 200)); // Délai plus long pour laisser le temps au graphique
  });

  // Appliquer les couleurs aux indicateurs circulaires avec glow amélioré
  document.querySelectorAll('.prob-color-0, .prob-color-1, .prob-color-2, .prob-color-3, .prob-color-4').forEach((circle, index) => {
    const colorIndex = index % colors.length;
    const color = colors[colorIndex];
    circle.style.background = `linear-gradient(135deg, ${color}, ${adjustBrightness(color, -20)})`;
    circle.style.boxShadow = `0 0 16px ${color}60, 0 2px 8px ${color}30`;
    circle.style.border = `2px solid rgba(255, 255, 255, 0.8)`;
  });
})();

// Fonction helper pour ajuster la luminosité
function adjustBrightness(color, percent) {
  const num = parseInt(color.replace('#', ''), 16);
  const amt = Math.round(2.55 * percent);
  const R = (num >> 16) + amt;
  const G = (num >> 8 & 0x00FF) + amt;
  const B = (num & 0x0000FF) + amt;
  return '#' + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
    (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
    (B < 255 ? B < 1 ? 0 : B : 255))
    .toString(16).slice(1);
}

// ====================================================================
// GRAPHIQUE DONUT AMÉLIORÉ - Chart.js Premium
// ====================================================================
(function(){
  try {
    const canvas = document.getElementById('probDonut');
    if (!canvas) return;
    
    const data = (window.__analysisData && window.__analysisData.probabilities) || {};
    const labels = Object.keys(data);
    const values = Object.values(data).map(v => Number(v) || 0);
    const colors = ['#3b82f6','#7c3aed','#10b981','#f59e0b','#ef4444'];
    
    // Créer des couleurs avec dégradés
    const gradients = colors.slice(0, labels.length).map((color, index) => {
      const gradient = canvas.getContext('2d').createLinearGradient(0, 0, 0, 400);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, adjustBrightness(color, -15));
      return gradient;
    });
    
    const chart = new Chart(canvas.getContext('2d'), {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: colors.slice(0, labels.length),
          borderWidth: 3,
          borderColor: '#ffffff',
          cutout: '65%',
          spacing: 2,
          hoverOffset: 15
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            enabled: true,
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            titleColor: '#ffffff',
            bodyColor: '#ffffff',
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderWidth: 1,
            cornerRadius: 12,
            padding: 12,
            titleFont: {
              size: 14,
              weight: 'bold'
            },
            bodyFont: {
              size: 13
            },
            displayColors: true,
            boxWidth: 12,
            boxHeight: 12,
            boxPadding: 6,
            callbacks: {
              label: function(context) {
                return ' ' + context.label + ': ' + context.parsed.toFixed(1) + '%';
              },
              title: function(context) {
                return 'Probabilité';
              }
            }
          }
        },
        animation: {
          animateRotate: true,
          animateScale: true,
          duration: 1500,
          easing: 'easeInOutQuart',
          onProgress: function(animation) {
            if (animation.currentStep === 1) {
              // Animation de pulse au début
              canvas.style.transform = 'scale(1.05)';
              setTimeout(() => {
                canvas.style.transform = 'scale(1)';
              }, 200);
            }
          }
        },
        onHover: (event, activeElements) => {
          event.native.target.style.cursor = activeElements.length > 0 ? 'pointer' : 'default';
        }
      }
    });
    
    // Ajouter un effet de rotation au hover
    canvas.addEventListener('mouseenter', () => {
      chart.options.animation.duration = 300;
      chart.update();
    });
    
  } catch (e) {
    console.error('Erreur lors de la création du graphique:', e);
    // Afficher un message d'erreur élégant
    const canvas = document.getElementById('probDonut');
    if (canvas && canvas.parentElement) {
      canvas.parentElement.innerHTML = `
        <div class="flex items-center justify-center h-64 text-gray-400">
          <div class="text-center">
            <i class="fas fa-chart-pie text-4xl mb-3"></i>
            <p class="text-sm">Graphique non disponible</p>
          </div>
        </div>
      `;
    }
  }
})();

// Actions des boutons
(function(){
  const pdfBtn = document.getElementById('detailPdfBtn');
  const shareBtn = document.getElementById('detailShareBtn');
  
  if (pdfBtn) {
    pdfBtn.addEventListener('click', async function() {
      this.disabled = true;
      const originalContent = this.innerHTML;
      this.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span class="hidden sm:inline ml-2">Génération...</span>';
      
      try {
        const data = window.__analysisData || {};
        const resp = await fetch('/generate-report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            patientName: data.patient_name,
            analysisData: data
          })
        });
        
        if (resp.ok) {
          // Créer un blob à partir de la réponse et déclencher le téléchargement
          const blob = await resp.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.style.display = 'none';
          a.href = url;
          
          // Générer un nom de fichier basé sur les données du patient
          const patientName = (data.patient_name || 'Patient').replace(/\s+/g, '_');
          const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
          a.download = `rapport_medical_${patientName}_${timestamp}.pdf`;
          
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
          
          // Notification de succès
          showNotification('Rapport PDF généré et téléchargé avec succès', 'success');
          
          // Effet confetti pour célébrer
          createConfetti();
        } else {
          const errorData = await resp.json().catch(() => ({ error: 'Erreur inconnue' }));
          showNotification(errorData.error || 'Erreur lors de la génération du rapport', 'error');
        }
      } catch (e) {
        console.error('Erreur lors du téléchargement du PDF:', e);
        showNotification('Erreur réseau lors de la génération du PDF', 'error');
      } finally {
        this.disabled = false;
        this.innerHTML = originalContent;
      }
    });
  }
  
  if (shareBtn) {
    shareBtn.addEventListener('click', async function() {
      const email = prompt('Email du destinataire:');
      if (!email) return;
      
      this.disabled = true;
      const originalContent = this.innerHTML;
      this.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span class="hidden sm:inline ml-2">Partage...</span>';
      
      try {
        const data = window.__analysisData || {};
        const resp = await fetch('/share-analysis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            recipientEmail: email,
            analysisData: data
          })
        });
        
        const result = await resp.json();
        
        if (result.success) {
          showNotification('Analyse partagée avec succès', 'success');
        } else {
          showNotification(result.error || 'Erreur lors du partage', 'error');
        }
      } catch (e) {
        showNotification('Erreur réseau lors du partage', 'error');
      } finally {
        this.disabled = false;
        this.innerHTML = originalContent;
      }
    });
  }
})();

// ====================================================================
// SYSTÈME DE NOTIFICATIONS - Design Premium
// ====================================================================
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification-toast fixed top-4 right-4 z-50 p-4 rounded-16 shadow-2xl max-w-sm transition-all duration-300 transform translate-x-full`;
  
  const icons = {
    success: '<i class="fas fa-check-circle text-xl"></i>',
    error: '<i class="fas fa-exclamation-circle text-xl"></i>',
    info: '<i class="fas fa-info-circle text-xl"></i>',
    warning: '<i class="fas fa-exclamation-triangle text-xl"></i>'
  };
  
  const colors = {
    success: 'bg-gradient-to-r from-green-500 to-green-600 text-white',
    error: 'bg-gradient-to-r from-red-500 to-red-600 text-white',
    info: 'bg-gradient-to-r from-blue-500 to-blue-600 text-white',
    warning: 'bg-gradient-to-r from-yellow-500 to-yellow-600 text-white'
  };
  
  notification.classList.add(...colors[type].split(' '));
  notification.innerHTML = `
    <div class="flex items-start gap-3">
      <div class="flex-shrink-0">
        ${icons[type]}
      </div>
      <div class="flex-1">
        <p class="font-semibold text-sm leading-relaxed">${message}</p>
      </div>
      <button onclick="this.parentElement.parentElement.remove()" class="flex-shrink-0 ml-2 hover:opacity-75 transition-opacity">
        <i class="fas fa-times"></i>
      </button>
    </div>
  `;
  
  // Ajouter un effet de blur backdrop
  notification.style.backdropFilter = 'blur(12px)';
  notification.style.boxShadow = '0 10px 40px rgba(0, 0, 0, 0.2), 0 4px 16px rgba(0, 0, 0, 0.15)';
  
  document.body.appendChild(notification);
  
  // Animation d'entrée avec bounce
  requestAnimationFrame(() => {
    notification.classList.remove('translate-x-full');
    notification.style.animation = 'slideInBounce 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
  });
  
  // Barre de progression
  const progressBar = document.createElement('div');
  progressBar.className = 'absolute bottom-0 left-0 h-1 bg-white bg-opacity-30 rounded-full';
  progressBar.style.width = '100%';
  progressBar.style.animation = 'progress 4s linear';
  notification.appendChild(progressBar);
  
  // Animation de sortie et suppression
  setTimeout(() => {
    notification.style.animation = 'slideOutRight 0.3s ease';
    setTimeout(() => {
      if (notification.parentElement) {
        document.body.removeChild(notification);
      }
    }, 300);
  }, 4000);
}

// Effet confetti pour les succès importants
function createConfetti() {
  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#7c3aed'];
  const confettiCount = 50;
  
  for (let i = 0; i < confettiCount; i++) {
    const confetti = document.createElement('div');
    confetti.style.cssText = `
      position: fixed;
      width: ${Math.random() * 10 + 5}px;
      height: ${Math.random() * 10 + 5}px;
      background: ${colors[Math.floor(Math.random() * colors.length)]};
      left: ${Math.random() * 100}vw;
      top: -20px;
      border-radius: ${Math.random() > 0.5 ? '50%' : '0'};
      opacity: ${Math.random() * 0.7 + 0.3};
      z-index: 9999;
      animation: confettiFall ${Math.random() * 3 + 2}s linear;
      transform: rotate(${Math.random() * 360}deg);
    `;
    
    document.body.appendChild(confetti);
    
    confetti.addEventListener('animationend', () => {
      confetti.remove();
    });
  }
  
  // Ajouter l'animation CSS
  if (!document.getElementById('confetti-style')) {
    const style = document.createElement('style');
    style.id = 'confetti-style';
    style.textContent = `
      @keyframes confettiFall {
        to {
          top: 100vh;
          transform: translateX(${Math.random() * 200 - 100}px) rotate(${Math.random() * 720}deg);
        }
      }
      @keyframes slideInBounce {
        0% { transform: translateX(100%); }
        60% { transform: translateX(-10px); }
        80% { transform: translateX(5px); }
        100% { transform: translateX(0); }
      }
      @keyframes slideOutRight {
        to { transform: translateX(120%); opacity: 0; }
      }
      @keyframes progress {
        from { width: 100%; }
        to { width: 0%; }
      }
    `;
    document.head.appendChild(style);
  }
}

// ====================================================================
// GESTION DE L'IMAGE D'ANALYSE - Modal et Interactions
// ====================================================================
function openImageModal() {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  const img = document.getElementById('analysisImage');
  
  modal.style.display = 'flex';
  modalImg.src = img.src;
  modalImg.alt = img.alt;
  
  // Animation d'ouverture fluide
  requestAnimationFrame(() => {
    modal.classList.add('opacity-100');
    modalImg.classList.add('scale-100');
    modalImg.style.animation = 'modalImageZoom 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
  });
  
  // Désactiver le scroll du body
  document.body.style.overflow = 'hidden';
}

function closeImageModal() {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  
  modal.classList.remove('opacity-100');
  modalImg.classList.remove('scale-100');
  modalImg.style.animation = 'modalImageZoomOut 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
  
  setTimeout(() => {
    modal.style.display = 'none';
    // Réactiver le scroll du body
    document.body.style.overflow = '';
  }, 300);
}

function toggleFullscreen() {
  openImageModal();
}

function downloadImage() {
  const img = document.getElementById('analysisImage');
  const link = document.createElement('a');
  const data = window.__analysisData || {};
  const filename = `analyse_${data.id}_${data.patient_name || 'patient'}_${Date.now()}`.replace(/\s+/g, '_');
  
  link.download = filename + '.jpg';
  link.href = img.src;
  link.click();
  
  // Notification de succès
  showNotification('Image téléchargée avec succès', 'success');
}

function toggleImageInfo() {
  const overlay = document.getElementById('imageOverlay');
  const isVisible = overlay.style.transform === 'translateY(0px)';
  overlay.style.transform = isVisible ? 'translateY(100%)' : 'translateY(0px)';
  overlay.style.transition = 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
}

function hideImageLoading() {
  const loading = document.getElementById('imageLoading');
  const img = document.getElementById('analysisImage');
  
  if (loading) {
    loading.style.opacity = '0';
    setTimeout(() => loading.style.display = 'none', 300);
  }
  
  // Obtenir la résolution de l'image
  const resolution = document.getElementById('imageResolution');
  if (resolution && img.naturalWidth && img.naturalHeight) {
    resolution.textContent = img.naturalWidth + ' × ' + img.naturalHeight + ' px';
  }
  
  // Animation d'apparition de l'image
  img.style.animation = 'fadeIn 0.6s ease';
}

function showImageError() {
  const container = document.querySelector('.image-container');
  const loading = document.getElementById('imageLoading');
  
  if (loading) loading.style.display = 'none';
  
  container.innerHTML = 
    '<div class="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200">' +
      '<div class="text-center text-gray-500 p-8">' +
        '<i class="fas fa-exclamation-triangle text-5xl mb-4 text-yellow-500"></i>' +
        '<div class="font-semibold text-lg mb-2">Image non disponible</div>' +
        '<div class="text-sm">Impossible de charger l\'image d\'analyse</div>' +
      '</div>' +
    '</div>';
}

// ====================================================================
// GESTION DES ÉVÉNEMENTS CLAVIER
// ====================================================================
document.addEventListener('keydown', (e) => {
  // Fermeture de la modal avec Escape
  if (e.key === 'Escape') {
    closeImageModal();
  }
  
  // Raccourcis clavier
  if (e.ctrlKey || e.metaKey) {
    switch(e.key.toLowerCase()) {
      case 'p':
        e.preventDefault();
        document.getElementById('detailPdfBtn')?.click();
        break;
      case 's':
        e.preventDefault();
        document.getElementById('detailShareBtn')?.click();
        break;
      case 'd':
        e.preventDefault();
        downloadImage();
        break;
    }
  }
});

// ====================================================================
// AJOUT DES STYLES D'ANIMATION DYNAMIQUES
// ====================================================================
const animationStyles = document.createElement('style');
animationStyles.textContent = `
  @keyframes modalImageZoom {
    from {
      opacity: 0;
      transform: scale(0.8);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }
  
  @keyframes modalImageZoomOut {
    from {
      opacity: 1;
      transform: scale(1);
    }
    to {
      opacity: 0;
      transform: scale(0.8);
    }
  }
  
  @keyframes tooltipFadeIn {
    from {
      opacity: 0;
      transform: translateY(-5px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  @keyframes tooltipFadeOut {
    from {
      opacity: 1;
      transform: translateY(0);
    }
    to {
      opacity: 0;
      transform: translateY(-5px);
    }
  }
  
  /* Amélioration du curseur sur les éléments interactifs */
  .analysis-card:hover,
  .action-btn:hover,
  .result-badge:hover,
  .info-chip:hover {
    cursor: pointer;
  }
  
  /* Animation de pulse pour les éléments importants */
  @keyframes pulse-subtle {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.85;
    }
  }
  
  /* Smooth scrolling amélioré */
  html {
    scroll-behavior: smooth;
  }
  
  /* Sélection de texte personnalisée */
  ::selection {
    background-color: rgba(102, 126, 234, 0.3);
    color: inherit;
  }
  
  /* Amélioration des focus pour l'accessibilité */
  *:focus-visible {
    outline: 2px solid rgba(59, 130, 246, 0.5);
    outline-offset: 2px;
    border-radius: 4px;
  }
`;
document.head.appendChild(animationStyles);

// ====================================================================
// PERFORMANCE - Préchargement des images importantes
// ====================================================================
if ('requestIdleCallback' in window) {
  requestIdleCallback(() => {
    const images = document.querySelectorAll('img[data-preload]');
    images.forEach(img => {
      const tempImg = new Image();
      tempImg.src = img.src;
    });
  });
}

// ====================================================================
// INTERACTIONS POUR LES AUTRES ANALYSES
// ====================================================================
document.addEventListener('DOMContentLoaded', function() {
  // Ajouter un effet de prévisualisation au hover sur les autres analyses
  const otherAnalysisLinks = document.querySelectorAll('.other-analyses-list a');
  
  otherAnalysisLinks.forEach(link => {
    link.addEventListener('mouseenter', function(e) {
      const card = this.querySelector('.analysis-item');
      if (card) {
        card.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        card.style.transform = 'translateX(8px) scale(1.02)';
        card.style.boxShadow = '0 12px 32px rgba(59, 130, 246, 0.2)';
      }
    });
    
    link.addEventListener('mouseleave', function(e) {
      const card = this.querySelector('.analysis-item');
      if (card && !card.classList.contains('current')) {
        card.style.transform = 'translateX(0) scale(1)';
        card.style.boxShadow = '';
      }
    });
  });
  
  // Smooth scroll vers l'analyse sélectionnée
  const currentAnalysis = document.querySelector('.analysis-item.current');
  if (currentAnalysis) {
    currentAnalysis.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
});

// ====================================================================
// LOG DE DÉMARRAGE
// ====================================================================
console.log('%c✨ NeuroScan Analytics Pro', 'font-size: 20px; font-weight: bold; color: #667eea;');
console.log('%cPage d\'analyse chargée avec succès', 'color: #10b981;');
console.log('%cVersion: 2.0.0 | Design System: Premium', 'color: #6b7280;');
