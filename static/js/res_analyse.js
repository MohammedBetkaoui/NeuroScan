// Données d'analyse pour JavaScript
window.__analysisData = JSON.parse(document.getElementById('analysis-json')?.textContent || '{}');

// Animation d'entrée des cartes
document.addEventListener('DOMContentLoaded', function() {
  const cards = document.querySelectorAll('.fade-in');
  cards.forEach((card, index) => {
    card.style.animationDelay = `${index * 0.1}s`;
  });
});

// Animation de la barre de confiance
(function(){
  const confFill = document.getElementById('confFill');
  if (confFill) {
    const width = parseFloat(confFill.dataset.width || '0');
    setTimeout(() => {
      confFill.style.width = Math.max(0, Math.min(100, width)) + '%';
    }, 500);
  }
})();

// Animation des barres de probabilités avec couleurs
(function(){
  const colors = ['#3b82f6','#7c3aed','#10b981','#f59e0b','#ef4444'];
  const probBars = document.querySelectorAll('.prob-bar-fill');
  
  probBars.forEach((bar, index) => {
    const width = parseFloat(bar.dataset.width || '0');
    const color = colors[index % colors.length];
    
    // Appliquer la couleur
    bar.style.backgroundColor = color;
    
    // Animation de largeur
    setTimeout(() => {
      bar.style.width = Math.max(0, Math.min(100, width)) + '%';
    }, 700 + (index * 100));
  });

  // Appliquer les couleurs aux indicateurs circulaires
  document.querySelectorAll('.w-4.h-4.rounded-full').forEach((circle, index) => {
    if (index < colors.length) {
      circle.style.backgroundColor = colors[index];
    }
  });
})();

// Graphique donut amélioré
(function(){
  try {
    const canvas = document.getElementById('probDonut');
    if (!canvas) return;
    
    const data = (window.__analysisData && window.__analysisData.probabilities) || {};
    const labels = Object.keys(data);
    const values = Object.values(data).map(v => Number(v) || 0);
    const colors = ['#3b82f6','#7c3aed','#10b981','#f59e0b','#ef4444'];
    
    new Chart(canvas.getContext('2d'), {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: colors.slice(0, labels.length),
          borderWidth: 0,
          cutout: '60%'
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
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: 'white',
            bodyColor: 'white',
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
            cornerRadius: 8,
            callbacks: {
              label: function(context) {
                return context.label + ': ' + context.parsed.toFixed(1) + '%';
              }
            }
          }
        },
        animation: {
          animateRotate: true,
          animateScale: true,
          duration: 1000,
          easing: 'easeOutCubic'
        }
      }
    });
  } catch (e) {
    console.error('Erreur lors de la création du graphique:', e);
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
        
        const result = await resp.json();
        
        if (result.success) {
          // Créer une notification de succès
          showNotification('Rapport PDF généré avec succès', 'success');
        } else {
          showNotification(result.error || 'Erreur lors de la génération du rapport', 'error');
        }
      } catch (e) {
        showNotification('Erreur réseau lors de la génération', 'error');
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

// Analyse avancée (Gemini)
(async function(){
  try {
    const analysisData = window.__analysisData || {};
    const payload = {
      predicted_label: analysisData.predicted_label,
      confidence: Number(analysisData.confidence) || 0,
      probabilities: Object.fromEntries(
        Object.entries(analysisData.probabilities || {})
          .map(([k, v]) => [k, Number(v) / 100])
      )
    };
    
    const response = await fetch('/api/advanced-analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const result = await response.json();
    const loading = document.getElementById('advLoading');
    const content = document.getElementById('advContent');
    
    if (!result.success) {
      if (loading) {
        loading.innerHTML = `
          <div class="flex items-center justify-center p-8">
            <div class="flex items-center gap-3 text-gray-500">
              <i class="fas fa-exclamation-triangle text-amber-500"></i>
              <span>Service d'analyse avancée temporairement indisponible</span>
            </div>
          </div>
        `;
      }
      return;
    }
    
    if (loading) loading.classList.add('hidden');
    if (content) content.classList.remove('hidden');
    
    if (result.data) {
      if (result.data.summary) {
        document.getElementById('advSummary').textContent = result.data.summary;
      }
      if (result.data.explanation) {
        document.getElementById('advExplanation').textContent = result.data.explanation;
      }
      
      const suggestionsList = document.getElementById('advSuggestions');
      suggestionsList.innerHTML = '';
      (result.data.suggestions || []).forEach(suggestion => {
        const li = document.createElement('li');
        li.className = 'flex items-start gap-3 p-3 bg-amber-50 rounded-12 border-l-4 border-amber-400';
        li.innerHTML = `
          <i class="fas fa-lightbulb text-amber-500 mt-1 flex-shrink-0"></i>
          <span class="text-gray-700">${suggestion}</span>
        `;
        suggestionsList.appendChild(li);
      });
    }
  } catch (e) {
    console.error('Erreur lors de l\'analyse avancée:', e);
    const loading = document.getElementById('advLoading');
    if (loading) {
      loading.innerHTML = `
        <div class="flex items-center justify-center p-8">
          <div class="flex items-center gap-3 text-red-500">
            <i class="fas fa-times-circle"></i>
            <span>Erreur lors du chargement de l'analyse avancée</span>
          </div>
        </div>
      `;
    }
  }
})();

// Fonction pour afficher des notifications
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `fixed top-4 right-4 z-50 p-4 rounded-16 shadow-lg max-w-sm transition-all duration-300 transform translate-x-full`;
  
  if (type === 'success') {
    notification.classList.add('bg-green-500', 'text-white');
    notification.innerHTML = `<i class="fas fa-check-circle mr-2"></i>${message}`;
  } else if (type === 'error') {
    notification.classList.add('bg-red-500', 'text-white');
    notification.innerHTML = `<i class="fas fa-exclamation-circle mr-2"></i>${message}`;
  } else {
    notification.classList.add('bg-blue-500', 'text-white');
    notification.innerHTML = `<i class="fas fa-info-circle mr-2"></i>${message}`;
  }
  
  document.body.appendChild(notification);
  
  // Animation d'entrée
  setTimeout(() => {
    notification.classList.remove('translate-x-full');
  }, 100);
  
  // Animation de sortie et suppression
  setTimeout(() => {
    notification.classList.add('translate-x-full');
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 300);
  }, 4000);
}

// Fonctions pour la gestion de l'image d'analyse
function openImageModal() {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  const img = document.getElementById('analysisImage');
  
  modal.style.display = 'flex';
  modalImg.src = img.src;
  modalImg.alt = img.alt;
  
  // Animation d'ouverture
  requestAnimationFrame(() => {
    modal.classList.add('opacity-100');
    modalImg.classList.add('scale-100');
  });
}

function closeImageModal() {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  
  modal.classList.remove('opacity-100');
  modalImg.classList.remove('scale-100');
  
  setTimeout(() => {
    modal.style.display = 'none';
  }, 300);
}

function toggleFullscreen() {
  openImageModal();
}

function downloadImage() {
  const img = document.getElementById('analysisImage');
  const link = document.createElement('a');
  link.download = 'analyse_{{ analysis.id }}_' + Date.now() + '.jpg';
  link.href = img.src;
  link.click();
}

function toggleImageInfo() {
  const overlay = document.getElementById('imageOverlay');
  overlay.style.transform = overlay.style.transform === 'translateY(0px)' ? 'translateY(100%)' : 'translateY(0px)';
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
}

function showImageError() {
  const container = document.querySelector('.image-container');
  const loading = document.getElementById('imageLoading');
  
  if (loading) loading.style.display = 'none';
  
  container.innerHTML += 
    '<div class="absolute inset-0 flex items-center justify-center bg-gray-100">' +
      '<div class="text-center text-gray-500">' +
        '<i class="fas fa-exclamation-triangle text-3xl mb-2"></i>' +
        '<div>Impossible de charger l\'image</div>' +
      '</div>' +
    '</div>';
}

// Fermeture de la modal avec Escape
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeImageModal();
  }
});

// Ajout de classes utilitaires CSS manquantes
const style = document.createElement('style');
style.textContent = `
  .rounded-12 { border-radius: 12px; }
  .rounded-16 { border-radius: 16px; }
`;
document.head.appendChild(style);
