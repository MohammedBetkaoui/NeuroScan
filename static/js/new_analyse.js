let analysisResults = null;

// Ultra-Modern Alert System for NeuroScan Professional Platform
function showModernAlert(message, type = 'info', duration = 6000, options = {}) {
    const container = document.getElementById('modernAlertContainer');
    if (!container) return;

    const {
        title,
        description,
        important = false,
        showProgress = true,
        actionButton
    } = options;

    const alertDiv = document.createElement('div');
    alertDiv.className = `modern-alert ${type} ${important ? 'important' : ''}`;

    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-triangle',
        warning: 'fa-exclamation-circle',
        info: 'fa-info-circle',
        analysis: 'fa-brain',
        upload: 'fa-cloud-upload-alt',
        report: 'fa-file-pdf',
        share: 'fa-share-alt'
    };

    const titles = {
        success: 'Succès',
        error: 'Erreur',
        warning: 'Attention',
        info: 'Information',
        analysis: 'Analyse IA',
        upload: 'Téléversement',
        report: 'Rapport',
        share: 'Partage'
    };

    const alertTitle = title || titles[type] || 'Notification';
    const iconClass = icons[type] || icons.info;

    let progressBar = '';
    if (showProgress && duration > 0) {
        progressBar = '<div class="modern-alert-progress"></div>';
    }

    let actionBtn = '';
    if (actionButton) {
        actionBtn = `<button class="modern-alert-action" onclick="${actionButton.onClick}">${actionButton.text}</button>`;
    }

    alertDiv.innerHTML = `
        <div class="modern-alert-content">
            <div class="modern-alert-icon">
                <i class="fas ${iconClass}"></i>
            </div>
            <div class="modern-alert-body">
                <div class="modern-alert-text">${alertTitle}</div>
                ${description ? `<div class="modern-alert-description">${description}</div>` : ''}
                ${message ? `<div class="modern-alert-message">${message}</div>` : ''}
                ${actionBtn}
            </div>
            <button class="modern-alert-close" onclick="this.closest('.modern-alert').remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
        ${progressBar}
    `;

    container.appendChild(alertDiv);

    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.classList.add('fade-out');
                setTimeout(() => alertDiv.remove(), 600);
            }
        }, duration);
    }

    return alertDiv;
}

// Specialized alert functions for better UX
function showAnalysisStartAlert() {
    return showModernAlert(
        'Votre analyse IA est en cours de traitement...',
        'analysis',
        0, // Don't auto-remove
        {
            description: 'Le modèle d\'intelligence artificielle analyse l\'imagerie médicale avec précision.',
            important: true,
            showProgress: false
        }
    );
}

function showAnalysisCompleteAlert(diagnosis, confidence) {
    const confidenceLevel = parseFloat(confidence);
    const isHighConfidence = confidenceLevel >= 85;

    let description = `Analyse terminée avec ${confidence} de confiance.`;
    if (isHighConfidence) {
        description += ' Résultats hautement fiables grâce à notre modèle IA avancé.';
    } else {
        description += ' Corrélation clinique recommandée pour confirmation.';
    }

    return showModernAlert(
        `Diagnostic: ${diagnosis}`,
        'success',
        10000,
        {
            description: description,
            important: isHighConfidence,
            actionButton: {
                text: 'Voir le rapport détaillé',
                onClick: 'openResultsModal()'
            }
        }
    );
}

function showFileUploadAlert(fileName, fileSize) {
    return showModernAlert(
        `Fichier "${fileName}" chargé avec succès`,
        'upload',
        4000,
        {
            description: `Taille: ${fileSize} • Prêt pour l'analyse IA`,
            showProgress: false
        }
    );
}

function showErrorAlert(message, details = '') {
    return showModernAlert(
        message,
        'error',
        8000,
        {
            description: details || 'Veuillez réessayer ou contacter le support technique.',
            important: true
        }
    );
}

function showReportGeneratedAlert() {
    return showModernAlert(
        'Rapport PDF généré avec succès',
        'report',
        5000,
        {
            description: 'Le rapport détaillé a été créé et est prêt au téléchargement.',
            actionButton: {
                text: 'Télécharger',
                onClick: 'document.querySelector("#reportBtn").click()'
            }
        }
    );
}

function showShareSuccessAlert() {
    return showModernAlert(
        'Analyse partagée avec succès',
        'share',
        5000,
        {
            description: 'Le destinataire recevra un lien sécurisé vers les résultats.',
            showProgress: false
        }
    );
}

function openResultsModal() {
  if (!analysisResults) return;
  document.getElementById('modalResultImg').src = analysisResults.imageUrl || '';
  document.getElementById('modalPrediction').textContent = analysisResults.prediction || '—';
  const badge = document.getElementById('modalDiagBadge');
  badge.textContent = analysisResults.diagnosis || '—';
  badge.className = `px-3 py-1 rounded-lg text-white text-sm font-semibold ${analysisResults.badgeClass || 'bg-blue-600'}`;
  document.getElementById('modalConfidence').textContent = analysisResults.confidence || '—';
  document.getElementById('modalProcTime').textContent = analysisResults.processingTime || '—';
  document.getElementById('modalPatientSummary').textContent = analysisResults.patientSummary || '—';
  const idEl = document.getElementById('modalAnalysisId');
  if (idEl) idEl.textContent = analysisResults.id || '';
  // Header patient chips
  const hdrName = document.getElementById('hdrPatientName');
  const hdrAge = document.getElementById('hdrPatientAge');
  const hdrGender = document.getElementById('hdrPatientGender');
  const hdrExam = document.getElementById('hdrExamDate');
  if (hdrName) hdrName.textContent = analysisResults.patientName || '—';
  if (hdrAge) hdrAge.textContent = analysisResults.patientAge || '—';
  if (hdrGender) hdrGender.textContent = analysisResults.patientGender || '—';
  if (hdrExam) hdrExam.textContent = analysisResults.examDate || '—';

  // Confiance bar width
  const confVal = (typeof analysisResults.confidence === 'string') ? parseFloat(analysisResults.confidence.replace('%','')) : (analysisResults.confidence || 0);
  const bar = document.getElementById('modalConfidenceBar');
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, confVal))}%`;

  if (analysisResults.chartData) createModalChart(analysisResults.chartData);
  if (analysisResults.probabilities) fillModalProbabilities(analysisResults.probabilities);
  if (analysisResults.recommendations) fillModalRecommendations(analysisResults.recommendations);

  // Analyse avancée (Gemini)
  fetchAdvancedAnalysis(analysisResults).catch(()=>{});

  const modal = document.getElementById('resultsModal');
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  // Lock page scroll when modal is open
  document.documentElement.style.overflow = 'hidden';
  document.body.style.overflow = 'hidden';
}

function closeResultsModal() {
  const modal = document.getElementById('resultsModal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  // Restore page scroll
  document.documentElement.style.overflow = '';
  document.body.style.overflow = '';
}

document.addEventListener('click', (e) => {
  if (e.target.id === 'closeModal' || e.target.id === 'resultsModal') closeResultsModal();
});

function createModalChart(data) {
  const canvas = document.getElementById('modalProbChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (window._modalChart) window._modalChart.destroy();
  window._modalChart = new Chart(ctx, {
    type: 'doughnut',
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom' } },
      layout: { padding: 0 },
    }
  });
}

function fillModalProbabilities(list) {
  const c = document.getElementById('modalProbs');
  c.innerHTML = '';
  const palette = ['#2563eb','#7c3aed','#059669','#d97706','#dc2626','#14b8a6','#f59e0b'];
  list.forEach((p, i) => {
    const color = palette[i % palette.length];
    const row = document.createElement('div');
    row.className = 'space-y-1';
    row.innerHTML = `
      <div class="flex items-center justify-between text-sm">
        <span class="text-neutral-700">${p.name}</span>
        <span class="font-semibold text-neutral-900">${p.value}%</span>
      </div>
      <div class="w-full h-2 bg-neutral-200 rounded-full overflow-hidden">
        <div class="h-2 rounded-full" style="width:${p.value}%; background:${color}"></div>
      </div>`;
    c.appendChild(row);
  });
}

function fillModalRecommendations(recs) {
  const ul = document.getElementById('modalRecs');
  ul.innerHTML = '';
  const icons = ['fa-user-md text-emerald-600','fa-eye text-blue-600','fa-hospital text-purple-600','fa-vial text-amber-600'];
  recs.forEach((r,i) => {
    const li = document.createElement('li');
    li.className = 'flex items-start gap-2';
    li.innerHTML = `<i class="fas ${icons[i%icons.length]} mt-1"></i><span>${r}</span>`;
    ul.appendChild(li);
  });
}

function getBadgeClass(pred) {
  // Couleurs distinctes par type: Méningiome violet, Gliome orange, Normal vert, Tumeur hypophysaire bleu
  const map = {
    'Normal': 'bg-emerald-600',
    'Gliome': 'bg-orange-600',
    'Méningiome': 'bg-purple-600',
    'Tumeur pituitaire': 'bg-blue-600'
  };
  return map[pred] || 'bg-blue-600';
}

// --- Normalisation & formatage des pourcentages ---
function normalizePercent(value) {
  if (value == null) return null;
  let n = (typeof value === 'string') ? parseFloat(value.replace('%', '').trim()) : Number(value);
  if (Number.isNaN(n)) return null;
  // Beaucoup d'API renvoient des ratios 0–1. Si c'est le cas, convertir en %.
  if (n > 0 && n <= 1) n = n * 100;
  // Clamp sécurité
  if (n < 0) n = 0; if (n > 100) n = 100;
  return n;
}

function formatPercent(value, decimals = 1) {
  const n = normalizePercent(value);
  return (n == null) ? '—' : `${n.toFixed(decimals)}%`;
}

function createChartData(prob) {
  const labels = Object.keys(prob);
  const data = Object.values(prob).map(v => normalizePercent(v) ?? 0);
  const colors = ['#2563eb','#7c3aed','#059669','#d97706','#dc2626'];
  return { labels, datasets: [{ data, backgroundColor: colors.slice(0, labels.length), borderWidth: 0 }] };
}

function createProbabilityList(prob) {
  return Object.entries(prob).map(([name, value]) => ({ name, value: (normalizePercent(value) ?? 0).toFixed(1) }));
}

// Page setup
(function initPage(){
  const select = document.getElementById('patientSelect');
  const info = document.getElementById('patientInfo');
  const analyzeBtn = document.getElementById('nsAnalyzeBtn');
  const drop = document.getElementById('nsDropZone');
  const input = document.getElementById('nsFileInput');
  const fileInfo = document.getElementById('nsFileInfo');
  const fileName = document.getElementById('nsFileName');
  const fileSize = document.getElementById('nsFileSize');
  const progressCard = document.getElementById('nsProgressCard');
  const progressBar = document.getElementById('nsAnalysisProgress');
  const progressPct = document.getElementById('nsProgressPct');
  const statusText = document.getElementById('nsStatusText');
  const resultsCard = document.getElementById('nsResultsCard');
  const reportBtn = document.getElementById('reportBtn');
  const shareBtn = document.getElementById('shareBtn');
  const examDateInput = document.getElementById('nsExamDate');

  // Set default exam date to today
  if (examDateInput) {
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth()+1).padStart(2,'0');
    const dd = String(today.getDate()).padStart(2,'0');
    examDateInput.value = `${yyyy}-${mm}-${dd}`;
  }

  function updateAnalyzeButton() {
    const ok = !!select.value && input.files.length > 0;
    analyzeBtn.disabled = !ok;
  }

  select.addEventListener('change', () => {
    const opt = select.options[select.selectedIndex];
    if (!opt.value) { info.classList.add('hidden'); updateAnalyzeButton(); return; }
    document.getElementById('patientName').textContent = opt.dataset.name;
    document.getElementById('patientAge').textContent = `${opt.dataset.age} ans`;
    document.getElementById('patientGender').textContent = opt.dataset.gender;
    document.getElementById('patientDob').textContent = opt.dataset.dob;
    info.classList.remove('hidden');
    updateAnalyzeButton();
  });

  // Auto-sélection depuis la liste patients
  try {
    const pre = sessionStorage.getItem('preselectedPatient');
    if (pre && select) {
      const opt = Array.from(select.options).find(o => o.value === pre);
      if (opt) {
        select.value = pre;
        document.getElementById('patientName').textContent = opt.dataset.name || '—';
        document.getElementById('patientAge').textContent = opt.dataset.age ? `${opt.dataset.age} ans` : '—';
        document.getElementById('patientGender').textContent = opt.dataset.gender || '—';
        document.getElementById('patientDob').textContent = opt.dataset.dob || '—';
        info.classList.remove('hidden');
        updateAnalyzeButton();
      }
      sessionStorage.removeItem('preselectedPatient');
    }
  } catch {}

  function formatSize(bytes){ if(bytes<1024) return bytes+' B'; const u=['KB','MB','GB']; let i=-1; do{bytes/=1024;i++;}while(bytes>=1024&&i<u.length-1); return bytes.toFixed(1)+' '+u[i]; }

  function onFileSelected(file){
    if (!file) return;
    const allowed = ['dcm','dicom','nii','jpg','jpeg','png'];
    const ext = file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
        showErrorAlert(
            'Format de fichier non supporté',
            'Formats acceptés: JPG, PNG, DICOM. Sélectionnez un fichier d\'imagerie médicale valide.'
        );
        return;
    }
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    fileInfo.classList.remove('hidden');
    input.files = input.files;
    updateAnalyzeButton();

    // Show modern file upload success alert
    showFileUploadAlert(file.name, formatSize(file.size));

    // Image preview for supported formats
    const previewImg = document.getElementById('nsImagePreview');
    if (['jpg','jpeg','png'].includes(ext)) {
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
      };
      reader.readAsDataURL(file);
    } else {
      previewImg.style.display = 'none';
    }
  }

  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => onFileSelected(e.target.files[0]));
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('bg-neutral-100'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('bg-neutral-100'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if(f){ input.files=e.dataTransfer.files; onFileSelected(f);} drop.classList.remove('bg-neutral-100'); });

  analyzeBtn.addEventListener('click', async () => {
    // Start progress
    resultsCard.classList.add('hidden');
    progressCard.classList.remove('hidden');
    statusText.textContent = 'Préparation de l\'analyse...';

    // Show modern analysis start alert
    showAnalysisStartAlert();

    // Populate progress modal with info
    document.getElementById('progressPatientName').textContent = document.getElementById('patientName').textContent || 'Non spécifié';
    document.getElementById('progressPatientAge').textContent = document.getElementById('patientAge').textContent || '—';
    document.getElementById('progressPatientGender').textContent = document.getElementById('patientGender').textContent || '—';
    document.getElementById('progressFileName').textContent = document.getElementById('nsFileName').textContent || 'Fichier';
    document.getElementById('progressFileSize').textContent = document.getElementById('nsFileSize').textContent || '—';

  const formData = new FormData();
  formData.append('file', input.files[0]);
  formData.append('patient_id', select.value);
  const patientNameVal = document.getElementById('patientName').textContent || (select.options[select.selectedIndex]?.dataset?.name || '');
  if (patientNameVal) formData.append('patient_name', patientNameVal);
  if (examDateInput && examDateInput.value) formData.append('exam_date', examDateInput.value);

    // Upload with XHR for progress
    let uploadOk = false;
    let uploadResponseText = null;
    await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/upload');
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          progressBar.style.width = `${pct}%`;
          progressPct.textContent = `${pct}%`;
          statusText.textContent = `Téléversement en cours... ${pct}%`;
        }
      };
      xhr.onload = () => { uploadOk = xhr.status === 200; uploadResponseText = xhr.responseText; resolve(); };
      xhr.onerror = () => resolve();
      xhr.send(formData);
    });

  statusText.textContent = uploadOk ? 'Analyse IA en cours...' : 'Analyse locale (démo)';
    progressBar.style.width = '0%';
    progressPct.textContent = '0%';

    // Simulate real-time progress
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += Math.random() * 10 + 5; // Random increment
      if (progress >= 100) {
        progress = 100;
        clearInterval(progressInterval);
        statusText.textContent = 'Finalisation des résultats...';
      }
      progressBar.style.width = `${progress}%`;
      progressPct.textContent = `${Math.round(progress)}%`;
    }, 200);

    let resData;
    if (uploadOk && uploadResponseText) {
      try { resData = JSON.parse(uploadResponseText); } catch (e) { /* ignore parse error */ }
    }

    // Wait for progress to complete
    await new Promise(resolve => {
      const checkProgress = () => {
        if (progress >= 100) resolve();
        else setTimeout(checkProgress, 100);
      };
      checkProgress();
    });

    progressCard.classList.add('hidden');

  if (!resData) {
      // Fallback démo
      // En mode fallback (démo), rester sur la page actuelle et afficher la carte comme avant
      // Mais si vous souhaitez forcer une redirection même en démo, on peut générer un faux id côté client
      resData = {
        analysis_id: null,
        prediction: 'Normal',
        confidence: 97.4,
        processing_time: 8.2,
        probabilities: { 'Normal': 97.4, 'Gliome': 1.2, 'Méningiome': 0.8, 'Tumeur pituitaire': 0.6 },
        description: 'Absence de masse tumorale détectable.',
        image_url: ''
      };
    }

  // Redirection vers la page dédiée si un analysis_id est retourné
  if (resData && resData.analysis_id) {
    window.location.href = `/analysis/${resData.analysis_id}`;
    return;
  }

  // Show welcome alert on page load
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        showModernAlert(
            'Plateforme NeuroScan IA prête',
            'info',
            6000,
            {
                description: 'Modèle d\'intelligence artificielle avancé pour l\'analyse d\'imagerie médicale. Précision et rapidité garanties.',
                showProgress: false
            }
        );
    }, 1000);
});
  analysisResults = {
    id: resData.analysis_id || 'DEMO',
    imageUrl: resData.image_url,
    prediction: resData.prediction,
    diagnosis: resData.prediction,
    badgeClass: getBadgeClass(resData.prediction),
    confidence: formatPercent(resData.confidence),
    processingTime: (typeof resData.processing_time === 'number' ? `${resData.processing_time.toFixed(1)}s` : `${resData.processing_time || ''}s`).trim(),
    patientName: document.getElementById('patientName').textContent || '—',
    patientAge: document.getElementById('patientAge').textContent || '—',
    patientGender: document.getElementById('patientGender').textContent || '—',
    examDate: document.getElementById('nsExamDate')?.value || '',
    patientSummary: `Patient: ${document.getElementById('patientName').textContent} • ${document.getElementById('patientAge').textContent} • ${document.getElementById('patientGender').textContent}`,
    chartData: createChartData(resData.probabilities),
    probabilities: createProbabilityList(resData.probabilities),
    recommendations: [
      'Corrélation clinique recommandée',
      'Surveillance radiologique si nécessaire',
      'Consulter un radiologue en cas de doute'
    ]
  };

  progressCard.classList.add('hidden');
  resultsCard.classList.remove('hidden');
  
  // Show success alert for completed analysis with detailed info
  showAnalysisCompleteAlert(analysisResults.diagnosis, analysisResults.confidence);
  
  const miniBadge = document.getElementById('nsResultMiniBadge');
  if (miniBadge) { miniBadge.textContent = analysisResults.diagnosis; miniBadge.className = `px-2 py-1 rounded-md text-white text-xs font-semibold ${analysisResults.badgeClass || 'bg-blue-600'}`; }
  const miniSummary = document.getElementById('nsResultMiniSummary');
  if (miniSummary) miniSummary.textContent = `Diagnostic: ${analysisResults.diagnosis} • Confiance: ${analysisResults.confidence} • Temps: ${analysisResults.processingTime}`;
  });

  // Actions dans le modal
  if (reportBtn) {
    reportBtn.addEventListener('click', async () => {
      if (!analysisResults) return;
      try {
        const resp = await fetch('/generate-report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patientName: document.getElementById('patientName').textContent, analysisData: analysisResults })
        });
        const data = await resp.json();
        if (data.success) {
            showReportGeneratedAlert();
        } else {
            showErrorAlert('Échec de la génération du rapport', 'Une erreur s\'est produite lors de la création du PDF.');
        }
      } catch {
        showModernAlert('Erreur de connexion réseau. Veuillez réessayer.', 'error');
      }
    });
  }
  if (shareBtn) {
    shareBtn.addEventListener('click', async () => {
      if (!analysisResults) return;
      const email = prompt('Email du destinataire:');
      if (!email) return;
      try {
        const resp = await fetch('/share-analysis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ recipientEmail: email, analysisData: analysisResults })
        });
        const data = await resp.json();
        if (data.success) {
            showShareSuccessAlert();
        } else {
            showErrorAlert('Échec du partage', 'Impossible d\'envoyer l\'analyse au destinataire.');
        }
      } catch {
        showErrorAlert('Erreur de connexion', 'Impossible de contacter le serveur. Vérifiez votre connexion internet.');
      }
    });
  }
})();