let analysisResults = null;

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
    if (!allowed.includes(ext)) { alert('Format non supporté'); return; }
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    fileInfo.classList.remove('hidden');
    input.files = input.files;
    updateAnalyzeButton();
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
    statusText.textContent = 'Téléversement du fichier...';

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
        }
      };
      xhr.onload = () => { uploadOk = xhr.status === 200; uploadResponseText = xhr.responseText; resolve(); };
      xhr.onerror = () => resolve();
      xhr.send(formData);
    });

  statusText.textContent = uploadOk ? 'Analyse IA en cours...' : 'Analyse locale (démo)';
    progressBar.style.width = '100%';
    progressPct.textContent = '100%';

    let resData;
    if (uploadOk && uploadResponseText) {
      try { resData = JSON.parse(uploadResponseText); } catch (e) { /* ignore parse error */ }
    }

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

  // Fallback: conserver l'ancien affichage si aucun analysis_id (mode démo)
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
        alert(data.success ? 'Rapport généré' : 'Erreur génération rapport');
      } catch {
        alert('Erreur réseau');
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
        alert(data.success ? 'Analyse partagée' : 'Erreur partage');
      } catch {
        alert('Erreur réseau');
      }
    });
  }
})();