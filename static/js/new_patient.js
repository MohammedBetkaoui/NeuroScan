  document.addEventListener('DOMContentLoaded', () => {
    // Proposer un ID automatiquement
    fetch('/api/patients/next-id')
      .then(r => r.ok ? r.json() : null)
      .then(j => { if (j && j.success && j.next_id) { const el = document.getElementById('patientId'); if (!el.value) el.placeholder = j.next_id; }});

    // Bouton Générer ID
    const genBtn = document.getElementById('genIdBtn');
    if (genBtn) {
      genBtn.addEventListener('click', async () => {
        try {
          const r = await fetch('/api/patients/next-id');
          const j = await r.json();
          if (j.success && j.next_id) {
            const el = document.getElementById('patientId');
            el.value = j.next_id;
            const help = document.getElementById('patientIdHelp');
            if (help) { help.textContent = 'ID proposé automatiquement'; help.className = 'text-xs mt-1 text-gray-500'; }
          }
        } catch {}
      });
    }

    // Vérif ID dispo en temps réel + statut
    const pid = document.getElementById('patientId');
    const help = document.getElementById('patientIdHelp');
    const status = document.getElementById('patientIdStatus');
    pid.addEventListener('input', debounce(async () => {
      const v = (pid.value || '').trim();
      pid.classList.remove('is-valid','is-invalid');
      status.textContent = '';
      status.className = 'status-badge';
      if (!v) { help.textContent = 'Sera généré automatiquement si vide.'; help.className = 'help'; return; }
      try {
        const res = await fetch(`/api/patients/check-id/${encodeURIComponent(v)}`);
        const j = await res.json();
        if (j.success && j.available) {
          help.textContent = 'ID disponible'; help.className = 'help ok';
          pid.classList.add('is-valid');
          status.textContent = 'Disponible'; status.classList.add('status-ok');
        } else {
          help.textContent = 'ID déjà utilisé'; help.className = 'help err';
          pid.classList.add('is-invalid');
          status.textContent = 'Indisponible'; status.classList.add('status-err');
        }
      } catch { help.textContent = 'Vérification impossible'; help.className = 'help'; }
    }, 300));

    // Limiter date de naissance au présent
    const dob = document.getElementById('dateOfBirth');
    if (dob) {
      const today = new Date();
      const m = String(today.getMonth()+1).padStart(2,'0');
      const d = String(today.getDate()).padStart(2,'0');
      dob.max = `${today.getFullYear()}-${m}-${d}`;
      dob.addEventListener('change', () => {
        const v = dob.value; const helpEl = document.getElementById('dobHelp');
        if (!v) { helpEl.textContent=''; dob.classList.remove('is-invalid','is-valid'); return; }
        const dt = new Date(v);
        if (isNaN(dt.getTime()) || dt > today) { helpEl.textContent='Date invalide (future)'; helpEl.className='help err'; dob.classList.add('is-invalid'); }
        else { helpEl.textContent=''; helpEl.className='help'; dob.classList.add('is-valid'); dob.classList.remove('is-invalid'); }
      });
    }

    // Validation email/phone
    const email = document.getElementById('email');
    const phone = document.getElementById('phone');
    email.addEventListener('input', () => validateEmail(email));
    phone.addEventListener('input', () => validatePhone(phone));

    // Réinitialiser le formulaire
    document.getElementById('resetBtn').addEventListener('click', () => document.getElementById('newPatientForm').reset());

    // Soumission
    document.getElementById('newPatientForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('submitBtn');
      // Validation client légère
      const nameEl = document.getElementById('patientName');
      if (!nameEl.value.trim()) { nameEl.classList.add('is-invalid'); nameEl.focus(); showNotification('Veuillez renseigner le nom du patient', 'error'); return; }
      if (!validateEmail(email)) { email.focus(); return; }
      if (!validatePhone(phone)) { phone.focus(); return; }
      // Loading state
      const prevHTML = submitBtn.innerHTML; submitBtn.disabled = true; submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Enregistrement…';
      const formData = new FormData(e.target);
      const data = Object.fromEntries(formData.entries());
      data.patientName = data.patientName || data.patient_name || data.patientName;
      data.patientId = (data.patientId || data.patient_id || '').trim();
      if (!data.patientId) delete data.patientId;
      try {
        const res = await fetch('/api/patients', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
        const json = await res.json();
        if (json.success) {
          const pid = json.patient_id ? ` (ID: ${json.patient_id})` : '';
          showNotification(`Patient ajouté avec succès${pid}`, 'success');
          window.location.href = "{{ url_for('patients_list') }}";
        } else {
          showNotification(json.error || 'Erreur lors de l\'ajout', 'error');
          if ((json.error || '').toLowerCase().includes('existe déjà')) { help.textContent = 'ID déjà utilisé — choisissez un autre ID ou laissez vide pour auto.'; help.className = 'text-xs mt-1 text-red-600'; }
        }
      } catch (err) {
        console.error(err);
        showNotification('Erreur de connexion', 'error');
      } finally { submitBtn.disabled = false; submitBtn.innerHTML = prevHTML; }
    });

    // Compteurs de caractères
    bindCounter('medicalHistory','medicalHistoryCounter');
    bindCounter('currentMedications','currentMedicationsCounter');
    bindCounter('notes','notesCounter');
  });

  function debounce(fn, wait) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn.apply(this, a), wait); }; }

  function validateEmail(el) {
    const helpEl = document.getElementById('emailHelp');
    const v = (el.value || '').trim();
    el.classList.remove('is-valid','is-invalid'); helpEl.textContent=''; helpEl.className='help';
    if (!v) return true; // facultatif
    const ok = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(v);
    if (!ok) { helpEl.textContent='Format email invalide'; helpEl.classList.add('err'); el.classList.add('is-invalid'); return false; }
    el.classList.add('is-valid'); return true;
  }

  function validatePhone(el) {
    const helpEl = document.getElementById('phoneHelp');
    const v = (el.value || '').replace(/\s+/g,'');
    el.classList.remove('is-valid','is-invalid'); helpEl.textContent=''; helpEl.className='help';
    if (!v) return true; // facultatif
    const digits = v.replace(/[^0-9+]/g,'');
    const ok = /^(\+)?[0-9]{8,15}$/.test(digits);
    if (!ok) { helpEl.textContent='Numéro invalide (min 8 chiffres)'; helpEl.classList.add('err'); el.classList.add('is-invalid'); return false; }
    el.classList.add('is-valid'); return true;
  }

  function bindCounter(textareaId, counterId) {
    const ta = document.getElementById(textareaId);
    const ctr = document.getElementById(counterId);
    if (!ta || !ctr) return;
    const max = parseInt(ta.dataset.max || ta.getAttribute('maxlength') || '1000', 10);
    const update = () => { ctr.textContent = `${ta.value.length}/${max}`; };
    ta.addEventListener('input', update);
    update();
  }
