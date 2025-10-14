const patientId = document.getElementById('editPage').dataset.patientId;

function toInputDate(value) {
  if (!value) return '';
  try {
    // Accept already formatted YYYY-MM-DD
    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return value;
    const d = new Date(value);
    if (isNaN(d.getTime())) return '';
    const m = String(d.getMonth()+1).padStart(2,'0');
    const day = String(d.getDate()).padStart(2,'0');
    return `${d.getFullYear()}-${m}-${day}`;
  } catch { return ''; }
}

async function loadPatient() {
  try {
    const res = await fetch(`/api/patients/${patientId}/details`);
    const data = await res.json();
    if (!data.success) {
      showNotification(data.error || 'Patient introuvable', 'error');
      return;
    }
    const p = data.data || {};
    const keys = [
      'patient_name','date_of_birth','gender','phone','email','address',
      'emergency_contact_name','emergency_contact_phone','medical_history',
      'allergies','current_medications','notes'
    ];
    for (const k of keys) {
      const el = document.getElementById(k);
      if (!el) continue;
      let val = p[k] || '';
      if (k === 'date_of_birth') {
        val = toInputDate(val);
      }
      el.value = val;
    }
  } catch (e) {
    console.error(e);
    showNotification('Erreur de chargement', 'error');
  }
}

async function submitEdit(e) {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData(form);
  const payload = Object.fromEntries(fd.entries());
  // Validation légère côté client
  const nameEl = document.getElementById('patient_name');
  const emailEl = document.getElementById('email');
  const phoneEl = document.getElementById('phone');
  if (!nameEl.value.trim()) { nameEl.classList.add('is-invalid'); nameEl.focus(); showNotification('Veuillez renseigner le nom du patient', 'error'); return; }
  if (!validateEmail(emailEl)) { emailEl.focus(); return; }
  if (!validatePhone(phoneEl)) { phoneEl.focus(); return; }
  const submitBtn = document.getElementById('submitBtn');
  const prevHTML = submitBtn.innerHTML; submitBtn.disabled = true; submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Enregistrement…';
  try {
    const res = await fetch(`/api/patients/${patientId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const out = await res.json();
    if (out.success) {
      showNotification(out.message || 'Patient mis à jour', 'success');
      window.location.href = `/patient/${patientId}`;
    } else {
      showNotification(out.error || 'Échec de la mise à jour', 'error');
    }
  } catch (e) {
    console.error(e);
    showNotification('Erreur de connexion', 'error');
  } finally { submitBtn.disabled = false; submitBtn.innerHTML = prevHTML; }
}

document.addEventListener('DOMContentLoaded', () => {
  loadPatient();
  document.getElementById('editPatientForm').addEventListener('submit', submitEdit);
  // Contraindre la date max au présent
  const dob = document.getElementById('date_of_birth');
  if (dob) {
    const today = new Date();
    const m = String(today.getMonth()+1).padStart(2,'0');
    const d = String(today.getDate()).padStart(2,'0');
    dob.max = `${today.getFullYear()}-${m}-${d}`;
    dob.addEventListener('change', () => {
      const v = dob.value; const helpEl = document.getElementById('dobHelp');
      if (!v) { helpEl.textContent=''; dob.classList.remove('is-invalid','is-valid'); return; }
      const dt = new Date(v);
      if (isNaN(dt.getTime()) || dt > today) { helpEl.textContent='Date invalide (future)'; helpEl.classList.add('err'); dob.classList.add('is-invalid'); }
      else { helpEl.textContent=''; helpEl.classList.remove('err'); dob.classList.add('is-valid'); dob.classList.remove('is-invalid'); }
    });
  }
  // Validation live email/phone
  const email = document.getElementById('email');
  const phone = document.getElementById('phone');
  email.addEventListener('input', () => validateEmail(email));
  phone.addEventListener('input', () => validatePhone(phone));
  // Compteurs
  bindCounter('medical_history','medicalHistoryCounter');
  bindCounter('current_medications','currentMedicationsCounter');
  bindCounter('notes','notesCounter');
});

function validateEmail(el) {
  const helpEl = document.getElementById('emailHelp');
  const v = (el.value || '').trim();
  el.classList.remove('is-valid','is-invalid'); helpEl.textContent=''; helpEl.className='help';
  if (!v) return true; // optionnel
  const ok = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(v);
  if (!ok) { helpEl.textContent='Format email invalide'; helpEl.classList.add('err'); el.classList.add('is-invalid'); return false; }
  el.classList.add('is-valid'); return true;
}

function validatePhone(el) {
  const helpEl = document.getElementById('phoneHelp');
  const v = (el.value || '').replace(/\s+/g,'');
  el.classList.remove('is-valid','is-invalid'); helpEl.textContent=''; helpEl.className='help';
  if (!v) return true; // optionnel
  const digits = v.replace(/[^0-9+]/g,'');
  const ok = /^(\+)?[0-9]{8,15}$/.test(digits);
  if (!ok) { helpEl.textContent='Numéro invalide (min 8 chiffres)'; helpEl.classList.add('err'); el.classList.add('is-invalid'); return false; }
  el.classList.add('is-valid'); return true;
}

function bindCounter(fieldId, counterId) {
  const el = document.getElementById(fieldId);
  const ctr = document.getElementById(counterId);
  if (!el || !ctr) return;
  const max = parseInt(el.dataset.max || el.getAttribute('maxlength') || '1000', 10);
  const update = () => { ctr.textContent = `${el.value.length}/${max}`; };
  el.addEventListener('input', update);
  update();
}
