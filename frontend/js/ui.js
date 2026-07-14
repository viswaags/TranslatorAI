function updateCamTip(w, h) {
  const mp = ((w * h) / 1000000).toFixed(1);
  const el = document.getElementById('camTipText');

  if (w < 1280) {
    el.textContent = ` Low res (${w}x${h}). For best OCR, ensure good lighting and hold steady.`;
    el.style.color = 'var(--accent4)';
  } else {
    el.textContent = `${mp}MP capture ready. Align text within guides, tap video to focus.`;
    el.style.color = '';
  }
}

function handleVideoTap(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const fi = document.getElementById('focusIndicator');

  fi.style.left = x + 'px';
  fi.style.top = y + 'px';
  fi.style.display = 'block';
  fi.style.animation = 'none';
  fi.offsetHeight;
  fi.style.animation = 'focus-anim 0.4s ease forwards';
  setTimeout(() => fi.style.display = 'none', 1200);

  const track = state.cameraStream?.getVideoTracks()[0];
  if (track && typeof ImageCapture !== 'undefined') {
    try {
      const ic = new ImageCapture(track);
      const xFrac = x / rect.width;
      const yFrac = y / rect.height;

      ic.getPhotoCapabilities().then(cap => {
        if (cap.focusMode && cap.focusMode.includes('manual')) {
          track.applyConstraints({
            advanced: [{ pointOfInterest: { x: xFrac, y: yFrac }, focusMode: 'single-shot' }],
          });
        }
      }).catch(() => {});
    } catch (_) {}
  }
}

function switchTab(tab, btn) {
  state.activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');

  const titles = {
    image: ['Document OCR & Translation', 'Tesseract + IndicTrans2', 'badge-blue'],
    speech: ['Speech Transcription & Translation', 'Whisper + IndicTrans2', 'badge-purple'],
    text: ['Text Translation', 'IndicTrans2 INT8', 'badge-green'],
  };
  const [title, badge, cls] = titles[tab];
  const badgeEl = document.getElementById('cardBadge');

  document.getElementById('cardTitle').textContent = title;
  badgeEl.textContent = badge;
  badgeEl.className = 'card-badge ' + cls;
  hideResults();
}

const BTN_LABELS = {
  image: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Process & Translate Document',
  speech: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Transcribe & Translate',
  text: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Translate Text',
};

function setLoading(btn, loading, tabKey) {
  if (loading) {
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Processing...';
  } else {
    btn.disabled = false;
    btn.innerHTML = BTN_LABELS[tabKey] || 'Submit';
  }
}

function showStatus(type, msg) {
  const bar = document.getElementById('statusBar');
  const spinner = document.getElementById('statusSpinner');

  bar.className = 'status-bar show ' + type;
  document.getElementById('statusMsg').textContent = msg;
  spinner.style.display = type === 'processing' ? 'block' : 'none';
}

function showResults(result, elapsed, isImage) {
  document.getElementById('extractedText').textContent = result.original_text || '-';
  document.getElementById('translatedText').textContent = result.translated_text || '-';

  const audioSection = document.getElementById('audioOutput');
  if (result.audio_url) {
    audioSection.classList.add('show');
    document.getElementById('audioPlayer').src = result.audio_url + '?t=' + Date.now();
  } else {
    audioSection.classList.remove('show');
  }

  const imgContainer = document.getElementById('translated-image-container');
  const imgEl = document.getElementById('backend-image-result');
  if (isImage && result.translated_image_url) {
    imgEl.src = result.translated_image_url + '?t=' + Date.now();
    imgContainer.style.display = 'block';
    const dlBtn = document.getElementById('downloadImgBtn');
    if (dlBtn) dlBtn.style.display = 'inline-block';
    setTimeout(() => imgContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 200);
  } else {
    imgContainer.style.display = 'none';
  }

  document.getElementById('statsRow').innerHTML = [
    { label: 'Time', value: elapsed + 's' },
    { label: 'Input', value: result.input_type || state.activeTab },
    { label: 'Detected', value: result.detected_lang || 'auto' },
    { label: 'Characters', value: (result.original_text || '').length },
  ].map(s => `<div class="stat-chip"><b>${s.value}</b> ${s.label}</div>`).join('');

  document.getElementById('resultsArea').style.display = 'block';
}

function hideResults() {
  document.getElementById('resultsArea').style.display = 'none';
  const dlBtn = document.getElementById('downloadImgBtn');
  if (dlBtn) dlBtn.style.display = 'none';
}

function downloadTranslatedImage() {
  const img = document.getElementById('backend-image-result');
  if (!img || !img.src || img.src === window.location.href) return;

  const a = document.createElement('a');
  a.href = img.src;
  a.download = 'translated_' + Date.now() + '.jpg';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function copyText(id) {
  navigator.clipboard.writeText(document.getElementById(id).textContent)
    .then(() => showStatus('success', 'Copied to clipboard'));
}
