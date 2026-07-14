function handleDrag(e, over) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.toggle('dragover', over);
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) showImagePreview(f);
}

function handleFileSelect(e) {
  const f = e.target.files[0];
  if (f) showImagePreview(f);
}

function showImagePreview(file) {
  state.originalImageBlob = file;
  state.enhancedImageBlob = null;

  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('previewImg');
    img.src = e.target.result;
    img.style.filter = '';
    document.getElementById('imagePreviewWrap').style.display = 'block';
    document.getElementById('inputChoiceRow').style.display = 'none';
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('enhancementPanel').classList.add('show');
    resetEnhancementSliders();
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  document.getElementById('imageInput').value = '';
  document.getElementById('imagePreviewWrap').style.display = 'none';
  document.getElementById('enhancementPanel').classList.remove('show');
  document.getElementById('translated-image-container').style.display = 'none';
  document.getElementById('inputChoiceRow').style.display = 'grid';
  document.getElementById('uploadZone').style.display = 'block';
  state.originalImageBlob = null;
  state.enhancedImageBlob = null;
  hideResults();
}

async function submitImage() {
  const fileToSend = state.enhancedImageBlob || state.originalImageBlob
    || document.getElementById('imageInput').files[0];

  if (!fileToSend) {
    showStatus('error', 'Please select or capture an image first');
    return;
  }

  const btn = document.getElementById('imageSubmitBtn');
  setLoading(btn, true, 'image');
  showStatus('processing', 'Running OCR -> translating -> generating image overlay...');
  hideResults();
  state.processingStart = Date.now();

  const fd = new FormData();
  fd.append('file', fileToSend);
  fd.append('target_lang', state.langs.image.tgt);
  fd.append('src_lang', state.langs.image.src);
  await callAPI('/translate/image', fd, btn, 'image', true);
}
