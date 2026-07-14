function resetEnhancementSliders() {
  document.getElementById('brightSlider').value = 100;
  document.getElementById('contrastSlider').value = 100;
  document.getElementById('sharpSlider').value = 0;
  document.getElementById('graySlider').value = 0;
  updateEnhancementLabels();
  document.getElementById('previewImg').style.filter = '';
}

function resetEnhancement() {
  state.enhancedImageBlob = null;
  resetEnhancementSliders();
}

function updateEnhancementLabels() {
  document.getElementById('brightVal').textContent = document.getElementById('brightSlider').value;
  document.getElementById('contrastVal').textContent = document.getElementById('contrastSlider').value;
  document.getElementById('sharpVal').textContent = document.getElementById('sharpSlider').value;
  document.getElementById('grayVal').textContent = document.getElementById('graySlider').value;
}

function updateEnhancement() {
  updateEnhancementLabels();

  const b = document.getElementById('brightSlider').value;
  const c = document.getElementById('contrastSlider').value;
  const sh = document.getElementById('sharpSlider').value;
  const g = document.getElementById('graySlider').value;
  const preview = document.getElementById('previewImg');

  preview.style.filter = `brightness(${b}%) contrast(${c}%) grayscale(${g}%) blur(0px)`;
  if (sh > 0) {
    preview.style.filter += ` saturate(${100 + sh * 20}%)`;
  }
}

function presetDocument() {
  document.getElementById('brightSlider').value = 115;
  document.getElementById('contrastSlider').value = 145;
  document.getElementById('sharpSlider').value = 1.5;
  document.getElementById('graySlider').value = 60;
  updateEnhancement();
}

function applyEnhancement() {
  const src = state.originalImageBlob;
  if (!src) return;

  const b = parseInt(document.getElementById('brightSlider').value, 10) / 100;
  const c = parseInt(document.getElementById('contrastSlider').value, 10) / 100;
  const sh = parseFloat(document.getElementById('sharpSlider').value);
  const g = parseInt(document.getElementById('graySlider').value, 10) / 100;
  const img = new Image();

  img.onload = () => {
    const canvas = document.getElementById('enhCanvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');

    ctx.filter = `brightness(${b * 100}%) contrast(${c * 100}%)`;
    ctx.drawImage(img, 0, 0);
    ctx.filter = 'none';

    if (g > 0) {
      const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const d = id.data;
      for (let i = 0; i < d.length; i += 4) {
        const gray = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        d[i] = d[i] * (1 - g) + gray * g;
        d[i + 1] = d[i + 1] * (1 - g) + gray * g;
        d[i + 2] = d[i + 2] * (1 - g) + gray * g;
      }
      ctx.putImageData(id, 0, 0);
    }

    if (sh > 0) {
      const id2 = ctx.getImageData(0, 0, canvas.width, canvas.height);
      applyUnsharpMask(ctx, id2, sh);
    }

    canvas.toBlob(blob => {
      state.enhancedImageBlob = new File([blob], 'enhanced.jpg', { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      const previewEl = document.getElementById('previewImg');
      previewEl.src = url;
      previewEl.style.filter = '';
      showStatus('success', 'Enhancement applied - ready to translate');
    }, 'image/jpeg', 0.97);
  };

  img.src = URL.createObjectURL(src);
}

function applyUnsharpMask(ctx, imageData, amount) {
  const d = imageData.data;
  const w = imageData.width;
  const h = imageData.height;
  const k = amount * 0.5;
  const kernel = [
    0, -k, 0,
    -k, 1 + 4 * k, -k,
    0, -k, 0,
  ];
  const src = new Uint8ClampedArray(d);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        let val = 0;
        let ki = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            val += src[((y + dy) * w + (x + dx)) * 4 + ch] * kernel[ki++];
          }
        }
        d[idx + ch] = Math.max(0, Math.min(255, val));
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}
