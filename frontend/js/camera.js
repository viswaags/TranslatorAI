async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showStatus('error', 'Camera not supported. Use HTTPS or localhost.');
    return;
  }
  await startCamera(state.facingMode);
}

async function startCamera(facingMode) {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }

  const constraints = {
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 3840, min: 1280 },
      height: { ideal: 2160, min: 720 },
      focusMode: { ideal: 'continuous' },
    },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.cameraStream = stream;
    state.facingMode = facingMode;

    const video = document.getElementById('cameraVideo');
    video.srcObject = stream;
    await video.play();

    video.addEventListener('loadedmetadata', () => {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      document.getElementById('camResBadge').textContent = `${vw}x${vh}`;
      updateCamTip(vw, vh);
    }, { once: true });

    document.getElementById('cameraModal').classList.add('show');
    document.getElementById('cameraVideo').addEventListener('click', handleVideoTap);
  } catch (err) {
    try {
      const fallback = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facingMode } },
      });
      state.cameraStream = fallback;
      state.facingMode = facingMode;

      const video = document.getElementById('cameraVideo');
      video.srcObject = fallback;
      await video.play();
      document.getElementById('camResBadge').textContent = 'fallback res';
      document.getElementById('cameraModal').classList.add('show');
      document.getElementById('cameraVideo').addEventListener('click', handleVideoTap);
    } catch (err2) {
      let msg = 'Camera access denied.';
      if (err.name === 'NotAllowedError') msg += ' Allow camera permissions in browser settings.';
      else if (err.name === 'NotFoundError') msg += ' No camera found on this device.';
      showStatus('error', msg);
    }
  }
}

async function switchCamera() {
  state.facingMode = state.facingMode === 'environment' ? 'user' : 'environment';
  state.torchOn = false;
  document.getElementById('torchBtn').classList.remove('active');
  await startCamera(state.facingMode);
}

async function toggleTorch() {
  const track = state.cameraStream?.getVideoTracks()[0];
  if (!track) return;

  try {
    state.torchOn = !state.torchOn;
    await track.applyConstraints({ advanced: [{ torch: state.torchOn }] });
    document.getElementById('torchBtn').classList.toggle('active', state.torchOn);
  } catch (_) {
    document.getElementById('torchBtn').classList.remove('active');
    state.torchOn = false;
  }
}

function closeCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }

  state.torchOn = false;
  document.getElementById('torchBtn').classList.remove('active');
  document.getElementById('cameraVideo').removeEventListener('click', handleVideoTap);
  document.getElementById('cameraModal').classList.remove('show');
  document.getElementById('cameraVideo').srcObject = null;
}

async function capturePhoto() {
  const video = document.getElementById('cameraVideo');
  const track = state.cameraStream?.getVideoTracks()[0];
  const flash = document.getElementById('flashOverlay');

  flash.style.opacity = '0.7';
  setTimeout(() => flash.style.opacity = '0', 150);

  if (track && typeof ImageCapture !== 'undefined') {
    try {
      const ic = new ImageCapture(track);
      const blob = await ic.takePhoto();
      const file = new File([blob], 'camera_capture.jpg', { type: blob.type });
      closeCamera();
      showImagePreview(file);
      return;
    } catch (err) {
      console.warn('ImageCapture failed, falling back to canvas:', err);
    }
  }

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
    closeCamera();
    showImagePreview(file);
  }, 'image/jpeg', 0.97);
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    const modal = document.getElementById('cameraModal');
    if (modal && modal.classList.contains('show')) closeCamera();
  }
});
