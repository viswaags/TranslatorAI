function handleAudioSelect(e) {
  const f = e.target.files[0];
  if (f) {
    state.selectedAudioFile = f;
    const el = document.getElementById('audioFileName');
    el.style.display = 'block';
    el.textContent = f.name;
  }
}

async function toggleRecording() {
  if (!state.recording) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showStatus('error', 'Microphone not supported. Use HTTPS or localhost.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1 },
      });
      state.audioChunks = [];

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')
          ? 'audio/ogg;codecs=opus'
          : 'audio/webm';

      state.mediaRecorder = new MediaRecorder(stream, { mimeType });
      state.mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) state.audioChunks.push(e.data);
      };
      state.mediaRecorder.onstop = () => {
        state.selectedAudioFile = new Blob(state.audioChunks, { type: mimeType });
        stream.getTracks().forEach(t => t.stop());
        document.getElementById('micSubLabel').textContent = 'Recording saved - click Transcribe & Translate';
        document.getElementById('micSubLabel').style.color = 'var(--accent3)';
      };

      state.mediaRecorder.start(250);
      state.recording = true;
      document.getElementById('micBtn').classList.add('recording');
      document.getElementById('waveform').classList.add('visible');
      document.getElementById('micLabel').textContent = 'Recording... tap to stop';
      document.getElementById('micSubLabel').textContent = 'Microphone active';
      document.getElementById('micSubLabel').style.color = 'var(--danger)';
    } catch (err) {
      let msg = 'Microphone access denied.';
      if (err.name === 'NotAllowedError') msg += ' Check browser permissions.';
      else if (err.name === 'NotFoundError') msg += ' No microphone found.';
      showStatus('error', msg);
    }
  } else {
    state.mediaRecorder.stop();
    state.recording = false;
    document.getElementById('micBtn').classList.remove('recording');
    document.getElementById('waveform').classList.remove('visible');
    document.getElementById('micLabel').textContent = 'Tap to start recording';
  }
}

async function submitSpeech() {
  if (!state.selectedAudioFile) {
    showStatus('error', 'Please record or upload audio first');
    return;
  }

  const btn = document.getElementById('speechSubmitBtn');
  setLoading(btn, true, 'speech');
  showStatus('processing', 'Running Whisper STT -> translating -> generating audio...');
  hideResults();
  state.processingStart = Date.now();

  const fd = new FormData();
  fd.append('file', state.selectedAudioFile, 'recording.wav');
  fd.append('target_lang', state.langs.speech.tgt);
  document.getElementById('micLabel').textContent = 'Tap to start recording';
  document.getElementById('micSubLabel').textContent = 'Speak -> Whisper STT -> Translation';
  document.getElementById('micSubLabel').style.color = 'var(--muted)';
  await callAPI('/translate/speech', fd, btn, 'speech', false);
}
