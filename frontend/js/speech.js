function handleAudioSelect(e) {
  const f = e.target.files[0];
  if (!f) return;
  try {
    state.selectedAudioFile = validateSpeechUpload(f);
    const label = document.getElementById('audioFileName');
    label.style.display = 'block';
    label.textContent = f.name;
  } catch (error) {
    e.target.value = '';
    state.selectedAudioFile = null;
    showStatus('error', errorMessage(error));
  }
}

function audioBufferToWavFile(audioBuffer) {
  const channels = audioBuffer.numberOfChannels;
  const frameCount = audioBuffer.length;
  const bytesPerSample = 2;
  const dataSize = frameCount * channels * bytesPerSample;
  const wav = new ArrayBuffer(44 + dataSize);
  const view = new DataView(wav);

  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };

  writeText(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeText(8, 'WAVE');
  writeText(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, audioBuffer.sampleRate, true);
  view.setUint32(28, audioBuffer.sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeText(36, 'data');
  view.setUint32(40, dataSize, true);

  const channelData = Array.from(
    { length: channels },
    (_, channel) => audioBuffer.getChannelData(channel),
  );
  let offset = 44;
  for (let frame = 0; frame < frameCount; frame++) {
    for (let channel = 0; channel < channels; channel++) {
      const sample = Math.max(-1, Math.min(1, channelData[channel][frame]));
      view.setInt16(
        offset,
        sample < 0 ? sample * 0x8000 : sample * 0x7fff,
        true,
      );
      offset += bytesPerSample;
    }
  }

  return new File([wav], 'recording.wav', { type: 'audio/wav' });
}

async function createSupportedRecording(chunks, mimeType) {
  const source = new Blob(chunks, { type: mimeType });
  if (mimeType.startsWith('audio/ogg')) {
    return new File([source], 'recording.ogg', { type: 'audio/ogg' });
  }

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error('This browser cannot convert microphone audio to WAV.');
  }

  const context = new AudioContextClass();
  try {
    const audioBuffer = await context.decodeAudioData(await source.arrayBuffer());
    return audioBufferToWavFile(audioBuffer);
  } finally {
    await context.close();
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
      state.mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        document.getElementById('micSubLabel').textContent =
          'Preparing recording for offline transcription...';
        try {
          const recording = await createSupportedRecording(state.audioChunks, mimeType);
          state.selectedAudioFile = validateSpeechUpload(recording);
          document.getElementById('micSubLabel').textContent =
            'Recording saved - click Transcribe & Translate';
          document.getElementById('micSubLabel').style.color = 'var(--accent3)';
        } catch (error) {
          state.selectedAudioFile = null;
          document.getElementById('micSubLabel').textContent =
            'The browser recording could not be prepared for transcription.';
          document.getElementById('micSubLabel').style.color = 'var(--danger)';
          const message = error instanceof ClientValidationError
            ? errorMessage(error)
            : `Recording error: ${error.message}`;
          showStatus('error', message);
        }
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

  try {
    validateSpeechUpload(state.selectedAudioFile);
  } catch (error) {
    showStatus('error', errorMessage(error));
    return;
  }

  const request = beginWorkflowRequest('speech');
  const btn = document.getElementById('speechSubmitBtn');
  setLoading(btn, true, 'speech');
  showStatus('processing', 'Running Faster Whisper -> IndicTrans2 -> Piper audio...');
  hideResults();
  const fd = new FormData();
  fd.append('file', state.selectedAudioFile);
  fd.append('target_lang', state.langs.speech.tgt);
  fd.append('tts_enabled', 'true');
  fd.append('tts_engine', 'auto');
  fd.append('tts_return_audio', 'true');
  fd.append('tts_speed', '150');
  document.getElementById('micLabel').textContent = 'Tap to start recording';
  document.getElementById('micSubLabel').textContent = 'Speak -> Faster Whisper -> IndicTrans2';
  document.getElementById('micSubLabel').style.color = 'var(--muted)';
  await callAPI(API_ENDPOINTS.speechTranslate, {
    method: 'POST',
    body: fd,
  }, btn, 'speech', request);
}
