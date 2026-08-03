function resolveAPIBaseURL() {
  if (window.location.protocol === 'file:') {
    return 'http://127.0.0.1:8000';
  }
  if (window.location.port && window.location.port !== '8000') {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return '';
}

const API_BASE_URL = resolveAPIBaseURL();

const API_ENDPOINTS = Object.freeze({
  health: `${API_BASE_URL}/health`,
  languages: `${API_BASE_URL}/health/languages`,
  detectLanguage: `${API_BASE_URL}/api/v1/text/detect-language`,
  textTranslate: `${API_BASE_URL}/api/v1/text/translate`,
  ocrTranslate: `${API_BASE_URL}/api/v1/ocr/extract-and-translate`,
  speechTranslate: `${API_BASE_URL}/api/v1/speech/transcribe-and-translate`,
});

const FRONTEND_POLICY = Object.freeze({
  requestTimeoutMs: 120000,
  metadataTimeoutMs: 10000,
  textMaxCharacters: 5000,
  image: Object.freeze({
    maxSizeBytes: 10 * 1024 * 1024,
    extensions: Object.freeze(['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']),
    mimeTypes: Object.freeze([
      'image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp',
    ]),
  }),
  speech: Object.freeze({
    maxSizeBytes: 25 * 1024 * 1024,
    extensions: Object.freeze(['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']),
    mimeTypes: Object.freeze([
      'audio/wav', 'audio/x-wav', 'audio/wave', 'audio/vnd.wave',
      'audio/mpeg', 'audio/mp3', 'audio/ogg', 'application/ogg',
      'audio/flac', 'audio/x-flac', 'audio/mp4', 'audio/x-m4a',
      'audio/aac', 'audio/x-aac',
    ]),
  }),
});

class BackendAPIError extends Error {
  constructor(message, status, type, category, details) {
    super(message);
    this.name = 'BackendAPIError';
    this.status = status;
    this.type = type;
    this.category = category;
    this.details = details;
  }
}

class ClientValidationError extends Error {
  constructor(message, category = 'validation') {
    super(message);
    this.name = 'ClientValidationError';
    this.category = category;
  }
}

function fileExtension(filename) {
  const dot = filename.lastIndexOf('.');
  return dot >= 0 ? filename.slice(dot).toLowerCase() : '';
}

function formatMegabytes(bytes) {
  return `${Math.round(bytes / (1024 * 1024))} MB`;
}

function validateUpload(file, policy, label) {
  if (!(file instanceof Blob) || file.size === 0) {
    throw new ClientValidationError(`${label} file is empty.`);
  }

  if (file.size > policy.maxSizeBytes) {
    throw new ClientValidationError(
      `${label} exceeds the ${formatMegabytes(policy.maxSizeBytes)} upload limit.`,
      'upload_too_large',
    );
  }

  const extension = fileExtension(file.name || '');
  if (!policy.extensions.includes(extension)) {
    throw new ClientValidationError(
      `Unsupported ${label.toLowerCase()} extension "${extension || 'none'}". `
      + `Allowed formats: ${policy.extensions.join(', ')}.`,
      'unsupported_input',
    );
  }

  const mimeType = (file.type || '').toLowerCase().split(';')[0];
  if (!policy.mimeTypes.includes(mimeType)) {
    throw new ClientValidationError(
      `Unsupported ${label.toLowerCase()} type "${mimeType || 'unknown'}".`,
      'unsupported_input',
    );
  }

  return file;
}

function validateImageUpload(file) {
  return validateUpload(file, FRONTEND_POLICY.image, 'Image');
}

function validateSpeechUpload(file) {
  return validateUpload(file, FRONTEND_POLICY.speech, 'Audio');
}

function isWorkflowActive(workflow) {
  return Boolean(state.requests[workflow]);
}

function isCurrentWorkflowRequest(workflow, request) {
  return state.requests[workflow] === request;
}

function beginWorkflowRequest(workflow) {
  const previous = state.requests[workflow];
  if (previous) {
    previous.controller.abort(new DOMException('Request replaced', 'AbortError'));
    clearTimeout(previous.timeoutId);
  }

  const controller = new AbortController();
  const request = {
    controller,
    startedAt: Date.now(),
    timedOut: false,
    timeoutId: null,
  };
  request.timeoutId = setTimeout(() => {
    request.timedOut = true;
    controller.abort(new DOMException('Request timed out', 'TimeoutError'));
  }, FRONTEND_POLICY.requestTimeoutMs);
  state.requests[workflow] = request;
  return request;
}

function finishWorkflowRequest(workflow, request) {
  clearTimeout(request.timeoutId);
  if (state.requests[workflow] === request) {
    state.requests[workflow] = null;
  }
}

function cancelWorkflowRequest(workflow) {
  const request = state.requests[workflow];
  if (!request) return false;
  state.requests[workflow] = null;
  clearTimeout(request.timeoutId);
  request.controller.abort(new DOMException('Request cancelled', 'AbortError'));
  return true;
}

async function requestBackend(endpoint, options = {}) {
  const response = await fetch(endpoint, options);
  const contentType = response.headers.get('content-type') || '';
  let payload = null;

  if (contentType.includes('application/json')) {
    payload = await response.json();
  } else {
    const text = await response.text();
    payload = text ? { error: { message: text } } : {};
  }

  if (!response.ok || payload?.success === false) {
    const error = payload?.error || {};
    throw new BackendAPIError(
      error.message || `Request failed with HTTP ${response.status}`,
      response.status,
      error.type || 'HTTPError',
      error.category || 'http',
      error.details,
    );
  }

  return payload;
}

function errorMessage(error) {
  if (error instanceof ClientValidationError) {
    return error.message;
  }
  if (error?.name === 'TimeoutError') {
    return 'Request timed out. The local model took too long to respond; please try again.';
  }
  if (error?.name === 'AbortError') {
    return 'Request cancelled.';
  }
  if (!(error instanceof BackendAPIError)) {
    return 'Network error - is the server running? ' + error.message;
  }

  if (error.status === 413) {
    return 'Upload too large: the selected file exceeds the server limit.';
  }
  if (error.status === 415) {
    return 'Unsupported file type: choose a format accepted by the server.';
  }

  const prefixes = {
    validation: 'Invalid request',
    unsupported_input: 'Unsupported input',
    model_unavailable: 'Model unavailable',
    model_load: 'Model loading failed',
    inference: 'Processing failed',
    lifecycle: 'Service temporarily unavailable',
    overload: 'Service busy',
  };
  const prefix = prefixes[error.category] || 'Request failed';
  return `${prefix}: ${error.message}`;
}

function normalizeTranslationResponse(result, tabKey) {
  const inputText = tabKey === 'image'
    ? result.extracted_text
    : tabKey === 'speech'
      ? result.transcribed_text
      : result.input_text;
  const detectedLanguage = tabKey === 'speech'
    ? result.detected_source_lang_indictrans
    : result.detected_source_lang;

  return {
    inputText: inputText || '',
    translatedText: result.translated_text || '',
    detectedLanguage: detectedLanguage || '',
    detectedLanguageName: result.detected_source_lang_name || '',
    targetLanguage: result.target_lang || '',
    targetLanguageName: result.target_lang_name || '',
    translationEngine: result.translation_engine || '',
    processingMs: result.processing_ms || 0,
    inputType: tabKey,
    tts: result.tts || null,
  };
}

async function callAPI(endpoint, options, btn, tabKey, request) {
  try {
    const result = await requestBackend(endpoint, {
      ...options,
      signal: request.controller.signal,
    });
    const elapsed = ((Date.now() - request.startedAt) / 1000).toFixed(1);
    const viewModel = normalizeTranslationResponse(result, tabKey);
    showStatus('success', `Translation complete in ${elapsed}s`);
    showResults(viewModel, elapsed);
    return result;
  } catch (error) {
    if (!isCurrentWorkflowRequest(tabKey, request)) {
      return null;
    }
    if (request.timedOut && error?.name === 'AbortError') {
      error = new DOMException('Request timed out', 'TimeoutError');
    }
    showStatus('error', errorMessage(error));
    return null;
  } finally {
    finishWorkflowRequest(tabKey, request);
    if (!isWorkflowActive(tabKey)) {
      setLoading(btn, false, tabKey);
    }
  }
}

async function detectLanguage(text, signal) {
  return requestBackend(API_ENDPOINTS.detectLanguage, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal,
  });
}

function applyLanguageCatalog(languages) {
  state.languages = new Map(languages.map(language => [language.code, language]));

  document.querySelectorAll('.lang-chip[data-val]').forEach(chip => {
    if (chip.dataset.val === 'auto') return;
    const language = state.languages.get(chip.dataset.val);
    if (!language) return;
    const labelNode = Array.from(chip.childNodes)
      .find(node => node.nodeType === Node.TEXT_NODE);
    if (labelNode) labelNode.nodeValue = language.name;
  });
}

function updateConnectivity(connected) {
  const indicator = document.querySelector('.header-status');
  if (!indicator) return;
  const textNode = Array.from(indicator.childNodes)
    .find(node => node.nodeType === Node.TEXT_NODE);
  if (textNode) {
    textNode.nodeValue = connected
      ? 'Backend Ready · On-Device AI'
      : 'Backend Unavailable';
  }
}

async function loadBackendMetadata() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FRONTEND_POLICY.metadataTimeoutMs);
  try {
    const [health, catalog] = await Promise.all([
      requestBackend(API_ENDPOINTS.health, { signal: controller.signal }),
      requestBackend(API_ENDPOINTS.languages, { signal: controller.signal }),
    ]);
    state.health = health;
    applyLanguageCatalog(catalog.languages || []);
    updateConnectivity(health.status === 'ok');
  } catch (error) {
    updateConnectivity(false);
    console.warn('Backend metadata unavailable:', errorMessage(error));
  } finally {
    clearTimeout(timeoutId);
  }
}

document.addEventListener('DOMContentLoaded', loadBackendMetadata);
