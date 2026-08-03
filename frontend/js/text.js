async function submitText() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) {
    showStatus('error', 'Please enter some text first');
    return;
  }
  if (text.length > FRONTEND_POLICY.textMaxCharacters) {
    showStatus(
      'error',
      `Text is too long. The maximum is ${FRONTEND_POLICY.textMaxCharacters} characters.`,
    );
    return;
  }

  const request = beginWorkflowRequest('text');
  const btn = document.getElementById('textSubmitBtn');
  setLoading(btn, true, 'text');
  showStatus('processing', 'Detecting language -> IndicTrans2 -> Piper audio...');
  hideResults();
  try {
    const sourceLanguage = state.langs.text.src === 'auto'
      ? (await detectLanguage(text, request.controller.signal)).detected_lang
      : state.langs.text.src;

    await callAPI(API_ENDPOINTS.textTranslate, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        target_lang: state.langs.text.tgt,
        source_lang: sourceLanguage,
        tts: {
          enabled: true,
          engine: 'auto',
          speed: 150,
          return_audio: true,
        },
      }),
    }, btn, 'text', request);
  } catch (error) {
    if (!isCurrentWorkflowRequest('text', request)) {
      return;
    }
    if (request.timedOut && error?.name === 'AbortError') {
      error = new DOMException('Request timed out', 'TimeoutError');
    }
    showStatus('error', errorMessage(error));
    finishWorkflowRequest('text', request);
    if (!isWorkflowActive('text')) {
      setLoading(btn, false, 'text');
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const textInput = document.getElementById('textInput');
  const charCount = document.getElementById('charCount');

  if (textInput && charCount) {
    textInput.addEventListener('input', function () {
      const remaining = FRONTEND_POLICY.textMaxCharacters - this.value.length;
      charCount.textContent = remaining >= 0
        ? `${remaining} characters remaining`
        : `${Math.abs(remaining)} characters over limit`;
    });
    charCount.textContent =
      `${FRONTEND_POLICY.textMaxCharacters} characters remaining`;
  }
});
