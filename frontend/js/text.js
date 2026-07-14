async function submitText() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) {
    showStatus('error', 'Please enter some text first');
    return;
  }

  const btn = document.getElementById('textSubmitBtn');
  setLoading(btn, true, 'text');
  showStatus('processing', 'Detecting language -> translating -> generating audio...');
  hideResults();
  state.processingStart = Date.now();

  const fd = new FormData();
  fd.append('text', text);
  fd.append('target_lang', state.langs.text.tgt);
  fd.append('src_lang', state.langs.text.src);
  await callAPI('/translate/text', fd, btn, 'text', false);
}

document.addEventListener('DOMContentLoaded', () => {
  const textInput = document.getElementById('textInput');
  const charCount = document.getElementById('charCount');

  if (textInput && charCount) {
    textInput.addEventListener('input', function () {
      charCount.textContent = this.value.length + ' characters';
    });
  }
});
