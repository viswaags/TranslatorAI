async function callAPI(endpoint, formData, btn, tabKey, isImage) {
  try {
    const res = await fetch(endpoint, { method: 'POST', body: formData });
    const result = await res.json();
    const elapsed = ((Date.now() - state.processingStart) / 1000).toFixed(1);

    if (result.status === 'success') {
      showStatus('success', `Translation complete in ${elapsed}s`);
      showResults(result, elapsed, isImage);
    } else {
      showStatus('error', 'Error: ' + (result.message || 'Unknown error'));
    }
  } catch (err) {
    showStatus('error', 'Network error - is the server running? ' + err.message);
  } finally {
    setLoading(btn, false, tabKey);
  }
}
