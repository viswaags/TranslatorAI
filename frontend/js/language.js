function selectLang(btn, type, panel) {
  const grid = document.getElementById((type==='src'?'srcLangGrid-':'tgtLangGrid-') + panel);
  if (!grid) return;
  grid.querySelectorAll('.lang-chip').forEach(c => c.classList.remove('selected'));
  btn.classList.add('selected');
  state.langs[panel][type] = btn.dataset.val;
}

// ════════════════════════════════════════════════════════════
// AUDIO
// ════════════════════════════════════════════════════════════
