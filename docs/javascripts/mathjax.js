// Configure MathJax *before* it loads
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  }
};

// Re-render math on every page change (Material instant navigation)
document$.subscribe(() => {
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetClear?.();
    MathJax.typesetPromise();
  }
});