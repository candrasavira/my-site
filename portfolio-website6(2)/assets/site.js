/* Hides optional files and screenshots until you add them to assets/. */
(function () {
  var live = location.protocol !== 'file:';
  document.querySelectorAll('img[data-optional]').forEach(function (img) {
    function drop() { var f = img.closest('figure'); if (f) f.remove(); }
    if (img.complete && img.naturalWidth === 0) drop();
    else img.addEventListener('error', drop);
  });
  setTimeout(function () {
    document.querySelectorAll('[data-shots]').forEach(function (s) { if (!s.querySelector('figure')) s.remove(); });
  }, 800);
  if (live) {
    var checks = [];
    document.querySelectorAll('a[data-optional]').forEach(function (a) {
      checks.push(fetch(a.href, { method: 'HEAD' }).then(function (r) { if (!r.ok) a.remove(); }).catch(function () { a.remove(); }));
    });
    Promise.all(checks).then(function () {
      document.querySelectorAll('.files').forEach(function (f) {
        if (!f.querySelector('a')) { var s = f.closest('section'); if (s) s.remove(); }
      });
    });
  }
})();
