---
layout: default
title: Anki 
permalink: /blogs/blog-glossary-anki/
---
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --w: 880px; --gap: 12px; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif; margin: 0; padding: 0; }
    .wrap { max-width: var(--w); margin: 0 auto; padding: 24px; }
    h1 { font-weight: 700; font-size: 1.35rem; margin: 0 0 8px; }
    p.meta { color: #666; margin: 0 0 16px; font-size: 0.95rem; }
    .bar { display: flex; flex-wrap: wrap; gap: var(--gap); align-items: center; margin: 16px 0 20px; }
    button { border: 0; padding: 10px 14px; font-size: 0.95rem; border-radius: 10px; cursor: pointer; background: #2b64ff; color: white; }
    button.secondary { background: #e9ecef; color: #111; border: 1px solid #d0d7de; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: var(--gap); }
    .card { background: #fff; border: 1px solid #d0d7de; padding: 14px; border-radius: 14px; min-height: 120px; display: flex; flex-direction: column; justify-content: space-between; }
    .term { font-weight: 700; line-height: 1.35; }
    .solution { margin-top: 10px; padding-top: 10px; border-top: 1px dashed #ced4da; display: none; color: #333; }
    .card.revealed .solution { display: block; }
    .card .row { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .muted { color: #666; font-size: 0.9rem; }
    code { background: #f6f8fa; border: 1px solid #d0d7de; padding: 2px 6px; border-radius: 6px; }
    a { color: #0969da; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <button id="btn-random">Random 10</button>
      <button id="btn-swap" class="secondary" title="swap">Eng/De</button>
      <span id="where" class="muted"></span>
    </div>
    <div id="cards" class="grid" aria-live="polite"></div>
  </div>

  <script>
    const $ = (s, el=document) => el.querySelector(s);
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

    // Fixed path under /blogs/ per your structure
    const GLOSSARY_URL = '{{ "/blogs/assets/glossary.json" | relative_url }}';

    // Normalize to {idx, de, en}
    function normalize(raw) {
      return raw.map((x, i) => Array.isArray(x)
        ? { idx: i, de: x[0], en: x[1] }
        : { idx: (typeof x.idx === 'number') ? x.idx : i, de: x.de, en: x.en }
      ).filter(x => x.de && x.en);
    }

    function escapeHTML(s) { return String(s).replaceAll('<','&lt;').replaceAll('>','&gt;'); }

    function sample(arr, k) {
      const a = arr.slice();
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a.slice(0, k);
    }

    let GLOSSARY = []; // full ordered list
    let swap = false;  // false: DE front, true: EN front
    let offset = 0;    // position in sequential mode

    function saveOffset() { localStorage.setItem('anki.offset', String(offset)); }
    function loadOffset() { const v = localStorage.getItem('anki.offset'); return v? parseInt(v,10) : 0; }

    async function loadGlossary() {
      try {
        const res = await fetch(GLOSSARY_URL, { cache: 'no-store' });
        if (res.ok) {
          const raw = await res.json();
          GLOSSARY = normalize(raw);
        }
      } catch (_) {}

      if (!GLOSSARY.length) {
        // Minimal fallback (remove in production)
        GLOSSARY = normalize([
          ["giftig", "poisonous(ly)"],
          ["im Freien", "in the outdoors"],
          ["die Haut (Singular)", "skin"],
          ["das Insekt, -en", "insect"],
          ["der Insektenschutz (Singular)", "insect repellent"],
          ["das Netz, -e", "net"],
          ["der Pilz, -e", "mushroom, toadstool"],
          ["der Schutz (Singular)", "protection"],
          ["übernachten", "to spend the night"],
          ["die Abneigung, -en", "dislike, aversion"],
        ]);
      }

    }

    function makeCard(pair) {
      const front = swap ? pair.en : pair.de;
      const back  = swap ? pair.de : pair.en;
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="row">
          <div class="term">${escapeHTML(front)}<div class="muted">idx ${pair.idx}</div></div>
          <button class="secondary btn-show" aria-expanded="false">Show</button>
        </div>
        <div class="solution" aria-hidden="true">${escapeHTML(back)}</div>
      `;
      const btn = card.querySelector('.btn-show');
      const sol = card.querySelector('.solution');
      btn.addEventListener('click', () => {
        const isOpen = card.classList.toggle('revealed');
        btn.textContent = isOpen ? 'Hide' : 'Show';
        btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        sol.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
      });
      return card;
    }

    function renderSlice(start) {
      const cardsEl = $('#cards');
      cardsEl.innerHTML = '';
      if (!GLOSSARY.length) return;
      const slice = GLOSSARY.slice(start, start + 10);
      slice.forEach(p => cardsEl.appendChild(makeCard(p)));
      const end = Math.min(start + 9, Math.max(0, GLOSSARY.length - 1));
      $('#where').textContent = `Showing idx ${GLOSSARY[start]?.idx}–${GLOSSARY[end]?.idx}`;
    }

    function dealRandom() {
      const cardsEl = $('#cards');
      cardsEl.innerHTML = '';
      if (!GLOSSARY.length) return;
      sample(GLOSSARY, Math.min(10, GLOSSARY.length)).forEach(p => cardsEl.appendChild(makeCard(p)));
      $('#where').textContent = `Random 10 from ${GLOSSARY.length}`;
    }

    (async function init() {
      await loadGlossary();
      offset = loadOffset();
      offset = clamp(offset, 0, Math.max(0, GLOSSARY.length - 1));
      renderSlice(offset);

      $('#btn-random').addEventListener('click', dealRandom);
      $('#btn-swap').addEventListener('click', () => { swap = !swap; renderSlice(offset); });
    })();
  </script>
</body>
</html>

