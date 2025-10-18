---
layout: default
title: anki-v2
permalink: /blogs/anki-v2/
---

<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
  :root { --w: 900px; --gap: 12px; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif; margin: 0; padding: 0; }
  .wrap { max-width: var(--w); margin: 0 auto; padding: 24px; }
  .bar { display: flex; gap: 12px; align-items: center; margin: 10px 0 16px; flex-wrap: wrap; }
  button { display: block; width: 100%; border: 0; padding: 10px 12px; font-size: 0.95rem; border-radius: 10px; cursor: pointer; background: #2b64ff; color: #fff; }
  button.secondary { background: #e9ecef; color: #111; border: 1px solid #d0d7de; }
  button:disabled { opacity: 0.6; cursor: not-allowed; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: var(--gap); }
  .card { background: #fff; border: 1px solid #d0d7de; padding: 14px; border-radius: 14px; display: flex; flex-direction: column; gap: 10px; }
  .term { font-weight: 700; line-height: 1.3; word-break: normal; overflow-wrap: anywhere; white-space: normal; }
  .muted { color: #666; font-size: 0.9rem; }
  .sect { display: none; padding: 10px; border: 1px dashed #ced4da; border-radius: 10px; background: #f8f9fb; }
  .sect.revealed { display: block; }
  .kvs { display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; }
  .kvs div { min-width: 0; overflow-wrap: anywhere; word-break: normal; }
	.actions {
	  display: grid;
	  grid-template-columns: 1fr 1fr;
	  gap: 8px;
	}
	.actions button { width: 100%; }
</style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <button id="btn-random">Pick me!</button>
    </div>
    <div id="cards" class="grid" aria-live="polite"></div>
  </div>
 <script>
(function(){
  var GLOSSARY_URL = '{{ "/blogs/assets/glossary.en.json" | relative_url }}';

  // --- helpers ---
  function $(s, el){ return (el||document).querySelector(s); }
  function sample(arr, k){
    var a = arr.slice();
    for (var i=a.length-1; i>0; i--){ var j=Math.floor(Math.random()*(i+1)); var t=a[i]; a[i]=a[j]; a[j]=t; }
    if (k>a.length) k=a.length; return a.slice(0,k);
  }
  function loadJSON(url, onOk, onErr){
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.onreadystatechange = function(){
      if (xhr.readyState === 4){
        if (xhr.status >= 200 && xhr.status < 300){
          try { onOk(JSON.parse(xhr.responseText)); } catch(e){ onErr(e); }
        } else { onErr(new Error('HTTP '+xhr.status)); }
      }
    };
    xhr.send(null);
  }
  function normalizeEntries(root){
    // Accepts array [{word,examples,translations}, ...] OR dict {"0":{...},...}
    if (Object.prototype.toString.call(root) === '[object Array]') return root;
    if (root && typeof root === 'object'){
      var keys = Object.keys(root).sort(function(a,b){
        var an=/^\d+$/.test(a), bn=/^\d+$/.test(b);
        if (an && bn) return parseInt(a,10)-parseInt(b,10);
        if (an && !bn) return -1; if (!an && bn) return 1;
        return a<b?-1:a>b?1:0;
      });
      var out=[]; for (var i=0;i<keys.length;i++) out.push(root[keys[i]]);
      return out;
    }
    return [];
  }

  // --- UI builders ---
  function makeList(items){
    var ol = document.createElement('ol');
    ol.style.margin = '0';
    ol.style.paddingLeft = '1.2rem';
    for (var i=0;i<items.length;i++){
      var li = document.createElement('li');
      li.textContent = items[i];
      li.style.margin = '4px 0';
      ol.appendChild(li);
    }
    return ol;
  }

  function makeCard(entry, entryIndex){
    var word = entry.word || '';
    var exs  = Object.prototype.toString.call(entry.examples)==='[object Array]' ? entry.examples : [];
    var trs  = Object.prototype.toString.call(entry.translations)==='[object Array]' ? entry.translations : [];

    var card = document.createElement('div'); card.className='card';
    
    var title = document.createElement('div'); title.className='term'; title.textContent = word;

    
    // Examples (DE)
    var btnEx = document.createElement('button');
    btnEx.appendChild(document.createTextNode('Beispiel(e)'));
    btnEx.setAttribute('aria-expanded','false');
    
    var sectEx = document.createElement('div'); sectEx.className='sect'; sectEx.setAttribute('aria-hidden','true');
    var exGrid = document.createElement('div'); exGrid.className='kvs';
    var exLbl  = document.createElement('div'); exLbl.className='muted'; exLbl.appendChild(document.createTextNode('DE'));
    var exBox  = document.createElement('div'); exBox.appendChild(makeList(exs));
    exGrid.appendChild(exLbl); exGrid.appendChild(exBox); sectEx.appendChild(exGrid);
    
    // Translations (EN) – aligned 1:1; we show up to exs.length
    var btnTr = document.createElement('button');
    btnTr.className='secondary';
    btnTr.appendChild(document.createTextNode('Übersetzung(en)'));
    btnTr.disabled = true;
    btnTr.setAttribute('aria-expanded','false');

	// added this:
	var actions = document.createElement('div');
	actions.className = 'actions';
	actions.appendChild(btnEx);
	actions.appendChild(btnTr);

    
    var sectTr = document.createElement('div'); sectTr.className='sect'; sectTr.setAttribute('aria-hidden','true');
    var trsShown = trs.slice(0, exs.length); // guard if lengths differ
    var trGrid = document.createElement('div'); trGrid.className='kvs';
    var trLbl  = document.createElement('div'); trLbl.className='muted'; trLbl.appendChild(document.createTextNode('EN'));
    var trBox  = document.createElement('div'); trBox.appendChild(makeList(trsShown));
    trGrid.appendChild(trLbl); trGrid.appendChild(trBox); sectTr.appendChild(trGrid);
    
    // Toggles
    btnEx.addEventListener('click', function(){
      var open = sectEx.className.indexOf('revealed') === -1;
      if (open){
        sectEx.className += ' revealed';
        sectEx.setAttribute('aria-hidden','false');
        btnEx.setAttribute('aria-expanded','true');
        btnTr.disabled = false;
      } else {
        sectEx.className = sectEx.className.replace(' revealed','');
        sectEx.setAttribute('aria-hidden','true');
        btnEx.setAttribute('aria-expanded','false');
        btnTr.disabled = true;
        sectTr.className = sectTr.className.replace(' revealed','');
        sectTr.setAttribute('aria-hidden','true');
        btnTr.setAttribute('aria-expanded','false');
      }
    });
    
    btnTr.addEventListener('click', function(){
      if (btnTr.disabled) return;
      var open = sectTr.className.indexOf('revealed') === -1;
      if (open){
        sectTr.className += ' revealed';
        sectTr.setAttribute('aria-hidden','false');
        btnTr.setAttribute('aria-expanded','true');
      } else {
        sectTr.className = sectTr.className.replace(' revealed','');
        sectTr.setAttribute('aria-hidden','true');
        btnTr.setAttribute('aria-expanded','false');
      }
    });
    
    // Assemble vertically

    //card.appendChild(title);
    //card.appendChild(btnEx);
    //card.appendChild(sectEx);
    //card.appendChild(btnTr);
    //card.appendChild(sectTr);

	card.appendChild(title);      
	card.appendChild(actions);
	card.appendChild(sectEx);
	card.appendChild(sectTr);





    return card;
  }

  function renderRandomEntries(entries){
    var cardsEl = $('#cards'); cardsEl.innerHTML = '';
    var pick = sample(entries, 5); // 5 random words
    for (var i=0;i<pick.length;i++){
      cardsEl.appendChild(makeCard(pick[i], i));
    }
    var where = $('#where');
  }

  function start(){
    loadJSON(GLOSSARY_URL, function(data){
      var entries = normalizeEntries(data)
        .filter(function(e){ return e && e.word && Object.prototype.toString.call(e.examples)==='[object Array]' && e.examples.length; });
      if (!entries.length){
        $('#cards').innerHTML = '<div class="muted">No entries with examples found.</div>';
        return;
      }
      renderRandomEntries(entries);
      var btn = $('#btn-random');
      if (btn){ btn.addEventListener('click', function(){ renderRandomEntries(entries); }); }
    }, function(){
      $('#cards').innerHTML = '<div class="muted">Failed to load /blogs/assets/glossary.en.json</div>';
    });
  }

  if (document.readyState === 'loading'){ document.addEventListener('DOMContentLoaded', start); } else { start(); }
})();
</script>
