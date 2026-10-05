/* =============================================================================
   La Ruche : interface du banc d'essai
   -----------------------------------------------------------------------------
   1. Chargement des modèles (modeles/index.js puis modeles/<nom>/modele.js + poids_*.js)
   2. Panneau des modèles (activation, architecture, précision mesurée)
   3. Images : dépôt, collage, webcam → redimensionnement → prédiction par chaque modèle
   4. Cartes de résultats : prédiction de chaque modèle, consensus, « En vrai », détails

   Le calcul du réseau lui-même est dans engine.js (window.LuckyEngine).
   ============================================================================= */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const E = window.LuckyEngine;
  const nextFrame = () => new Promise((r) => setTimeout(r, 0));

  const statusEl = $("#status"), statusText = $("#status-text");
  const dropzone = $("#dropzone"), fileInput = $("#file-input");
  const results = $("#results"), empty = $("#empty"), resultsHead = $("#results-head");
  const modelsEl = $("#models"), tpl = $("#tpl-card");

  const COLORS = ["#f5b301", "#8a5cf6", "#14a39a", "#e8613c", "#3a86e8", "#5da83a", "#d6479b", "#8c6a3a"];

  /** @type {{name:string, color:string, meta?:any, net?:any, error?:string, active:boolean}[]} */
  const models = [];
  const cards = [];
  let queue = Promise.resolve();

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* rien */ } },
  };

  function setStatus(text, kind) {
    statusText.textContent = text;
    statusEl.classList.toggle("ready", kind === "ready");
    statusEl.classList.toggle("error", kind === "error");
  }

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const activeModels = () => models.filter((m) => m.net && m.active);

  // ---------------------------------------------------------------------
  // Comparer des modèles qui n'ont pas les mêmes classes
  // Le modèle 2 classes (Autre / Photo) range les peintures dans « Autre ».
  // Pour lui, répondre « Autre » sur une peinture est donc juste, et ce n'est
  // pas un désaccord avec le modèle 3 classes qui répond « Peinture ».
  // ---------------------------------------------------------------------
  // Icônes « juste » / « faux » (petits tracés SVG, pas d'emoji)
  const ICON_OK = `<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5 5 9.5 10 3"/></svg>`;
  const ICON_KO = `<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3 9 9M9 3 3 9"/></svg>`;

  const CATCH_ALL = ["autre", "autres", "other", "others"];
  const catchAll = (meta) => meta.classes.find((n) => CATCH_ALL.includes(n.toLowerCase())) || null;

  /** Classe que le modèle aurait dû répondre si l'image est vraiment `truth` (null : on ne peut pas juger). */
  function expectedLabel(meta, truth) {
    if (!truth) return null;
    if (meta.classes.includes(truth)) return truth;
    return catchAll(meta);
  }

  /** Les modèles sont-ils d'accord ? Renvoie { agree, label, text }. */
  function consensus(entries) { // entries : [{ meta, label }]
    // Une réponse « précise » est une classe autre que la classe fourre-tout du modèle.
    const precise = entries.filter((e) => e.label !== catchAll(e.meta));
    const labels = [...new Set(precise.map((e) => e.label))];
    if (!labels.length) return { agree: true, text: `Unanimes : ${entries[0].label}` };
    if (labels.length === 1) {
      const L = labels[0];
      // Ceux qui ont répondu « Autre » sont cohérents seulement s'ils ne connaissent pas L.
      const ok = entries.every((e) => e.label === L || !e.meta.classes.includes(L));
      if (ok) return { agree: true, text: precise.length === entries.length ? `Unanimes : ${L}` : `Cohérents : ${L}` };
    }
    const counts = {};
    entries.forEach((e) => { counts[e.label] = (counts[e.label] || 0) + 1; });
    const parts = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([l, n]) => `${l} (${n})`);
    return { agree: false, text: "Avis partagés : " + parts.join(" · ") };
  }

  function fmtPct(v) {
    const pc = v * 100;
    if (pc > 99.9 && pc < 100) return "> 99,9 %";
    if (pc < 0.1 && pc > 0) return "< 0,1 %";
    return pc.toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " %";
  }

  // =====================================================================
  // Chargement des modèles : modeles/index.js puis modeles/<nom>/modele.js + poids_*.js
  // =====================================================================
  let loading = null;
  window.RUCHE = {
    liste(names) { window.RUCHE._names = names; },
    modele(meta) { if (loading) loading.meta = meta; },
    poids(i, s) { if (loading) loading.parts[i] = s; },
  };

  const loadScript = (src) => new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src; s.onload = () => { s.remove(); res(); };
    s.onerror = () => { s.remove(); rej(new Error(src + " introuvable")); };
    document.body.appendChild(s);
  });

  async function loadAll() {
    setStatus("Réveil de la ruche…");
    try {
      await loadScript("modeles/index.js");
    } catch (_) {
      setStatus("modeles/index.js introuvable : lance convertir_modeles.bat", "error");
      renderModels();
      return;
    }
    const names = window.RUCHE._names || [];
    names.forEach((name, i) => models.push({ name, color: COLORS[i % COLORS.length], active: store.get("ruche.off." + name) !== "1" }));
    renderModels();
    if (!names.length) { setStatus("Aucun modèle : ajoute-en un dans modeles/", "error"); return; }

    for (let i = 0; i < models.length; i++) {
      const m = models[i];
      setStatus(`Les abeilles rapportent ${m.name}… (${i + 1}/${models.length})`);
      try {
        loading = { meta: null, parts: [] };
        await loadScript(`modeles/${encodeURIComponent(m.name)}/modele.js`);
        if (!loading.meta) throw new Error("modele.js invalide");
        for (const p of loading.meta.parts) await loadScript(`modeles/${encodeURIComponent(m.name)}/${p}`);
        await nextFrame();
        m.meta = loading.meta;
        m.net = E.createModel(loading.meta, loading.parts.join(""));
      } catch (e) {
        console.error(e);
        m.error = e.message;
      } finally {
        loading = null;
      }
      renderModels();
    }
    const ok = models.filter((m) => m.net).length;
    $("#model-count").textContent = ok + (ok > 1 ? " modèles en compétition" : " modèle");
    if (!ok) { setStatus("Aucun modèle n'a pu être chargé", "error"); return; }
    setStatus(ok > 1 ? `${ok} modèles prêts` : "Ruche prête", "ready");
    dropzone.classList.remove("disabled");
  }

  // =====================================================================
  // Panneau des modèles
  // =====================================================================
  function modelScore(m) {
    let ok = 0, n = 0, ms = 0, k = 0;
    for (const c of cards) {
      const r = c.results[m.name];
      if (r && r.ms != null) { ms += r.ms; k++; }
      const expected = expectedLabel(m.meta, c.truth);
      if (!r || !r.probs || !expected) continue;
      n++; if (r.label === expected) ok++;
    }
    return { ok, n, ms: k ? ms / k : null };
  }

  function renderModels() {
    if (!models.length) {
      modelsEl.innerHTML = `<p class="hint">Aucun modèle trouvé. Mets un fichier <code>.keras</code> dans <code>modeles/V1/</code> puis lance <code>convertir_modeles.bat</code>.</p>`;
      return;
    }
    modelsEl.innerHTML = "";
    for (const m of models) {
      const div = document.createElement("div");
      div.className = "model" + (m.active ? "" : " off") + (m.error ? " broken" : "");
      div.style.setProperty("--mc", m.color);
      let body;
      if (m.error) body = `<p class="model-err">Erreur : ${esc(m.error)}</p>`;
      else if (!m.meta) body = `<p class="model-sub">Chargement…</p>`;
      else {
        const mt = m.meta, [h, w, c] = mt.input;
        const sc = modelScore(m);
        const date = mt.date ? mt.date.split("@")[0].split("-").reverse().join("/") : null;
        body = `
          <p class="model-sub">${esc(mt.source)}${date ? " · entraîné le " + date : ""}</p>
          <div class="model-meta">
            <span>${w}×${h} ${c === 1 ? "gris" : "RGB"}</span>
            <span>${(mt.params / 1e6 >= 1 ? (mt.params / 1e6).toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " M" : mt.params.toLocaleString("fr-FR"))} param.</span>
            ${sc.ms != null ? `<span>~${Math.round(sc.ms)} ms</span>` : ""}
          </div>
          <div class="model-classes">${mt.classes.map((n, i) => `<span class="cls"><b>${i}</b>${esc(n)}</span>`).join("")}</div>
          ${sc.n ? `<div class="model-score"><span>Précision</span><b>${Math.round((sc.ok / sc.n) * 100)} %</b><span>${sc.ok}/${sc.n}</span></div>` : ""}
          <details class="arch"><summary>Architecture (${mt.layers.length} couches)</summary><ol class="stack">${mt.layers.map(layerLine).join("")}</ol></details>`;
      }
      div.innerHTML = `
        <label class="model-head">
          <span class="mtag">${esc(m.name)}</span>
          <span class="model-state">${m.error ? "en erreur" : !m.net ? "chargement…" : m.active ? "actif" : "en pause"}</span>
          <input type="checkbox" class="switch" ${m.active ? "checked" : ""} ${m.net ? "" : "disabled"} aria-label="Activer ${esc(m.name)}">
        </label>${body}`;
      $(".switch", div).addEventListener("change", (e) => toggleModel(m, e.target.checked));
      modelsEl.appendChild(div);
    }
  }

  function layerLine(L) {
    let d = "";
    const act = L.activation && L.activation !== "linear" ? " · " + L.activation : "";
    if (L.type === "Conv2D") d = `${L.filters} × ${L.kernel.join("×")}${act}`;
    else if (L.type === "Dense") d = `${L.units} unités${act}`;
    else if (L.type === "Pool") d = `${L.mode} ${L.pool.join("×")}`;
    else if (L.type === "GlobalPool") d = L.mode;
    else if (L.type === "Activation") d = L.activation;
    else if (L.type === "Affine") d = L.name;
    return `<li><span>${esc(L.type)}</span><span>${esc(d)}</span></li>`;
  }

  function toggleModel(m, on) {
    m.active = on;
    store.set("ruche.off." + m.name, on ? "0" : "1");
    renderModels();
    cards.forEach(renderCard);
    if (on) cards.forEach((c) => { queue = queue.then(() => analyse(c)); });
  }

  // =====================================================================
  // Images
  // =====================================================================
  function fileToImage(blob) {
    return new Promise((res, rej) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => res({ img, url });
      img.onerror = () => { URL.revokeObjectURL(url); rej(new Error("image illisible")); };
      img.src = url;
    });
  }

  // Pixels de l'image en pleine résolution (aucun pré-rétrécissement par le navigateur).
  function fullPixels(img) {
    const MAX = 16000; // garde-fou : taille max d'un canvas
    const k = Math.min(1, MAX / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.max(1, Math.round(img.naturalWidth * k)), h = Math.max(1, Math.round(img.naturalHeight * k));
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    const ctx = cv.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    return { rgba: ctx.getImageData(0, 0, w, h).data, w, h };
  }

  function inputFor(c, meta) {
    const [H, W, C] = meta.input, key = `${W}x${H}x${C}`;
    if (!c.inputs[key]) {
      if (!c.full) c.full = fullPixels(c.img);
      c.inputs[key] = E.resizeBilinear(c.full.rgba, c.full.w, c.full.h, W, H, C);
    }
    return c.inputs[key];
  }

  function addFiles(files) {
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (!list.length) return;
    empty.hidden = true; resultsHead.hidden = false;
    for (const f of list) {
      const c = makeCard(f.name || "image collée");
      queue = queue.then(() => loadCardImage(c, f)).then(() => analyse(c));
    }
  }

  async function loadCardImage(c, file) {
    try {
      const { img, url } = await fileToImage(file);
      c.img = img; c.url = url;
      const im = $("img", c.el);
      im.src = url; im.alt = c.name;
      $(".res", c.el).textContent = `${img.naturalWidth}×${img.naturalHeight}`;
    } catch (e) {
      c.error = e.message;
      renderCard(c);
    }
  }

  async function analyse(c) {
    if (!c.img) return;
    for (const m of activeModels()) {
      if (c.results[m.name]) continue;
      c.el.classList.remove("done");
      await nextFrame();
      try {
        const x = inputFor(c, m.meta);
        const t0 = performance.now();
        const { raw, probs } = m.net.predict(x);
        const ms = performance.now() - t0;
        const best = probs.indexOf(Math.max(...probs));
        c.results[m.name] = { raw, probs, ms, best, label: m.meta.classes[best], fresh: true };
      } catch (e) {
        console.error(e);
        c.results[m.name] = { error: e.message };
      }
      renderCard(c);
    }
    c.full = null; // libère la mémoire de l'image pleine résolution
    c.el.classList.add("done");
    renderCard(c);
    renderModels();
    renderBoard();
  }

  // =====================================================================
  // Cartes de résultats
  // =====================================================================
  function makeCard(name) {
    const el = tpl.content.firstElementChild.cloneNode(true);
    const c = { el, name, img: null, url: null, full: null, inputs: {}, results: {}, truth: null, view: -1, error: null };
    $(".fname-text", el).textContent = name;
    $(".fname", el).title = name;
    results.prepend(el);
    cards.push(c);

    $(".btn-view", el).addEventListener("click", () => cycleView(c));
    $(".btn-details", el).addEventListener("click", (e) => {
      const d = $(".details", el), on = d.hidden;
      d.hidden = !on; e.currentTarget.setAttribute("aria-pressed", String(on));
    });
    renderCard(c);
    return c;
  }

  // Vue du modèle : alterne entre les tailles d'entrée des modèles actifs, puis revient à la photo.
  function cycleView(c) {
    if (!c.img) return;
    const seen = new Set(), views = [];
    for (const m of activeModels()) {
      const key = m.meta.input.join("x");
      if (!seen.has(key)) { seen.add(key); views.push(m); }
    }
    c.view = c.view + 1 >= views.length ? -1 : c.view + 1;
    const cv = $(".model-view", c.el), badge = $(".mv-badge", c.el), btn = $(".btn-view", c.el);
    if (c.view < 0) { cv.hidden = badge.hidden = true; btn.setAttribute("aria-pressed", "false"); return; }
    const m = views[c.view], [H, W, C] = m.meta.input;
    const px = inputFor(c, m.meta);
    c.full = null;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d"), id = ctx.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      for (let k = 0; k < 3; k++) id.data[i * 4 + k] = px[i * C + (C === 1 ? 0 : k)];
      id.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    const same = activeModels().filter((x) => x.meta.input.join("x") === m.meta.input.join("x")).map((x) => x.name);
    badge.textContent = `Vue de ${same.join(", ")} · ${W}×${H}`;
    cv.hidden = badge.hidden = false;
    btn.setAttribute("aria-pressed", "true");
  }

  function renderCard(c) {
    const el = c.el, act = activeModels();
    const preds = $(".preds", el);
    if (c.error) {
      preds.innerHTML = `<li class="pred-err">Oups : ${esc(c.error)}</li>`;
      el.classList.add("done", "err");
      return;
    }
    const done = act.filter((m) => c.results[m.name] && c.results[m.name].probs);

    // --- lignes par modèle ---
    // Nom du modèle sur sa propre ligne (les noms sont longs), puis classe prédite + confiance.
    const head = (m) => `<span class="pmodel" title="${esc(m.name)}"><i class="pdot"></i><span>${esc(m.name)}</span></span>`;
    preds.innerHTML = act.map((m) => {
      const r = c.results[m.name];
      if (!r) return `<li class="pred wait" style="--mc:${m.color}">${head(m)}<span class="plabel">Analyse…</span></li>`;
      if (r.error) return `<li class="pred" style="--mc:${m.color}">${head(m)}<span class="plabel perr">Erreur</span></li>`;
      const p = r.probs[r.best];
      let mark = "";
      const expected = expectedLabel(m.meta, c.truth);
      if (expected) {
        const why = expected === c.truth ? "" : ` (ce modèle ne connaît pas « ${c.truth} » : il devait répondre « ${expected} »)`;
        mark = r.label === expected
          ? `<span class="pmark ok" title="Juste${esc(why)}" aria-label="Juste">${ICON_OK}</span>`
          : `<span class="pmark ko" title="Faux${esc(why)}" aria-label="Faux">${ICON_KO}</span>`;
      } else if (c.truth) mark = `<span class="pmark na" title="Ce modèle ne connaît pas cette classe" aria-label="Non jugé">-</span>`;
      const fresh = r.fresh ? " fresh" : "";
      r.fresh = false;
      return `<li class="pred${fresh}" style="--mc:${m.color}">${head(m)}
        <span class="plabel">${esc(r.label)}</span><span class="pconf">${fmtPct(p)}</span>${mark}
        <span class="ptrack"><span class="pfill" style="width:${(p * 100).toFixed(2)}%"></span></span></li>`;
    }).join("") || `<li class="pred-err">Aucun modèle actif.</li>`;

    // --- consensus (tient compte des classes que chaque modèle connaît) ---
    const cons = $(".consensus", el);
    if (done.length >= 2) {
      const v = consensus(done.map((m) => ({ meta: m.meta, label: c.results[m.name].label })));
      cons.className = "consensus " + (v.agree ? "agree" : "split");
      cons.textContent = v.text;
      cons.hidden = false;
    } else cons.hidden = true;

    // --- « En vrai » : union des classes des modèles actifs ---
    const opts = [];
    act.forEach((m) => m.meta.classes.forEach((n) => { if (!opts.includes(n)) opts.push(n); }));
    const box = $(".truth-opts", el);
    box.innerHTML = opts.map((n) => `<button type="button" class="tbtn" aria-pressed="${c.truth === n}">${esc(n)}</button>`).join("");
    box.querySelectorAll(".tbtn").forEach((b, i) => b.addEventListener("click", () => {
      c.truth = c.truth === opts[i] ? null : opts[i];
      renderCard(c); renderModels(); renderBoard();
    }));

    // --- détails ---
    $(".details", el).innerHTML = act.map((m) => {
      const r = c.results[m.name];
      if (!r || !r.probs) return "";
      const bars = r.probs.map((p, i) => `
        <div class="bar"><span class="name">${esc(m.meta.classes[i])}</span><span class="val">${fmtPct(p)}</span>
        <div class="track"><div class="fill" style="width:${(p * 100).toFixed(2)}%;background:${m.color}"></div></div></div>`).join("");
      const kind = { logits: "logits", probas: "probas", sigmoid: "sigmoïde", logit: "logit" }[m.meta.sortie] || "sortie";
      return `<div class="det" style="--mc:${m.color}"><p class="det-head"><span class="mtag">${esc(m.name)}</span>${m.meta.input[1]}×${m.meta.input[0]} · ${Math.round(r.ms)} ms</p>${bars}
        <p class="logits">${kind} [${r.raw.map((v) => v.toFixed(3)).join(", ")}]</p></div>`;
    }).join("");
  }

  function renderBoard() {
    const board = $("#board");
    const rows = models.filter((m) => m.net && m.active).map((m) => ({ m, s: modelScore(m) })).filter((x) => x.s.n);
    if (!rows.length) { board.innerHTML = ""; return; }
    rows.sort((a, b) => b.s.ok / b.s.n - a.s.ok / a.s.n);
    board.innerHTML = rows.map(({ m, s }, i) =>
      `<span class="bpill${i === 0 && rows.length > 1 ? " top" : ""}" style="--mc:${m.color}"><span class="mtag">${esc(m.name)}</span>${Math.round((s.ok / s.n) * 100)} %<small>${s.ok}/${s.n}</small></span>`).join("");
  }

  // =====================================================================
  // Interactions
  // =====================================================================
  dropzone.classList.add("disabled");
  // Un clic n'importe où dans la zone ouvre le sélecteur (au clavier : bouton « Choisir des images »).
  dropzone.addEventListener("click", (e) => { if (!e.target.closest("#btn-cam")) fileInput.click(); });
  $("#btn-browse").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  fileInput.addEventListener("change", () => { addFiles(fileInput.files); fileInput.value = ""; });

  const ready = () => models.some((m) => m.net);
  ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => {
    e.preventDefault(); if (ready()) dropzone.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "dragleave" && e.relatedTarget) return;
    dropzone.classList.remove("drag");
    if (ev === "drop" && ready() && e.dataTransfer) addFiles(e.dataTransfer.files);
  }));
  document.addEventListener("paste", (e) => {
    if (!ready() || !e.clipboardData) return;
    const files = Array.from(e.clipboardData.items).filter((i) => i.kind === "file").map((i) => i.getAsFile()).filter(Boolean);
    if (files.length) { e.preventDefault(); addFiles(files); }
  });

  $("#btn-clear").addEventListener("click", () => {
    cards.forEach((c) => c.url && URL.revokeObjectURL(c.url));
    results.innerHTML = ""; cards.length = 0;
    empty.hidden = false; resultsHead.hidden = true;
    renderModels(); renderBoard();
  });

  // ---------- webcam ----------
  const cam = $("#cam"), video = $("#cam-video");
  let stream = null;
  $("#btn-cam").addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 } }, audio: false });
      video.srcObject = stream; cam.hidden = false;
      cam.scrollIntoView({ behavior: "smooth", block: "center" });
    } catch (err) {
      setStatus("Webcam indisponible (" + (err.name || "erreur") + ")", ready() ? "ready" : undefined);
    }
  });
  $("#btn-cam-close").addEventListener("click", () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null; cam.hidden = true;
  });
  $("#btn-snap").addEventListener("click", () => {
    if (!video.videoWidth) return;
    const cv = document.createElement("canvas");
    cv.width = video.videoWidth; cv.height = video.videoHeight;
    cv.getContext("2d").drawImage(video, 0, 0);
    cv.toBlob((blob) => {
      addFiles([new File([blob], "photo-" + new Date().toLocaleTimeString("fr-FR").replace(/:/g, "h") + ".png", { type: "image/png" })]);
    }, "image/png");
  });

  loadAll();
})();
