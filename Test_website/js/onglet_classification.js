/* =============================================================================
   Onglet 1 : classification (livrable 1)
   -----------------------------------------------------------------------------
   Chaque image déposée passe dans tous les classifieurs chargés (cochés) :
   redimensionnement comme à l'entraînement -> prédiction -> carte de résultats
   (classe prédite, confiance, consensus, « En vrai » pour noter les modèles).
   ============================================================================= */
(function () {
  "use strict";
  const RUCHE = window.RUCHE;
  const { $, esc, fmtPct, prochaineImage, lireImage, pixelsPleineTaille } = RUCHE.outils;
  const TACHE = "L1_classification";
  const M = RUCHE.modeles;

  const resultats = $("#l1-resultats"), vide = $("#l1-vide"), entete = $("#l1-entete");
  const panneau = $("#l1-modeles"), tpl = $("#tpl-l1-carte");

  const cartes = [];
  let file = Promise.resolve(); // les images sont traitées une par une, dans l'ordre
  const enFile = (f) => { file = file.then(f).catch((e) => console.error(e)); };
  const actifs = () => M.actifs(TACHE);

  // ---------------------------------------------------------------------------
  // Comparer des modèles qui n'ont pas les mêmes classes
  // Le modèle 2 classes (Autre / Photo) range les peintures dans « Autre ».
  // Pour lui, répondre « Autre » sur une peinture est donc juste, et ce n'est
  // pas un désaccord avec le modèle 3 classes qui répond « Peinture ».
  // ---------------------------------------------------------------------------
  const ICON_OK = `<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5 5 9.5 10 3"/></svg>`;
  const ICON_KO = `<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3 9 9M9 3 3 9"/></svg>`;
  const FOURRE_TOUT = ["autre", "autres", "other", "others"];
  const fourreTout = (meta) => meta.classes.find((n) => FOURRE_TOUT.includes(n.toLowerCase())) || null;

  /** Classe que le modèle aurait dû répondre si l'image est vraiment `verite` (null : on ne peut pas juger). */
  function attendue(meta, verite) {
    if (!verite) return null;
    if (meta.classes.includes(verite)) return verite;
    return fourreTout(meta);
  }

  /** Les modèles sont-ils d'accord ? entrees : [{ meta, label }] -> { agree, text } */
  function consensus(entrees) {
    const precises = entrees.filter((e) => e.label !== fourreTout(e.meta)); // réponses autres que « Autre »
    const labels = [...new Set(precises.map((e) => e.label))];
    if (!labels.length) return { agree: true, text: `Unanimes : ${entrees[0].label}` };
    if (labels.length === 1) {
      const L = labels[0];
      // ceux qui ont répondu « Autre » sont cohérents seulement s'ils ne connaissent pas L
      if (entrees.every((e) => e.label === L || !e.meta.classes.includes(L))) {
        return { agree: true, text: precises.length === entrees.length ? `Unanimes : ${L}` : `Cohérents : ${L}` };
      }
    }
    const n = {};
    entrees.forEach((e) => { n[e.label] = (n[e.label] || 0) + 1; });
    return { agree: false, text: "Avis partagés : " + Object.entries(n).sort((a, b) => b[1] - a[1]).map(([l, k]) => `${l} (${k})`).join(" · ") };
  }

  // ---------------------------------------------------------------------------
  // Panneau des modèles
  // ---------------------------------------------------------------------------
  function score(m) {
    let ok = 0, n = 0, ms = 0, k = 0;
    for (const c of cartes) {
      const r = c.resultats[m.nom];
      if (r && r.ms != null) { ms += r.ms; k++; }
      const att = attendue(m.meta, c.verite);
      if (!r || !r.probs || !att) continue;
      n++; if (r.label === att) ok++;
    }
    return { ok, n, ms: k ? ms / k : null };
  }

  function afficherModeles() {
    RUCHE.interface.panneauModeles(panneau, TACHE, {
      vide: `Aucun classifieur. Ajoute-le dans <code>catalogue.json</code> (et, pour le mode navigateur, lance <code>convertir_modeles.bat</code>).`,
      extra: (m) => { const s = score(m); return s.ms != null ? `<span>~${Math.round(s.ms)} ms</span>` : ""; },
    });
    for (const m of M.liste(TACHE)) {
      const el = panneau.querySelector(`[data-modele="${CSS.escape(m.nom)}"] [data-score]`);
      const s = m.meta ? score(m) : { n: 0 };
      if (el && s.n) el.innerHTML = `<span>Précision</span><b>${Math.round((s.ok / s.n) * 100)} %</b><span>${s.ok}/${s.n}</span>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Images
  // ---------------------------------------------------------------------------
  function entreePour(c, meta) { // image redimensionnée à la taille d'entrée du modèle (mise en cache)
    const [H, W, C] = meta.input, cle = `${W}x${H}x${C}`;
    if (!c.entrees[cle]) {
      if (!c.pleine) c.pleine = pixelsPleineTaille(c.img);
      c.entrees[cle] = RUCHE.moteur.redimensionnerBilineaire(c.pleine.rgba, c.pleine.w, c.pleine.h, W, H, C);
    }
    return c.entrees[cle];
  }

  function ajouter(items) {
    vide.hidden = true; entete.hidden = false;
    for (const { fichier, chemin } of items) {
      const c = creerCarte(chemin || fichier.name || "image collée");
      enFile(() => chargerImage(c, fichier).then(() => analyser(c)));
    }
  }

  async function chargerImage(c, fichier) {
    try {
      const { img, url } = await lireImage(fichier);
      c.img = img; c.url = url;
      const im = $("img", c.el);
      im.src = url; im.alt = c.nom;
      $(".res", c.el).textContent = `${img.naturalWidth}×${img.naturalHeight}`;
    } catch (e) {
      c.erreur = e.message;
      afficherCarte(c);
    }
  }

  async function analyser(c) {
    if (!c.img) return;
    for (const m of actifs()) {
      if (c.resultats[m.nom]) continue;
      c.el.classList.remove("done");
      await prochaineImage();
      try {
        const x = entreePour(c, m.meta);
        const t0 = performance.now();
        const { raw, probs } = await m.net.classer(x);
        const best = probs.indexOf(Math.max(...probs));
        c.resultats[m.nom] = { raw, probs, ms: performance.now() - t0, best, label: m.meta.classes[best], fresh: true };
      } catch (e) {
        console.error(e);
        c.resultats[m.nom] = { erreur: e.message };
      }
      afficherCarte(c);
    }
    c.pleine = null; // libère l'image pleine résolution
    c.el.classList.add("done");
    afficherCarte(c);
    afficherModeles();
    afficherClassement();
  }

  // ---------------------------------------------------------------------------
  // Cartes
  // ---------------------------------------------------------------------------
  function creerCarte(nom) {
    const el = tpl.content.firstElementChild.cloneNode(true);
    const c = { el, nom, img: null, url: null, pleine: null, entrees: {}, resultats: {}, verite: null, vue: -1, erreur: null };
    $(".fname-text", el).textContent = nom;
    $(".fname", el).title = nom;
    resultats.prepend(el);
    cartes.push(c);
    $(".btn-view", el).addEventListener("click", () => vueSuivante(c));
    $(".btn-details", el).addEventListener("click", (e) => {
      const d = $(".details", el), on = d.hidden;
      d.hidden = !on; e.currentTarget.setAttribute("aria-pressed", String(on));
    });
    afficherCarte(c);
    return c;
  }

  // « Vue du modèle » : alterne entre les tailles d'entrée des modèles actifs, puis revient à la photo.
  function vueSuivante(c) {
    if (!c.img) return;
    const vues = [], vus = new Set();
    for (const m of actifs()) {
      const cle = m.meta.input.join("x");
      if (!vus.has(cle)) { vus.add(cle); vues.push(m); }
    }
    c.vue = c.vue + 1 >= vues.length ? -1 : c.vue + 1;
    const cv = $(".model-view", c.el), badge = $(".mv-badge", c.el), btn = $(".btn-view", c.el);
    if (c.vue < 0) { cv.hidden = badge.hidden = true; btn.setAttribute("aria-pressed", "false"); return; }
    const m = vues[c.vue], [H, W, C] = m.meta.input;
    const px = entreePour(c, m.meta);
    c.pleine = null;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d"), id = ctx.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      for (let k = 0; k < 3; k++) id.data[i * 4 + k] = px[i * C + (C === 1 ? 0 : k)];
      id.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    const memes = actifs().filter((x) => x.meta.input.join("x") === m.meta.input.join("x")).map((x) => x.nom);
    badge.textContent = `Vue de ${memes.join(", ")} · ${W}×${H}`;
    cv.hidden = badge.hidden = false;
    btn.setAttribute("aria-pressed", "true");
  }

  function afficherCarte(c) {
    const el = c.el, act = actifs(), preds = $(".preds", el);
    if (c.erreur) {
      preds.innerHTML = `<li class="pred-err">Oups : ${esc(c.erreur)}</li>`;
      el.classList.add("done", "err");
      return;
    }
    const faits = act.filter((m) => c.resultats[m.nom] && c.resultats[m.nom].probs);

    // une ligne par modèle : nom, puis classe prédite + confiance, puis barre
    const tete = (m) => `<span class="pmodel" title="${esc(m.nom)}"><i class="pdot"></i><span>${esc(m.nom)}</span></span>`;
    preds.innerHTML = act.map((m) => {
      const r = c.resultats[m.nom];
      if (!r) return `<li class="pred wait" style="--mc:${m.couleur}">${tete(m)}<span class="plabel">Analyse…</span></li>`;
      if (r.erreur) return `<li class="pred" style="--mc:${m.couleur}">${tete(m)}<span class="plabel perr">Erreur</span></li>`;
      const p = r.probs[r.best];
      let marque = "";
      const att = attendue(m.meta, c.verite);
      if (att) {
        const pourquoi = att === c.verite ? "" : ` (ce modèle ne connaît pas « ${c.verite} » : il devait répondre « ${att} »)`;
        marque = r.label === att
          ? `<span class="pmark ok" title="Juste${esc(pourquoi)}" aria-label="Juste">${ICON_OK}</span>`
          : `<span class="pmark ko" title="Faux${esc(pourquoi)}" aria-label="Faux">${ICON_KO}</span>`;
      } else if (c.verite) marque = `<span class="pmark na" title="Ce modèle ne connaît pas cette classe" aria-label="Non jugé">-</span>`;
      const neuf = r.fresh ? " fresh" : "";
      r.fresh = false;
      return `<li class="pred${neuf}" style="--mc:${m.couleur}">${tete(m)}
        <span class="plabel">${esc(r.label)}</span><span class="pconf">${fmtPct(p)}</span>${marque}
        <span class="ptrack"><span class="pfill" style="width:${(p * 100).toFixed(2)}%"></span></span></li>`;
    }).join("") || `<li class="pred-err">Aucun classifieur chargé : coche-en un dans le panneau « Les classifieurs ».</li>`;

    // consensus (tient compte des classes que chaque modèle connaît)
    const cons = $(".consensus", el);
    if (faits.length >= 2) {
      const v = consensus(faits.map((m) => ({ meta: m.meta, label: c.resultats[m.nom].label })));
      cons.className = "consensus " + (v.agree ? "agree" : "split");
      cons.textContent = v.text;
      cons.hidden = false;
    } else cons.hidden = true;

    // « En vrai » : union des classes des modèles actifs
    const opts = [];
    act.forEach((m) => m.meta.classes.forEach((n) => { if (!opts.includes(n)) opts.push(n); }));
    const boite = $(".truth-opts", el);
    boite.innerHTML = opts.map((n) => `<button type="button" class="tbtn" aria-pressed="${c.verite === n}">${esc(n)}</button>`).join("");
    boite.querySelectorAll(".tbtn").forEach((b, i) => b.addEventListener("click", () => {
      c.verite = c.verite === opts[i] ? null : opts[i];
      afficherCarte(c); afficherModeles(); afficherClassement();
    }));

    // détails : probabilité de chaque classe + sorties brutes
    $(".details", el).innerHTML = act.map((m) => {
      const r = c.resultats[m.nom];
      if (!r || !r.probs) return "";
      const barres = r.probs.map((p, i) => `
        <div class="bar"><span class="name">${esc(m.meta.classes[i])}</span><span class="val">${fmtPct(p)}</span>
        <div class="track"><div class="fill" style="width:${(p * 100).toFixed(2)}%;background:${m.couleur}"></div></div></div>`).join("");
      const sorte = { logits: "logits", probas: "probas", sigmoid: "sigmoïde", logit: "logit" }[m.meta.sortie] || "sortie";
      return `<div class="det" style="--mc:${m.couleur}"><p class="det-head"><span class="mtag">${esc(m.nom)}</span>${m.meta.input[1]}×${m.meta.input[0]} · ${Math.round(r.ms)} ms</p>${barres}
        <p class="logits">${sorte} [${r.raw.map((v) => v.toFixed(3)).join(", ")}]</p></div>`;
    }).join("");
  }

  function afficherClassement() {
    const board = $("#l1-classement");
    const lignes = actifs().map((m) => ({ m, s: score(m) })).filter((x) => x.s.n);
    if (!lignes.length) { board.innerHTML = ""; return; }
    lignes.sort((a, b) => b.s.ok / b.s.n - a.s.ok / a.s.n);
    board.innerHTML = lignes.map(({ m, s }, i) =>
      `<span class="bpill${i === 0 && lignes.length > 1 ? " top" : ""}" style="--mc:${m.couleur}"><span class="mtag">${esc(m.nom)}</span>${Math.round((s.ok / s.n) * 100)} %<small>${s.ok}/${s.n}</small></span>`).join("");
  }

  // ---------------------------------------------------------------------------
  // Branchements
  // ---------------------------------------------------------------------------
  RUCHE.interface.zoneDepot({
    zone: $("#l1-depot"),
    inputs: [$("#l1-fichiers")],
    boutons: [[$("#l1-parcourir"), $("#l1-fichiers")]],
    camera: { bouton: $("#l1-webcam"), cadre: $("#l1-cam"), video: $("#l1-cam-video"), capturer: $("#l1-capturer"), fermer: $("#l1-cam-fermer") },
    recevoir: ajouter,
  });

  $("#l1-vider").addEventListener("click", () => {
    cartes.forEach((c) => c.url && URL.revokeObjectURL(c.url));
    resultats.innerHTML = ""; cartes.length = 0;
    vide.hidden = false; entete.hidden = true;
    afficherModeles(); afficherClassement();
  });

  RUCHE.on("modeles", () => {
    afficherModeles();
    cartes.forEach(afficherCarte);
    cartes.forEach((c) => enFile(() => analyser(c))); // un modèle activé ou chargé : on complète les cartes
    afficherClassement();
  });
})();
