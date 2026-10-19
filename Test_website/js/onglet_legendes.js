/* =============================================================================
   Onglet L3 : légendes (livrable 3, captioning avec BLIP)
   -----------------------------------------------------------------------------
   Chaque image déposée est envoyée au serveur local, où BLIP (models/
   blip-captioning-base, enregistré par Livrable3/L3_2_BLIP.ipynb) écrit une phrase qui la
   décrit. Les réglages de génération (amorce, recherche en faisceau, longueur,
   répétitions, échantillonnage) se changent ici ; chaque nouvelle légende d'une
   même image est gardée, pour comparer l'effet des réglages.
   Les réglages servent aussi à l'étape 3 de l'onglet Pipeline.
   ============================================================================= */
(function () {
  "use strict";
  const RUCHE = window.RUCHE;
  const O = RUCHE.outils;
  const { $, $$, esc, nombre, prochaineImage } = O;
  const TACHE = "L3_captioning";
  const M = RUCHE.modeles;

  const resultats = $("#l3-resultats"), vide = $("#l3-vide"), entete = $("#l3-entete");
  const panneau = $("#l3-modeles"), tpl = $("#tpl-l3-carte");

  // ---------------------------------------------------------------------------
  // Réglages de génération (défauts = Livrable3/L3_2_BLIP.ipynb et workshop)
  // ---------------------------------------------------------------------------
  const DEFAUTS = {
    amorce: "a photograph of", num_beams: 5, max_new_tokens: 30, min_new_tokens: 0, no_repeat_ngram_size: 2,
    repetition_penalty: 1.0, length_penalty: 1.0, do_sample: false, temperature: 1.0, top_p: 0.9, top_k: 50,
  };
  const AMORCES = ["a photograph of", "a picture of", ""];
  const reglages = { ...DEFAUTS };
  try {
    const r = JSON.parse(O.stockage.get("ruche.l3.reglages") || "null");
    if (r && typeof r === "object") for (const k of Object.keys(DEFAUTS)) if (typeof r[k] === typeof DEFAUTS[k]) reglages[k] = r[k];
  } catch (_) { /* réglages illisibles : défauts */ }

  const champs = {
    num_beams: { el: $("#l3-beams"), txt: (v) => (v === 1 ? "1 (glouton : le mot le plus probable à chaque pas)" : `${v} hypothèses gardées à chaque pas`) },
    max_new_tokens: { el: $("#l3-max"), txt: (v) => `${v} sous-mots au plus` },
    min_new_tokens: { el: $("#l3-min"), txt: (v) => (v ? `${v} sous-mots au moins` : "pas de minimum") },
    no_repeat_ngram_size: { el: $("#l3-ngram"), txt: (v) => (v === 0 ? "aucune interdiction" : v === 1 ? "chaque sous-mot une seule fois" : `pas deux fois la même suite de ${v} sous-mots`) },
    repetition_penalty: { el: $("#l3-repet"), txt: (v) => (v === 1 ? "1 (aucune pénalité)" : `${nombre(v, 2)} (mots déjà écrits moins probables)`) },
    length_penalty: { el: $("#l3-longueur"), txt: (v) => `${nombre(v, 1)} (${v > 1 ? "favorise les phrases longues" : v < 1 ? "favorise les phrases courtes" : "neutre"})` },
    temperature: { el: $("#l3-temp"), txt: (v) => `${nombre(v, 2)} (${v < 1 ? "plus prudent" : v > 1 ? "plus inventif" : "probabilités du modèle"})` },
    top_p: { el: $("#l3-topp"), txt: (v) => `${nombre(v, 2)} (mots les plus probables jusqu'à ${Math.round(v * 100)} % de probabilité cumulée)` },
  };
  const selAmorce = $("#l3-amorce"), txtAmorce = $("#l3-amorce-texte"), chkSample = $("#l3-sample");

  const ENTIERS = new Set(["num_beams", "max_new_tokens", "min_new_tokens", "no_repeat_ngram_size", "top_k"]);
  const lireChamp = (k) => (ENTIERS.has(k) ? Math.round(Number(champs[k].el.value)) : Number(champs[k].el.value));

  function afficherReglages() {
    const choix = AMORCES.includes(reglages.amorce) ? reglages.amorce : "autre";
    selAmorce.value = choix;
    txtAmorce.hidden = choix !== "autre";
    if (choix === "autre" && document.activeElement !== txtAmorce) txtAmorce.value = reglages.amorce;
    for (const [k, c] of Object.entries(champs)) {
      c.el.value = reglages[k];
      $(`output[for="${c.el.id}"]`).textContent = c.txt(reglages[k]);
    }
    chkSample.checked = reglages.do_sample;
    $("#l3-champs-sample").hidden = !reglages.do_sample;
    champs.length_penalty.el.disabled = reglages.num_beams === 1;
    $("#l3-resume").textContent = resume(reglages);
    const defaut = Object.keys(DEFAUTS).every((k) => reglages[k] === DEFAUTS[k]);
    $("#l3-defaut").disabled = defaut;
  }

  function changer() {
    O.stockage.set("ruche.l3.reglages", JSON.stringify(reglages));
    afficherReglages();
    $("#l3-relancer").disabled = !cartes.length;
  }

  selAmorce.addEventListener("change", () => {
    reglages.amorce = selAmorce.value === "autre" ? (txtAmorce.value || "a photo of") : selAmorce.value;
    changer();
    if (selAmorce.value === "autre") txtAmorce.focus();
  });
  txtAmorce.addEventListener("input", () => { reglages.amorce = txtAmorce.value.slice(0, 120); changer(); });
  for (const [k, c] of Object.entries(champs)) {
    c.el.addEventListener("input", () => { reglages[k] = lireChamp(k); $(`output[for="${c.el.id}"]`).textContent = c.txt(reglages[k]); });
    c.el.addEventListener("change", () => { reglages[k] = lireChamp(k); changer(); });
  }
  chkSample.addEventListener("change", () => { reglages.do_sample = chkSample.checked; changer(); });
  $("#l3-defaut").addEventListener("click", () => { Object.assign(reglages, DEFAUTS); changer(); });

  function resume(r) {
    const morceaux = [`amorce « ${r.amorce || "aucune"} »`, r.num_beams > 1 ? `faisceau ${r.num_beams}` : "glouton",
      `≤ ${r.max_new_tokens} sous-mots`];
    if (r.min_new_tokens) morceaux.push(`≥ ${r.min_new_tokens}`);
    if (r.no_repeat_ngram_size) morceaux.push(`sans répétition de ${r.no_repeat_ngram_size}-grammes`);
    if (r.repetition_penalty !== 1) morceaux.push(`pénalité de répétition ${nombre(r.repetition_penalty, 2)}`);
    if (r.num_beams > 1 && r.length_penalty !== 1) morceaux.push(`pénalité de longueur ${nombre(r.length_penalty, 1)}`);
    if (r.do_sample) morceaux.push(`échantillonnage (T ${nombre(r.temperature, 2)}, top-p ${nombre(r.top_p, 2)})`);
    return morceaux.join(" · ");
  }

  // ---------------------------------------------------------------------------
  // Cartes
  // ---------------------------------------------------------------------------
  const cartes = [];
  let file = Promise.resolve();
  const enFile = (f) => { file = file.then(f).catch((e) => console.error(e)); };
  const legendeur = () => M.actifs(TACHE)[0] || null;

  function ajouter(items) {
    vide.hidden = true; entete.hidden = false;
    for (const { fichier, chemin } of items) {
      const c = creerCarte(chemin || fichier.name || "image collée");
      enFile(async () => {
        try {
          const { img, url } = await O.lireImage(fichier);
          c.img = img; c.url = url;
          $("img", c.el).src = url; $("img", c.el).alt = c.nom;
          $(".res", c.el).textContent = `${img.naturalWidth}×${img.naturalHeight}`;
          c.image = await imageAEnvoyer(img);
        } catch (e) {
          c.erreur = e.message;
        }
        afficherCarte(c);
        await legender(c);
      });
    }
    $("#l3-relancer").disabled = false;
  }

  /** L'image envoyée à BLIP : réduite à 1024 pixels au plus (BLIP travaille en 384 × 384), en JPEG de qualité 95. */
  function imageAEnvoyer(img) {
    const k = Math.min(1, 1024 / Math.max(img.naturalWidth, img.naturalHeight));
    const cv = document.createElement("canvas");
    cv.width = Math.max(1, Math.round(img.naturalWidth * k)); cv.height = Math.max(1, Math.round(img.naturalHeight * k));
    const ctx = cv.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
    return new Promise((res, rej) => cv.toBlob((b) => (b ? res(b) : rej(new Error("conversion de l'image impossible"))), "image/jpeg", 0.95));
  }

  /** Une légende de plus pour la carte, avec les réglages actuels (si elle n'existe pas déjà pour ces réglages). */
  async function legender(c) {
    const m = legendeur();
    if (!c.image || c.erreur || !m) { afficherCarte(c); return; }
    const r = { ...reglages }, cle = JSON.stringify(r);
    if (c.versions.some((v) => v.cle === cle && !v.erreur) && !r.do_sample) { afficherCarte(c); return; }
    c.attente = true;
    c.el.classList.remove("done");
    afficherCarte(c);
    await prochaineImage();
    let v;
    try {
      const res = await m.net.legender(c.image, r);
      v = { cle, reglages: r, legende: res.legende, suite: res.suite, amorce: res.amorce, ms: res.ms, modele: m.nom };
    } catch (e) {
      v = { cle, reglages: r, erreur: e.message };
    }
    c.versions.unshift(v);
    c.attente = false;
    c.el.classList.add("done");
    afficherCarte(c);
    afficherModeles();
  }

  function creerCarte(nom) {
    const el = tpl.content.firstElementChild.cloneNode(true);
    const c = { el, nom, img: null, url: null, image: null, versions: [], attente: false, erreur: null };
    $(".fname-text", el).textContent = nom;
    $(".fname", el).title = nom;
    $(".btn-copier", el).addEventListener("click", async (e) => {
      const v = c.versions.find((x) => !x.erreur);
      if (!v) return;
      try { await navigator.clipboard.writeText(v.legende); e.currentTarget.textContent = "Copiée"; } catch (_) { RUCHE.emettre("message", "Copie impossible dans ce navigateur"); }
      setTimeout(() => { $(".btn-copier", el).textContent = "Copier"; }, 1500);
    });
    $(".btn-relancer", el).addEventListener("click", () => enFile(() => legender(c)));
    resultats.prepend(el);
    cartes.push(c);
    afficherCarte(c);
    return c;
  }

  function texteVersion(v) {
    if (v.erreur) return `<span class="perr">Erreur : ${esc(v.erreur)}</span>`;
    return `${v.amorce ? `<span class="lamorce">${esc(v.amorce)}</span> ` : ""}${esc(v.suite || v.legende)}`;
  }

  function afficherCarte(c) {
    const el = c.el, m = legendeur(), [v, ...anciennes] = c.versions;
    const leg = $(".legende", el), info = $(".lmeta", el);
    if (c.erreur) { leg.innerHTML = `<span class="perr">Oups : ${esc(c.erreur)}</span>`; el.classList.add("done", "err"); info.textContent = ""; return; }
    if (c.attente) leg.innerHTML = `<span class="lattente">BLIP regarde l'image…</span>`;
    else if (v) leg.innerHTML = texteVersion(v);
    else leg.innerHTML = `<span class="lattente">${M.serveurPret() ? "Coche BLIP dans le panneau « Le légendeur » pour écrire la légende." : "En attente du serveur local (lancer_serveur.bat)."}</span>`;
    if (!c.attente && (v || !m)) el.classList.add("done");
    info.innerHTML = v && !v.erreur ? `<span>${Math.round(v.ms)} ms</span><span>${esc(resume(v.reglages))}</span>` : "";
    const hist = $(".lhistorique", el);
    hist.hidden = !anciennes.length;
    $("summary", hist).textContent = `${anciennes.length} légende${anciennes.length > 1 ? "s" : ""} précédente${anciennes.length > 1 ? "s" : ""}`;
    $("ol", hist).innerHTML = anciennes.map((x) => `<li><p>${texteVersion(x)}</p><small>${esc(resume(x.reglages))}</small></li>`).join("");
    $(".btn-copier", el).disabled = !v || !!v.erreur;
    $(".btn-relancer", el).disabled = !m || !c.image || c.attente;
  }

  // ---------------------------------------------------------------------------
  // Panneau du modèle
  // ---------------------------------------------------------------------------
  function afficherModeles() {
    const faites = cartes.flatMap((c) => c.versions.filter((v) => !v.erreur));
    RUCHE.interface.panneauModeles(panneau, TACHE, {
      vide: M.mode() === "navigateur"
        ? "BLIP (224 millions de paramètres) tourne avec PyTorch : il a besoin du serveur local. Double-clique sur <code>lancer_serveur.bat</code>, puis recharge la page."
        : "Aucun légendeur dans <code>catalogue.json</code> (rubrique <code>L3_captioning</code>).",
      corps: (m) => {
        const mt = m.meta, ms = faites.filter((v) => v.modele === m.nom).map((v) => v.ms);
        const date = mt.date ? mt.date.split("@")[0].split("-").reverse().join("/") : null;
        return `
          ${mt.description ? `<p class="model-desc">${esc(mt.description)}</p>` : ""}
          <p class="model-sub">models/${esc(mt.source)}${date ? " · enregistré le " + date : ""}</p>
          <div class="model-meta"><span>${nombre(mt.params / 1e6, 0)} M param.</span><span>${esc(mt.precision)}</span>${ms.length ? `<span>~${Math.round(ms.reduce((a, b) => a + b, 0) / ms.length)} ms</span>` : ""}</div>
          <p class="model-ctrl">Calculé par PyTorch sur ${esc(mt.appareil || "?")}.</p>
          <ol class="stack">
            <li><span>Image</span><span>${mt.image}×${mt.image} · ${(mt.image / mt.patch) ** 2} patchs ${mt.patch}×${mt.patch}</span></li>
            <li><span>Encodeur ViT</span><span>${mt.couches_vision} couches · ${mt.dim} · ${nombre(mt.params_vision / 1e6, 0)} M</span></li>
            <li><span>Décodeur texte</span><span>${mt.couches_texte} couches · ${nombre(mt.params_texte / 1e6, 0)} M</span></li>
            <li><span>Vocabulaire</span><span>${mt.vocabulaire.toLocaleString("fr-FR")} sous-mots</span></li>
          </ol>`;
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Export
  // ---------------------------------------------------------------------------
  function exporterCSV() {
    const lignes = [["fichier", "legende", "temps_ms", ...Object.keys(DEFAUTS)]];
    for (const c of cartes.slice().reverse()) {
      for (const v of c.versions.slice().reverse()) {
        if (v.erreur) continue;
        lignes.push([c.nom, v.legende, Math.round(v.ms), ...Object.keys(DEFAUTS).map((k) => v.reglages[k])]);
      }
    }
    O.telecharger(O.versCSV(lignes), "legendes_blip.csv");
  }

  // ---------------------------------------------------------------------------
  // Branchements
  // ---------------------------------------------------------------------------
  RUCHE.interface.zoneDepot({
    zone: $("#l3-depot"),
    inputs: [$("#l3-fichiers")],
    boutons: [[$("#l3-parcourir"), $("#l3-fichiers")]],
    camera: { bouton: $("#l3-webcam"), cadre: $("#l3-cam"), video: $("#l3-cam-video"), capturer: $("#l3-capturer"), fermer: $("#l3-cam-fermer") },
    recevoir: ajouter,
  });
  $("#l3-vider").addEventListener("click", () => {
    cartes.forEach((c) => c.url && URL.revokeObjectURL(c.url));
    resultats.innerHTML = ""; cartes.length = 0;
    vide.hidden = false; entete.hidden = true;
    $("#l3-relancer").disabled = true;
    afficherModeles();
  });
  $("#l3-relancer").addEventListener("click", () => cartes.forEach((c) => enFile(() => legender(c))));
  $("#l3-csv").addEventListener("click", exporterCSV);

  RUCHE.on("modeles", (m) => {
    if (m && m.tache !== TACHE) return;
    afficherModeles();
    cartes.forEach(afficherCarte);
    if (legendeur()) cartes.filter((c) => !c.versions.length).forEach((c) => enFile(() => legender(c)));
  });
  afficherReglages();
  $("#l3-relancer").disabled = true;

  /** Réglages actuels (utilisés par l'étape 3 de la pipeline). */
  RUCHE.legendes = { reglages: () => ({ ...reglages }), resume };
})();
