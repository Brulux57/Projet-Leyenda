/* =============================================================================
   Onglet 2 : débruitage (livrable 2)
   -----------------------------------------------------------------------------
   Chaque image déposée passe dans tous les débruiteurs activés. Deux modes :
   - « bruit simulé » (par défaut, comme dans le notebook L2) : l'image déposée est
     la référence propre, on lui ajoute un bruit gaussien de sigma choisi, puis on
     débruite. Le gain est mesuré exactement : PSNR et SSIM avant / après ;
   - « image déjà bruitée » : l'image est débruitée telle quelle. Sans image propre
     de référence, le gain ne se mesure pas : on affiche le bruit estimé avant / après.
   ============================================================================= */
(function () {
  "use strict";
  const RUCHE = window.RUCHE;
  const O = RUCHE.outils;
  const { $, $$, esc, nombre, fmtDb, prochaineImage } = O;
  const TACHE = "L2_debruitage";
  const M = RUCHE.modeles;

  const resultats = $("#l2-resultats"), vide = $("#l2-vide"), entete = $("#l2-entete");
  const panneau = $("#l2-modeles"), tpl = $("#tpl-l2-carte");

  // ---------------------------------------------------------------------------
  // Réglages (gardés d'une visite à l'autre)
  // ---------------------------------------------------------------------------
  const reglages = {
    mode: O.stockage.get("ruche.l2.mode") === "reel" ? "reel" : "simule",
    sigma: Number(O.stockage.get("ruche.l2.sigma")) || 0.15,
    maxCote: O.stockage.get("ruche.l2.taille") != null ? Number(O.stockage.get("ruche.l2.taille")) : 1024,
  };
  const niveauxGris = (s) => nombre(s * 255, 1); // écart-type sur 0..1 -> en niveaux de gris (0..255)

  const cartes = [];
  let file = Promise.resolve();
  const enFile = (f) => { file = file.then(f).catch((e) => console.error(e)); };
  const actifs = () => M.actifs(TACHE);
  const simule = () => reglages.mode === "simule";

  // ---------------------------------------------------------------------------
  // Traitement d'une image
  // ---------------------------------------------------------------------------
  /** Image d'entrée du débruiteur pour les réglages actuels (le bruit simulé est reproductible). */
  function preparer(c) {
    if (!c.entree) {
      c.travail = O.imageDeTravail(c.img, reglages.maxCote);
      c.entree = simule() ? O.ajouterBruit(c.travail.pixels, reglages.sigma, O.graine(c.nom)) : c.travail.pixels;
      c.taille = { w: c.travail.w, h: c.travail.h };
    }
    return { travail: c.travail, entree: c.entree };
  }

  async function analyser(c) {
    if (!c.img || c.erreur) return;
    const version = c.version, perime = () => c.version !== version; // réglages changés pendant le calcul
    try {
      const { travail, entree } = preparer(c);
      const { w, h } = travail;
      if (!c.urls.entree) { // première préparation pour ces réglages : image d'entrée et mesures « avant »
        const urls = { entree: URL.createObjectURL(await O.versPNG(entree, w, h)) };
        if (simule()) urls.propre = URL.createObjectURL(await O.versPNG(travail.pixels, w, h));
        if (perime()) { Object.values(urls).forEach((u) => URL.revokeObjectURL(u)); return; }
        c.urls = urls;
        c.avant = { bruit: O.bruitEstime(entree, w, h) };
        if (simule()) Object.assign(c.avant, { psnr: O.psnr(entree, travail.pixels), ssim: O.ssim(entree, travail.pixels, w, h) });
        afficherCarte(c);
      }
      for (const m of actifs()) {
        if (c.resultats[m.nom]) continue;
        c.el.classList.remove("done");
        await prochaineImage();
        const t0 = performance.now();
        const sortie = await m.net.debruiter(entree, h, w);
        const ms = performance.now() - t0;
        const blob = await O.versPNG(sortie, w, h);
        if (perime()) return; // un autre passage est déjà prévu dans la file
        const r = { ms, blob, url: URL.createObjectURL(blob), bruit: O.bruitEstime(sortie, w, h) };
        if (simule()) { r.psnr = O.psnr(sortie, travail.pixels); r.ssim = O.ssim(sortie, travail.pixels, w, h); }
        c.resultats[m.nom] = r;
        if (!c.choisi || !actifs().some((x) => x.nom === c.choisi && c.resultats[x.nom])) c.choisi = m.nom;
        afficherCarte(c);
      }
    } catch (e) {
      console.error(e);
      if (perime()) return;
      c.erreur = e.message;
    }
    if (perime()) return;
    c.travail = c.entree = null; // libère la mémoire ; preparer() sait les refaire à l'identique
    c.el.classList.add("done");
    afficherCarte(c);
    afficherClassement();
    afficherModeles();
  }

  function oublierResultats(c) {
    c.version++;
    [c.urls.entree, c.urls.propre, ...Object.values(c.resultats).map((r) => r.url)].forEach((u) => u && URL.revokeObjectURL(u));
    c.urls = {}; c.resultats = {}; c.avant = null; c.travail = c.entree = c.taille = null; c.erreur = null; c.originale = false;
  }

  // ---------------------------------------------------------------------------
  // Cartes
  // ---------------------------------------------------------------------------
  function ajouter(items) {
    vide.hidden = true; entete.hidden = false;
    for (const { fichier, chemin } of items) {
      const c = creerCarte(chemin || fichier.name || "image collée");
      enFile(async () => {
        try {
          const { img, url } = await O.lireImage(fichier);
          c.img = img; c.url = url;
          $(".res", c.el).textContent = `${img.naturalWidth}×${img.naturalHeight}`;
        } catch (e) {
          c.erreur = e.message;
          afficherCarte(c);
          return;
        }
        await analyser(c);
      });
    }
  }

  function creerCarte(nom) {
    const el = tpl.content.firstElementChild.cloneNode(true);
    const cmp = RUCHE.interface.comparateur();
    $(".dthumb", el).prepend(cmp.el);
    const c = { el, nom, cmp, img: null, url: null, travail: null, entree: null, taille: null, urls: {}, avant: null, resultats: {}, choisi: null, originale: false, version: 0, erreur: null };
    $(".fname-text", el).textContent = nom;
    $(".fname", el).title = nom;
    $(".btn-originale", el).addEventListener("click", () => { c.originale = !c.originale; afficherCarte(c); });
    $(".btn-agrandir", el).addEventListener("click", (e) => {
      const on = el.classList.toggle("wide");
      e.currentTarget.setAttribute("aria-pressed", String(on));
      if (on) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    $(".btn-png", el).addEventListener("click", () => {
      const r = c.resultats[c.choisi];
      if (r) O.telecharger(r.blob, `${nomSansExtension(c.nom)}_debruitee_${c.choisi}.png`);
    });
    resultats.prepend(el);
    cartes.push(c);
    afficherCarte(c);
    return c;
  }

  const nomSansExtension = (chemin) => chemin.split("/").pop().replace(/\.[^.]+$/, "");

  function afficherCarte(c) {
    const el = c.el, act = actifs(), liste = $(".dlist", el);
    if (c.erreur) {
      liste.innerHTML = `<li class="pred-err">Oups : ${esc(c.erreur)}</li>`;
      $(".dgain", el).hidden = true;
      el.classList.add("done", "err");
      return;
    }
    el.classList.remove("err");
    const r = c.resultats[c.choisi];
    const sim = simule() && c.urls.propre;

    // image avant / après
    const etiquetteAvant = sim ? (c.originale ? "Originale" : `Bruitée σ ${nombre(reglages.sigma, 2)}`) : "Avant";
    c.cmp.definir({
      avant: (c.originale && c.urls.propre) || c.urls.entree || c.url || null,
      apres: r ? r.url : "",
      ratio: c.img ? c.img.naturalWidth / c.img.naturalHeight : null,
      etiquettes: [etiquetteAvant, r ? `Après · ${c.choisi}` : "Après"],
    });
    if (c.img) $(".res", el).textContent = `${c.img.naturalWidth}×${c.img.naturalHeight}` +
      (c.taille && (c.taille.w !== c.img.naturalWidth || c.taille.h !== c.img.naturalHeight) ? ` → ${c.taille.w}×${c.taille.h}` : "");
    const bo = $(".btn-originale", el);
    bo.hidden = !sim;
    bo.setAttribute("aria-pressed", String(c.originale));
    $(".btn-png", el).disabled = !r;

    // gain du modèle affiché
    const g = $(".dgain", el);
    g.hidden = !r || !c.avant;
    if (r && c.avant) {
      g.style.setProperty("--mc", (M.trouver(TACHE, c.choisi) || {}).couleur || "var(--honey)");
      g.innerHTML = sim
        ? `<b>${fmtDb(r.psnr - c.avant.psnr, true)}</b><span>gain de PSNR avec ${esc(c.choisi)}<br>SSIM ${nombre(c.avant.ssim, 2)} → ${nombre(r.ssim, 2)}</span>`
        : `<b>σ ${niveauxGris(c.avant.bruit)} → ${niveauxGris(r.bruit)}</b><span>bruit estimé en niveaux de gris avec ${esc(c.choisi)}<br>gain non mesurable sans image propre</span>`;
    }

    // une ligne par débruiteur actif ; un clic l'affiche dans le comparateur
    liste.innerHTML = act.map((m) => {
      const x = c.resultats[m.nom];
      const tete = `<button type="button" class="dpick" data-nom="${esc(m.nom)}" aria-pressed="${m.nom === c.choisi}" ${x ? "" : "disabled"}><span class="mtag">${esc(m.nom)}</span></button>`;
      if (!x) return `<li class="drow wait" style="--mc:${m.couleur}">${tete}<span class="dval">Débruitage…</span></li>`;
      const mesure = sim && c.avant
        ? `<span class="dval">PSNR ${fmtDb(c.avant.psnr)} → <b>${fmtDb(x.psnr)}</b></span><span class="chip ${x.psnr >= c.avant.psnr ? "gain" : "perte"}">${fmtDb(x.psnr - c.avant.psnr, true)}</span>`
        : `<span class="dval">bruit estimé σ ${niveauxGris(c.avant.bruit)} → <b>${niveauxGris(x.bruit)}</b></span>`;
      return `<li class="drow" style="--mc:${m.couleur}">${tete}${mesure}<span class="dms">${Math.round(x.ms)} ms</span></li>`;
    }).join("") || `<li class="pred-err">Aucun débruiteur actif.</li>`;
    $$(".dpick", liste).forEach((b) => b.addEventListener("click", () => { c.choisi = b.dataset.nom; afficherCarte(c); }));
  }

  function afficherClassement() {
    const board = $("#l2-classement");
    const lignes = actifs().map((m) => {
      const faits = cartes.filter((c) => c.resultats[m.nom] && c.avant);
      if (!faits.length) return null;
      const moy = (f) => faits.reduce((s, c) => s + f(c), 0) / faits.length;
      return { m, n: faits.length, v: simule() ? moy((c) => c.resultats[m.nom].psnr - c.avant.psnr) : moy((c) => c.resultats[m.nom].bruit) };
    }).filter(Boolean);
    lignes.sort((a, b) => (simule() ? b.v - a.v : a.v - b.v));
    board.innerHTML = lignes.map(({ m, n, v }, i) => `<span class="bpill${i === 0 && lignes.length > 1 ? " top" : ""}" style="--mc:${m.couleur}">
      <span class="mtag">${esc(m.nom)}</span>${simule() ? fmtDb(v, true) : "σ " + niveauxGris(v)}<small>moy. sur ${n}</small></span>`).join("");
  }

  function afficherModeles() {
    RUCHE.interface.panneauModeles(panneau, TACHE, {
      vide: `Aucun débruiteur. Ajoute-le dans <code>catalogue.json</code> puis lance <code>convertir_modeles.bat</code>.`,
      extra: (m) => {
        const t = cartes.map((c) => c.resultats[m.nom]).filter(Boolean);
        return t.length ? `<span>~${Math.round(t.reduce((s, r) => s + r.ms, 0) / t.length)} ms</span>` : "";
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Réglages
  // ---------------------------------------------------------------------------
  const sigmaInput = $("#l2-sigma"), sigmaVal = $("#l2-sigma-val"), tailleInput = $("#l2-taille");

  function afficherReglages() {
    // valeur enregistrée par une ancienne version du site, absente de la liste : on revient à 1024 px
    if (![...tailleInput.options].some((o) => Number(o.value) === reglages.maxCote)) reglages.maxCote = 1024;
    $$("#l2-mode [data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === reglages.mode)));
    $("#l2-sigma-champ").hidden = !simule();
    sigmaInput.value = reglages.sigma;
    sigmaVal.textContent = `${nombre(reglages.sigma, 2)} (≈ ${Math.round(reglages.sigma * 255)} niveaux de gris)`;
    tailleInput.value = String(reglages.maxCote);
    $("#l2-aide-mode").innerHTML = simule()
      ? "L'image déposée sert de référence propre : on lui ajoute un bruit gaussien, puis on mesure le PSNR et le SSIM avant et après le débruitage, comme dans le notebook."
      : "L'image est débruitée telle quelle. Sans version propre de référence, le gain ne peut pas être mesuré : on affiche le bruit estimé avant et après (indicatif, les détails fins comptent aussi).";
  }

  function appliquerReglages() {
    O.stockage.set("ruche.l2.mode", reglages.mode);
    O.stockage.set("ruche.l2.sigma", String(reglages.sigma));
    O.stockage.set("ruche.l2.taille", String(reglages.maxCote));
    afficherReglages();
    cartes.forEach((c) => { oublierResultats(c); afficherCarte(c); enFile(() => analyser(c)); });
    afficherClassement();
  }

  $$("#l2-mode [data-mode]").forEach((b) => b.addEventListener("click", () => {
    if (reglages.mode === b.dataset.mode) return;
    reglages.mode = b.dataset.mode;
    appliquerReglages();
  }));
  sigmaInput.addEventListener("input", () => { sigmaVal.textContent = `${nombre(Number(sigmaInput.value), 2)} (≈ ${Math.round(sigmaInput.value * 255)} niveaux de gris)`; });
  sigmaInput.addEventListener("change", () => { reglages.sigma = Number(sigmaInput.value); appliquerReglages(); });
  tailleInput.addEventListener("change", () => { reglages.maxCote = Number(tailleInput.value); appliquerReglages(); });
  afficherReglages();

  // ---------------------------------------------------------------------------
  // Branchements
  // ---------------------------------------------------------------------------
  const depot = RUCHE.interface.zoneDepot({
    zone: $("#l2-depot"),
    inputs: [$("#l2-fichiers")],
    boutons: [[$("#l2-parcourir"), $("#l2-fichiers")]],
    camera: { bouton: $("#l2-webcam"), cadre: $("#l2-cam"), video: $("#l2-cam-video"), capturer: $("#l2-capturer"), fermer: $("#l2-cam-fermer") },
    recevoir: ajouter,
    pret: () => M.prets(TACHE).length > 0,
  });
  depot.activer(false);

  $("#l2-vider").addEventListener("click", () => {
    cartes.forEach((c) => { oublierResultats(c); if (c.url) URL.revokeObjectURL(c.url); });
    resultats.innerHTML = ""; cartes.length = 0;
    vide.hidden = false; entete.hidden = true;
    afficherModeles(); afficherClassement();
  });

  RUCHE.on("modeles", () => {
    depot.activer(M.prets(TACHE).length > 0);
    afficherModeles();
    cartes.forEach(afficherCarte);
    cartes.forEach((c) => enFile(() => analyser(c)));
    afficherClassement();
  });
})();
