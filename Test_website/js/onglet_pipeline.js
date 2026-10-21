/* =============================================================================
   Onglet Pipeline : livrable 1 -> livrable 2 -> livrable 3
   -----------------------------------------------------------------------------
   Un dossier d'images entre, les étapes s'enchaînent comme en production :
     1. classification (L1) : on ne garde que les images dont P(Photo) >= seuil ;
     2. débruitage (L2) : chaque photo gardée est débruitée. Avec « Simuler un lot bruité »,
        le bruit est ajouté juste avant cette étape : le classifieur voit les images d'origine ;
     3. captioning (L3) : BLIP écrit une légende pour chaque photo débruitée
        (réglages de l'onglet L3 ; étape sautée si aucun légendeur n'est choisi).
   Les modèles choisis sont chargés au lancement s'ils ne le sont pas encore.
   À la fin : photos débruitées et rapport (.csv, légendes comprises) dans un .zip.
   ============================================================================= */
(function () {
  "use strict";
  const RUCHE = window.RUCHE;
  const O = RUCHE.outils;
  const { $, esc, nombre, fmtPct, fmtDb, prochaineImage } = O;
  const M = RUCHE.modeles;
  const L1 = "L1_classification", L2 = "L2_debruitage", L3 = "L3_captioning";

  const selClf = $("#pl-classifieur"), selDen = $("#pl-debruiteur"), selLeg = $("#pl-legendeur");
  const seuil = $("#pl-seuil"), taille = $("#pl-taille");
  const simuler = $("#pl-simuler"), sigma = $("#pl-sigma"), sigmaVal = $("#pl-sigma-val");
  const btnLancer = $("#pl-lancer"), btnArreter = $("#pl-arreter"), btnVider = $("#pl-vider");
  const grille1 = $("#pl-grille1"), grille2 = $("#pl-grille2"), grille3 = $("#pl-grille3");

  let items = [];       // [{ fichier, chemin }] : images de la source
  let passage = null;   // exécution en cours ou terminée
  const urls = [];      // URL d'objets à libérer au prochain passage
  const url = (blob) => { const u = URL.createObjectURL(blob); urls.push(u); return u; };
  const niveauxGris = (s) => nombre(s * 255, 1);
  const sansExtension = (chemin) => chemin.replace(/\.[^./]+$/, "");
  const pluriel = (n, mot, s = "s") => `${n} ${mot}${n > 1 ? s : ""}`;

  // ---------------------------------------------------------------------------
  // Réglages
  // ---------------------------------------------------------------------------
  /** Liste tous les modèles de la tâche : ceux qui ne sont pas encore chargés le seront au lancement. */
  function remplirSelect(sel, tache, cle, parDefaut, aucun = null) {
    const tousM = M.liste(tache), dispo = tousM.filter((m) => m.disponible || m.etat === "pret");
    const voulu = [O.stockage.get(cle), parDefaut(dispo)].find((n) => n === "" ? aucun != null : dispo.some((m) => m.nom === n));
    sel.innerHTML = (aucun != null ? `<option value="">${esc(aucun)}</option>` : "") + tousM.map((m) => {
      const etat = m.etat === "pret" ? "chargé" : m.etat === "chargement" ? "chargement…" : m.disponible ? "chargé au lancement" : "indisponible";
      return `<option value="${esc(m.nom)}" ${m.disponible || m.etat === "pret" ? "" : "disabled"}>${esc(m.nom)} (${etat})</option>`;
    }).join("");
    if (voulu != null) sel.value = voulu;
    else if (aucun != null) sel.value = "";
    sel.disabled = (!dispo.length && aucun == null) || !!(passage && passage.enCours);
  }

  function afficherReglages() {
    // par défaut : le classifieur qui connaît le plus de classes (il sait écarter les peintures)
    const classes = (m) => (m.meta && m.meta.classes ? m.meta.classes.length : /peinture/i.test(m.nom) ? 3 : 2);
    remplirSelect(selClf, L1, "ruche.pl.clf", (p) => (p.slice().sort((a, b) => classes(b) - classes(a))[0] || {}).nom);
    remplirSelect(selDen, L2, "ruche.pl.den", (p) => (p[0] || {}).nom);
    remplirSelect(selLeg, L3, "ruche.pl.leg", (p) => (p[0] || {}).nom, "aucun (étape 3 sautée)");
    $("#pl-sigma-champ").hidden = !simuler.checked;
    sigmaVal.textContent = `${nombre(Number(sigma.value), 2)} (≈ ${Math.round(sigma.value * 255)} niveaux de gris)`;
    const pret = items.length && selClf.value && selDen.value && !(passage && passage.enCours);
    btnLancer.disabled = !pret;
    const attente = $("#pl-attente");
    attente.hidden = !!(M.liste(L1).length && M.liste(L2).length);
    attente.textContent = M.mode() ? "Aucun classifieur ou débruiteur au catalogue." : "Recherche des modèles…";
    $("#pl-aide-leg").textContent = M.liste(L3).some((m) => m.disponible)
      ? "Réglages de génération : ceux de l'onglet L3 Légendes."
      : "BLIP a besoin du serveur local (lancer_serveur.bat).";
  }

  [[selClf, "ruche.pl.clf"], [selDen, "ruche.pl.den"], [selLeg, "ruche.pl.leg"]].forEach(([s, cle]) => s.addEventListener("change", () => O.stockage.set(cle, s.value)));
  [seuil, taille, simuler, sigma].forEach((el) => {
    const cle = "ruche.pl." + el.id.slice(3);
    const v = O.stockage.get(cle);
    if (v != null) {
      if (el.type === "checkbox") el.checked = v === "1";
      else if (el.tagName !== "SELECT" || [...el.options].some((o) => o.value === v)) el.value = v; // option encore proposée
    }
    el.addEventListener("input", afficherReglages);
    el.addEventListener("change", () => O.stockage.set(cle, el.type === "checkbox" ? (el.checked ? "1" : "0") : el.value));
  });

  // ---------------------------------------------------------------------------
  // Source : un dossier (sous-dossiers compris) ou une sélection d'images
  // ---------------------------------------------------------------------------
  function definirSource(nouveaux) {
    if (passage && passage.enCours) return;
    items = nouveaux.slice().sort((a, b) => O.triNaturel(a.chemin, b.chemin));
    const dossiers = [...new Set(items.map((it) => (it.chemin.includes("/") ? it.chemin.split("/")[0] : "")))];
    const nom = dossiers.length === 1 && dossiers[0] ? `dossier « ${dossiers[0]} »` : "sélection";
    $("#pl-source-info").innerHTML = `<b>${pluriel(items.length, "image")}</b> · ${esc(nom)}`;
    $("#pl-source").classList.add("chargee");
    vider(false);
    majFlux({ source: items.length });
    afficherReglages();
  }

  // ---------------------------------------------------------------------------
  // Affichage
  // ---------------------------------------------------------------------------
  function majFlux({ source, etape1, etape2, etape3 } = {}) {
    if (source != null) $("#pl-flux-source").textContent = pluriel(source, "image");
    if (etape1 != null) $("#pl-flux-1").textContent = etape1;
    if (etape2 != null) $("#pl-flux-2").textContent = etape2;
    if (etape3 != null) $("#pl-flux-3").textContent = etape3;
  }

  function progression(etape, fait, total, texte) {
    const sec = $(`#pl-etape${etape}`);
    sec.hidden = false;
    $(".progress span", sec).style.width = total ? `${(100 * fait) / total}%` : "0%";
    $(".etape-etat", sec).textContent = texte;
  }

  function carteClassification(l) {
    const el = document.createElement("article");
    el.className = "pcard";
    el.innerHTML = `<div class="pthumb"><img alt="" loading="lazy"></div>
      <div class="pbody"><p class="pname" title="${esc(l.chemin)}">${esc(l.chemin.split("/").pop())}</p><p class="pclasse">Analyse…</p><span class="pbadge"></span></div>`;
    grille1.appendChild(el);
    l.carte1 = el;
  }

  function majCarteClassification(l) {
    const el = l.carte1;
    if (l.urlOriginale) $("img", el).src = l.urlOriginale;
    el.classList.toggle("gardee", !!l.gardee);
    el.classList.toggle("ecartee", !l.gardee);
    el.classList.toggle("err", !!l.erreur);
    $(".pclasse", el).innerHTML = l.erreur ? `Erreur : ${esc(l.erreur)}`
      : `<span><b>${esc(l.classe)}</b></span><span>P(Photo) ${fmtPct(l.pPhoto)}</span>`;
    $(".pbadge", el).textContent = l.erreur ? "Ignorée" : l.gardee ? "Gardée" : "Écartée";
  }

  function carteDebruitage(l, den) {
    const el = document.createElement("article");
    el.className = "card dcard pcard2 done";
    const cmp = RUCHE.interface.comparateur();
    el.innerHTML = `<div class="dthumb"></div><div class="card-body">
      <p class="fname"><span class="fname-text"></span><span class="res"></span></p><p class="pmesure"></p>
      <div class="card-foot"><button class="btn btn-small btn-ghost btn-png" type="button">PNG</button></div></div>`;
    $(".dthumb", el).appendChild(cmp.el);
    $(".fname-text", el).textContent = l.chemin.split("/").pop();
    $(".fname", el).title = l.chemin;
    if (l.erreur2) {
      $(".pmesure", el).innerHTML = `<span class="perr">Erreur : ${esc(l.erreur2)}</span>`;
      $(".btn-png", el).disabled = true;
      cmp.definir({ avant: l.urlOriginale, apres: "", etiquettes: ["Avant", "Après"] });
    } else {
      $(".res", el).textContent = l.taille;
      cmp.definir({ avant: l.urlAvant, apres: l.urlApres, ratio: l.ratio, etiquettes: [l.simule ? "Bruitée" : "Avant", `Après · ${den.nom}`] });
      $(".pmesure", el).innerHTML = l.simule
        ? `PSNR ${fmtDb(l.psnrAvant)} → <b>${fmtDb(l.psnrApres)}</b> <span class="chip ${l.psnrApres >= l.psnrAvant ? "gain" : "perte"}">${fmtDb(l.psnrApres - l.psnrAvant, true)}</span>`
        : `Bruit estimé σ ${niveauxGris(l.bruitAvant)} → <b>${niveauxGris(l.bruitApres)}</b> <small>niveaux de gris</small>`;
      $(".btn-png", el).addEventListener("click", () => O.telecharger(l.blob, sansExtension(l.chemin.split("/").pop()) + ".png"));
    }
    el.style.setProperty("--mc", den.couleur);
    grille2.appendChild(el);
  }

  function carteLegende(l, leg) {
    const el = document.createElement("article");
    el.className = "pcard pcard3";
    el.style.setProperty("--mc", leg.couleur);
    el.innerHTML = `<div class="pthumb"><img alt=""></div><div class="pbody">
      <p class="pname" title="${esc(l.chemin)}">${esc(l.chemin.split("/").pop())}</p>
      <p class="plegende">${l.erreur3 ? `<span class="perr">Erreur : ${esc(l.erreur3)}</span>`
        : `${l.amorce ? `<span class="lamorce">${esc(l.amorce)}</span> ` : ""}${esc(l.suite || l.legende)}`}</p>
      ${l.msLegende != null ? `<span class="pbadge">${Math.round(l.msLegende)} ms</span>` : ""}</div>`;
    $("img", el).src = l.urlApres;
    $("img", el).alt = l.legende || "";
    grille3.appendChild(el);
  }

  // ---------------------------------------------------------------------------
  // Exécution
  // ---------------------------------------------------------------------------
  async function lancer() {
    const clf = M.trouver(L1, selClf.value), den = M.trouver(L2, selDen.value), leg = selLeg.value ? M.trouver(L3, selLeg.value) : null;
    if (!items.length || !clf || !den) return;

    vider(false);
    const p = passage = {
      enCours: true, arret: false, lignes: [], clf, den, leg, debut: performance.now(),
      seuil: Number(seuil.value), maxCote: Number(taille.value), simule: simuler.checked, sigma: Number(sigma.value),
      reglages: leg && RUCHE.legendes ? RUCHE.legendes.reglages() : null,
    };
    etatBoutons();

    // --- Modèles choisis : chargés s'ils ne le sont pas encore (comme si on cochait leur case) ---
    for (const m of [clf, den, leg].filter(Boolean)) {
      if (m.etat === "pret") continue;
      progression(1, 0, 1, `Chargement de ${m.nom}…`);
      try {
        await M.activer(m);
      } catch (e) {
        p.enCours = false;
        progression(1, 0, 1, `Impossible de charger ${m.nom} : ${e.message}`);
        etatBoutons();
        return;
      }
    }
    const iPhoto = clf.meta.classes.findIndex((n) => n.toLowerCase() === "photo");
    if (iPhoto < 0) {
      p.enCours = false;
      RUCHE.emettre("message", `${clf.nom} n'a pas de classe « Photo »`);
      etatBoutons();
      return;
    }
    const [H, W, C] = clf.meta.input;

    // --- Étape 1 : classification, on garde les photos ---
    for (let i = 0; i < items.length && !p.arret; i++) {
      const it = items[i], l = { chemin: it.chemin, fichier: it.fichier, simule: p.simule };
      p.lignes.push(l);
      carteClassification(l);
      progression(1, i, items.length, `Image ${i + 1} sur ${items.length}…`);
      try {
        const { img, url: u } = await O.lireImage(it.fichier);
        urls.push(u);
        l.urlOriginale = u; l.largeur = img.naturalWidth; l.hauteur = img.naturalHeight;
        // le classifieur voit toujours l'image d'origine : le bruit simulé n'est ajouté qu'à l'étape 2
        const px = O.pixelsPleineTaille(img);
        const x = RUCHE.moteur.redimensionnerBilineaire(px.rgba, px.w, px.h, W, H, C);
        const { probs } = await clf.net.classer(x);
        l.classe = clf.meta.classes[probs.indexOf(Math.max(...probs))];
        l.pPhoto = probs[iPhoto];
        l.gardee = l.pPhoto >= p.seuil;
      } catch (e) {
        console.error(e);
        l.erreur = e.message; l.gardee = false;
      }
      majCarteClassification(l);
      const gardees = p.lignes.filter((x) => x.gardee).length;
      majFlux({ etape1: `${pluriel(gardees, "photo")} gardée${gardees > 1 ? "s" : ""}` });
      await prochaineImage();
    }
    const photos = p.lignes.filter((l) => l.gardee);
    const comptes = {};
    p.lignes.forEach((l) => { const k = l.erreur ? "Erreurs" : l.classe; comptes[k] = (comptes[k] || 0) + 1; });
    progression(1, p.lignes.length, items.length, `${pluriel(p.lignes.length, "image")} analysée${p.lignes.length > 1 ? "s" : ""} avec ${clf.nom} · seuil P(Photo) ≥ ${fmtPct(p.seuil)}`);
    $("#pl-resume1").innerHTML = Object.entries(comptes).map(([k, n]) => `<span class="chip">${esc(k)} : ${n}</span>`).join("") +
      `<span class="chip chip-fort">${photos.length} gardée${photos.length > 1 ? "s" : ""} pour l'étape 2</span>`;

    // --- Étape 2 : débruitage des photos gardées (lot simulé : bruit ajouté juste avant le débruiteur) ---
    let faites = 0;
    for (let i = 0; i < photos.length && !p.arret; i++) {
      const l = photos[i];
      progression(2, i, photos.length, `Photo ${i + 1} sur ${photos.length}…`);
      try {
        const { img, url: u } = await O.lireImage(l.fichier);
        URL.revokeObjectURL(u);
        const t = O.imageDeTravail(img, p.maxCote);
        const entree = p.simule ? O.ajouterBruit(t.pixels, p.sigma, O.graine(l.chemin)) : t.pixels;
        const t0 = performance.now();
        const sortie = await den.net.debruiter(entree, t.h, t.w);
        l.ms = performance.now() - t0;
        l.taille = `${t.w}×${t.h}`; l.ratio = t.w / t.h;
        l.bruitAvant = O.bruitEstime(entree, t.w, t.h);
        l.bruitApres = O.bruitEstime(sortie, t.w, t.h);
        if (p.simule) { l.psnrAvant = O.psnr(entree, t.pixels); l.psnrApres = O.psnr(sortie, t.pixels); }
        l.blob = await O.versPNG(sortie, t.w, t.h);
        l.urlApres = url(l.blob);
        l.urlAvant = url(await O.versPNG(entree, t.w, t.h));
        l.sortie = "photos_debruitees/" + sansExtension(l.chemin) + ".png";
        faites++;
      } catch (e) {
        console.error(e);
        l.erreur2 = e.message;
      }
      carteDebruitage(l, den);
      majFlux({ etape2: `${pluriel(faites, "photo")} débruitée${faites > 1 ? "s" : ""}` });
      await prochaineImage();
    }
    const fin = photos.filter((l) => l.blob);
    progression(2, photos.length && !p.arret ? photos.length : faites, photos.length,
      photos.length ? `${pluriel(faites, "photo")} débruitée${faites > 1 ? "s" : ""} avec ${den.nom}` : "Aucune photo à débruiter");
    const moy = (f) => fin.reduce((s, l) => s + f(l), 0) / fin.length;
    $("#pl-resume2").innerHTML = !fin.length ? "" : p.simule
      ? `<span class="chip chip-fort">Gain moyen ${fmtDb(moy((l) => l.psnrApres - l.psnrAvant), true)}</span><span class="chip">PSNR moyen ${fmtDb(moy((l) => l.psnrAvant))} → ${fmtDb(moy((l) => l.psnrApres))}</span>`
      : `<span class="chip">Bruit estimé moyen σ ${niveauxGris(moy((l) => l.bruitAvant))} → ${niveauxGris(moy((l) => l.bruitApres))}</span>`;

    // --- Étape 3 : une légende par photo débruitée ---
    if (!leg) {
      progression(3, 0, 0, "Étape sautée : aucun légendeur choisi.");
      majFlux({ etape3: "sautée" });
    } else {
      let ecrites = 0;
      for (let i = 0; i < fin.length && !p.arret; i++) {
        const l = fin[i];
        progression(3, i, fin.length, `Photo ${i + 1} sur ${fin.length}…`);
        try {
          const r = await leg.net.legender(l.blob, p.reglages);
          l.legende = r.legende; l.suite = r.suite; l.amorce = r.amorce; l.msLegende = r.ms;
          ecrites++;
        } catch (e) {
          console.error(e);
          l.erreur3 = e.message;
        }
        carteLegende(l, leg);
        majFlux({ etape3: `${pluriel(ecrites, "légende")} écrite${ecrites > 1 ? "s" : ""}` });
        await prochaineImage();
      }
      progression(3, fin.length && !p.arret ? fin.length : ecrites, fin.length,
        fin.length ? `${pluriel(ecrites, "légende")} écrite${ecrites > 1 ? "s" : ""} par ${leg.nom}` : "Aucune photo à légender");
      const ms = fin.filter((l) => l.msLegende != null).map((l) => l.msLegende);
      $("#pl-resume3").innerHTML = ms.length ? `<span class="chip">${Math.round(ms.reduce((a, b) => a + b, 0) / ms.length)} ms par légende</span>`
        + `<span class="chip">amorce « ${esc(p.reglages.amorce || "aucune")} » · faisceau ${p.reglages.num_beams}</span>` : "";
    }

    // --- Bilan ---
    p.enCours = false;
    p.duree = (performance.now() - p.debut) / 1000;
    afficherBilan(p);
    etatBoutons();
  }

  function afficherBilan(p) {
    const fin = p.lignes.filter((l) => l.blob), legendees = p.lignes.filter((l) => l.legende);
    $("#pl-bilan").hidden = false;
    $("#pl-bilan-texte").innerHTML = (p.arret ? "<b>Pipeline arrêtée.</b> " : "") +
      `${pluriel(p.lignes.length, "image")} classée${p.lignes.length > 1 ? "s" : ""}, ` +
      `${p.lignes.filter((l) => l.gardee).length} gardée${p.lignes.filter((l) => l.gardee).length > 1 ? "s" : ""}, ` +
      `${fin.length} débruitée${fin.length > 1 ? "s" : ""}` + (p.leg ? `, ${legendees.length} légendée${legendees.length > 1 ? "s" : ""}` : "") +
      ` en ${nombre(p.duree, 1)} s` + (p.simule ? ` · lot simulé (bruit σ ${nombre(p.sigma, 2)} ajouté juste avant l'étape 2)` : "");
    $("#pl-zip").disabled = !fin.length;
    $("#pl-tableau tbody").innerHTML = p.lignes.map((l) => {
      const etape2 = !l.gardee ? `<span class="muted">écartée</span>` : l.erreur2 ? `<span class="perr">erreur</span>` : !l.blob ? `<span class="muted">non traitée</span>`
        : p.simule ? `${fmtDb(l.psnrAvant)} → ${fmtDb(l.psnrApres)} (${fmtDb(l.psnrApres - l.psnrAvant, true)})`
          : `σ ${niveauxGris(l.bruitAvant)} → ${niveauxGris(l.bruitApres)}`;
      const etape3 = !l.blob || !p.leg ? `<span class="muted">–</span>` : l.erreur3 ? `<span class="perr">erreur</span>`
        : l.legende ? esc(l.legende) : `<span class="muted">non traitée</span>`;
      return `<tr class="${l.gardee ? "" : "ligne-ecartee"}"><td title="${esc(l.chemin)}">${esc(l.chemin)}</td>
        <td>${l.erreur ? `<span class="perr">${esc(l.erreur)}</span>` : esc(l.classe)}</td>
        <td class="num">${l.pPhoto != null ? fmtPct(l.pPhoto) : "–"}</td><td>${etape2}</td><td class="td-legende">${etape3}</td></tr>`;
    }).join("");
  }

  function rapportCSV(p) {
    const lignes = [["fichier", "largeur", "hauteur", "classe_predite", "p_photo", "gardee", "taille_traitee", "bruit_estime_avant", "bruit_estime_apres",
      "psnr_avant_db", "psnr_apres_db", "gain_db", "temps_debruitage_ms", "sortie", "legende", "temps_legende_ms", "erreur"]];
    for (const l of p.lignes) {
      lignes.push([l.chemin, l.largeur, l.hauteur, l.classe, l.pPhoto, l.gardee ? "oui" : "non", l.taille,
        l.bruitAvant, l.bruitApres, l.psnrAvant, l.psnrApres, l.psnrAvant != null ? l.psnrApres - l.psnrAvant : null,
        l.ms != null ? Math.round(l.ms) : null, l.sortie, l.legende, l.msLegende != null ? Math.round(l.msLegende) : null,
        l.erreur || l.erreur2 || l.erreur3 || ""]);
    }
    lignes.push([]);
    lignes.push(["classifieur", p.clf.nom, "seuil_p_photo", p.seuil, "debruiteur", p.den.nom, "taille_max", p.maxCote,
      "bruit_simule", p.simule ? p.sigma : "non", "legendeur", p.leg ? p.leg.nom : "aucun",
      ...(p.reglages ? Object.entries(p.reglages).flat() : [])]);
    return O.versCSV(lignes);
  }

  async function telechargerZip() {
    const p = passage;
    if (!p) return;
    const fichiers = [];
    for (const l of p.lignes.filter((x) => x.blob)) fichiers.push({ nom: l.sortie, donnees: new Uint8Array(await l.blob.arrayBuffer()) });
    fichiers.push({ nom: "rapport.csv", donnees: new Uint8Array(await rapportCSV(p).arrayBuffer()) });
    O.telecharger(O.creerZip(fichiers), "pipeline_leyenda.zip");
  }

  function etatBoutons() {
    const enCours = !!(passage && passage.enCours);
    btnArreter.hidden = !enCours;
    btnLancer.hidden = enCours;
    btnVider.disabled = enCours;
    [selClf, selDen, selLeg, seuil, taille, simuler, sigma].forEach((el) => { el.disabled = enCours; });
    $("#pl-source").classList.toggle("disabled", enCours);
    afficherReglages();
  }

  function vider(toutEffacer) {
    if (passage && passage.enCours) return;
    urls.splice(0).forEach((u) => URL.revokeObjectURL(u));
    passage = null;
    grille1.innerHTML = ""; grille2.innerHTML = ""; grille3.innerHTML = "";
    ["#pl-etape1", "#pl-etape2", "#pl-etape3", "#pl-bilan"].forEach((s) => { $(s).hidden = true; });
    $("#pl-resume1").innerHTML = ""; $("#pl-resume2").innerHTML = ""; $("#pl-resume3").innerHTML = "";
    majFlux({ etape1: "photos gardées", etape2: "photos débruitées", etape3: "légendes" });
    if (toutEffacer) {
      items = [];
      $("#pl-source-info").textContent = "Aucun dossier choisi";
      $("#pl-source").classList.remove("chargee");
      majFlux({ source: 0 });
    }
    afficherReglages();
  }

  // ---------------------------------------------------------------------------
  // Branchements
  // ---------------------------------------------------------------------------
  RUCHE.interface.zoneDepot({
    zone: $("#pl-source"),
    inputs: [$("#pl-input-dossier"), $("#pl-input-images")],
    boutons: [[$("#pl-btn-dossier"), $("#pl-input-dossier")], [$("#pl-btn-images"), $("#pl-input-images")]],
    recevoir: definirSource,
    pret: () => !(passage && passage.enCours),
  });
  btnLancer.addEventListener("click", () => lancer().catch((e) => {
    console.error(e);
    if (passage) passage.enCours = false;
    RUCHE.emettre("message", "Pipeline interrompue : " + e.message);
    etatBoutons();
  }));
  btnArreter.addEventListener("click", () => { if (passage) passage.arret = true; });
  btnVider.addEventListener("click", () => vider(true));
  $("#pl-zip").addEventListener("click", () => telechargerZip().catch((e) => RUCHE.emettre("message", e.message)));
  $("#pl-csv").addEventListener("click", () => { if (passage) O.telecharger(rapportCSV(passage), "rapport_pipeline.csv"); });

  RUCHE.on("modeles", afficherReglages);
  vider(true);
})();
