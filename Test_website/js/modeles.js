/* =============================================================================
   La Ruche : chargement des modèles
   -----------------------------------------------------------------------------
   modeles/index.js liste les modèles par tâche (généré par convertir_modeles.py) :
       RUCHE.liste({ "L1_classification": [...], "L2_debruitage": [...] })
   puis, pour chaque modèle, modeles/<tâche>/<nom>/modele.js appelle RUCHE.modele(meta)
   et chaque poids_<i>.js appelle RUCHE.poids(i, "base64...").
   Chaque fois qu'un modèle change d'état, l'événement « modeles » est émis.
   ============================================================================= */
(function () {
  "use strict";
  const RUCHE = window.RUCHE;
  const { chargerScript, stockage, prochaineImage } = RUCHE.outils;

  const TACHES = { L1_classification: "classifieur", L2_debruitage: "débruiteur" };
  const COULEURS = ["#f5b301", "#8a5cf6", "#14a39a", "#e8613c", "#3a86e8", "#5da83a", "#d6479b", "#8c6a3a"];
  const modeles = Object.fromEntries(Object.keys(TACHES).map((t) => [t, []]));
  let backend = null;

  // Fonctions appelées par les fichiers générés
  let enCours = null;
  RUCHE.liste = (l) => { RUCHE._liste = l; };
  RUCHE.modele = (meta) => { if (enCours) enCours.meta = meta; };
  RUCHE.poids = (i, s) => { if (enCours) enCours.parts[i] = s; };

  const cleOff = (m) => (m.tache === "L1_classification" ? "ruche.off." : `ruche.off.${m.tache}.`) + m.nom;

  async function charger(statut) {
    statut("Démarrage de TensorFlow.js…");
    try {
      backend = await RUCHE.moteur.demarrer();
    } catch (e) {
      statut(e.message, "error");
      return;
    }
    try {
      await chargerScript("modeles/index.js");
    } catch (_) {
      statut("modeles/index.js introuvable : lance convertir_modeles.bat", "error");
      RUCHE.emettre("modeles");
      return;
    }
    const liste = RUCHE._liste;
    if (!liste || Array.isArray(liste)) {
      statut("modeles/index.js est d'une ancienne version : relance convertir_modeles.bat", "error");
      RUCHE.emettre("modeles");
      return;
    }
    let k = 0;
    for (const tache of Object.keys(TACHES)) {
      for (const nom of liste[tache] || []) {
        const m = { nom, tache, couleur: COULEURS[k++ % COULEURS.length], actif: true, meta: null, net: null, erreur: null };
        m.actif = stockage.get(cleOff(m)) !== "1";
        modeles[tache].push(m);
      }
    }
    RUCHE.emettre("modeles");
    const tous = Object.values(modeles).flat();
    if (!tous.length) { statut("Aucun modèle : lance convertir_modeles.bat", "error"); return; }

    for (let i = 0; i < tous.length; i++) {
      const m = tous[i];
      statut(`Les abeilles rapportent ${m.nom}… (${i + 1}/${tous.length})`);
      try {
        enCours = { meta: null, parts: [] };
        const dossier = `modeles/${m.tache}/${encodeURIComponent(m.nom)}/`;
        await chargerScript(dossier + "modele.js");
        if (!enCours.meta) throw new Error("modele.js invalide");
        for (const p of enCours.meta.parts) await chargerScript(dossier + p);
        await prochaineImage();
        m.meta = enCours.meta;
        m.net = RUCHE.moteur.creerModele(enCours.meta, enCours.parts.join(""));
        enCours = null;
        if (backend === "webgl") await controler(m, statut);
      } catch (e) {
        console.error(e);
        m.erreur = e.message;
        m.net = null;
      } finally {
        enCours = null;
      }
      RUCHE.emettre("modeles");
    }
    const ok = tous.filter((m) => m.net).length;
    if (!ok) { statut("Aucun modèle n'a pu être chargé", "error"); return; }
    // Un modèle reste faux sur la carte graphique, même avec de petites tuiles : tout passe sur le processeur,
    // plus lent mais exact (le calcul y est celui de Keras).
    const faux = tous.filter((m) => m.controle && !m.controle.ok);
    if (faux.length) {
      await RUCHE.moteur.passerSurProcesseur();
      backend = "cpu";
      faux.forEach((m) => { m.controle = { ok: true, ecart: null, processeur: true }; });
      RUCHE.emettre("modeles");
    }
    const ou = backend === "webgl" ? "carte graphique" : faux.length ? "processeur (carte graphique inexacte)" : "processeur (lent)";
    statut(`${ok} modèle${ok > 1 ? "s" : ""} prêt${ok > 1 ? "s" : ""} · ${ou}`, "ready");
  }

  /** Vérifie un modèle sur la carte graphique (voir controler() dans moteur.js). En cas d'écart avec Keras,
      on réessaie avec des tuiles plus petites ; le résultat est gardé dans m.controle. */
  async function controler(m, statut) {
    statut(`Vérification de ${m.nom} sur la carte graphique…`);
    let c = await m.net.controler();
    if (!c.ok && m.meta.sortie === "image") {
      console.warn(`${m.nom} : écart ${c.ecart} avec Keras, nouvel essai avec des tuiles plus petites`);
      m.net.reduireTuiles();
      c = await m.net.controler();
    }
    if (!c.ok) console.warn(`${m.nom} : résultats faux sur cette carte graphique (écart ${c.ecart})`);
    m.controle = c;
  }

  RUCHE.modeles = {
    TACHES,
    charger,
    backend: () => backend,
    liste: (tache) => modeles[tache],
    prets: (tache) => modeles[tache].filter((m) => m.net),
    actifs: (tache) => modeles[tache].filter((m) => m.net && m.actif),
    trouver: (tache, nom) => modeles[tache].find((m) => m.nom === nom && m.net) || null,
    basculer(m, on) {
      m.actif = on;
      stockage.set(cleOff(m), on ? "0" : "1");
      RUCHE.emettre("modeles", m);
    },
  };
})();
