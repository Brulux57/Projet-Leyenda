/* =============================================================================
   La Ruche : registre des modèles
   -----------------------------------------------------------------------------
   Deux façons de calculer, choisies au démarrage :
   - serveur local (serveur.py) : les modèles tournent avec TensorFlow / PyTorch
     sur la carte graphique, le navigateur envoie les pixels. C'est le mode
     normal, et le seul qui fait tourner BLIP (livrable 3) ;
   - navigateur (secours, si le serveur n'est pas lancé) : TensorFlow.js
     exécute les exports de convertir_modeles.py (modeles/<tâche>/<nom>/).

   Aucun modèle n'est chargé au démarrage : chacun est chargé quand on coche sa
   case, et déchargé (mémoire libérée) quand on la décoche. Chaque fois qu'un
   modèle change d'état, l'événement « modeles » est émis.

   Un modèle : { nom, tache, couleur, description, disponible, etat, meta, net,
   erreur, progression } avec etat = "libre" | "chargement" | "pret" | "erreur".
   net expose classer(pixels), debruiter(pixels, h, w) ou legender(image, reglages).
   ============================================================================= */
(function () {
  "use strict";
  const RUCHE = window.RUCHE;
  const { chargerScript, prochaineImage } = RUCHE.outils;

  const TACHES = { L1_classification: "classifieur", L2_debruitage: "débruiteur", L3_captioning: "légendeur" };
  const COULEURS = ["#f5b301", "#8a5cf6", "#14a39a", "#e8613c", "#3a86e8", "#5da83a", "#d6479b", "#8c6a3a"];
  const modeles = Object.fromEntries(Object.keys(TACHES).map((t) => [t, []]));
  let mode = null;          // "serveur" | "navigateur"
  let etatServeur = null;   // dernière réponse de /api/etat
  let backend = null;       // backend TensorFlow.js (mode navigateur), démarré au premier modèle chargé
  let statut = () => {};

  // Fichiers générés par convertir_modeles.py (mode navigateur)
  let enCours = null;
  RUCHE.liste = (l) => { RUCHE._liste = l; };
  RUCHE.modele = (meta) => { if (enCours) enCours.meta = meta; };
  RUCHE.poids = (i, s) => { if (enCours) enCours.parts[i] = s; };

  const tous = () => Object.values(modeles).flat();
  const emettre = (m) => { RUCHE.emettre("modeles", m); majStatut(); };

  function nouveau(tache, nom, k, extra = {}) {
    return { nom, tache, couleur: COULEURS[k % COULEURS.length], description: "", disponible: true, etat: "libre",
      meta: null, net: null, erreur: null, progression: null, controle: null, ...extra };
  }

  // ---------------------------------------------------------------------------
  // Démarrage
  // ---------------------------------------------------------------------------
  async function demarrer(fonctionStatut) {
    statut = fonctionStatut;
    statut("Recherche du serveur local…");
    const e = await RUCHE.serveur.etat();
    if (e) modeServeur(e);
    else await modeNavigateur();
    // Serveur lancé après l'ouverture de la page, ou arrêté puis relancé : on le détecte (toutes les 4 s quand on
    // l'utilise, toutes les 10 s sinon ; la console du navigateur note chaque essai manqué, c'est sans gravité)
    const boucle = async () => { await surveiller(); setTimeout(boucle, mode === "serveur" ? 4000 : 10000); };
    setTimeout(boucle, mode === "serveur" ? 4000 : 10000);
  }

  function modeServeur(e) {
    mode = "serveur";
    etatServeur = e;
    let k = 0;
    for (const tache of Object.keys(TACHES)) {
      modeles[tache].forEach((m) => { if (m.net && m.net.liberer) m.net.liberer(); });
      modeles[tache] = (e.modeles[tache] || []).map((d) => {
        const m = nouveau(tache, d.nom, k++, { description: d.description, disponible: d.disponible, fichier: d.fichier });
        synchroniser(m, d);
        return m;
      });
    }
    emettre();
  }

  /** Met un modèle d'accord avec l'état renvoyé par le serveur (modèle chargé par une page précédente, serveur relancé...). */
  function synchroniser(m, d) {
    const avant = m.etat + m.disponible;
    m.disponible = d.disponible;
    if (m.etat === "chargement") return false;      // chargement lancé par cette page : activer() s'en occupe
    if (d.charge && d.meta) {
      if (m.etat !== "pret") { m.meta = d.meta; m.net = reseauServeur(m); m.controle = { serveur: true, appareil: d.meta.appareil }; m.etat = "pret"; }
    } else if (m.etat === "pret") {
      m.etat = "libre"; m.net = null;                // serveur relancé : le modèle n'est plus en mémoire
    }
    return avant !== m.etat + m.disponible;
  }

  async function modeNavigateur() {
    mode = "navigateur";
    try {
      await chargerScript("modeles/index.js");
    } catch (_) {
      statut("Serveur local non lancé et modeles/index.js introuvable : lance lancer_serveur.bat", "error");
      emettre();
      return;
    }
    const liste = RUCHE._liste || {};
    let k = 0;
    for (const tache of Object.keys(TACHES)) {
      modeles[tache] = (liste[tache] || []).map((nom) => nouveau(tache, nom, k++, tache === "L3_captioning"
        ? { disponible: false, erreur: null, description: "BLIP a besoin du serveur local (lancer_serveur.bat)." } : {}));
    }
    emettre();
  }

  async function surveiller() {
    const e = await RUCHE.serveur.etat(2500);
    if (mode === "navigateur") {
      if (!e) return;
      if (tous().some((m) => m.etat === "pret" || m.etat === "chargement")) {
        RUCHE.emettre("message", "Serveur local détecté : recharge la page (F5) pour l'utiliser");
        return;
      }
      modeServeur(e);
      RUCHE.emettre("message", "Serveur local détecté : les modèles tourneront sur la carte graphique");
      return;
    }
    if (!e) {
      if (etatServeur) { etatServeur = null; majStatut(); }
      return;
    }
    let change = !etatServeur;
    etatServeur = e;
    for (const tache of Object.keys(TACHES)) {
      for (const d of e.modeles[tache] || []) {
        let m = modeles[tache].find((x) => x.nom === d.nom);
        if (!m) { m = nouveau(tache, d.nom, tous().length, { description: d.description }); modeles[tache].push(m); change = true; }
        change = synchroniser(m, d) || change;
      }
    }
    if (change) emettre(); else majStatut();
  }

  function majStatut() {
    const prets = tous().filter((m) => m.etat === "pret");
    const charge = tous().find((m) => m.etat === "chargement");
    const n = prets.length, s = n > 1 ? "s" : "";
    if (charge) return statut(`Chargement de ${charge.nom}…` + (charge.progression != null ? ` ${Math.round(charge.progression * 100)} %` : ""));
    if (mode === "serveur") {
      if (!etatServeur) return statut("Serveur local arrêté : relance lancer_serveur.bat", "error");
      const ou = [...new Set(prets.map((m) => m.meta && m.meta.appareil).filter(Boolean))].join(", ");
      return statut(n ? `Serveur local · ${n} modèle${s} chargé${s}` + (ou ? ` · ${ou}` : "") : "Serveur local prêt · coche un modèle pour le charger", "ready");
    }
    if (mode === "navigateur") {
      const ou = backend === "webgl" ? "carte graphique (WebGL)" : backend ? "processeur, lent" : "";
      return statut(n ? `Navigateur · ${n} modèle${s} chargé${s}${ou ? " · " + ou : ""}` : "Navigateur (serveur non lancé) · coche un modèle pour le charger", n ? "ready" : undefined);
    }
  }

  // ---------------------------------------------------------------------------
  // Calcul par le serveur
  // ---------------------------------------------------------------------------
  function reseauServeur(m) {
    const S = RUCHE.serveur;
    return {
      async classer(pixels) {
        const [h, w, c] = m.meta.input;
        const r = await S.classer(m.nom, pixels, h, w, c);
        return { raw: r.raw, probs: r.probs };
      },
      debruiter: (pixels, h, w) => S.debruiter(m.nom, pixels, h, w),
      legender: (image, reglages) => S.legender(m.nom, image, reglages),
    };
  }

  // ---------------------------------------------------------------------------
  // Calcul dans le navigateur (secours) : chargement d'un export de convertir_modeles.py
  // ---------------------------------------------------------------------------
  /** base64 -> octets. fetch() d'une URL data: fait le décodage hors du fil principal de la page. */
  async function base64VersOctets(b64) {
    try {
      const r = await fetch("data:application/octet-stream;base64," + b64);
      return new Uint8Array(await r.arrayBuffer());
    } catch (_) { // navigateur qui refuse : décodage par morceaux, en rendant la main à la page entre deux morceaux
      const out = new Uint8Array(Math.floor((b64.length * 3) / 4) - (b64.endsWith("==") ? 2 : b64.endsWith("=") ? 1 : 0));
      const PAS = 1 << 20; // multiple de 4 caractères
      for (let i = 0, o = 0; i < b64.length; i += PAS) {
        const bin = atob(b64.slice(i, i + PAS));
        for (let j = 0; j < bin.length; j++) out[o++] = bin.charCodeAt(j);
        await prochaineImage();
      }
      return out;
    }
  }

  let fileNavigateur = Promise.resolve(); // un chargement à la fois (les fichiers générés écrivent dans enCours)

  function chargerNavigateur(m) {
    const tache = fileNavigateur.then(async () => {
      if (!backend) {
        statut("Démarrage de TensorFlow.js…");
        backend = await RUCHE.moteur.demarrer();
      }
      const dossier = `modeles/${m.tache}/${encodeURIComponent(m.nom)}/`;
      enCours = { meta: null, parts: [] };
      try {
        await chargerScript(dossier + "modele.js");
        const meta = enCours.meta;
        if (!meta) throw new Error("modele.js invalide");
        const morceaux = [];
        for (let i = 0; i < meta.parts.length; i++) {
          await chargerScript(dossier + meta.parts[i]);
          morceaux.push(base64VersOctets(enCours.parts[i]));
          enCours.parts[i] = null;
          m.progression = 0.8 * (i + 1) / meta.parts.length;
          emettre(m);
        }
        const blocs = await Promise.all(morceaux);
        const octets = new Uint8Array(blocs.reduce((s, b) => s + b.length, 0));
        let o = 0;
        for (const b of blocs) { octets.set(b, o); o += b.length; }
        const poids = await RUCHE.moteur.decoderPoidsAsync(octets, meta.poids_type);
        await prochaineImage();
        m.meta = meta;
        m.net = RUCHE.moteur.creerModele(meta, poids);
        m.progression = 0.9;
        emettre(m);
        if (backend === "webgl") await controler(m);
      } finally {
        enCours = null;
      }
    });
    fileNavigateur = tache.catch(() => {});
    return tache;
  }

  /** Vérifie un modèle sur la carte graphique (voir controler() dans moteur.js). En cas d'écart avec Keras, on
      réessaie avec des tuiles plus petites, puis on passe tout le calcul sur le processeur (plus lent mais exact). */
  async function controler(m) {
    statut(`Vérification de ${m.nom} sur la carte graphique…`);
    await prochaineImage();
    let c = await m.net.controler();
    if (!c.ok && m.meta.sortie === "image") {
      console.warn(`${m.nom} : écart ${c.ecart} avec Keras, nouvel essai avec des tuiles plus petites`);
      m.net.reduireTuiles();
      c = await m.net.controler();
    }
    if (!c.ok) {
      console.warn(`${m.nom} : résultats faux sur cette carte graphique (écart ${c.ecart}), passage sur le processeur`);
      await RUCHE.moteur.passerSurProcesseur();
      backend = "cpu";
      tous().forEach((x) => { if (x.net) x.controle = { ok: true, ecart: null, processeur: true }; });
      c = { ok: true, ecart: null, processeur: true };
    }
    m.controle = c;
  }

  // ---------------------------------------------------------------------------
  // Cocher / décocher
  // ---------------------------------------------------------------------------
  async function activer(m) {
    if (m.etat === "pret") return m;
    if (m.etat === "chargement") return m.attente;
    if (!m.disponible) throw new Error(mode === "navigateur" && m.tache === "L3_captioning" ? "BLIP a besoin du serveur local" : `${m.nom} : fichier du modèle introuvable`);
    m.etat = "chargement"; m.erreur = null; m.progression = mode === "navigateur" ? 0 : null; m.annuler = false;
    emettre(m);
    m.attente = (async () => {
      try {
        if (mode === "serveur") {
          const meta = await RUCHE.serveur.charger(m.tache, m.nom);
          m.meta = meta;
          m.net = reseauServeur(m);
          m.controle = { serveur: true, appareil: meta.appareil };
        } else {
          await chargerNavigateur(m);
        }
        m.etat = "pret";
        m.progression = null;
        if (m.annuler) await desactiver(m);
      } catch (e) {
        console.error(e);
        m.etat = "erreur"; m.erreur = e.message; m.net = null; m.progression = null;
        emettre(m);
        throw e;
      }
      emettre(m);
      return m;
    })();
    return m.attente;
  }

  async function desactiver(m) {
    if (m.etat === "chargement") { m.annuler = true; return; }
    if (m.etat !== "pret") { m.etat = "libre"; m.erreur = null; emettre(m); return; }
    m.etat = "libre";
    const net = m.net;
    m.net = null;
    emettre(m);
    try {
      if (mode === "serveur") await RUCHE.serveur.decharger(m.tache, m.nom);
      else if (net && net.liberer) net.liberer();
    } catch (e) {
      console.error(e);
    }
  }

  RUCHE.modeles = {
    TACHES,
    demarrer,
    mode: () => mode,
    serveurPret: () => mode === "serveur" && !!etatServeur,
    etatServeur: () => etatServeur,
    backend: () => backend,
    liste: (tache) => modeles[tache],
    prets: (tache) => modeles[tache].filter((m) => m.etat === "pret"),
    actifs: (tache) => modeles[tache].filter((m) => m.etat === "pret"),
    trouver: (tache, nom) => modeles[tache].find((m) => m.nom === nom) || null,
    /** Charge le modèle s'il ne l'est pas (renvoie une promesse résolue quand il est prêt). */
    activer,
    desactiver,
    basculer(m, on) {
      (on ? activer(m) : desactiver(m)).catch((e) => RUCHE.emettre("message", `${m.nom} : ${e.message}`));
    },
  };
})();
