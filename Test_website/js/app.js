/* =============================================================================
   La Ruche : démarrage du site
   -----------------------------------------------------------------------------
   Onglets de l'en-tête (un par étape du projet, l'adresse #classification,
   #debruitage, #legendes ou #pipeline garde l'onglet ouvert au rechargement),
   pastille d'état, puis recherche du serveur local et liste des modèles (aucun
   n'est chargé tant qu'on ne le coche pas).
   ============================================================================= */
(function () {
  "use strict";
  const RUCHE = window.RUCHE;
  const { $, $$ } = RUCHE.outils;

  // ---------------------------------------------------------------------------
  // Onglets
  // ---------------------------------------------------------------------------
  const onglets = $$(".tab[data-onglet]");
  const noms = onglets.map((t) => t.dataset.onglet);

  function afficher(nom, focus = false) {
    if (!noms.includes(nom)) nom = noms[0];
    for (const t of onglets) {
      const on = t.dataset.onglet === nom;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $("#" + t.getAttribute("aria-controls")).hidden = !on;
      if (on && focus) t.focus();
    }
    if (location.hash !== "#" + nom) history.replaceState(null, "", "#" + nom);
    RUCHE.emettre("onglet", nom);
  }

  onglets.forEach((t, i) => {
    t.addEventListener("click", () => afficher(t.dataset.onglet));
    t.addEventListener("keydown", (e) => { // flèches gauche / droite entre les onglets (motif ARIA)
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (e.key === "Home" || e.key === "End") { e.preventDefault(); afficher(noms[e.key === "Home" ? 0 : noms.length - 1], true); }
      if (!d) return;
      e.preventDefault();
      afficher(noms[(i + d + noms.length) % noms.length], true);
    });
  });
  window.addEventListener("hashchange", () => afficher(location.hash.slice(1)));
  afficher(location.hash.slice(1));

  // ---------------------------------------------------------------------------
  // Pastille d'état
  // ---------------------------------------------------------------------------
  const pastille = $("#status"), texte = $("#status-text");
  let etatFixe = { texte: "", type: undefined }, minuterie = null;

  function statut(t, type) {
    etatFixe = { texte: t, type };
    if (!minuterie) montrer(t, type);
  }
  function montrer(t, type) {
    texte.textContent = t;
    pastille.classList.toggle("ready", type === "ready");
    pastille.classList.toggle("error", type === "error");
  }
  // message passager (ex. webcam refusée), puis retour à l'état normal
  RUCHE.on("message", (t) => {
    clearTimeout(minuterie);
    montrer(t, "error");
    minuterie = setTimeout(() => { minuterie = null; montrer(etatFixe.texte, etatFixe.type); }, 5000);
  });

  RUCHE.modeles.demarrer(statut);
})();
