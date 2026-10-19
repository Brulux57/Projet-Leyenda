/* =============================================================================
   La Ruche : liaison avec le serveur local (serveur.py)
   -----------------------------------------------------------------------------
   Quand serveur.py tourne, les modèles sont calculés par leurs vraies
   bibliothèques (TensorFlow pour L1 et L2, PyTorch pour BLIP), sur la carte
   graphique : le navigateur n'envoie que les pixels et affiche les résultats.
   La page ouverte par le serveur (http://localhost:8765/) l'appelle à la même
   adresse ; ouverte directement (file://), elle essaie http://127.0.0.1:8765.
   ============================================================================= */
(function () {
  "use strict";
  const RUCHE = window.RUCHE;
  const PORT = 8765;
  const base = location.protocol.startsWith("http") ? "" : `http://127.0.0.1:${PORT}`;

  async function appel(chemin, { methode = "GET", corps, type, delai = 0 } = {}) {
    const ctrl = delai ? new AbortController() : null;
    const minuterie = ctrl ? setTimeout(() => ctrl.abort(), delai) : null;
    let r;
    try {
      r = await fetch(base + chemin, {
        method: methode,
        body: corps,
        headers: type ? { "Content-Type": type } : undefined,
        signal: ctrl ? ctrl.signal : undefined,
        cache: "no-store",
      });
    } catch (e) {
      const err = new Error(e.name === "AbortError" ? "le serveur ne répond pas" : "serveur local injoignable");
      err.injoignable = true;
      throw err;
    } finally {
      clearTimeout(minuterie);
    }
    if (!r.ok) {
      let message = `erreur ${r.status}`;
      try { message = (await r.json()).erreur || message; } catch (_) { /* réponse non JSON */ }
      throw new Error(message);
    }
    return r;
  }

  const json = async (chemin, options) => (await appel(chemin, options)).json();
  const post = (chemin, objet) => json(chemin, { methode: "POST", corps: JSON.stringify(objet), type: "application/json" });
  const requete = (params) => new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();

  RUCHE.serveur = {
    PORT,
    adresse: base || location.origin,
    /** État du serveur et des modèles du catalogue, ou null s'il ne répond pas. */
    async etat(delai = 1500) {
      try { return await json("/api/etat", { delai }); } catch (_) { return null; }
    },
    charger: (tache, nom) => post("/api/charger", { tache, nom }).then((r) => r.meta),
    decharger: (tache, nom) => post("/api/decharger", { tache, nom }),

    /** pixels : Float32Array H x W x C (0..255). Renvoie { raw, probs, ms }. */
    classer: (nom, pixels, h, w, c) => json(`/api/classer?${requete({ nom, h, w, c })}`, { methode: "POST", corps: pixels, type: "application/octet-stream" }),

    /** pixels : Float32Array H x W x 3 (0..1). Renvoie un Float32Array de même taille. */
    async debruiter(nom, pixels, h, w) {
      const r = await appel(`/api/debruiter?${requete({ nom, h, w })}`, { methode: "POST", corps: pixels, type: "application/octet-stream" });
      return new Float32Array(await r.arrayBuffer());
    },

    /** image : Blob (JPEG, PNG...) ; reglages : réglages de génération de BLIP. Renvoie { legende, suite, amorce, ms, ... }. */
    legender: (nom, image, reglages) => json(`/api/legender?${requete({ nom, ...reglages })}`, { methode: "POST", corps: image, type: image.type || "application/octet-stream" }),
  };
})();
