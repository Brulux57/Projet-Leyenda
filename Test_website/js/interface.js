/* =============================================================================
   La Ruche : composants d'interface partagés par les onglets
   -----------------------------------------------------------------------------
   zoneDepot   : glisser-déposer (fichiers et dossiers), sélecteur, coller, webcam
   comparateur : image avant / après avec un curseur à faire glisser
   panneauModeles : liste des modèles d'une tâche (activation, architecture)
   ============================================================================= */
(function () {
  "use strict";
  const RUCHE = window.RUCHE;
  const { $, esc, estImage, depuisInput, lireDepot, nombre } = RUCHE.outils;

  /* ---------------------------------------------------------------------------
     Zone de dépôt. Les événements de glisser-déposer et de collage sont écoutés sur
     toute la page, mais seule la zone de l'onglet affiché les reçoit.
     options : zone, inputs (sélecteurs de fichiers / dossier), boutons { bouton: input },
               camera { bouton, cadre, video, capturer, fermer }, recevoir(items), pret()
     ------------------------------------------------------------------------- */
  function zoneDepot({ zone, inputs = [], boutons = [], camera = null, recevoir, pret = () => true }) {
    const vue = zone.closest(".vue");
    const actif = () => (!vue || !vue.hidden) && pret();
    const transmettre = (items) => {
      const images = items.filter((it) => estImage(it.fichier));
      if (images.length) recevoir(images);
    };

    zone.addEventListener("click", (e) => {
      if (!e.target.closest("button, a, input, label, video") && inputs[0]) inputs[0].click();
    });
    boutons.forEach(([bouton, input]) => bouton.addEventListener("click", (e) => { e.stopPropagation(); input.click(); }));
    inputs.forEach((input) => input.addEventListener("change", () => { transmettre(depuisInput(input.files)); input.value = ""; }));

    ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => {
      e.preventDefault(); // sinon le navigateur ouvre l'image à la place de la page
      if (actif()) zone.classList.add("drag");
    }));
    ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => {
      e.preventDefault();
      if (ev === "dragleave" && e.relatedTarget) return;
      zone.classList.remove("drag");
      if (ev === "drop" && actif() && e.dataTransfer) lireDepot(e.dataTransfer).then(transmettre);
    }));
    document.addEventListener("paste", (e) => {
      if (!actif() || !e.clipboardData) return;
      const fichiers = Array.from(e.clipboardData.items).filter((i) => i.kind === "file").map((i) => i.getAsFile()).filter(Boolean);
      if (fichiers.length) { e.preventDefault(); transmettre(depuisInput(fichiers)); }
    });

    if (camera) {
      let flux = null;
      const fermer = () => { if (flux) flux.getTracks().forEach((t) => t.stop()); flux = null; camera.cadre.hidden = true; };
      camera.bouton.addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          flux = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 } }, audio: false });
          camera.video.srcObject = flux;
          camera.cadre.hidden = false;
          camera.cadre.scrollIntoView({ behavior: "smooth", block: "center" });
        } catch (err) {
          RUCHE.emettre("message", "Webcam indisponible (" + (err.name || "erreur") + ")");
        }
      });
      camera.fermer.addEventListener("click", fermer);
      camera.capturer.addEventListener("click", () => {
        const v = camera.video;
        if (!v.videoWidth) return;
        const cv = document.createElement("canvas");
        cv.width = v.videoWidth; cv.height = v.videoHeight;
        cv.getContext("2d").drawImage(v, 0, 0);
        cv.toBlob((blob) => {
          const nom = "photo-" + new Date().toLocaleTimeString("fr-FR").replace(/:/g, "h") + ".png";
          transmettre([{ fichier: new File([blob], nom, { type: "image/png" }), chemin: nom }]);
        }, "image/png");
      });
      if (vue) RUCHE.on("onglet", () => { if (vue.hidden) fermer(); });
    }

    return { activer(on) { zone.classList.toggle("disabled", !on); } };
  }

  /* ---------------------------------------------------------------------------
     Comparateur avant / après : deux images superposées, l'image « après » est
     découpée à droite du curseur. Le curseur est un <input type="range"> transparent
     qui couvre l'image (souris, doigt et clavier).
     ------------------------------------------------------------------------- */
  function comparateur() {
    const el = document.createElement("div");
    el.className = "compare";
    el.innerHTML = `
      <img class="cmp-avant" alt="" draggable="false">
      <img class="cmp-apres" alt="" draggable="false">
      <div class="cmp-ligne" aria-hidden="true"><span class="cmp-poignee"></span></div>
      <span class="cmp-tag cmp-g"></span><span class="cmp-tag cmp-d"></span>
      <input type="range" class="cmp-range" min="0" max="100" step="0.5" value="50">`;
    const range = $(".cmp-range", el), avant = $(".cmp-avant", el), apres = $(".cmp-apres", el);
    const placer = (v) => el.style.setProperty("--pos", v + "%");
    range.addEventListener("input", () => placer(range.value));
    placer(50);
    return {
      el,
      /** avant, apres : URL d'images ; ratio : largeur / hauteur ; etiquettes : [gauche, droite] */
      definir({ avant: ua, apres: ub, ratio, etiquettes = ["Avant", "Après"] }) {
        if (ua != null) avant.src = ua;
        if (ub != null) apres.src = ub;
        apres.hidden = !apres.getAttribute("src");
        if (ratio) el.style.setProperty("--ratio", Math.min(1.8, Math.max(0.75, ratio)));
        $(".cmp-g", el).textContent = etiquettes[0];
        $(".cmp-d", el).textContent = etiquettes[1];
        range.setAttribute("aria-label", `Curseur ${etiquettes[0]} / ${etiquettes[1]}`);
      },
    };
  }

  /* ---------------------------------------------------------------------------
     Panneau « modèles » d'une tâche : interrupteur actif / en pause, fichier source,
     taille d'entrée, paramètres, classes, architecture. extra(m) ajoute du HTML.
     ------------------------------------------------------------------------- */
  function panneauModeles(conteneur, tache, { extra = () => "", vide = "" } = {}) {
    const liste = RUCHE.modeles.liste(tache);
    // garde ouvert le détail « Architecture » des modèles qui l'étaient avant de redessiner
    const ouverts = new Set(Array.from(conteneur.querySelectorAll(".model")).filter((d) => d.querySelector("details[open]")).map((d) => d.dataset.modele));
    if (!liste.length) {
      conteneur.innerHTML = `<p class="hint">${vide}</p>`;
      return;
    }
    conteneur.innerHTML = "";
    for (const m of liste) {
      const div = document.createElement("div");
      div.className = "model" + (m.actif ? "" : " off") + (m.erreur ? " broken" : "");
      div.style.setProperty("--mc", m.couleur);
      let corps;
      if (m.erreur) corps = `<p class="model-err">Erreur : ${esc(m.erreur)}</p>`;
      else if (!m.meta) corps = `<p class="model-sub">Chargement…</p>`;
      else {
        const mt = m.meta, [h, w, c] = mt.input;
        const date = mt.date ? mt.date.split("@")[0].split("-").reverse().join("/") : null;
        const params = mt.params >= 1e6 ? nombre(mt.params / 1e6) + " M" : mt.params.toLocaleString("fr-FR");
        const taille = w == null ? `taille libre (×${mt.multiple || 1})` : `${w}×${h}`;
        corps = `
          ${mt.description ? `<p class="model-desc">${esc(mt.description)}</p>` : ""}
          <p class="model-sub">${esc(mt.source)}${date ? " · entraîné le " + date : ""}</p>
          <div class="model-meta"><span>${taille} ${c === 1 ? "gris" : "RGB"}</span><span>${params} param.</span>${extra(m)}</div>
          ${ligneControle(m)}
          ${mt.classes ? `<div class="model-classes">${mt.classes.map((n, i) => `<span class="cls"><b>${i}</b>${esc(n)}</span>`).join("")}</div>` : ""}
          <div class="model-score" data-score></div>
          <details class="arch"><summary>Architecture (${mt.layers.length} couches)</summary><ol class="stack">${mt.layers.map(ligneCouche).join("")}</ol></details>`;
      }
      div.innerHTML = `
        <label class="model-head">
          <span class="mtag">${esc(m.nom)}</span>
          <span class="model-state">${m.erreur ? "en erreur" : !m.net ? "chargement…" : m.actif ? "actif" : "en pause"}</span>
          <input type="checkbox" class="switch" ${m.actif ? "checked" : ""} ${m.net ? "" : "disabled"} aria-label="Activer ${esc(m.nom)}">
        </label>${corps}`;
      $(".switch", div).addEventListener("change", (e) => RUCHE.modeles.basculer(m, e.target.checked));
      div.dataset.modele = m.nom;
      if (ouverts.has(m.nom) && div.querySelector("details")) div.querySelector("details").open = true;
      conteneur.appendChild(div);
    }
  }

  const EXPOSANTS = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

  /** Résultat du contrôle au chargement : le navigateur calcule-t-il comme Keras ? */
  function ligneControle(m) {
    const c = m.controle;
    if (!c) return "";
    if (c.processeur) return `<p class="model-ctrl warn">Calcul sur le processeur : la carte graphique donnait un résultat différent de Keras.</p>`;
    if (c.ecart == null) return "";
    const [mant, puiss] = c.ecart.toExponential(0).split("e");
    const e = `${mant}·10${[...String(Number(puiss))].map((ch) => EXPOSANTS[ch]).join("")}`;
    return c.ok ? `<p class="model-ctrl">Vérifié sur ce navigateur : même résultat que Keras (écart ${esc(e)}).</p>`
      : `<p class="model-ctrl warn">Écart avec Keras sur ce navigateur (${esc(e)}).</p>`;
  }

  function ligneCouche(L) {
    const act = L.activation && L.activation !== "linear" ? " · " + L.activation + (L.max_value != null ? " ≤ " + L.max_value : "") : "";
    let d = "";
    if (L.type === "Conv2D") d = `${L.filters} × ${L.kernel.join("×")}${act}`;
    else if (L.type === "Dense") d = `${L.units} unités${act}`;
    else if (L.type === "Pool") d = `${L.mode} ${L.pool.join("×")}`;
    else if (L.type === "GlobalPool") d = L.mode;
    else if (L.type === "UpSampling") d = `×${L.size.join("×")}`;
    else if (L.type === "Concatenate" || L.type === "Add") d = (L.inputs || []).join(" + ");
    else if (L.type === "Activation") d = act.slice(3);
    else if (L.type === "Affine") d = L.name;
    return `<li><span>${esc(L.type)}</span><span>${esc(d)}</span></li>`;
  }

  RUCHE.interface = { zoneDepot, comparateur, panneauModeles };
})();
