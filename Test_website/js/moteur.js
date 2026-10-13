/* =============================================================================
   La Ruche : moteur de calcul
   -----------------------------------------------------------------------------
   Exécute dans le navigateur, avec TensorFlow.js, les modèles exportés par
   convertir_modeles.py (modeles/<tâche>/<nom>/modele.js + poids_*.js) :
     - un graphe de couches (Conv2D, Pool, Dense, UpSampling, Concatenate, Add...)
       décrit en JSON, chaque couche indique les tenseurs qu'elle reçoit ;
     - les poids, en base64 (float16 ou float32).
   Le calcul se fait sur la carte graphique (WebGL) si possible, sinon sur le processeur.

   Précision sur carte graphique. Sur certaines cartes (NVIDIA sous Windows notamment),
   WebGL calcule les positions dans un tenseur avec une arithmétique approchée au-delà
   de 2^24 valeurs : les pixels lus sont faux (rayures, teintes). Le moteur garde donc
   chaque tenseur sous LIMITE valeurs :
     - les grandes images sont débruitées par tuiles qui se chevauchent : chaque tuile
       a une marge au moins égale au champ récepteur du réseau, on ne garde que son
       centre, et le résultat est identique au calcul sur l'image entière ;
     - les convolutions n'utilisent pas « im2col » (matrice intermédiaire 9 fois plus
       grosse que l'image) ;
     - les très grosses couches denses (classifieur : 20 M de poids) sont découpées.
   Enfin chaque modèle est contrôlé au chargement contre une sortie de référence calculée
   par convertir_modeles.py : si la carte graphique se trompe, le site le détecte.
   ============================================================================= */
(function (racine) {
  "use strict";
  const RUCHE = (racine.RUCHE = racine.RUCHE || {});
  const tf = racine.tf;
  const LIMITE = 1 << 23;          // nombre max de valeurs par tenseur (2^23, marge sous 2^24)
  const LIMITE_DENSE = 1 << 22;    // taille max d'un morceau de matrice dense

  // ---------------------------------------------------------------------------
  // Démarrage : WebGL (carte graphique) si disponible, sinon processeur
  // ---------------------------------------------------------------------------
  async function demarrer() {
    if (!tf) throw new Error("TensorFlow.js introuvable (js/lib/tf.min.js)");
    let webgl = false;
    try { webgl = await tf.setBackend("webgl"); } catch (_) { /* WebGL absent */ }
    if (!webgl) await tf.setBackend("cpu");
    await tf.ready();
    if (webgl) {
      tf.env().set("WEBGL_CONV_IM2COL", false); // pas de matrice im2col géante (voir en-tête)
      // WebGL garde les textures libérées pour les réutiliser, sans limite par défaut : avec des images de
      // tailles différentes, la mémoire de la carte graphique gonflerait. Au-delà de 256 Mo, on les supprime.
      tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 256 * 1024 * 1024);
    }
    return tf.getBackend();
  }

  /** Passe tout le calcul sur le processeur (exact, mais plus lent). */
  async function passerSurProcesseur() {
    await tf.setBackend("cpu");
    await tf.ready();
  }

  // ---------------------------------------------------------------------------
  // Décodage des poids : base64 -> octets -> Float32Array
  // ---------------------------------------------------------------------------
  function base64EnOctets(b64) {
    if (typeof atob === "undefined") return new Uint8Array(Buffer.from(b64, "base64")); // Node (tests) : copie alignée
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  let tableFloat16 = null;
  function float16VersFloat32() {
    if (tableFloat16) return tableFloat16;
    tableFloat16 = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      if (e === 0) tableFloat16[h] = s * Math.pow(2, -14) * (f / 1024);
      else if (e === 31) tableFloat16[h] = f ? NaN : s * Infinity;
      else tableFloat16[h] = s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return tableFloat16;
  }

  function decoderPoids(b64, type) {
    const octets = base64EnOctets(b64);
    if (type === "float32") return new Float32Array(octets.buffer, octets.byteOffset, octets.byteLength >> 2);
    const u16 = new Uint16Array(octets.buffer, octets.byteOffset, octets.byteLength >> 1);
    const table = float16VersFloat32(), out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = table[u16[i]];
    return out;
  }

  /** Image de contrôle, identique à motif_controle() de convertir_modeles.py :
      moitié dégradé horizontal, moitié valeurs pseudo-aléatoires, ramenée à la plage du modèle. */
  function motifControle(h, w, c, [lo, hi]) {
    const x = new Float32Array(h * w * c);
    for (let i = 0; i < x.length; i++) {
      const col = Math.floor(i / c) % w;
      const hasard = (Math.imul(i, 2654435761 | 0) >>> 0) / 4294967296;
      x[i] = lo + (hi - lo) * (0.5 * hasard + 0.5 * (col / (w - 1)));
    }
    return x;
  }

  // ---------------------------------------------------------------------------
  // Couches
  // ---------------------------------------------------------------------------
  function activer(x, L) {
    let y;
    switch (L.activation) {
      case undefined: case null: case "linear": y = x; break;
      case "relu": y = tf.relu(x); break;
      case "relu6": y = tf.relu6(x); break;
      case "leaky_relu": y = tf.leakyRelu(x, L.alpha == null ? 0.2 : L.alpha); break; // 0,2 : défaut de Keras 3
      case "elu": y = tf.elu(x); break;
      case "selu": y = tf.selu(x); break;
      case "sigmoid": y = tf.sigmoid(x); break;
      case "tanh": y = tf.tanh(x); break;
      case "silu": y = tf.mul(x, tf.sigmoid(x)); break;
      case "softplus": y = tf.softplus(x); break;
      case "gelu": y = tf.mul(tf.mul(x, 0.5), tf.add(1, tf.erf(tf.div(x, Math.SQRT2)))); break;
      case "softmax": y = tf.softmax(x); break;
      default: throw new Error("Activation non gérée : " + L.activation);
    }
    return L.max_value != null ? tf.minimum(y, L.max_value) : y;
  }

  // UpSampling2D (nearest) de Keras : chaque pixel est recopié sh x sw fois (calcul exact, sans interpolation)
  function surechantillonner(x, [sh, sw]) {
    const [n, h, w, c] = x.shape;
    let y = x;
    if (sh > 1) y = tf.reshape(tf.tile(tf.reshape(y, [n * h, 1, w, c]), [1, sh, 1, 1]), [n, h * sh, w, c]);
    if (sw > 1) y = tf.reshape(tf.tile(tf.reshape(y, [n * h * sh, w, 1, c]), [1, 1, sw, 1]), [n, h * sh, w * sw, c]);
    return y;
  }

  function calculerCouche(L, entrees, P) {
    const x = entrees[0];
    switch (L.type) {
      case "Affine": return tf.add(tf.mul(x, P.scale), P.offset);
      case "Conv2D": {
        let y = tf.conv2d(x, P.w[0], L.strides, L.padding);
        if (P.w[1]) y = tf.add(y, P.w[1]);
        return activer(y, L);
      }
      case "Pool":
        return (L.mode === "max" ? tf.maxPool : tf.avgPool)(x, L.pool, L.strides, L.padding);
      case "GlobalPool": return L.mode === "max" ? tf.max(x, [1, 2]) : tf.mean(x, [1, 2]);
      case "Flatten": return tf.reshape(x, [x.shape[0], -1]);
      case "Dense": { // comme Keras : la couche dense agit sur le dernier axe
        const c = x.shape[x.rank - 1], x2 = tf.reshape(x, [-1, c]);
        // matrice découpée en morceaux de lignes : y = somme des x[:, morceau] · W[morceau, :]
        let y = null;
        for (const { debut, poids } of P.morceaux) {
          const t = tf.matMul(tf.slice(x2, [0, debut], [-1, poids.shape[0]]), poids);
          y = y ? tf.add(y, t) : t;
        }
        if (P.w[1]) y = tf.add(y, P.w[1]);
        y = activer(y, L);
        return tf.reshape(y, [...x.shape.slice(0, -1), L.units]);
      }
      case "UpSampling": return surechantillonner(x, L.size);
      case "Concatenate": return tf.concat(entrees, -1);
      case "Add": return tf.addN(entrees);
      case "Activation": return activer(x, L);
      default: throw new Error("Couche non gérée : " + L.type);
    }
  }

  /** Pour un réseau image -> image : valeurs par pixel d'entrée dans le plus gros tenseur, et rayon du
      champ récepteur (en pixels d'entrée). Renvoie null si le graphe contient des couches non spatiales. */
  function analyserGraphe(couches, recus, ENTREE, SORTIE, canaux) {
    const info = new Map([[ENTREE, { c: canaux, s: 1, r: 0 }]]);
    let parPixel = canaux;
    for (let i = 0; i < couches.length; i++) {
      const L = couches[i], ins = recus[i].map((n) => info.get(n)), a = ins[0];
      let o;
      switch (L.type) {
        case "Conv2D": o = { c: L.filters, s: a.s * Math.max(...L.strides), r: a.r + ((Math.max(...L.kernel) - 1) / 2) * a.s }; break;
        case "Pool": o = { c: a.c, s: a.s * Math.max(...L.strides), r: a.r + ((Math.max(...L.pool) - 1) / 2) * a.s }; break;
        case "UpSampling": o = { c: a.c, s: a.s / Math.max(...L.size), r: a.r }; break;
        case "Concatenate": o = { c: ins.reduce((t, x) => t + x.c, 0), s: a.s, r: Math.max(...ins.map((x) => x.r)) }; break;
        case "Add": o = { c: a.c, s: a.s, r: Math.max(...ins.map((x) => x.r)) }; break;
        case "Affine": case "Activation": o = { ...a }; break;
        default: return null;
      }
      info.set(L.name, o);
      parPixel = Math.max(parPixel, o.c / (o.s * o.s));
    }
    return { parPixel, rf: info.get(SORTIE).r };
  }

  // ---------------------------------------------------------------------------
  // Modèle
  // ---------------------------------------------------------------------------
  function creerModele(meta, b64) {
    const W = decoderPoids(b64, meta.poids_type);
    const ENTREE = meta.entree || "entree";
    const couches = meta.layers;

    // Poids de chaque couche, envoyés une fois pour toutes à TensorFlow.js
    const P = couches.map((L) => {
      const p = { w: (L.weights || []).map((r) => tf.tensor(W.subarray(r.offset, r.offset + r.size), r.shape, "float32")) };
      if (L.type === "Affine") { p.scale = tf.tensor1d(L.scale); p.offset = tf.tensor1d(L.offset); }
      if (L.type === "Dense") { // matrice découpée en morceaux de moins de LIMITE_DENSE valeurs
        const [n, u] = p.w[0].shape, pas = Math.max(1, Math.floor(LIMITE_DENSE / u));
        if (n <= pas) p.morceaux = [{ debut: 0, poids: p.w[0] }];
        else {
          p.morceaux = [];
          for (let debut = 0; debut < n; debut += pas) {
            p.morceaux.push({ debut, poids: tf.slice(p.w[0], [debut, 0], [Math.min(pas, n - debut), u]) });
          }
          p.w[0].dispose();
        }
        p.w[0] = null;
      }
      return p;
    });

    // Anciennes exportations : pas de champ « inputs », chaque couche reçoit la sortie de la précédente
    const recus = couches.map((L, i) => L.inputs || [i ? couches[i - 1].name : ENTREE]);
    const SORTIE = meta.sortie_noeud || couches[couches.length - 1].name;

    // Dernière couche qui utilise chaque tenseur : on le libère juste après (mémoire de la carte graphique)
    const dernierUsage = new Map();
    recus.forEach((noms, i) => noms.forEach((n) => dernierUsage.set(n, i)));

    // Débruiteurs : taille des tuiles pour que chaque tenseur reste sous la limite
    const m = meta.multiple || 1;
    const analyse = meta.sortie === "image" ? analyserGraphe(couches, recus, ENTREE, SORTIE, 3) : null;
    let limite = LIMITE, tuile = null;
    function calculerTuile() {
      if (!analyse) return;
      const marge = Math.ceil(analyse.rf / m) * m;
      const cote = Math.max(Math.floor(Math.sqrt(limite / analyse.parPixel) / m) * m, 2 * marge + m);
      tuile = { cote, marge, coeur: cote - 2 * marge };
    }
    calculerTuile();

    /** Calcule le graphe sur x (tenseur 4D [1, H, W, C]). Renvoie le tenseur de sortie. */
    function executer(x) {
      const valeurs = new Map([[ENTREE, x]]);
      couches.forEach((L, i) => {
        const entrees = recus[i].map((n) => {
          if (!valeurs.has(n)) throw new Error(`Tenseur « ${n} » inconnu (couche ${L.name})`);
          return valeurs.get(n);
        });
        valeurs.set(L.name, tf.tidy(() => calculerCouche(L, entrees, P[i])));
        for (const n of new Set(recus[i])) {
          if (dernierUsage.get(n) !== i || n === ENTREE || n === SORTIE) continue;
          const t = valeurs.get(n);
          valeurs.delete(n);
          if (![...valeurs.values()].includes(t)) t.dispose();
        }
      });
      return valeurs.get(SORTIE);
    }

    const modele = {
      meta,
      tuile: () => tuile,

      /** Classification. pixels : Float32Array H x W x C à la taille d'entrée, dans meta.plage (0..255).
          Renvoie { raw, probs } : sorties brutes du réseau et une probabilité par classe. */
      async classer(pixels) {
        const [H, Wd, C] = meta.input;
        const y = tf.tidy(() => executer(tf.tensor4d(pixels, [1, H, Wd, C])));
        const raw = Array.from(await y.data());
        y.dispose();
        let probs;
        switch (meta.sortie) {
          case "probas": probs = raw.slice(); break;
          case "sigmoid": probs = [1 - raw[0], raw[0]]; break;
          case "logit": { const p = 1 / (1 + Math.exp(-raw[0])); probs = [1 - p, p]; break; }
          default: probs = softmax(raw); // logits
        }
        return { raw, probs };
      },

      /** Débruitage. pixels : Float32Array H x W x 3 dans [0, 1]. Renvoie un Float32Array de même taille.
          Les côtés sont complétés par symétrie jusqu'à un multiple de meta.multiple (8), puis l'image est
          traitée par tuiles (voir l'en-tête du fichier) et rognée à sa taille d'origine. */
      async debruiter(pixels, H, Wd) {
        const ph = (m - (H % m)) % m, pw = (m - (Wd % m)) % m;
        if (H <= ph || Wd <= pw) throw new Error("image trop petite");
        const Hp = H + ph, Wp = Wd + pw, out = new Float32Array(H * Wd * 3);
        const x = tf.tidy(() => {
          const t = tf.tensor4d(pixels, [1, H, Wd, 3]);
          return ph || pw ? tf.mirrorPad(t, [[0, 0], [0, ph], [0, pw], [0, 0]], "reflect") : t;
        });
        const { cote, marge, coeur } = tuile;
        const pasY = Hp <= cote ? Hp : coeur, pasX = Wp <= cote ? Wp : coeur;
        try {
          for (let y0 = 0; y0 < Hp; y0 += pasY) {
            for (let x0 = 0; x0 < Wp; x0 += pasX) {
              // centre gardé [y0, y1) x [x0, x1) ; fenêtre calculée = centre + marge (dans l'image)
              const y1 = Math.min(y0 + pasY, Hp), x1 = Math.min(x0 + pasX, Wp);
              const fy0 = Math.max(0, y0 - marge), fy1 = Math.min(Hp, y1 + marge);
              const fx0 = Math.max(0, x0 - marge), fx1 = Math.min(Wp, x1 + marge);
              const r = tf.tidy(() => {
                const fenetre = Hp === fy1 - fy0 && Wp === fx1 - fx0 ? x : tf.slice(x, [0, fy0, fx0, 0], [1, fy1 - fy0, fx1 - fx0, 3]);
                return tf.slice(executer(fenetre), [0, y0 - fy0, x0 - fx0, 0], [1, y1 - y0, x1 - x0, 3]);
              });
              const d = await r.data();
              r.dispose();
              const lignes = Math.min(y1, H) - y0, colonnes = Math.min(x1, Wd) - x0, largeur = x1 - x0;
              for (let yy = 0; yy < lignes; yy++) {
                const src = yy * largeur * 3, dst = ((y0 + yy) * Wd + x0) * 3;
                for (let k = 0; k < colonnes * 3; k++) out[dst + k] = Math.min(1, Math.max(0, d[src + k]));
              }
            }
          }
        } finally {
          x.dispose();
        }
        return out;
      },

      /** Contrôle : calcule la sortie d'une image de référence et la compare à celle de Keras, stockée
          dans modele.js par convertir_modeles.py. Sert aussi de premier calcul (WebGL compile ses programmes).
          Renvoie { ok, ecart } (ecart : plus grande différence observée). */
      async controler() {
        const c = meta.controle;
        if (!c) return { ok: true, ecart: null };
        const [h, w, ch] = c.forme, x = motifControle(h, w, ch, meta.plage);
        let ecart = 0;
        if (meta.sortie === "image") {
          const y = await this.debruiter(x, h, w), b = c.bloc, attendu = decoderPoids(c.attendu, "float32");
          const bh = h / b, bw = w / b;
          for (let i = 0; i < bh; i++) for (let j = 0; j < bw; j++) for (let k = 0; k < 3; k++) {
            let s = 0;
            for (let u = 0; u < b; u++) for (let v = 0; v < b; v++) s += y[((i * b + u) * w + j * b + v) * 3 + k];
            ecart = Math.max(ecart, Math.abs(s / (b * b) - attendu[(i * bw + j) * 3 + k]));
          }
          return { ok: ecart <= 2e-3, ecart };
        }
        const { raw } = await this.classer(x);
        raw.forEach((v, i) => { ecart = Math.max(ecart, Math.abs(v - c.attendu[i]) / Math.max(1, Math.abs(c.attendu[i]))); });
        return { ok: ecart <= 2e-2, ecart };
      },

      /** Tuiles 2 fois plus petites (si le contrôle échoue sur une carte graphique). */
      reduireTuiles() { limite /= 4; calculerTuile(); },

      liberer() {
        P.forEach((p) => tf.dispose([...p.w, p.scale, p.offset, ...(p.morceaux || []).map((q) => q.poids)].filter(Boolean)));
      },
    };
    return modele;
  }

  function softmax(a) {
    const m = Math.max(...a), e = a.map((v) => Math.exp(v - m)), s = e.reduce((x, y) => x + y, 0);
    return e.map((v) => v / s);
  }

  /* Redimensionnement bilinéaire identique à tf.image.resize (half_pixel_centers, sans antialias),
     ce qu'utilise image_dataset_from_directory à l'entraînement du livrable 1.
     rgba : pixels RGBA (canvas) ; channels = 3 (RGB) ou 1 (niveaux de gris). Renvoie des valeurs 0..255. */
  function redimensionnerBilineaire(rgba, srcW, srcH, dstW, dstH, channels = 3) {
    let src = rgba, stride = 4;
    if (channels === 1) { // gris comme libjpeg / TensorFlow : 0,299 R + 0,587 G + 0,114 B
      src = new Float32Array(srcW * srcH);
      for (let i = 0; i < src.length; i++) src[i] = Math.round(0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]);
      stride = 1;
    }
    const out = new Float32Array(dstW * dstH * channels);
    const sy = srcH / dstH, sx = srcW / dstW;
    for (let y = 0; y < dstH; y++) {
      let fy = (y + 0.5) * sy - 0.5; if (fy < 0) fy = 0;
      const y0 = Math.min(Math.floor(fy), srcH - 1), y1 = Math.min(y0 + 1, srcH - 1), dy = fy - y0;
      for (let x = 0; x < dstW; x++) {
        let fx = (x + 0.5) * sx - 0.5; if (fx < 0) fx = 0;
        const x0 = Math.min(Math.floor(fx), srcW - 1), x1 = Math.min(x0 + 1, srcW - 1), dx = fx - x0;
        const a = (y0 * srcW + x0) * stride, b = (y0 * srcW + x1) * stride, c = (y1 * srcW + x0) * stride, d = (y1 * srcW + x1) * stride;
        const o = (y * dstW + x) * channels;
        for (let ch = 0; ch < channels; ch++) {
          const haut = src[a + ch] + (src[b + ch] - src[a + ch]) * dx;
          const bas = src[c + ch] + (src[d + ch] - src[c + ch]) * dx;
          out[o + ch] = haut + (bas - haut) * dy;
        }
      }
    }
    return out;
  }

  RUCHE.moteur = {
    demarrer, passerSurProcesseur, creerModele, redimensionnerBilineaire, motifControle,
    backend: () => (tf ? tf.getBackend() : null),
  };
})(typeof window !== "undefined" ? window : globalThis);
