/* =============================================================================
   La Ruche : outils communs aux onglets
   -----------------------------------------------------------------------------
   DOM et formats · événements · images (lecture, taille de traitement, PNG)
   · bruit simulé · mesures (PSNR, SSIM, bruit estimé) · fichiers (dépôt de
   dossiers, téléchargement, ZIP, CSV)
   ============================================================================= */
(function (racine) {
  "use strict";
  const RUCHE = (racine.RUCHE = racine.RUCHE || {});

  // ---------------------------------------------------------------------------
  // DOM, formats, stockage
  // ---------------------------------------------------------------------------
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const prochaineImage = () => new Promise((r) => setTimeout(r, 0));

  const nombre = (v, dec = 1) => v.toLocaleString("fr-FR", { minimumFractionDigits: dec, maximumFractionDigits: dec });
  function fmtPct(v) {
    const pc = v * 100;
    if (pc > 99.9 && pc < 100) return "> 99,9\u00a0%"; // espaces insécables : « 70 % » ne se coupe pas
    if (pc < 0.1 && pc > 0) return "< 0,1\u00a0%";
    return pc.toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + "\u00a0%";
  }
  function fmtDb(v, signe = false) {
    if (Number.isNaN(v) || v == null) return "–";
    if (!Number.isFinite(v)) return "∞";
    return (signe && v >= 0 ? "+" : "") + nombre(v) + "\u00a0dB";
  }
  const fmtTaille = (w, h) => `${w}×${h}`;

  // Le stockage local peut être indisponible (navigation privée, fichier ouvert en file://...)
  const stockage = {
    get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* rien */ } },
  };

  // Les pages ouvertes en file:// ne peuvent pas lire de fichiers avec fetch() : les modèles sont des .js
  // chargés par une balise <script>.
  const chargerScript = (src) => new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => { s.remove(); res(); };
    s.onerror = () => { s.remove(); rej(new Error(src + " introuvable")); };
    document.body.appendChild(s);
  });

  // Petit système d'événements entre modules (ex. « modeles » quand un modèle finit de charger)
  const abonnes = {};
  RUCHE.on = (nom, f) => { (abonnes[nom] = abonnes[nom] || []).push(f); };
  RUCHE.emettre = (nom, ...args) => (abonnes[nom] || []).forEach((f) => f(...args));

  // ---------------------------------------------------------------------------
  // Images
  // ---------------------------------------------------------------------------
  const EXTENSIONS = /\.(jpe?g|png|gif|bmp|webp|avif)$/i;
  const estImage = (f) => (f.type ? f.type.startsWith("image/") : EXTENSIONS.test(f.name));

  function lireImage(blob) {
    return new Promise((res, rej) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => res({ img, url });
      img.onerror = () => { URL.revokeObjectURL(url); rej(new Error("image illisible")); };
      img.src = url;
    });
  }

  function contexte(w, h) {
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    return { cv, ctx: cv.getContext("2d", { willReadFrequently: true }) };
  }

  /** Pixels RGBA en pleine résolution (pour le classifieur, qui redimensionne comme TensorFlow). */
  function pixelsPleineTaille(img) {
    const MAX = 16000; // garde-fou : taille max d'un canvas
    const k = Math.min(1, MAX / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.max(1, Math.round(img.naturalWidth * k)), h = Math.max(1, Math.round(img.naturalHeight * k));
    const { ctx } = contexte(w, h);
    ctx.drawImage(img, 0, 0, w, h);
    return { rgba: ctx.getImageData(0, 0, w, h).data, w, h };
  }

  /** Image réduite pour que son grand côté ne dépasse pas maxCote (0 : taille d'origine), avec un lissage de
      qualité comme tf.image.resize(antialias=True) dans le notebook. Renvoie { pixels RGB dans [0, 1], w, h }. */
  function imageDeTravail(img, maxCote) {
    const W = img.naturalWidth, H = img.naturalHeight;
    const k = maxCote > 0 ? Math.min(1, maxCote / Math.max(W, H)) : 1;
    const w = Math.max(1, Math.round(W * k)), h = Math.max(1, Math.round(H * k));
    if (Math.min(w, h) < 16) throw new Error(`image trop petite (${w}×${h}, 16 pixels minimum)`);
    const { ctx } = contexte(w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, w, h);
    const rgba = ctx.getImageData(0, 0, w, h).data, pixels = new Float32Array(w * h * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      pixels[j] = rgba[i] / 255; pixels[j + 1] = rgba[i + 1] / 255; pixels[j + 2] = rgba[i + 2] / 255;
    }
    return { pixels, w, h };
  }

  /** Pixels RGB [0, 1] -> ImageData (RGBA 0..255, arrondi au plus proche). */
  function versImageData(pixels, w, h) {
    const id = new ImageData(w, h);
    for (let i = 0, j = 0; j < pixels.length; i += 4, j += 3) {
      id.data[i] = pixels[j] * 255; id.data[i + 1] = pixels[j + 1] * 255; id.data[i + 2] = pixels[j + 2] * 255;
      id.data[i + 3] = 255;
    }
    return id;
  }

  /** Pixels RGB [0, 1] -> image PNG (sans perte : on n'ajoute pas d'artefacts de compression). */
  function versPNG(pixels, w, h) {
    const { cv, ctx } = contexte(w, h);
    ctx.putImageData(versImageData(pixels, w, h), 0, 0);
    return new Promise((res, rej) => cv.toBlob((b) => (b ? res(b) : rej(new Error("export PNG impossible"))), "image/png"));
  }

  // ---------------------------------------------------------------------------
  // Bruit simulé : y = clip(x + n), n ~ N(0, sigma²), comme dans le notebook L2.
  // Graine fixe par image : la même image reçoit toujours le même bruit.
  // ---------------------------------------------------------------------------
  function graine(texte) { // FNV-1a 32 bits
    let h = 0x811c9dc5;
    for (let i = 0; i < texte.length; i++) { h ^= texte.charCodeAt(i); h = Math.imul(h, 0x01000193); }
    return h >>> 0;
  }

  function aleatoire(g) { // mulberry32 : générateur rapide et reproductible
    return () => {
      g = (g + 0x6d2b79f5) >>> 0;
      let t = g;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function ajouterBruit(pixels, sigma, g) {
    const u = aleatoire(g), out = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i += 2) { // Box-Muller : deux tirages gaussiens à la fois
      const r = Math.sqrt(-2 * Math.log(1 - u())), a = 2 * Math.PI * u();
      out[i] = Math.min(1, Math.max(0, pixels[i] + sigma * r * Math.cos(a)));
      if (i + 1 < pixels.length) out[i + 1] = Math.min(1, Math.max(0, pixels[i + 1] + sigma * r * Math.sin(a)));
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Mesures de qualité (mêmes définitions que tf.image.psnr et tf.image.ssim)
  // ---------------------------------------------------------------------------
  /** PSNR en dB entre deux images dans [0, 1] : 10 log10(1 / MSE). Plus haut = plus proche. */
  function psnr(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return 10 * Math.log10(1 / (s / a.length));
  }

  /** SSIM entre deux images RGB dans [0, 1] : fenêtre gaussienne 11x11 (sigma 1,5), k1 = 0,01, k2 = 0,03,
      moyenne sur les positions puis sur les canaux, comme tf.image.ssim. Entre 0 et 1, plus haut = mieux. */
  function ssim(a, b, w, h) {
    const R = 5, T = 2 * R + 1, g = new Float64Array(T);
    let s = 0;
    for (let i = 0; i < T; i++) { g[i] = Math.exp(-((i - R) ** 2) / (2 * 1.5 * 1.5)); s += g[i]; }
    for (let i = 0; i < T; i++) g[i] /= s;
    const ow = w - 2 * R, oh = h - 2 * R;
    if (ow <= 0 || oh <= 0) return NaN;
    const C1 = 0.01 ** 2, C2 = 0.03 ** 2, n = h * ow;
    const ha = new Float32Array(n), hb = new Float32Array(n), haa = new Float32Array(n), hbb = new Float32Array(n), hab = new Float32Array(n);
    let total = 0;
    for (let c = 0; c < 3; c++) {
      for (let y = 0; y < h; y++) for (let x = 0; x < ow; x++) { // filtre horizontal
        let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (let k = 0; k < T; k++) {
          const i = (y * w + x + k) * 3 + c, va = a[i], vb = b[i], gk = g[k];
          sa += gk * va; sb += gk * vb; saa += gk * va * va; sbb += gk * vb * vb; sab += gk * va * vb;
        }
        const o = y * ow + x;
        ha[o] = sa; hb[o] = sb; haa[o] = saa; hbb[o] = sbb; hab[o] = sab;
      }
      let somme = 0;
      for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) { // filtre vertical
        let ma = 0, mb = 0, maa = 0, mbb = 0, mab = 0;
        for (let k = 0; k < T; k++) {
          const o = (y + k) * ow + x, gk = g[k];
          ma += gk * ha[o]; mb += gk * hb[o]; maa += gk * haa[o]; mbb += gk * hbb[o]; mab += gk * hab[o];
        }
        const lum = (2 * ma * mb + C1) / (ma * ma + mb * mb + C1);
        const cs = (2 * (mab - ma * mb) + C2) / ((maa - ma * ma) + (mbb - mb * mb) + C2);
        somme += lum * cs;
      }
      total += somme / (ow * oh);
    }
    return total / 3;
  }

  /** Écart-type du bruit estimé sur une seule image, sans référence (méthode d'Immerkær, 1996),
      sur la luminance dans [0, 1]. Indicatif : les détails fins comptent aussi comme du « bruit ». */
  function bruitEstime(pixels, w, h) {
    if (w < 3 || h < 3) return NaN;
    const L = new Float32Array(w * h);
    for (let i = 0; i < L.length; i++) L[i] = 0.299 * pixels[3 * i] + 0.587 * pixels[3 * i + 1] + 0.114 * pixels[3 * i + 2];
    let s = 0;
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      s += Math.abs(L[i - w - 1] - 2 * L[i - w] + L[i - w + 1] - 2 * L[i - 1] + 4 * L[i] - 2 * L[i + 1] + L[i + w - 1] - 2 * L[i + w] + L[i + w + 1]);
    }
    return Math.sqrt(Math.PI / 2) * s / (6 * (w - 2) * (h - 2));
  }

  // ---------------------------------------------------------------------------
  // Fichiers
  // ---------------------------------------------------------------------------
  /** Fichiers choisis avec <input type="file"> -> [{ fichier, chemin }] (chemin relatif si c'est un dossier). */
  const depuisInput = (liste) => Array.from(liste || []).map((f) => ({ fichier: f, chemin: f.webkitRelativePath || f.name }));

  /** Fichiers déposés (glisser-déposer), dossiers et sous-dossiers compris -> [{ fichier, chemin }].
      À appeler pendant l'événement « drop » : le navigateur invalide les éléments ensuite. */
  function lireDepot(dt) {
    const entrees = Array.from(dt.items || []).filter((i) => i.kind === "file").map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null));
    if (!entrees.length || entrees.some((e) => !e)) return Promise.resolve(depuisInput(dt.files));
    const out = [];
    const lireLot = (lecteur) => new Promise((res, rej) => lecteur.readEntries(res, rej));
    async function visiter(e, prefixe) {
      if (e.isFile) {
        const f = await new Promise((res, rej) => e.file(res, rej));
        out.push({ fichier: f, chemin: prefixe + f.name });
      } else if (e.isDirectory) {
        const lecteur = e.createReader();
        for (let lot = await lireLot(lecteur); lot.length; lot = await lireLot(lecteur)) {
          for (const s of lot) await visiter(s, prefixe + e.name + "/");
        }
      }
    }
    return (async () => { for (const e of entrees) await visiter(e, ""); return out; })();
  }

  const triNaturel = (a, b) => a.localeCompare(b, "fr", { numeric: true, sensitivity: "base" });

  function telecharger(blob, nom) {
    const url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = nom;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  // ZIP sans compression (les PNG sont déjà compressés) : en-têtes locaux, répertoire central, fin.
  const TABLE_CRC = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
    return t;
  })();
  function crc32(u8) {
    let c = 0xffffffff;
    for (let i = 0; i < u8.length; i++) c = TABLE_CRC[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  /** fichiers : [{ nom: "dossier/image.png", donnees: Uint8Array }] -> Blob .zip */
  function creerZip(fichiers, date = new Date()) {
    const enc = new TextEncoder(), parts = [], central = [];
    const heure = (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
    const jour = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
    let offset = 0;
    for (const f of fichiers) {
      const nom = enc.encode(f.nom), n = f.donnees.length, crc = crc32(f.donnees);
      const l = new DataView(new ArrayBuffer(30));
      l.setUint32(0, 0x04034b50, true); l.setUint16(4, 20, true); l.setUint16(6, 0x0800, true); // noms en UTF-8
      l.setUint16(8, 0, true); l.setUint16(10, heure, true); l.setUint16(12, jour, true);
      l.setUint32(14, crc, true); l.setUint32(18, n, true); l.setUint32(22, n, true);
      l.setUint16(26, nom.length, true); l.setUint16(28, 0, true);
      parts.push(new Uint8Array(l.buffer), nom, f.donnees);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
      c.setUint16(10, 0, true); c.setUint16(12, heure, true); c.setUint16(14, jour, true);
      c.setUint32(16, crc, true); c.setUint32(20, n, true); c.setUint32(24, n, true);
      c.setUint16(28, nom.length, true); c.setUint16(30, 0, true); c.setUint16(32, 0, true);
      c.setUint16(34, 0, true); c.setUint16(36, 0, true); c.setUint32(38, 0, true); c.setUint32(42, offset, true);
      central.push(new Uint8Array(c.buffer), nom);
      offset += 30 + nom.length + n;
    }
    const tailleCentral = central.reduce((s, u) => s + u.length, 0);
    const fin = new DataView(new ArrayBuffer(22));
    fin.setUint32(0, 0x06054b50, true); fin.setUint16(8, fichiers.length, true); fin.setUint16(10, fichiers.length, true);
    fin.setUint32(12, tailleCentral, true); fin.setUint32(16, offset, true);
    return new Blob([...parts, ...central, new Uint8Array(fin.buffer)], { type: "application/zip" });
  }

  /** Tableau de lignes -> CSV lisible par Excel en français (séparateur « ; », UTF-8 avec BOM). */
  function versCSV(lignes) {
    const cellule = (v) => {
      const s = v == null ? "" : typeof v === "number" ? (Number.isFinite(v) ? String(v).replace(".", ",") : "") : String(v);
      return /[;"\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    return new Blob(["﻿" + lignes.map((l) => l.map(cellule).join(";")).join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
  }

  RUCHE.outils = {
    $, $$, esc, prochaineImage, nombre, fmtPct, fmtDb, fmtTaille, stockage, chargerScript,
    estImage, lireImage, pixelsPleineTaille, imageDeTravail, versImageData, versPNG,
    graine, ajouterBruit, psnr, ssim, bruitEstime,
    depuisInput, lireDepot, triNaturel, telecharger, crc32, creerZip, versCSV,
  };
})(typeof window !== "undefined" ? window : globalThis);
