/* Mini moteur d'inférence pour les modèles Keras exportés par convertir_modeles.py
   (format NHWC, une image à la fois). Couches : Affine (Rescaling / Normalization /
   BatchNormalization), Conv2D, Pool, GlobalPool, Flatten, Dense, Activation. */
(function (root) {
  "use strict";

  // ---------- décodage base64 -> float16 -> float32 ----------
  function base64ToBytes(b64) {
    if (typeof atob === "undefined") return new Uint8Array(Buffer.from(b64, "base64"));
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  let LUT = null;
  function halfTable() {
    if (LUT) return LUT;
    LUT = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      if (e === 0) LUT[h] = s * Math.pow(2, -14) * (f / 1024);
      else if (e === 31) LUT[h] = f ? NaN : s * Infinity;
      else LUT[h] = s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return LUT;
  }

  function decodeWeights(b64) {
    const bytes = base64ToBytes(b64);
    const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
    const lut = halfTable();
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = lut[u16[i]];
    return out;
  }

  // ---------- activations (valeurs par défaut de Keras 3) ----------
  function erf(x) { // Abramowitz-Stegun 7.1.26 raffinée (erreur < 1.2e-7)
    const t = 1 / (1 + 0.5 * Math.abs(x));
    const y = 1 - t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? y : -y;
  }

  function activate(x, name, alpha, channels) {
    const n = x.length;
    switch (name) {
      case "linear": case undefined: case null: return;
      case "relu": for (let i = 0; i < n; i++) if (x[i] < 0) x[i] = 0; return;
      case "relu6": for (let i = 0; i < n; i++) x[i] = Math.min(Math.max(x[i], 0), 6); return;
      case "leaky_relu": { const a = alpha == null ? 0.2 : alpha; for (let i = 0; i < n; i++) if (x[i] < 0) x[i] *= a; return; }
      case "elu": for (let i = 0; i < n; i++) if (x[i] < 0) x[i] = Math.expm1(x[i]); return;
      case "selu": for (let i = 0; i < n; i++) x[i] = 1.0507009873554805 * (x[i] > 0 ? x[i] : 1.6732632423543772 * Math.expm1(x[i])); return;
      case "sigmoid": for (let i = 0; i < n; i++) x[i] = 1 / (1 + Math.exp(-x[i])); return;
      case "tanh": for (let i = 0; i < n; i++) x[i] = Math.tanh(x[i]); return;
      case "silu": for (let i = 0; i < n; i++) x[i] = x[i] / (1 + Math.exp(-x[i])); return;
      case "softplus": for (let i = 0; i < n; i++) x[i] = x[i] > 30 ? x[i] : Math.log1p(Math.exp(x[i])); return;
      case "gelu": for (let i = 0; i < n; i++) x[i] = 0.5 * x[i] * (1 + erf(x[i] / Math.SQRT2)); return;
      case "softmax": { const C = channels || n; for (let o = 0; o < n; o += C) softmaxInPlace(x, o, C); return; }
      default: throw new Error("Activation non gérée : " + name);
    }
  }

  function softmaxInPlace(x, off = 0, len = x.length) {
    let m = -Infinity, s = 0;
    for (let i = off; i < off + len; i++) m = Math.max(m, x[i]);
    for (let i = off; i < off + len; i++) { x[i] = Math.exp(x[i] - m); s += x[i]; }
    for (let i = off; i < off + len; i++) x[i] /= s;
  }

  // Keras "same" : pad total = max((ceil(in/s)-1)*s + k - in, 0), moitié (arrondie en bas) avant
  function padInfo(inSize, k, s, padding) {
    if (padding === "valid") return { out: Math.floor((inSize - k) / s) + 1, before: 0 };
    const out = Math.ceil(inSize / s);
    const total = Math.max((out - 1) * s + k - inSize, 0);
    return { out, before: Math.floor(total / 2) };
  }

  const view = (W, ref) => W.subarray(ref.offset, ref.offset + ref.size);

  function affine(t, L) {
    const d = t.data, C = t.shape[t.shape.length - 1];
    const sc = L.scale, of = L.offset, ns = sc.length, no = of.length;
    for (let i = 0; i < d.length; i++) {
      const c = i % C;
      d[i] = d[i] * sc[ns === 1 ? 0 : c] + of[no === 1 ? 0 : c];
    }
    return t;
  }

  function conv2d(t, L, W) {
    const [H, Wd, C] = t.shape;
    const [kh, kw] = L.kernel, [sh, sw] = L.strides, F = L.filters;
    const py = padInfo(H, kh, sh, L.padding), px = padInfo(Wd, kw, sw, L.padding);
    const k = view(W, L.weights[0]), b = L.weights[1] ? view(W, L.weights[1]) : null;
    const OH = py.out, OW = px.out, out = new Float32Array(OH * OW * F);
    const x = t.data;
    for (let oy = 0; oy < OH; oy++) {
      for (let ox = 0; ox < OW; ox++) {
        const o = (oy * OW + ox) * F;
        if (b) for (let f = 0; f < F; f++) out[o + f] = b[f];
        for (let ky = 0; ky < kh; ky++) {
          const iy = oy * sh + ky - py.before;
          if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < kw; kx++) {
            const ix = ox * sw + kx - px.before;
            if (ix < 0 || ix >= Wd) continue;
            const xi = (iy * Wd + ix) * C, ki = (ky * kw + kx) * C * F;
            for (let c = 0; c < C; c++) {
              const v = x[xi + c];
              if (v === 0) continue;
              const kk = ki + c * F;
              for (let f = 0; f < F; f++) out[o + f] += v * k[kk + f];
            }
          }
        }
      }
    }
    activate(out, L.activation, L.alpha, F);
    return { shape: [OH, OW, F], data: out };
  }

  function pool2d(t, L) {
    const isMax = L.mode === "max";
    const [H, Wd, C] = t.shape;
    const [ph, pw] = L.pool, [sh, sw] = L.strides;
    const py = padInfo(H, ph, sh, L.padding), px = padInfo(Wd, pw, sw, L.padding);
    const OH = py.out, OW = px.out, out = new Float32Array(OH * OW * C), x = t.data;
    for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) for (let c = 0; c < C; c++) {
      let acc = isMax ? -Infinity : 0, n = 0;
      for (let ky = 0; ky < ph; ky++) {
        const iy = oy * sh + ky - py.before; if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < pw; kx++) {
          const ix = ox * sw + kx - px.before; if (ix < 0 || ix >= Wd) continue;
          const v = x[(iy * Wd + ix) * C + c];
          if (isMax) { if (v > acc) acc = v; } else { acc += v; n++; }
        }
      }
      out[(oy * OW + ox) * C + c] = isMax ? acc : acc / n;
    }
    return { shape: [OH, OW, C], data: out };
  }

  function globalPool(t, L) {
    const C = t.shape[t.shape.length - 1], P = t.data.length / C, x = t.data;
    const out = new Float32Array(C).fill(L.mode === "max" ? -Infinity : 0);
    for (let p = 0; p < P; p++) for (let c = 0; c < C; c++) {
      const v = x[p * C + c];
      if (L.mode === "max") { if (v > out[c]) out[c] = v; } else out[c] += v;
    }
    if (L.mode !== "max") for (let c = 0; c < C; c++) out[c] /= P;
    return { shape: [C], data: out };
  }

  function dense(t, L, W) {
    const x = t.data, N = x.length, U = L.units;
    const k = view(W, L.weights[0]);
    const out = new Float32Array(U);
    if (L.weights[1]) out.set(view(W, L.weights[1]));
    for (let i = 0; i < N; i++) {
      const v = x[i];
      if (v === 0) continue;
      const r = i * U;
      for (let u = 0; u < U; u++) out[u] += v * k[r + u];
    }
    activate(out, L.activation, L.alpha, U);
    return { shape: [U], data: out };
  }

  function createModel(meta, b64) {
    const W = decodeWeights(b64);
    return {
      meta,
      /** pixels : Float32Array H x W x C (0..255), déjà à la taille d'entrée. Renvoie les sorties brutes. */
      raw(pixels) {
        let t = { shape: meta.input.slice(), data: Float32Array.from(pixels) };
        for (const L of meta.layers) {
          switch (L.type) {
            case "Affine": t = affine(t, L); break;
            case "Conv2D": t = conv2d(t, L, W); break;
            case "Pool": t = pool2d(t, L); break;
            case "GlobalPool": t = globalPool(t, L); break;
            case "Flatten": t = { shape: [t.data.length], data: t.data }; break;
            case "Dense": t = dense(t, L, W); break;
            case "Activation": activate(t.data, L.activation, L.alpha, t.shape[t.shape.length - 1]); break;
            default: throw new Error("Couche non gérée : " + L.type);
          }
        }
        return Array.from(t.data);
      },
      /** Renvoie { raw, probs } avec une probabilité par classe. */
      predict(pixels) {
        const raw = this.raw(pixels);
        let probs;
        switch (meta.sortie) {
          case "probas": probs = raw.slice(); break;
          case "sigmoid": probs = [1 - raw[0], raw[0]]; break;
          case "logit": { const p = 1 / (1 + Math.exp(-raw[0])); probs = [1 - p, p]; break; }
          default: probs = softmax(raw);
        }
        return { raw, probs };
      },
    };
  }

  function softmax(a) { const x = Float32Array.from(a); softmaxInPlace(x); return Array.from(x); }

  /* Redimensionnement bilinéaire identique à tf.image.resize (half_pixel_centers, sans antialias),
     ce qu'utilise image_dataset_from_directory. channels = 3 (RGB) ou 1 (niveaux de gris). */
  function resizeBilinear(rgba, srcW, srcH, dstW, dstH, channels = 3) {
    let src = rgba, stride = 4;
    if (channels === 1) { // conversion en gris comme libjpeg / tf : 0.299 R + 0.587 G + 0.114 B
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
          const top = src[a + ch] + (src[b + ch] - src[a + ch]) * dx;
          const bot = src[c + ch] + (src[d + ch] - src[c + ch]) * dx;
          out[o + ch] = top + (bot - top) * dy;
        }
      }
    }
    return out;
  }

  const api = { createModel, resizeBilinear, softmax };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LuckyEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
