"""
LA RUCHE : conversion des modèles Keras pour le site
=====================================================

Organisation attendue :

    Test_website/
      modeles/
        V1/
          lucky_model.keras   <- ton modèle (n'importe quel nom en .keras)
          classes.txt         <- optionnel : un nom de classe par ligne (ligne 1 = classe 0)
        V2/
          ...

Utilisation (ou double-clic sur convertir_modeles.bat) :

    python convertir_modeles.py
        Parcourt modeles/*, convertit les modèles nouveaux ou modifiés, met à jour la liste du site.

    python convertir_modeles.py chemin/vers/modele.keras V3 --classes Autre Photo
        Crée modeles/V3/, y copie le .keras, écrit classes.txt, puis convertit.

    python convertir_modeles.py --force
        Reconvertit tous les modèles.

Chaque conversion est vérifiée : le script refait le calcul du réseau avec les poids exportés
et le compare à Keras. Si ça ne correspond pas, le modèle n'est pas ajouté.

Couches gérées : Rescaling, Normalization, Conv2D, MaxPooling2D, AveragePooling2D,
GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, BatchNormalization,
Activation, ReLU, LeakyReLU, Softmax, Dropout et couches d'augmentation (ignorées).
Entrée : image 1 canal (niveaux de gris) ou 3 canaux (RGB).
"""
import argparse
import base64
import json
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODELES = os.path.join(HERE, "modeles")
CHUNK = 12_000_000  # taille max d'un fichier de poids (caractères base64)

IGNORED = {"InputLayer", "Dropout", "SpatialDropout1D", "SpatialDropout2D", "GaussianNoise",
           "GaussianDropout", "AlphaDropout", "ActivityRegularization",
           "RandomFlip", "RandomRotation", "RandomZoom", "RandomContrast", "RandomBrightness",
           "RandomTranslation", "RandomCrop", "RandomHeight", "RandomWidth"}
ACTIVATIONS = {"linear", "relu", "relu6", "leaky_relu", "elu", "selu", "sigmoid", "tanh",
               "softmax", "silu", "swish", "gelu", "softplus"}


try:  # évite les erreurs d'accents dans certaines consoles Windows
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def log(msg):
    print(msg, flush=True)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ----------------------------------------------------------------------------------------
# Export d'un modèle
# ----------------------------------------------------------------------------------------
def export_model(keras_path, folder, classes):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import numpy as np
    import keras

    model = keras.models.load_model(keras_path, compile=False)
    in_shape = [int(v) for v in model.input_shape[1:]]
    if len(in_shape) != 3 or in_shape[2] not in (1, 3):
        raise ValueError(f"entrée non gérée {in_shape} (il faut une image H x L x 1 ou 3)")

    def flat_layers(m):
        for layer in m.layers:
            if isinstance(layer, keras.Model):
                yield from flat_layers(layer)
            else:
                yield layer

    def act(a, alpha=None):
        name = a if isinstance(a, str) else getattr(a, "__name__", str(a))
        name = {"swish": "silu"}.get(name, name)
        if name not in ACTIVATIONS - {"swish"}:
            raise ValueError(f"activation non gérée : {name}")
        return {"activation": name, "alpha": alpha}

    spec, blobs = [], []
    offset = 0

    def add_weights(*arrays):
        nonlocal offset
        out = []
        for arr in arrays:
            arr = np.asarray(arr, dtype=np.float16)
            out.append({"shape": list(arr.shape), "offset": offset, "size": int(arr.size)})
            blobs.append(arr.ravel())
            offset += arr.size
        return out

    def vec(v, n=None):
        v = np.ravel(np.asarray(v, dtype=np.float64))
        return [float(x) for x in v]

    for layer in flat_layers(model):
        kind = layer.__class__.__name__
        cfg = layer.get_config()
        if kind in IGNORED:
            continue
        e = {"type": kind, "name": layer.name}
        if kind == "Rescaling":
            e = {"type": "Affine", "name": layer.name,
                 "scale": vec(cfg["scale"]), "offset": vec(cfg["offset"])}
        elif kind == "Normalization":
            if cfg.get("invert"):
                raise ValueError("Normalization(invert=True) non gérée")
            mean = np.ravel(np.asarray(layer.mean))
            var = np.ravel(np.asarray(layer.variance))
            std = np.maximum(np.sqrt(var), keras.backend.epsilon())
            e = {"type": "Affine", "name": layer.name, "scale": vec(1 / std), "offset": vec(-mean / std)}
        elif kind == "BatchNormalization":
            if cfg.get("axis") not in (-1, [-1], 3, [3]):
                raise ValueError("BatchNormalization doit porter sur le dernier axe")
            mm = np.asarray(layer.moving_mean); mv = np.asarray(layer.moving_variance)
            g = np.asarray(layer.gamma) if layer.gamma is not None else np.ones_like(mm)
            b = np.asarray(layer.beta) if layer.beta is not None else np.zeros_like(mm)
            s = g / np.sqrt(mv + cfg["epsilon"])
            e = {"type": "Affine", "name": layer.name, "scale": vec(s), "offset": vec(b - mm * s)}
        elif kind == "Conv2D":
            if list(cfg["dilation_rate"]) != [1, 1] or cfg.get("groups", 1) != 1:
                raise ValueError("Conv2D avec dilation ou groups non géré")
            if cfg.get("data_format", "channels_last") != "channels_last":
                raise ValueError("Conv2D channels_first non géré")
            w = layer.get_weights()
            e.update(kernel=list(cfg["kernel_size"]), strides=list(cfg["strides"]),
                     padding=cfg["padding"], filters=cfg["filters"], **act(cfg["activation"]),
                     weights=add_weights(*w))
        elif kind == "Dense":
            w = layer.get_weights()
            e.update(units=cfg["units"], **act(cfg["activation"]), weights=add_weights(*w))
        elif kind in ("MaxPooling2D", "AveragePooling2D"):
            e = {"type": "Pool", "name": layer.name, "mode": "max" if kind.startswith("Max") else "avg",
                 "pool": list(cfg["pool_size"]), "strides": list(cfg["strides"] or cfg["pool_size"]),
                 "padding": cfg["padding"]}
        elif kind in ("GlobalAveragePooling2D", "GlobalMaxPooling2D"):
            if cfg.get("keepdims"):
                raise ValueError(f"{kind}(keepdims=True) non géré")
            e = {"type": "GlobalPool", "name": layer.name, "mode": "max" if "Max" in kind else "avg"}
        elif kind == "Flatten":
            e = {"type": "Flatten", "name": layer.name}
        elif kind == "Activation":
            e = {"type": "Activation", "name": layer.name, **act(cfg["activation"])}
        elif kind == "ReLU":
            if cfg.get("max_value") is not None or cfg.get("threshold", 0) != 0:
                raise ValueError("ReLU avec max_value/threshold non géré")
            ns = float(cfg.get("negative_slope", 0) or 0)
            e = {"type": "Activation", "name": layer.name,
                 **(act("leaky_relu", ns) if ns else act("relu"))}
        elif kind == "LeakyReLU":
            e = {"type": "Activation", "name": layer.name,
                 **act("leaky_relu", float(cfg.get("negative_slope", cfg.get("alpha", 0.3))))}
        elif kind == "Softmax":
            e = {"type": "Activation", "name": layer.name, **act("softmax")}
        else:
            raise ValueError(f"couche non gérée par le site : {kind} ({layer.name})")
        spec.append(e)

    if not blobs:
        raise ValueError("aucun poids trouvé dans le modèle")
    weights16 = np.concatenate(blobs).astype("<f2")
    if not np.all(np.isfinite(weights16)):
        raise ValueError("des poids dépassent la plage float16 (±65504)")

    # --- type de sortie ---
    n_out = int(model.output_shape[-1])
    last_act = next((L.get("activation") for L in reversed(spec) if L.get("activation")), "linear")
    if n_out == 1:
        sortie = "sigmoid" if last_act == "sigmoid" else "logit"
        n_classes = 2
    else:
        sortie = "probas" if last_act == "softmax" else "logits"
        n_classes = n_out
    classes = list(classes or [])[:n_classes]
    classes += [f"Classe {i}" for i in range(len(classes), n_classes)]

    # --- vérification : calcul numpy avec les poids exportés vs Keras ---
    rng = np.random.default_rng(0)
    xs = [rng.uniform(0, 255, in_shape).astype("float32"),
          np.clip(np.linspace(0, 255, int(np.prod(in_shape))).reshape(in_shape)
                  + rng.normal(0, 30, in_shape), 0, 255).astype("float32")]
    ref = model.predict(np.stack(xs), verbose=0)
    wf = weights16.astype(np.float32)
    worst = 0.0
    for x, r in zip(xs, ref):
        mine = numpy_forward(x, spec, wf)
        tol = 0.02 * max(1.0, float(np.max(np.abs(r))))
        worst = max(worst, float(np.max(np.abs(mine - r))) / tol)
    if worst > 1:
        raise ValueError("la vérification a échoué : les résultats exportés ne correspondent pas à "
                         "Keras (architecture non linéaire ou couche mal gérée ?)")

    # --- métadonnées (date d'entraînement) ---
    date_saved = None
    try:
        with zipfile.ZipFile(keras_path) as z:
            date_saved = json.loads(z.read("metadata.json")).get("date_saved")
    except Exception:
        pass

    # --- écriture ---
    for f in os.listdir(folder):
        if (f.startswith("poids_") and f.endswith(".js")) or f == "modele.js":
            os.remove(os.path.join(folder, f))
    b64 = base64.b64encode(weights16.tobytes()).decode("ascii")
    parts = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
    files = []
    for i, part in enumerate(parts):
        name = f"poids_{i + 1}.js"
        files.append(name)
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write(f'RUCHE.poids({i}, "{part}");\n')
    meta = {"input": in_shape, "outputs": n_out, "sortie": sortie, "classes": classes,
            "layers": spec, "source": os.path.basename(keras_path), "date": date_saved,
            "params": int(offset), "parts": files}
    with open(os.path.join(folder, "modele.js"), "w", encoding="utf-8") as f:
        f.write("// Généré par convertir_modeles.py, ne pas modifier à la main.\n")
        f.write("RUCHE.modele(" + json.dumps(meta, ensure_ascii=False) + ");\n")
    return meta


def numpy_forward(x, spec, W):
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view as swv

    def pad_info(n, k, s, padding):
        if padding == "valid":
            return (n - k) // s + 1, 0, 0
        out = -(-n // s)
        total = max((out - 1) * s + k - n, 0)
        return out, total // 2, total - total // 2

    def activate(t, name, alpha):
        if name == "linear": return t
        if name == "relu": return np.maximum(t, 0)
        if name == "relu6": return np.clip(t, 0, 6)
        if name == "leaky_relu": return np.where(t < 0, t * (0.2 if alpha is None else alpha), t)
        if name == "elu": return np.where(t > 0, t, np.expm1(t))
        if name == "selu": return 1.0507009873554805 * np.where(t > 0, t, 1.6732632423543772 * np.expm1(t))
        if name == "sigmoid": return 1 / (1 + np.exp(-t))
        if name == "tanh": return np.tanh(t)
        if name == "silu": return t / (1 + np.exp(-t))
        if name == "softplus": return np.logaddexp(0, t)
        if name == "gelu":
            from math import erf
            return 0.5 * t * (1 + np.vectorize(erf)(t / np.sqrt(2)))
        if name == "softmax":
            e = np.exp(t - t.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)
        raise ValueError(name)

    def w(ref):
        return W[ref["offset"]:ref["offset"] + ref["size"]].reshape(ref["shape"])

    t = x.astype(np.float64)
    for L in spec:
        typ = L["type"]
        if typ == "Affine":
            t = t * np.array(L["scale"]) + np.array(L["offset"])
        elif typ == "Conv2D":
            k = w(L["weights"][0]); kh, kw = L["kernel"]; sh, sw = L["strides"]
            oh, pt, pb = pad_info(t.shape[0], kh, sh, L["padding"])
            ow, pl, pr = pad_info(t.shape[1], kw, sw, L["padding"])
            tp = np.pad(t, ((pt, pb), (pl, pr), (0, 0)))
            win = swv(tp, (kh, kw), axis=(0, 1))[::sh, ::sw][:oh, :ow]  # oh,ow,C,kh,kw
            t = np.einsum("hwcij,ijcf->hwf", win, k)
            if len(L["weights"]) > 1:
                t = t + w(L["weights"][1])
            t = activate(t, L["activation"], L["alpha"])
        elif typ == "Pool":
            ph, pw = L["pool"]; sh, sw = L["strides"]
            oh, pt, pb = pad_info(t.shape[0], ph, sh, L["padding"])
            ow, pl, pr = pad_info(t.shape[1], pw, sw, L["padding"])
            fill = -np.inf if L["mode"] == "max" else np.nan
            tp = np.pad(t, ((pt, pb), (pl, pr), (0, 0)), constant_values=fill)
            win = swv(tp, (ph, pw), axis=(0, 1))[::sh, ::sw][:oh, :ow]
            t = win.max(axis=(3, 4)) if L["mode"] == "max" else np.nanmean(win, axis=(3, 4))
        elif typ == "GlobalPool":
            t = t.max(axis=(0, 1)) if L["mode"] == "max" else t.mean(axis=(0, 1))
        elif typ == "Flatten":
            t = t.ravel()
        elif typ == "Dense":
            t = t @ w(L["weights"][0])
            if len(L["weights"]) > 1:
                t = t + w(L["weights"][1])
            t = activate(t, L["activation"], L["alpha"])
        elif typ == "Activation":
            t = activate(t, L["activation"], L["alpha"])
    return t


# ----------------------------------------------------------------------------------------
# Parcours du dossier modeles/
# ----------------------------------------------------------------------------------------
def read_classes(folder):
    p = os.path.join(folder, "classes.txt")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def needs_update(folder, keras_path):
    out = os.path.join(folder, "modele.js")
    if not os.path.exists(out):
        return True
    t = os.path.getmtime(out)
    cls = os.path.join(folder, "classes.txt")
    return os.path.getmtime(keras_path) > t or (os.path.exists(cls) and os.path.getmtime(cls) > t)


def write_index(names):
    with open(os.path.join(MODELES, "index.js"), "w", encoding="utf-8") as f:
        f.write("// Généré par convertir_modeles.py : liste des modèles affichés par le site.\n")
        f.write("RUCHE.liste(" + json.dumps(names, ensure_ascii=False) + ");\n")


def main():
    ap = argparse.ArgumentParser(description="Convertit les modèles Keras pour La Ruche.")
    ap.add_argument("keras", nargs="?", help="fichier .keras à ajouter (optionnel)")
    ap.add_argument("nom", nargs="?", help="nom du dossier du modèle, ex : V2")
    ap.add_argument("--classes", nargs="+", help="noms des classes, dans l'ordre (classe 0 en premier)")
    ap.add_argument("--force", action="store_true", help="reconvertit tous les modèles")
    args = ap.parse_args()

    os.makedirs(MODELES, exist_ok=True)

    if args.keras:
        if not args.keras.lower().endswith(".keras") or not os.path.isfile(args.keras):
            sys.exit(f"Fichier .keras introuvable : {args.keras}")
        nom = args.nom or os.path.splitext(os.path.basename(args.keras))[0]
        dest = os.path.join(MODELES, nom)
        os.makedirs(dest, exist_ok=True)
        for f in os.listdir(dest):
            if f.lower().endswith(".keras"):
                os.remove(os.path.join(dest, f))
        shutil.copy2(args.keras, dest)
        os.utime(os.path.join(dest, os.path.basename(args.keras)))  # force la conversion
        log(f"Copié dans modeles/{nom}/")
    if args.classes:
        if not args.keras:
            sys.exit("--classes s'utilise avec un fichier .keras et un nom de modèle")
        with open(os.path.join(MODELES, nom, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(args.classes) + "\n")

    ok, erreurs = [], []
    for nom in sorted(os.listdir(MODELES), key=natural_key):
        folder = os.path.join(MODELES, nom)
        if not os.path.isdir(folder) or nom.startswith((".", "_")):
            continue
        kfiles = sorted((f for f in os.listdir(folder) if f.lower().endswith(".keras")),
                        key=lambda f: os.path.getmtime(os.path.join(folder, f)))
        has_export = os.path.exists(os.path.join(folder, "modele.js"))
        if not kfiles:
            if has_export:
                log(f"[{nom}] pas de .keras, on garde l'export existant")
                ok.append(nom)
            else:
                log(f"[{nom}] ignoré : aucun fichier .keras dans le dossier")
            continue
        if len(kfiles) > 1:
            log(f"[{nom}] plusieurs .keras, on prend le plus récent : {kfiles[-1]}")
        kpath = os.path.join(folder, kfiles[-1])
        if not args.force and not needs_update(folder, kpath):
            log(f"[{nom}] à jour")
            ok.append(nom)
            continue
        log(f"[{nom}] conversion de {kfiles[-1]}…")
        try:
            meta = export_model(kpath, folder, read_classes(folder))
            log(f"[{nom}] OK : entrée {meta['input'][1]}x{meta['input'][0]}x{meta['input'][2]}, "
                f"classes {meta['classes']}, {meta['params']:,} paramètres (vérifié vs Keras)")
            ok.append(nom)
        except Exception as e:
            erreurs.append(nom)
            log(f"[{nom}] ERREUR : {e}")
            if has_export:
                log(f"[{nom}] l'ancien export est conservé")
                ok.append(nom)

    write_index(ok)
    log("")
    log(f"Modèles disponibles sur le site : {', '.join(ok) if ok else 'aucun'}")
    if erreurs:
        log(f"Modèles en erreur : {', '.join(erreurs)}")
        sys.exit(1)
    log("Recharge la page du site (F5).")


if __name__ == "__main__":
    main()
