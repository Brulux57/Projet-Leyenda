"""
LA RUCHE : conversion des modèles Keras pour le site de test
=============================================================

Le site (index.html) fait tous ses calculs dans le navigateur, avec TensorFlow.js. Ce script prend les
modèles entraînés par les notebooks (dossier ../models/), exporte leur architecture et leurs poids dans
modeles/<tâche>/<nom>/ et vérifie que le calcul exporté donne les mêmes résultats que Keras.

Les modèles affichés sur le site sont listés dans catalogue.json, rangés par livrable :

    "L1_classification" : classifieurs photo / peinture / autre (une probabilité par classe)
    "L2_debruitage"     : débruiteurs (image bruitée en entrée, image débruitée de même taille en sortie)

Ajouter un modèle : l'enregistrer dans ../models/ (les notebooks le font en fin d'entraînement), ajouter
une ligne dans catalogue.json, puis lancer ce script et recharger la page.

Utilisation (ou double-clic sur convertir_modeles.bat) :

    python convertir_modeles.py           convertit les modèles nouveaux ou modifiés
    python convertir_modeles.py --force   reconvertit tous les modèles

Couches gérées : Rescaling, Normalization, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D,
GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, UpSampling2D (nearest), Concatenate, Add,
Activation, ReLU, LeakyReLU, Softmax ; Dropout et couches d'augmentation sont ignorées (inactives en
prédiction). Modèles Sequential ou fonctionnels (connexions de saut), une entrée et une sortie.
"""
import argparse
import base64
import importlib.util
import json
import os
import sys
import zipfile

ICI = os.path.dirname(os.path.abspath(__file__))
CATALOGUE = os.path.join(ICI, "catalogue.json")
EXPORTS = os.path.join(ICI, "modeles")
TAILLE_MORCEAU = 12_000_000   # taille max d'un fichier de poids (caractères base64)
SEUIL_FLOAT16 = 4_000_000     # au-delà de ce nombre de poids, on les stocke en float16 (fichiers 2x plus petits)
ENTREE = "entree"             # nom du tenseur d'entrée dans le graphe exporté
FORMAT = 2                    # version du format d'export : un export plus ancien est refait automatiquement
CONTROLE_DEBRUITEUR = (320, 416)  # image de contrôle des débruiteurs : plus grande qu'une tuile du site

TACHES = {
    # plage : valeurs des pixels attendues par le modèle (le classifieur contient sa couche Rescaling 1/255)
    "L1_classification": {"plage": [0, 255]},
    "L2_debruitage": {"plage": [0, 1]},
}
IGNOREES = {"InputLayer", "Dropout", "SpatialDropout1D", "SpatialDropout2D", "GaussianNoise",
            "GaussianDropout", "AlphaDropout", "ActivityRegularization",
            "RandomFlip", "RandomRotation", "RandomZoom", "RandomContrast", "RandomBrightness",
            "RandomTranslation", "RandomCrop", "RandomHeight", "RandomWidth"}
ACTIVATIONS = {"linear", "relu", "relu6", "leaky_relu", "elu", "selu", "sigmoid", "tanh",
               "softmax", "silu", "gelu", "softplus"}

try:  # évite les erreurs d'accents dans certaines consoles Windows
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def log(msg=""):
    print(msg, flush=True)


def choisir_backend():
    """Keras 3 a besoin d'un backend : TensorFlow si installé (WSL), sinon PyTorch ou JAX (venv Windows)."""
    if os.environ.get("KERAS_BACKEND"):
        return
    for nom in ("tensorflow", "torch", "jax"):
        if importlib.util.find_spec(nom):
            os.environ["KERAS_BACKEND"] = nom
            return


# ----------------------------------------------------------------------------------------
# Lecture du graphe d'un modèle Keras
# ----------------------------------------------------------------------------------------
def references(obj):
    """Noms des couches dans input_layers / output_layers de la config (un ou plusieurs tenseurs)."""
    if isinstance(obj, (list, tuple)) and len(obj) == 3 and isinstance(obj[0], str) and isinstance(obj[1], int):
        return [obj[0]]
    return [nom for o in obj for nom in references(o)]


def tenseurs_recus(noeuds_entrants):
    """Noms des couches dont la couche reçoit la sortie, dans l'ordre (formats Keras 3 et Keras 2).
    L'ordre compte : c'est l'ordre des canaux dans un Concatenate."""
    noms = []

    def visite(o):
        if isinstance(o, dict):
            if o.get("class_name") == "__keras_tensor__":
                noms.append(o["config"]["keras_history"][0])
                return
            for v in o.values():
                visite(v)
        elif isinstance(o, (list, tuple)):
            if len(o) >= 3 and isinstance(o[0], str) and isinstance(o[1], int) and isinstance(o[2], int):
                noms.append(o[0])  # format Keras 2 : [nom, indice du noeud, indice du tenseur, kwargs]
                return
            for v in o:
                visite(v)

    visite(noeuds_entrants)
    return noms


def graphe(model, keras):
    """Liste (couche, [noms des tenseurs reçus]) dans l'ordre de calcul, et nom du tenseur de sortie."""
    if isinstance(model, keras.Sequential):
        def a_plat(m):
            for couche in m.layers:
                if isinstance(couche, keras.Sequential):
                    yield from a_plat(couche)   # ex. bloc d'augmentation de données
                elif isinstance(couche, keras.Model):
                    raise ValueError(f"sous-modèle non séquentiel non géré : {couche.name}")
                else:
                    yield couche
        noeuds, prec = [], ENTREE
        for couche in a_plat(model):
            noeuds.append((couche, [prec]))
            prec = couche.name
        return noeuds, prec

    cfg = model.get_config()
    entrees, sorties = references(cfg["input_layers"]), references(cfg["output_layers"])
    if len(entrees) != 1 or len(sorties) != 1:
        raise ValueError("il faut un modèle à une seule entrée et une seule sortie")
    recus = {}
    for L in cfg["layers"]:
        if len(L.get("inbound_nodes") or []) > 1:
            raise ValueError(f"couche partagée (appelée plusieurs fois) non gérée : {L['name']}")
        recus[L["name"]] = tenseurs_recus(L.get("inbound_nodes") or [])
    # tri topologique : une couche n'est calculée qu'une fois toutes ses entrées connues
    ordre, faits, restants = [], set(), [L["name"] for L in cfg["layers"]]
    while restants:
        pret = next((n for n in restants if all(r in faits for r in recus[n])), None)
        if pret is None:
            raise ValueError("graphe du modèle illisible (dépendances circulaires ?)")
        restants.remove(pret)
        faits.add(pret)
        ordre.append(pret)
    noeuds = []
    for nom in ordre:
        if nom == entrees[0]:
            continue
        couche = model.get_layer(nom)
        if isinstance(couche, keras.Model):
            raise ValueError(f"sous-modèle dans un graphe non géré : {nom}")
        noeuds.append((couche, [ENTREE if r == entrees[0] else r for r in recus[nom]]))
    return noeuds, sorties[0]


# ----------------------------------------------------------------------------------------
# Export d'un modèle
# ----------------------------------------------------------------------------------------
def exporter(chemin_keras, dossier, tache, entree_catalogue):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    choisir_backend()
    import numpy as np
    import keras

    model = keras.models.load_model(chemin_keras, compile=False)
    in_shape = [None if v is None else int(v) for v in model.input_shape[1:]]
    if len(in_shape) != 3 or in_shape[2] not in (1, 3):
        raise ValueError(f"entrée non gérée {in_shape} (il faut une image H x L x 1 ou 3)")
    if tache == "L1_classification" and None in in_shape:
        raise ValueError("un classifieur doit avoir une taille d'entrée fixe")
    if tache == "L2_debruitage" and in_shape[2] != 3:
        raise ValueError("un débruiteur doit recevoir une image RGB")

    def act(a, alpha=None, max_value=None):
        nom = a if isinstance(a, str) else getattr(a, "__name__", str(a))
        nom = {"swish": "silu"}.get(nom, nom)
        if nom not in ACTIVATIONS:
            raise ValueError(f"activation non gérée : {nom}")
        e = {"activation": nom, "alpha": alpha}
        if max_value is not None:
            e["max_value"] = float(max_value)
        return e

    def vec(v):
        return [float(x) for x in np.ravel(np.asarray(v, dtype=np.float64))]

    couches = []   # (entrée de spec, [tableaux de poids])
    alias = {}     # couche ignorée -> tenseur qu'elle laisse passer tel quel
    noeuds, sortie = graphe(model, keras)
    for couche, recus in noeuds:
        kind = couche.__class__.__name__
        cfg = couche.get_config()
        recus = [alias.get(r, r) for r in recus]
        if kind in IGNOREES:
            alias[couche.name] = recus[0]
            continue
        multi = kind in ("Concatenate", "Add")
        if (len(recus) < 2) if multi else (len(recus) != 1):
            raise ValueError(f"nombre d'entrées inattendu pour {couche.name} ({kind})")
        e, poids = {"type": kind}, []
        if kind == "Rescaling":
            e = {"type": "Affine", "scale": vec(cfg["scale"]), "offset": vec(cfg["offset"])}
        elif kind == "Normalization":
            if cfg.get("invert"):
                raise ValueError("Normalization(invert=True) non gérée")
            mean = np.ravel(np.asarray(couche.mean))
            std = np.maximum(np.sqrt(np.ravel(np.asarray(couche.variance))), keras.backend.epsilon())
            e = {"type": "Affine", "scale": vec(1 / std), "offset": vec(-mean / std)}
        elif kind == "BatchNormalization":
            if cfg.get("axis") not in (-1, [-1], 3, [3]):
                raise ValueError("BatchNormalization doit porter sur le dernier axe")
            mm, mv = np.asarray(couche.moving_mean), np.asarray(couche.moving_variance)
            g = np.asarray(couche.gamma) if couche.gamma is not None else np.ones_like(mm)
            b = np.asarray(couche.beta) if couche.beta is not None else np.zeros_like(mm)
            s = g / np.sqrt(mv + cfg["epsilon"])
            e = {"type": "Affine", "scale": vec(s), "offset": vec(b - mm * s)}
        elif kind == "Conv2D":
            if list(cfg["dilation_rate"]) != [1, 1] or cfg.get("groups", 1) != 1:
                raise ValueError("Conv2D avec dilation ou groups non géré")
            if cfg.get("data_format", "channels_last") != "channels_last":
                raise ValueError("Conv2D channels_first non géré")
            e.update(kernel=list(cfg["kernel_size"]), strides=list(cfg["strides"]), padding=cfg["padding"],
                     filters=cfg["filters"], **act(cfg["activation"]))
            poids = couche.get_weights()
        elif kind == "Dense":
            e.update(units=cfg["units"], **act(cfg["activation"]))
            poids = couche.get_weights()
        elif kind in ("MaxPooling2D", "AveragePooling2D"):
            e = {"type": "Pool", "mode": "max" if kind.startswith("Max") else "avg",
                 "pool": list(cfg["pool_size"]), "strides": list(cfg["strides"] or cfg["pool_size"]),
                 "padding": cfg["padding"]}
        elif kind in ("GlobalAveragePooling2D", "GlobalMaxPooling2D"):
            if cfg.get("keepdims"):
                raise ValueError(f"{kind}(keepdims=True) non géré")
            e = {"type": "GlobalPool", "mode": "max" if "Max" in kind else "avg"}
        elif kind == "Flatten":
            e = {"type": "Flatten"}
        elif kind == "UpSampling2D":
            if cfg.get("interpolation", "nearest") != "nearest":
                raise ValueError("UpSampling2D : seule l'interpolation 'nearest' est gérée")
            if cfg.get("data_format", "channels_last") != "channels_last":
                raise ValueError("UpSampling2D channels_first non géré")
            e = {"type": "UpSampling", "size": [int(v) for v in cfg["size"]]}
        elif kind == "Concatenate":
            if cfg.get("axis", -1) not in (-1, 3):
                raise ValueError("Concatenate : seule la concaténation des canaux (axis=-1) est gérée")
            e = {"type": "Concatenate"}
        elif kind == "Add":
            e = {"type": "Add"}
        elif kind == "Activation":
            e = {"type": "Activation", **act(cfg["activation"])}
        elif kind == "ReLU":
            if float(cfg.get("threshold") or 0) != 0:
                raise ValueError("ReLU avec threshold non géré")
            pente, plafond = float(cfg.get("negative_slope") or 0), cfg.get("max_value")
            if pente and plafond is not None:
                raise ValueError("ReLU avec negative_slope et max_value à la fois non géré")
            e = {"type": "Activation", **(act("leaky_relu", pente) if pente else act("relu", max_value=plafond))}
        elif kind == "LeakyReLU":
            e = {"type": "Activation", **act("leaky_relu", float(cfg.get("negative_slope", cfg.get("alpha", 0.3))))}
        elif kind == "Softmax":
            e = {"type": "Activation", **act("softmax")}
        else:
            raise ValueError(f"couche non gérée par le site : {kind} ({couche.name})")
        e["name"], e["inputs"] = couche.name, recus
        couches.append((e, poids))
    sortie = alias.get(sortie, sortie)
    if sortie == ENTREE:
        raise ValueError("le modèle ne fait aucun calcul")

    # --- poids : float16 pour les gros modèles (taille des fichiers), float32 sinon (précision) ---
    n_poids = sum(int(np.asarray(p).size) for _, ps in couches for p in ps)
    if not n_poids:
        raise ValueError("aucun poids trouvé dans le modèle")
    dtype = "<f2" if n_poids > SEUIL_FLOAT16 else "<f4"
    spec, blobs, offset = [], [], 0
    for e, ps in couches:
        if ps:
            refs = []
            for p in ps:
                p = np.asarray(p, dtype=dtype)
                refs.append({"shape": list(p.shape), "offset": offset, "size": int(p.size)})
                blobs.append(p.ravel())
                offset += p.size
            e["weights"] = refs
        spec.append(e)
    poids = np.concatenate(blobs).astype(dtype)
    if not np.all(np.isfinite(poids)):
        raise ValueError("des poids dépassent la plage float16 (±65504)")

    # --- sortie du modèle ---
    meta = {"tache": tache, "input": in_shape, "plage": TACHES[tache]["plage"]}
    if tache == "L1_classification":
        if len(model.output_shape) != 2:
            raise ValueError(f"un classifieur doit sortir un vecteur, pas {model.output_shape}")
        n_out = int(model.output_shape[-1])
        dernier = next((L.get("activation") for L in reversed(spec) if L.get("activation")), "linear")
        if n_out == 1:
            meta["sortie"], n_classes = ("sigmoid" if dernier == "sigmoid" else "logit"), 2
        else:
            meta["sortie"], n_classes = ("probas" if dernier == "softmax" else "logits"), n_out
        classes = list(entree_catalogue.get("classes") or [])[:n_classes]
        meta["classes"] = classes + [f"Classe {i}" for i in range(len(classes), n_classes)]
    else:
        meta["sortie"] = "image"
        # hauteur et largeur doivent être multiples du produit des pas (3 MaxPooling 2x2 -> 8)
        multiple = 1
        for L in spec:
            if L["type"] in ("Pool", "Conv2D"):
                multiple *= max(L["strides"])
        meta["multiple"] = multiple

    # --- vérification : calcul numpy avec les poids exportés contre Keras ---
    lo, hi = TACHES[tache]["plage"]
    m = meta.get("multiple", 1)
    formes = [in_shape] if None not in in_shape else [[6 * m, 8 * m, 3], [5 * m, 5 * m, 3]]
    rng = np.random.default_rng(0)
    pire, ecart = 0.0, 0.0
    for forme in formes:
        n = int(np.prod(forme))
        xs = np.stack([rng.uniform(lo, hi, forme),
                       np.clip(np.linspace(lo, hi, n).reshape(forme) + rng.normal(0, (hi - lo) / 8, forme), lo, hi)]
                      ).astype("float32")
        ref = np.asarray(model.predict(xs, verbose=0))
        if tache == "L2_debruitage" and ref.shape != xs.shape:
            raise ValueError(f"un débruiteur doit rendre une image de même taille ({ref.shape} pour {xs.shape})")
        for x, r in zip(xs, ref):
            mien = calcul_numpy(x, spec, poids.astype(np.float64), sortie)
            diff = float(np.max(np.abs(mien - r)))
            ecart = max(ecart, diff)
            pire = max(pire, diff / (0.02 * max(1.0, float(np.max(np.abs(r))))))
    if pire > 1:
        raise ValueError("la vérification a échoué : le calcul exporté ne correspond pas à Keras")

    # --- contrôle : sortie de référence que le site recalcule au chargement pour vérifier la carte graphique ---
    if tache == "L1_classification":
        x = motif_controle(in_shape, (lo, hi))
        controle = {"forme": in_shape, "attendu": [float(v) for v in np.ravel(calcul_numpy(x, spec, poids.astype(np.float64), sortie))]}
    else:
        h, w = CONTROLE_DEBRUITEUR
        y = calcul_numpy(motif_controle([h, w, 3], (lo, hi)), spec, poids.astype(np.float64), sortie)
        blocs = np.clip(y, 0, 1).reshape(h // 8, 8, w // 8, 8, 3).mean(axis=(1, 3)).astype("<f4")
        controle = {"forme": [h, w, 3], "bloc": 8, "attendu": base64.b64encode(blocs.tobytes()).decode("ascii")}

    # --- date d'entraînement (enregistrée par Keras dans le .keras) ---
    date = None
    try:
        with zipfile.ZipFile(chemin_keras) as z:
            date = json.loads(z.read("metadata.json")).get("date_saved")
    except Exception:
        pass

    # --- écriture ---
    os.makedirs(dossier, exist_ok=True)
    for f in os.listdir(dossier):
        if (f.startswith("poids_") and f.endswith(".js")) or f == "modele.js":
            os.remove(os.path.join(dossier, f))
    b64 = base64.b64encode(poids.tobytes()).decode("ascii")
    fichiers = []
    for i in range(0, len(b64), TAILLE_MORCEAU):
        nom = f"poids_{len(fichiers) + 1}.js"
        with open(os.path.join(dossier, nom), "w", encoding="utf-8") as f:
            f.write(f'RUCHE.poids({len(fichiers)}, "{b64[i:i + TAILLE_MORCEAU]}");\n')
        fichiers.append(nom)
    meta.update(format=FORMAT, controle=controle, entree=ENTREE, sortie_noeud=sortie, layers=spec,
                poids_type="float16" if dtype == "<f2" else "float32",
                source=os.path.basename(chemin_keras), date=date, params=int(offset), parts=fichiers,
                description=entree_catalogue.get("description", ""), catalogue=signature(entree_catalogue),
                ecart_verification=float(f"{ecart:.2e}"))
    with open(os.path.join(dossier, "modele.js"), "w", encoding="utf-8") as f:
        f.write("// Généré par convertir_modeles.py, ne pas modifier à la main.\n")
        f.write("RUCHE.modele(" + json.dumps(meta, ensure_ascii=False) + ");\n")
    return meta


def motif_controle(forme, plage):
    """Image de contrôle, identique à motifControle() de js/moteur.js : moitié dégradé horizontal,
    moitié valeurs pseudo-aléatoires (hachage entier, même résultat en Python et en JavaScript)."""
    import numpy as np
    h, w, c = forme
    i = np.arange(h * w * c, dtype=np.uint64)
    hasard = ((i * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0
    colonne = (i // np.uint64(c)) % np.uint64(w)
    lo, hi = plage
    v = lo + (hi - lo) * (0.5 * hasard + 0.5 * colonne.astype(np.float64) / (w - 1))
    return v.astype(np.float32).reshape(h, w, c)


def calcul_numpy(x, spec, W, sortie):
    """Refait le calcul du réseau avec numpy (référence pour la vérification)."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view as fenetres

    def pad_info(n, k, s, padding):
        if padding == "valid":
            return (n - k) // s + 1, 0, 0
        out = -(-n // s)
        total = max((out - 1) * s + k - n, 0)
        return out, total // 2, total - total // 2

    def activer(t, L):
        nom, alpha = L.get("activation", "linear"), L.get("alpha")
        if nom == "relu": t = np.maximum(t, 0)
        elif nom == "relu6": t = np.clip(t, 0, 6)
        elif nom == "leaky_relu": t = np.where(t < 0, t * (0.2 if alpha is None else alpha), t)
        elif nom == "elu": t = np.where(t > 0, t, np.expm1(t))
        elif nom == "selu": t = 1.0507009873554805 * np.where(t > 0, t, 1.6732632423543772 * np.expm1(t))
        elif nom == "sigmoid": t = 1 / (1 + np.exp(-t))
        elif nom == "tanh": t = np.tanh(t)
        elif nom == "silu": t = t / (1 + np.exp(-t))
        elif nom == "softplus": t = np.logaddexp(0, t)
        elif nom == "gelu":
            from math import erf
            t = 0.5 * t * (1 + np.vectorize(erf)(t / np.sqrt(2)))
        elif nom == "softmax":
            e = np.exp(t - t.max(axis=-1, keepdims=True))
            t = e / e.sum(axis=-1, keepdims=True)
        elif nom != "linear":
            raise ValueError(nom)
        if L.get("max_value") is not None:
            t = np.minimum(t, L["max_value"])
        return t

    def w(ref):
        return W[ref["offset"]:ref["offset"] + ref["size"]].reshape(ref["shape"])

    valeurs, prec = {ENTREE: x.astype(np.float64)}, ENTREE
    for L in spec:
        ins = [valeurs[n] for n in L.get("inputs", [prec])]
        t, typ = ins[0], L["type"]
        if typ == "Affine":
            t = t * np.array(L["scale"]) + np.array(L["offset"])
        elif typ == "Conv2D":
            k = w(L["weights"][0]); kh, kw = L["kernel"]; sh, sw = L["strides"]
            oh, pt, pb = pad_info(t.shape[0], kh, sh, L["padding"])
            ow, pl, pr = pad_info(t.shape[1], kw, sw, L["padding"])
            tp = np.pad(t, ((pt, pb), (pl, pr), (0, 0)))
            win = fenetres(tp, (kh, kw), axis=(0, 1))[::sh, ::sw][:oh, :ow]  # oh, ow, C, kh, kw
            t = np.einsum("hwcij,ijcf->hwf", win, k, optimize=True)
            if len(L["weights"]) > 1:
                t = t + w(L["weights"][1])
            t = activer(t, L)
        elif typ == "Pool":
            ph, pw = L["pool"]; sh, sw = L["strides"]
            oh, pt, pb = pad_info(t.shape[0], ph, sh, L["padding"])
            ow, pl, pr = pad_info(t.shape[1], pw, sw, L["padding"])
            vide = -np.inf if L["mode"] == "max" else np.nan
            tp = np.pad(t, ((pt, pb), (pl, pr), (0, 0)), constant_values=vide)
            win = fenetres(tp, (ph, pw), axis=(0, 1))[::sh, ::sw][:oh, :ow]
            t = win.max(axis=(3, 4)) if L["mode"] == "max" else np.nanmean(win, axis=(3, 4))
        elif typ == "GlobalPool":
            t = t.max(axis=(0, 1)) if L["mode"] == "max" else t.mean(axis=(0, 1))
        elif typ == "Flatten":
            t = t.ravel()
        elif typ == "Dense":
            t = t @ w(L["weights"][0])
            if len(L["weights"]) > 1:
                t = t + w(L["weights"][1])
            t = activer(t, L)
        elif typ == "UpSampling":
            t = t.repeat(L["size"][0], axis=0).repeat(L["size"][1], axis=1)
        elif typ == "Concatenate":
            t = np.concatenate(ins, axis=-1)
        elif typ == "Add":
            t = np.sum(ins, axis=0)
        elif typ == "Activation":
            t = activer(t, L)
        else:
            raise ValueError(typ)
        valeurs[L["name"]] = t
        prec = L["name"]
    return valeurs[sortie]


# ----------------------------------------------------------------------------------------
# Catalogue et dossier modeles/
# ----------------------------------------------------------------------------------------
def signature(entree_catalogue):
    return json.dumps(entree_catalogue, sort_keys=True, ensure_ascii=False)


def lire_meta(dossier):
    try:
        with open(os.path.join(dossier, "modele.js"), encoding="utf-8") as f:
            texte = f.read()
        return json.loads(texte[texte.index("RUCHE.modele(") + len("RUCHE.modele("):texte.rindex(")")])
    except (OSError, ValueError):
        return None


def a_jour(dossier, chemin_keras, entree_catalogue):
    meta = lire_meta(dossier)
    if meta is None or meta.get("format") != FORMAT or meta.get("catalogue") != signature(entree_catalogue):
        return False
    return os.path.getmtime(chemin_keras) <= os.path.getmtime(os.path.join(dossier, "modele.js"))


def ecrire_index(disponibles):
    with open(os.path.join(EXPORTS, "index.js"), "w", encoding="utf-8") as f:
        f.write("// Généré par convertir_modeles.py : modèles affichés par le site, rangés par tâche.\n")
        f.write("RUCHE.liste(" + json.dumps(disponibles, ensure_ascii=False) + ");\n")


def main():
    ap = argparse.ArgumentParser(description="Convertit les modèles Keras du catalogue pour le site.")
    ap.add_argument("--force", action="store_true", help="reconvertit tous les modèles")
    args = ap.parse_args()

    try:
        with open(CATALOGUE, encoding="utf-8") as f:
            catalogue = json.load(f)
    except (OSError, ValueError) as e:
        sys.exit(f"catalogue.json illisible : {e}")
    dossier_modeles = os.path.normpath(os.path.join(ICI, catalogue.get("dossier_modeles", "../models")))
    inconnues = [k for k in catalogue if k != "dossier_modeles" and k not in TACHES]
    if inconnues:
        sys.exit(f"tâche(s) inconnue(s) dans catalogue.json : {inconnues} (attendu : {list(TACHES)})")

    disponibles, erreurs = {}, []
    for tache in TACHES:
        disponibles[tache] = []
        for nom, entree in (catalogue.get(tache) or {}).items():
            dossier = os.path.join(EXPORTS, tache, nom)
            etiquette = f"[{tache}/{nom}]"
            chemin = os.path.join(dossier_modeles, entree.get("fichier", ""))
            existe = lire_meta(dossier) is not None
            if not entree.get("fichier") or not os.path.isfile(chemin):
                if existe:
                    log(f"{etiquette} {entree.get('fichier')} absent de {dossier_modeles} : on garde l'export existant")
                    disponibles[tache].append(nom)
                else:
                    log(f"{etiquette} ignoré : {entree.get('fichier')} introuvable dans {dossier_modeles}")
                continue
            if not args.force and a_jour(dossier, chemin, entree):
                log(f"{etiquette} à jour")
                disponibles[tache].append(nom)
                continue
            log(f"{etiquette} conversion de {entree['fichier']}...")
            try:
                meta = exporter(chemin, dossier, tache, entree)
                forme = "x".join("?" if v is None else str(v) for v in meta["input"])
                quoi = f"classes {meta['classes']}" if "classes" in meta else f"image -> image (côtés multiples de {meta['multiple']})"
                log(f"{etiquette} OK : entrée {forme}, {quoi}, {meta['params']:,} paramètres en {meta['poids_type']}, "
                    f"écart max avec Keras {meta['ecart_verification']:.1e}")
                disponibles[tache].append(nom)
            except Exception as e:
                erreurs.append(f"{tache}/{nom}")
                log(f"{etiquette} ERREUR : {e}")
                if existe:
                    log(f"{etiquette} l'ancien export est conservé")
                    disponibles[tache].append(nom)

    os.makedirs(EXPORTS, exist_ok=True)
    ecrire_index(disponibles)
    log()
    for tache, noms in disponibles.items():
        log(f"{tache} : {', '.join(noms) if noms else 'aucun modèle'}")
    if erreurs:
        log(f"Modèles en erreur : {', '.join(erreurs)}")
        sys.exit(1)
    log("Recharge la page du site (F5).")


if __name__ == "__main__":
    main()
