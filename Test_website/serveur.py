"""
LA RUCHE : serveur local du site de test
=========================================

Fait tourner les modèles du projet avec leurs vraies bibliothèques, sur la carte graphique si elle est disponible :

    L1_classification   classifieurs Keras (.keras)          TensorFlow
    L2_debruitage       débruiteurs Keras (.keras)           TensorFlow
    L3_captioning       BLIP (dossier save_pretrained)       PyTorch + Transformers

et sert le site lui-même : ouvrir http://localhost:8765/ une fois le serveur lancé. Les modèles sont ceux de
catalogue.json, lus dans ../models/. Aucun n'est chargé au démarrage : le site les charge quand on les coche, et les
décharge quand on les décoche. Tout reste sur l'ordinateur (le serveur n'écoute que sur 127.0.0.1).

Lancement (ou double-clic sur lancer_serveur.bat sous Windows, qui passe par WSL) :

    python serveur.py                    port 8765
    python serveur.py --port 9000
    python serveur.py --processeur       force le calcul sur le processeur

Il faut TensorFlow (classifieurs, débruiteurs) et PyTorch + transformers (BLIP) : ce sont les bibliothèques des
notebooks. Chaque bibliothèque n'est importée qu'au premier modèle qui en a besoin : le serveur démarre en une seconde,
et il fonctionne même si l'une des deux manque (les modèles correspondants sont alors signalés en erreur).

BLIP tourne dans un PROCESSUS À PART, lancé quand on le coche et arrêté quand on le décoche : TensorFlow et PyTorch ne
sont jamais chargés dans le même processus (ensemble sur la carte graphique, ils arrêtaient le serveur sans message au
chargement de BLIP, comme le notebook L3 avant sa séparation en L3_1 et L3_2). Si ce processus s'arrête, le serveur
continue, affiche la raison et décoche BLIP dans le site.

API (utilisée par js/serveur.js) :

    GET  /api/etat                                     modèles du catalogue, état de chargement, appareils
    POST /api/charger      {"tache", "nom"}            charge un modèle (et le prépare), renvoie sa description
    POST /api/decharger    {"tache", "nom"}            libère sa mémoire
    POST /api/classer?nom=&h=&w=&c=                    corps : pixels float32 (h x w x c, 0..255)  -> {raw, probs, ms}
    POST /api/debruiter?nom=&h=&w=                     corps : pixels float32 (h x w x 3, 0..1)    -> pixels float32
    POST /api/legender?nom=&<réglages>                 corps : image (JPEG, PNG...)                -> {legende, ...}
"""
import argparse
import faulthandler
import gc
import io
import json
import math
import mimetypes
import multiprocessing
import os
import pathlib
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ICI = pathlib.Path(__file__).resolve().parent
CATALOGUE = ICI / "catalogue.json"
PORT = 8765
TACHES = ("L1_classification", "L2_debruitage", "L3_captioning")
COTE_TUILE = 1024          # débruitage : au-delà, l'image est traitée par tuiles (mémoire de la carte graphique)

try:  # évite les erreurs d'accents dans certaines consoles Windows
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class ErreurApi(Exception):
    def __init__(self, message, statut=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.statut = statut


# ----------------------------------------------------------------------------------------------
# Bibliothèques, importées à la demande
# ----------------------------------------------------------------------------------------------
FORCER_PROCESSEUR = False
_tf = None
_APPAREIL_TORCH = None       # appareil de BLIP, donné par son processus une fois chargé
_verrou_import = threading.Lock()


def tensorflow():
    global _tf
    with _verrou_import:
        if _tf is None:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            if FORCER_PROCESSEUR:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            t0 = time.time()
            try:
                import tensorflow as tf
            except ImportError as e:
                raise ErreurApi("TensorFlow n'est pas installé dans cet environnement Python "
                                "(pip install tensorflow, ou lancer le serveur depuis l'environnement des notebooks)",
                                HTTPStatus.INTERNAL_SERVER_ERROR) from e
            # Mémoire de la carte graphique réservée à la demande : le processus de BLIP l'utilise aussi
            for gpu in tf.config.list_physical_devices("GPU"):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            _tf = tf
            log(f"TensorFlow {tf.__version__} importé en {time.time() - t0:.1f} s ({appareil_tf()})")
    return _tf


def appareil_tf():
    if _tf is None:
        return None
    gpus = _tf.config.list_physical_devices("GPU")
    if not gpus:
        return "processeur"
    try:
        nom = _tf.config.experimental.get_device_details(gpus[0]).get("device_name")
    except Exception:
        nom = None
    return f"carte graphique ({nom})" if nom else "carte graphique"


def appareil_torch():
    return _APPAREIL_TORCH


def copie_locale_wsl(dossier):
    """Sous WSL, copie un dossier de modèle d'un disque Windows (/mnt/c, /mnt/d...) sur le disque Linux, une seule fois.

    Transformers lit les poids safetensors par projection en mémoire (mmap) ; un disque Windows vu depuis WSL passe par
    un système de fichiers réseau (9P), sur lequel cette lecture est très lente et peut arrêter le processus sans
    message. La copie est gardée dans ~/.cache/leyenda/ (la même que celle de Livrable3/L3_2_BLIP.ipynb)."""
    dossier = pathlib.Path(dossier)
    wsl = sys.platform.startswith("linux") and "microsoft" in os.uname().release.lower()
    if not (wsl and dossier.is_dir() and str(dossier.resolve()).startswith("/mnt/")):
        return dossier
    copie = pathlib.Path.home() / ".cache" / "leyenda" / dossier.name
    for f in sorted(dossier.iterdir()):
        cible = copie / f.name
        if f.is_file() and not (cible.is_file() and cible.stat().st_size == f.stat().st_size
                                and int(cible.stat().st_mtime) == int(f.stat().st_mtime)):
            log(f"copie de {f.name} ({f.stat().st_size / 2**20:.0f} Mo) sur le disque de WSL, une seule fois")
            copie.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, cible)          # copy2 garde la date : le fichier n'est pas recopié la fois suivante
    return copie


# Un seul calcul à la fois : deux images en parallèle sur la même carte graphique ne vont pas plus vite et
# doubleraient la mémoire utilisée.
VERROU_CALCUL = threading.Lock()


# ----------------------------------------------------------------------------------------------
# Modèles Keras (L1, L2)
# ----------------------------------------------------------------------------------------------
IGNOREES = {"InputLayer", "Dropout", "SpatialDropout2D", "GaussianNoise", "GaussianDropout", "RandomFlip",
            "RandomRotation", "RandomZoom", "RandomContrast", "RandomBrightness", "RandomTranslation", "RandomCrop"}


def nom_activation(a):
    nom = a if isinstance(a, str) else getattr(a, "__name__", str(a))
    return {"swish": "silu"}.get(nom, nom)


def couches_keras(model):
    """Description des couches pour le panneau « Architecture » du site (même format que l'export navigateur)."""
    out = []

    def a_plat(m):
        for L in m.layers:
            if hasattr(L, "layers") and L.__class__.__name__ in ("Sequential", "Functional"):
                yield from a_plat(L)
            else:
                yield L

    cfg_entrees = {}
    try:
        for L in model.get_config().get("layers", []):
            noms = []

            def visite(o):
                if isinstance(o, dict):
                    if o.get("class_name") == "__keras_tensor__":
                        noms.append(o["config"]["keras_history"][0])
                        return
                    for v in o.values():
                        visite(v)
                elif isinstance(o, (list, tuple)):
                    for v in o:
                        visite(v)
            visite(L.get("inbound_nodes") or [])
            cfg_entrees[L["name"]] = noms
    except Exception:
        pass

    for L in a_plat(model):
        k = L.__class__.__name__
        if k in IGNOREES:
            continue
        c = L.get_config()
        e = {"type": k, "name": L.name}
        if k == "Conv2D":
            e.update(type="Conv2D", filters=c["filters"], kernel=list(c["kernel_size"]), strides=list(c["strides"]),
                     activation=nom_activation(c["activation"]))
        elif k == "Dense":
            e.update(units=c["units"], activation=nom_activation(c["activation"]))
        elif k in ("MaxPooling2D", "AveragePooling2D"):
            e.update(type="Pool", mode="max" if k.startswith("Max") else "avg", pool=list(c["pool_size"]),
                     strides=list(c["strides"] or c["pool_size"]))
        elif k in ("GlobalAveragePooling2D", "GlobalMaxPooling2D"):
            e.update(type="GlobalPool", mode="max" if "Max" in k else "avg")
        elif k == "UpSampling2D":
            e.update(type="UpSampling", size=[int(v) for v in c["size"]])
        elif k in ("Concatenate", "Add"):
            e["inputs"] = cfg_entrees.get(L.name, [])
        elif k in ("Rescaling", "Normalization", "BatchNormalization"):
            e["type"] = "Affine"
        elif k in ("ReLU", "LeakyReLU", "Softmax", "Activation"):
            act = {"ReLU": "relu", "LeakyReLU": "leaky_relu", "Softmax": "softmax"}.get(k) or nom_activation(c.get("activation"))
            e.update(type="Activation", activation=act)
            if k == "ReLU" and c.get("max_value") is not None:
                e["max_value"] = float(c["max_value"])
        out.append(e)
    return out


def date_keras(chemin):
    try:
        with zipfile.ZipFile(chemin) as z:
            return json.loads(z.read("metadata.json")).get("date_saved")
    except Exception:
        return None


class ModeleKeras:
    def __init__(self, tache, nom, entree, dossier_modeles):
        self.tache, self.nom, self.entree = tache, nom, entree
        self.chemin = dossier_modeles / entree.get("fichier", "")
        self.model = None
        self.meta = None

    def disponible(self):
        return self.chemin.is_file()

    def charger(self):
        tf = tensorflow()
        import keras
        t0 = time.time()
        model = keras.models.load_model(self.chemin, compile=False)
        forme = [None if v is None else int(v) for v in model.input_shape[1:]]
        if len(forme) != 3:
            raise ErreurApi(f"entrée non gérée {forme} : il faut une image H x L x C")
        couches = couches_keras(model)
        meta = {
            "tache": self.tache, "input": forme, "source": self.chemin.name, "date": date_keras(self.chemin),
            "params": int(model.count_params()), "layers": couches, "description": self.entree.get("description", ""),
        }
        if self.tache == "L1_classification":
            n = int(model.output_shape[-1])
            derniere = next((L.get("activation") for L in reversed(couches) if L.get("activation")), "linear")
            if n == 1:
                meta["sortie"], n_classes = ("sigmoid" if derniere == "sigmoid" else "logit"), 2
            else:
                meta["sortie"], n_classes = ("probas" if derniere == "softmax" else "logits"), n
            classes = list(self.entree.get("classes") or [])[:n_classes]
            meta["classes"] = classes + [f"Classe {i}" for i in range(len(classes), n_classes)]
            meta["plage"] = [0, 255]
        else:
            meta["sortie"], meta["plage"] = "image", [0, 1]
            meta["multiple"] = multiple_keras(model)
            meta["champ"] = champ_recepteur(model)
        # Calcul dans un graphe TensorFlow (tf.function) : bien plus rapide que l'appel couche par couche.
        # Les côtés de l'image varient : reduce_retracing évite de recompiler pour chaque taille.
        self.appel = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
        self.model, self.meta = model, meta
        # Premier calcul (compilation, choix des algorithmes de convolution) fait maintenant plutôt qu'à la première image
        h, w, c = [v or 64 for v in forme]
        self.appel(tf.zeros([1, h, w, c]))
        meta["appareil"] = appareil_tf()
        meta["charge_en_s"] = round(time.time() - t0, 2)
        return meta

    def decharger(self):
        # On lâche les références : TensorFlow réutilise ensuite cette mémoire pour les autres modèles.
        # (keras.backend.clear_session() n'est pas appelé : il réinitialiserait aussi les modèles encore chargés.)
        self.model = self.appel = self.meta = None
        gc.collect()

    def classer(self, pixels):
        import numpy as np
        tf = tensorflow()
        h, w, c = self.meta["input"]
        if pixels.shape != (h, w, c):
            raise ErreurApi(f"image de forme {pixels.shape}, attendu {(h, w, c)}")
        raw = np.asarray(self.appel(tf.constant(pixels[None]))[0], dtype=np.float64)
        sortie = self.meta["sortie"]
        if sortie == "probas":
            probs = raw
        elif sortie in ("sigmoid", "logit"):
            p = float(raw[0]) if sortie == "sigmoid" else 1 / (1 + math.exp(-float(raw[0])))
            probs = np.array([1 - p, p])
        else:
            e = np.exp(raw - raw.max())
            probs = e / e.sum()
        return raw.tolist(), probs.tolist()

    def debruiter(self, pixels):
        """Image H x W x 3 dans [0, 1] -> image débruitée de même taille. Les côtés sont complétés par symétrie jusqu'à
        un multiple de meta.multiple ; au-delà de COTE_TUILE pixels, calcul par tuiles qui se chevauchent d'au moins le
        champ récepteur du réseau (résultat identique au calcul d'un seul bloc)."""
        import numpy as np
        tf = tensorflow()
        H, W, _ = pixels.shape
        m = self.meta["multiple"]
        ph, pw = (-H) % m, (-W) % m
        x = np.pad(pixels, ((0, ph), (0, pw), (0, 0)), mode="reflect") if (ph or pw) else pixels
        Hp, Wp = x.shape[:2]
        marge = int(math.ceil(self.meta["champ"] / m) * m)
        cote = max(COTE_TUILE // m * m, 2 * marge + m)
        if Hp <= cote + 2 * marge and Wp <= cote + 2 * marge:
            y = np.asarray(self.appel(tf.constant(x[None]))[0])
        else:
            y = np.empty_like(x)
            fen = cote + 2 * marge              # toutes les fenêtres ont la même taille : un seul graphe compilé
            for y0 in range(0, Hp, cote):
                for x0 in range(0, Wp, cote):
                    y1, x1 = min(y0 + cote, Hp), min(x0 + cote, Wp)
                    fy0 = min(max(0, y0 - marge), max(0, Hp - fen))
                    fx0 = min(max(0, x0 - marge), max(0, Wp - fen))
                    bloc = x[fy0:fy0 + fen, fx0:fx0 + fen]
                    r = np.asarray(self.appel(tf.constant(bloc[None]))[0])
                    y[y0:y1, x0:x1] = r[y0 - fy0:y1 - fy0, x0 - fx0:x1 - fx0]
        return np.clip(y[:H, :W], 0, 1).astype(np.float32)


def multiple_keras(model):
    """Les côtés de l'image doivent être multiples du produit des pas (3 MaxPooling 2x2 -> 8)."""
    m = 1
    for L in model.layers:
        c = L.get_config()
        if L.__class__.__name__ in ("MaxPooling2D", "AveragePooling2D"):
            m *= max(c["strides"] or c["pool_size"])
        elif L.__class__.__name__ == "Conv2D":
            m *= max(c["strides"])
    return m


def champ_recepteur(model):
    """Rayon du champ récepteur, en pixels d'entrée : jusqu'où un pixel de sortie dépend de l'entrée. Les tuiles
    ont une marge au moins aussi large, ce qui rend le calcul par tuiles exact."""
    try:
        cfg = model.get_config()
        entree = cfg["input_layers"][0][0] if isinstance(cfg["input_layers"][0], list) else cfg["input_layers"][0]
        recus = {}
        for L in cfg["layers"]:
            noms = []

            def visite(o):
                if isinstance(o, dict):
                    if o.get("class_name") == "__keras_tensor__":
                        noms.append(o["config"]["keras_history"][0])
                        return
                    for v in o.values():
                        visite(v)
                elif isinstance(o, (list, tuple)):
                    for v in o:
                        visite(v)
            visite(L.get("inbound_nodes") or [])
            recus[L["name"]] = noms
        info = {entree: (1.0, 0.0)}             # nom -> (pas cumulé, rayon)
        restants = [L["name"] for L in cfg["layers"] if L["name"] != entree]
        while restants:
            nom = next(n for n in restants if all(r in info for r in recus[n]))
            restants.remove(nom)
            L = model.get_layer(nom)
            c = L.get_config()
            ins = [info[r] for r in recus[nom]] or [(1.0, 0.0)]
            s, r = ins[0][0], max(x[1] for x in ins)
            k = L.__class__.__name__
            if k == "Conv2D":
                info[nom] = (s * max(c["strides"]), r + (max(c["kernel_size"]) - 1) / 2 * s)
            elif k in ("MaxPooling2D", "AveragePooling2D"):
                info[nom] = (s * max(c["strides"] or c["pool_size"]), r + (max(c["pool_size"]) - 1) / 2 * s)
            elif k == "UpSampling2D":
                info[nom] = (s / max(c["size"]), r)
            else:
                info[nom] = (s, r)
        sortie = cfg["output_layers"][0][0] if isinstance(cfg["output_layers"][0], list) else cfg["output_layers"][0]
        return int(math.ceil(info[sortie][1]))
    except Exception:
        return 128      # valeur prudente si le graphe n'a pas pu être analysé


# ----------------------------------------------------------------------------------------------
# BLIP (L3)
# ----------------------------------------------------------------------------------------------
REGLAGES_BLIP = {   # nom : (type, défaut, minimum, maximum) ; défauts = réglages de Livrable3/L3_2_BLIP.ipynb
    "amorce": (str, "a photograph of", None, 120),
    "num_beams": (int, 5, 1, 10),
    "max_new_tokens": (int, 30, 5, 80),
    "min_new_tokens": (int, 0, 0, 40),
    "no_repeat_ngram_size": (int, 2, 0, 6),
    "repetition_penalty": (float, 1.0, 1.0, 3.0),
    "length_penalty": (float, 1.0, -2.0, 3.0),
    "do_sample": (bool, False, None, None),
    "temperature": (float, 1.0, 0.05, 2.0),
    "top_p": (float, 0.9, 0.05, 1.0),
    "top_k": (int, 50, 0, 500),
}


def lire_reglages(params):
    out = {}
    for nom, (typ, defaut, lo, hi) in REGLAGES_BLIP.items():
        v = params.get(nom)
        if v is None or v == "":
            out[nom] = defaut if nom != "amorce" or v is None else ""
            continue
        try:
            if typ is bool:
                v = str(v).lower() in ("1", "true", "oui", "on")
            elif typ is str:
                v = str(v)[:hi]
            else:
                v = typ(v)
                if not math.isfinite(v):
                    raise ValueError
                v = min(max(v, lo), hi)
        except ValueError:
            raise ErreurApi(f"réglage {nom} invalide : {v!r}")
        out[nom] = v
    out["min_new_tokens"] = min(out["min_new_tokens"], out["max_new_tokens"])
    return out


class MoteurBLIP:
    """BLIP lui-même. Ne sert QUE dans le processus de BLIP (voir _processus_blip) : PyTorch n'est jamais importé par
    le processus principal du serveur, qui fait tourner TensorFlow."""

    def __init__(self, tache, dossier, description):
        self.tache, self.dossier, self.description = tache, pathlib.Path(dossier), description

    def charger(self):
        t0 = time.time()
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as e:
            raise ErreurApi("PyTorch ou transformers n'est pas installé dans cet environnement Python "
                            "(pip install torch transformers)", HTTPStatus.INTERNAL_SERVER_ERROR) from e
        self.torch = torch
        self.device = torch.device("cuda" if (not FORCER_PROCESSEUR and torch.cuda.is_available()) else "cpu")
        appareil = f"carte graphique ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else "processeur"
        log(f"[BLIP] PyTorch {torch.__version__} importé en {time.time() - t0:.1f} s ({appareil})")
        # float16 sur carte graphique : deux fois moins de mémoire, nettement plus rapide, même légende en pratique
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        source = copie_locale_wsl(self.dossier)
        self.processor = BlipProcessor.from_pretrained(source)
        try:
            model = BlipForConditionalGeneration.from_pretrained(source, dtype=self.dtype)
        except TypeError:   # transformers < 4.56 : l'argument s'appelait torch_dtype
            model = BlipForConditionalGeneration.from_pretrained(source, torch_dtype=self.dtype)
        self.model = model.to(self.device).eval()
        c = self.model.config
        compter = lambda m: sum(p.numel() for p in m.parameters())
        poids = self.dossier / "model.safetensors"
        meta = {
            "tache": self.tache, "source": self.dossier.name, "description": self.description,
            "params": int(compter(self.model)), "params_vision": int(compter(self.model.vision_model)),
            "params_texte": int(compter(self.model.text_decoder)),
            "image": int(c.vision_config.image_size), "patch": int(c.vision_config.patch_size),
            "couches_vision": int(c.vision_config.num_hidden_layers), "couches_texte": int(c.text_config.num_hidden_layers),
            "dim": int(c.vision_config.hidden_size), "vocabulaire": int(c.text_config.vocab_size),
            "precision": "float16" if self.dtype == torch.float16 else "float32",
            "date": time.strftime("%Y-%m-%d@%H:%M:%S", time.localtime(poids.stat().st_mtime)) if poids.exists() else None,
            "reglages": {k: v[1] for k, v in REGLAGES_BLIP.items()},
            "bornes": {k: [v[2], v[3]] for k, v in REGLAGES_BLIP.items() if v[0] in (int, float)},
        }
        from PIL import Image
        self.legender_image(Image.new("RGB", (384, 384), (128, 128, 128)), lire_reglages({"max_new_tokens": "5"}))  # préchauffage
        meta["appareil"] = appareil
        meta["charge_en_s"] = round(time.time() - t0, 2)
        return meta

    def legender_image(self, image, r):
        torch = self.torch
        amorce = r["amorce"].strip()
        entrees = self.processor(images=image, text=amorce or None, return_tensors="pt").to(self.device)
        entrees["pixel_values"] = entrees["pixel_values"].to(self.dtype)
        gen = dict(num_beams=r["num_beams"], max_new_tokens=r["max_new_tokens"], min_new_tokens=r["min_new_tokens"],
                   no_repeat_ngram_size=r["no_repeat_ngram_size"], repetition_penalty=r["repetition_penalty"],
                   do_sample=r["do_sample"])
        if r["num_beams"] > 1:
            gen.update(early_stopping=True, length_penalty=r["length_penalty"])
        if r["do_sample"]:
            gen.update(temperature=r["temperature"], top_p=r["top_p"], top_k=r["top_k"])
        with torch.inference_mode():
            sortie = self.model.generate(**entrees, **gen)
        texte = self.processor.decode(sortie[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        suite = texte[len(amorce):].strip() if amorce and texte.lower().startswith(amorce.lower()) else texte
        return {"legende": texte, "suite": suite, "amorce": amorce if suite != texte else "", "jetons": int(sortie.shape[-1])}

    def legender(self, octets, r):
        from PIL import Image, ImageOps
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(octets))).convert("RGB")
        except Exception:
            raise ErreurApi("image illisible")
        res = self.legender_image(image, r)
        res["reglages"] = r
        res["taille"] = list(image.size)
        return res


def _processus_blip(conn, dossier, tache, description, forcer_processeur):
    """Point d'entrée du processus de BLIP : charge le modèle, puis répond aux demandes de légende jusqu'à « stop »
    (ou jusqu'à la fermeture du serveur)."""
    global FORCER_PROCESSEUR
    FORCER_PROCESSEUR = forcer_processeur
    faulthandler.enable()        # en cas de plantage du code C / CUDA : la ligne en cause dans la console du serveur
    try:
        moteur = MoteurBLIP(tache, dossier, description)
        conn.send(("ok", moteur.charger()))
    except ErreurApi as e:
        conn.send(("erreur", str(e)))
        return
    except Exception as e:
        traceback.print_exc()
        conn.send(("erreur", f"{e.__class__.__name__} : {e}"))
        return
    while True:
        try:
            demande = conn.recv()
        except (EOFError, OSError):          # serveur arrêté
            return
        if demande[0] == "stop":
            return
        try:
            conn.send(("ok", moteur.legender(demande[1], demande[2])))
        except ErreurApi as e:
            conn.send(("erreur_api", str(e)))
        except Exception as e:
            traceback.print_exc()
            conn.send(("erreur", f"{e.__class__.__name__} : {e}"))


def explication_arret(code):
    "Raison lisible de l'arrêt d'un processus d'après son code de sortie (négatif : signal reçu)."
    signaux = {-9: "tué par le système, sans doute faute de mémoire vive", -11: "erreur de segmentation dans une bibliothèque C ou CUDA",
               -7: "erreur de bus, lecture d'un fichier projeté en mémoire", -6: "abandon (abort) d'une bibliothèque C ou CUDA"}
    return signaux.get(code, f"code de sortie {code}")


class ModeleBLIP:
    """BLIP vu du serveur : le modèle tourne dans un processus séparé (multiprocessing, méthode « spawn » : un Python
    neuf, sans TensorFlow), lancé par charger() et arrêté par decharger(). Images et légendes passent par un tube."""
    ATTENTE_CHARGEMENT = 900       # s : la première fois sous WSL, copie de 854 Mo puis chargement
    ATTENTE_LEGENDE = 300

    def __init__(self, tache, nom, entree, dossier_modeles):
        self.tache, self.nom, self.entree = tache, nom, entree
        self.dossier = dossier_modeles / entree.get("dossier", nom)
        self.meta = None
        self.processus = self.conn = None

    def disponible(self):
        return (self.dossier / "config.json").is_file()

    def actif(self):
        return self.processus is not None and self.processus.is_alive()

    def charger(self):
        global _APPAREIL_TORCH
        self._arreter()
        ctx = multiprocessing.get_context("spawn")
        self.conn, bout_enfant = ctx.Pipe()
        self.processus = ctx.Process(target=_processus_blip, name="BLIP", daemon=True,
                                     args=(bout_enfant, str(self.dossier), self.tache,
                                           self.entree.get("description", ""), FORCER_PROCESSEUR))
        self.processus.start()
        bout_enfant.close()
        log(f"[BLIP] processus séparé lancé (pid {self.processus.pid})")
        statut, valeur = self._reponse(self.ATTENTE_CHARGEMENT, "le chargement")
        if statut != "ok":
            self._arreter()
            raise ErreurApi(valeur, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.meta = valeur
        _APPAREIL_TORCH = valeur.get("appareil")
        return self.meta

    def _reponse(self, attente_max, etape):
        fin = time.time() + attente_max
        while True:
            try:
                if self.conn.poll(0.5):
                    return self.conn.recv()
            except (EOFError, OSError):
                pass                              # le processus vient de s'arrêter : traité ci-dessous
            if not self.processus.is_alive():
                self.processus.join(1)
                raison = explication_arret(self.processus.exitcode)
                self._arreter()
                log(f"[BLIP] le processus s'est arrêté pendant {etape} ({raison})")
                raise ErreurApi(f"le processus de BLIP s'est arrêté pendant {etape} ({raison}) ; "
                                "le détail est dans la console du serveur", HTTPStatus.INTERNAL_SERVER_ERROR)
            if time.time() > fin:
                self._arreter()
                raise ErreurApi(f"BLIP n'a pas répondu en {attente_max} s ({etape})", HTTPStatus.GATEWAY_TIMEOUT)

    def legender(self, octets, params):
        if not self.actif():
            raison = explication_arret(self.processus.exitcode) if self.processus is not None else None
            self._arreter()
            raise ErreurApi(f"le processus de BLIP s'est arrêté ({raison}) : coche BLIP de nouveau dans le site" if raison
                            else "BLIP n'est plus chargé : coche-le de nouveau dans le site", HTTPStatus.CONFLICT)
        r = lire_reglages(params)
        self.conn.send(("legender", octets, r))
        statut, valeur = self._reponse(self.ATTENTE_LEGENDE, "le calcul d'une légende")
        if statut == "erreur_api":
            raise ErreurApi(valeur)
        if statut != "ok":
            raise ErreurApi(valeur, HTTPStatus.INTERNAL_SERVER_ERROR)
        return valeur

    def decharger(self):
        self._arreter()
        self.meta = None

    def _arreter(self):
        "Arrête le processus de BLIP (s'il tourne) : toute sa mémoire, vive et carte graphique, est rendue."
        global _APPAREIL_TORCH
        if self.conn is not None:
            try:
                self.conn.send(("stop",))
            except (OSError, ValueError, BrokenPipeError):
                pass
        if self.processus is not None:
            self.processus.join(10)
            if self.processus.is_alive():
                self.processus.terminate()
                self.processus.join(5)
        if self.conn is not None:
            self.conn.close()
        self.processus = self.conn = None
        _APPAREIL_TORCH = None


# ----------------------------------------------------------------------------------------------
# Catalogue et registre des modèles
# ----------------------------------------------------------------------------------------------
class Registre:
    def __init__(self):
        self.modeles = {}            # (tache, nom) -> objet modèle
        self.etat = {}               # (tache, nom) -> {"charge", "en_cours", "erreur"}
        self.verrou = threading.Lock()
        self.relire()

    def relire(self):
        with open(CATALOGUE, encoding="utf-8") as f:
            cat = json.load(f)
        dossier = (ICI / cat.get("dossier_modeles", "../models")).resolve()
        self.dossier_modeles = dossier
        vus = set()
        for tache in TACHES:
            for nom, entree in (cat.get(tache) or {}).items():
                cle = (tache, nom)
                vus.add(cle)
                ancien = self.modeles.get(cle)
                if ancien is not None and ancien.entree == entree:
                    continue
                cls = ModeleBLIP if tache == "L3_captioning" else ModeleKeras
                self.modeles[cle] = cls(tache, nom, entree, dossier)
                self.etat[cle] = {"charge": False, "en_cours": False, "erreur": None}
        for cle in list(self.modeles):
            if cle not in vus and not self.etat[cle]["charge"]:
                del self.modeles[cle], self.etat[cle]

    def trouver(self, tache, nom):
        m = self.modeles.get((tache, nom))
        if m is None:
            raise ErreurApi(f"modèle inconnu : {tache}/{nom}", HTTPStatus.NOT_FOUND)
        return m

    def resume(self):
        self.relire()
        modeles = {t: [] for t in TACHES}
        for (tache, nom), m in self.modeles.items():
            e = self.etat[(tache, nom)]
            modeles[tache].append({"nom": nom, "description": m.entree.get("description", ""), "disponible": m.disponible(),
                                   "fichier": m.entree.get("fichier") or m.entree.get("dossier", nom),
                                   "charge": e["charge"], "en_cours": e["en_cours"], "erreur": e["erreur"],
                                   "meta": m.meta if e["charge"] else None})
        return {"serveur": "La Ruche", "version": 1, "dossier_modeles": str(self.dossier_modeles),
                "appareils": {"tensorflow": appareil_tf(), "pytorch": appareil_torch()},
                "processeur_force": FORCER_PROCESSEUR, "modeles": modeles}

    def charger(self, tache, nom):
        m = self.trouver(tache, nom)
        e = self.etat[(tache, nom)]
        if e["charge"]:
            return m.meta
        if not m.disponible():
            raise ErreurApi(f"{m.entree.get('fichier') or m.entree.get('dossier', nom)} introuvable dans {self.dossier_modeles}",
                            HTTPStatus.NOT_FOUND)
        e.update(en_cours=True, erreur=None)
        try:
            with VERROU_CALCUL:
                if e["charge"]:          # chargé entre-temps par une autre requête (deux onglets ouverts...)
                    return m.meta
                log(f"chargement de {tache}/{nom}...")
                meta = m.charger()
            e["charge"] = True
            log(f"{tache}/{nom} prêt en {meta['charge_en_s']} s sur {meta['appareil']}")
            return meta
        except Exception as ex:
            e["erreur"] = str(ex)
            m.decharger()
            raise
        finally:
            e["en_cours"] = False

    def decharger(self, tache, nom):
        m = self.trouver(tache, nom)
        with VERROU_CALCUL:
            m.decharger()
        self.etat[(tache, nom)].update(charge=False, erreur=None)
        log(f"{tache}/{nom} déchargé")

    def modele_pret(self, tache, nom):
        m = self.trouver(tache, nom)
        if not self.etat[(tache, nom)]["charge"]:
            raise ErreurApi(f"{nom} n'est pas chargé : coche-le dans le site", HTTPStatus.CONFLICT)
        return m


REGISTRE = None


# ----------------------------------------------------------------------------------------------
# Serveur HTTP
# ----------------------------------------------------------------------------------------------
class Gestionnaire(BaseHTTPRequestHandler):
    server_version = "LaRuche/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):   # pas de ligne par requête dans la console
        pass

    # --- réponses ---
    def entetes_communs(self):
        # Le site peut aussi être ouvert directement (file://) : il appelle alors le serveur depuis une autre origine
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Expose-Headers", "X-Duree-Ms")

    def repondre(self, statut, corps, type_mime, entetes=None):
        self.send_response(statut)
        self.entetes_communs()
        self.send_header("Content-Type", type_mime)
        self.send_header("Content-Length", str(len(corps)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (entetes or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(corps)

    def json(self, obj, statut=HTTPStatus.OK):
        self.repondre(statut, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def corps(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 512 * 2**20:
            raise ErreurApi("requête trop grosse", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        return self.rfile.read(n) if n else b""

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.entetes_communs()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        try:
            if url.path == "/api/etat":
                return self.json(REGISTRE.resume())
            if url.path.startswith("/api/"):
                raise ErreurApi("route inconnue", HTTPStatus.NOT_FOUND)
            self.fichier_statique(url.path)
        except ErreurApi as e:
            self.json({"erreur": str(e)}, e.statut)

    def do_POST(self):
        url = urllib.parse.urlsplit(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query, keep_blank_values=True).items()}
        try:
            import numpy as np
            if url.path in ("/api/charger", "/api/decharger"):
                d = json.loads(self.corps() or b"{}")
                if url.path == "/api/charger":
                    return self.json({"meta": REGISTRE.charger(d.get("tache"), d.get("nom"))})
                REGISTRE.decharger(d.get("tache"), d.get("nom"))
                return self.json({"ok": True})
            if url.path == "/api/classer":
                m = REGISTRE.modele_pret("L1_classification", q.get("nom"))
                h, w, c = int(q["h"]), int(q["w"]), int(q.get("c", 3))
                x = np.frombuffer(self.corps(), dtype="<f4")
                if x.size != h * w * c:
                    raise ErreurApi(f"{x.size} valeurs reçues pour une image {h} x {w} x {c}")
                t0 = time.perf_counter()
                with VERROU_CALCUL:
                    raw, probs = m.classer(x.reshape(h, w, c))
                return self.json({"raw": raw, "probs": probs, "ms": (time.perf_counter() - t0) * 1000})
            if url.path == "/api/debruiter":
                m = REGISTRE.modele_pret("L2_debruitage", q.get("nom"))
                h, w = int(q["h"]), int(q["w"])
                x = np.frombuffer(self.corps(), dtype="<f4")
                if x.size != h * w * 3:
                    raise ErreurApi(f"{x.size} valeurs reçues pour une image {h} x {w} x 3")
                if min(h, w) < 16:
                    raise ErreurApi("image trop petite (16 pixels minimum)")
                t0 = time.perf_counter()
                with VERROU_CALCUL:
                    y = m.debruiter(x.reshape(h, w, 3))
                ms = (time.perf_counter() - t0) * 1000
                return self.repondre(HTTPStatus.OK, y.astype("<f4").tobytes(), "application/octet-stream", {"X-Duree-Ms": f"{ms:.1f}"})
            if url.path == "/api/legender":
                m = REGISTRE.modele_pret("L3_captioning", q.get("nom"))
                octets = self.corps()
                t0 = time.perf_counter()
                try:
                    with VERROU_CALCUL:
                        res = m.legender(octets, q)
                except ErreurApi as e:
                    if not m.actif():         # processus de BLIP arrêté : le site le verra décoché, avec la raison
                        REGISTRE.etat[("L3_captioning", q.get("nom"))].update(charge=False, erreur=str(e))
                    raise
                res["ms"] = (time.perf_counter() - t0) * 1000
                return self.json(res)
            raise ErreurApi("route inconnue", HTTPStatus.NOT_FOUND)
        except ErreurApi as e:
            self.json({"erreur": str(e)}, e.statut)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.json({"erreur": f"requête invalide ({e})"}, HTTPStatus.BAD_REQUEST)
        except Exception as e:  # erreur du modèle : on la renvoie au site et on l'affiche dans la console
            traceback.print_exc()
            self.json({"erreur": f"{e.__class__.__name__} : {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def fichier_statique(self, chemin):
        rel = urllib.parse.unquote(chemin).lstrip("/") or "index.html"
        cible = (ICI / rel).resolve()
        if ICI not in cible.parents and cible != ICI:     # pas de sortie du dossier du site
            raise ErreurApi("accès refusé", HTTPStatus.FORBIDDEN)
        if cible.is_dir():
            cible = cible / "index.html"
        if not cible.is_file() or cible.suffix in (".py", ".bat", ".sh"):
            raise ErreurApi("fichier introuvable", HTTPStatus.NOT_FOUND)
        type_mime = mimetypes.guess_type(cible.name)[0] or "application/octet-stream"
        if type_mime.startswith("text/") or type_mime in ("application/javascript", "application/json"):
            type_mime += "; charset=utf-8"
        self.repondre(HTTPStatus.OK, cible.read_bytes(), type_mime)


def main():
    global REGISTRE, FORCER_PROCESSEUR
    ap = argparse.ArgumentParser(description="Serveur local de La Ruche : modèles du projet sur la carte graphique.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("RUCHE_PORT", PORT)))
    ap.add_argument("--hote", default=os.environ.get("RUCHE_HOTE", "127.0.0.1"),
                    help="adresse d'écoute (127.0.0.1 : seulement cet ordinateur)")
    ap.add_argument("--processeur", action="store_true", help="calcul sur le processeur même si une carte graphique est là")
    args = ap.parse_args()
    FORCER_PROCESSEUR = args.processeur
    faulthandler.enable()        # en cas de plantage du code C / CUDA : la ligne en cause dans cette console
    mimetypes.add_type("application/javascript", ".js")
    REGISTRE = Registre()
    try:
        serveur = ThreadingHTTPServer((args.hote, args.port), Gestionnaire)
    except OSError as e:
        sys.exit(f"Port {args.port} indisponible ({e}). Le serveur tourne peut-être déjà : ouvre http://localhost:{args.port}/")
    serveur.daemon_threads = True
    n = sum(len(v) for v in REGISTRE.resume()["modeles"].values())
    log(f"La Ruche : {n} modèles au catalogue ({REGISTRE.dossier_modeles}), aucun chargé tant qu'on ne le coche pas")
    log(f"Site : http://localhost:{args.port}/   (Ctrl+C pour arrêter)")
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        log("arrêt du serveur")
    finally:
        for m in REGISTRE.modeles.values():
            if isinstance(m, ModeleBLIP):
                m.decharger()


if __name__ == "__main__":
    main()
