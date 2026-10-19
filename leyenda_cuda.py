"""Projet Leyenda : précharge cuSOLVER avant TensorFlow. Installé dans ~/venvs/leyenda par installer_environnement.sh.

TensorFlow 2.21 trouve seul les bibliothèques nvidia-*-cu12 de l'environnement, sauf libcusolver.so.11 : sans elle, il
abandonne la carte graphique (« Cannot dlopen some GPU libraries », aucune carte graphique détectée). Ce module est
chargé au démarrage de Python par le fichier leyenda_cuda.pth du même dossier. Il attend le premier
« import tensorflow » et charge juste avant cuSOLVER et les bibliothèques dont elle dépend : TensorFlow la trouve alors
déjà chargée. Cela vaut pour tous les notebooks (L1, L2, L3), le serveur du site et tout autre script, quelle que soit
la façon dont ils sont lancés. Un programme qui n'importe pas TensorFlow ne charge rien.
"""
import sys

BIBLIOTHEQUES = (("nvjitlink", "libnvJitLink.so.12"), ("cublas", "libcublasLt.so.12"), ("cublas", "libcublas.so.12"),
                 ("cusparse", "libcusparse.so.12"), ("cusolver", "libcusolver.so.11"))


def precharger():
    "Charge cuSOLVER et ses dépendances depuis site-packages/nvidia/ (paquets nvidia-*-cu12)."
    import ctypes
    import importlib.util
    import pathlib

    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for sous_dossier, fichier in BIBLIOTHEQUES:
        for racine in spec.submodule_search_locations:
            chemin = pathlib.Path(racine) / sous_dossier / "lib" / fichier
            if chemin.is_file():
                try:
                    ctypes.CDLL(str(chemin), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


class AvantTensorFlow:
    "Chercheur d'imports qui ne trouve jamais rien lui-même : il précharge cuSOLVER au premier import de tensorflow."
    fait = False

    def find_spec(self, nom, chemin=None, cible=None):
        if nom == "tensorflow" and not self.fait:
            self.fait = True
            try:
                precharger()
            except Exception:
                pass
        return None


if sys.platform.startswith("linux") and not any(isinstance(f, AvantTensorFlow) for f in sys.meta_path):
    sys.meta_path.insert(0, AvantTensorFlow())
