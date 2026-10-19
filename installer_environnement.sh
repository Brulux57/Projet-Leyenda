#!/usr/bin/env bash
# Projet Leyenda : installe l'environnement Python des notebooks et du site (WSL2 ou Linux, carte NVIDIA).
#
#   bash installer_environnement.sh                       environnement créé (ou complété) dans ~/venvs/leyenda
#   bash installer_environnement.sh --supprimer-ancien    idem, puis supprime l'ancien ~/venvs/tf-gpu s'il existe
#   LEYENDA_VENV=/autre/dossier bash installer_environnement.sh
#   Depuis Windows : double-clic sur installer_environnement.bat (lance ce script dans WSL).
#
# 1. crée l'environnement virtuel ;
# 2. y installe requirements.txt : TensorFlow 2.21 et PyTorch 2.13, tous deux sur CUDA 12.9 (mêmes bibliothèques
#    nvidia-*-cu12), puis leyenda_cuda.py, qui permet à TensorFlow de trouver cuSOLVER (voir ce fichier) ;
# 3. enregistre le noyau Jupyter « Leyenda (TensorFlow + PyTorch GPU) », à choisir dans VS Code ou Jupyter pour
#    TOUS les notebooks du projet (L1, L2, L3) ;
# 4. vérifie que TensorFlow PUIS PyTorch calculent sur la carte graphique dans le même processus ;
# 5. avec --supprimer-ancien, et seulement si la vérification a réussi : supprime ~/venvs/tf-gpu et son noyau Jupyter.
# Tout est recopié dans installation.log, à côté de ce script. Environ 5 Go à télécharger la première fois.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PROJET="$PWD"
VENV="${LEYENDA_VENV:-$HOME/venvs/leyenda}"
NOYAU="leyenda"
NOM_NOYAU="Leyenda (TensorFlow + PyTorch GPU)"
ANCIEN="$HOME/venvs/tf-gpu"
SUPPRIMER_ANCIEN=0
for a in "$@"; do
  case "$a" in
    --supprimer-ancien) SUPPRIMER_ANCIEN=1 ;;
    *) echo "option inconnue : $a" ; exit 2 ;;
  esac
done
exec > >(tee "$PROJET/installation.log") 2>&1

echec() { echo; echo "RESULTAT : ECHEC - $*"; exit 1; }
etape() { echo; echo "=== $(date +%H:%M:%S)  $* ==="; }

etape "Projet $PROJET ; environnement $VENV"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

libre=$(df -Pk "$HOME" | awk 'NR==2 {print int($4 / 1048576)}')
echo "Espace libre : ${libre} Go"
[ "$libre" -ge 12 ] || echo "ATTENTION : l'installation occupe environ 8 Go, plus 5 Go de cache de pip."

etape "1/5 Environnement virtuel"
if [ -x "$VENV/bin/python" ]; then
  echo "$VENV existe déjà : il est complété."
else
  [ -e "$VENV" ] && echec "$VENV existe mais ne contient pas d'environnement Python : le déplacer ou choisir LEYENDA_VENV"
  # Python de base : LEYENDA_PYTHON, sinon celui de l'ancien environnement s'il existe, sinon python3.12 ou python3
  PY_BASE="${LEYENDA_PYTHON:-}"
  if [ -z "$PY_BASE" ] && [ -x "$ANCIEN/bin/python" ]; then
    PY_BASE="$("$ANCIEN/bin/python" -c 'import sys; print(sys._base_executable)' 2>/dev/null)"
  fi
  [ -n "$PY_BASE" ] && [ -x "$PY_BASE" ] || PY_BASE="$(command -v python3.12 || command -v python3 || true)"
  [ -n "$PY_BASE" ] || echec "python3 introuvable (sudo apt install python3 python3-venv)"
  "$PY_BASE" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' \
    || echec "$("$PY_BASE" -V 2>&1) : TensorFlow 2.21 et PyTorch 2.13 demandent Python 3.10 à 3.13"
  echo "Python de base : $PY_BASE ($("$PY_BASE" -V 2>&1))"
  mkdir -p "$(dirname "$VENV")"
  if ! "$PY_BASE" -m venv "$VENV"; then
    # Module ensurepip absent (paquet python3-venv non installé) : environnement sans pip, puis pip installé avec
    # celui de l'ancien environnement.
    rm -rf "$VENV"
    "$PY_BASE" -m venv --without-pip "$VENV" || echec "création de $VENV impossible"
    [ -x "$ANCIEN/bin/pip" ] || echec "module venv incomplet : sudo apt install python3-venv, puis relancer"
    "$ANCIEN/bin/pip" --python "$VENV/bin/python" install pip || echec "installation de pip dans $VENV"
  fi
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip || echec "mise à jour de pip"

etape "2/5 Bibliothèques de requirements.txt (TensorFlow, PyTorch CUDA 12.9, Transformers...)"
"$PY" -m pip install -r requirements.txt || echec "installation des bibliothèques (message de pip ci-dessus)"
# Aucune bibliothèque CUDA 13 ne doit rester : les paquets nvidia-* sans suffixe -cu12 sont ceux de CUDA 13.
cu13=$("$PY" -m pip list --format=freeze 2>/dev/null | grep -i -E '^nvidia-' | grep -v -i -E -- '-cu12==')
[ -z "$cu13" ] || echec "bibliothèques CUDA 13 présentes dans $VENV : $(echo $cu13) ; les désinstaller (pip uninstall) puis relancer"
"$PY" -m pip list 2>/dev/null | grep -i -E '^(tensorflow|keras|torch|torchvision|transformers|numpy|nvidia-cudnn-cu12|nvidia-nccl-cu12) '
# TensorFlow 2.21 charge seul les bibliothèques nvidia-*-cu12 de l'environnement, sauf libcusolver.so.11 : il abandonne
# alors la carte graphique (« Cannot dlopen some GPU libraries »). leyenda_cuda.py, chargé au démarrage de Python par
# leyenda_cuda.pth, la charge juste avant le premier « import tensorflow » : cela marche pour tous les notebooks et
# scripts, quelle que soit la façon dont ils sont lancés (VS Code n'applique pas toujours les variables d'un noyau).
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NV_LIBS="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | paste -sd:)"
[ -n "$NV_LIBS" ] || echec "aucun dossier nvidia/*/lib dans $SITE"
cp "$PROJET/leyenda_cuda.py" "$SITE/leyenda_cuda.py" || echec "copie de leyenda_cuda.py dans $SITE"
echo "import leyenda_cuda" > "$SITE/leyenda_cuda.pth"
echo "leyenda_cuda.py installé dans $SITE (cuSOLVER préchargé avant TensorFlow)"

etape "3/5 Noyau Jupyter « $NOM_NOYAU »"
# CUDA_CACHE_MAXSIZE : TensorFlow compile ses noyaux CUDA à la volée pour les cartes récentes (RTX 50xx) ; un cache de
# 4 Go (256 Mo par défaut) évite de les recompiler à chaque lancement du notebook.
"$PY" -m ipykernel install --user --name "$NOYAU" --display-name "$NOM_NOYAU" \
  --env LD_LIBRARY_PATH "$NV_LIBS" --env CUDA_CACHE_MAXSIZE 4294967296 \
  || echec "enregistrement du noyau Jupyter"

etape "4/5 Vérification : TensorFlow puis PyTorch, dans le même processus, sur la carte graphique"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
# Sans LD_LIBRARY_PATH, comme un notebook lancé par VS Code : c'est leyenda_cuda.py qui doit suffire
env -u LD_LIBRARY_PATH "$PY" -u - <<'EOF'
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ok = True

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)   # comme le notebook : laisse de la place à PyTorch
info = tf.sysconfig.get_build_info()
print(f"TensorFlow {tf.__version__} (CUDA {info.get('cuda_version')}, cuDNN {info.get('cudnn_version')}) : "
      f"{len(gpus)} carte(s) graphique(s)")
if gpus:
    d = tf.config.experimental.get_device_details(gpus[0])
    print(f"  {d.get('device_name')}, capacité de calcul {d.get('compute_capability')}")
    t0 = time.time()
    with tf.device("/GPU:0"):
        x = tf.random.normal((8, 224, 224, 3))
        y = tf.keras.layers.Conv2D(32, 3)(x)                 # convolution : cuDNN
        z = tf.reduce_sum(tf.matmul(x[:, :, :, 0], x[:, :, :, 0], transpose_b=True))   # cuBLAS
        _ = float(z) + float(tf.reduce_sum(y))
    print(f"  calcul de contrôle sur {y.device} : {time.time() - t0:.1f} s "
          "(la première fois, compilation des noyaux pour la carte : jusqu'à quelques minutes)")
else:
    ok = False
    print("  ERREUR : TensorFlow ne voit pas la carte graphique")

import torch

dispo = torch.cuda.is_available()
print(f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}) : "
      f"carte graphique {'oui' if dispo else 'NON'}")
if dispo:
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    gere = arch in torch.cuda.get_arch_list()
    print(f"  {torch.cuda.get_device_name(0)} ({arch}) : {'gérée' if gere else 'NON gérée'} par cette version "
          f"({' '.join(torch.cuda.get_arch_list())})")
    ok = ok and gere
    t0 = time.time()
    x = torch.randn(8, 3, 224, 224, device="cuda", dtype=torch.float16)
    w = torch.randn(32, 3, 3, 3, device="cuda", dtype=torch.float16)
    s = float(torch.nn.functional.conv2d(x, w).float().sum()) + float((x[:, 0] @ x[:, 0].transpose(1, 2)).float().sum())
    torch.cuda.synchronize()
    libre, total = torch.cuda.mem_get_info()
    print(f"  calcul de contrôle en float16 : {time.time() - t0:.1f} s ; mémoire de la carte libre : "
          f"{libre / 2**30:.1f} Go sur {total / 2**30:.1f} Go")
else:
    ok = False

print()
print("RESULTAT : OK - TensorFlow et PyTorch utilisent la carte graphique" if ok else
      "RESULTAT : ECHEC - voir les messages ci-dessus")
sys.exit(0 if ok else 3)
EOF
statut=$?
# 3 : échec déjà expliqué ci-dessus ; autre code : le processus s'est arrêté avant la fin (erreur Python, plantage).
[ $statut -eq 0 ] || [ $statut -eq 3 ] || echo "RESULTAT : ECHEC - vérification interrompue (code $statut)"
[ $statut -eq 0 ] || exit $statut

etape "5/5 Ancien environnement ($ANCIEN)"
if [ ! -e "$ANCIEN" ]; then
  echo "absent : rien à faire"
elif [ "$SUPPRIMER_ANCIEN" = 1 ]; then
  # Noyaux Jupyter qui lancent le Python de l'ancien environnement, puis l'environnement lui-même
  "$PY" - "$ANCIEN" <<'EOF'
import json, pathlib, shutil, sys
ancien = sys.argv[1]
for k in sorted((pathlib.Path.home() / ".local/share/jupyter/kernels").glob("*/kernel.json")):
    try:
        argv = json.loads(k.read_text(encoding="utf-8")).get("argv") or [""]
    except (OSError, ValueError):
        continue
    if argv[0].startswith(ancien + "/"):
        shutil.rmtree(k.parent)
        print(f"noyau Jupyter supprimé : {k.parent.name}")
EOF
  rm -rf "$ANCIEN" && echo "$ANCIEN supprimé : tous les notebooks utilisent $VENV"
else
  echo "toujours présent ; les notebooks et le site n'en ont plus besoin. Pour le supprimer :"
  echo "  bash $PROJET/installer_environnement.sh --supprimer-ancien"
fi

echo
echo "Notebooks (L1, L2, L3) : dans VS Code (fenêtre WSL), « Select Kernel » > « Jupyter Kernel... » > « $NOM_NOYAU »."
echo "Site : Test_website/lancer_serveur.bat (ou .sh) utilise $VENV."
exit 0
