#!/usr/bin/env bash
# La Ruche : lance le serveur local du site (WSL ou Linux).
#   ./lancer_serveur.sh              puis ouvrir http://localhost:8765/
#   ./lancer_serveur.sh --processeur sans carte graphique
# Python utilisé : $RUCHE_PYTHON s'il est défini, sinon l'environnement créé par installer_environnement.sh
# (~/venvs/leyenda : TensorFlow et PyTorch sur la carte graphique), sinon l'ancien ~/venvs/tf-gpu, sinon .venv du
# projet, sinon python3.
cd "$(dirname "$0")" || exit 1
PY="${RUCHE_PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$HOME/venvs/leyenda/bin/python" "$HOME/venvs/tf-gpu/bin/python" "../.venv/bin/python" "../../.venv/bin/python"; do
    if [ -x "$c" ]; then PY="$c"; break; fi
  done
fi
PY="${PY:-python3}"
# TensorFlow 2.21 ne trouve pas seul libcusolver.so.11 parmi les bibliothèques nvidia-*-cu12 de l'environnement et
# abandonne alors la carte graphique : on lui donne les dossiers nvidia/*/lib de l'environnement choisi.
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
NV_LIBS="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | paste -sd:)"
[ -n "$NV_LIBS" ] && export LD_LIBRARY_PATH="$NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# TensorFlow compile ses noyaux CUDA à la volée pour les cartes récentes (RTX 50xx) : un cache assez grand
# évite de les recompiler à chaque lancement (le cache par défaut, 256 Mo, est trop petit).
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
echo "=== La Ruche : serveur local (Python : $PY) ==="
exec "$PY" serveur.py "$@"
