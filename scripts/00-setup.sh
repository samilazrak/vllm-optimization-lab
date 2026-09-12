#!/usr/bin/env bash
# Étapes 1 et 2 du lab : qualifier l'hôte, inspecter le matériel, installer vLLM,
# récupérer le trafic. À lancer une seule fois sur le pod GPU.
#
# Les trois premières sections sont des contrôles de qualification, ajoutés après
# avoir perdu une heure sur trois pods successifs : un hôte à 478 kB/s, un pip
# sans version qui remonte jusqu'à vLLM 0.7.2, et un driver trop vieux pour les
# wheels installées. Chacun se détecte en quelques secondes et coûte cher à
# découvrir plus tard.
#
# SKIP_HOST_CHECKS=1 contourne les contrôles bloquants.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw data figures

VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
# Version de CUDA que visent les wheels de cette version de vLLM. Le driver de
# l'hôte doit la supporter : entre 12.x et 13.x le saut est majeur, la
# compatibilité mineure ne joue pas.
REQUIRED_CUDA="${REQUIRED_CUDA:-13.0}"
# En dessous, les 28 Go de poids du modèle prennent des heures.
MIN_SPEED_MB="${MIN_SPEED_MB:-5}"

echo "=== Qualification 1 : le débit vers Hugging Face ============="
# 50 Mo depuis un fichier public et stable. C'est le chemin qui compte : les
# poids du modèle viennent de là.
SPEED=$(curl -s -o /dev/null -w '%{speed_download}' -L -r 0-52428800 \
  https://huggingface.co/gpt2/resolve/main/model.safetensors || echo 0)
SPEED_MB=$(awk -v s="$SPEED" 'BEGIN { printf "%.1f", s / 1048576 }')
# Parenthèses obligatoires : sans elles, awk lit le « > » du ternaire comme une
# redirection de sortie.
WEIGHTS_MIN=$(awk -v s="$SPEED" 'BEGIN { printf "%.0f", (s > 0 ? 28672 * 1048576 / s / 60 : 9999) }')
echo "Débit mesuré : ${SPEED_MB} Mo/s  (28 Go de poids ≈ ${WEIGHTS_MIN} min)"

if awk -v s="$SPEED_MB" -v m="$MIN_SPEED_MB" 'BEGIN { exit !(s < m) }'; then
  echo "ÉCHEC : en dessous de ${MIN_SPEED_MB} Mo/s, cet hôte n'est pas exploitable." >&2
  echo "Détruis le pod et reprends-en un autre, c'est moins cher que d'attendre." >&2
  [ "${SKIP_HOST_CHECKS:-0}" = "1" ] || exit 1
elif awk -v s="$SPEED_MB" 'BEGIN { exit !(s < 20) }'; then
  echo "ATTENTION : hôte lent. Utilisable, mais tu paieras l'attente."
fi

echo
echo "=== Qualification 2 : le driver =============================="
DRIVER_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
echo "CUDA supporté par le driver : ${DRIVER_CUDA:-inconnu}  |  requis par vLLM ${VLLM_VERSION} : ${REQUIRED_CUDA}"
if [ -n "$DRIVER_CUDA" ] && [ "$(printf '%s\n%s\n' "$REQUIRED_CUDA" "$DRIVER_CUDA" | sort -V | head -1)" != "$REQUIRED_CUDA" ]; then
  echo "ÉCHEC : driver trop ancien pour les wheels de vLLM ${VLLM_VERSION}." >&2
  echo "Sur RunPod, filtre les hôtes sur « Available CUDA versions » >= ${REQUIRED_CUDA}." >&2
  [ "${SKIP_HOST_CHECKS:-0}" = "1" ] || exit 1
fi

echo
echo "=== Étape 1 : le matériel ==================================="
nvidia-smi | tee results/raw/nvidia-smi-idle.txt
echo
nvidia-smi --query-gpu=name,compute_cap,memory.free,memory.used,memory.total,pstate \
  --format=csv | tee results/raw/gpu-specs.csv
echo
echo "Rappel de lecture :"
echo "  pstate P8 = GPU inactif   |  P0/P1 = pleine vitesse"
echo "  Référence du livre : NVIDIA L40S, cc 8.9, 46068 MiB total"

echo
echo "=== Installation de vLLM ===================================="
# Version épinglée : un `pip install vllm` nu entre en conflit avec le torch
# préinstallé de l'image et fait reculer le résolveur de version en version,
# jusqu'à des vLLM de 2025 où `vllm bench serve` n'existe pas encore.
if python -c "import vllm" 2>/dev/null; then
  echo "vLLM déjà présent : $(python -c 'import vllm; print(vllm.__version__)')"
else
  pip install --upgrade pip
  pip install "vllm==${VLLM_VERSION}"
fi
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)" \
  | tee results/raw/versions.txt

echo
echo "=== Qualification 3 : CUDA vu depuis torch =================="
# Le contrôle qui fait autorité : le driver et la build de torch doivent
# s'entendre. Un échec ici rend tout le reste inutile.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ÉCHEC : torch ne voit pas le GPU." >&2
  python -c "import torch; print(torch.__version__)" >&2 || true
  echo "Le driver de l'hôte est trop ancien pour cette build. Change de pod," >&2
  echo "ou réinstalle avec : uv pip install --system --break-system-packages \\" >&2
  echo "  vllm==${VLLM_VERSION} --torch-backend=auto" >&2
  [ "${SKIP_HOST_CHECKS:-0}" = "1" ] || exit 1
fi
echo "torch voit le GPU : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

echo
echo "=== Étape 2 : le trafic de benchmark ========================"
SHAREGPT="data/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ -f "$SHAREGPT" ]; then
  echo "ShareGPT déjà téléchargé ($(du -h "$SHAREGPT" | cut -f1))"
else
  echo "Téléchargement de ShareGPT (~650 Mo)..."
  wget -q --show-progress -O "$SHAREGPT" \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
fi

echo
echo "=== Les options réellement disponibles dans ce vLLM ========="
# --help=all et pas --help : depuis vLLM 0.29, l'aide courte ne liste que les
# groupes de configuration, plus les flags. Un grep sur --help ne remonte rien
# et laisse croire à tort que le dataset prefix_repetition a disparu.
vllm bench serve --help=all > results/raw/bench-serve-help.txt 2>&1 || true
echo "Aide sauvegardée dans results/raw/bench-serve-help.txt"
# Le résultat passe par une variable plutôt que par le statut d'un pipeline :
# avec « | head », un SIGPIPE suffirait à déclencher une fausse alerte.
FLAGS=$(grep -E -- "--prefix-repetition|--dataset-name" results/raw/bench-serve-help.txt | head -20 || true)
if [ -n "$FLAGS" ]; then
  echo "$FLAGS"
else
  echo "ATTENTION : les flags de dataset attendus par 02-bench.sh sont absents." >&2
  echo "Vérifie results/raw/bench-serve-help.txt avant de lancer le sweep." >&2
fi

echo
echo "Setup terminé. Prochaine étape : make sweep (ou scripts/01-serve.sh base)"
