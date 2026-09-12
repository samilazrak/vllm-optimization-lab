#!/usr/bin/env bash
# Étapes 1 et 2 du lab : inspecter le matériel, installer vLLM, récupérer le trafic.
# À lancer une seule fois sur le pod GPU.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/raw data figures

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
if python -c "import vllm" 2>/dev/null; then
  echo "vLLM déjà présent : $(python -c 'import vllm; print(vllm.__version__)')"
else
  pip install --upgrade pip
  pip install vllm
fi
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)" \
  | tee results/raw/versions.txt

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
# Les noms de flags du bench bougent d'une version à l'autre : on garde une trace
# de ceux que cette version expose, notamment pour prefix_repetition.
vllm bench serve --help > results/raw/bench-serve-help.txt 2>&1 || true
echo "Aide sauvegardée dans results/raw/bench-serve-help.txt"
grep -E -- "--prefix-repetition|--dataset-name" results/raw/bench-serve-help.txt | head -20 || true

echo
echo "Setup terminé. Prochaine étape : make sweep (ou scripts/01-serve.sh base)"
