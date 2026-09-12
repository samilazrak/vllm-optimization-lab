#!/usr/bin/env bash
# Étapes 5 et 6 : envoyer le trafic sur un serveur déjà démarré et sauvegarder
# les métriques en JSON.
#
#   ./scripts/02-bench.sh <config> <sharegpt|prefix>
#
# <config> ne sert qu'à nommer le fichier de résultat : c'est la clé de jointure
# avec le log de démarrage correspondant, dans collect.py.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:?usage: 02-bench.sh <config> <sharegpt|prefix>}"
DATASET="${2:?usage: 02-bench.sh <config> <sharegpt|prefix>}"
PORT="${PORT:-8000}"

# Le modèle passé au bench doit correspondre à celui que sert le serveur : c'est
# lui qui détermine le tokenizer utilisé pour compter les tokens.
case "$CONFIG" in
  base) MODEL="${BASE_MODEL:-Qwen/Qwen3-14B}" ;;
  *)    MODEL="${AWQ_MODEL:-Qwen/Qwen3-14B-AWQ}" ;;
esac

OUT="results/raw/${CONFIG}__${DATASET}.json"

COMMON=(--backend vllm --base-url "http://localhost:${PORT}"
        --model "$MODEL" --burstiness 1.0
        --percentile-metrics ttft,tpot,itl
        --save-result --result-filename "$OUT")

# Deux charges volontairement différentes, reprises du livre :
#   sharegpt = trafic réaliste, longueurs très dispersées (écart-type > moyenne)
#   prefix   = sonde de cache ; moins de préfixes uniques = signal plus fort
case "$DATASET" in
  sharegpt)
    ARGS=(--dataset-name sharegpt
          --dataset-path data/ShareGPT_V3_unfiltered_cleaned_split.json
          --num-prompts 2000 --request-rate 10 --max-concurrency 10) ;;
  prefix)
    ARGS=(--dataset-name prefix_repetition
          --num-prompts 1000 --request-rate 5 --max-concurrency 10
          --prefix-repetition-num-prefixes 10) ;;
  *) echo "Dataset inconnu : $DATASET" >&2; exit 1 ;;
esac

echo "Bench $CONFIG / $DATASET → $OUT"
vllm bench serve "${COMMON[@]}" "${ARGS[@]}" | tee "results/raw/${CONFIG}__${DATASET}.console.log"

echo
echo "Utilisation GPU pendant ce run (à relancer en parallèle du bench pour être utile) :"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used,pstate --format=csv -l 5"
