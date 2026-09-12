#!/usr/bin/env bash
# Étape 4 : monter le serveur vLLM dans une configuration donnée, et surtout
# CAPTURER LES LOGS DE DÉMARRAGE : c'est le diagnostic central du chapitre.
#
#   ./scripts/01-serve.sh base     modèle fp16, flags par défaut (baseline du livre)
#   ./scripts/01-serve.sh awq      même modèle en AWQ 4 bits
#   ./scripts/01-serve.sh tuned    AWQ + prefix caching, chunked prefill, batching élargi
#   ./scripts/01-serve.sh tp2      AWQ sur 2 GPU en tensor parallelism
#   ./scripts/01-serve.sh tp4      AWQ sur 4 GPU
#
# Le serveur tourne en arrière-plan ; le PID est écrit dans results/raw/<config>.pid.
# Arrêt : ./scripts/01-serve.sh stop
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:?usage: 01-serve.sh <base|awq|tuned|tp2|tp4|stop>}"
PORT="${PORT:-8000}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-14B}"
# Si ce dépôt AWQ n'existe pas pour ta version du modèle, remplace-le par un
# dépôt AWQ communautaire équivalent : c'est la quantization qui compte ici, pas
# l'auteur du dépôt.
AWQ_MODEL="${AWQ_MODEL:-Qwen/Qwen3-14B-AWQ}"

if [ "$CONFIG" = "stop" ]; then
  for pidfile in results/raw/*.pid; do
    [ -f "$pidfile" ] || continue
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Arrêt du serveur $(basename "$pidfile" .pid) (PID $pid)"
      kill "$pid"
    fi
    rm -f "$pidfile"
  done
  exit 0
fi

# Les configs base et awq restent aux flags PAR DÉFAUT, comme le livre
# (`vllm serve Qwen/Qwen3-14B`). C'est la condition pour que les lignes
# « Available KV cache memory » et « Maximum concurrency » soient comparables aux
# siennes. Le réglage n'arrive qu'avec la config tuned.
case "$CONFIG" in
  base)  MODEL="$BASE_MODEL"; ARGS=() ;;
  awq)   MODEL="$AWQ_MODEL";  ARGS=(--quantization awq) ;;
  tuned) MODEL="$AWQ_MODEL";  ARGS=(--quantization awq
                                    --gpu-memory-utilization 0.95
                                    --enable-prefix-caching
                                    --enable-chunked-prefill
                                    --max-num-seqs 512
                                    --max-num-batched-tokens 8192
                                    --block-size 16) ;;
  tp2)   MODEL="$AWQ_MODEL";  ARGS=(--quantization awq --tensor-parallel-size 2) ;;
  tp4)   MODEL="$AWQ_MODEL";  ARGS=(--quantization awq --tensor-parallel-size 4) ;;
  *) echo "Config inconnue : $CONFIG" >&2; exit 1 ;;
esac

LOG="results/raw/${CONFIG}.startup.log"
echo "Lancement de $MODEL (config $CONFIG) sur le port $PORT"
echo "  flags : ${ARGS[*]:-aucun (défauts vLLM)}"
echo "  logs  : $LOG"

vllm serve "$MODEL" --port "$PORT" "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "results/raw/${CONFIG}.pid"

echo -n "Attente de la disponibilité du serveur (le chargement des poids peut prendre quelques minutes)"
for _ in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo " prêt."
    echo
    echo "=== Les 4 lignes qui comptent dans les logs de chargement ==="
    grep -E "Model loading took|Loading model weights took|Available KV cache memory|GPU KV cache size|Maximum concurrency" "$LOG" || \
      echo "(formulations différentes dans cette version de vLLM : voir $LOG)"
    exit 0
  fi
  sleep 5
  echo -n "."
done

echo
echo "Le serveur n'a pas répondu à temps. Fin du log :" >&2
tail -30 "$LOG" >&2
exit 1
