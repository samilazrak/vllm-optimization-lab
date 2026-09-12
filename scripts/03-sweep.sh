#!/usr/bin/env bash
# La matrice complète des étapes 1 à 7, sans surveillance : pour chaque config,
# démarrer, benchmarker les deux datasets, arrêter.
#
# Les configs multi-GPU (tp2, tp4) ne sont PAS dans la matrice par défaut :
# l'étape 8 du livre demande un pod multi-GPU, et les conclusions du livre sur
# l'interconnexion n'ont pas été reproduites ici (cf. docs/findings.md).
# Pour les ajouter sur une machine adaptée : CONFIGS="base awq tuned tp2 tp4" ./scripts/03-sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIGS="${CONFIGS:-base awq tuned}"
DATASETS="${DATASETS:-sharegpt prefix}"

for config in $CONFIGS; do
  echo "############ CONFIG $config ############"
  ./scripts/01-serve.sh "$config"
  for dataset in $DATASETS; do
    ./scripts/02-bench.sh "$config" "$dataset"
  done
  ./scripts/01-serve.sh stop
  # Laisser la mémoire GPU se libérer avant la config suivante.
  sleep 20
done

echo
echo "Matrice terminée. Consolidation :"
PYTHONPATH=src python -m lab.collect
PYTHONPATH=src python -m lab.report
