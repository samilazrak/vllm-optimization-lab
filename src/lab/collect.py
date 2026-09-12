"""Consolidation de results/raw/ en un seul tableau : results/summary.csv.

Joint deux sources par la clé `config` :
  - results/raw/<config>__<dataset>.json  les métriques de `vllm bench serve`
  - results/raw/<config>.startup.log      la répartition mémoire au chargement

C'est cette jointure qui fait l'intérêt du tableau : mettre le throughput en face
de la taille du KV cache qui l'explique.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from lab.parse_startup import parse_file

RAW = Path("results/raw")
SUMMARY = Path("results/summary.csv")

# Les clés reprises du JSON de vllm bench serve, dans l'ordre d'affichage.
BENCH_FIELDS = [
    "num_prompts",
    "request_rate",
    "max_concurrency",
    "duration",
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
]

STARTUP_FIELDS = [
    "weights_gib",
    "kv_cache_gib",
    "kv_cache_tokens",
    "max_model_len",
    "max_concurrency_x",
]

COLUMNS = ["run_id", "config", "dataset", "model_id"] + BENCH_FIELDS + STARTUP_FIELDS


def collect() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    startup_cache: dict[str, dict[str, float | None]] = {}

    for bench_path in sorted(RAW.glob("*__*.json")):
        config, dataset = bench_path.stem.split("__", 1)
        bench = json.loads(bench_path.read_text())

        if config not in startup_cache:
            log_path = RAW / f"{config}.startup.log"
            startup_cache[config] = parse_file(log_path) if log_path.exists() else {}
        startup = startup_cache[config]

        row: dict[str, object] = {
            "run_id": bench_path.stem,
            "config": config,
            "dataset": dataset,
            "model_id": bench.get("model_id", ""),
        }
        for field in BENCH_FIELDS:
            row[field] = bench.get(field, "")
        for field in STARTUP_FIELDS:
            # max_concurrency existe des deux côtés avec deux sens différents
            # (le plafond demandé au bench vs le multiplicateur calculé par vLLM).
            key = "max_concurrency" if field == "max_concurrency_x" else field
            value = startup.get(key)
            row[field] = "" if value is None else value
        rows.append(row)

    return rows


def main() -> int:
    rows = collect()
    if not rows:
        print(f"Aucun résultat dans {RAW}/ (fichiers attendus : <config>__<dataset>.json)")
        return 1

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} run(s) → {SUMMARY}")
    for row in rows:
        print(
            f"  {row['run_id']:<24} "
            f"{row['total_token_throughput'] or '?'} TPS  "
            f"TTFT {row['mean_ttft_ms'] or '?'} ms  "
            f"KV {row['kv_cache_gib'] or '?'} GiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
