"""Extraction des quatre chiffres clés des logs de démarrage de vLLM.

`vllm bench serve` mesure le throughput et la latence, mais ne dit rien de la
répartition de la mémoire GPU. Or c'est elle qui explique les résultats : la
capacité de KV cache limite directement le batching et la concurrence, donc le
throughput. Ces chiffres ne vivent que dans les logs de chargement.

Les quatre lignes cherchées, telles que le livre les montre :

    Model loading took 27.5185 GiB
    Available KV cache memory: 11.00 GiB
    GPU KV cache size: 72,064 tokens
    Maximum concurrency for 40,960 tokens per request: 1.76x

Les formulations changent d'une version de vLLM à l'autre, d'où des motifs
tolérants (GiB ou GB, « Model loading took » ou « Loading model weights took »).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# GiB et GB sont traités comme équivalents : l'écart est sous le bruit de mesure
# et vLLM lui-même a alterné entre les deux étiquettes selon les versions.
PATTERNS = {
    "weights_gib": re.compile(
        r"(?:Model loading took|Loading model weights took)\s+([\d.]+)\s*G[iI]?B"
    ),
    "kv_cache_gib": re.compile(r"Available KV cache memory:\s*([\d.]+)\s*G[iI]?B"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    "max_concurrency": re.compile(
        r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)\s*x"
    ),
}


def _number(raw: str) -> float:
    return float(raw.replace(",", ""))


def parse(log_text: str) -> dict[str, float | None]:
    """Retourne les quatre chiffres, à None quand la ligne est absente du log."""
    out: dict[str, float | None] = {
        "weights_gib": None,
        "kv_cache_gib": None,
        "kv_cache_tokens": None,
        "max_model_len": None,
        "max_concurrency": None,
    }

    for key in ("weights_gib", "kv_cache_gib", "kv_cache_tokens"):
        match = PATTERNS[key].search(log_text)
        if match:
            out[key] = _number(match.group(1))

    match = PATTERNS["max_concurrency"].search(log_text)
    if match:
        out["max_model_len"] = _number(match.group(1))
        out["max_concurrency"] = _number(match.group(2))

    return out


def parse_file(path: Path) -> dict[str, float | None]:
    return parse(path.read_text(errors="replace"))


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m lab.parse_startup <fichier.startup.log>...", file=sys.stderr)
        return 1
    for arg in argv:
        path = Path(arg)
        print(f"# {path.name}")
        print(json.dumps(parse_file(path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
