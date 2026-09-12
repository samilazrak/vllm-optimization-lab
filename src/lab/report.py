"""Génération du rapport depuis results/summary.csv.

Produit results/report.md et deux figures. Les chiffres du livre sont posés en
dur comme référence : tout l'intérêt du rapport est la colonne d'écart, qui dit
où la reproduction diverge et invite à expliquer pourquoi.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend à fixer avant l'import)

SUMMARY = Path("results/summary.csv")
REPORT = Path("results/report.md")
FIGURES = Path("figures")

# Hands-On LLM Serving and Optimization, ch. 9 : Qwen3-14B sur un L40S 46 Go.
BOOK = {
    "base__sharegpt": {"total_token_throughput": 474.38, "mean_ttft_ms": 104.15,
                       "mean_itl_ms": 43.24, "weights_gib": 27.52,
                       "kv_cache_gib": 11.00, "kv_cache_tokens": 72064,
                       "max_concurrency_x": 1.76},
    "base__prefix":   {"total_token_throughput": 1123.13, "mean_ttft_ms": 104.64,
                       "mean_itl_ms": 43.95},
    "awq__sharegpt":  {"total_token_throughput": 1280.0, "mean_ttft_ms": 59.29,
                       "weights_gib": 9.36, "kv_cache_gib": 29.15,
                       "kv_cache_tokens": 191056, "max_concurrency_x": 4.66},
}


# Ordre narratif du rapport, pas l'ordre alphabétique : la baseline d'abord,
# puis ce qui s'y compare.
CONFIG_ORDER = ["base", "awq", "tuned", "tp2", "tp4"]
DATASET_ORDER = ["sharegpt", "prefix"]


def sort_key(row: dict[str, str]) -> tuple[int, int, str]:
    config, dataset = row["config"], row["dataset"]
    return (
        CONFIG_ORDER.index(config) if config in CONFIG_ORDER else len(CONFIG_ORDER),
        DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER),
        row["run_id"],
    )


def load() -> list[dict[str, str]]:
    if not SUMMARY.exists():
        raise SystemExit(f"{SUMMARY} absent : lancer d'abord `python -m lab.collect`.")
    with SUMMARY.open() as handle:
        return sorted(csv.DictReader(handle), key=sort_key)


def num(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}".replace(",", " ")


def gap(measured: float | None, reference: float | None) -> str:
    """Écart relatif du mesuré par rapport au livre."""
    if measured is None or not reference:
        return "n/a"
    return f"{(measured / reference - 1) * 100:+.0f} %"


def table_metrics(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Run | Total TPS | Output TPS | TTFT moy. (ms) | ITL moy. (ms) | TPS du livre | Écart |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        run = row["run_id"]
        tps = num(row, "total_token_throughput")
        book_tps = BOOK.get(run, {}).get("total_token_throughput")
        lines.append(
            f"| `{run}` | {fmt(tps)} | {fmt(num(row, 'output_throughput'))} "
            f"| {fmt(num(row, 'mean_ttft_ms'))} | {fmt(num(row, 'mean_itl_ms'))} "
            f"| {fmt(book_tps) if book_tps else 'n/a'} | {gap(tps, book_tps)} |"
        )
    return "\n".join(lines)


def table_memory(rows: list[dict[str, str]]) -> str:
    """Le tableau qui explique les précédents : ce que vLLM annonce au chargement."""
    lines = [
        "| Config | Poids (GiB) | KV cache (GiB) | Tokens cachables | Concurrence max |",
        "|---|---|---|---|---|",
    ]
    seen: set[str] = set()
    for row in rows:
        config = row["config"]
        if config in seen:
            continue
        seen.add(config)
        lines.append(
            f"| `{config}` | {fmt(num(row, 'weights_gib'), 2)} "
            f"| {fmt(num(row, 'kv_cache_gib'), 2)} "
            f"| {fmt(num(row, 'kv_cache_tokens'), 0)} "
            f"| {fmt(num(row, 'max_concurrency_x'), 2)}× |"
        )
    return "\n".join(lines)


def figure_throughput(rows: list[dict[str, str]]) -> Path | None:
    labelled = [(r["run_id"], num(r, "total_token_throughput")) for r in rows]
    labelled = [(label, value) for label, value in labelled if value is not None]
    if not labelled:
        return None

    labels = [label for label, _ in labelled]
    measured = [value for _, value in labelled]
    reference = [BOOK.get(label, {}).get("total_token_throughput", 0) for label in labels]

    positions = range(len(labels))
    width = 0.38
    fig, axes = plt.subplots(figsize=(9, 4.5))
    axes.bar([p - width / 2 for p in positions], measured, width, label="mesuré")
    axes.bar([p + width / 2 for p in positions], reference, width, label="livre", alpha=0.55)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=20, ha="right")
    axes.set_ylabel("Total token throughput (tok/s)")
    axes.set_title("Throughput par configuration")
    axes.legend()
    fig.tight_layout()

    path = FIGURES / "throughput.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_memory(rows: list[dict[str, str]]) -> Path | None:
    """La figure centrale du lab : la quantization ne gagne pas du calcul, elle
    déplace de la mémoire des poids vers le KV cache."""
    configs: dict[str, tuple[float | None, float | None]] = {}
    for row in rows:
        configs.setdefault(row["config"], (num(row, "weights_gib"), num(row, "kv_cache_gib")))
    usable = {k: v for k, v in configs.items() if v[0] is not None and v[1] is not None}
    if not usable:
        return None

    labels = list(usable)
    weights = [usable[label][0] for label in labels]
    caches = [usable[label][1] for label in labels]

    fig, axes = plt.subplots(figsize=(7, 4.5))
    axes.bar(labels, weights, label="poids du modèle")
    axes.bar(labels, caches, bottom=weights, label="KV cache disponible")
    axes.set_ylabel("Mémoire GPU (GiB)")
    axes.set_title("Répartition de la mémoire GPU au chargement")
    axes.legend()
    fig.tight_layout()

    path = FIGURES / "memoire.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    rows = load()
    FIGURES.mkdir(exist_ok=True)

    sections = [
        "# Rapport de benchmark",
        "",
        "Généré par `python -m lab.report` depuis `results/summary.csv`.",
        "Référence : *Hands-On LLM Serving and Optimization*, ch. 9, Qwen3-14B sur L40S 46 Go.",
        "",
        "## Métriques de service",
        "",
        table_metrics(rows),
        "",
        "## Mémoire GPU au chargement",
        "",
        "Le tableau qui explique le précédent : le KV cache disponible plafonne le",
        "batching, donc le throughput.",
        "",
        table_memory(rows),
        "",
    ]

    for figure in (figure_throughput(rows), figure_memory(rows)):
        if figure:
            sections += [f"![{figure.stem}](../{figure})", ""]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(sections))
    print(f"{REPORT} écrit ({len(rows)} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
