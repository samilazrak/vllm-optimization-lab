# Rapport de benchmark

Généré par `python -m lab.report` depuis `results/summary.csv`.
Référence : *Hands-On LLM Serving and Optimization*, ch. 9, Qwen3-14B sur L40S 46 Go.

## Métriques de service

| Run | Total TPS | Output TPS | TTFT moy. (ms) | ITL moy. (ms) | TPS du livre | Écart |
|---|---|---|---|---|---|---|
| `base__sharegpt` | 481.4 | 230.8 | 145.2 | 42.4 | 474.4 | +1 % |
| `base__prefix` | 1 135.6 | 225.4 | 150.2 | 43.2 | 1 123.1 | +1 % |
| `awq__sharegpt` | 1 334.2 | 640.1 | 75.6 | 15.2 | 1 280.0 | +4 % |
| `awq__prefix` | 2 873.0 | 570.8 | 110.7 | 16.5 | n/a | n/a |
| `tuned__sharegpt` | 1 335.3 | 639.8 | 73.8 | 15.2 | n/a | n/a |
| `tuned__prefix` | 2 867.1 | 570.3 | 104.5 | 16.6 | n/a | n/a |

## Mémoire GPU au chargement

Le tableau qui explique le précédent : le KV cache disponible plafonne le
batching, donc le throughput.

| Config | Poids (GiB) | KV cache (GiB) | Tokens cachables | Concurrence max |
|---|---|---|---|---|
| `base` | 27.52 | 11.06 | 72 496 | 1.77× |
| `awq` | 9.44 | 29.09 | 190 672 | 4.66× |
| `tuned` | 9.44 | 30.03 | 196 784 | 4.80× |

![throughput](../figures/throughput.png)

![memoire](../figures/memoire.png)
