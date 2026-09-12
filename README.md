# vllm-optimization-lab

Un harnais de benchmark reproductible pour le **serving LLM sur vLLM** : une
commande lance la matrice complète des configurations, sauvegarde les résultats
bruts, et génère le rapport depuis les données.

Le protocole reprend le chapitre 9 de *Hands-On LLM Serving and Optimization*
(Chi Wang & Peiheng Hu, O'Reilly) : **Qwen3-14B sur un seul NVIDIA L40S 46 Go**.
Les chiffres du livre sont posés en référence dans le rapport, en face des
mesures, avec la colonne d'écart. Projet d'apprentissage : le code est court et
commenté.

> **État** : le harnais est écrit et validé sur des jeux de résultats de test.
> Les runs sur GPU ne sont pas encore lancés, donc `results/` ne contient pas
> encore de mesures réelles. Les tableaux ci-dessous donnent les chiffres du
> livre, que la campagne doit confirmer ou contredire.

## Ce que le lab cherche à montrer

**1. Le trafic répétitif double le throughput sans aucune configuration.**
Sur un trafic où les prompts partagent un préfixe, vLLM applique automatiquement
prefix caching, continuous batching et partage de blocs mémoire. Le livre mesure
1 123 contre 474 tok/s, à TTFT et ITL identiques.

**2. La quantization ne gagne pas du calcul, elle gagne de la place pour le KV
cache.** C'est le résultat le moins intuitif du chapitre. En AWQ 4 bits, les
poids passent de 27,5 à 9,4 Go, et les 17 Go libérés vont au KV cache, qui passe
de 11 à 29 Go, soit de 72 064 à 191 056 tokens cachables. Plus de cache veut dire
des batches plus gros et moins d'évictions, donc plus de throughput : 474 → 1 280
tok/s. Le gain vient de la mémoire, pas des kernels. Pour gagner du calcul, il
faut quantifier les activations.

**3. Le multi-GPU dépend de l'interconnexion, pas du GPU.** Non reproduit ici,
voir [docs/findings.md](docs/findings.md).

## Le point technique du harnais

`vllm bench serve` mesure le throughput et la latence, mais ne dit rien de la
répartition de la mémoire GPU. Or c'est elle qui explique les résultats. Ces
chiffres ne vivent que dans les logs de chargement :

```
Model loading took 27.5185 GiB
Available KV cache memory: 11.00 GiB
GPU KV cache size: 72,064 tokens
Maximum concurrency for 40,960 tokens per request: 1.76x
```

[`src/lab/parse_startup.py`](src/lab/parse_startup.py) les extrait,
[`collect.py`](src/lab/collect.py) les joint aux métriques de bench par
configuration. C'est cette jointure qui permet de mettre le throughput en face de
la taille de KV cache qui le plafonne, au lieu de constater un gain sans pouvoir
l'expliquer.

## La matrice

| `run_id` | Modèle | Dataset | Flags |
|---|---|---|---|
| `base__sharegpt` | Qwen3-14B | ShareGPT | défauts vLLM |
| `base__prefix` | Qwen3-14B | Prefix Repetition | défauts vLLM |
| `awq__sharegpt` | Qwen3-14B-AWQ | ShareGPT | `--quantization awq` |
| `awq__prefix` | Qwen3-14B-AWQ | Prefix Repetition | `--quantization awq` |
| `tuned__sharegpt` | Qwen3-14B-AWQ | ShareGPT | + prefix caching, chunked prefill, batching élargi |
| `tuned__prefix` | Qwen3-14B-AWQ | Prefix Repetition | idem |

Les deux datasets ne mesurent pas la même chose. **ShareGPT** est du trafic réel,
aux longueurs très dispersées (écart-type 241 pour une moyenne de 233), ce qui
est précisément le cas que le continuous batching sert à absorber. **Prefix
Repetition** est synthétique et sert de sonde de cache : moins il y a de préfixes
uniques, plus le signal de réutilisation est fort.

`base` et `awq` tournent aux **flags par défaut**, comme dans le livre
(`vllm serve Qwen/Qwen3-14B`). C'est la condition pour que les lignes de
chargement soient comparables aux siennes : fixer `--max-model-len` changerait la
concurrence annoncée et casserait la comparaison. Le réglage n'arrive qu'avec
`tuned`.

## Prérequis de l'hôte

Trois contraintes, apprises en brûlant trois pods loués avant d'en tenir un bon.
`make setup` les vérifie et s'arrête net si l'une n'est pas remplie, parce que
chacune se détecte en quelques secondes et coûte cher à découvrir plus tard.

| Contrainte | Seuil | Ce qui arrive sinon |
|---|---|---|
| **VRAM** | 48 Go | Le modèle fp16 occupe 27,5 Go : sur 24 Go la baseline n'existe pas |
| **Débit vers Hugging Face** | 20 Mo/s visés, 5 Mo/s minimum | À 0,5 Mo/s, les 28 Go de poids demandent 17 heures |
| **CUDA supporté par le driver** | 13.0 pour vLLM 0.29 | `torch.cuda.is_available()` à `False`, rien ne tourne |

La troisième est la moins évidente. Les wheels de vLLM sont compilées contre une
version de CUDA précise, et entre 12.x et 13.x le saut est majeur : la
compatibilité mineure ne joue plus, il faut un driver r580 ou plus récent. Sur
RunPod, le filtre « Available CUDA versions » de la page de déploiement est le
bon levier, et un hôte annonçant seulement 12.8 est à écarter.

La version de vLLM est épinglée dans `00-setup.sh` (`VLLM_VERSION`, 0.29.0 par
défaut). Un `pip install vllm` nu entre en conflit avec le torch préinstallé des
images RunPod et fait reculer le résolveur jusqu'à des versions de 2025, où
`vllm bench serve` n'existe pas encore.

## Utilisation

```bash
git clone https://github.com/samilazrak/vllm-optimization-lab && cd vllm-optimization-lab
make setup    # qualification de l'hôte, vLLM épinglé, ShareGPT
make sweep    # la matrice complète, puis collect + report
```

Pour passer outre les contrôles de qualification : `SKIP_HOST_CHECKS=1 make setup`.

Un run isolé :

```bash
make serve CONFIG=awq
make bench CONFIG=awq DATASET=sharegpt
make stop
```

L'analyse ne demande pas de GPU et se relance partout, sur les JSON versionnés :

```bash
uv venv && uv pip install -e .
make collect   # results/raw/*.json → results/summary.csv
make report    # results/summary.csv → results/report.md + figures/
```

## Structure

| Chemin | Rôle |
|---|---|
| [`scripts/00-setup.sh`](scripts/00-setup.sh) | Étapes 1 et 2 : matériel, vLLM, datasets |
| [`scripts/01-serve.sh`](scripts/01-serve.sh) | Étape 4 : serveur paramétré, capture des logs de chargement |
| [`scripts/02-bench.sh`](scripts/02-bench.sh) | Étapes 5 et 6 : trafic et métriques |
| [`scripts/03-sweep.sh`](scripts/03-sweep.sh) | La matrice de bout en bout, sans surveillance |
| [`src/lab/parse_startup.py`](src/lab/parse_startup.py) | Extraction des chiffres mémoire des logs vLLM |
| [`src/lab/collect.py`](src/lab/collect.py) | Jointure bench + chargement → `results/summary.csv` |
| [`src/lab/report.py`](src/lab/report.py) | Tableaux comparatifs et figures |
| `results/raw/` | JSON de bench et logs de démarrage, versionnés |
| [`docs/findings.md`](docs/findings.md) | Interprétation, écarts au livre, ce qui n'est pas reproduit |

## Limites

- **Le serving distribué (étape 8 du livre) n'est pas reproduit.** Il demande deux
  pods multi-GPU aux interconnexions différentes. Les configs `tp2` et `tp4`
  existent dans `01-serve.sh` mais ne sont pas dans la matrice par défaut. Le
  raisonnement et les chiffres du livre sont dans `docs/findings.md`, présentés
  comme tels et non comme des mesures.
- **La qualité du modèle quantifié n'est pas évaluée.** Le lab mesure le débit,
  pas la dégradation en 4 bits. Un throughput multiplié par 2,7 ne dit rien de ce
  qu'on perd en précision.
- **Les résultats ne valent que pour ce GPU et ce trafic.** C'est le fond du
  chapitre : une configuration surajustée ne se généralise pas.

## Référence

Chi Wang, Peiheng Hu, *Hands-On LLM Serving and Optimization*, O'Reilly,
chapitre 9 « LLM Optimization in Practice ».
