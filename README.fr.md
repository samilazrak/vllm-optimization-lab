# vllm-optimization-lab

*[English version](README.md)*

**Les chiffres du livre se reproduisent à 1 % près. Son explication, elle, ne
survit pas à la vérification.**

Ce dépôt rejoue le lab du chapitre 9 de *Hands-On LLM Serving and Optimization*
(Chi Wang & Peiheng Hu, O'Reilly) : Qwen3-14B sur un seul NVIDIA L40S 46 Go. La
quantification AWQ produit bien le gain de débit annoncé par le chapitre, 2,77×
ici. Mais le mécanisme auquel il l'attribue, un KV cache plus grand qui permet
des batches plus gros, n'est pas ce qui l'a produit : à `--max-concurrency 10`,
le KV cache n'a jamais dépassé **6 % de sa capacité utile**, pas même sur la
baseline non quantifiée. Le gain vient de la bande passante mémoire sur la
lecture des poids pendant le décodage, et les mesures le démontrent à la
décimale près.

Y arriver demandait plus que lancer `vllm bench serve`, qui donne le débit et la
latence mais ne dit rien de la répartition de la mémoire GPU. Or c'est elle qui
explique les résultats, et elle ne vit que dans les logs de démarrage du
serveur. Ce harnais les en extrait et les joint aux métriques de bench, ce qui
est précisément ce qui a rendu le vrai goulot visible.

Une commande lance la matrice complète des configurations, sauvegarde les
résultats bruts et régénère le rapport depuis les données. Le code est court et
commenté.

## Environnement de mesure

| | |
|---|---|
| GPU | NVIDIA L40S, 46 068 MiB, compute capability 8.9 |
| Pile | vLLM 0.29.0, torch 2.13.0+cu130, CUDA 13.0, driver 580.159.04 |
| Hôte | RunPod, région EU-NL-1, 32 vCPU, 125 Go de RAM |
| Modèles | `Qwen/Qwen3-14B` et `Qwen/Qwen3-14B-AWQ` |
| Charge | 2 000 prompts ShareGPT et 1 000 prompts Prefix Repetition, `--max-concurrency 10` |

Un point de méthode : la version de vLLM est nettement postérieure à celle du
livre. Les écarts qui suivent sont donc à lire comme la robustesse du protocole
à un changement de version, pas comme une réplication à l'identique.

## Résultats

### Débit et latence

| Run | Total TPS | Output TPS | TTFT moy. | ITL moy. | TPS du livre | Écart |
|---|---|---|---|---|---|---|
| `base__sharegpt` | 481,4 | 230,8 | 145,2 ms | 42,4 ms | 474,4 | +1 % |
| `base__prefix` | 1 135,6 | 225,4 | 150,2 ms | 43,2 ms | 1 123,1 | +1 % |
| `awq__sharegpt` | 1 334,2 | 640,1 | 75,6 ms | 15,2 ms | 1 280,0 | +4 % |
| `awq__prefix` | 2 873,0 | 570,8 | 110,7 ms | 16,5 ms | n/a | n/a |
| `tuned__sharegpt` | 1 335,3 | 639,8 | 73,8 ms | 15,2 ms | n/a | n/a |
| `tuned__prefix` | 2 867,1 | 570,3 | 104,5 ms | 16,6 ms | n/a | n/a |

![throughput](figures/throughput.png)

Les totaux ne sont pas comparables entre les deux datasets : Prefix Repetition
envoie 512 k tokens d'entrée pour 127 k de sortie, ShareGPT 447 k pour 412 k. Le
« total token throughput » compte les deux, et un token d'entrée coûte bien moins
cher qu'un token de sortie. **C'est l'output TPS qui compare les datasets**, et
lui reste plat (230,8 contre 225,4 en base).

### Mémoire GPU au chargement

| Config | Poids | KV cache | Tokens cachables | Concurrence max |
|---|---|---|---|---|
| `base` | 27,52 GiB | 11,06 GiB | 72 496 | 1,77× |
| `awq` | 9,44 GiB | 29,09 GiB | 190 672 | 4,66× |
| `tuned` | 9,44 GiB | 30,03 GiB | 196 784 | 4,80× |

![memoire](figures/memoire.png)

La barre totale ne bouge pas, c'est la frontière à l'intérieur qui se déplace.

## Ce que les mesures montrent

**1. AWQ multiplie le débit par 2,77, mais pas par le mécanisme annoncé.**
Le livre explique le gain par la chaîne « moins de poids → plus de KV cache →
plus de batching → plus de débit ». Les mesures valident les deux premiers
maillons et invalident le troisième dans ces conditions : à `--max-concurrency
10`, le KV cache n'a jamais été contraignant, même en baseline. Une requête
ShareGPT pèse 429 tokens en moyenne, le cache de `base` en tient 72 496, soit
**169 requêtes simultanées quand on en demande 10**. Le goulot est ailleurs :

| | `base` | `awq` | rapport |
|---|---|---|---|
| Poids du modèle | 27,52 GiB | 9,44 GiB | **2,92×** |
| ITL moyen | 42,4 ms | 15,2 ms | **2,79×** |

Le décodage est borné par la bande passante mémoire sur la lecture des poids :
pour chaque token généré, le GPU relit l'intégralité du modèle. Diviser les
poids par 2,92 divise le temps par token par 2,79, et le débit total suit
exactement (2,77×). Ici la quantification accélère le décodage ; elle n'a pas eu
à débloquer le batching. L'expansion du cache (1,77× à 4,66× de concurrence
soutenable) est réelle, mais ce protocole ne l'a jamais sollicitée.

**2. Le réglage manuel n'apporte rien.** `tuned` ajoute
`--gpu-memory-utilization 0.95`, `--enable-prefix-caching`,
`--enable-chunked-prefill`, `--max-num-seqs 512`, `--max-num-batched-tokens 8192`
et `--block-size 16`. Résultat : 1 335,3 contre 1 334,2 TPS, soit +0,08 %, dans
le bruit. Deux causes se cumulent. Le prefix caching et le chunked prefill sont
**activés par défaut depuis vLLM 0.29**, donc ces flags ne font que redemander
l'existant. Et le seul gain réel, 0,94 GiB de KV cache supplémentaire, porte sur
la ressource dont le point précédent montre qu'elle n'était pas le goulot.

**3. Le prefix caching fonctionne et reste invisible sur le débit.** Les logs du
serveur donnent un taux de réutilisation cumulé de 0,2 % à la fin de la phase
ShareGPT, et 49,5 % à la fin de la phase Prefix Repetition, ce qui situe la
phase prefix seule autour de 90 %. Le cache a donc bien opéré. Mais il ne touche
que le prefill, qui pèse 145 ms face à environ 8,6 s de décodage par requête : le
TTFT de `base__prefix` reste à 150 ms pour 512 tokens d'entrée, contre 145 ms
pour 223 tokens sans cache. Économiser du prefill ne se voit pas sur une charge
dominée par le décodage.

**4. Le multi-GPU dépend de l'interconnexion, pas du GPU.** Non reproduit ici,
voir [docs/findings.fr.md](docs/findings.fr.md).

## Écarts au livre

La reproduction du profil mémoire est presque exacte, malgré une version de vLLM
très postérieure :

| Mesure | Ce dépôt | Livre | Écart |
|---|---|---|---|
| Poids `base` | 27,52 GiB | 27,5185 GiB | identique |
| KV cache `base` | 11,06 GiB | 11,00 GiB | +0,5 % |
| Tokens cachables `base` | 72 496 | 72 064 | +0,6 % |
| Concurrence max `base` | 1,77× | 1,76× | +0,6 % |
| Poids `awq` | 9,44 GiB | 9,36 GiB | +0,9 % |
| Tokens cachables `awq` | 190 672 | 191 056 | -0,2 % |

C'est attendu une fois qu'on voit d'où viennent ces nombres : ils sont dictés par
la taille du modèle et celle de la carte, pas par la version du serveur. Le
serveur n'intervient que sur la marge qu'il se réserve, et celle-ci a très peu
bougé.

Les débits tiennent aussi, à +1 % sur les deux runs `base` et +4 % sur
`awq__sharegpt`. L'ITL suit (42,4 ms contre 43,2 chez les auteurs sur
`base__sharegpt`), ce qui indique que l'exécution du modèle elle-même est
identique.

**Une métrique ne se reproduit pas : le TTFT.** 145,2 ms contre 104,2 chez les
auteurs sur `base__sharegpt`, et 75,6 contre 59,3 sur `awq__sharegpt`, soit +27 à
+39 %. Comme le débit et l'ITL tombent juste, l'écart ne vient pas du GPU : il se
situe du côté de l'admission des requêtes, du tokenizer ou de l'hôte, pas du
décodage. Faute d'avoir instrumenté cette partie, la cause reste non identifiée,
et c'est signalé comme tel plutôt que lissé.

**Le vrai écart n'est pas numérique, il est explicatif.** Les chiffres du livre
se reproduisent ; son interprétation du gain AWQ ne survit pas à la
vérification, parce que le protocole plafonne la concurrence à 10 et ne met
jamais le KV cache sous pression.

## Le point technique du harnais

`vllm bench serve` mesure le throughput et la latence, mais ne dit rien de la
répartition de la mémoire GPU. Or c'est elle qui explique les résultats. Ces
chiffres ne vivent que dans les logs de chargement :

```
Model loading took 27.52 GiB memory and 51.881960 seconds
Available KV cache memory: 11.06 GiB
GPU KV cache size: 72,496 tokens
Maximum concurrency for 40,960 tokens per request: 1.77x
```

[`src/lab/parse_startup.py`](src/lab/parse_startup.py) les extrait,
[`collect.py`](src/lab/collect.py) les joint aux métriques de bench par
configuration. C'est cette jointure qui permet de mettre le throughput en face de
la taille de KV cache censée le plafonner, au lieu de constater un gain sans
pouvoir l'expliquer. Sans elle, le constat n°1 ci-dessus était hors de portée :
c'est en divisant les tokens cachables par la longueur moyenne des requêtes que
le vrai goulot apparaît.

## La matrice

| `run_id` | Modèle | Dataset | Flags |
|---|---|---|---|
| `base__sharegpt` | Qwen3-14B | ShareGPT | défauts vLLM |
| `base__prefix` | Qwen3-14B | Prefix Repetition | défauts vLLM |
| `awq__sharegpt` | Qwen3-14B-AWQ | ShareGPT | `--quantization awq` |
| `awq__prefix` | Qwen3-14B-AWQ | Prefix Repetition | `--quantization awq` |
| `tuned__sharegpt` | Qwen3-14B-AWQ | ShareGPT | + prefix caching, chunked prefill, batching et cache élargis |
| `tuned__prefix` | Qwen3-14B-AWQ | Prefix Repetition | idem |

Les deux datasets ne mesurent pas la même chose. **ShareGPT** est du trafic réel,
aux longueurs très dispersées, ce qui est précisément le cas que le continuous
batching sert à absorber. **Prefix Repetition** est synthétique et sert de sonde
de cache : moins il y a de préfixes uniques, plus le signal de réutilisation est
fort.

`base` et `awq` tournent aux **flags par défaut**, comme dans le livre
(`vllm serve Qwen/Qwen3-14B`). C'est la condition pour que les lignes de
chargement soient comparables aux siennes : fixer `--max-model-len` changerait la
concurrence annoncée et casserait la comparaison. Le réglage n'arrive qu'avec
`tuned`.

La matrice complète prend environ deux heures sur un L40S, dont 30 minutes pour
le seul `base__sharegpt`.

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
| [`results/summary.csv`](results/summary.csv) | Les 6 runs, une ligne chacun |
| [`results/report.md`](results/report.md) | Rapport généré |
| `results/raw/` | JSON de bench et logs de démarrage, versionnés |
| [`docs/findings.fr.md`](docs/findings.fr.md) | Interprétation, écarts au livre, ce qui n'est pas reproduit |

## Limites

- **La concurrence est plafonnée à 10 côté client.** C'est la limite qui pèse le
  plus sur les conclusions : elle laisse le KV cache à 6 % de sa capacité utile
  et empêche donc de mesurer l'effet que le livre attribue à la quantification.
  Pour le tester il faudrait monter `--max-concurrency` au-delà de 170 sur
  ShareGPT, ou allonger fortement les contextes.
- **Le serving distribué (étape 8 du livre) n'est pas reproduit.** Il demande deux
  pods multi-GPU aux interconnexions différentes. Les configs `tp2` et `tp4`
  existent dans `01-serve.sh` mais ne sont pas dans la matrice par défaut. Le
  raisonnement et les chiffres du livre sont dans `docs/findings.fr.md`, présentés
  comme tels et non comme des mesures.
- **La qualité du modèle quantifié n'est pas évaluée.** Le lab mesure le débit,
  pas la dégradation en 4 bits. Un throughput multiplié par 2,77 ne dit rien de ce
  qu'on perd en précision.
- **Chaque point est un run unique.** Pas de répétition, donc pas de mesure de
  dispersion. Les écarts de l'ordre du pourcent ci-dessus sont à lire avec cette
  réserve.
- **Les résultats ne valent que pour ce GPU et ce trafic.** C'est le fond du
  chapitre : une configuration surajustée ne se généralise pas.

## Référence

Chi Wang, Peiheng Hu, *Hands-On LLM Serving and Optimization*, O'Reilly,
chapitre 9 « LLM Optimization in Practice ».
