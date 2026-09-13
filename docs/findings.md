# Interprétation

Ce document sépare volontairement trois choses : ce que cette campagne mesure, ce
que le livre mesure, et ce qui n'est pas reproduit du tout.

Toutes les mesures citées viennent de [`results/summary.csv`](../results/summary.csv)
et des logs de [`results/raw/`](../results/raw/), produits sur un L40S 46 Go avec
vLLM 0.29.0.

## 1. Le profil mémoire se reproduit à moins de 1 %

| Mesure | Ce dépôt | Livre | Écart |
|---|---|---|---|
| Poids `base` | 27,52 GiB | 27,5185 GiB | identique |
| KV cache `base` | 11,06 GiB | 11,00 GiB | +0,5 % |
| Tokens cachables `base` | 72 496 | 72 064 | +0,6 % |
| Concurrence max `base` | 1,77× | 1,76× | +0,6 % |
| Poids `awq` | 9,44 GiB | 9,36 GiB | +0,9 % |
| KV cache `awq` | 29,09 GiB | 29,15 GiB | -0,2 % |
| Tokens cachables `awq` | 190 672 | 191 056 | -0,2 % |

La version de vLLM utilisée ici est nettement postérieure à celle du livre, et
pourtant rien ne bouge. La raison est mécanique : ces nombres sont une
soustraction entre la capacité de la carte et la taille du modèle. Le serveur
n'intervient que sur la marge qu'il se réserve pour les activations et les graphes
CUDA, et cette marge a très peu changé entre les deux versions.

C'est une bonne nouvelle méthodologique : **le profil mémoire est le point le plus
stable et le plus transposable du chapitre.** Il se calcule à l'avance et se
vérifie en lisant quatre lignes de log, sans lancer un seul benchmark.

Côté service, la reproduction est bonne sur le débit (+1 % sur les deux runs
`base`, +4 % sur `awq__sharegpt`) et sur l'ITL (42,4 ms contre 43,2 sur
`base__sharegpt`), ce qui situe l'exécution du modèle au même niveau.

**Une seule métrique diverge franchement : le TTFT.**

| Run | Ce dépôt | Livre | Écart |
|---|---|---|---|
| `base__sharegpt` | 145,2 ms | 104,2 ms | +39 % |
| `awq__sharegpt` | 75,6 ms | 59,3 ms | +27 % |

L'écart est trop large pour être ignoré et trop isolé pour être attribué au GPU :
si le calcul était plus lent, l'ITL et le débit le montreraient aussi. Il pointe
vers le chemin d'admission d'une requête, le tokenizer ou l'hôte plutôt que vers
le décodage. Cette partie n'a pas été instrumentée pendant la campagne, donc la
cause reste inconnue. Elle est consignée ici comme un écart non résolu, pas comme
un détail.

## 2. Le gain AWQ est réel, son explication ne l'est pas

Le livre pose la chaîne causale suivante : les poids 4 bits libèrent de la VRAM,
la VRAM libérée va au KV cache, un cache plus grand permet des batches plus gros
et moins d'évictions, donc le débit monte. Les deux premiers maillons sont
confirmés par le tableau ci-dessus. **Le troisième ne tient pas dans ces
conditions de mesure.**

Le calcul qui le montre tient en une division. Sur ShareGPT, une requête pèse
429 tokens en moyenne (223 d'entrée, 206 de sortie). Le KV cache de `base`, le
plus petit des trois, en contient 72 496, soit de la place pour **169 requêtes
simultanées**. Le benchmark en lance 10. Le cache est utilisé à 6 % de sa
capacité utile, et sur Prefix Repetition à 9 %.

Autrement dit, `base` et `awq` tenaient toutes les deux les 10 requêtes en vol
sans la moindre pression mémoire. Le batching ne peut donc pas expliquer l'écart
de débit. Ce qui l'explique :

| | `base` | `awq` | rapport |
|---|---|---|---|
| Poids du modèle | 27,52 GiB | 9,44 GiB | **2,92×** |
| ITL moyen | 42,4 ms | 15,2 ms | **2,79×** |
| Output TPS (ShareGPT) | 230,8 | 640,1 | **2,77×** |
| Durée du run | 1 783 s | 643 s | **2,77×** |

À faible batch, le décodage est borné par la bande passante mémoire : à chaque
token généré, le GPU relit l'intégralité des poids. Diviser les poids par 2,92
divise donc le temps par token par un facteur voisin, et c'est exactement ce
qu'on observe (2,79). Les trois autres rapports suivent à 2,77, ce qui est la
signature d'un système où **une seule grandeur a changé**.

La nuance importe, parce qu'elle change la prédiction qu'on tire de la mesure.
Si le gain venait du batching, il grandirait avec la concurrence. S'il vient de
la bande passante des poids, il est déjà là à une requête et **s'érode** quand le
batch grossit, puisque le coût de lecture des poids s'amortit alors sur plusieurs
séquences.

Ce n'est pas une erreur du livre : l'expansion du cache de 1,77× à 4,66× de
concurrence soutenable est bien réelle, et c'est elle qui compte sur une charge
saturée ou à contextes longs. C'est le protocole de mesure qui ne l'atteint pas,
en plafonnant la concurrence à 10. La conclusion correcte est plus étroite que
celle affichée : **sur cette charge, AWQ accélère le décodage ; le bénéfice
mémoire reste en réserve.**

Pour le tester il faudrait relancer la matrice avec `--max-concurrency` bien
au-delà de 170 sur ShareGPT, et regarder si le rapport `awq` / `base` s'écrase ou
tient. C'est la suite naturelle de ce lab.

## 3. Le réglage manuel n'a aucun effet mesurable

| Run | `awq` | `tuned` | Écart |
|---|---|---|---|
| ShareGPT | 1 334,2 TPS | 1 335,3 TPS | +0,08 % |
| Prefix Repetition | 2 873,0 TPS | 2 867,1 TPS | -0,2 % |

`tuned` ajoute pourtant six flags : `--gpu-memory-utilization 0.95`,
`--enable-prefix-caching`, `--enable-chunked-prefill`, `--max-num-seqs 512`,
`--max-num-batched-tokens 8192` et `--block-size 16`. Deux causes se cumulent
pour annuler leur effet.

**Le prefix caching et le chunked prefill sont activés par défaut depuis vLLM
0.29.** Les demander explicitement ne change rien : `base` en bénéficiait déjà,
ce que le taux de réutilisation du point 4 confirme directement. Le livre les
présente comme des optimisations à activer ; ce sont désormais des valeurs par
défaut, et une part du gain que le chapitre attribue au réglage est déjà dans la
baseline.

**Le seul gain réel est de 0,94 GiB de KV cache** (29,09 à 30,03 GiB), obtenu en
poussant `--gpu-memory-utilization` de 0,90 à 0,95. Il porte sur la ressource dont
le point 2 montre qu'elle n'était pas contraignante. Élargir une réserve déjà
utilisée à 6 % ne produit rien.

Le résultat est négatif, et c'est le plus utile du lot : il rappelle qu'un flag
n'améliore une charge que s'il desserre la contrainte qui la limite, et qu'il faut
donc avoir identifié cette contrainte avant de régler quoi que ce soit.

## 4. Le prefix caching a fonctionné sans se voir

Le serveur journalise un taux de réutilisation cumulé. Sur le run `base`, qui
enchaîne ShareGPT puis Prefix Repetition sans redémarrage :

| Moment | Taux cumulé |
|---|---|
| Fin de la phase ShareGPT | 0,2 % |
| Fin de la phase Prefix Repetition | 49,5 % |

Les deux phases totalisent 958 647 tokens d'entrée, dont 512 028 pour la phase
prefix. Comme la phase ShareGPT n'apporte quasiment aucun hit, les 49,5 % cumulés
se concentrent sur la seconde, ce qui la situe **autour de 90 % de réutilisation**.
Le mécanisme a donc parfaitement opéré, et l'écart 0,2 % contre 90 % est une
bonne illustration de ce que le choix du dataset décide : du trafic
conversationnel réel ne partage presque rien, du synthétique à 10 préfixes
partage presque tout.

Et pourtant le TTFT ne bouge pas : 150 ms sur `base__prefix` avec 512 tokens
d'entrée, contre 145 ms sur `base__sharegpt` avec 223 tokens. Servir deux fois
plus de prompt au même prix est bien le signe du cache, mais le gain absolu est
noyé. Le prefill pèse 145 ms quand le décodage d'une réponse en pèse environ
8 600.

**Le prefix caching se paie en TTFT, et cette campagne mesure du débit.** Sur une
charge interactive à réponses courtes, la hiérarchie s'inverserait.

## 5. Le serving distribué (non reproduit)

Cette section rapporte les résultats du livre. **Aucune mesure de ce dépôt ne les
confirme ni ne les infirme** : l'étape 8 demande deux pods multi-GPU aux
interconnexions différentes, ce qui sort du budget de cette campagne. Les configs
`tp2` et `tp4` existent dans `scripts/01-serve.sh` pour qui a le matériel.

**Résultat A : le multi-GPU peut être pire que le mono-GPU.**

| Instance | GPU | Interconnexion | Gagnant |
|---|---|---|---|
| g6e.12xlarge | 4× L40S | PCIe, pas de NVLink | **le mono-GPU** |
| p4d.24xlarge | 8× A100 | NVLink | le 4-GPU |

Le L40S est pourtant meilleur que l'A100 en inférence mono-GPU. C'est
l'interconnexion qui décide : sans NVLink, l'overhead de synchronisation du
tensor parallelism annule le gain de parallélisme.

**Résultat B : même quand le distribué gagne, quatre instances indépendantes le
battent en throughput** : 9 816 contre 3 926 TPS sur p4d.

D'où la conclusion : **le bénéfice du distribué n'est pas le débit, c'est la
latence et la taille de modèle accessible.** Sur p4d, passer de 1 à 4 GPU fait
tomber le TTFT de 66 à 33 ms. Ce gain-là, le scaling horizontal ne l'atteint
jamais : ajouter des répliques n'améliore pas la latence d'une requête.

## Ce que le lab ne mesure pas

- **Le KV cache sous pression.** La limite principale de cette campagne, détaillée
  au point 2 : `--max-concurrency 10` laisse le cache à 6 % de sa capacité utile,
  donc le levier central du chapitre n'est jamais sollicité.
- **La dégradation du modèle quantifié.** Le harnais mesure le débit, pas la
  qualité des sorties en 4 bits. Un ×2,77 de throughput ne dit rien du coût en
  précision, et l'arbitrage mémoire contre qualité est le deuxième des cinq
  arbitrages du chapitre.
- **Les techniques ciblées de l'étape 7.** LMCache pour les charges
  prefill-heavy et le speculative decoding pour les charges decode-heavy
  demandent chacun leur propre protocole, avec un dataset qui penche franchement
  d'un côté. La config `tuned` ne couvre que les leviers généraux de cache et de
  batching.
- **La variance entre runs.** Chaque point est un run unique. Les écarts de
  l'ordre du pourcent cités ici, notamment les +1 % et +4 % contre le livre, ne
  sont pas distinguables du bruit sans répétition.

## Le principe directeur

> In practice, we spend most of our effort identifying which optimization
> techniques to apply rather than chasing the perfect configuration.

Cette campagne en donne une illustration involontaire. Les deux résultats les plus
instructifs sont un résultat négatif (le réglage `tuned` n'apporte rien) et une
explication qui s'effondre à la vérification (le gain AWQ ne vient pas du
batching). Dans les deux cas, ce qui manquait n'était pas un flag mais
l'identification de la contrainte qui limitait réellement la charge.

L'ordre de travail qui en découle : comprendre le scénario, se donner un dataset
et des métriques représentatifs, appliquer les optimisations générales pour avoir
une baseline solide, et seulement ensuite cibler la charge. Une configuration
surajustée à un GPU et à un pattern de trafic n'est pas portable, et peut être
mauvaise ailleurs.
