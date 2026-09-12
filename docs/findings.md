# Interprétation

Ce document sépare volontairement trois choses : ce que le livre mesure, ce que
cette campagne mesure, et ce qui n'est pas reproduit du tout.

## 1. Pourquoi le trafic répétitif double le throughput

Le livre mesure, sur le même serveur et sans changer un seul flag :

| Dataset | Total TPS | TTFT moy. | ITL moy. |
|---|---|---|---|
| ShareGPT | 474 | 104 ms | 43 ms |
| Prefix Repetition (10 préfixes uniques) | **1 123** | 105 ms | 44 ms |

Le point important est que **la latence ne bouge pas**. Ce n'est donc pas un
arbitrage débit contre latence, c'est du calcul qui n'a pas lieu : sous trafic
répétitif, vLLM réutilise les blocs de KV déjà calculés pour le préfixe commun.
Le prefix caching, le continuous batching et le partage de blocs sont appliqués
automatiquement, sans configuration.

Conséquence pratique : **un benchmark mené sur un dataset non représentatif donne
un chiffre faux dans les deux sens.** Mesurer sur du trafic synthétique répétitif
quand la production est hétérogène surestime largement la capacité réelle.

## 2. Pourquoi la quantization améliore le throughput

C'est le résultat le plus contre-intuitif, parce que le mécanisme n'est pas celui
qu'on suppose. En AWQ 4 bits :

| | Original | AWQ 4 bits |
|---|---|---|
| Poids du modèle | 27,52 GiB | 9,36 GiB |
| KV cache disponible | 11,00 GiB | 29,15 GiB |
| Tokens cachables | 72 064 | 191 056 |
| Concurrence max | 1,76× | 4,66× |
| Total throughput | 474 TPS | 1 280 TPS |
| TTFT moyen | 104 ms | 59 ms |

Sur 46 Go de GPU, le modèle fp16 en occupe 27,5, soit plus de 65 %. Ce qui reste
au KV cache plafonne le nombre de séquences que le scheduler peut garder en vol.
Cache contraint, le serveur réduit le batch ou évince plus souvent, et une
éviction au décodage se paie en recalcul.

La quantization desserre cette contrainte : les 17 Go libérés vont au cache, qui
passe de 72 000 à 191 000 tokens, et la concurrence soutenable de 1,76× à 4,66×.
**Le gain vient de la place, pas des kernels.** Les auteurs le disent
explicitement : cette expérience démontre la réduction du mouvement de données et
de l'usage mémoire. Pour un gain de calcul, il faut quantifier aussi les
activations.

C'est le sens de la figure `figures/memoire.png` : la barre totale ne change
presque pas, c'est la frontière à l'intérieur qui se déplace.

## 3. Le serving distribué (non reproduit)

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

- **La dégradation du modèle quantifié.** Le harnais mesure le débit, pas la
  qualité des sorties en 4 bits. Un ×2,7 de throughput ne dit rien du coût en
  précision, et l'arbitrage mémoire contre qualité est le deuxième des cinq
  arbitrages du chapitre.
- **Les techniques ciblées de l'étape 7.** LMCache pour les charges
  prefill-heavy et le speculative decoding pour les charges decode-heavy
  demandent chacun leur propre protocole, avec un dataset qui penche franchement
  d'un côté. La config `tuned` ne couvre que les leviers généraux de cache et de
  batching.
- **La variance entre runs.** Chaque point est un run unique. Pour conclure sur
  des écarts de quelques pourcents, il faudrait répéter et regarder la dispersion.

## Le principe directeur

> In practice, we spend most of our effort identifying which optimization
> techniques to apply rather than chasing the perfect configuration.

Une configuration surajustée à un GPU et à un pattern de trafic n'est pas
portable, et peut être mauvaise ailleurs. L'ordre de travail qui en découle :
comprendre le scénario, se donner un dataset et des métriques représentatifs,
appliquer les optimisations générales pour avoir une baseline solide, et
seulement ensuite cibler la charge.
