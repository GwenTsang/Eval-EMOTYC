# Évaluation du modèle EMOTYC

Ce dépôt a été conçu pour évaluer les performances du modèle **[EMOTYC](https://huggingface.co/TextToKids/CamemBERT-base-EmoTextToKids)** sur le corpus [CyberAgression-Large](https://github.com/aollagnier/CyberAgression-Large), contenant des messages de cyberharcèlement en français rédigés par des jeunes âgés de 11 à 18 ans. EMOTYC a été conçu par Etienne ([2023](https://bdr.parisnanterre.fr/theses/internet/2023/2023PA100047/2023PA100047.pdf)) dans le cadre du projet [ANR TextToKids](https://texttokids.irisa.fr/publications/)


# Table des matières

- [1. Cadre théorique et schéma d'annotation utilisé](#1-cadre-théorique-et-schéma-dannotation-utilisé)
  - [1.1 L'Unité d'annotation](#11-lunité-dannotation)
  - [1.2 Les catégories émotionnelles](#12-les-catégories-émotionnelles)
  - [1.3 Les modes d'expression](#13-les-modes-dexpression)
  - [1.4 Les trois types](#14-les-trois-types)
  - [1.5 Transposition au niveau phrastique : le vecteur à 19 labels](#15-transposition-au-niveau-phrastique-le-vecteur-à-19-labels)
- [2. Architecture du modèle EMOTYC](#2-architecture-du-modèle-emotyc)
  - [2.1 De CamemBERT-base à EMOTYC](#21-de-camembert-base-à-emotyc)
  - [2.2 Format d'entrée](#22-format-dentrée)
- [3. Données évaluées](#3-données-évaluées)
  - [3.1. Echantillons](#31-echantillons)
- [4. Performances du modèle EMOTYC](#4-performances-du-modèle-emotyc)
  - [4.1 Métriques utilisées](#41-métriques-utilisées)
  - [4.2 Répliquer les résultats officiels sur le corpus Test](#42-répliquer-les-résultats-officiels-sur-le-corpus-test)
  - [4.3 Performance sur CyberAggAdo avec les mêmes paramètres](#43-performance-sur-cyberaggado-avec-les-mêmes-paramètres)
  - [4.4 CyberAggAdo — contexte + seuil modes 0.06](#44-cyberaggado-contexte-+-seuil-modes-006)
  - [4.5 TTK — contexte + seuil modes 0.06](#45-ttk-contexte-+-seuil-modes-006)
  - [4.6 CyberAggAdo — sans contexte + seuil modes 0.06](#46-cyberaggado-sans-contexte-+-seuil-modes-006)
  - [4.7 TTK — sans contexte + seuil modes 0.06](#47-ttk-sans-contexte-+-seuil-modes-006)
  - [4.8 Échantillons XLSX aléatoires contigus](#48-échantillons-contigus)
  - [4.9 Échantillon XLSX  non contigus](#49-échantillon-aléatoire)
  - [4.10 Écarts TTK vs. CyberAggAdo — avec contexte](#410-écarts-ttk-vs-cyberaggado-avec-contexte)
  - [4.11 Écarts TTK vs. CyberAggAdo — sans contexte](#411-écarts-ttk-vs-cyberaggado-sans-contexte)
- [5. Remarques relatives à la configuration et aux hyperparamètres](#5-remarques-relatives-à-la-configuration-et-aux-hyperparamètres)
  - [5.1 Génération d'un rapport HTML](#51-génération-dun-rapport-html)
  - [5.2 Contiguité et non-contiguité](#52-contiguité-et-non-contiguité)
- [6. Remarques relatives à l'optimisation des scripts d'inférence](#6-remarques-relatives-à-loptimisation-des-scripts-dinférence)
- [7. Reproductibilité et commandes utilisées](#7-reproductibilité-et-commandes-utilisées)

# 1. Cadre théorique et schéma d'annotation utilisé

## 1.1 L'Unité d'annotation

Le schéma d'annotation utilisé est celui proposé par Etienne et Battistelli ([2021](https://hal.science/hal-03263194v1/document)) et développé dans Etienne ([2023](https://bdr.parisnanterre.fr/theses/internet/2023/2023PA100047/2023PA100047.pdf)). Il modélise l'expression émotionnelle dans les textes à travers un triplet :

<p align="center">
  <code>SitEmo = (Span ; Catégorie émotionnelle ; Mode d'expression)</code>
</p>

- **Span** : un intervalle `[i, j]` qui délimite le segment textuel porteur de l'émotion au sein d'une phrase. Ce segment peut aller d'un seul signe de ponctuation (`!`) à une proposition entière.
- **Catégorie émotionnelle** : l'émotion exprimée (parmi 12 catégories, voir §1.2).
- **Mode d'expression** : la *manière* dont l'émotion est linguistiquement réalisée (parmi 4 modes, voir §1.3).

Une phrase peut contenir zéro, une ou plusieurs unités SitEmo, et les segments de deux SitEmo distinctes peuvent se chevaucher.

## 1.2 Les catégories émotionnelles

Le schéma distingue 12 catégories émotionnelles, chacune regroupant des émotions fines :

| Catégorie | Émotions fines associées |
|:---|:---|
| **Colère** | agacement, colère, contestation, désapprobation, énervement, fureur/rage, indignation, irritation, mécontentement, révolte… |
| **Dégoût** | dégoût, lassitude, répulsion |
| **Joie** | amusement, enthousiasme, exaltation, joie, plaisir |
| **Peur** | angoisse, appréhension, effroi, horreur, inquiétude, méfiance, peur, stress |
| **Surprise** | étonnement, stupeur, surprise |
| **Tristesse** | blues, chagrin, déception, désespoir, peine, souffrance, tristesse |
| **Admiration** | admiration |
| **Culpabilité** | culpabilité |
| **Embarras** | embarras, gêne, honte, humiliation |
| **Fierté** | fierté, orgueil |
| **Jalousie** | jalousie |
| **Autre** | amour, courage, curiosité, désir, espoir, haine, mépris, soulagement… |

## 1.3 Les modes d'expression

Le mode qualifie la *relation* entre le segment textuel et l'émotion qu'il exprime. Il repose sur la typologie de Micheli (2014), adaptée par Etienne ([2023](https://bdr.parisnanterre.fr/theses/internet/2023/2023PA100047/2023PA100047.pdf)) :

| Mode | Définition | Exemples |
|:---|:---|:---|
|  **Désigné** | L'émotion est nommée explicitement par un terme du lexique émotionnel. | « Paul est *heureux*. » → Joie |
|  **Comportemental** | L'émotion est inférée à partir de la description d'une manifestation physique ou comportementale. | « Elle *éclata en sanglots*. » → Tristesse |
| **Suggéré** | L'émotion est inférée par le lecteur à partir d'une situation décrite, conventionnellement associée à un ressenti. | « Paul *a gagné la course*. » → Joie/Fierté |
| **Montré** | L'émotion transparaît à travers les caractéristiques formelles de l'énoncé (interjections, ponctuation expressive, syntaxe fragmentée, etc.). | « *DEHORSSSSS* » → Colère |

Une unité SitEmo ne peut recevoir qu'un seul mode.

## 1.4 Les trois types

Les 12 catégories émotinonnelles sont regroupées en trois types :

<br>
<p align="center">
  <img src="illustrations/types_emotions.svg" width="700">
</p>

## 1.5 Transposition au niveau phrastique : le vecteur à 19 labels


Pour l'entraînement et l'évaluation d'EMOTYC, les annotations fines (au niveau des segments) sont agrégées au niveau de la phrase par un "aplatissement". Cette agrégation rompt le lien entre une émotion spécifique et son mode d'expression. Pour une phrase contenant deux SitEmo (p. ex. une colère montrée et une tristesse désignée), le vecteur activera `Colère=1`, `Tristesse=1`, `Montré=1` et `Désigné=1`, sans permettre de reconstruire quel mode s'applique à quelle émotion.

Si au moins un segment de la phrase porte une propriété donnée, le label correspondant est activé (`1`) pour la phrase entière (si une phrase contient deux segments exprimant la colère, elle est associée à un vecteur dont le 10ème indice est 1, tout comme une phrase qui l'exprime sur un seul segment).



Ainsi, si une instance est étiquetée `Base = 1` dans le gold, cela peut être interprété comme une disjonction entre toutes les émotions appartenant à l'ensemble des « émotions de base » (cette disjonction étant inclusive, car plusieurs émotions peuvent être activées à la fois sur une même unité textuelle). Cette logique de disjonction est la même pour `Complexe = 1` (avec l'ensemble des émotions complexes) et pour `Emo = 1` (avec tous les labels émotionnels).

Il est possible de mesurer la « cohérence » des prédictions du modèle EMOTYC avec ce cadre théorique (p. ex., il ne devrait pas prédire `Base = 1` si aucune émotion de base n'est activée, ni prédire une émotion complexe (p. ex. `Culpabilité = 1`) sans prédire `Complexe = 1`). Cette cohérence n'est pas mesurée ici, mais elle l'est [dans ce script](https://github.com/GwenTsang/EMOTYC/blob/master/scripts/emotyc_sanity_check.py).





## 2. Architecture du modèle EMOTYC

### 2.1 De CamemBERT-base à EMOTYC

EMOTYC est une version fine-tunée de [CamemBERT-base](https://arxiv.org/abs/1911.03894) avec une tête de classification multi-label ajoutée. La sortie est un vecteur de prédictions :

$$\hat{\mathbf{y}} = [\hat{y}_1, \ldots, \hat{y}_{19}] $$

où chaque $\hat{y}_i$ est dans l'intervalle {0, 1}.

Concrètement, une phrase qui exprime la joie sur un mode comportemental (p. ex. la phrase "Il lui a adressé un sourire") sera représentée par ce vecteur :
<br>
<p align="center">
  <img src="illustrations/emotyc_output_vector.svg" width="650">
</p>

Pour ce qui concerne le fine-tuning, Etienne et al. (2024, p. 5) rapportent une stratégorie en deux temps. Dans une première phase, ils ont fait l'affinage sur la seule tâche de détection de présence/absence d'émotion (1 sortie binaire). Dans un second temps, ils ont fait un affinage multi-tâches sur les 19 labels simultanément, à partir des poids de la phase 1. L'optimiseur est Adam (lr = 10⁻⁵, pas de decay, batch size = 8) avec une pondération des classes plafonnée à 50 pour gérer le déséquilibre.

### 2.2 Format d'entrée

Le modèle a été entraîné avec le template :

```txt
before:{previous_sentence}current:{target_sentence}after:{next_sentence}
```

ce template est désigné "template BCA" (pour _Before, Current, After_).
Le fine-tuning a été réalisé avec `add_special_tokens=False`. En conséquence le premier token est `_be` (premier sous-mot de `"before"`). C'est l'état caché de ce token en position 0 à la 12ᵉ couche qui est transmis à la tête de classification.





Les labels `Emo`, `Base` et `Complexe` sont logiquement impliqués par les autres labels. Par exemple, `Base = 1` si et seulement si au moins l'une des 6 émotions de base est à `1`.



## 3. Données évaluées

Nous testons les performances d'EMOTYC sur deux corpus. D'une part, [`emotexttokids_gold_flat.xlsx`](golds/emotexttokids_gold_flat.xlsx), qui est le corpus d'entraînement d'EMOTYC, contenant des articles de presse jeunesse et de la littérature pour enfants. Le sous-ensemble TEST est disponible sur [HuggingFace](https://huggingface.co/datasets/TextToKids/EmoTextToKids-sentences).

D'autre part, un corpus contenant des messages de Cyber Harcèlement qui est sous-partie du corpus [CyberAgression-Large-v2](https://github.com/aollagnier/CyberAgression-Large) publié par Ollagnier ([2024](https://hal.science/hal-04514689v1/document)). Ce corpus peut être dit "hors-domaine" dans la mesure où le corpus de fine-tuning d'EMOTYC ne contient pas de messages numériques similaires. Nous avons annoté 781 lignes selon le schéma d'Etienne (2023) via Label Studio pour produire [`golds/CyberAdoAgg_gold_global_total.xlsx`](golds/CyberAdoAgg_gold_global_total.xlsx) en utilisant [ce script d'annotation](https://github.com/42009221/AnnotationsCyberAggAdo).



## 3.1. Echantillons

Le script [`prepare_xlsx_samples.py`](prepare_xlsx_samples.py) permet un échantilonnage aléatoire


Le dossier [`results`](results) contient l'ensemble des inférences déjà générées par les scripts d'inférence sont organisées par corpus évalué et par configuration testée.


Ce dossier gold contient également deux autres corpus qui sont des versions échantillonnées aléatoirement de CyberAggAdo. Le script d'échantillonage aléatoire utilisé est [prepare_xlsx_samples.py](prepare_xlsx_samples.py), dans lequel un `argparse` permet de choisir entre un échantillonage aléatoire ou non.

Ainsi, d'un côté, par un échantillonage non-contigu, nous avons obtenu [randomSample120.xlsx](`golds/random_sample_120.xlsx`), on expose les performances d'EMOTYC sur ce XLSX dans la section 2.2.6. Dans la mesure où, dans ce XLSX, les phrases ne se suivent pas, il n'y aurait pas de sens à utiliser l'option `use-context`.
C'est la raison pour laquelle nous échantillonnons aussi en "blocs contigus". Cela permet d'avoir pouvoir XLSX séparés, et ainsi d'utiliser le script [`orchestrate_emotyc_folder.py`](orchestrate_emotyc_folder.py). Les résultats sur ce corpus sont exposés dans la section 2.2.5.




## 4. Performances du modèle EMOTYC

### 4.1 Métriques utilisées

La précision mesure la fiabilité des prédictions positives :

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Elle évalue, parmi les instances prédites comme positives par le modèle, la proportion réellement correcte. Une baisse de précision sur CyberAggAdo indique une augmentation des faux positifs : EMOTYC attribue à tort un label émotionnel. Cela suggère que certains indices lexicaux ou contextuels valides dans TTK deviennent trompeurs dans CyberAggAdo.

Le rappel mesure la capacité du modèle à retrouver les instances réellement positives :

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Il porte sur l’ensemble des instances pour lesquelles `y=1`. Une baisse de rappel indique une augmentation des faux négatifs : EMOTYC ne détecte plus certaines occurrences. Cela suggère par ex. que l’émotion concernée est exprimée dans CyberAggAdo par des formes lexicales, discursives ou contextuelles différentes de celles apprises sur TTK (EMOTYC n'ayant jamais vu ces formes, il ne les détecte pas).




### 4.2 Répliquer les résultats officiels sur le corpus Test

Etienne et al. ([2024](https://arxiv.org/abs/2405.14385)) rapportent les performances suivantes, sur le sous ensemble TEST du corpus TTK, avec les phrases adjacentes (contexte) injectées dans le template BCA et des seuils à 0.5 pour tous les labels :

|  | Rappel (Macro) | Précision (Macro) | Macro F1 |
| :--- | :---: | :---: | :---: |
| Présence d'une émotion | 0.76 | 0.74 | 0.75 |
| Mode d'expression | 0.63 | 0.67 | 0.64 |
| Type | 0.56 | 0.66 | 0.60 |
| Catégorie émotionnelle | 0.40 | 0.46 | 0.42 |

Nous avons essayé de reproduire à l'identique ces paramètres, en partant du sous-ensemble TEST du [corpus TTK donné sur HuggingFace](https://huggingface.co/datasets/TextToKids/EmoTextToKids-sentences/blob/main/data/test-00000-of-00001.parquet) ainsi qu'avec les poids du modèle donnés sur [HuggingFace](https://huggingface.co/TextToKids/CamemBERT-base-EmoTextToKids) à travers le script [`emotyc_predict_parquet.py`](emotyc_predict_parquet.py), qui a été exécuté dans ce [notebook Colab T4](https://colab.research.google.com/drive/17dVMtpKE4Ca2eKJ_tDvaUa1FF-e6igjn?usp=sharing). Mais les performances obtenues sont supérieures à celles qui sont documentées dans l'article :

|  | Rappel (Macro) | Précision (Macro) | Macro F1 |
| :--- | :---: | :---: | :---: |
| Présence d'une émotion | 0.93 | 0.92 | 0.92 |
| Mode d'expression | 0.81 | 0.82 | 0.81 |
| Type | 0.76 | 0.83 | 0.79 |
| Catégorie émotionnelle | 0.55 | 0.60 | 0.57 |

Une hypothèse pour expliquer ces écarts serait que les résultats donnés dans l'article découlent d'une moyenne des performances des différents "checkpoints" du modèle EMOTYC (une moyenne de ses performances à travers les epochs), et qu'on accède, via le dépôt HuggingFace, aux meilleurs checkpoints (aux meilleurs poids).

Performances détaillées label par label :

*(Les illustrations SVG de cette section ont été retirées)*


### 4.3 Performance sur CyberAggAdo avec les mêmes paramètres

Le script [`orchestrate_emotyc_folder.py`](orchestrate_emotyc_folder.py) (avec l'option `--groups`) permet de faire une comparaison honnête en utilisant exactement la même configuration que celle ayant donné les résultats exposé dans la section 2.1. ci-dessus. On obtient donc :

|  | Rappel (Macro) | Précision (Macro) | Macro F1 |
| :--- | :---: | :---: | :---: |
| Présence d'une émotion | 0.63 | 0.63 | 0.63 |
| Mode d'expression | 0.25 | 0.34 | 0.28 |
| Type | 0.49 | 0.42 | 0.42 |
| Catégorie émotionnelle | 0.35 | 0.20 | 0.23 |

Performances détaillées par label :


> **Observation** : dans CyberAggAdo, les erreurs sont légèrement plus élevées sur les domaines Religion et Homophobie que sur Obésité et Racisme. Obésité étant un corpus plus grand, l'agrégation par tirage aléatoire donne des performances très légèrement inférieures.

*(Les illustrations SVG de cette section ont été retirées)*

### 4.8 Échantillons XLSX aléatoires contigus

**Configuration** : 4 échantillons de 50 unités textuelles contiguës extraites aléatoirement. Template BCA + contexte + seuil 0.5.

*(Les illustrations SVG de cette section ont été retirées)*

### 4.9 Échantillon XLSX  non contigus

**Configuration** : 120 unités non contiguës extraites aléatoirement. Sans contexte.

*(Les illustrations SVG de cette section ont été retirées)*

### 4.10 Écarts TTK vs. CyberAggAdo — avec contexte

Écarts par label : Δ = score(TTK) − score(CyberAggAdo). Un Δ positif indique une performance supérieure sur TTK.

$$\Delta = \text{score}_{TTK} - \text{score}_{CyberAggAdo}$$

*(Les illustrations SVG de cette section ont été retirées)*

### 4.11 Écarts TTK vs. CyberAggAdo — sans contexte

*(Les illustrations SVG de cette section ont été retirées)*




## 5. Remarques relatives à la configuration et aux hyperparamètres

Le script [`emotyc_predict.py`](emotyc_predict.py) reprend le template "BCA" (_Before, Current, After_) qui est utilisé lors du fine-tuning du modèle :

```txt
before:{previous_sentence}</s>current: {target_sentence}</s>after:{next_sentence}</s>
```

Lorsque l’option `--use-context` est activée, le script injecte dans le template BCA les phrases immédiatement voisines de la phrase cible : la phrase précédente est placée dans le champ `before`, la phrase courante dans le champ `current`, et la phrase suivante dans le champ `after`. Pour la première et la dernière ligne du fichier, lorsqu’il n’existe pas respectivement de phrase précédente ou suivante, le script remplace le contexte manquant par le token de fin de séquence `</s>`.

L'utilisation de ce template est documentée dans Étienne ([2023](https://theses.hal.science/tel-04210908v1/document), p. 141), dans Étienne et al. (2024, p. 5) (voir l'article sur [ArXiv](https://arxiv.org/pdf/2405.14385) ou sur [ACL](https://aclanthology.org/2024.wassa-1.14.pdf)), ainsi que dans le [README](https://huggingface.co/TextToKids/CamemBERT-base-EmoTextToKids) présent sur le dépôt Hugging Face du modèle. Cela est cohérent avec nos tests, dans lesquels ce template donne les meilleures performances sur le corpus TextToKids.

Par ailleurs, comme dans [l'implémentation officielle d'EMOTYC sur TextToKids](https://gitlab.huma-num.fr/texttokids/ttkwp3-2025/-/blob/main/text_complexity/server/src/processor/semantique/emotyc.py), nous désactivons l'ajout de tokens spéciaux :

```python
add_special_tokens=False
```

Nos tests montrent que les performances d'EMOTYC diminuent quand `add_special_tokens=True`, ce qui suggère que l'ajout de tokens spéciaux était bien désactivé pendant le fine-tuning. Avec `add_special_tokens=False`, le premier token de la séquence n’est pas le token spécial `<s>`, mais le premier token produit par la tokenisation du template BCA, qui correspond au fragment lexical `_be` (car `CamembertTokenizer` ajoute le préfixe `_` lorsqu’un mot est précédé d’un espace). L’état caché associé à ce token en position 0 à la 12e couche du modèle sert de représentation globale utilisée pour la classification.

D'autres tests montrent également que la configuration avec template BCA et `add_special_tokens=True` reste assez performante, bien qu’inférieure à la configuration sans tokens spéciaux. Cela suggère que, dans les deux cas, l'architecture Transformer parvient à diriger l'information pertinente vers la position 0 (qu'il s'agisse du token `_be` lorsque `add_special_tokens=False`, ou du token spécial `<s>` lorsque `add_special_tokens=True`).





### 5.1 Génération d'un rapport HTML

Convertit le fichier standardisé `emotyc_predictions_summary.json` (généré par les scripts d'inférence) en un rapport HTML lisible, avec possibilité de regrouper les métriques par dimension sémantique.

**Exemple d'utilisation :**
```bash
python visualizations/json_summary_to_html.py \
    --json ./results/mon_run/emotyc_predictions_summary.json \
    --out ./results/mon_run/rapport.html \
    --groups
```

### 5.2 Génération d'une Heatmap de transferabilité

Permet de générer une heatmap HTML comparant les performances entre TextToKids et CyberAggAdo.

**Exemple d'utilisation :**
```bash
python visualizations/delta_heatmap.py \
    --ttk ./results/TextToKids/.../emotyc_predictions_summary.json \
    --cyber ./results/orchestrated_emotyc_CyberAggAdo/.../emotyc_predictions_summary.json \
    --out ./results/heatmap.html
```

### 5.3 Contiguité et non-contiguité

Lance l'inférence de manière séquentielle sur plusieurs fichiers Excel, puis fusionne tous les résultats dans un unique dossier `merged`.

L'objectif principal de cet orchestrateur est de préserver l'intégrité du contexte BCA. Si on produit un XLSX qui résulte d'une concaténation, puis qu'on utilise l'option `--use-context`, une phrase située à la fin d'un fichier XLSX se retrouve injectée comme contexte "before" de la première phrase du fichier XLSX suivant. C'est pourquoi il faut lançer le script d'inférence sur chaque bloc ou fichier individuel. L'échantillonnage contigu doit, par défaut, tirer un bloc de taille 50. Si l'indice de départ est `i` (tiré dans l'intervalle `[0 ; len(xlsx) - 50]`), le bloc sélectionné va de `i` jusqu’à `i + 50` exclu.

Si les phrases sont mélangées ou sélectionnées aléatoirement, il faut être très prudent avec `--use-context`. Le contexte `before`/`after` suivrait alors l’ordre du sous-ensemble et non pas les vraies phrases voisines du document source. La recommandation est donc de ne pas utiliser **`--use-context`** pour tout échantillonnage non contigu.

**Exemple d'utilisation :**
```bash
# Générer des sous-ensembles (par défaut dans ./results/prepared_xlsx_samples/subsets)
python prepare_xlsx_samples.py

# Lancer l'inférence (utilise le contexte par défaut sur le dossier généré ci-dessus)
python orchestrate_emotyc_folder.py --groups
```


## 6. Remarques relatives à l'optimisation des scripts d'inférence

Les scripts d'inférence utilisent ONNX Runtime et la bibliothèque Rust `tokenizers` :
- ONNX Runtime applique des optimisations de graphe (fusions de nœuds, élimination de sous-graphes redondants) grâce à `ORT_ENABLE_ALL`.
- Le chargement utilise `CUDAExecutionProvider` si un GPU CUDA est détecté, avec un repli automatique et performant sur `CPUExecutionProvider` le cas échéant.
- Les paramètres de mémoire `enable_cpu_mem_arena` et `enable_mem_pattern` sont activés pour réduire l'allocation dynamique de mémoire lors de l'inférence.
- Le parallélisme interne est contrôlé via `intra_op_num_threads` (2 par défaut) pour limiter l'utilisation CPU excédentaire.

## 7. Reproductibilité et commandes utilisées

Inférence sur les fichiers XLSX du gold (TextToKids ou CyberAggAdo) :

```bash
# Inférence rapide (par défaut avec le modèle model_onnx/)
python emotyc_predict.py --xlsx ./golds/emotexttokids_gold_flat.xlsx --out_dir ./results/TTK/NoContextTemplateMode05

# Inférence avec contexte
python emotyc_predict.py --xlsx ./golds/emotexttokids_gold_flat.xlsx --out_dir ./results/TTK/ContextTemplateMode05 --use-context
```

Pour obtenir un rapport détaillé (divergences et matrices de confusion par dimensions) :

```bash
# Rapport détaillé (contexte + bca + seuil modes à 0.06)
python emotyc_predict_details.py \
    --xlsx ./golds/emotexttokids_gold_flat.xlsx \
    --out_dir ./results/TTK/ContextTemplateAvecEspaceMode006 \
    --use-context \
    --template bca \
    --mode-threshold 0.06
```

Pour orchestrer l'inférence sur tout un dossier de fichiers XLSX :

```bash
# Évaluation sur un dossier (charge le modèle ONNX une seule fois)
python orchestrate_emotyc_folder.py ./results/prepared_xlsx_samples/subsets
```

