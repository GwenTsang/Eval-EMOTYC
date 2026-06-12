# Rapport de performance EMOTYC sur CyberAggAdo

Ce rapport synthétise les résultats présents dans `results/`. L'objectif n'est pas seulement de constater une baisse de score, mais d'expliquer pourquoi le modèle se trompe quand il est appliqué aux messages de cyberagression adolescente.

## 1. Ce que le modèle a appris

EMOTYC est un CamemBERT-base fine-tuné pour produire 19 sorties binaires au niveau de la phrase : `Emo`, 4 modes d'expression, `Base`/`Complexe`, et 12 catégories émotionnelles. Etienne et al. (2024) précisent que le corpus d'origine contient 1 594 textes français destinés aux enfants de 6 à 14 ans, soit environ 28k phrases et 515k mots. Il s'agit surtout de textes journalistiques jeunesse (91 % des phrases), avec des articles encyclopédiques (9 %) et très peu de roman (1 %).

Le modèle a donc été entraîné sur des textes écrits, édités et non conversationnels. Ainsi, d'après Etienne et al. (2024), la tâche vise les émotions dans des textes narratifs, journalistiques ou explicatifs, pas l'état affectif de locuteurs dans des chats. Les annotations sont d'abord faites au niveau du segment émotionnel, puis aplaties au niveau de la phrase en un vecteur de 19 booléens. Cet aplatissement supprime le lien direct entre une catégorie et son mode : une phrase peut avoir `Colere=1`, `Degout=1`, `Montree=1`, `Suggeree=1`, sans indiquer quelle émotion est montrée et laquelle est suggérée.

Architecture et entraînement :

- CamemBERT-base, 110M paramètres, 12 couches, pré-entraîné sur 138 Go de français.
- Remplacement de la tête finale par une tête multi-label avec binary cross-entropy.
- Fine-tuning en deux temps : 3 epochs sur la seule tâche `Emo`, puis 6 epochs sur les 19 labels.
- Adam, learning rate `1e-5`, batch size 8, pondération de classes plafonnée à 50.
- Entrée attendue : `before:{previous}</s>current:{target}</s>after:{next}</s>`.

Conséquence directe : EMOTYC connaît bien les marqueurs émotionnels de textes propres et contextualisés, mais il n'a pas appris la pragmatique des interactions courtes, bruitées et insultantes de CyberAggAdo.

## 2. Résultats globaux

| Résultat | N | Template | Macro-F1 | Micro-F1 | Exact match |
|---|---:|---|---:|---:|---:|
| TextToKids, avec contexte | 27 911 | `bca_spaced_context` | 0.739 | 0.897 | n.d. |
| TextToKids, sans contexte | 27 911 | `bca_spaced_no_context` | 0.684 | 0.850 | n.d. |
| CyberAggAdo complet, détaillé | 781 | `bca_context` | 0.291 | 0.472 | 0.346 |
| CyberAggAdo complet, orchestré | 781 | `bca_spaced_context` | 0.282 | 0.472 | n.d. |
| Sous-échantillons contigus | 200 | `bca_spaced_context` | 0.270 | 0.470 | n.d. |
| Random sample 120 | 120 | `bca_spaced_context` | 0.247 | 0.492 | 0.283 |

Le point important est la robustesse de l'effondrement : la version détaillée et la version orchestrée de CyberAggAdo donnent quasiment le même micro-F1 autour de 0.47. Les sous-échantillons contigus et l'échantillon aléatoire confirment que le problème n'est pas un artefact d'un seul fichier.

Attention toutefois au `random_sample_120` : le résumé et le JSONL indiquent `bca_spaced_context`, alors que l'échantillon est non contigu. Les champs `text_prev` et `text_next` existent donc, mais ils suivent l'ordre de l'échantillon et non nécessairement l'ordre conversationnel réel. Ce résultat est utile comme test de robustesse, mais pas comme mesure propre de l'effet du contexte.

## 3. Chute par famille de labels

Comparaison principale : TextToKids avec contexte vs CyberAggAdo complet détaillé.

| Groupe | F1 TTK | F1 Cyber | Précision Cyber | Rappel Cyber | Support Cyber |
|---|---:|---:|---:|---:|---:|
| `Emo` | 0.930 | 0.657 | 0.626 | 0.692 | 399 |
| Modes | 0.881 | 0.266 | 0.326 | 0.234 | 446 |
| Types | 0.882 | 0.418 | 0.425 | 0.489 | 378 |
| Catégories émotionnelles | 0.652 | 0.248 | 0.235 | 0.344 | 499 |
| Tous les 19 labels | 0.739 | 0.291 | 0.294 | 0.354 | 1 722 |

Le modèle ne s'effondre pas uniformément. Il conserve une capacité moyenne à dire qu'une phrase est émotionnelle (`Emo` F1 0.657), mais il ne sait plus qualifier correctement cette émotion : les modes tombent à 0.266 de macro-F1 et les catégories à 0.248.

Cela indique une erreur de granularité. Le modèle repère souvent une charge affective ou agressive, mais il ne parvient pas à choisir la bonne catégorie ni le bon mode dans le registre cyberharcèlement.

## 4. Résultats par label

| Label | Gold + | Pred + | TP | FP | FN | F1 | Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---|
| `Emo` | 399 | 441 | 276 | 165 | 123 | 0.657 | Détection globale moyenne, mais beaucoup de FP/FN. |
| `Colere` | 290 | 243 | 127 | 116 | 163 | 0.477 | Label central, mais confondu avec agressivité et menace. |
| `Degout` | 80 | 0 | 0 | 0 | 80 | 0.000 | Le modèle ne prédit jamais le dégoût. |
| `Autre` | 62 | 138 | 13 | 125 | 49 | 0.130 | Forte sur-prédiction, précision très faible. |
| `Montree` | 301 | 279 | 133 | 146 | 168 | 0.459 | Très présent dans CyberAggAdo, mais bruité. |
| `Suggeree` | 54 | 17 | 3 | 14 | 51 | 0.085 | Quasi-incapacité à inférer l'émotion situationnelle. |
| `Comportementale` | 36 | 18 | 4 | 14 | 32 | 0.148 | Menaces/actions mal reconnues comme comportements. |
| `Designee` | 55 | 42 | 18 | 24 | 37 | 0.371 | Termes explicites moins stables que sur TTK. |
| `Base` | 364 | 268 | 174 | 94 | 190 | 0.551 | Les catégories de base sont souvent ratées malgré `Emo`. |
| `Complexe` | 14 | 35 | 7 | 28 | 7 | 0.286 | Support faible et sur-prédiction. |
| `Joie` | 30 | 25 | 11 | 14 | 19 | 0.400 | Souvent déclenchée par rire/blague, parfois ironique. |
| `Tristesse` | 14 | 23 | 3 | 20 | 11 | 0.162 | Signal faible, sur-prédiction et sous-détection. |

Les labels rares (`Culpabilite`, `Jalousie`) ne sont jamais prédits, comme déjà observé dans le papier sur certaines classes rares du domaine source. Mais l'échec le plus grave ici concerne `Degout` : il est fréquent dans CyberAggAdo (80 occurrences, 10.2 % des lignes), alors qu'il est totalement absent des prédictions.

## 5. Distribution des erreurs

Sur les 781 lignes du JSONL complet :

- 270 lignes seulement sont exactes sur les 19 labels, soit 34.6 %.
- 59 lignes ont 1 erreur, 96 ont 2 erreurs.
- 356 lignes ont au moins 3 erreurs, soit 45.6 %.
- La moyenne est de 2.23 erreurs par ligne sur 19 labels.
- Sur les seuls labels catégories + modes, l'exact match tombe aussi à 36.0 %.

Le modèle ne fait donc pas seulement des erreurs ponctuelles sur des labels rares. Près d'une ligne sur deux accumule plusieurs erreurs, ce qui montre une mauvaise représentation globale du registre.

## 6. Cohérence structurelle

Les 19 sorties sont prédites indépendamment. Cela produit des incohérences logiques :

| Violation | Nombre | Taux |
|---|---:|---:|
| Au moins une violation | 190 | 24.3 % |
| Émotion prédite sans mode prédit | 124 | 15.9 % |
| Émotion de base prédite mais `Base=0` | 38 | 4.9 % |
| `Emo=1` sans aucune catégorie émotionnelle | 31 | 4.0 % |
| Mode prédit sans catégorie émotionnelle | 22 | 2.8 % |
| Catégorie émotionnelle prédite mais `Emo=0` | 18 | 2.3 % |

La violation dominante est "émotion sans mode". C'est cohérent avec la chute des modes : EMOTYC détecte souvent qu'il se passe quelque chose d'émotionnel, mais il ne sait pas comment cette émotion est exprimée dans des messages courts, sarcastiques ou insultants.

## 7. Domaines CyberAggAdo

| Domaine | N | All19 macro-F1 | All19 micro-F1 | Exact match | Catégories macro-F1 | Modes macro-F1 | `Emo` F1 positif |
|---|---:|---:|---:|---:|---:|---:|---:|
| Homophobie | 103 | 0.149 | 0.469 | 0.184 | 0.052 | 0.257 | 0.641 |
| Obésité | 373 | 0.205 | 0.481 | 0.391 | 0.119 | 0.248 | 0.637 |
| Racisme | 201 | 0.251 | 0.408 | 0.358 | 0.206 | 0.211 | 0.616 |
| Religion | 104 | 0.344 | 0.541 | 0.317 | 0.297 | 0.291 | 0.788 |

Homophobie est le plus difficile pour les catégories émotionnelles. Religion obtient les meilleurs scores relatifs, mais reste faible. Les différences par domaine doivent être lues avec prudence parce que certaines catégories ont très peu d'occurrences, mais elles confirment que le problème est transversal.

## 8. Pourquoi le modèle se trompe

### 8.1. Changement de domaine massif

Le modèle a appris sur des textes édités destinés aux enfants. CyberAggAdo contient des conversations adolescentes, des tours elliptiques, des insultes, des abréviations, du sarcasme, des messages mono-syntagmatiques et des réponses dépendantes du rôle interactionnel. Le papier positionne explicitement EMOTYC hors du cadre conversationnel : il vise les émotions de personnages ou d'événements dans des textes écrits, pas la dynamique sociale d'un fil de cyberharcèlement.

Ce changement de domaine explique la forme des erreurs :

- le modèle sur-détecte parfois l'émotion globale dans un contexte agressif ;
- il rate des émotions implicites portées par l'acte social, par exemple l'exclusion, la menace ou la dévalorisation ;
- il confond le rôle pragmatique du message avec la catégorie émotionnelle.

### 8.2. Agressivité et émotion ne sont pas équivalentes

Dans CyberAggAdo, une phrase peut être agressive sans être annotée comme émotionnelle, ou émotionnelle sans contenir un mot émotionnel explicite. EMOTYC tend à projeter l'agressivité vers `Colere`, `Autre` ou `Montree`.

Exemples de faux positifs très confiants :

- `le mec il a un nom de films de clichés américain et il se permet de critiquer le culot` : `Colere` prédite à 0.999879, gold sans catégorie émotionnelle.
- `y'a rien à assumer, puisqu'y a r` : `Colere` prédite à 0.998299, probablement contaminée par le contexte homophobe adjacent.
- `je rigole roh` : `Joie` et `Montree` prédites, alors que le gold n'active aucune émotion. Le modèle lit "rigole" littéralement et ne met pas en contexte (même si, je note que cette entrée aurait pu être annotée "joie" + "comportementale").

Inversement, des insultes courtes ou des menaces explicites sont parfois ignorées :

- `bouffon` : gold `Autre` + `Colere`, probabilité `Emo=0.000140`.
- `un gros tas de graisse arabe` : gold `Degout`, probabilité `Emo=0.000163`.
- `ca va chauffer` : gold `Colere` + `Suggeree`, probabilité `Colere=0.000012`.

Le modèle ne possède donc pas une représentation stable de l'insulte comme acte discursif. Selon les cas, il sur-interprète ou ignore complètement le signal.

### 8.3. `Degout` est invisible

Le problème le plus net est `Degout` :

- 80 occurrences dans le gold CyberAggAdo ;
- 0 prédiction positive à seuil 0.5 ;
- F1 = 0.000 ;
- même avec un seuil optimisé sur CyberAggAdo, le meilleur F1 n'atteint que 0.309.

La distribution des probabilités montre que les positifs gold de `Degout` restent presque toujours très bas : médiane 0.0033, 75e percentile 0.0179, maximum 0.0812. Le modèle ne place donc pas les cas de dégoût cyberharcèlement près du seuil.

Cela vient probablement de deux facteurs combinés :

- Dans TextToKids, `Degout` est extrêmement rare et déjà mal appris.
- Dans CyberAggAdo, le dégoût est souvent exprimé par dévalorisation corporelle, saleté morale, rejet social ou insulte identitaire, pas par des formes lexicales proches de "dégoûtant", "beurk", etc.

Exemples de faux négatifs :

- `parce qu'il est gros` : gold `Colere` + `Degout`, probabilité `Degout=0.000002`.
- `en même temps vu comme il est gros` : gold `Autre` + `Degout`, probabilité `Degout=0.000002`.
- `pierre il pu, il pète, il polue la planète` : gold `Degout`, probabilité `Degout=0.000002`.

### 8.4. Les modes implicites s'effondrent

`Suggeree` et `Comportementale` sont les labels de mode les plus problématiques :

| Mode | Gold + | F1 à 0.5 | Meilleur F1 après sweep | Interprétation |
|---|---:|---:|---:|---|
| `Suggeree` | 54 | 0.085 | 0.225 | L'inférence situationnelle cyber est hors distribution. |
| `Comportementale` | 36 | 0.148 | 0.377 | Les menaces/actions ne ressemblent pas aux comportements appris. |
| `Designee` | 55 | 0.371 | 0.484 | Les termes explicites existent, mais sont bruités ou ironiques. |
| `Montree` | 301 | 0.459 | 0.565 | Le mode est fréquent, mais les indices de surface sont ambigus. |

Etienne et al. (2024) signale déjà que les émotions suggérées sont difficiles sur le domaine source. Dans CyberAggAdo, elles exigent souvent de comprendre une situation sociale : menace, humiliation, exclusion, défense de la victime, sous-entendu sexiste/raciste/homophobe, etc. Ce n'est pas le même type de "situation suggérée" que dans des textes narratifs ou journalistiques.

### 8.5. Le contexte BCA peut devenir du bruit

Le template BCA a été conçu pour des phrases adjacentes d'un même texte. Dans CyberAggAdo, les phrases voisines sont souvent des tours de parole de personnes différentes, avec changement de rôle, cible ou intention. Le contexte peut alors contaminer la phrase cible.

Exemple :

- Cible : `taimes les filles`
- Contexte précédent : `bande de crasseuse`
- Contexte suivant : `y'a rien à assumer...`
- Gold cible : aucune émotion
- Prédiction : `Autre` avec probabilité 0.999706

Le contexte aide sur TextToKids : macro-F1 0.684 sans contexte contre 0.739 avec contexte. Mais les sanity checks consultés dans `EMOTYC/Documentation/sanity_checks.md` indiquent qu'en OOD le contexte dégrade la cohérence structurelle. Il faut donc distinguer deux fonctions du contexte : utile dans des textes continus et édités, risqué dans des dialogues agressifs multi-locuteurs.

### 8.6. Bruit lexical, argot et abréviations

Le notebook `Relations_entre_le_rôle_et_l'émotion_détectée_+_détection_des_chaînes_de_caractères_sémantiquement_pauvres.ipynb` contient une analyse de fragmentation CamemBERT. Sur le corpus cyber complet, il relève environ 5 003 formes uniques, 4 943 formes retenues avec au moins deux caractères, et 1 041 formes suspectes avec un ratio caractères/sous-tokens très bas. Parmi les exemples qui ressortent : `tg`, `ftg`, `ntm`, `ptn`, `vrm`, `wlh`, `tjrs`, `jsp`, `frr`, etc.

Cette analyse ne prouve pas seule la causalité, mais elle correspond aux erreurs observées. Les abréviations et formes conversationnelles portent beaucoup de sens pragmatique dans CyberAggAdo, mais elles sont mal segmentées ou peu représentées par CamemBERT.

Les variables du gold confirment l'effet du registre :

| Sous-groupe | N | Exact match 19 labels | Erreurs moyennes | Commentaire |
|---|---:|---:|---:|---|
| `abréviation=1` | 83 | 0.120 | 3.145 | Très fort marqueur d'échec. |
| `argot=1` | 156 | 0.244 | 2.731 | Forte baisse vs lignes sans argot. |
| `insulte=1` | 200 | 0.235 | 2.795 | Les insultes augmentent le nombre d'erreurs. |
| `mépris / haine=1` | 344 | 0.253 | 2.555 | Confusion entre émotion, haine et attaque sociale. |
| `ironie=1` | 57 | 0.228 | 2.561 | Sur-détection de `Emo` : préd. 0.737 vs gold 0.421. |
| `elongation=1` | 17 | 0.059 | 2.882 | Petit support, mais très mauvais exact match. |

Ces effets doivent être lus comme des corrélations descriptives, pas comme une preuve causale isolée. Ils pointent cependant vers le même mécanisme : la langue numérique adolescente est hors distribution.

### 8.7. Les seuils ne suffisent pas

Un sweep de seuils améliore certains F1, mais ne restaure pas les performances source :

| Label | F1 à 0.5 | Meilleur F1 | Seuil optimal approximatif |
|---|---:|---:|---:|
| `Colere` | 0.477 | 0.556 | 0.000227 |
| `Degout` | 0.000 | 0.309 | 0.008158 |
| `Autre` | 0.130 | 0.162 | 0.033830 |
| `Montree` | 0.459 | 0.565 | 0.000013 |
| `Suggeree` | 0.085 | 0.225 | 0.024581 |
| `Comportementale` | 0.148 | 0.377 | 0.110000 |
| `Emo` | 0.657 | 0.692 | 0.000244 |

Ces seuils optimaux très bas montrent une mauvaise calibration OOD. Pour `Colere`, `Montree`, `Emo` et `Base`, le meilleur F1 est obtenu en activant massivement les labels, ce qui augmente le rappel mais dégrade la précision. Pour `Autre`, même le meilleur seuil reste presque inutile. Pour `Degout`, le seuil aide un peu mais le classement reste mauvais.

## 9. Lecture des erreurs par rôle et agressivité

Les variables CyberAggAdo aident à comprendre les erreurs :

- `CAG` (agression couverte) : gold `Emo=1` dans 28.0 % des lignes, mais prédiction `Emo=1` dans 51.6 %. Le taux de faux positifs `Emo` monte à 35.5 %. Le modèle emotionalise des stratégies rhétoriques ou sarcastiques qui ne sont pas toujours annotées comme émotion.
- `OAG` (agression ouverte) : gold `Emo=1` dans 65.8 %, prédiction 59.7 %, avec 22.0 % de faux négatifs `Emo`. Le modèle rate donc aussi des agressions explicites, surtout quand elles sont courtes, argotiques ou très insultantes.
- Sentiment `NEG` : exact match 0.293 contre 0.511 pour `NEU`. Les messages négatifs concentrent les cas difficiles.
- Les rôles `bully` et `bully_support` ont des F1 plus faibles sur les catégories que `victim` ou `victim_support`, ce qui suggère que les productions offensives sont plus hors distribution que les réponses de défense.

Le modèle ne sait pas utiliser explicitement `ROLE`, `HATE`, `INTENTION` ou `SENTIMENT`. Or ces variables sont essentielles pour distinguer une attaque, une défense, un sarcasme, une menace, une dévalorisation ou un commentaire neutre.

## 10. Synthèse : les causes principales

1. **Domain shift textuel** : EMOTYC a appris sur des textes jeunesse édités, pas sur des conversations adolescentes bruitées.

2. **Domain shift pragmatique** : dans CyberAggAdo, l'émotion est souvent dans l'acte social, pas dans la phrase seule. Une insulte, une menace ou une défense exige de connaître le rôle du locuteur, la cible et le fil interactionnel.

3. **Labels rares mal appris** : `Degout`, `Culpabilite`, `Jalousie` sont faibles ou absents dans les prédictions. Pour `Degout`, le problème devient critique car CyberAggAdo en contient beaucoup.

4. **Confusion agression/émotion** : EMOTYC transforme parfois l'agressivité en `Colere` ou `Autre`, mais ignore d'autres agressions très explicites quand elles sont courtes ou argotiques.

5. **Modes d'expression non transférables** : `Suggeree` et `Comportementale` changent de nature entre les textes source et les conversations de harcèlement.

6. **Bruit lexical et tokenizer** : abréviations, argot, allongements et graphies non standard portent le signal, mais sont mal représentés par CamemBERT.

7. **Contexte mal aligné** : le BCA est adapté à des textes continus ; dans un chat multi-locuteurs, les phrases adjacentes peuvent apporter du bruit ou une émotion d'un autre locuteur.

8. **Têtes indépendantes** : les sorties ne respectent pas toujours les contraintes logiques du schéma (`Emo`, `Base`, `Complexe`, modes), ce qui amplifie les erreurs.

## 11. Recommandations

1. **Créer un split de calibration CyberAggAdo** pour fixer des seuils par label. Les seuils à 0.5 sont clairement inadaptés, mais le sweep montre qu'une calibration seule ne suffira pas.

2. **Appliquer un post-traitement logique** :
   - si une catégorie émotionnelle est prédite, forcer `Emo=1` ;
   - si une émotion de base est prédite, forcer `Base=1` ;
   - si une émotion complexe est prédite, forcer `Complexe=1` ;
   - si une émotion est prédite sans mode, activer le mode de probabilité maximale ou renvoyer une alerte d'incertitude.

3. **Évaluer explicitement avec et sans contexte sur CyberAggAdo**, en respectant les frontières conversationnelles et les changements de locuteur. Le contexte ne doit pas être utilisé sur un échantillon non contigu.

4. **Ajouter une normalisation contrôlée du langage numérique** : dictionnaire d'abréviations (`tg`, `ftg`, `vrm`, `ptn`, etc.), réduction d'élongations, correction légère des graphies fréquentes, tout en conservant les signaux expressifs utiles.

5. **Fine-tuner ou adapter EMOTYC sur CyberAggAdo**. Les catégories `Degout`, `Autre`, `Colere`, `Montree`, `Suggeree` nécessitent des exemples cyber spécifiques.

6. **Réviser les guidelines pour le domaine cyber** sur les frontières `Colere` / `Degout` / `Autre` / absence d'émotion. Plusieurs erreurs reflètent aussi une tension d'annotation : l'agression verbale n'est pas toujours une émotion, mais elle porte souvent mépris, haine, rejet ou dégoût.

7. **Exploiter les variables conversationnelles** (`ROLE`, `HATE`, `INTENTION`, `TARGET`, `SENTIMENT`) dans un modèle complémentaire ou une couche de décision. Ces variables expliquent des différences d'erreurs que la phrase seule ne peut pas résoudre.

## 12. Conclusion

EMOTYC ne généralise pas correctement à CyberAggAdo. Il reste capable de détecter partiellement la présence globale d'émotion, mais il échoue sur la catégorie fine, le mode d'expression, la distinction entre émotion et agression, et les signaux pragmatiques propres au cyberharcèlement adolescent.

Le modèle ne se trompe pas seulement parce que les labels sont rares ou parce que le seuil est mal choisi. Il se trompe parce que les indices pertinents dans CyberAggAdo ne sont pas ceux appris dans TextToKids. Dans TextToKids, les émotions sont souvent portées par des structures linguistiques éditées, narratives ou explicatives. Dans CyberAggAdo, elles sont portées par des actes de parole, des rôles sociaux, des insultes, de l'implicite conversationnel, des abréviations et des graphies non standard.

Pour obtenir de bonnes performances, il faudra probablement une adaptation de domaine ou un modèle spécialisé cyberharcèlement.

## 13. Enrichissement par `error_analysis.py`, SHAP et FP-Growth

Le pipeline `EMOTYC/experimentations/error_analysis.py` a été relancé sur `EMOTYC/data/CyberAdoAgg_gold_global_total_latest.xlsx`, avec `context=no` et les seuils émotionnels optimisés. Les sorties sont dans `EMOTYC/experimentations/error_analysis_results/`. Les dépendances `shap` et `mlxtend` sont maintenant actives : le script produit `plots/shap_summary.png`, `plots/shap_bar.png`, `structured/explainability/shap_mean_abs.csv` et `association_rules_high_error.csv`.

### 13.1. Signaux descriptifs confirmés

Les erreurs EMOTYC restent surtout associées aux messages agressifs, insultants, méprisants/haineux, négatifs, longs, et aux formes non standard. La relation la plus structurante demeure la densité émotionnelle : Spearman ρ = 0.72, p ≈ 9.9e-126. L'erreur moyenne passe de 0.016 pour les lignes sans émotion gold à 0.087 pour densité 1, 0.155 pour densité 2 et 0.201 pour densité 3.

| Signal | Lecture principale |
|---|---|
| `HATE=OAG` | erreur moyenne 0.078, contre 0.043 pour `CAG` et 0.038 pour `NAG`; surreprésenté dans les lignes high-error : 74.8 % vs 39.7 % des exact-match. |
| `mépris / haine=1` | erreur moyenne 0.078 vs 0.047 sans mépris/haine; 67.1 % des high-error vs 35.2 % des exact-match. |
| `insulte=1` | erreur moyenne 0.087 vs 0.051; 41.3 % des high-error vs 16.0 % des exact-match. |
| `SENTIMENT=NEG` | erreur moyenne 0.069 vs 0.034 pour `NEU`; 87.4 % des high-error vs 64.5 % des exact-match. |
| `text_long` | signal le plus contrastif en support : 60.8 % des high-error vs 22.4 % des exact-match. |
| `abréviation=1`, `argot=1`, `elongation=1` | erreurs moyennes 0.090, 0.080 et 0.127; l'élongation a un petit support (17 lignes), mais reste un amplificateur visible dans les interactions bivariées. |

Les interactions bivariées vont dans le même sens : les plus forts écarts d'erreur impliquent `HATE × elongation`, `elongation × abréviation`, `INTENTION × elongation`, `ROLE × elongation` et `elongation × mépris / haine`. Les formes expressives/non standard deviennent donc particulièrement problématiques quand elles apparaissent dans un contexte agressif.

### 13.2. Apport de SHAP

Le Random Forest explicatif obtient un OOB R² de 0.371. Les importances MDI et SHAP convergent sur quelques familles de variables : longueur du texte, mépris/haine, agression ouverte, cible/rôle et type d'abus verbal.

| Rang SHAP | Feature | mean \|SHAP\| | Lecture |
|---:|---|---:|---|
| 1 | `mépris / haine` | 0.0059 | Confirme que la haine/mépris pèse dans l'erreur émotionnelle. |
| 2 | `text_length` | 0.0059 | Les messages longs concentrent plus de configurations émotionnelles et d'actes discursifs. |
| 3 | `word_count` | 0.0056 | Même effet que la longueur brute, sous une autre forme. |
| 4 | `HATE_OAG` | 0.0021 | Les agressions ouvertes sont plus difficiles que les contenus neutres ou couverts. |
| 5 | `TARGET_victim` | 0.0011 | La cible interactionnelle contribue, mais moins que longueur/agressivité. |

### 13.3. Apport et limites de `mlxtend`

`mlxtend` produit 17 034 règles FP-Growth dans `association_rules_high_error.csv`. Le sous-ensemble miné correspond aux lignes avec `hamming_12 > 1/12`, soit 143 lignes sur 781. Ces règles sont utiles pour décrire des profils récurrents dans les erreurs, mais elles ne sont pas directement des facteurs de risque : le minage est fait uniquement dans la sous-population high-error, sans contraste intégré contre les lignes correctes.

Les premières règles par lift illustrent surtout des dépendances internes aux variables CyberAggAdo :

| Règle | Support | Confiance | Lift | Interprétation |
|---|---:|---:|---:|---|
| `HATE=NAG ∧ text_long` → `SENTIMENT=NEU` | 0.084 | 0.706 | 6.73 | Association de codage neutre; pas un signal d'erreur en soi. |
| `SENTIMENT=NEU` → `HATE=NAG` | 0.105 | 1.000 | 5.50 | Tautologie descriptive du corpus : le neutre est non agressif. |
| `CONTEXT=ATK ∧ ROLE=victim` → `INTENTION=DFN ∧ text_long` | 0.084 | 1.000 | 4.61 | Profil conversationnel de défense de victime. |
| `INTENTION=DFN ∧ ROLE=victim_support ∧ VERBAL_ABUSE=OTH` → `TARGET=bully` | 0.084 | 1.000 | 4.33 | Profil de défense/support dirigé vers l'agresseur. |
| `TARGET=bully ∧ mépris / haine=1` → `INTENTION=DFN ∧ ROLE=victim_support ∧ SENTIMENT=NEG` | 0.091 | 0.684 | 4.25 | Cas plus interprétable : défense négative et mépris contre le bully. |

La lecture la plus fiable ne vient donc pas du lift brut des règles, mais du contraste high-error vs exact-match. Sur ce contraste, les profils réellement surreprésentés dans les erreurs sont `text_long`, `HATE=OAG`, `mépris / haine=1`, `insulte=1`, `SENTIMENT=NEG`, `VERBAL_ABUSE=DNG`, `argot=1`, `TARGET=victim/bully`, `INTENTION=DFN/ATK` et certaines formes `nature_linguistique` comme `Syntagme nominal`, `Enonce exclamatif`, `Proposition` ou `Interjection`.

Pour rendre `mlxtend` plus diagnostique, il faudrait modifier l'analyse en ajoutant explicitement un item `HIGH_ERROR=1` et en filtrant les règles qui l'ont comme conséquent, ou calculer des ratios de support high-error vs exact-match pour chaque itemset. Il faudrait aussi supprimer les items constants (`domain=CyberAggAdo`) et limiter les règles qui ne font que reproduire la logique d'annotation entre `HATE`, `SENTIMENT`, `ROLE`, `TARGET`, `INTENTION` et `CONTEXT`.

### 13.4. Synthèse de l'enrichissement

SHAP confirme que le modèle explicatif retrouve les mêmes mécanismes que l'analyse descriptive : l'erreur augmente avec la complexité textuelle, l'agressivité ouverte, le mépris/haine et certains profils interactionnels. `mlxtend` donne des profils de cooccurrence, mais son résultat brut est surtout descriptif. L'enrichissement renforce donc la conclusion du rapport : EMOTYC échoue moins par un simple mauvais seuil que par inadéquation de domaine, parce que les indices pertinents dans CyberAggAdo sont pragmatiques, interactionnels et bruités.