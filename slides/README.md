# Slides Beamer pour la heatmap de transferabilité

Ce dossier contient une version Beamer de la visualisation
`results/heatmap_delta.html`.

## Fichiers

- `heatmap_delta_beamer.tex` : présentation Beamer principale.
- `make_heatmap_assets.py` : régénère les assets vectoriels depuis les JSON.
- `figures/heatmap_delta_table_compact.tex` : tableau LaTeX natif inclus dans Beamer.
- `figures/heatmap_delta_compact.svg` : version SVG compacte.

## Régénérer les assets

Depuis la racine du dépôt :

```bash
python3 slides/make_heatmap_assets.py
```

Par défaut, la version utilisée dans Beamer garde 5 lignes : les 4 labels les
plus représentés dans CyberAggAdo, plus le plus fort décrochage parmi les
labels avec au moins 50 occurrences positives. Pour changer ces paramètres :

```bash
python3 slides/make_heatmap_assets.py --min-cyber-support 30 --max-compact-labels 6
```

## Compiler la présentation

Depuis la racine du dépôt :

```bash
./slides/compile_beamer.sh
```

Le script compile dans un dossier temporaire, copie uniquement le PDF final vers
`slides/heatmap_delta_beamer.pdf`, puis supprime les fichiers auxiliaires
(`.aux`, `.log`, `.nav`, etc.) si la compilation réussit. En cas d'erreur, le
dossier temporaire est conservé pour inspecter les logs.

Pour écrire le PDF ailleurs :

```bash
OUTPUT_PDF=/tmp/heatmap_delta_beamer.pdf ./slides/compile_beamer.sh
```

La slide de heatmap utilise un tableau LaTeX natif filtré : le rendu final
dans le PDF est vectoriel, sans capture PNG.

## Option SVG vers PDF

Si vous préférez insérer la heatmap comme figure externe, convertissez le SVG :

```bash
inkscape slides/figures/heatmap_delta_compact.svg \
  --export-filename=slides/figures/heatmap_delta_compact.pdf
```
