# Slides Beamer pour la heatmap de transferabilité

Ce dossier contient une version Beamer de la visualisation
`results/heatmap_delta.html`.

## Fichiers

- `heatmap_delta_beamer.tex` : présentation Beamer principale.
- `make_heatmap_assets.py` : régénère les assets vectoriels depuis les JSON.
- `figures/heatmap_delta_table.tex` : tableau LaTeX natif inclus dans Beamer.
- `figures/heatmap_delta.svg` : version SVG autonome de la heatmap.

## Régénérer les assets

Depuis la racine du dépôt :

```bash
python3 slides/make_heatmap_assets.py
```

## Compiler la présentation

Depuis la racine du dépôt :

```bash
pdflatex slides/heatmap_delta_beamer.tex
```

Ou depuis `slides/` :

```bash
pdflatex heatmap_delta_beamer.tex
```

La slide de heatmap utilise un tableau LaTeX natif : le rendu final dans le
PDF est vectoriel, sans capture PNG.

## Option SVG vers PDF

Si vous préférez insérer la heatmap comme figure externe, convertissez le SVG :

```bash
inkscape slides/figures/heatmap_delta.svg \
  --export-filename=slides/figures/heatmap_delta.pdf
```
