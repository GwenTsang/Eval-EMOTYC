# Documents LaTeX

Ce dossier contient les documents LaTeX du projet (rapport de résumé de stage et présentation Beamer) ainsi que les fichiers nécessaires à leur compilation.

## Contenu

- `stage_resume.tex` : le fichier source principal du résumé de stage.
- `stage_resume.pdf` : le PDF compilé du résumé de stage.
- `template/` : dossier contenant les modèles de style et configurations pour le résumé (`stageOnera.cls`, `onera.bst`, `references.bib`).
- `beamer_main.tex` : le fichier source principal des slides Beamer.
- `beamer_main.pdf` : le PDF compilé des slides Beamer.
- `baseline_comparison_beamer_draft.tex` : brouillon de comparaison des baselines pour les slides.
- `compile.sh` : script de compilation pour les documents LaTeX.
- `figures/` : dossier contenant les figures et sous-dossiers (`logos/` - contenant le logo ONERA et l'icône GitHub, `plots/`, `schemas/`, `tables/`) utilisés par les documents.

## Compilation

Le script `compile.sh` permet de compiler les documents. Il utilise `latexmk` s'il est disponible, ou bascule sur `pdflatex` par défaut.

Pour compiler un document spécifique, passez son nom en argument :
```bash
./compile.sh stage_resume.tex
# ou
./compile.sh beamer_main.tex
```

Si aucun argument n'est fourni, le script tente de compiler à la fois `stage_resume.tex` et `beamer_main.tex` s'ils sont présents dans le dossier.
