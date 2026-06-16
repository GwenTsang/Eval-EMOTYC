## Compilation

Le script `compile.sh` permet de compiler les documents. Il utilise `latexmk` s'il est disponible et bascule sur `pdflatex` par défaut.

Pour compiler un document spécifique, passez son nom en argument :
```bash
./compile.sh stage_resume.tex
# ou
./compile.sh beamer_main.tex
```

Si aucun argument n'est fourni, le script tente de compiler à la fois `stage_resume.tex` et `beamer_main.tex` s'ils sont présents dans le dossier.
