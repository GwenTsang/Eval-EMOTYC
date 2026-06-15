# Résumé de stage

Ce dossier contient le résumé de stage LaTeX et les fichiers nécessaires à sa
compilation :

- `stage_resume.tex` : source principale ;
- `stageOnera.cls` : classe LaTeX ONERA ;
- `onera.bst` et `references.bib` : bibliographie ;
- `logo-onera-ident-quadri-HD.png` : logo utilisé par la classe ;
- `figures/schema.pdf` : schéma EMOTYC intégré au document ;
- `compile.sh` : script de compilation avec `latexmk` ;
- `stage_resume.pdf` : PDF généré.

Compilation :

```bash
./compile.sh stage_resume.tex
```
