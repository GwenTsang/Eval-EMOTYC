#!/bin/bash

# Script simple pour compiler le fichier LaTeX et nettoyer les fichiers générés

echo "Compilation en cours..."
pdflatex -interaction=batchmode -halt-on-error beamer_main.tex > /dev/null 2>&1

if [ $? -eq 0 ]; then
    # Nettoyage des fichiers temporaires
    rm -f beamer_main.aux beamer_main.log beamer_main.nav beamer_main.out beamer_main.snm beamer_main.toc
    echo "Compilation réussie ! Le fichier PDF a été généré et les fichiers auxiliaires ont été supprimés."
else
    echo "Erreur lors de la compilation. (Vous pouvez retirer > /dev/null pour voir les détails)."
fi
