#!/usr/bin/env bash
# Script de compilation pour les documents LaTeX (présentation et résumé)
set -euo pipefail

# Se positionner dans le dossier du script pour que les chemins relatifs soient corrects
cd "$(dirname "${BASH_SOURCE[0]}")"

tex_file="${1:-}"

# Détection des fichiers à compiler si aucun argument n'est fourni
files_to_compile=()
if [[ -z "$tex_file" ]]; then
  if [[ -f "stage_resume.tex" ]]; then
    files_to_compile+=("stage_resume.tex")
  fi
  if [[ -f "beamer_main.tex" ]]; then
    files_to_compile+=("beamer_main.tex")
  fi
  if [[ ${#files_to_compile[@]} -eq 0 ]]; then
    echo "Erreur: aucun fichier .tex trouvé à compiler." >&2
    exit 1
  fi
else
  files_to_compile+=("$tex_file")
fi

for f in "${files_to_compile[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Erreur: fichier introuvable: $f" >&2
    exit 1
  fi
  
  filename="$(basename "$f")"
  basename="${filename%.tex}"
  
  echo "Compilation de ${filename}..."
  
  # Utilisation de latexmk si disponible, sinon fallback sur pdflatex
  if command -v latexmk >/dev/null 2>&1; then
    build_log="$(mktemp "${TMPDIR:-/tmp}/${basename}.latexmk.XXXXXX.log")"
    cleanup() {
      rm -f "$build_log"
    }
    trap cleanup EXIT
    
    if ! latexmk -pdf -silent -interaction=nonstopmode -halt-on-error "$filename" >"$build_log" 2>&1; then
      echo "Erreur: la compilation a échoué. Dernières lignes du journal:" >&2
      tail -n 40 "$build_log" >&2
      exit 1
    fi
    latexmk -c "$filename" >/dev/null 2>&1 || true
    rm -f \
      "${basename}".{aux,bbl,bcf,blg,fdb_latexmk,fls,log,out,run.xml,synctex.gz,toc,lof,lot,nav,snm,vrb} \
      "${basename}".synctex.gz\(busy\)
    echo "PDF généré: ${basename}.pdf"
  else
    # Fallback sur pdflatex
    if ! pdflatex -interaction=nonstopmode -halt-on-error "$filename" >/dev/null 2>&1; then
      echo "Erreur lors de la compilation de $filename avec pdflatex." >&2
      exit 1
    fi
    # Si c'est stage_resume et bibtex est disponible, on gère la biblio
    if [[ "$basename" == "stage_resume" ]] && command -v bibtex >/dev/null 2>&1; then
      bibtex "$basename" >/dev/null 2>&1 || true
      pdflatex -interaction=nonstopmode -halt-on-error "$filename" >/dev/null 2>&1 || true
      pdflatex -interaction=nonstopmode -halt-on-error "$filename" >/dev/null 2>&1 || true
    fi
    rm -f "${basename}".{aux,log,nav,out,snm,toc,vrb,bbl,blg,lof,lot}
    echo "Compilation réussie ! Le fichier PDF ${basename}.pdf a été généré."
  fi
done
