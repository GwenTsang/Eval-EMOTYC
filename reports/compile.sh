#!/usr/bin/env bash
set -euo pipefail

tex_file="${1:-stage_resume.tex}"

if [[ ! -f "$tex_file" ]]; then
  echo "Erreur: fichier introuvable: $tex_file" >&2
  exit 1
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "Erreur: latexmk n'est pas installé ou n'est pas dans le PATH." >&2
  exit 1
fi

workdir="$(cd "$(dirname "$tex_file")" && pwd)"
filename="$(basename "$tex_file")"
basename="${filename%.tex}"
build_log="$(mktemp "${TMPDIR:-/tmp}/${basename}.latexmk.XXXXXX.log")"

cleanup() {
  rm -f "$build_log"
}
trap cleanup EXIT

cd "$workdir"

echo "Compilation de ${filename}..."

if ! latexmk -pdf -silent -interaction=nonstopmode -halt-on-error "$filename" >"$build_log" 2>&1; then
  echo "Erreur: la compilation a échoué. Dernières lignes du journal:" >&2
  tail -n 40 "$build_log" >&2
  exit 1
fi

# Nettoie les fichiers auxiliaires en conservant le PDF généré.
latexmk -c "$filename" >/dev/null 2>&1 || true
rm -f \
  "${basename}".{aux,bbl,bcf,blg,fdb_latexmk,fls,log,out,run.xml,synctex.gz,toc,lof,lot,nav,snm,vrb} \
  "${basename}".synctex.gz\(busy\)

echo "PDF généré: ${workdir}/${basename}.pdf"
