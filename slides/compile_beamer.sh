#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEX_NAME="heatmap_delta_beamer.tex"
PDF_NAME="${TEX_NAME%.tex}.pdf"
OUTPUT_PDF="${OUTPUT_PDF:-$SCRIPT_DIR/$PDF_NAME}"
PASSES="${PASSES:-2}"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "Error: pdflatex is not available in PATH." >&2
  exit 127
fi

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/heatmap_delta_beamer.XXXXXX")"

cleanup() {
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    rm -rf "$BUILD_DIR"
  else
    echo "Compilation failed. Auxiliary files kept in: $BUILD_DIR" >&2
  fi
}
trap cleanup EXIT

cd "$SCRIPT_DIR"

for pass in $(seq 1 "$PASSES"); do
  pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="$BUILD_DIR" \
    "$TEX_NAME" \
    >"$BUILD_DIR/pdflatex-pass-$pass.log"
done

mkdir -p "$(dirname -- "$OUTPUT_PDF")"
cp "$BUILD_DIR/$PDF_NAME" "$OUTPUT_PDF"

echo "PDF generated: $OUTPUT_PDF"
