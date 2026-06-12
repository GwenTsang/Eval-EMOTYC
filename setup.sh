#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REQUIREMENTS_FILE="${EMOTYC_REQUIREMENTS:-$ROOT_DIR/requirements.txt}"
MODEL_URL="${EMOTYC_ONNX_URL:-https://huggingface.co/GwendalTsang/EMOTYC-ONNX/resolve/main/model.onnx}"
MODEL_PATH="${EMOTYC_ONNX_PATH:-$ROOT_DIR/model_onnx/model.onnx}"
MODEL_TMP="${MODEL_PATH}.tmp"
EXPECTED_SHA256="${EMOTYC_ONNX_SHA256:-e0c18514933453452929c9f699d68e1fd253414dd44046cc9ea77c445fcfd642}"
PYTHON_BIN="${PYTHON:-python3}"

usage() {
    cat <<EOF
Usage: bash setup.sh

Installs Python dependencies from requirements.txt, then downloads and verifies
model_onnx/model.onnx.

Options:
  -h, --help     Show this help message.

Environment overrides:
  PYTHON             Python executable used for pip, default: python3
  EMOTYC_ONNX_URL    Model URL
  EMOTYC_ONNX_PATH   Destination path
  EMOTYC_ONNX_SHA256 Expected SHA-256
EOF
}

log() {
    printf '\n==> %s\n' "$*"
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
    usage
    exit 0
fi

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    else
        die "sha256sum or shasum is required to verify the ONNX weights."
    fi
}

validate_model() {
    local path="$1"
    local digest

    if [[ ! -f "$path" ]]; then
        printf 'file is missing\n'
        return 1
    fi

    digest="$(sha256_file "$path")"
    if [[ "$digest" != "$EXPECTED_SHA256" ]]; then
        printf 'unexpected sha256: got %s, expected %s\n' "$digest" "$EXPECTED_SHA256"
        return 1
    fi

    printf 'valid model (sha256=%s)\n' "$digest"
}

install_requirements() {
    [[ -f "$REQUIREMENTS_FILE" ]] || die "Requirements file not found: $REQUIREMENTS_FILE"
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

    log "Installing Python dependencies from $REQUIREMENTS_FILE"
    "$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"
}

download_file() {
    rm -f "$MODEL_TMP"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --show-error --progress-bar --output "$MODEL_TMP" "$MODEL_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$MODEL_TMP" "$MODEL_URL"
    else
        die "curl or wget is required to download the ONNX weights."
    fi
}

download_model() {
    local validation_message

    mkdir -p "$(dirname "$MODEL_PATH")"

    if validation_message="$(validate_model "$MODEL_PATH")"; then
        log "ONNX weights already present"
        printf 'OK: %s\n%s\n' "$MODEL_PATH" "$validation_message"
        return 0
    fi

    if [[ -e "$MODEL_PATH" ]]; then
        warn "Local ONNX weights are not valid: $validation_message"
        warn "Downloading a fresh copy."
    fi

    log "Downloading ONNX weights"
    printf 'URL: %s\n' "$MODEL_URL"
    printf 'Destination: %s\n' "$MODEL_PATH"

    download_file

    if ! validation_message="$(validate_model "$MODEL_TMP")"; then
        rm -f "$MODEL_TMP"
        die "Downloaded ONNX weights are not valid: $validation_message"
    fi

    mv -f "$MODEL_TMP" "$MODEL_PATH"
    printf 'OK: %s\n%s\n' "$MODEL_PATH" "$validation_message"
}

cleanup() {
    rm -f "$MODEL_TMP"
}
trap cleanup EXIT

install_requirements
download_model
log "Setup complete"