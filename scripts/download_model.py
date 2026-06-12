#!/usr/bin/env python3
"""Download the EMOTYC ONNX weights required for local inference."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_URL = os.environ.get(
    "EMOTYC_ONNX_URL",
    "https://huggingface.co/GwendalTsang/EMOTYC-ONNX/resolve/main/model.onnx",
)
EXPECTED_SHA256 = "e0c18514933453452929c9f699d68e1fd253414dd44046cc9ea77c445fcfd642"
EXPECTED_SIZE = 442_769_218
CHUNK_SIZE = 1024 * 1024
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "model_onnx" / "model.onnx"


def format_bytes(value: int) -> str:
    mib = value / (1024 * 1024)
    return f"{mib:.1f} MiB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def validate_model(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "file is missing"
    if is_git_lfs_pointer(path):
        return False, "file is a Git LFS pointer, not the ONNX weights"

    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        return (
            False,
            f"unexpected size: got {size} bytes, expected {EXPECTED_SIZE} bytes",
        )

    digest = sha256_file(path)
    if digest != EXPECTED_SHA256:
        return False, f"unexpected sha256: got {digest}, expected {EXPECTED_SHA256}"

    return True, f"valid model ({format_bytes(size)}, sha256={digest})"


def download(url: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f"{output.name}.tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Eval-EMOTYC-model-downloader/1.0"},
    )

    try:
        with urllib.request.urlopen(request) as response, tmp_output.open("wb") as handle:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            downloaded = 0
            next_progress = time.monotonic()

            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now >= next_progress:
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\rDownloaded {format_bytes(downloaded)} / "
                            f"{format_bytes(total)} ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\rDownloaded {format_bytes(downloaded)}",
                            end="",
                            flush=True,
                        )
                    next_progress = now + 0.5
    except urllib.error.URLError as exc:
        if tmp_output.exists():
            tmp_output.unlink()
        raise RuntimeError(f"download failed from {url}: {exc}") from exc

    print(f"\rDownloaded {format_bytes(tmp_output.stat().st_size)}".ljust(80))
    return tmp_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify Eval-EMOTYC/model_onnx/model.onnx.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Model URL. Defaults to EMOTYC_ONNX_URL or the public Hugging Face URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if the local model is already valid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()

    if not args.force:
        valid, message = validate_model(output)
        if valid:
            print(f"OK: {output}")
            print(message)
            return 0
        if output.exists():
            print(f"Local model is not valid: {message}", file=sys.stderr)
            print("Downloading a fresh copy...", file=sys.stderr)

    print(f"Downloading model from {args.url}")
    print(f"Destination: {output}")
    tmp_output = download(args.url, output)

    valid, message = validate_model(tmp_output)
    if not valid:
        tmp_output.unlink(missing_ok=True)
        print(f"Downloaded file is not valid: {message}", file=sys.stderr)
        return 1

    tmp_output.replace(output)
    print(f"OK: {output}")
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
