#!/usr/bin/env python3
"""Prépare des sous-ensembles XLSX EMOTYC, avec ou sans contexte."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = [
    ("homophobie", Path("golds/CyberAggAdo/homophobie_annotations_gold_flat_updated.xlsx")),
    ("obésité", Path("golds/CyberAggAdo/obésité_annotations_gold_flat_updated.xlsx")),
    ("religion", Path("golds/CyberAggAdo/religion_annotations_gold_flat_updated.xlsx")),
    ("racisme", Path("golds/CyberAggAdo/racisme_annotations_gold_flat_updated.xlsx")),
]


def repo(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-files", nargs="+", type=Path,
                        help="Remplace les 4 XLSX par défaut. Chemins relatifs au repo.")
    parser.add_argument("--mode", choices=("context", "nocontext"), default="context")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir", type=Path,
                        help="Défaut: results/prepared_xlsx_samples/subsets")
    args = parser.parse_args()

    if args.sample_size <= 0:
        parser.error("--sample-size doit être > 0")

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)
    sources = [(p.stem, p) for p in args.source_files] if args.source_files else DEFAULT_SOURCES
    out_dir = repo(args.out_dir) if args.out_dir else ROOT / "results" / "prepared_xlsx_samples" / "subsets"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seed: {seed}")
    print(f"Mode: {args.mode}")
    print(f"Sample size: {args.sample_size}")

    nocontext_samples = []
    for order, (label, raw_path) in enumerate(sources, start=1):
        source = repo(raw_path)
        if not source.exists():
            raise FileNotFoundError(source)

        df = pd.read_excel(source)
        if len(df) < args.sample_size:
            raise ValueError(f"{source.name}: {len(df)} lignes < sample-size {args.sample_size}")

        if args.mode == "context":
            start = rng.randint(0, len(df) - args.sample_size)
            rows = list(range(start, start + args.sample_size))
            range_tag = f"rows_{rows[0]}_{rows[-1]}"
            row_desc = f"start={rows[0]} end_exclusive={rows[-1] + 1} excel_rows={rows[0] + 2}-{rows[-1] + 2}"
        else:
            rows = sorted(rng.sample(range(len(df)), args.sample_size))
            range_tag = f"{args.sample_size}_rows"
            preview = rows[:8]
            row_desc = f"rows_0based={preview}{'...' if len(rows) > len(preview) else ''}"

        print(f"  {order:02d}. {label}: {source.name} len={len(df)} {row_desc}")

        sample = df.iloc[rows].copy().reset_index(drop=True)
        for col, values in reversed([
            ("sample_seed", seed),
            ("sample_mode", args.mode),
            ("sample_label", label),
            ("sample_source_file", source.name),
            ("sample_source_row_0based", rows),
            ("sample_excel_row", [r + 2 for r in rows]),
            ("sample_subset_pos", list(range(len(rows)))),
        ]):
            sample.insert(0, col, values)

        if args.mode == "context":
            out_file = out_dir / f"{order:02d}_{label}_{args.mode}_{range_tag}.xlsx"
            sample.to_excel(out_file, index=False)
        else:
            nocontext_samples.append(sample)

    if args.mode == "nocontext":
        out_file = out_dir / f"all_sources_{args.mode}_{args.sample_size}_rows_each.xlsx"
        pd.concat(nocontext_samples, ignore_index=True).to_excel(out_file, index=False)

    print(f"\nSous-ensembles créés dans: {out_dir}")


if __name__ == "__main__":
    main()
