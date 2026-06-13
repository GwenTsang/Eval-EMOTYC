#!/usr/bin/env python3
"""
Orchestrateur EMOTYC — charge le modèle ONNX une seule fois
et évalue tous les fichiers XLSX d'un dossier.

Remplace l'ancien orchestration_cyberaggado.py (supprimé).
Usage équivalent :
    python orchestrate_emotyc_folder.py
"""

import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

from common import (
    ALL_LABELS,
    build_context_texts,
    build_prediction_summary,
    compute_group_metrics,
    get_predictor,
    load_gold_xlsx,
    compute_metrics,
    write_json,
)

DEFAULT_GOLD_DIR = ROOT / "results" / "prepared_xlsx_samples" / "subsets"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrateur EMOTYC — inférence ONNX sur un dossier de fichiers XLSX gold."
    )
    parser.add_argument(
        "gold_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_GOLD_DIR,
        help="Dossier contenant les fichiers XLSX gold (défaut: ./results/prepared_xlsx_samples/subsets)",
    )
    parser.add_argument(
        "--no-context", action="store_true", default=False,
        help="Désactiver l'utilisation des phrases voisines (i-1, i+1) comme contexte BCA",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Taille du batch pour l'inférence (défaut: 128)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Seuil de binarisation des probabilités (défaut: 0.5)",
    )
    parser.add_argument(
        "--groups", action="store_true",
        help="Afficher les métriques par groupe sémantique (émo, émotions, modes, types)",
    )
    return parser.parse_args()


def build_texts(sentences, use_context):
    """Construit les inputs BCA pour un fichier."""
    return build_context_texts(sentences, use_context=use_context)


def print_group_metrics(gold, pred, threshold):
    """Affiche les métriques par groupe sémantique (rappel, précision, F1)."""
    print(f"\n{'═' * 65}")
    print(f"  MÉTRIQUES PAR GROUPE SÉMANTIQUE  (seuil: {threshold})")
    print(f"{'═' * 65}")
    for metrics in compute_group_metrics(gold, pred, ALL_LABELS).values():
        labels = ", ".join(metrics["labels"])
        print(f"\n── {metrics['display_name']} ({labels}) ──")
        print(f"   Macro Rappel    : {metrics['recall']:.3f}")
        print(f"   Macro Précision : {metrics['precision']:.3f}")
        print(f"   Macro F1        : {metrics['macro_f1']:.3f}")
    print(f"\n{'═' * 65}")


def main() -> None:
    args = parse_args()
    gold_dir = args.gold_dir.resolve()

    if not gold_dir.is_dir():
        raise NotADirectoryError(f"Dossier introuvable : {gold_dir}")

    files = sorted(gold_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier XLSX dans : {gold_dir}")

    out_dir = ROOT / "results" / f"orchestrated_emotyc_{gold_dir.name}"

    print(f"{len(files)} fichiers sélectionnés dans {gold_dir}")
    use_context = not args.no_context
    print(f"Contexte: {'oui' if use_context else 'non'}")

    # Charger le modèle ONNX
    try:
        predictor = get_predictor()
    except Exception as e:
        sys.exit(f"ERREUR lors du chargement du modèle : {e}")

    all_gold, all_probs = [], []

    for xlsx in files:
        try:
            _, sentences, gold = load_gold_xlsx(str(xlsx))
        except Exception as e:
            print(f"  ✗ {xlsx.name} — Erreur lors du chargement: {e}")
            continue

        texts = build_texts(sentences, use_context)
        probs = predictor.predict_texts(texts, batch_size=args.batch_size)
        all_gold.append(gold)
        all_probs.append(probs)
        print(f"  ✓ {xlsx.name} — {len(sentences)} phrases")

    if not all_gold:
        sys.exit("Aucun fichier n'a pu être traité avec succès.")

    gold_cat = np.vstack(all_gold)
    probs_cat = np.vstack(all_probs)
    pred_cat = (probs_cat >= args.threshold).astype(int)

    # Métriques par groupe sémantique (si --groups)
    if args.groups:
        print_group_metrics(gold_cat, pred_cat, args.threshold)

    # Métriques per-label (toujours calculées et exportées)
    per_label, global_metrics = compute_metrics(gold_cat, pred_cat, ALL_LABELS)

    summary = build_prediction_summary(
        source=gold_dir.name,
        n_samples=len(gold_cat),
        threshold=args.threshold,
        per_label=per_label,
        global_metrics=global_metrics,
        source_key="source",
        extra={
            "n_files": len(files),
            "use_context": use_context,
        },
    )
    out_path = out_dir / "emotyc_predictions_summary.json"
    write_json(out_path, summary)
    print(f"\nMétriques globales exportées : {out_path}")


if __name__ == "__main__":
    main()
