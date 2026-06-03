#!/usr/bin/env python3
"""
Orchestrateur EMOTYC — charge le modèle ONNX une seule fois
et évalue tous les fichiers XLSX d'un dossier.

Remplace l'ancien orchestration_cyberaggado.py (supprimé).
Usage équivalent :
    python orchestrate_emotyc_folder.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent

from emotyc_common import (
    ALL_LABELS,
    EMOTYC_LABEL2ID,
    LABEL_GROUPS,
    GROUP_DISPLAY_NAMES,
    get_predictor,
    format_input,
    load_gold_xlsx,
    compute_metrics,
)

DEFAULT_GOLD_DIR = ROOT / "results" / "prepared_xlsx_samples" / "subsets"

GROUP_INDICES = {
    g: [EMOTYC_LABEL2ID[l] for l in labels]
    for g, labels in LABEL_GROUPS.items()
}


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
        "--template", choices=["bca", "bca_spaced"], default="bca_spaced",
        help="Format du template d'input (défaut: bca_spaced)",
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


def build_texts(sentences, use_context, template):
    """Construit les inputs BCA pour un fichier."""
    n = len(sentences)
    return [
        format_input(
            sentences[i],
            sentences[i - 1] if i > 0 and use_context else None,
            sentences[i + 1] if i < n - 1 and use_context else None,
            use_context,
            template=template,
        )
        for i in range(n)
    ]


def print_group_metrics(gold, pred, threshold):
    """Affiche les métriques par groupe sémantique (rappel, précision, F1)."""
    print(f"\n{'═' * 65}")
    print(f"  MÉTRIQUES PAR GROUPE SÉMANTIQUE  (seuil: {threshold})")
    print(f"{'═' * 65}")
    for group, indices in GROUP_INDICES.items():
        g, p = gold[:, indices], pred[:, indices]
        labels = ", ".join(LABEL_GROUPS[group])
        print(f"\n── {GROUP_DISPLAY_NAMES[group]} ({labels}) ──")
        print(f"   Macro Rappel    : {recall_score(g, p, average='macro', zero_division=0):.3f}")
        print(f"   Macro Précision : {precision_score(g, p, average='macro', zero_division=0):.3f}")
        print(f"   Macro F1        : {f1_score(g, p, average='macro', zero_division=0):.3f}")
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
    print(f"Contexte: {'oui' if use_context else 'non'}, template: {args.template}")

    # Charger le modèle ONNX
    model_dir = os.path.join(os.path.dirname(__file__), "model_onnx")
    try:
        predictor = get_predictor(model_dir=model_dir)
    except Exception as e:
        sys.exit(f"ERREUR lors du chargement du modèle : {e}")

    all_gold, all_probs = [], []

    for xlsx in files:
        try:
            _, sentences, gold = load_gold_xlsx(str(xlsx))
        except Exception as e:
            print(f"  ✗ {xlsx.name} — Erreur lors du chargement: {e}")
            continue

        texts = build_texts(sentences, use_context, args.template)
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

    global_metrics_3dec = {
        "micro_f1": round(global_metrics["micro_f1"], 3),
        "macro_f1": round(global_metrics["macro_f1"], 3),
    }

    per_label_3dec = []
    for r in per_label:
        r_3 = dict(r)
        for k in ["accuracy", "kappa", "f1", "precision", "recall", "prevalence_gold", "prevalence_pred"]:
            if r_3.get(k) is not None:
                r_3[k] = round(r_3[k], 3)
        per_label_3dec.append(r_3)

    ctx_tag = "context" if use_context else "no_context"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_files": len(files),
        "n_samples": len(gold_cat),
        "template": f"{args.template}_{ctx_tag}",
        "threshold": args.threshold,
        "use_context": use_context,
        "global_metrics": global_metrics_3dec,
        "per_label": per_label_3dec,
    }
    out_path = out_dir / "emotyc_predictions_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMétriques globales exportées : {out_path}")


if __name__ == "__main__":
    main()