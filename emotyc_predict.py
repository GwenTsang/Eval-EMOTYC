#!/usr/bin/env python3
"""
Inférence EMOTYC minimale — 19 labels traités uniformément.
Charge le modèle EMOTYC au format ONNX, applique les prédictions
sur chaque ligne du gold label, calcule des métriques globales agrégées,
et exporte un unique fichier emotyc_predictions_summary.json.
"""
import argparse
import os
import sys
import numpy as np

from common import (
    ALL_LABELS,
    build_context_texts,
    build_prediction_summary,
    get_predictor,
    load_gold_xlsx,
    compute_metrics,
    write_json,
)

def main():
    p = argparse.ArgumentParser(description="Inférence EMOTYC — 19 labels, métriques globales")
    p.add_argument("--xlsx", required=True, help="Fichier gold label (.xlsx)")
    p.add_argument("--out_dir", required=True, help="Dossier de sortie")
    p.add_argument("--use-context", action="store_true", help="Utiliser les phrases voisines comme contexte")
    p.add_argument("--batch-size", type=int, default=128, help="Taille du batch")
    args = p.parse_args()

    # 1. Gold
    xlsx_path = os.path.abspath(args.xlsx)
    try:
        df, sentences, gold = load_gold_xlsx(xlsx_path)
    except Exception as e:
        sys.exit(str(e))
    
    N = len(sentences)

    # 2. Modèle
    try:
        predictor = get_predictor()
    except Exception as e:
        sys.exit(f"ERREUR lors du chargement du modèle : {e}")

    # 3. Inputs formatés (bca_spaced)
    use_ctx = args.use_context
    texts = build_context_texts(sentences, use_context=use_ctx, template="bca_spaced")
    ctx_tag = "context" if use_ctx else "no_context"

    # 4. Inférence
    print(f"\nInférence sur {N} phrases (batch_size={args.batch_size})…")
    probs = predictor.predict_texts(texts, batch_size=args.batch_size)
    pred = (probs >= 0.5).astype(int)
    print(f"Inférence terminée — shape: {probs.shape}")

    # 5. Métriques
    per_label, global_metrics = compute_metrics(gold, pred, ALL_LABELS)

    # 6. Export résumé JSON uniquement
    summary = build_prediction_summary(
        source=os.path.basename(xlsx_path),
        n_samples=N,
        template=f"bca_spaced_{ctx_tag}",
        threshold=0.5,
        per_label=per_label,
        global_metrics=global_metrics,
    )
    out = os.path.join(args.out_dir, "emotyc_predictions_summary.json")
    write_json(out, summary)
    print(f"\nRésumé exporté : {out}")

if __name__ == "__main__":
    main()
