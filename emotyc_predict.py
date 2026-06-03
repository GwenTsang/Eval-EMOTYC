#!/usr/bin/env python3
"""
Inférence EMOTYC minimale — 19 labels traités uniformément.
Charge le modèle EMOTYC au format ONNX, applique les prédictions
sur chaque ligne du gold label, calcule des métriques globales agrégées,
et exporte un unique fichier emotyc_predictions_summary.json.
"""
import argparse
import json
import os
import sys
import numpy as np

from emotyc_common import (
    ALL_LABELS,
    get_predictor,
    format_input,
    load_gold_xlsx,
    compute_metrics,
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
    model_dir = os.path.join(os.path.dirname(__file__), "model_onnx")
    try:
        predictor = get_predictor(model_dir=model_dir)
    except Exception as e:
        sys.exit(f"ERREUR lors du chargement du modèle : {e}")

    # 3. Inputs formatés (bca_spaced)
    use_ctx = args.use_context
    texts = [
        format_input(
            sentences[i],
            sentences[i - 1] if (i > 0 and use_ctx) else None,
            sentences[i + 1] if (i < N - 1 and use_ctx) else None,
            use_ctx,
            template="bca_spaced"
        )
        for i in range(N)
    ]
    ctx_tag = "context" if use_ctx else "no_context"

    # 4. Inférence
    print(f"\nInférence sur {N} phrases (batch_size={args.batch_size})…")
    probs = predictor.predict_texts(texts, batch_size=args.batch_size)
    pred = (probs >= 0.5).astype(int)
    print(f"Inférence terminée — shape: {probs.shape}")

    # 5. Métriques
    per_label, global_metrics = compute_metrics(gold, pred, ALL_LABELS)

    # Pour des raisons de compatibilité avec les anciens rapports, on ajoute micro/macro F1 arrondis à 3 décimales
    global_metrics_3dec = {
        "micro_f1": round(global_metrics["micro_f1"], 3),
        "macro_f1": round(global_metrics["macro_f1"], 3),
    }

    # On formate aussi per_label pour garder 3 décimales
    per_label_3dec = []
    for r in per_label:
        r_3 = dict(r)
        for k in ["accuracy", "kappa", "f1", "precision", "recall", "prevalence_gold", "prevalence_pred"]:
            if r_3.get(k) is not None:
                r_3[k] = round(r_3[k], 3)
        per_label_3dec.append(r_3)

    # 6. Export résumé JSON uniquement
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "source_xlsx": os.path.basename(xlsx_path),
        "n_samples": N,
        "template": f"bca_spaced_{ctx_tag}",
        "threshold": 0.5,
        "per_label": per_label_3dec,
        "global_metrics": global_metrics_3dec,
    }
    out = os.path.join(args.out_dir, "emotyc_predictions_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nRésumé exporté : {out}")

if __name__ == "__main__":
    main()
