#!/usr/bin/env python3
"""
Inférence EMOTYC détaillée et comparaison au gold label via ONNX.
Charge le modèle EMOTYC au format ONNX, applique les prédictions
sur chaque ligne du gold label, compare avec les annotations humaines,
et exporte les résultats sous forme d'un tableau XLSX, d'un JSONL détaillé
et d'un résumé JSON.
"""
import argparse
import json
import math
import os
import sys
import numpy as np
import pandas as pd

from emotyc_common import (
    EMOTYC_LABEL2ID,
    ALL_LABELS,
    EMOTION_LABELS,
    MODE_LABELS,
    TYPE_LABELS,
    DEFAULT_THRESHOLD,
    DEFAULT_MODE_THRESHOLD,
    get_predictor,
    format_input,
    load_gold_xlsx,
    compute_metrics,
)

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTES DÉRIVÉES
# ═══════════════════════════════════════════════════════════════════════════

EMO_LABEL = "Emo"

# Indices dans le vecteur de 19 logits/probas
EMOTION_INDICES = [EMOTYC_LABEL2ID[e] for e in EMOTION_LABELS]
MODE_INDICES = [EMOTYC_LABEL2ID[m] for m in MODE_LABELS]
TYPE_INDICES = [EMOTYC_LABEL2ID[t] for t in TYPE_LABELS]
EMO_INDEX = EMOTYC_LABEL2ID[EMO_LABEL]


# ═══════════════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════

def _print_metrics_table(title, per_label, global_metrics, threshold_info=None):
    """Affiche un tableau de métriques formaté."""
    t_info = f"  (seuil: {threshold_info})" if threshold_info else ""
    print(f"\n{'—' * 75}")
    print(f"  {title}{t_info}")
    print(f"{'—' * 75}")
    print(f"  {'Label':<20s} {'Acc':>7s} {'Kappa':>7s} {'F1':>7s} "
          f"{'Prec':>7s} {'Recall':>7s} {'FP':>5s} {'FN':>5s}")
    print(f"  {'-' * 68}")
    for r in per_label:
        k_str = f"{r['kappa']:.3f}" if r['kappa'] is not None else "  N/A"
        print(f"  {r['label']:<20s} {r['accuracy']:>7.3f} {k_str:>7s} "
              f"{r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['fp']:>5d} {r['fn']:>5d}")
    print(f"  {'-' * 68}")
    print(f"  Macro-F1    : {global_metrics['macro_f1']:.4f}")
    print(f"  Micro-F1    : {global_metrics['micro_f1']:.4f}")
    print(f"  Exact Match : {global_metrics['exact_match']:.4f}")
    print(f"{'—' * 75}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True,
                   help="Chemin vers le fichier gold label (.xlsx)")
    p.add_argument("--out_dir", required=True,
                   help="Dossier de sortie pour les résultats")
    p.add_argument("--use-context", action="store_true",
                   help="Utiliser les phrases voisines (i-1, i+1) comme contexte")
    p.add_argument("--template", choices=["bca", "bca_spaced"], default="bca",
                   help="Format du template d'input. "
                        "'bca' = before:{prev}</s>current:{s}</s>after:{next}</s> (défaut), "
                        "'bca_spaced' = before:{prev}</s>current: {s}</s>after:{next}</s> "
                        "(espace uniquement après current:)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Taille du batch pour l'inférence (défaut: 32)")
    p.add_argument("--mode-threshold", type=float, default=DEFAULT_MODE_THRESHOLD,
                   help=f"Seuil pour les prédictions des modes d'expression (défaut: {DEFAULT_MODE_THRESHOLD})")
    return p.parse_args()


def main():
    args = parse_args()
    EMOTION_THRESHOLD = DEFAULT_THRESHOLD

    # ── 1. Chargement du gold ─────────────────────────────────────────
    xlsx_path = os.path.abspath(args.xlsx)
    try:
        df, sentences, gold = load_gold_xlsx(xlsx_path)
    except Exception as e:
        sys.exit(str(e))

    N = len(sentences)

    # Gold matrices par groupe
    gold_emotion = gold[:, EMOTION_INDICES]
    gold_mode = gold[:, MODE_INDICES]
    gold_emo = gold[:, [EMO_INDEX]]
    gold_type = gold[:, TYPE_INDICES]

    # ── 2. Chargement du modèle ───────────────────────────────────────
    model_dir = os.path.join(os.path.dirname(__file__), "model_onnx")
    try:
        predictor = get_predictor(model_dir=model_dir)
    except Exception as e:
        sys.exit(f"ERREUR lors du chargement du modèle : {e}")

    # ── 3. Préparation des inputs ─────────────────────────────────────
    use_context = args.use_context
    formatted_texts = [
        format_input(
            sentences[i],
            sentences[i - 1] if (i > 0 and use_context) else None,
            sentences[i + 1] if (i < N - 1 and use_context) else None,
            use_context,
            template=args.template,
        )
        for i in range(N)
    ]

    ctx_suffix = "_context" if use_context else "_no_context"
    template_name = f"{args.template}{ctx_suffix}"
    print(f"Template : {template_name}")
    print(f"Exemple  : {formatted_texts[0][:120]}…")

    # ── 4. Inférence ──────────────────────────────────────────────────
    print(f"\nInférence sur {N} phrases (batch_size={args.batch_size})…")
    all_probs = predictor.predict_texts(formatted_texts, batch_size=args.batch_size)
    print(f"Inférence terminée — shape: {all_probs.shape}")

    # ── 5. Extraction des probas par groupe ───────────────────────────
    emotion_probs = all_probs[:, EMOTION_INDICES]
    mode_probs = all_probs[:, MODE_INDICES]
    emo_probs = all_probs[:, EMO_INDEX]
    type_probs = all_probs[:, TYPE_INDICES]

    # ── 6. Prédictions binaires ───────────────────────────────────────
    print(f"▸ Seuil émotions : {EMOTION_THRESHOLD}")
    print(f"▸ Seuil modes : {args.mode_threshold}")
    pred_emotion = (emotion_probs >= EMOTION_THRESHOLD).astype(int)
    pred_mode = (mode_probs >= args.mode_threshold).astype(int)
    pred_emo = (emo_probs >= 0.5).astype(int)
    pred_type = (type_probs >= 0.5).astype(int)

    # ── 7. Métriques (affichage terminal par groupe) ──────────────────
    per_emotion, global_emotion = compute_metrics(gold_emotion, pred_emotion, EMOTION_LABELS)
    _print_metrics_table("MÉTRIQUES PAR ÉMOTION", per_emotion, global_emotion,
                         threshold_info=f"{EMOTION_THRESHOLD}")

    per_mode, global_mode = compute_metrics(gold_mode, pred_mode, MODE_LABELS)
    _print_metrics_table("MÉTRIQUES PAR MODE D'EXPRESSION", per_mode, global_mode,
                         threshold_info=f"{args.mode_threshold}")

    per_emo, global_emo = compute_metrics(gold_emo, pred_emo.reshape(-1, 1), [EMO_LABEL])
    _print_metrics_table("MÉTRIQUES — CARACTÈRE ÉMOTIONNEL (Emo)", per_emo, global_emo)

    per_type, global_type = compute_metrics(gold_type, pred_type, TYPE_LABELS)
    _print_metrics_table("MÉTRIQUES — TYPE (Base/Complexe)", per_type, global_type)

    # ── 8. Export XLSX et JSONL ────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    # XLSX des prédictions
    export_dict = {"TEXT": sentences}
    for j, emo in enumerate(EMOTION_LABELS):
        export_dict[emo] = pred_emotion[:, j]
    for j, mode in enumerate(MODE_LABELS):
        export_dict[mode] = pred_mode[:, j]
    out_xlsx = os.path.join(args.out_dir, "emotyc_predictions_output.xlsx")
    pd.DataFrame(export_dict).to_excel(out_xlsx, index=False)
    print(f"Prédictions exportées en XLSX : {out_xlsx}")

    # JSONL détaillé
    out_jsonl = os.path.join(args.out_dir, "emotyc_predictions.jsonl")
    n_divergent = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(N):
            # Divergences émotions + modes
            divergences = []
            for labels, gold_m, pred_m, probs_m, threshold, dim in [
                (EMOTION_LABELS, gold_emotion, pred_emotion, emotion_probs, EMOTION_THRESHOLD, "emotion"),
                (MODE_LABELS, gold_mode, pred_mode, mode_probs, args.mode_threshold, "mode"),
            ]:
                for j, label in enumerate(labels):
                    g, p = int(gold_m[i, j]), int(pred_m[i, j])
                    if g != p:
                        divergences.append({
                            "dimension": dim, "label": label,
                            "gold": g, "pred": p,
                            "proba": round(float(probs_m[i, j]), 6),
                            "seuil": threshold,
                            "type_divergence": "faux_positif" if p == 1 else "faux_negatif",
                        })

            if divergences:
                n_divergent += 1

            id_val = df.iloc[i].get("ID", i)
            record = {
                "idx": i,
                "id": str(id_val) if id_val is not None and not (isinstance(id_val, float) and math.isnan(id_val)) else str(i),
                "text": sentences[i],
                "text_prev": sentences[i - 1] if i > 0 else None,
                "text_next": sentences[i + 1] if i < N - 1 else None,
                "template_used": template_name,
                "emotion_threshold": EMOTION_THRESHOLD,
                "mode_threshold": args.mode_threshold,
                # Émotions
                "probas": {e: round(float(emotion_probs[i, j]), 6) for j, e in enumerate(EMOTION_LABELS)},
                "preds": {e: int(pred_emotion[i, j]) for j, e in enumerate(EMOTION_LABELS)},
                "golds": {e: int(gold_emotion[i, j]) for j, e in enumerate(EMOTION_LABELS)},
                # Modes
                "probas_mode": {m: round(float(mode_probs[i, j]), 6) for j, m in enumerate(MODE_LABELS)},
                "preds_mode": {m: int(pred_mode[i, j]) for j, m in enumerate(MODE_LABELS)},
                "golds_mode": {m: int(gold_mode[i, j]) for j, m in enumerate(MODE_LABELS)},
                # Caractère émotionnel
                "proba_emo": round(float(emo_probs[i]), 6),
                "pred_emo": int(pred_emo[i]),
                "gold_emo": int(gold_emo[i, 0]),
                # Type (Base/Complexe)
                "probas_type": {t: round(float(type_probs[i, j]), 6) for j, t in enumerate(TYPE_LABELS)},
                "preds_type": {t: int(pred_type[i, j]) for j, t in enumerate(TYPE_LABELS)},
                "golds_type": {t: int(gold_type[i, j]) for j, t in enumerate(TYPE_LABELS)},
                # Divergences
                "n_divergences": len(divergences),
                "divergences": divergences,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n Résultats exportés : {out_jsonl}")
    print(f"  {N} lignes, {n_divergent} avec ≥1 divergence")

    # ── 9. Export du résumé JSON (format normalisé) ─────────────────
    per_label = per_emo + per_emotion + per_mode + per_type

    gold_all = np.hstack([gold_emo, gold_emotion, gold_mode, gold_type])
    pred_all = np.hstack([pred_emo.reshape(-1, 1), pred_emotion, pred_mode, pred_type])
    
    from sklearn.metrics import f1_score as _f1_score
    global_all = {
        "macro_f1": round(float(np.mean([r["f1"] for r in per_label])), 4),
        "micro_f1": round(float(_f1_score(gold_all.ravel(), pred_all.ravel(), zero_division=0)), 4),
        "exact_match": round(float(np.all(gold_all == pred_all, axis=1).mean()), 4),
        "n_samples": N,
        "n_labels": len(per_label),
    }

    summary = {
        "source_xlsx": os.path.basename(xlsx_path),
        "n_samples": N,
        "template": template_name,
        "threshold": EMOTION_THRESHOLD,
        "per_label": per_label,
        "global_metrics": global_all,
    }
    summary_path = os.path.join(args.out_dir, "emotyc_predictions_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Résumé : {summary_path}")


if __name__ == "__main__":
    main()
