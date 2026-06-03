#!/usr/bin/env python3
"""
Inférence EMOTYC sur fichier Parquet avec ONNX.
Charge le modèle EMOTYC au format ONNX, applique les prédictions
sur le corpus de test TTK au format Parquet, et exporte un résumé JSON.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from emotyc_common import (
    EMOTYC_LABEL2ID,
    LABEL_GROUPS,
    GROUP_DISPLAY_NAMES,
    get_predictor,
)

PARQUET_PATH = os.path.join(os.path.dirname(__file__), "golds", "TTK_test.parquet")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "results")
THRESHOLD    = 0.5
BATCH_SIZE   = 910

# Indices dans le vecteur 19-d pour chaque groupe
GROUP_INDICES = {
    group: [EMOTYC_LABEL2ID[l] for l in labels]
    for group, labels in LABEL_GROUPS.items()
}

# Mapping colonne parquet → labels EMOTYC qu'elle alimente
_PARQUET_COL_TO_LABELS = {
    "Emo":        ["Emo"],
    "types":      ["Base", "Complexe"],
    "modes":      ["Comportementale", "Designee", "Montree", "Suggeree"],
    "categories": [
        "Admiration", "Autre", "Colere", "Culpabilite", "Degout",
        "Embarras", "Fierte", "Jalousie", "Joie", "Peur",
        "Surprise", "Tristesse",
    ],
}


def format_inputs(df):
    eos = "</s>"
    texts = []
    for prev, sent, nxt in zip(
        df["previous_sentence"], df["target_sentence"], df["next_sentence"]
    ):
        prev = prev if isinstance(prev, str) else eos
        nxt  = nxt  if isinstance(nxt, str)  else eos
        texts.append(f"before:{prev}{eos}current: {sent}{eos}after:{nxt}{eos}")
    return texts


def build_gold(df):
    N = len(df)
    gold = np.zeros((N, 19), dtype=np.int8)

    # Emo : chaîne "0"/"1"
    gold[:, EMOTYC_LABEL2ID["Emo"]] = df["Emo"].astype(int).values

    # Colonnes array (types, modes, categories)
    for col, labels in _PARQUET_COL_TO_LABELS.items():
        if col == "Emo":
            continue
        series = df[col]
        for label in labels:
            idx = EMOTYC_LABEL2ID[label]
            gold[:, idx] = series.apply(lambda arr: int(label in arr)).values

    return gold


# ── Métriques par groupe sémantique ─────────────────────────────────────
def compute_group_metrics(gold, pred):
    results = {}
    for group, indices in GROUP_INDICES.items():
        g = gold[:, indices]
        p = pred[:, indices]
        results[group] = {
            "display_name": GROUP_DISPLAY_NAMES[group],
            "labels": LABEL_GROUPS[group],
            "macro_f1":   round(f1_score(g, p, average="macro", zero_division=0), 3),
            "precision":  round(precision_score(g, p, average="macro", zero_division=0), 3),
            "recall":     round(recall_score(g, p, average="macro", zero_division=0), 3),
        }
    return results


# ── Main ────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(PARQUET_PATH):
        sys.exit(f"ERREUR : Fichier Parquet introuvable à {PARQUET_PATH}")

    # 1. Gold
    df = pd.read_parquet(PARQUET_PATH)
    gold = build_gold(df)
    N = len(df)
    print(f"Gold chargé : {N} phrases, 19 labels")

    # 2. Modèle
    model_dir = os.path.join(os.path.dirname(__file__), "model_onnx")
    try:
        predictor = get_predictor(model_dir=model_dir)
    except Exception as e:
        sys.exit(f"ERREUR lors du chargement du modèle : {e}")

    # 3. Inputs formatés (contexte utilisé car présent dans le parquet)
    texts = format_inputs(df)

    # 4. Inférence
    print(f"Inférence sur {N} phrases (batch_size={BATCH_SIZE})…")
    probs = predictor.predict_texts(texts, batch_size=BATCH_SIZE)
    pred = (probs >= THRESHOLD).astype(np.int8)
    print(f"Inférence terminée — shape: {probs.shape}")

    # 5. Métriques par groupe
    group_metrics = compute_group_metrics(gold, pred)

    for group, m in group_metrics.items():
        print(f"\n── {m['display_name']} ({', '.join(m['labels'])}) ──")
        print(f"   Macro F1  : {m['macro_f1']}")
        print(f"   Precision : {m['precision']}")
        print(f"   Recall    : {m['recall']}")

    # 6. Export JSON
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {
        "source": os.path.basename(PARQUET_PATH),
        "n_samples": N,
        "threshold": THRESHOLD,
        "group_metrics": group_metrics,
    }
    out = os.path.join(OUT_DIR, "emotyc_parquet_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Résumé exporté : {out}")


if __name__ == "__main__":
    main()