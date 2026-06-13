#!/usr/bin/env python3
"""
Baseline classifiers for CyberAggAdo emotion classification.
Compares TF-IDF + SVM and TF-IDF + RandomForest against EMOTYC.

Uses stratified 5-fold cross-validation on CyberAggAdo (781 samples)
to give a fair estimate. Since the entire corpus is used both for
training and evaluation, these baselines have a significant advantage
over EMOTYC (zero-shot OOD), making the comparison informative:
if EMOTYC beats trained baselines, the transfer is strong;
if baselines beat EMOTYC, it confirms the domain gap.

Labels evaluated: the 19 EMOTYC labels (Emo, 4 modes, 2 types, 12 categories).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────

XLSX_PATH = Path(__file__).parent / "golds" / "CyberAdoAgg_gold_global_total.xlsx"

LABELS_19 = [
    "Emo",
    "Comportementale", "Designee", "Montree", "Suggeree",
    "Base", "Complexe",
    "Admiration", "Autre", "Colere", "Culpabilite",
    "Degout", "Embarras", "Fierte", "Jalousie",
    "Joie", "Peur", "Surprise", "Tristesse",
]

EMOTION_LABELS = [
    "Admiration", "Autre", "Colere", "Culpabilite",
    "Degout", "Embarras", "Fierte", "Jalousie",
    "Joie", "Peur", "Surprise", "Tristesse",
]

MODE_LABELS = ["Comportementale", "Designee", "Montree", "Suggeree"]
TYPE_LABELS = ["Base", "Complexe"]
META_LABELS = ["Emo"]

OUT_DIR = Path(__file__).parent / "results" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTYC_SUMMARY_PATH = (
    Path(__file__).parent
    / "results"
    / "All_cyberadoagg_context"
    / "emotyc_predictions_summary.json"
)

N_FOLDS = 5
RANDOM_STATE = 42


def load_data():
    """Load gold data and extract text + labels."""
    df = pd.read_excel(XLSX_PATH)
    missing_labels = [label for label in LABELS_19 if label not in df.columns]
    if missing_labels:
        missing = ", ".join(missing_labels)
        raise ValueError(f"Gold file is missing expected label column(s): {missing}")

    texts = df["TEXT"].fillna("").astype(str).tolist()
    # Replace any remaining "nan" strings
    texts = [t if t != "nan" else "" for t in texts]

    # Build gold matrix (N, 19)
    gold = np.zeros((len(df), len(LABELS_19)), dtype=int)
    for j, label in enumerate(LABELS_19):
        gold[:, j] = df[label].fillna(0).astype(int).values

    return texts, gold, df


def load_emotyc_reference():
    """Load EMOTYC reference metrics from the CyberAggAdo summary JSON."""
    with open(EMOTYC_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    global_metrics = summary.get("global_metrics", {})
    missing_metrics = [
        metric for metric in ("macro_f1", "micro_f1") if metric not in global_metrics
    ]
    if missing_metrics:
        missing = ", ".join(missing_metrics)
        raise ValueError(f"EMOTYC summary is missing global metric(s): {missing}")

    per_label_rows = summary.get("per_label", [])
    per_label_f1 = {row["label"]: row["f1"] for row in per_label_rows}
    missing_labels = [label for label in LABELS_19 if label not in per_label_f1]
    if missing_labels:
        missing = ", ".join(missing_labels)
        raise ValueError(f"EMOTYC summary is missing expected label(s): {missing}")

    return {
        "macro_f1": global_metrics["macro_f1"],
        "micro_f1": global_metrics["micro_f1"],
        "exact_match": global_metrics.get("exact_match"),
        "per_label_f1": {label: per_label_f1[label] for label in LABELS_19},
        "source": str(EMOTYC_SUMMARY_PATH),
    }


def evaluate_per_label(y_true, y_pred, labels):
    """Compute per-label and group metrics."""
    results = {}
    for j, label in enumerate(labels):
        tp = int(np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 1)))
        fp = int(np.sum((y_true[:, j] == 0) & (y_pred[:, j] == 1)))
        fn = int(np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 0)))
        tn = int(np.sum((y_true[:, j] == 0) & (y_pred[:, j] == 0)))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results[label] = {
            "gold_positive": int(np.sum(y_true[:, j])),
            "pred_positive": int(np.sum(y_pred[:, j])),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }

    # Global metrics
    macro_f1 = np.mean([results[l]["f1"] for l in labels])
    micro_tp = sum(results[l]["tp"] for l in labels)
    micro_fp = sum(results[l]["fp"] for l in labels)
    micro_fn = sum(results[l]["fn"] for l in labels)
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Exact match
    exact = int(np.all(y_true == y_pred, axis=1).sum())
    exact_ratio = exact / len(y_true)

    # Group F1s
    group_f1 = {}
    for name, group_labels in [("meta", META_LABELS), ("modes", MODE_LABELS),
                                ("types", TYPE_LABELS), ("emotions", EMOTION_LABELS)]:
        idxs = [labels.index(l) for l in group_labels if l in labels]
        if idxs:
            group_f1[name] = round(np.mean([results[labels[i]]["f1"] for i in idxs]), 4)

    return {
        "per_label": results,
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "exact_match": round(exact_ratio, 4),
        "exact_match_count": exact,
        "n_samples": len(y_true),
        "group_f1": group_f1,
    }


def run_cross_validation(texts, gold, model_name, model_factory, tfidf_params=None):
    """Run N-fold cross-validation for a multi-label classifier."""
    if tfidf_params is None:
        tfidf_params = {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "sublinear_tf": True,
            "min_df": 2,
        }

    # Use KFold since StratifiedKFold doesn't work directly with multi-label
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_preds = np.zeros_like(gold)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(texts)):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train = gold[train_idx]
        y_test = gold[test_idx]

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(**tfidf_params)
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        # Train one classifier per label
        y_pred = np.zeros_like(y_test)
        for j in range(gold.shape[1]):
            if y_train[:, j].sum() == 0:
                # No positive samples for this label in train fold
                y_pred[:, j] = 0
                continue

            if y_train[:, j].sum() < 3:
                # Too few samples, predict 0
                y_pred[:, j] = 0
                continue

            clf = model_factory()
            clf.fit(X_train, y_train[:, j])
            y_pred[:, j] = clf.predict(X_test)

        all_preds[test_idx] = y_pred

        fold_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_results.append({
            "fold": fold_idx + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "macro_f1": round(fold_macro_f1, 4),
        })
        print(f"  Fold {fold_idx+1}/{N_FOLDS}: macro-F1 = {fold_macro_f1:.4f}")

    # Global evaluation on all folds combined
    results = evaluate_per_label(gold, all_preds, LABELS_19)
    results["model"] = model_name
    results["method"] = "5-fold cross-validation"
    results["fold_results"] = fold_results
    results["tfidf_params"] = tfidf_params

    return results


def main():
    print("")
    print("  Baseline Classifiers — CyberAggAdo")
    print("")

    texts, gold, df = load_data()
    emotyc_reference = load_emotyc_reference()
    print(f"\nLoaded {len(texts)} samples, {gold.shape[1]} labels")
    print(f"Label prevalences:")
    for j, label in enumerate(LABELS_19):
        n_pos = int(gold[:, j].sum())
        print(f"  {label:20s}: {n_pos:4d} ({n_pos/len(texts)*100:5.1f}%)")

    all_results = {}

    # ── Baseline 1: TF-IDF + LinearSVC ────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Model 1: TF-IDF + LinearSVC (One-vs-Rest)")
    print(f"{'─' * 70}")

    svm_results = run_cross_validation(
        texts, gold, "TF-IDF + LinearSVC",
        lambda: LinearSVC(class_weight="balanced", max_iter=5000, C=1.0, random_state=RANDOM_STATE),
    )
    all_results["tfidf_svm"] = svm_results
    print(f"\n  Global: macro-F1 = {svm_results['macro_f1']}, micro-F1 = {svm_results['micro_f1']}, "
          f"exact match = {svm_results['exact_match']}")

    # ── Baseline 2: TF-IDF + RandomForest ─────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Model 2: TF-IDF + RandomForest (One-vs-Rest)")
    print(f"{'─' * 70}")

    rf_results = run_cross_validation(
        texts, gold, "TF-IDF + RandomForest",
        lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", max_depth=None,
            min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    )
    all_results["tfidf_rf"] = rf_results
    print(f"\n  Global: macro-F1 = {rf_results['macro_f1']}, micro-F1 = {rf_results['micro_f1']}, "
          f"exact match = {rf_results['exact_match']}")

    # ── Baseline 3: TF-IDF + LinearSVC with char n-grams ─────────────────
    print("")
    print("  Model 3: TF-IDF (char 2-5) + LinearSVC")
    print("")

    svm_char_results = run_cross_validation(
        texts, gold, "TF-IDF (char) + LinearSVC",
        lambda: LinearSVC(class_weight="balanced", max_iter=5000, C=1.0, random_state=RANDOM_STATE),
        tfidf_params={
            "analyzer": "char_wb",
            "ngram_range": (2, 5),
            "max_features": 20000,
            "sublinear_tf": True,
            "min_df": 2,
        },
    )
    all_results["tfidf_char_svm"] = svm_char_results
    print(f"\n  Global: macro-F1 = {svm_char_results['macro_f1']}, micro-F1 = {svm_char_results['micro_f1']}, "
          f"exact match = {svm_char_results['exact_match']}")

    # ── Summary comparison ────────────────────────────────────────────────
    print("")
    print("  SUMMARY: Baseline Classifiers vs EMOTYC")
    print("")

    # EMOTYC results from results/All_cyberadoagg_context/emotyc_predictions_summary.json
    emotyc_macro = emotyc_reference["macro_f1"]
    emotyc_micro = emotyc_reference["micro_f1"]
    emotyc_exact = emotyc_reference["exact_match"]
    emotyc_exact_display = (
        f"{emotyc_exact:>12.4f}" if emotyc_exact is not None else f"{'n/a':>12s}"
    )

    print(f"\n  {'Model':<35s} {'Macro-F1':>10s} {'Micro-F1':>10s} {'Exact Match':>12s} {'Training':>20s}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*12} {'─'*20}")
    print(
        f"  {'EMOTYC (zero-shot OOD)':<35s} {emotyc_macro:>10.4f} "
        f"{emotyc_micro:>10.4f} {emotyc_exact_display} "
        f"{'TTK (27k, other domain)':>20s}"
    )

    for key, label in [("tfidf_svm", "TF-IDF + SVM"),
                       ("tfidf_rf", "TF-IDF + RF"),
                       ("tfidf_char_svm", "TF-IDF (char) + SVM")]:
        r = all_results[key]
        print(
            f"  {label:<35s} {r['macro_f1']:>10.4f} "
            f"{r['micro_f1']:>10.4f} {r['exact_match']:>12.4f} "
            f"{'CyberAggAdo (5-fold CV)':>20s}"
        )

    # ── Per-label comparison table ────────────────────────────────────────
    print(f"\n  Per-label F1 comparison:")
    print(f"  {'Label':<20s} {'EMOTYC':>8s} {'SVM':>8s} {'RF':>8s} {'Char-SVM':>8s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    emotyc_f1 = emotyc_reference["per_label_f1"]

    for label in LABELS_19:
        ef1 = emotyc_f1[label]
        sf1 = all_results["tfidf_svm"]["per_label"][label]["f1"]
        rf1 = all_results["tfidf_rf"]["per_label"][label]["f1"]
        cf1 = all_results["tfidf_char_svm"]["per_label"][label]["f1"]
        best = max(ef1, sf1, rf1, cf1)
        markers = ""
        if ef1 == best and ef1 > 0:
            markers = " ← EMOTYC best"
        print(f"  {label:<20s} {ef1:>8.3f} {sf1:>8.3f} {rf1:>8.3f} {cf1:>8.3f}{markers}")

    # ── Group-level comparison ────────────────────────────────────────────
    print(f"\n  Group macro-F1 comparison:")
    print(f"  {'Group':<20s} {'EMOTYC':>8s} {'SVM':>8s} {'RF':>8s} {'Char-SVM':>8s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    emotyc_groups = {
        name: round(float(np.mean([emotyc_f1[label] for label in group_labels])), 4)
        for name, group_labels in [
            ("meta", META_LABELS),
            ("modes", MODE_LABELS),
            ("types", TYPE_LABELS),
            ("emotions", EMOTION_LABELS),
        ]
    }
    for group in ["meta", "modes", "types", "emotions"]:
        eg = emotyc_groups[group]
        sg = all_results["tfidf_svm"]["group_f1"].get(group, 0.0)
        rg = all_results["tfidf_rf"]["group_f1"].get(group, 0.0)
        cg = all_results["tfidf_char_svm"]["group_f1"].get(group, 0.0)
        print(f"  {group:<20s} {eg:>8.3f} {sg:>8.3f} {rg:>8.3f} {cg:>8.3f}")

    # ── Export JSON ───────────────────────────────────────────────────────
    output = {
        "description": "Baseline classifiers for CyberAggAdo emotion classification (5-fold CV)",
        "n_samples": len(texts),
        "n_labels": len(LABELS_19),
        "labels": LABELS_19,
        "models": all_results,
        "emotyc_reference": {
            "macro_f1": emotyc_macro,
            "micro_f1": emotyc_micro,
            "exact_match": emotyc_exact,
            "per_label_f1": emotyc_f1,
            "group_f1": emotyc_groups,
            "source": emotyc_reference["source"],
            "training_data": "TextToKids (27,911 samples, different domain)",
            "note": (
                "Zero-shot OOD, no training on CyberAggAdo; exact_match is null "
                "because the source summary does not include it"
            ),
        },
    }

    out_path = OUT_DIR / "baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved to: {out_path}")

if __name__ == "__main__":
    main()
