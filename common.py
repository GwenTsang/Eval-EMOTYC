"""
Module commun centralisant les constantes, la configuration des étiquettes,
les métriques d'évaluation et l'inférence.
"""
from __future__ import annotations

import os
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tokenizers import Tokenizer

#  LABELS (noms canoniques, sans accents, ordre du modèle)
EMOTYC_LABEL2ID = {
    "Emo": 0, "Comportementale": 1, "Designee": 2, "Montree": 3,
    "Suggeree": 4, "Base": 5, "Complexe": 6, "Admiration": 7,
    "Autre": 8, "Colere": 9, "Culpabilite": 10, "Degout": 11,
    "Embarras": 12, "Fierte": 13, "Jalousie": 14, "Joie": 15,
    "Peur": 16, "Surprise": 17, "Tristesse": 18,
}

ALL_LABELS = list(EMOTYC_LABEL2ID.keys())  # 19 labels
LABELS_19 = ALL_LABELS

#  GROUPES SÉMANTIQUES

META_LABELS = ["Emo"]

EMOTION_LABELS = [
    "Admiration", "Autre", "Colere", "Culpabilite", "Degout",
    "Embarras", "Fierte", "Jalousie", "Joie", "Peur",
    "Surprise", "Tristesse",
]

MODE_LABELS = ["Comportementale", "Designee", "Montree", "Suggeree"]

TYPE_LABELS = ["Base", "Complexe"]

# Dictionnaire de groupes pour itération programmatique
LABEL_GROUPS = {
    "emo":     META_LABELS,
    "emotion": EMOTION_LABELS,
    "mode":    MODE_LABELS,
    "type":    TYPE_LABELS,
}

EVALUATION_LABEL_GROUPS = {
    "meta": META_LABELS,
    "modes": MODE_LABELS,
    "types": TYPE_LABELS,
    "emotions": EMOTION_LABELS,
}

#  SEUILS PAR DÉFAUT
THRESHOLD = 0.5

#  NOMS D'AFFICHAGE ET NORMALISATION

DISPLAY_NAMES = {
    "Emo": "Émo",
    "Comportementale": "Comportementale",
    "Designee": "Désignée",
    "Montree": "Montrée",
    "Suggeree": "Suggérée",
    "Base": "Base",
    "Complexe": "Complexe",
    "Admiration": "Admiration",
    "Autre": "Autre",
    "Colere": "Colère",
    "Culpabilite": "Culpabilité",
    "Degout": "Dégoût",
    "Embarras": "Embarras",
    "Fierte": "Fierté",
    "Jalousie": "Jalousie",
    "Joie": "Joie",
    "Peur": "Peur",
    "Surprise": "Surprise",
    "Tristesse": "Tristesse",
}

GROUP_DISPLAY_NAMES = {
    "emo":     "Caractère émotionnel (Émo)",
    "emotion": "Émotions",
    "mode":    "Modes d'expression",
    "type":    "Types (Base / Complexe)",
}

#  INFÉRENCE ONNX RUNTIME
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_onnx")

@dataclass(frozen=True)
class EmotycPredictor:
    session: Any
    tokenizer: "Tokenizer"
    labels: list[str]
    input_names: set[str]
    pad_id: int

    def predict_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Inférence par batch. Retourne une matrice (N, 19) de probabilités sigmoid."""
        all_probs: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = _encode_batch(
                tokenizer=self.tokenizer,
                texts=batch_texts,
                pad_id=self.pad_id,
                input_names=self.input_names,
            )
            logits = np.asarray(self.session.run(None, inputs)[0], dtype=np.float32)
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)
            probs = _sigmoid(logits)
            all_probs.append(probs)

        if not all_probs:
            return np.empty((0, len(self.labels)), dtype=np.float32)
        return np.vstack(all_probs)

def _sigmoid(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out

def _encode_batch(
    tokenizer: Tokenizer,
    texts: list[str],
    pad_id: int,
    input_names: set[str],
) -> dict[str, np.ndarray]:
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    max_len = max((len(encoding.ids) for encoding in encodings), default=1)
    max_len = max(max_len, 1)

    input_ids = np.full((len(encodings), max_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)

    for row_index, encoding in enumerate(encodings):
        ids = encoding.ids or [pad_id]
        input_ids[row_index, : len(ids)] = ids
        attention_mask[row_index, : len(ids)] = 1

    inputs: dict[str, np.ndarray] = {"input_ids": input_ids}
    if "attention_mask" in input_names:
        inputs["attention_mask"] = attention_mask
    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
    return inputs

def get_predictor(model_dir: str = DEFAULT_MODEL_DIR, intra_threads: int = 2) -> EmotycPredictor:
    """Charge le modèle ONNX et le tokenizer."""
    from tokenizers import Tokenizer

    onnx_path = os.path.join(model_dir, "model.onnx")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    config_path = os.path.join(model_dir, "config.json")

    # Charger la config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    raw = config.get("id2label", {})
    labels = [raw.get(str(index), f"LABEL_{index}") for index in range(len(raw))]
    if len(labels) != 19:
        raise RuntimeError("Configuration modèle invalide : 19 labels attendus.")

    # Charger le tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=512)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 1

    import onnxruntime as ort

    # Configurer la session ONNX Runtime
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    options.log_severity_level = 3

    # Utiliser le GPU CUDA si disponible
    providers = ["CPUExecutionProvider"]
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(onnx_path, sess_options=options, providers=providers)
    input_names = {inp.name for inp in session.get_inputs()}

    return EmotycPredictor(
        session=session,
        tokenizer=tokenizer,
        labels=labels,
        input_names=input_names,
        pad_id=int(pad_id),
    )

# ═══════════════════════════════════════════════════════════════════════════
#  FORMATTAGE DE L'ENTRÉE & CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def format_input(
    sentence: str,
    prev_sentence: str = None,
    next_sentence: str = None,
    use_context: bool = False,
) -> str:
    """Formate l'input au format BCA espacé."""
    eos = "</s>"
    if use_context:
        prev = prev_sentence or eos
        nxt = next_sentence or eos
        return f"before:{prev}{eos}current: {sentence}{eos}after:{nxt}{eos}"
    return f"before:{eos}current: {sentence}{eos}after:{eos}"

def load_gold_xlsx(xlsx_path: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Charge le fichier XLSX du gold.
    Retourne (df, sentences, gold_matrix)."""
    df = pd.read_excel(xlsx_path)
    if "TEXT" not in df.columns:
        raise ValueError("ERREUR : colonne 'TEXT' absente.")

    sentences = df["TEXT"].fillna("").astype(str).tolist()
    gold = labels_to_gold_matrix(df)
    return df, sentences, gold

def labels_to_gold_matrix(df: pd.DataFrame, label_names: list[str] = ALL_LABELS) -> np.ndarray:
    """Extrait une matrice binaire ordonnée depuis un DataFrame gold."""
    missing = [label for label in label_names if label not in df.columns]
    if missing:
        raise ValueError(f"ERREUR : colonnes EMOTYC manquantes ({len(missing)}/19) : {missing}")
    return df[label_names].fillna(0).astype(int).to_numpy()

def build_context_texts(
    sentences: list[str],
    use_context: bool,
    template: str = "bca_spaced",
) -> list[str]:
    """Construit les inputs BCA avec ou sans phrases voisines."""
    n = len(sentences)
    return [
        format_input(
            sentences[i],
            sentences[i - 1] if i > 0 and use_context else None,
            sentences[i + 1] if i < n - 1 and use_context else None,
            use_context,
        )
        for i in range(n)
    ]

# ═══════════════════════════════════════════════════════════════════════════
#  CALCUL DE MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(gold: np.ndarray, pred: np.ndarray, label_names: list[str]) -> tuple[list[dict], dict]:
    """Calcule les métriques d'évaluation par label et globales."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        cohen_kappa_score,
    )

    results = []
    for j, label in enumerate(label_names):
        g, p = gold[:, j], pred[:, j]
        tp = int(((g == 1) & (p == 1)).sum())
        fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        tn = int(((g == 0) & (p == 0)).sum())

        acc = accuracy_score(g, p)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                kappa = cohen_kappa_score(g, p, labels=[0, 1])
        except Exception:
            kappa = float("nan")
        f1 = f1_score(g, p, zero_division=0)
        prec = precision_score(g, p, zero_division=0)
        rec = recall_score(g, p, zero_division=0)

        results.append({
            "label": label,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4),
            "kappa": round(kappa, 4) if not math.isnan(kappa) else None,
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "prevalence_gold": round(g.sum() / len(g), 4),
            "prevalence_pred": round(p.sum() / len(p), 4),
        })

    macro_f1 = np.mean([r["f1"] for r in results]) if results else 0.0
    micro_f1 = f1_score(gold.ravel(), pred.ravel(), zero_division=0) if gold.size > 0 else 0.0
    exact_match_count = int(np.all(gold == pred, axis=1).sum()) if gold.size > 0 else 0
    exact_match = exact_match_count / len(gold) if len(gold) > 0 else 0.0

    return results, {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "exact_match": round(float(exact_match), 4),
        "exact_match_count": exact_match_count,
        "n_samples": len(gold),
        "n_labels": len(label_names),
    }

def compute_group_metrics(
    gold: np.ndarray,
    pred: np.ndarray,
    label_names: list[str] = ALL_LABELS,
    groups: dict[str, list[str]] = LABEL_GROUPS,
    display_names: dict[str, str] | None = GROUP_DISPLAY_NAMES,
    decimals: int = 3,
) -> dict[str, dict]:
    """Calcule précision, rappel et macro-F1 par groupe sémantique."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    results: dict[str, dict] = {}
    for group, labels in groups.items():
        indices = [label_names.index(label) for label in labels if label in label_names]
        if not indices:
            continue

        g = gold[:, indices]
        p = pred[:, indices]
        row = {
            "labels": [label_names[index] for index in indices],
            "macro_f1": round(float(f1_score(g, p, average="macro", zero_division=0)), decimals),
            "precision": round(float(precision_score(g, p, average="macro", zero_division=0)), decimals),
            "recall": round(float(recall_score(g, p, average="macro", zero_division=0)), decimals),
        }
        if display_names and group in display_names:
            row["display_name"] = display_names[group]
        results[group] = row
    return results

def compute_group_f1_from_per_label(
    per_label: list[dict] | dict[str, dict],
    groups: dict[str, list[str]] = EVALUATION_LABEL_GROUPS,
    decimals: int = 4,
) -> dict[str, float]:
    """Agrège les F1 par label en macro-F1 par groupe."""
    if isinstance(per_label, dict):
        by_label = per_label
    else:
        by_label = {row["label"]: row for row in per_label}

    grouped: dict[str, float] = {}
    for group, labels in groups.items():
        values = [
            float(by_label[label]["f1"])
            for label in labels
            if label in by_label and by_label[label].get("f1") is not None
        ]
        if values:
            grouped[group] = round(float(np.mean(values)), decimals)
    return grouped

def per_label_list_to_dict(rows: list[dict]) -> dict[str, dict]:
    """Indexe une liste de métriques per-label par nom de label."""
    return {row["label"]: {k: v for k, v in row.items() if k != "label"} for row in rows}

def round_metric_rows(rows: list[dict], decimals: int = 3) -> list[dict]:
    """Arrondit les champs numériques de métriques pour les exports lisibles."""
    rounded_rows = []
    for row in rows:
        rounded = dict(row)
        for key, value in rounded.items():
            if isinstance(value, float):
                rounded[key] = round(value, decimals)
        rounded_rows.append(rounded)
    return rounded_rows

def compact_global_metrics(
    global_metrics: dict,
    decimals: int = 3,
    keys: tuple[str, ...] = ("micro_f1", "macro_f1"),
) -> dict:
    """Garde les métriques globales publiées dans les résumés historiques."""
    return {
        key: round(global_metrics[key], decimals)
        for key in keys
        if key in global_metrics and global_metrics[key] is not None
    }

def build_prediction_summary(
    *,
    source: str,
    n_samples: int,
    threshold: float,
    per_label: list[dict] | None = None,
    global_metrics: dict | None = None,
    template: str = "bca_spaced",
    extra: dict | None = None,
    source_key: str = "source_xlsx",
    decimals: int = 3,
) -> dict:
    """Construit le JSON résumé commun aux scripts d'inférence."""
    summary = {
        source_key: source,
        "n_samples": n_samples,
        "threshold": threshold,
        "template": template,
    }
    if per_label is not None:
        summary["per_label"] = round_metric_rows(per_label, decimals=decimals)
    if global_metrics is not None:
        summary["global_metrics"] = compact_global_metrics(global_metrics, decimals=decimals)
    if extra:
        summary.update(extra)
    return summary

def write_json(path: str | Path, payload: dict) -> None:
    """Écrit un JSON UTF-8 indenté après création du dossier parent."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════
#  UTILITAIRES HEATMAP / SUPPORT
# ═══════════════════════════════════════════════════════════════════════════

def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def gold_support(entry: dict) -> int:
    """Nombre d'instances positives dans le gold (TP + FN)."""
    return int((entry.get("tp", 0) or 0) + (entry.get("fn", 0) or 0))

def sample_count(summary: dict) -> int:
    """Retourne n_samples malgré les variations historiques des exports."""
    for section in (summary, summary.get("global_metrics") or {}):
        value = section.get("n_samples")
        if value is not None:
            return int(value)

    for entry in summary.get("per_label", []):
        counts = [entry.get(key) for key in ("tp", "fp", "fn", "tn")]
        if all(value is not None for value in counts):
            return int(sum(counts))

    raise KeyError("Impossible de déterminer n_samples dans le résumé JSON.")

def hsl_for_delta(delta: float) -> tuple[float, float, float]:
    """Palette séquentielle commune pour l'amplitude absolue d'un delta."""
    d = max(0.0, min(1.0, abs(delta)))
    retention = 1.0 - d

    if retention >= 0.5:
        t = 2.0 * (1.0 - retention)
        h = 145 + t * (35 - 145)
        s = 45 + t * (85 - 45)
        light = 88 + t * (65 - 88)
    else:
        t = 2.0 * (0.5 - retention)
        h = 35 + t * (0 - 35)
        s = 85 + t * (70 - 85)
        light = 65 + t * (42 - 65)

    return h, s, light

def css_delta_color(delta: float) -> str:
    h, s, light = hsl_for_delta(delta)
    return f"hsl({h:.0f}, {s:.0f}%, {light:.0f}%)"

def hex_for_delta(delta: float) -> str:
    import colorsys

    h, s, light = hsl_for_delta(delta)
    red, green, blue = colorsys.hls_to_rgb(h / 360.0, light / 100.0, s / 100.0)
    return f"{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"

def delta_text_color(delta: float, *, light: str = "#fff", dark: str = "#1a1a1a") -> str:
    return light if abs(delta) > 0.75 else dark

def format_delta(value: float, decimals: int = 2) -> str:
    if abs(value) < 0.5 * 10 ** (-decimals):
        return f"{0:.{decimals}f}"
    return f"{value:+.{decimals}f}"
