"""
Module commun EMOTYC centralisant les constantes, la configuration des étiquettes,
les métriques d'évaluation et l'inférence via ONNX Runtime.
"""
import os
import json
import math
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
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
MODEL_DOWNLOAD_HINT = (
    "Téléchargez-le depuis le dossier Eval-EMOTYC avec : "
    "bash setup.sh"
)

@dataclass(frozen=True)
class EmotycPredictor:
    session: Any
    tokenizer: Tokenizer
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

def _is_git_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False

def get_predictor(model_dir: str = DEFAULT_MODEL_DIR, intra_threads: int = 2) -> EmotycPredictor:
    """Charge le modèle ONNX et le tokenizer."""
    onnx_path = os.path.join(model_dir, "model.onnx")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"Fichier modèle introuvable à {onnx_path}. {MODEL_DOWNLOAD_HINT}"
        )
    if _is_git_lfs_pointer(onnx_path):
        raise RuntimeError(
            f"{onnx_path} est un pointeur Git LFS, pas le modèle ONNX complet. "
            f"{MODEL_DOWNLOAD_HINT}"
        )
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Fichier tokenizer introuvable à {tokenizer_path}")

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

def format_input(sentence: str, prev_sentence: str = None, next_sentence: str = None,
                 use_context: bool = False, template: str = "bca") -> str:
    """
    Formate l'input selon le template BCA.
    template='bca'        : before:{prev}</s>current:{s}</s>after:{next}</s>
    template='bca_spaced' : before:{prev}</s>current: {s}</s>after:{next}</s>
    """
    eos = "</s>"
    current_sep = " " if template == "bca_spaced" else ""
    if use_context:
        prev = prev_sentence or eos
        nxt = next_sentence or eos
        return f"before:{prev}{eos}current:{current_sep}{sentence}{eos}after:{nxt}{eos}"
    return f"before:{eos}current:{current_sep}{sentence}{eos}after:{eos}"

def load_gold_xlsx(xlsx_path: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Charge le fichier XLSX du gold.
    Retourne (df, sentences, gold_matrix)."""
    df = pd.read_excel(xlsx_path)
    if "TEXT" not in df.columns:
        raise ValueError("ERREUR : colonne 'TEXT' absente.")
    missing = [l for l in ALL_LABELS if l not in df.columns]
    if missing:
        raise ValueError(f"ERREUR : colonnes EMOTYC manquantes ({len(missing)}/19) : {missing}")

    sentences = df["TEXT"].fillna("").astype(str).tolist()
    gold = df[ALL_LABELS].astype(int).to_numpy()
    return df, sentences, gold

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
    exact_match = np.all(gold == pred, axis=1).mean() if gold.size > 0 else 0.0

    return results, {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "exact_match": round(float(exact_match), 4),
        "n_samples": len(gold),
        "n_labels": len(label_names),
    }
