from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "slides" / "figures" / "plots"
TABLES_DIR = REPO_ROOT / "slides" / "figures" / "tables"
RESULTS_DIR = REPO_ROOT / "results"

EMOTION_LABELS = [
    "Admiration",
    "Autre",
    "Colere",
    "Culpabilite",
    "Degout",
    "Embarras",
    "Fierte",
    "Jalousie",
    "Joie",
    "Peur",
    "Surprise",
    "Tristesse",
]

DISPLAY_LABELS = {
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

EMOTION_COLORS = {
    "Admiration": "#1abc9c",
    "Autre": "#95a5a6",
    "Colere": "#e74c3c",
    "Culpabilite": "#8e44ad",
    "Degout": "#2ecc71",
    "Embarras": "#e84393",
    "Fierte": "#e67e22",
    "Jalousie": "#7f8c1f",
    "Joie": "#f1c40f",
    "Peur": "#9b59b6",
    "Surprise": "#00a8cc",
    "Tristesse": "#3498db",
}

CORPORA = {
    "CyberAdoAgg": REPO_ROOT / "golds" / "CyberAdoAgg_gold_global_total.xlsx",
    "EmoTextToKids": REPO_ROOT / "golds" / "emotexttokids_gold_flat.xlsx",
}

PERFORMANCE_JSONS = {
    "CyberAdoAgg": REPO_ROOT
    / "results"
    / "All_cyberadoagg_context"
    / "emotyc_predictions_summary.json",
    "EmoTextToKids": REPO_ROOT
    / "results"
    / "TextToKids"
    / "ContextTemplateAvecEspace"
    / "emotyc_predictions_summary.json",
}


def save_emotion_pie(counts, output_name):
    plot_df = pd.DataFrame(
        {"label": EMOTION_LABELS, "count": [counts.get(label, 0) for label in EMOTION_LABELS]}
    )
    plot_df = plot_df[plot_df["count"] > 0]
    sizes = plot_df["count"].tolist()
    colors = [EMOTION_COLORS[label] for label in plot_df["label"].tolist()]

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    ax.axis("equal")
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(PLOTS_DIR / output_name, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_static_reference_pies():
    save_emotion_pie(
        {
            "Colere": 45,
            "Peur": 19,
            "Tristesse": 13,
            "Joie": 7,
            "Fierte": 5,
            "Autre": 11,
        },
        "emotions_extremiste.pdf",
    )
    save_emotion_pie(
        {
            "Tristesse": 25,
            "Colere": 20,
            "Peur": 17,
            "Joie": 16,
            "Fierte": 4,
            "Autre": 18,
        },
        "emotions_non_extremiste.pdf",
    )


def extract_emotion_distribution(corpus_name, path):
    df = pd.read_excel(path)
    missing = [label for label in EMOTION_LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"{path} ne contient pas les colonnes attendues: {missing}")

    counts = {}
    for label in EMOTION_LABELS:
        values = pd.to_numeric(df[label], errors="coerce").fillna(0)
        counts[label] = int(values.gt(0).sum())

    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"Aucune activation émotionnelle trouvée dans {path}")

    rows = []
    for label in EMOTION_LABELS:
        count = counts[label]
        rows.append(
            {
                "corpus": corpus_name,
                "label": label,
                "label_fr": DISPLAY_LABELS[label],
                "count": count,
                "proportion": count / total,
                "proportion_percent": count / total * 100,
                "n_rows": len(df),
                "n_emotion_activations": total,
            }
        )
    return pd.DataFrame(rows)


def save_xlsx_pie(distribution, output_name):
    counts = dict(zip(distribution["label"], distribution["count"], strict=True))
    save_emotion_pie(counts, output_name)


def latex_escape(value):
    return str(value).replace("_", r"\_")


def percent_fr(value):
    return f"{value:.1f}".replace(".", ",") + r"\,\%"


def save_latex_table(distributions):
    cyber = distributions["CyberAdoAgg"].set_index("label")
    ttk = distributions["EmoTextToKids"].set_index("label")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Émotion} & \multicolumn{2}{c}{\textbf{EmoTextToKids}} & \multicolumn{2}{c}{\textbf{CyberAggAdo}} \\",
        r" & \textbf{n} & \textbf{\%} & \textbf{n} & \textbf{\%} \\",
        r"\midrule",
    ]
    for label in EMOTION_LABELS:
        cyber_row = cyber.loc[label]
        ttk_row = ttk.loc[label]
        lines.append(
            f"{latex_escape(DISPLAY_LABELS[label])} & "
            f"{int(ttk_row['count'])} & {percent_fr(ttk_row['proportion_percent'])} & "
            f"{int(cyber_row['count'])} & {percent_fr(cyber_row['proportion_percent'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLES_DIR / "emotion_proportions_table.tex").write_text("\n".join(lines), encoding="utf-8")


def format_int_fr(value):
    return f"{int(value):,}".replace(",", r"\,")


def format_float_fr(value):
    return f"{value:.2f}".replace(".", ",")


def read_micro_metrics(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    per_label = data["per_label"]
    tp = sum(row["tp"] for row in per_label)
    fp = sum(row["fp"] for row in per_label)
    fn = sum(row["fn"] for row in per_label)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        "n_samples": data["n_samples"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_corpus_sizes_table(distributions):
    ttk_rows = int(distributions["EmoTextToKids"]["n_rows"].iloc[0])
    cyber_rows = int(distributions["CyberAdoAgg"]["n_rows"].iloc[0])
    lines = [
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"\textbf{Référence} & \textbf{Corpus} & \textbf{Taille} \\",
        r"\midrule",
        rf"Etienne (2023) & EmoTextToKids & {format_int_fr(ttk_rows)} phrases \\",
        rf"Dragos et al. (2022) & Textes extrémistes / radicaux & {format_int_fr(1129)} textes \\",
        rf"Dragos et al. (2022) & Textes non-extrémistes / non-radicaux & {format_int_fr(599)} textes \\",
        rf"Annotations CyberAggAdo & CyberAggAdo & {format_int_fr(cyber_rows)} messages \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (TABLES_DIR / "corpus_sizes_table.tex").write_text("\n".join(lines), encoding="utf-8")


def save_emotyc_performance_table():
    ttk = read_micro_metrics(PERFORMANCE_JSONS["EmoTextToKids"])
    cyber = read_micro_metrics(PERFORMANCE_JSONS["CyberAdoAgg"])
    rows = [
        ("Etienne (2023)", "EmoTextToKids", ttk["precision"], ttk["recall"], ttk["f1"]),
        ("Dragos et al. (2022)", "Extrémisme", 0.80, 0.25, 0.38),
        ("Annotations CyberAggAdo", "CyberAggAdo", cyber["precision"], cyber["recall"], cyber["f1"]),
    ]
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Référence} & \textbf{Corpus} & \textbf{Précision} & \textbf{Rappel} & \textbf{F1} \\",
        r"\midrule",
    ]
    for ref, corpus, precision, recall, f1 in rows:
        lines.append(
            f"{latex_escape(ref)} & {latex_escape(corpus)} & "
            f"{format_float_fr(precision)} & {format_float_fr(recall)} & {format_float_fr(f1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLES_DIR / "emotyc_performance_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    save_static_reference_pies()

    distributions = {
        corpus_name: extract_emotion_distribution(corpus_name, path)
        for corpus_name, path in CORPORA.items()
    }
    all_distributions = pd.concat(distributions.values(), ignore_index=True)
    all_distributions.to_csv(RESULTS_DIR / "emotion_proportions.csv", index=False)

    save_xlsx_pie(distributions["CyberAdoAgg"], "emotions_cyberadoagg.pdf")
    save_xlsx_pie(distributions["EmoTextToKids"], "emotions_emotexttokids.pdf")
    save_latex_table(distributions)
    save_corpus_sizes_table(distributions)
    save_emotyc_performance_table()

    for output in [
        PLOTS_DIR / "emotions_extremiste.pdf",
        PLOTS_DIR / "emotions_non_extremiste.pdf",
        PLOTS_DIR / "emotions_cyberadoagg.pdf",
        PLOTS_DIR / "emotions_emotexttokids.pdf",
        RESULTS_DIR / "emotion_proportions.csv",
        TABLES_DIR / "emotion_proportions_table.tex",
        TABLES_DIR / "corpus_sizes_table.tex",
        TABLES_DIR / "emotyc_performance_table.tex",
    ]:
        print(output.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
