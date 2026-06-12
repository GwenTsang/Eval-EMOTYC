#!/usr/bin/env python3
"""Generate vector assets for the EMOTYC transferability Beamer slides."""

from __future__ import annotations

import argparse
import colorsys
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from emotyc_common import ALL_LABELS, DISPLAY_NAMES  # noqa: E402
from visualizations.delta_heatmap import (  # noqa: E402
    DEFAULT_CYBER,
    DEFAULT_TTK,
    METRICS,
    gold_support,
    sample_count,
)

METRIC_TEX = {
    "f1": r"$\Delta F_1$",
    "precision": r"$\Delta P$",
    "recall": r"$\Delta R$",
}
METRIC_SVG = {
    "f1": "ΔF1",
    "precision": "ΔPrécision",
    "recall": "ΔRappel",
}


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hsl_for_delta(delta: float) -> tuple[float, float, float]:
    """Same sequential palette as visualizations/delta_heatmap.py."""
    d = max(0.0, min(1.0, abs(delta)))
    r = 1.0 - d

    if r >= 0.5:
        t = 2.0 * (1.0 - r)
        h = 145 + t * (35 - 145)
        s = 45 + t * (85 - 45)
        light = 88 + t * (65 - 88)
    else:
        t = 2.0 * (0.5 - r)
        h = 35 + t * (0 - 35)
        s = 85 + t * (70 - 85)
        light = 65 + t * (42 - 65)

    return h, s, light


def hex_for_delta(delta: float) -> str:
    h, s, light = hsl_for_delta(delta)
    red, green, blue = colorsys.hls_to_rgb(h / 360.0, light / 100.0, s / 100.0)
    return f"{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"


def text_color(delta: float) -> str:
    return "white" if abs(delta) > 0.75 else "black"


def format_delta(value: float, decimals: int = 2) -> str:
    if abs(value) < 0.5 * 10 ** (-decimals):
        return f"{0:.{decimals}f}"
    return f"{value:+.{decimals}f}"


def format_int_tex(value: int) -> str:
    return f"{value:,}".replace(",", r"\,")


def format_int_svg(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def collect_data(cyber_path: Path, ttk_path: Path) -> dict:
    cyber = load_summary(cyber_path)
    ttk = load_summary(ttk_path)
    cyber_by_label = {entry["label"]: entry for entry in cyber["per_label"]}
    ttk_by_label = {entry["label"]: entry for entry in ttk["per_label"]}

    sorted_labels = sorted(
        ALL_LABELS,
        key=lambda label: gold_support(cyber_by_label.get(label, {})),
        reverse=True,
    )

    rows = []
    for label in sorted_labels:
        cyber_entry = cyber_by_label.get(label, {})
        ttk_entry = ttk_by_label.get(label, {})
        rows.append(
            {
                "label": DISPLAY_NAMES.get(label, label),
                "ttk_support": gold_support(ttk_entry),
                "cyber_support": gold_support(cyber_entry),
                "deltas": {
                    metric: (ttk_entry.get(metric, 0) or 0)
                    - (cyber_entry.get(metric, 0) or 0)
                    for metric in METRICS
                },
            }
        )

    ttk_n = sample_count(ttk)
    cyber_n = sample_count(cyber)
    global_rows = [
        {
            "label": "Macro F1",
            "ttk_support": ttk_n,
            "cyber_support": cyber_n,
            "delta": ttk["global_metrics"]["macro_f1"]
            - cyber["global_metrics"]["macro_f1"],
        },
        {
            "label": "Micro F1",
            "ttk_support": ttk_n,
            "cyber_support": cyber_n,
            "delta": ttk["global_metrics"]["micro_f1"]
            - cyber["global_metrics"]["micro_f1"],
        },
    ]

    return {
        "rows": rows,
        "global_rows": global_rows,
        "ttk_n": ttk_n,
        "cyber_n": cyber_n,
    }


def latex_heat_cell(value: float, decimals: int = 2) -> str:
    color = hex_for_delta(value)
    foreground = text_color(value)
    return (
        rf"\cellcolor[HTML]{{{color}}}"
        rf"\textcolor{{{foreground}}}{{{format_delta(value, decimals)}}}"
    )


def write_latex_table(data: dict, out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrccc}",
        r"\toprule",
        r"\textbf{Label} & \textbf{TTK} & \textbf{Cyber} & "
        + " & ".join(rf"\textbf{{{METRIC_TEX[metric]}}}" for metric in METRICS)
        + r" \\",
        r"\midrule",
    ]

    for row in data["rows"]:
        metric_cells = " & ".join(
            latex_heat_cell(row["deltas"][metric]) for metric in METRICS
        )
        lines.append(
            f"{latex_escape(row['label'])} & "
            f"{format_int_tex(row['ttk_support'])} & "
            f"{format_int_tex(row['cyber_support'])} & "
            f"{metric_cells} \\\\"
        )

    lines.extend(
        [
            r"\midrule",
            r"\rowcolor[HTML]{3B3B3F}",
            r"\multicolumn{6}{l}{\textcolor{white}{\textbf{Métriques globales}}} \\",
        ]
    )

    for row in data["global_rows"]:
        lines.append(
            f"{latex_escape(row['label'])} & "
            f"{format_int_tex(row['ttk_support'])} & "
            f"{format_int_tex(row['cyber_support'])} & "
            f"{latex_heat_cell(row['delta'], decimals=4)} & & \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    weight: int = 400,
    fill: str = "#1A1A1A",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        'dominant-baseline="middle">'
        f"{html.escape(value)}</text>"
    )


def write_svg(data: dict, out_path: Path) -> None:
    margin = 36
    table_x = margin
    table_y = 132
    col_widths = [170, 95, 95, 125, 125, 125]
    row_h = 32
    header_h = 56
    table_w = sum(col_widths)
    rows_total = len(data["rows"]) + 1 + len(data["global_rows"])
    table_h = header_h + row_h * rows_total
    width = max(900, table_x + table_w + margin)
    height = table_y + table_h + 92

    col_x = [table_x]
    for width_i in col_widths[:-1]:
        col_x.append(col_x[-1] + width_i)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<defs>",
        "<linearGradient id=\"deltaGradient\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"0%\">",
        '<stop offset="0%" stop-color="#D7EBDD"/>',
        '<stop offset="35%" stop-color="#B8E19A"/>',
        '<stop offset="55%" stop-color="#F2B959"/>',
        '<stop offset="78%" stop-color="#CB673D"/>',
        '<stop offset="100%" stop-color="#B82020"/>',
        "</linearGradient>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="#F7F7F8"/>',
        svg_text(
            margin,
            34,
            "Transferabilité d'EMOTYC : TextToKids → CyberAggAdo",
            size=23,
            weight=700,
        ),
        svg_text(
            margin,
            66,
            "Δ = performance TextToKids - performance CyberAggAdo ; couleur = amplitude absolue du delta",
            size=14,
            fill="#71717A",
        ),
        svg_text(
            margin,
            101,
            f"TextToKids : {format_int_svg(data['ttk_n'])} unités textuelles",
            size=14,
            weight=600,
            fill="#4F46E5",
        ),
        svg_text(
            margin + 330,
            101,
            f"CyberAggAdo : {format_int_svg(data['cyber_n'])} unités textuelles",
            size=14,
            weight=600,
            fill="#B45309",
        ),
        f'<rect x="{table_x}" y="{table_y}" width="{table_w}" height="{table_h}" '
        'rx="8" fill="#FFFFFF" stroke="#E2E2E5"/>',
        f'<rect x="{table_x}" y="{table_y}" width="{table_w}" height="{header_h}" '
        'rx="8" fill="#FAFAFA"/>',
    ]

    headers = ["Label", "TTK", "Cyber"] + [METRIC_SVG[metric] for metric in METRICS]
    for index, header in enumerate(headers):
        x = col_x[index] + (12 if index == 0 else col_widths[index] / 2)
        anchor = "start" if index == 0 else "middle"
        lines.append(svg_text(x, table_y + header_h / 2, header, size=13, weight=700, anchor=anchor))

    y = table_y + header_h
    for row in data["rows"]:
        lines.append(
            f'<line x1="{table_x}" y1="{y}" x2="{table_x + table_w}" y2="{y}" '
            'stroke="#E2E2E5"/>'
        )
        lines.append(svg_text(col_x[0] + 12, y + row_h / 2, row["label"], size=13, weight=600))
        lines.append(
            svg_text(
                col_x[1] + col_widths[1] / 2,
                y + row_h / 2,
                format_int_svg(row["ttk_support"]),
                size=13,
                fill="#71717A",
                anchor="middle",
            )
        )
        lines.append(
            svg_text(
                col_x[2] + col_widths[2] / 2,
                y + row_h / 2,
                format_int_svg(row["cyber_support"]),
                size=13,
                fill="#71717A",
                anchor="middle",
            )
        )

        for offset, metric in enumerate(METRICS, start=3):
            value = row["deltas"][metric]
            fill = f"#{hex_for_delta(value)}"
            fg = "#FFFFFF" if text_color(value) == "white" else "#1A1A1A"
            lines.append(
                f'<rect x="{col_x[offset]}" y="{y}" width="{col_widths[offset]}" '
                f'height="{row_h}" fill="{fill}"/>'
            )
            lines.append(
                svg_text(
                    col_x[offset] + col_widths[offset] / 2,
                    y + row_h / 2,
                    format_delta(value),
                    size=13,
                    weight=700,
                    fill=fg,
                    anchor="middle",
                )
            )
        y += row_h

    lines.append(
        f'<rect x="{table_x}" y="{y}" width="{table_w}" height="{row_h}" fill="#3B3B3F"/>'
    )
    lines.append(svg_text(table_x + 12, y + row_h / 2, "Métriques globales", size=13, weight=700, fill="#FFFFFF"))
    y += row_h

    for row in data["global_rows"]:
        lines.append(
            f'<line x1="{table_x}" y1="{y}" x2="{table_x + table_w}" y2="{y}" '
            'stroke="#E2E2E5"/>'
        )
        lines.append(svg_text(col_x[0] + 12, y + row_h / 2, row["label"], size=13, weight=600))
        lines.append(
            svg_text(
                col_x[1] + col_widths[1] / 2,
                y + row_h / 2,
                format_int_svg(row["ttk_support"]),
                size=13,
                fill="#71717A",
                anchor="middle",
            )
        )
        lines.append(
            svg_text(
                col_x[2] + col_widths[2] / 2,
                y + row_h / 2,
                format_int_svg(row["cyber_support"]),
                size=13,
                fill="#71717A",
                anchor="middle",
            )
        )
        value = row["delta"]
        fill = f"#{hex_for_delta(value)}"
        fg = "#FFFFFF" if text_color(value) == "white" else "#1A1A1A"
        lines.append(
            f'<rect x="{col_x[3]}" y="{y}" width="{col_widths[3]}" '
            f'height="{row_h}" fill="{fill}"/>'
        )
        lines.append(
            svg_text(
                col_x[3] + col_widths[3] / 2,
                y + row_h / 2,
                format_delta(value, decimals=4),
                size=13,
                weight=700,
                fill=fg,
                anchor="middle",
            )
        )
        y += row_h

    legend_y = table_y + table_h + 38
    lines.extend(
        [
            svg_text(margin, legend_y, "Delta", size=13, weight=700, fill="#71717A"),
            svg_text(margin + 62, legend_y, "0.00", size=13, fill="#71717A"),
            f'<rect x="{margin + 105}" y="{legend_y - 8}" width="220" height="16" '
            'rx="3" fill="url(#deltaGradient)" stroke="#E2E2E5"/>',
            svg_text(margin + 345, legend_y, "1.00", size=13, fill="#71717A"),
            svg_text(
                margin,
                legend_y + 32,
                "Valeur positive : TextToKids meilleur ; valeur négative : CyberAggAdo meilleur.",
                size=12,
                fill="#71717A",
            ),
        ]
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cyber", type=Path, default=DEFAULT_CYBER)
    parser.add_argument("--ttk", type=Path, default=DEFAULT_TTK)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()

    data = collect_data(args.cyber, args.ttk)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_latex_table(data, args.out_dir / "heatmap_delta_table.tex")
    write_svg(data, args.out_dir / "heatmap_delta.svg")

    print(f"Wrote {args.out_dir / 'heatmap_delta_table.tex'}")
    print(f"Wrote {args.out_dir / 'heatmap_delta.svg'}")


if __name__ == "__main__":
    main()
