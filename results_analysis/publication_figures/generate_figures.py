#!/usr/bin/env python3
"""Generate publication-grade figures for the Mandarin BabyLM comparison."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch


SUITES = (
    "BabyLM official Chinese",
    "Chinese zero-shot",
    "Chinese fine-tune",
    "CogBench",
)
SUITE_SHORT = {
    "BabyLM official Chinese": "Official BabyLM",
    "Chinese zero-shot": "Chinese zero-shot",
    "Chinese fine-tune": "Chinese fine-tune",
    "CogBench": "CogBench",
}
SUITE_COLORS = {
    "BabyLM official Chinese": "#0072B2",
    "Chinese zero-shot": "#009E73",
    "Chinese fine-tune": "#D55E00",
    "CogBench": "#CC79A7",
}
TOKENIZER_COLORS = {
    "Hybrid": "#0072B2",
    "BPE": "#7A7A7A",
    "Atomic BPE within": "#009E73",
    "Atomic BPE cross": "#D55E00",
}
ARCH_MARKERS = {"GPT2": "o", "Qwen2": "s"}
WITHIN_COLOR = "#009E73"
CROSS_COLOR = "#D55E00"
NEUTRAL = "#666666"
FIGURE_KEYS = (
    "task_heatmap",
    "paired_effects",
    "experimental_design",
    "within_cross",
    "suite_averages",
    "cogbench_regions",
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "results_analysis/processed")
    parser.add_argument("--registry", type=Path, default=root / "results_analysis/models.json")
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "svg", "png"),
        default=("pdf", "svg"),
    )
    parser.add_argument(
        "--figures", nargs="+", choices=FIGURE_KEYS, default=FIGURE_KEYS,
        help="Generate only the selected figure families (default: all).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate registry/table compatibility without rendering figures.",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_registry(path: Path) -> tuple[pd.DataFrame, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["models"])
    required_columns = {"model_id", "label", "architecture", "tokenizer", "scale", "cohort"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Registry is missing metadata columns: {missing_columns}")
    labels = frame["label"].tolist()
    if len(labels) != len(set(labels)):
        raise ValueError("Registry model labels are not unique")
    unknown_tokenizers = sorted(set(frame["tokenizer"]) - set(TOKENIZER_COLORS))
    unknown_architectures = sorted(set(frame["architecture"]) - set(ARCH_MARKERS))
    if unknown_tokenizers:
        raise ValueError(f"Add plotting colors for new tokenizers: {unknown_tokenizers}")
    if unknown_architectures:
        raise ValueError(f"Add plotting markers for new architectures: {unknown_architectures}")
    return frame, labels


def ordered_metadata_values(models: pd.DataFrame, column: str) -> list[str]:
    """Return registry values in first-seen order, preserving the experiment design."""
    return models[column].drop_duplicates().astype(str).tolist()


def scale_sort_key(scale: str) -> tuple[float, str]:
    """Sort labels such as 30M and 100M numerically while accepting future names."""
    digits = "".join(character for character in str(scale) if character.isdigit() or character == ".")
    return (float(digits) if digits else math.inf, str(scale))


def scale_point_sizes(scales: Iterable[str]) -> dict[str, float]:
    """Map any number of registered model scales onto readable marker sizes."""
    ordered = sorted(set(scales), key=scale_sort_key)
    if len(ordered) == 1:
        return {ordered[0]: 58.0}
    sizes = np.linspace(47.0, 78.0, len(ordered))
    return dict(zip(ordered, sizes))


def atomic_model_pairs(models: pd.DataFrame) -> dict[str, list[tuple[str, str, str]]]:
    """Discover complete within/cross atomic-BPE pairs from registry metadata.

    Returns ``scale -> [(architecture, within_label, cross_label), ...]``.
    Adding final models therefore requires registry entries and regenerated
    processed tables, but no plotting-code changes.
    """
    required = {"Atomic BPE within", "Atomic BPE cross"}
    result: dict[str, list[tuple[str, str, str]]] = {}
    for (scale, architecture), group in models.groupby(["scale", "architecture"], sort=False):
        by_tokenizer = group.set_index("tokenizer")["label"].to_dict()
        present = required.intersection(by_tokenizer)
        if present and present != required:
            missing = sorted(required - present)
            raise ValueError(
                f"Incomplete atomic-BPE pair for {scale} {architecture}; missing {missing}"
            )
        if present == required:
            result.setdefault(str(scale), []).append((
                str(architecture),
                str(by_tokenizer["Atomic BPE within"]),
                str(by_tokenizer["Atomic BPE cross"]),
            ))
    return dict(sorted(result.items(), key=lambda item: scale_sort_key(item[0])))


def load_inputs(input_dir: Path, registry: Path) -> dict[str, object]:
    models, labels = load_registry(registry)
    data: dict[str, object] = {
        "models": models,
        "labels": labels,
        "primary": pd.read_csv(input_dir / "primary_scores_wide.csv"),
        "effects": pd.read_csv(input_dir / "pairwise_effects.csv"),
        "contrast_summary": pd.read_csv(input_dir / "contrast_summary.csv"),
        "averages": pd.read_csv(input_dir / "suite_averages.csv"),
        "cogbench": pd.read_csv(input_dir / "cogbench_detail.csv"),
    }
    primary = data["primary"]
    assert isinstance(primary, pd.DataFrame)
    missing = [label for label in labels if label not in primary.columns]
    if missing:
        raise ValueError(f"Primary table is missing registered models: {missing}")
    if primary[labels].isna().any().any():
        raise ValueError("Primary score matrix contains missing values")
    if set(primary["suite"]) != set(SUITES):
        raise ValueError(f"Unexpected suite coverage: {sorted(primary['suite'].unique())}")
    averages = data["averages"]
    cogbench = data["cogbench"]
    effects = data["effects"]
    assert isinstance(averages, pd.DataFrame) and isinstance(cogbench, pd.DataFrame)
    assert isinstance(effects, pd.DataFrame)
    expected_average_pairs = {(label, suite) for label in labels for suite in SUITES}
    observed_average_pairs = set(zip(averages["model"], averages["suite"]))
    missing_average_pairs = expected_average_pairs - observed_average_pairs
    if missing_average_pairs:
        examples = sorted(missing_average_pairs)[:5]
        raise ValueError(f"Suite averages are incomplete; examples: {examples}")
    missing_cogbench = sorted(set(labels) - set(cogbench["model"]))
    if missing_cogbench:
        raise ValueError(f"CogBench detail is missing registered models: {missing_cogbench}")
    contrast_models = set(effects["minuend"]).union(effects["subtrahend"])
    unknown_contrast_models = sorted(contrast_models - set(labels))
    if unknown_contrast_models:
        raise ValueError(f"Pairwise effects reference unknown models: {unknown_contrast_models}")
    atomic_model_pairs(models)
    return data


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str, formats: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "results_analysis/publication_figures/generate_figures.py"}
    for extension in formats:
        kwargs: dict[str, object] = {"metadata": metadata}
        if extension == "png":
            kwargs["dpi"] = 600
        fig.savefig(output_dir / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right")


def model_tick_label(label: str) -> str:
    replacements = {
        " Atomic Within": "\nAtomic within",
        " Atomic Cross": "\nAtomic cross",
        " Hybrid": "\nHybrid",
        " BPE": "\nBPE",
    }
    for suffix, replacement in replacements.items():
        if label.endswith(suffix):
            return label.removesuffix(suffix) + replacement
    return label


def task_label(task: str) -> str:
    mapping = {
        "global_piqa_nonparallel_zh": "Global PIQA nonparallel",
        "global_piqa_parallel_zh": "Global PIQA parallel",
        "hellaswag_zh_mubench": "HellaSwag",
        "winogrande_zh_mubench": "WinoGrande",
        "xstorycloze_zh_mubench": "XStoryCloze",
        "xcomps_zh": "XCOMPS",
        "zhoblimp": "ZhoBLiMP",
        "hanzi_pinyin": "Hanzi–pinyin",
        "hanzi_structure": "Hanzi structure",
        "cluewsc2020": "CLUE WSC 2020",
        "word_fmri": "Word fMRI",
        "fmri": "fMRI",
    }
    return mapping.get(task, task.upper() if len(task) <= 7 else task.replace("_", " ").title())


def ordered_primary(primary: pd.DataFrame) -> pd.DataFrame:
    result = primary.copy()
    result["suite"] = pd.Categorical(result["suite"], categories=SUITES, ordered=True)
    return result.sort_values(["suite", "task"], kind="stable").reset_index(drop=True)


def task_by_model_heatmap(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    primary = ordered_primary(data["primary"])
    labels = data["labels"]
    models = data["models"]
    assert isinstance(primary, pd.DataFrame) and isinstance(labels, list)
    assert isinstance(models, pd.DataFrame)
    raw = primary[labels].to_numpy(float) * 100.0
    centered = raw - raw.mean(axis=1, keepdims=True)
    limit = max(1.0, float(np.max(np.abs(centered))))

    width = max(13.8, 4.0 + 0.82 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 11.2), constrained_layout=True)
    image = ax.imshow(
        centered, aspect="auto", cmap="PuOr_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(labels)), [model_tick_label(x) for x in labels],
                  rotation=42, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(primary)), [task_label(x) for x in primary["task"]])
    ax.tick_params(length=0)
    ax.set_title("Task-level performance across Mandarin BabyLM models", pad=42, fontweight="bold")
    ax.set_xlabel("Model")
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            color = "white" if abs(centered[row, column]) > limit * 0.53 else "#202020"
            ax.text(column, row, f"{raw[row, column]:.1f}", ha="center", va="center",
                    fontsize=5.7, color=color)

    suite_values = primary["suite"].astype(str).tolist()
    boundaries = [index for index in range(1, len(suite_values))
                  if suite_values[index] != suite_values[index - 1]]
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=2.2)
        ax.axhline(boundary - 0.5, color="#333333", linewidth=0.6)
    starts = [0, *boundaries]
    ends = [*boundaries, len(primary)]
    for start, end in zip(starts, ends):
        suite = suite_values[start]
        ax.text(len(labels) + 0.25, (start + end - 1) / 2, SUITE_SHORT[suite],
                rotation=90, ha="center", va="center", fontsize=7.5,
                color=SUITE_COLORS[suite], fontweight="bold", clip_on=False)

    cohort_starts: list[tuple[int, int, str]] = []
    for cohort in models["cohort"].drop_duplicates():
        indices = [labels.index(label) for label in models.loc[models["cohort"] == cohort, "label"]]
        cohort_starts.append((min(indices), max(indices), cohort))
    for start, end, cohort in cohort_starts:
        ax.plot([start - 0.42, end + 0.42], [-1.32, -1.32], color="#333333", lw=0.8, clip_on=False)
        ax.text((start + end) / 2, -1.72, cohort, ha="center", va="bottom",
                fontsize=8, fontweight="bold", clip_on=False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.075)
    colorbar.set_label("Difference from task mean (score ×100)")
    ax.text(0, -0.10,
            "Cell labels are raw score ×100; color shows within-task centering. CogBench is correlation-based.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color=NEUTRAL)
    panel_label(ax, "A")
    save_figure(fig, output, "01_task_by_model_heatmap", formats)


def wrap_contrast(value: str, width: int = 32) -> str:
    return "\n".join(textwrap.wrap(value.replace("Atomic BPE", "Atomic"), width=width))


def paired_effect_forest(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    effects = data["effects"].copy()
    summary = data["contrast_summary"].copy()
    assert isinstance(effects, pd.DataFrame) and isinstance(summary, pd.DataFrame)
    panels = [
        ("scale", "Model scale", "100M minus 30M"),
        (("baseline_tokenizer", "atomic_vs_bpe"), "Tokenizer family", "Tokenizer contrast"),
        ("boundary_policy", "Boundary policy", "Cross-word minus within-word"),
        ("architecture", "Architecture", "Qwen2 minus GPT2"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.2))
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.075, top=0.885,
                        hspace=0.34, wspace=0.43)
    suite_offsets = np.linspace(-0.24, 0.24, len(SUITES))
    for panel_index, (ax, panel) in enumerate(zip(axes.flat, panels)):
        categories, title, xlabel = panel
        category_set = {categories} if isinstance(categories, str) else set(categories)
        panel_summary = summary[summary["category"].isin(category_set)]
        contrasts = panel_summary["contrast"].drop_duplicates().tolist()
        ybase = np.arange(len(contrasts))
        for suite_index, suite in enumerate(SUITES):
            color = SUITE_COLORS[suite]
            for contrast_index, contrast in enumerate(contrasts):
                y = ybase[contrast_index] + suite_offsets[suite_index]
                task_rows = effects[(effects["suite"] == suite) & (effects["contrast"] == contrast)]
                mean_rows = panel_summary[(panel_summary["suite"] == suite) &
                                          (panel_summary["contrast"] == contrast)]
                if task_rows.empty or mean_rows.empty:
                    continue
                ax.scatter(task_rows["difference"] * 100, np.full(len(task_rows), y),
                           s=8, color=color, alpha=0.20, linewidths=0, zorder=1)
                ax.scatter(float(mean_rows.iloc[0]["mean_difference"]) * 100, y,
                           s=38, marker="D", color=color, edgecolor="white",
                           linewidth=0.55, zorder=3)
        ax.axvline(0, color="#222222", linewidth=0.8, zorder=0)
        ax.set_yticks(ybase, [wrap_contrast(x) for x in contrasts])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(f"{xlabel} (score-point difference ×100)")
        ax.grid(axis="x")
        panel_label(ax, chr(ord("A") + panel_index))
    handles = [Line2D([], [], marker="D", linestyle="", color=SUITE_COLORS[suite],
                      label=SUITE_SHORT[suite], markersize=6) for suite in SUITES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.948),
               ncol=4, frameon=False)
    fig.suptitle("Matched evaluation contrasts across tasks", fontweight="bold", y=0.992)
    fig.text(0.5, 0.015,
             "Small points are task-level paired differences; diamonds are unweighted suite means. "
             "Dispersion is descriptive and is not a confidence interval.",
             ha="center", fontsize=7.5, color=NEUTRAL)
    save_figure(fig, output, "02_paired_effect_forest", formats)


def experimental_design(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    models = data["models"]
    assert isinstance(models, pd.DataFrame)
    tokenizers = tuple(TOKENIZER_COLORS)
    architectures = tuple(ordered_metadata_values(models, "architecture"))
    scales = tuple(sorted(ordered_metadata_values(models, "scale"), key=scale_sort_key))
    fig, axes_array = plt.subplots(
        1, len(scales), figsize=(5.75 * len(scales), 5.8), constrained_layout=True,
        squeeze=False,
    )
    axes = axes_array.flat
    for panel_index, (ax, scale) in enumerate(zip(axes, scales)):
        ax.set_xlim(0, len(architectures))
        ax.set_ylim(0, 4.55)
        ax.axis("off")
        ax.set_title(f"{scale} model family", fontsize=12, fontweight="bold", pad=14)
        for column, architecture in enumerate(architectures):
            ax.text(column + 0.5, 4.25, architecture, ha="center", va="center",
                    fontsize=10, fontweight="bold")
            for row, tokenizer in enumerate(tokenizers):
                y = 3.65 - row * 0.9
                match = models[(models["scale"] == scale) &
                               (models["architecture"] == architecture) &
                               (models["tokenizer"] == tokenizer)]
                evaluated = not match.empty
                color = TOKENIZER_COLORS[tokenizer] if evaluated else "#E6E6E6"
                edge = TOKENIZER_COLORS[tokenizer] if evaluated else "#999999"
                box = FancyBboxPatch(
                    (column + 0.10, y - 0.30), 0.80, 0.60,
                    boxstyle="round,pad=0.025,rounding_size=0.04",
                    facecolor=color, edgecolor=edge, linewidth=1.0,
                    alpha=0.90 if evaluated else 1.0,
                    hatch=None if evaluated else "///",
                )
                ax.add_patch(box)
                status = "evaluated" if evaluated else "training /\nevaluation pending"
                text_color = "white" if evaluated and tokenizer != "BPE" else "#222222"
                ax.text(column + 0.5, y + 0.04, tokenizer.replace("Atomic BPE ", "Atomic "),
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color=text_color)
                ax.text(column + 0.5, y - 0.16, status, ha="center", va="center",
                        fontsize=6.5, color=text_color if evaluated else NEUTRAL)
        if len(architectures) == 2:
            ax.annotate("", xy=(0.88, 4.02), xytext=(1.12, 4.02),
                        arrowprops={"arrowstyle": "<->", "color": NEUTRAL, "lw": 0.8})
            ax.text(1.0, 4.06, "Architecture contrast", ha="center", va="bottom",
                    fontsize=7, color=NEUTRAL)
        panel_label(ax, chr(ord("A") + panel_index))
    fig.suptitle("Factorial experimental design", fontweight="bold")
    fig.text(0.5, 0.01,
             "Comparisons isolate model scale, architecture, tokenizer family, and atomic-BPE boundary policy.",
             ha="center", fontsize=8, color=NEUTRAL)
    save_figure(fig, output, "03_experimental_design", formats)


def within_cross_dumbbell(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    primary = ordered_primary(data["primary"])
    models = data["models"]
    assert isinstance(primary, pd.DataFrame) and isinstance(models, pd.DataFrame)
    pairs_by_scale = atomic_model_pairs(models)
    if not pairs_by_scale:
        raise ValueError("No complete within/cross atomic-BPE model pairs are registered")

    for scale, pairs in pairs_by_scale.items():
        column_count = len(pairs)
        fig = plt.figure(figsize=(6.9 * column_count, 12.8))
        grid = fig.add_gridspec(
            len(SUITES), column_count, height_ratios=[4.4, 1.5, 1.8, 1.25],
            left=0.12, right=0.985, bottom=0.055, top=0.895,
            hspace=0.50, wspace=0.27,
        )
        for row_index, suite in enumerate(SUITES):
            suite_rows = primary[primary["suite"].astype(str) == suite].copy()
            for column_index, (architecture, within, cross) in enumerate(pairs):
                ax = fig.add_subplot(grid[row_index, column_index])
                subset = suite_rows.assign(
                    delta=suite_rows[within] - suite_rows[cross]
                ).sort_values("delta")
                y = np.arange(len(subset))
                x_within = subset[within].to_numpy(float) * 100
                x_cross = subset[cross].to_numpy(float) * 100
                for yi, left, right in zip(y, x_within, x_cross):
                    ax.plot([left, right], [yi, yi], color="#B5B5B5", linewidth=1.2, zorder=1)
                ax.scatter(x_within, y, color=WITHIN_COLOR, marker="o", s=29,
                           edgecolor="white", linewidth=0.45, zorder=3)
                ax.scatter(x_cross, y, color=CROSS_COLOR, marker="s", s=27,
                           edgecolor="white", linewidth=0.45, zorder=3)
                ax.set_yticks(y, [task_label(x) for x in subset["task"]])
                ax.set_xlabel("Score ×100")
                ax.grid(axis="x")
                ax.set_title(f"{SUITE_SHORT[suite]} — {architecture}",
                             loc="left", fontweight="bold")
                panel_label(ax, chr(ord("A") + row_index * column_count + column_index))
        handles = [
            Line2D([], [], marker="o", linestyle="", color=WITHIN_COLOR,
                   label="Within-word atomic BPE", markersize=6),
            Line2D([], [], marker="s", linestyle="", color=CROSS_COLOR,
                   label="Cross-word atomic BPE", markersize=6),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.948),
                   ncol=2, frameon=False)
        fig.suptitle(f"Within-word versus cross-word atomic BPE — {scale}",
                     fontweight="bold", y=0.992)
        fig.text(
            0.5, 0.012,
            "Each line connects matched models on one task. Panels use independent x-axis ranges; "
            "CogBench is correlation-based.",
            ha="center", fontsize=7.5, color=NEUTRAL,
        )
        scale_slug = "".join(character.lower() for character in scale if character.isalnum())
        save_figure(fig, output, f"04_within_vs_cross_dumbbell_{scale_slug}", formats)


def suite_average_dotplot(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    averages = data["averages"]
    models = data["models"]
    labels = data["labels"]
    assert isinstance(averages, pd.DataFrame) and isinstance(models, pd.DataFrame)
    assert isinstance(labels, list)
    metadata = models.set_index("label")
    y = np.arange(len(labels))
    sizes = scale_point_sizes(models["scale"].astype(str))
    figure_height = max(9.2, 4.8 + 0.36 * len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, figure_height), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.075, top=0.86,
                        hspace=0.30, wspace=0.13)
    for panel_index, (ax, suite) in enumerate(zip(axes.flat, SUITES)):
        subset = averages[averages["suite"] == suite].set_index("model").loc[labels]
        for yi, label in zip(y, labels):
            row = metadata.loc[label]
            ax.scatter(
                float(subset.loc[label, "mean_score"]) * 100, yi,
                color=TOKENIZER_COLORS[row["tokenizer"]],
                marker=ARCH_MARKERS[row["architecture"]],
                s=sizes[str(row["scale"])],
                edgecolor="white", linewidth=0.55, zorder=3,
            )
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Unweighted mean score ×100")
        ax.set_title(f"{SUITE_SHORT[suite]} (n={int(subset['task_count'].iloc[0])} tasks)",
                     loc="left", fontweight="bold")
        ax.grid(axis="x")
        panel_label(ax, chr(ord("A") + panel_index))
    tokenizer_handles = [Patch(facecolor=color, edgecolor="none", label=tokenizer)
                         for tokenizer, color in TOKENIZER_COLORS.items()]
    architecture_handles = [Line2D([], [], marker=marker, linestyle="", color="#333333",
                                   label=architecture, markersize=6)
                            for architecture, marker in ARCH_MARKERS.items()]
    size_handles = [Line2D([], [], marker="o", linestyle="", color="#777777",
                           label=scale, markersize=math.sqrt(size))
                    for scale, size in sizes.items()]
    fig.legend(handles=[*tokenizer_handles, *architecture_handles, *size_handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=8, frameon=False)
    fig.suptitle("Descriptive performance averages within evaluation suites",
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.018,
             "Task means are unweighted. Panels use independent x-axis ranges and must not be combined into one overall score.",
             ha="center", fontsize=7.5, color=NEUTRAL)
    save_figure(fig, output, "05_suite_average_dotplot", formats)


def cogbench_region_heatmap(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    cogbench = data["cogbench"]
    labels = data["labels"]
    models = data["models"]
    assert isinstance(cogbench, pd.DataFrame) and isinstance(labels, list)
    assert isinstance(models, pd.DataFrame)
    fmri = cogbench[(cogbench["task"] == "fmri") & cogbench["region"].notna()].copy()
    pivot = fmri.pivot_table(index="region", columns="model", values="score", aggfunc="mean")
    pivot = pivot.loc[sorted(pivot.index), labels]
    raw = pivot.to_numpy(float) * 100
    centered = raw - raw.mean(axis=1, keepdims=True)
    limit = max(0.1, float(np.max(np.abs(centered))))

    width = max(13.6, 4.0 + 0.80 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 5.3), constrained_layout=True)
    image = ax.imshow(
        centered, aspect="auto", cmap="PuOr_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(labels)), [model_tick_label(x) for x in labels],
                  rotation=42, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.tick_params(length=0)
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            color = "white" if abs(centered[row, column]) > limit * 0.53 else "#202020"
            ax.text(column, row, f"{raw[row, column]:.2f}", ha="center", va="center",
                    fontsize=6.4, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.025)
    colorbar.set_label("Difference from region mean (correlation ×100)")
    ax.set_title("CogBench fMRI performance by brain region", fontweight="bold", pad=12)
    ax.set_xlabel("Model")
    ax.set_ylabel("Region")
    ax.text(0, -0.24,
            "Cell labels are mean subject-level correlation ×100; color shows within-region centering.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color=NEUTRAL)
    panel_label(ax, "A")
    save_figure(fig, output, "06_cogbench_region_heatmap", formats)


def main() -> None:
    args = parse_args()
    configure_style()
    data = load_inputs(args.input_dir.resolve(), args.registry.resolve())
    if args.validate_only:
        print(f"validated {len(data['labels'])} models across {len(SUITES)} suites")
        return
    output = args.output_dir.resolve()
    generators: dict[str, Callable[[dict[str, object], Path, Iterable[str]], None]] = {
        "task_heatmap": task_by_model_heatmap,
        "paired_effects": paired_effect_forest,
        "experimental_design": experimental_design,
        "within_cross": within_cross_dumbbell,
        "suite_averages": suite_average_dotplot,
        "cogbench_regions": cogbench_region_heatmap,
    }
    for key in args.figures:
        generator = generators[key]
        generator(data, output, args.formats)
        print(f"generated {key}")
    print(f"output directory: {output}")


if __name__ == "__main__":
    main()
