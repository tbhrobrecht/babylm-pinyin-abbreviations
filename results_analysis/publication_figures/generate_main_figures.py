#!/usr/bin/env python3
"""Generate compact, publication-grade figures for the paper's main text.

The script is registry-driven: adding completed models to ``models.json`` and
rebuilding ``results_analysis/processed`` is sufficient to extend every plot.
Shared loading, validation, styling, and export behavior comes from
``generate_figures.py`` so the main-text and supplementary figures cannot drift.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import generate_figures as common


FIGURE_KEYS = ("overview", "scaling", "boundaries", "task_profiles")
ARCHITECTURE_COLORS = {"GPT2": "#0072B2", "Qwen2": "#D55E00"}
SCALE_LINESTYLES = ("-", "--", ":", "-.")
TASK_FAMILIES = (
    "Reasoning and commonsense",
    "Cross-lingual understanding",
    "Chinese linguistic form",
    "Chinese downstream",
    "Cognitive alignment",
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "results_analysis/processed")
    parser.add_argument("--registry", type=Path, default=root / "results_analysis/models.json")
    parser.add_argument("--output-dir", type=Path, default=here / "main_body_output")
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "svg", "png"),
        default=("pdf", "svg"),
    )
    parser.add_argument(
        "--figures", nargs="+", choices=FIGURE_KEYS, default=FIGURE_KEYS,
        help="Generate only the selected main-text figure families (default: all).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate inputs and matched comparisons without rendering.",
    )
    return parser.parse_args()


def scale_styles(scales: Iterable[str]) -> dict[str, str]:
    ordered = sorted(set(scales), key=common.scale_sort_key)
    return {scale: SCALE_LINESTYLES[index % len(SCALE_LINESTYLES)]
            for index, scale in enumerate(ordered)}


def add_panel_labels(axes: Iterable[mpl.axes.Axes]) -> None:
    for index, ax in enumerate(axes):
        common.panel_label(ax, chr(ord("A") + index))


def model_design_overview(
    data: dict[str, object], output: Path, formats: Iterable[str],
) -> None:
    """Draw a compact flow diagram of the experimental design."""
    models = data["models"]
    assert isinstance(models, pd.DataFrame)
    scales = sorted(common.ordered_metadata_values(models, "scale"), key=common.scale_sort_key)
    architectures = common.ordered_metadata_values(models, "architecture")
    tokenizers = list(common.TOKENIZER_COLORS)
    complete = len(models)
    factorial_total = len(scales) * len(architectures) * len(tokenizers)

    fig, ax = plt.subplots(figsize=(11.5, 4.1))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.82)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = (
        (0.02, 0.15, "Training data", "Mandarin BabyLM\n100M-word corpus", "#E6F2F8"),
        (0.22, 0.18, "Tokenizer", "Hybrid · BPE\nAtomic within · Atomic cross", "#E7F5EF"),
        (0.46, 0.15, "Architecture", "GPT2 · Qwen2", "#FFF0E5"),
        (0.66, 0.13, "Model scale", "30M · 100M\nparameters", "#F4EAF2"),
        (0.84, 0.14, "Evaluation", "Official BabyLM · Chinese\nCogBench", "#F0F0F0"),
    )
    for x, width, title, detail, color in stages:
        box = FancyBboxPatch(
            (x, 0.34), width, 0.40,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=color, edgecolor="#555555", linewidth=0.9,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, 0.61, title, ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x + width / 2, 0.46, detail, ha="center", va="center",
                fontsize=7.6, linespacing=1.35, color="#333333")
    for left, right in zip(stages, stages[1:]):
        start = left[0] + left[1]
        end = right[0]
        ax.annotate("", xy=(end - 0.008, 0.54), xytext=(start + 0.008, 0.54),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#555555"})

    ax.text(
        0.5, 0.17,
        f"{complete} evaluated cells currently registered; {factorial_total} cells in the complete factorial design",
        ha="center", va="center", fontsize=8, color=common.NEUTRAL,
    )
    ax.text(
        0.5, 0.07,
        "Matched contrasts isolate scale, architecture, tokenizer family, and atomic-BPE boundary policy.",
        ha="center", va="center", fontsize=8, color=common.NEUTRAL,
    )
    fig.suptitle("Mandarin BabyLM experimental pipeline", fontweight="bold", y=0.96)
    common.panel_label(ax, "A")
    common.save_figure(fig, output, "M1_experimental_overview", formats)


def scaling_trajectories(
    data: dict[str, object], output: Path, formats: Iterable[str],
) -> None:
    """Plot matched performance trajectories over model scale within each suite."""
    averages = data["averages"]
    models = data["models"]
    assert isinstance(averages, pd.DataFrame) and isinstance(models, pd.DataFrame)
    scales = sorted(common.ordered_metadata_values(models, "scale"), key=common.scale_sort_key)
    x_positions = {scale: index for index, scale in enumerate(scales)}
    metadata = models[["label", "architecture", "tokenizer", "scale"]].copy()
    matched: list[tuple[str, str, pd.DataFrame]] = []
    for (architecture, tokenizer), group in metadata.groupby(
        ["architecture", "tokenizer"], sort=False,
    ):
        if group["scale"].nunique() >= 2:
            matched.append((str(architecture), str(tokenizer), group))
    if not matched:
        raise ValueError("Scaling figure requires at least one condition evaluated at two scales")

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.82,
                        hspace=0.38, wspace=0.24)
    for ax, suite in zip(axes.flat, common.SUITES):
        suite_scores = averages[averages["suite"] == suite].set_index("model")["mean_score"]
        for architecture, tokenizer, group in matched:
            ordered = group.sort_values("scale", key=lambda values: values.map(common.scale_sort_key))
            xs = [x_positions[str(scale)] for scale in ordered["scale"]]
            ys = [float(suite_scores[label]) * 100 for label in ordered["label"]]
            ax.plot(
                xs, ys, color=common.TOKENIZER_COLORS[tokenizer],
                marker=common.ARCH_MARKERS[architecture], markersize=5.8,
                linewidth=1.7, markeredgecolor="white", markeredgewidth=0.5,
            )
        ax.set_xticks(range(len(scales)), scales)
        ax.set_xlim(-0.15, len(scales) - 0.85)
        ax.set_ylabel("Unweighted suite mean ×100")
        ax.set_title(common.SUITE_SHORT[suite], loc="left", fontweight="bold")
        ax.grid(axis="y")
    add_panel_labels(axes.flat)
    tokenizer_handles = [Line2D([], [], color=color, linewidth=2, label=tokenizer)
                         for tokenizer, color in common.TOKENIZER_COLORS.items()
                         if any(item[1] == tokenizer for item in matched)]
    architecture_handles = [Line2D([], [], color="#333333", marker=marker,
                                   linestyle="", label=architecture, markersize=6)
                            for architecture, marker in common.ARCH_MARKERS.items()]
    fig.legend(handles=[*tokenizer_handles, *architecture_handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.90), ncol=len(tokenizer_handles) + 2,
               frameon=False)
    fig.suptitle("Performance trajectories with increasing model scale",
                 fontweight="bold", y=0.975)
    fig.text(
        0.5, 0.025,
        "Lines connect matched architecture–tokenizer conditions only. Suites use independent y-axis ranges.",
        ha="center", fontsize=7.5, color=common.NEUTRAL,
    )
    common.save_figure(fig, output, "M2_scaling_trajectories", formats)


def boundary_policy_trajectories(
    data: dict[str, object], output: Path, formats: Iterable[str],
) -> None:
    """Plot within-word to cross-word changes for every complete atomic pair."""
    averages = data["averages"]
    models = data["models"]
    assert isinstance(averages, pd.DataFrame) and isinstance(models, pd.DataFrame)
    pairs = common.atomic_model_pairs(models)
    styles = scale_styles(pairs)
    flat_pairs = [(scale, architecture, within, cross)
                  for scale, items in pairs.items()
                  for architecture, within, cross in items]
    if not flat_pairs:
        raise ValueError("Boundary figure requires complete within/cross atomic-BPE pairs")

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.82,
                        hspace=0.38, wspace=0.24)
    for ax, suite in zip(axes.flat, common.SUITES):
        suite_scores = averages[averages["suite"] == suite].set_index("model")["mean_score"]
        for scale, architecture, within, cross in flat_pairs:
            values = [float(suite_scores[within]) * 100, float(suite_scores[cross]) * 100]
            ax.plot(
                [0, 1], values, color=ARCHITECTURE_COLORS[architecture],
                linestyle=styles[scale], marker=common.ARCH_MARKERS[architecture],
                markersize=5.8, linewidth=1.7, markeredgecolor="white",
                markeredgewidth=0.5,
            )
        ax.set_xticks((0, 1), ("Within-word", "Cross-word"))
        ax.set_xlim(-0.16, 1.16)
        ax.set_ylabel("Unweighted suite mean ×100")
        ax.set_title(common.SUITE_SHORT[suite], loc="left", fontweight="bold")
        ax.grid(axis="y")
    add_panel_labels(axes.flat)
    architecture_handles = [Line2D([], [], color=color, marker=common.ARCH_MARKERS[architecture],
                                   linewidth=1.7, label=architecture, markersize=6)
                            for architecture, color in ARCHITECTURE_COLORS.items()]
    scale_handles = [Line2D([], [], color="#555555", linestyle=linestyle,
                            linewidth=1.7, label=scale)
                     for scale, linestyle in styles.items()]
    fig.legend(handles=[*architecture_handles, *scale_handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.90), ncol=len(architecture_handles) + len(scale_handles),
               frameon=False)
    fig.suptitle("Effect of atomic-BPE boundary policy", fontweight="bold", y=0.975)
    fig.text(
        0.5, 0.025,
        "Each line connects matched within-word and cross-word models. Suites use independent y-axis ranges.",
        ha="center", fontsize=7.5, color=common.NEUTRAL,
    )
    common.save_figure(fig, output, "M3_boundary_policy_trajectories", formats)


def task_family(suite: str, task: str) -> str:
    """Map every primary evaluation row to a preregistered reporting family."""
    if suite == "CogBench":
        return "Cognitive alignment"
    if suite == "Chinese fine-tune":
        return "Chinese downstream"
    if task in {"pos", "zhoblimp", "hanzi_pinyin", "hanzi_structure"}:
        return "Chinese linguistic form"
    if task in {"belebele", "bmlama", "include", "mnli", "sib200", "xnli"}:
        return "Cross-lingual understanding"
    return "Reasoning and commonsense"


def task_family_profiles(
    data: dict[str, object], output: Path, formats: Iterable[str],
) -> None:
    """Compare metric-robust within-task percentile profiles by task family."""
    primary = common.ordered_primary(data["primary"])
    models = data["models"]
    labels = data["labels"]
    assert isinstance(primary, pd.DataFrame) and isinstance(models, pd.DataFrame)
    assert isinstance(labels, list)
    rank_rows = primary[labels].rank(axis=1, method="average")
    percentiles = (rank_rows - 1.0) / max(1, len(labels) - 1) * 100.0
    percentiles["family"] = [task_family(str(suite), str(task))
                              for suite, task in zip(primary["suite"], primary["task"])]
    profiles = percentiles.groupby("family", sort=False)[labels].mean().loc[list(TASK_FAMILIES)]
    styles = scale_styles(models["scale"].astype(str))

    architectures = common.ordered_metadata_values(models, "architecture")
    fig, axes_array = plt.subplots(1, len(architectures), figsize=(12.4, 5.4),
                                   sharey=True, squeeze=False)
    axes = axes_array.flat
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.25, top=0.78, wspace=0.12)
    x = np.arange(len(TASK_FAMILIES))
    for ax, architecture in zip(axes, architectures):
        subset = models[models["architecture"] == architecture]
        for _, model in subset.iterrows():
            ax.plot(
                x, profiles[str(model["label"])].to_numpy(float),
                color=common.TOKENIZER_COLORS[str(model["tokenizer"])],
                linestyle=styles[str(model["scale"])], linewidth=1.35,
                marker="o", markersize=3.8, alpha=0.90,
            )
        ax.set_xticks(x, [name.replace(" and ", " & ") for name in TASK_FAMILIES],
                      rotation=28, ha="right", rotation_mode="anchor")
        ax.set_ylim(-3, 103)
        ax.set_ylabel("Mean within-task percentile rank")
        ax.set_title(architecture, loc="left", fontweight="bold")
        ax.grid(axis="y")
    add_panel_labels(axes)
    tokenizer_handles = [Patch(facecolor=color, edgecolor="none", label=tokenizer)
                         for tokenizer, color in common.TOKENIZER_COLORS.items()]
    scale_handles = [Line2D([], [], color="#444444", linestyle=linestyle,
                            linewidth=1.6, label=scale)
                     for scale, linestyle in styles.items()]
    fig.legend(handles=[*tokenizer_handles, *scale_handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.88), ncol=len(tokenizer_handles) + len(scale_handles),
               frameon=False)
    fig.suptitle("Relative performance profiles across task families",
                 fontweight="bold", y=0.975)
    fig.text(
        0.5, 0.045,
        "Ranks are calculated within each task before averaging, avoiding direct aggregation of incompatible metrics. "
        "Lines are categorical profiles, not a continuous task axis.",
        ha="center", fontsize=7.5, color=common.NEUTRAL,
    )
    common.save_figure(fig, output, "M4_task_family_profiles", formats)


def validate_main_figure_inputs(data: dict[str, object]) -> None:
    models = data["models"]
    primary = data["primary"]
    assert isinstance(models, pd.DataFrame) and isinstance(primary, pd.DataFrame)
    common.atomic_model_pairs(models)
    observed_families = {task_family(str(suite), str(task))
                         for suite, task in zip(primary["suite"], primary["task"])}
    missing_families = set(TASK_FAMILIES) - observed_families
    if missing_families:
        raise ValueError(f"No tasks were assigned to families: {sorted(missing_families)}")


def main() -> None:
    args = parse_args()
    common.configure_style()
    data = common.load_inputs(args.input_dir.resolve(), args.registry.resolve())
    validate_main_figure_inputs(data)
    if args.validate_only:
        print(f"validated main-text inputs for {len(data['labels'])} models")
        return
    generators: dict[str, Callable[[dict[str, object], Path, Iterable[str]], None]] = {
        "overview": model_design_overview,
        "scaling": scaling_trajectories,
        "boundaries": boundary_policy_trajectories,
        "task_profiles": task_family_profiles,
    }
    output = args.output_dir.resolve()
    for key in args.figures:
        generators[key](data, output, args.formats)
        print(f"generated {key}")
    print(f"output directory: {output}")


if __name__ == "__main__":
    main()
