#!/usr/bin/env python3
"""Generate the paper-focused figures for the six matched 100M GPT-2 models.

The complete 18-model figures remain untouched.  This generator reads the
same processed result tables, filters through ``paper_focus_models.json``, and
writes both figures and their exact source tables to a separate directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import generate_figures as common


FIGURE_KEYS = (
    "design", "suites", "tasks", "families", "composite", "efficiency",
    "baseline_benefits",
)
GROUP_COLORS = {
    "Coded-pinyin systems": "#E7F5EF",
    "Standard baselines": "#FFF0E5",
}
TASK_FAMILIES = (
    "Reasoning & commonsense",
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
    parser.add_argument(
        "--focus-config", type=Path,
        default=root / "results_analysis/paper_focus_models.json",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "paper_focus_output")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"),
                        default=("pdf", "svg"))
    parser.add_argument("--figures", nargs="+", choices=FIGURE_KEYS, default=FIGURE_KEYS)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def task_family(suite: str, task: str) -> str:
    if suite == "CogBench":
        return "Cognitive alignment"
    if suite == "Chinese fine-tune":
        return "Chinese downstream"
    if task in {"pos", "zhoblimp", "hanzi_pinyin", "hanzi_structure"}:
        return "Chinese linguistic form"
    if task in {"belebele", "bmlama", "include", "mnli", "sib200", "xnli"}:
        return "Cross-lingual understanding"
    return "Reasoning & commonsense"


def load_focus(path: Path, registry: Path) -> pd.DataFrame:
    configured = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["models"])
    all_models, _ = common.load_registry(registry)
    required = {"model_id", "short_label", "group", "processed_text", "train_binary"}
    missing = required - set(configured.columns)
    if missing:
        raise ValueError(f"Focus configuration lacks fields: {sorted(missing)}")
    if configured["model_id"].duplicated().any() or configured["short_label"].duplicated().any():
        raise ValueError("Focused model IDs and short labels must be unique")
    unknown = set(configured["model_id"]) - set(all_models["model_id"])
    if unknown:
        raise ValueError(f"Focused models are absent from the main registry: {sorted(unknown)}")
    focused = configured.merge(all_models, on="model_id", how="left", validate="one_to_one")
    if len(focused) != 6:
        raise ValueError(f"The main-paper comparison must contain exactly six models, found {len(focused)}")
    expected = {"Coded-pinyin systems": 4, "Standard baselines": 2}
    if focused["group"].value_counts().to_dict() != expected:
        raise ValueError("Expected four coded-pinyin systems and two standard baselines")
    if set(focused["architecture"]) != {"GPT2"} or set(focused["scale"]) != {"100M"}:
        raise ValueError("Focused comparisons must hold architecture and scale fixed at 100M GPT-2")
    return focused


def percentile_matrix(
    primary: pd.DataFrame, reference_labels: list[str], display_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Rank over the complete experiment, then optionally retain display columns."""
    ranks = primary[reference_labels].rank(axis=1, method="average")
    percentiles = (ranks - 1.0) / (len(reference_labels) - 1) * 100.0
    return percentiles[display_labels] if display_labels is not None else percentiles


def focused_summary(
    primary: pd.DataFrame, reference_labels: list[str], labels: list[str],
    composite_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter the established full-experiment composite and add six-way task wins."""
    percentiles = percentile_matrix(primary, reference_labels, labels)
    percentiles["suite"] = primary["suite"].to_numpy()
    suite_scores = percentiles.groupby("suite", sort=False)[labels].mean().loc[list(common.SUITES)]
    wins = np.zeros(len(labels), dtype=float)
    for row in primary[labels].to_numpy(float):
        winners = np.flatnonzero(np.isclose(row, row.max(), rtol=0, atol=1e-12))
        wins[winners] += 1.0 / len(winners)
    established = pd.read_csv(composite_path)
    result = established[established["model"].isin(labels)].copy()
    if set(result["model"]) != set(labels) or len(result) != len(labels):
        raise ValueError("Established composite table does not cover the focused models")
    result["fractional_task_wins_among_six"] = result["model"].map(dict(zip(labels, wins)))
    suite_long = (
        suite_scores.T.rename_axis("model").reset_index()
        .melt(id_vars="model", var_name="suite", value_name="mean_within_task_percentile")
    )
    return result, suite_long


def load_data(args: argparse.Namespace) -> dict[str, object]:
    focus = load_focus(args.focus_config.resolve(), args.registry.resolve())
    labels = focus["label"].tolist()
    _, reference_labels = common.load_registry(args.registry.resolve())
    short = dict(zip(focus["label"], focus["short_label"]))
    primary = pd.read_csv(args.input_dir.resolve() / "primary_scores_wide.csv")
    averages = pd.read_csv(args.input_dir.resolve() / "suite_averages.csv")
    missing = set(labels) - set(primary.columns)
    if missing or primary[labels].isna().any().any():
        raise ValueError(f"Focused primary scores are missing or incomplete: {sorted(missing)}")
    averages = averages[averages["model"].isin(labels)].copy()
    if len(averages) != len(labels) * len(common.SUITES):
        raise ValueError("Focused suite averages are incomplete")
    composite, normalized_suites = focused_summary(
        primary, reference_labels, labels,
        args.input_dir.resolve() / "composite_summary.csv",
    )

    root = args.focus_config.resolve().parents[1]
    efficiency_rows = []
    for row in focus.itertuples(index=False):
        text_path = root / row.processed_text
        binary_path = root / row.train_binary
        metadata_path = root / row.model_path / "training_metadata.json"
        for required_path in (text_path, binary_path, metadata_path):
            if not required_path.is_file():
                raise FileNotFoundError(required_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        binary_bytes = binary_path.stat().st_size
        if binary_bytes % 4:
            raise ValueError(f"Expected uint32 training binary: {binary_path}")
        efficiency_rows.append({
            "model_id": row.model_id,
            "model": row.label,
            "short_label": row.short_label,
            "group": row.group,
            "processed_text_bytes": text_path.stat().st_size,
            "train_binary_bytes": binary_bytes,
            "train_tokens": binary_bytes // 4,
            "epochs": int(metadata["epoch"]),
            "optimizer_steps": int(metadata["global_step"]),
            "validation_loss": float(metadata["validation_loss"]),
            "observed_training_seconds": getattr(row, "observed_training_seconds", np.nan),
        })
    efficiency = pd.DataFrame(efficiency_rows).merge(
        composite[["model", "composite_percentile", "composite_rank"]], on="model",
        validate="one_to_one",
    )
    return {
        "focus": focus, "labels": labels, "reference_labels": reference_labels, "short": short,
        "primary": primary, "averages": averages,
        "composite": composite, "normalized_suites": normalized_suites,
        "efficiency": efficiency,
    }


def colors(data: dict[str, object]) -> dict[str, str]:
    focus = data["focus"]
    assert isinstance(focus, pd.DataFrame)
    return {
        str(row.label): common.TOKENIZER_COLORS[str(row.tokenizer)]
        for row in focus.itertuples(index=False)
    }


def add_legend(fig: mpl.figure.Figure, data: dict[str, object], y: float = 0.92) -> None:
    focus = data["focus"]
    assert isinstance(focus, pd.DataFrame)
    handles = [
        Patch(facecolor=common.TOKENIZER_COLORS[str(row.tokenizer)], edgecolor="none",
              label=str(row.short_label))
        for row in focus.itertuples(index=False)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=6, frameon=False)


def design_figure(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    focus = data["focus"]
    assert isinstance(focus, pd.DataFrame)
    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.08, top=0.88)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    source = FancyBboxPatch((0.02, 0.36), 0.15, 0.28, boxstyle="round,pad=0.015",
                            facecolor="#E6F2F8", edgecolor="#555", linewidth=0.9)
    ax.add_patch(source)
    ax.text(0.095, 0.56, "Same source corpus", ha="center", fontweight="bold", fontsize=9)
    ax.text(0.095, 0.44, "identical documents\nand split", ha="center", fontsize=8)
    groups = (("Coded-pinyin systems", 0.58), ("Standard baselines", 0.16))
    for group, y in groups:
        rows = focus[focus["group"] == group]
        panel = FancyBboxPatch((0.25, y), 0.43, 0.25, boxstyle="round,pad=0.015",
                               facecolor=GROUP_COLORS[group], edgecolor="#666", linewidth=0.8)
        ax.add_patch(panel)
        ax.text(0.275, y + 0.205, group, ha="left", va="center", fontweight="bold", fontsize=9)
        xs = np.linspace(0.30, 0.63, len(rows))
        for x, row in zip(xs, rows.itertuples(index=False)):
            ax.scatter(x, y + 0.10, s=210, color=common.TOKENIZER_COLORS[str(row.tokenizer)],
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(x, y + 0.035, str(row.short_label).replace(" ", "\n", 1),
                    ha="center", va="top", fontsize=7.2, linespacing=1.05)
        ax.annotate("", xy=(0.24, y + 0.125), xytext=(0.18, 0.50),
                    arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#666"})
    shared = FancyBboxPatch((0.77, 0.28), 0.20, 0.44, boxstyle="round,pad=0.015",
                            facecolor="#F2F2F2", edgecolor="#555", linewidth=0.9)
    ax.add_patch(shared)
    ax.text(0.87, 0.63, "Controlled training", ha="center", fontweight="bold", fontsize=9)
    ax.text(0.87, 0.51, "GPT-2 · 97.7M parameters\n16k vocabulary · 512 context\n5 epochs · one trained seed",
            ha="center", va="center", fontsize=8, linespacing=1.35)
    ax.text(0.87, 0.36, "Same evaluation", ha="center", fontweight="bold", fontsize=9)
    ax.text(0.87, 0.30, "25 primary metrics · 4 suites", ha="center", fontsize=8)
    for y in (0.705, 0.285):
        ax.annotate("", xy=(0.76, 0.50), xytext=(0.69, y),
                    arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#666"})
    fig.suptitle("Paper-focused comparison: representation changes, model capacity does not",
                 fontweight="bold", y=0.965)
    common.save_figure(fig, output, "P1_focused_experimental_design", formats)


def suite_figure(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    averages = data["averages"]; focus = data["focus"]; short = data["short"]
    assert isinstance(averages, pd.DataFrame) and isinstance(focus, pd.DataFrame)
    assert isinstance(short, dict)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.5))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.82, hspace=0.42, wspace=0.24)
    y = np.arange(len(focus))
    for ax, suite in zip(axes.flat, common.SUITES):
        scores = averages[averages["suite"] == suite].set_index("model")["mean_score"]
        values = np.array([float(scores[label]) * 100 for label in focus["label"]])
        ax.hlines(y, values.min() - 1, values, color="#D4D4D4", lw=0.8)
        for yi, row, value in zip(y, focus.itertuples(index=False), values):
            ax.scatter(value, yi, s=57, color=common.TOKENIZER_COLORS[str(row.tokenizer)],
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.text(value, yi - 0.20, f"{value:.1f}", ha="center", va="bottom", fontsize=6.6)
        ax.set_yticks(y, [short[label] for label in focus["label"]])
        ax.invert_yaxis(); ax.grid(axis="x")
        ax.set_xlabel("Unweighted suite mean ×100")
        ax.set_title(common.SUITE_SHORT[suite], loc="left", fontweight="bold")
    for index, ax in enumerate(axes.flat): common.panel_label(ax, chr(65 + index))
    add_legend(fig, data, 0.91)
    fig.suptitle("Evaluation-suite profiles of the six focal models", fontweight="bold", y=0.975)
    fig.text(0.5, 0.025, "Suite means are descriptive; panels have independent x-axis ranges.",
             ha="center", fontsize=7.5, color=common.NEUTRAL)
    common.save_figure(fig, output, "P2_suite_profiles", formats)


def task_figure(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    primary = common.ordered_primary(data["primary"])
    labels = data["labels"]; reference_labels = data["reference_labels"]; short = data["short"]
    assert isinstance(primary, pd.DataFrame) and isinstance(labels, list)
    assert isinstance(reference_labels, list) and isinstance(short, dict)
    matrix = percentile_matrix(primary, reference_labels, labels)
    fig, ax = plt.subplots(figsize=(10.8, 11.4))
    fig.subplots_adjust(left=0.27, right=0.91, bottom=0.12, top=0.90)
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="cividis", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(labels)), [short[label] for label in labels], rotation=28,
                  ha="right", rotation_mode="anchor")
    task_labels = [common.task_label(str(task)) for task in primary["task"]]
    ax.set_yticks(np.arange(len(primary)), task_labels)
    for row in range(len(primary)):
        for column, label in enumerate(labels):
            raw = float(primary.iloc[row][label]) * 100
            rank = float(matrix.iloc[row, column])
            ax.text(column, row, f"{raw:.1f}", ha="center", va="center", fontsize=5.8,
                    color="white" if rank < 28 or rank > 82 else "#171717")
    previous = primary.iloc[0]["suite"]
    for index, suite in enumerate(primary["suite"]):
        if suite != previous:
            ax.axhline(index - 0.5, color="white", lw=2.2)
            previous = suite
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("Within-task percentile among all 18 models")
    fig.suptitle("Task-level strengths and weaknesses", fontweight="bold", y=0.975)
    fig.text(0.5, 0.025,
             "Cell text is the raw reported score ×100; color is the within-task percentile. White rules separate evaluation suites.",
             ha="center", fontsize=7.5, color=common.NEUTRAL)
    common.save_figure(fig, output, "P3_task_performance_heatmap", formats)


def family_figure(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    primary = data["primary"]; labels = data["labels"]; reference_labels = data["reference_labels"]; short = data["short"]
    assert isinstance(primary, pd.DataFrame) and isinstance(labels, list)
    assert isinstance(reference_labels, list) and isinstance(short, dict)
    ranks = percentile_matrix(primary, reference_labels, labels)
    ranks["family"] = [task_family(str(s), str(t)) for s, t in zip(primary["suite"], primary["task"])]
    profiles = ranks.groupby("family", sort=False)[labels].mean().loc[list(TASK_FAMILIES)]
    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    fig.subplots_adjust(left=0.25, right=0.91, bottom=0.18, top=0.86)
    image = ax.imshow(profiles.T.to_numpy(float), aspect="auto", cmap="cividis", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(TASK_FAMILIES)), TASK_FAMILIES, rotation=24, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(np.arange(len(labels)), [short[label] for label in labels])
    for row in range(len(labels)):
        for col in range(len(TASK_FAMILIES)):
            value = float(profiles.iloc[col, row])
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if value < 28 or value > 82 else "#171717")
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Mean within-task percentile")
    fig.suptitle("Relative performance by task family", fontweight="bold", y=0.97)
    fig.text(0.5, 0.035,
             "Tasks are ranked among all 18 evaluated models before averaging; only the six focal models are displayed.",
             ha="center", fontsize=7.5, color=common.NEUTRAL)
    common.save_figure(fig, output, "P4_task_family_profiles", formats)


def composite_figure(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    result = data["composite"]; focus = data["focus"]; short = data["short"]
    assert isinstance(result, pd.DataFrame) and isinstance(focus, pd.DataFrame) and isinstance(short, dict)
    result = result.sort_values("composite_percentile", ascending=False).reset_index(drop=True)
    y = np.arange(len(result))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), sharey=True,
                             gridspec_kw={"width_ratios": (1.55, 1.0)})
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.16, top=0.80, wspace=0.12)
    for yi, row in result.iterrows():
        color = common.TOKENIZER_COLORS[str(row["tokenizer"])]
        score = float(row["composite_percentile"]); low = float(row["bootstrap_ci_low"]); high = float(row["bootstrap_ci_high"])
        axes[0].errorbar(score, yi, xerr=[[score-low], [high-score]], fmt="none",
                         ecolor=color, elinewidth=1.4, capsize=2.5, alpha=0.8)
        axes[0].scatter(score, yi, color=color, s=65, edgecolor="white", linewidth=0.6, zorder=3)
        axes[1].barh(yi, float(row["fractional_task_wins_among_six"]), color=color, height=0.58)
        axes[1].text(float(row["fractional_task_wins_among_six"]), yi, f" {row['fractional_task_wins_among_six']:.1f}",
                     va="center", fontsize=7)
    axes[0].set_yticks(y, [short[m] for m in result["model"]]); axes[0].invert_yaxis()
    axes[0].set_xlabel("Equal-suite composite percentile"); axes[0].grid(axis="x")
    axes[0].set_title("Aggregate relative performance", loc="left", fontweight="bold")
    axes[1].set_xlabel("Wins among six focal models (of 25)"); axes[1].grid(axis="x")
    axes[1].set_title("Breadth of task leadership", loc="left", fontweight="bold")
    common.panel_label(axes[0], "A"); common.panel_label(axes[1], "B")
    add_legend(fig, data, 0.90)
    fig.suptitle("Focused six-model summary", fontweight="bold", y=0.975)
    fig.text(0.5, 0.035,
             "Bars in A are 95% task-resampling intervals, not training-seed uncertainty; tied task wins in B are split fractionally.",
             ha="center", fontsize=7.5, color=common.NEUTRAL)
    common.save_figure(fig, output, "P5_composite_and_task_wins", formats)


def efficiency_figure(data: dict[str, object], output: Path, formats: Iterable[str]) -> None:
    efficiency = data["efficiency"]; focus = data["focus"]
    assert isinstance(efficiency, pd.DataFrame) and isinstance(focus, pd.DataFrame)
    efficiency = efficiency.set_index("model").loc[focus["label"]].reset_index()
    y = np.arange(len(efficiency))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.8),
                             gridspec_kw={"width_ratios": (1.0, 1.0, 1.35)})
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.17, top=0.79, wspace=0.16)
    model_colors = [common.TOKENIZER_COLORS[str(t)] for t in focus["tokenizer"]]
    text_mib = efficiency["processed_text_bytes"].to_numpy(float) / 2**20
    tokens_m = efficiency["train_tokens"].to_numpy(float) / 1e6
    axes[0].barh(y, text_mib, color=model_colors, height=0.62)
    axes[1].barh(y, tokens_m, color=model_colors, height=0.62)
    for yi, value in enumerate(text_mib): axes[0].text(value, yi, f" {value:.0f}", va="center", fontsize=7)
    for yi, value in enumerate(tokens_m): axes[1].text(value, yi, f" {value:.1f}", va="center", fontsize=7)
    axes[0].set_yticks(y, efficiency["short_label"]); axes[0].invert_yaxis()
    axes[1].set_yticks(y, []); axes[1].invert_yaxis()
    axes[0].set_xlim(0, text_mib.max() * 1.12)
    axes[1].set_xlim(0, tokens_m.max() * 1.12)
    axes[0].set_xlabel("Processed representation (MiB)"); axes[0].grid(axis="x")
    axes[0].set_title("On-disk text", loc="left", fontweight="bold")
    axes[1].set_xlabel("Training tokens (millions)"); axes[1].grid(axis="x")
    axes[1].set_title("Tokenized sequence", loc="left", fontweight="bold")
    annotation_offsets = {"Hybrid": (5, -11), "Atomic within": (5, 6)}
    for row, color in zip(efficiency.itertuples(index=False), model_colors):
        axes[2].scatter(row.train_tokens / 1e6, row.composite_percentile, s=72, color=color,
                        edgecolor="white", linewidth=0.7, zorder=3)
        axes[2].annotate(row.short_label, (row.train_tokens / 1e6, row.composite_percentile),
                         xytext=annotation_offsets.get(row.short_label, (4, 4)),
                         textcoords="offset points", fontsize=7)
    axes[2].set_xlabel("Training tokens per epoch (millions)")
    axes[2].set_ylabel("Equal-suite composite percentile")
    axes[2].grid(); axes[2].set_title("Performance–token trade-off", loc="left", fontweight="bold")
    for index, ax in enumerate(axes): common.panel_label(ax, chr(65 + index))
    add_legend(fig, data, 0.89)
    fig.suptitle("Representation and token efficiency on the same source documents",
                 fontweight="bold", y=0.975)
    fig.text(0.5, 0.035,
             "All models see five epochs. UTF-8 text size is representation-dependent and is a storage measure, not corpus information content.",
             ha="center", fontsize=7.5, color=common.NEUTRAL)
    common.save_figure(fig, output, "P6_efficiency_tradeoffs", formats)


def baseline_benefits_figure(
    data: dict[str, object], output: Path, formats: Iterable[str],
) -> None:
    """Summarize Atomic-within's resource/performance trade-off against baselines."""
    efficiency = data["efficiency"]
    assert isinstance(efficiency, pd.DataFrame)
    wanted = ["Atomic within", "Hanzi BPE", "Full-pinyin BPE"]
    comparison = efficiency.set_index("short_label").loc[wanted].reset_index()
    if comparison["observed_training_seconds"].isna().any():
        raise ValueError("Baseline-benefit figure requires observed training times")
    tokenizer_by_short = {
        str(row.short_label): str(row.tokenizer)
        for row in data["focus"].itertuples(index=False)
    }
    bar_colors = [common.TOKENIZER_COLORS[tokenizer_by_short[label]] for label in wanted]
    atomic_tokens = float(comparison.loc[comparison["short_label"] == "Atomic within", "train_tokens"].iloc[0])
    comparison["context_coverage_index"] = atomic_tokens / comparison["train_tokens"] * 100.0

    panels = (
        ("processed_text_bytes", 2**20, "Processed text storage", "MiB · lower is better", ".0f"),
        ("train_tokens", 1e6, "Tokens per epoch", "millions · lower is better", ".1f"),
        ("optimizer_steps", 1e3, "Optimizer steps", "thousands over 5 epochs · lower is better", ".2f"),
        ("observed_training_seconds", 60, "Observed A100 time", "minutes · lower is better", ".1f"),
        ("context_coverage_index", 1, "Fixed-context coverage", "Atomic within = 100 · higher is better", ".1f"),
        ("composite_percentile", 1, "Composite performance", "all-model percentile · higher is better", ".1f"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.1))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.13, top=0.82, hspace=0.55, wspace=0.32)
    y = np.arange(len(comparison))
    for index, (ax, (column, divisor, title, subtitle, value_format)) in enumerate(zip(axes.flat, panels)):
        values = comparison[column].to_numpy(float) / divisor
        ax.barh(y, values, color=bar_colors, height=0.60)
        ax.set_yticks(y, wanted if index % 3 == 0 else [])
        ax.invert_yaxis()
        ax.set_xlim(0, values.max() * 1.20)
        for yi, value in enumerate(values):
            ax.text(value, yi, f" {value:{value_format}}", va="center", fontsize=7.5)
        ax.set_title(title, loc="left", fontweight="bold", pad=16)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=7.2,
                color=common.NEUTRAL, va="bottom")
        ax.grid(axis="x")
        common.panel_label(ax, chr(65 + index))

    atomic = comparison.set_index("short_label").loc["Atomic within"]
    hanzi = comparison.set_index("short_label").loc["Hanzi BPE"]
    full = comparison.set_index("short_label").loc["Full-pinyin BPE"]
    def saving(base: pd.Series, column: str) -> float:
        return (1.0 - float(atomic[column]) / float(base[column])) * 100.0
    benefit_text = (
        f"Versus Hanzi BPE: {saving(hanzi, 'processed_text_bytes'):.1f}% less text storage · "
        f"{saving(hanzi, 'train_tokens'):.1f}% fewer tokens · "
        f"{saving(hanzi, 'optimizer_steps'):.1f}% fewer steps · "
        f"{saving(hanzi, 'observed_training_seconds'):.1f}% shorter observed run · "
        f"{float(atomic['context_coverage_index'] / hanzi['context_coverage_index'] - 1) * 100:.1f}% more source coverage per context\n"
        f"Versus full-pinyin BPE: {saving(full, 'processed_text_bytes'):.1f}% less text storage · "
        f"{saving(full, 'train_tokens'):.1f}% fewer tokens · "
        f"{saving(full, 'optimizer_steps'):.1f}% fewer steps · "
        f"{saving(full, 'observed_training_seconds'):.1f}% shorter observed run · "
        f"{float(atomic['context_coverage_index'] / full['context_coverage_index'] - 1) * 100:.1f}% more source coverage per context"
    )
    fig.text(0.5, 0.055, benefit_text, ha="center", va="center", fontsize=7.5,
             color="#333333", linespacing=1.45)
    handles = [Patch(facecolor=color, edgecolor="none", label=label)
               for label, color in zip(wanted, bar_colors)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=3, frameon=False)
    fig.suptitle("Atomic-within efficiency relative to standard representation baselines",
                 fontweight="bold", y=0.975)
    fig.text(0.5, 0.012,
             "Same source documents, 97.7M-parameter GPT-2, 16k vocabulary, 512-token context, and five epochs. "
             "Training time is one observed run per model, not a replicated hardware benchmark.",
             ha="center", fontsize=7.2, color=common.NEUTRAL)
    common.save_figure(fig, output, "P7_atomic_within_baseline_benefits", formats)


def write_source_tables(data: dict[str, object], output: Path) -> None:
    tables = output / "source_tables"; tables.mkdir(parents=True, exist_ok=True)
    focus = data["focus"]; labels = data["labels"]; primary = data["primary"]
    assert isinstance(focus, pd.DataFrame) and isinstance(labels, list) and isinstance(primary, pd.DataFrame)
    focus.to_csv(tables / "focused_models.csv", index=False)
    primary[["suite", "task", "subtask", "metric", *labels]].to_csv(
        tables / "focused_primary_scores.csv", index=False,
    )
    data["averages"].to_csv(tables / "focused_suite_averages.csv", index=False)
    data["normalized_suites"].to_csv(tables / "focused_normalized_suite_scores.csv", index=False)
    data["composite"].to_csv(tables / "focused_composite.csv", index=False)
    efficiency = data["efficiency"]
    efficiency.to_csv(tables / "focused_efficiency.csv", index=False)
    indexed = efficiency.set_index("short_label")
    atomic = indexed.loc["Atomic within"]
    benefit_rows = []
    for baseline_label in ("Hanzi BPE", "Full-pinyin BPE"):
        baseline = indexed.loc[baseline_label]
        benefit_rows.append({
            "comparison": f"Atomic within vs {baseline_label}",
            "text_storage_reduction_percent":
                (1 - atomic["processed_text_bytes"] / baseline["processed_text_bytes"]) * 100,
            "training_token_reduction_percent":
                (1 - atomic["train_tokens"] / baseline["train_tokens"]) * 100,
            "optimizer_step_reduction_percent":
                (1 - atomic["optimizer_steps"] / baseline["optimizer_steps"]) * 100,
            "observed_time_reduction_percent":
                (1 - atomic["observed_training_seconds"] / baseline["observed_training_seconds"]) * 100,
            "fixed_context_coverage_gain_percent":
                (baseline["train_tokens"] / atomic["train_tokens"] - 1) * 100,
            "composite_percentile_difference":
                atomic["composite_percentile"] - baseline["composite_percentile"],
        })
    pd.DataFrame(benefit_rows).to_csv(tables / "atomic_within_baseline_benefits.csv", index=False)


def main() -> None:
    args = parse_args(); common.configure_style()
    data = load_data(args)
    if args.validate_only:
        print(f"validated paper-focused inputs for {len(data['labels'])} models")
        return
    output = args.output_dir.resolve(); write_source_tables(data, output)
    generators: dict[str, Callable[[dict[str, object], Path, Iterable[str]], None]] = {
        "design": design_figure, "suites": suite_figure, "tasks": task_figure,
        "families": family_figure, "composite": composite_figure,
        "efficiency": efficiency_figure, "baseline_benefits": baseline_benefits_figure,
    }
    for key in args.figures:
        generators[key](data, output, args.formats); print(f"generated {key}")
    print(f"output directory: {output}")


if __name__ == "__main__":
    main()
