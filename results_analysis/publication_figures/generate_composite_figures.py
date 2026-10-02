#!/usr/bin/env python3
"""Generate publication figures for normalized cross-task composites."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import generate_figures as common


FIGURE_KEYS = ("main", "sensitivity")
SCHEME_LABELS = {
    "percentile_equal_suite": "Percentile · equal suite",
    "percentile_equal_task": "Percentile · equal task",
    "zscore_equal_suite": "z-score · equal suite",
    "zscore_equal_task": "z-score · equal task",
    "percentile_accuracy_only_equal_suite": "Percentile · no CogBench",
}
SCHEME_MARKERS = ("o", "s", "D", "^", "P")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "results_analysis/processed")
    parser.add_argument("--registry", type=Path, default=root / "results_analysis/models.json")
    parser.add_argument("--main-output-dir", type=Path, default=here / "main_body_output")
    parser.add_argument("--supp-output-dir", type=Path, default=here / "output")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"),
                        default=("pdf", "svg"))
    parser.add_argument("--figures", nargs="+", choices=FIGURE_KEYS, default=FIGURE_KEYS)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_inputs(input_dir: Path, registry: Path) -> dict[str, object]:
    models, labels = common.load_registry(registry)
    summary = pd.read_csv(input_dir / "composite_summary.csv")
    sensitivity = pd.read_csv(input_dir / "composite_sensitivity.csv")
    suites = pd.read_csv(input_dir / "composite_suite_scores.csv")
    if set(summary["model"]) != set(labels) or len(summary) != len(labels):
        raise ValueError("Composite summary does not contain exactly one row per registered model")
    if summary.isna().any().any():
        raise ValueError("Composite summary contains missing values")
    expected_sensitivity = {(label, scheme) for label in labels for scheme in SCHEME_LABELS}
    observed_sensitivity = set(zip(sensitivity["model"], sensitivity["scheme"]))
    if expected_sensitivity != observed_sensitivity:
        raise ValueError("Composite sensitivity coverage is incomplete")
    expected_suites = {(label, suite) for label in labels for suite in common.SUITES}
    if expected_suites != set(zip(suites["model"], suites["suite"])):
        raise ValueError("Composite suite coverage is incomplete")
    return {
        "models": models, "labels": labels, "summary": summary,
        "sensitivity": sensitivity, "suites": suites,
    }


def style_for_model(metadata: pd.DataFrame, label: str) -> dict[str, object]:
    row = metadata.loc[label]
    sizes = common.scale_point_sizes(metadata["scale"].astype(str))
    return {
        "color": common.TOKENIZER_COLORS[str(row["tokenizer"])],
        "marker": common.ARCH_MARKERS[str(row["architecture"])],
        "s": sizes[str(row["scale"])],
    }


def composite_main_figure(
    data: dict[str, object], output: Path, formats: Iterable[str],
) -> None:
    summary = data["summary"]
    models = data["models"]
    assert isinstance(summary, pd.DataFrame) and isinstance(models, pd.DataFrame)
    ordered = summary.sort_values("composite_rank", kind="stable").reset_index(drop=True)
    metadata = models.set_index("label")
    y = np.arange(len(ordered))
    height = max(7.3, 3.3 + 0.37 * len(ordered))
    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, height), sharey=True,
        gridspec_kw={"width_ratios": (1.7, 1.0)},
    )
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.14, top=0.84, wspace=0.10)
    score_ax, variability_ax = axes
    for yi, row in ordered.iterrows():
        label = str(row["model"])
        style = style_for_model(metadata, label)
        score = float(row["composite_percentile"])
        low = float(row["bootstrap_ci_low"])
        high = float(row["bootstrap_ci_high"])
        score_ax.errorbar(
            score, yi, xerr=[[score - low], [high - score]], fmt="none",
            ecolor=style["color"], elinewidth=1.25, capsize=2.5, alpha=0.78,
            zorder=1,
        )
        score_ax.scatter(
            score, yi, color=style["color"], marker=style["marker"], s=style["s"],
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        variability_ax.scatter(
            float(row["task_percentile_sd"]), yi, color=style["color"],
            marker=style["marker"], s=style["s"], edgecolor="white",
            linewidth=0.6, zorder=3,
        )
    score_ax.set_yticks(y, ordered["model"])
    score_ax.invert_yaxis()
    score_ax.set_xlabel("Equal-suite mean within-task percentile")
    score_ax.set_title("Overall relative performance", loc="left", fontweight="bold")
    score_ax.grid(axis="x")
    variability_ax.set_xlabel("Task-to-task percentile SD")
    variability_ax.set_title("Performance variability", loc="left", fontweight="bold")
    variability_ax.grid(axis="x")
    common.panel_label(score_ax, "A")
    common.panel_label(variability_ax, "B")

    tokenizer_handles = [Patch(facecolor=color, edgecolor="none", label=tokenizer)
                         for tokenizer, color in common.TOKENIZER_COLORS.items()]
    architecture_handles = [Line2D([], [], marker=marker, linestyle="", color="#333333",
                                   label=architecture, markersize=6)
                            for architecture, marker in common.ARCH_MARKERS.items()]
    fig.legend(
        handles=[*tokenizer_handles, *architecture_handles], loc="upper center",
        bbox_to_anchor=(0.5, 0.915), ncol=6, frameon=False,
    )
    fig.suptitle("Cross-suite performance and consistency", fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.045,
        "Tasks are percentile-ranked across models, averaged within suite, then averaged equally across suites. "
        "Bars are 95% task-resampling intervals; they do not represent training-seed uncertainty. Lower SD means more consistent relative performance.",
        ha="center", fontsize=7.5, color=common.NEUTRAL, wrap=True,
    )
    common.save_figure(fig, output, "M5_composite_performance", formats)


def composite_sensitivity_figure(
    data: dict[str, object], output: Path, formats: Iterable[str],
) -> None:
    summary = data["summary"]
    sensitivity = data["sensitivity"]
    suites = data["suites"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(sensitivity, pd.DataFrame) and isinstance(suites, pd.DataFrame)
    labels = summary.sort_values("composite_rank", kind="stable")["model"].tolist()
    y = np.arange(len(labels))
    figure_height = max(8.2, 3.8 + 0.39 * len(labels))
    fig, axes = plt.subplots(
        1, 2, figsize=(14.2, figure_height), sharey=True,
        gridspec_kw={"width_ratios": (1.05, 1.2)},
    )
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.17, top=0.82, wspace=0.17)
    rank_ax, suite_ax = axes
    scheme_colors = plt.get_cmap("Dark2")(np.linspace(0, 1, len(SCHEME_LABELS)))
    for yi, label in enumerate(labels):
        subset = sensitivity[sensitivity["model"] == label].set_index("scheme")
        ranks = [float(subset.loc[scheme, "rank"]) for scheme in SCHEME_LABELS]
        rank_ax.plot([min(ranks), max(ranks)], [yi, yi], color="#C5C5C5", lw=1.2, zorder=1)
        for index, scheme in enumerate(SCHEME_LABELS):
            rank_ax.scatter(
                ranks[index], yi, s=30, marker=SCHEME_MARKERS[index],
                color=scheme_colors[index], edgecolor="white", linewidth=0.45, zorder=3,
            )
    rank_ax.set_yticks(y, labels)
    rank_ax.invert_yaxis()
    rank_ax.set_xlim(0.4, len(labels) + 0.6)
    rank_ax.set_xticks(range(1, len(labels) + 1))
    rank_ax.set_xlabel("Model rank (1 = best)")
    rank_ax.set_title("Sensitivity to aggregation choice", loc="left", fontweight="bold")
    rank_ax.grid(axis="x")

    suite_matrix = (
        suites.pivot(index="model", columns="suite", values="mean_percentile")
        .loc[labels, list(common.SUITES)]
    )
    image = suite_ax.imshow(
        suite_matrix.to_numpy(float), aspect="auto", cmap="cividis", vmin=0, vmax=100,
        interpolation="nearest",
    )
    suite_ax.set_xticks(
        np.arange(len(common.SUITES)),
        [common.SUITE_SHORT[suite] for suite in common.SUITES],
        rotation=28, ha="right", rotation_mode="anchor",
    )
    suite_ax.set_yticks(y, labels)
    suite_ax.tick_params(length=0)
    for row in range(len(labels)):
        for column in range(len(common.SUITES)):
            value = float(suite_matrix.iloc[row, column])
            color = "white" if value < 36 or value > 82 else "#171717"
            suite_ax.text(column, row, f"{value:.1f}", ha="center", va="center",
                          fontsize=6.5, color=color)
    suite_ax.set_title("Normalized performance by suite", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=suite_ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Mean within-task percentile")
    common.panel_label(rank_ax, "A")
    common.panel_label(suite_ax, "B")

    handles = [Line2D([], [], marker=SCHEME_MARKERS[index], linestyle="",
                      color=scheme_colors[index], label=label, markersize=6)
               for index, label in enumerate(SCHEME_LABELS.values())]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, frameon=False)
    fig.suptitle("Robustness of the cross-task model ordering", fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.045,
        "Panel A compares rank under five prespecified normalization and weighting choices; horizontal lines span each model's rank range. "
        "Panel B reports suite-specific percentile means and does not combine raw accuracy, F1, MCC, or correlation values.",
        ha="center", fontsize=7.5, color=common.NEUTRAL, wrap=True,
    )
    common.save_figure(fig, output, "07_composite_sensitivity", formats)


def main() -> None:
    args = parse_args()
    common.configure_style()
    data = load_inputs(args.input_dir.resolve(), args.registry.resolve())
    if args.validate_only:
        print(f"validated composite figures for {len(data['labels'])} models")
        return
    if "main" in args.figures:
        composite_main_figure(data, args.main_output_dir.resolve(), args.formats)
        print("generated main composite figure")
    if "sensitivity" in args.figures:
        composite_sensitivity_figure(data, args.supp_output_dir.resolve(), args.formats)
        print("generated composite sensitivity figure")


if __name__ == "__main__":
    main()
