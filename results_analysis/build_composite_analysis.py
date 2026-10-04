#!/usr/bin/env python3
"""Build metric-compatible, cross-task composite summaries.

The primary composite first converts every task to a within-task percentile
rank, averages tasks within each suite, and then gives each suite equal weight.
Bootstrap intervals resample tasks within suites and quantify sensitivity to
task composition.  They are not uncertainty intervals over training seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable


SUITES = (
    "BabyLM official Chinese",
    "Chinese zero-shot",
    "Chinese fine-tune",
    "CogBench",
)
RECOMMENDED_SCHEME = "percentile_equal_suite"
SCHEMES = (
    RECOMMENDED_SCHEME,
    "percentile_equal_task",
    "zscore_equal_suite",
    "zscore_equal_task",
    "percentile_accuracy_only_equal_suite",
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=here / "processed")
    parser.add_argument("--registry", type=Path, default=here / "models.json")
    parser.add_argument("--output-dir", type=Path, default=here / "processed")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261002)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def average_ranks(values: list[float], *, descending: bool = False) -> list[float]:
    """Return one-based average ranks, assigning tied observations their mean rank."""
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=descending)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and math.isclose(
            values[order[end]], values[order[start]], rel_tol=0.0, abs_tol=1e-12
        ):
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate(
    values: dict[str, list[list[float]]], model_index: int, *, equal_suite: bool,
    included_suites: tuple[str, ...] = SUITES,
) -> float:
    suite_values = [values[suite][model_index] for suite in included_suites]
    if equal_suite:
        return fmean(fmean(items) for items in suite_values)
    return fmean(item for items in suite_values for item in items)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))["models"]
    labels = [str(model["label"]) for model in registry]
    model_ids = [str(model["model_id"]) for model in registry]
    metadata = {str(model["label"]): model for model in registry}
    rows = read_csv(args.input_dir / "primary_scores_wide.csv")
    if not rows or set(row["suite"] for row in rows) != set(SUITES):
        raise ValueError("Primary score table does not contain the expected four suites")
    if any(not row.get(label) for row in rows for label in labels):
        raise ValueError("Primary score table contains missing registered-model scores")

    percentile_by_suite: dict[str, list[list[float]]] = {
        suite: [[] for _ in labels] for suite in SUITES
    }
    zscore_by_suite: dict[str, list[list[float]]] = {
        suite: [[] for _ in labels] for suite in SUITES
    }
    task_rank_by_suite: dict[str, list[list[float]]] = {
        suite: [[] for _ in labels] for suite in SUITES
    }
    fractional_wins_by_suite: dict[str, list[list[float]]] = {
        suite: [[] for _ in labels] for suite in SUITES
    }
    degenerate_tasks: list[str] = []
    for row in rows:
        suite = row["suite"]
        scores = [float(row[label]) for label in labels]
        ascending_ranks = average_ranks(scores)
        descending_ranks = average_ranks(scores, descending=True)
        percentiles = [(rank - 1.0) / (len(labels) - 1.0) * 100.0
                       for rank in ascending_ranks]
        mean = fmean(scores)
        standard_deviation = pstdev(scores)
        if math.isclose(standard_deviation, 0.0, abs_tol=1e-15):
            zscores = [0.0] * len(labels)
            degenerate_tasks.append(row["comparison_key"])
        else:
            zscores = [(score - mean) / standard_deviation for score in scores]
        best = max(scores)
        winners = [index for index, score in enumerate(scores)
                   if math.isclose(score, best, rel_tol=0.0, abs_tol=1e-12)]
        win_share = 1.0 / len(winners)
        for index in range(len(labels)):
            percentile_by_suite[suite][index].append(percentiles[index])
            zscore_by_suite[suite][index].append(zscores[index])
            task_rank_by_suite[suite][index].append(descending_ranks[index])
            fractional_wins_by_suite[suite][index].append(
                win_share if index in winners else 0.0
            )

    task_counts = {suite: len(percentile_by_suite[suite][0]) for suite in SUITES}
    if task_counts != {
        "BabyLM official Chinese": 16,
        "Chinese zero-shot": 3,
        "Chinese fine-tune": 4,
        "CogBench": 2,
    }:
        raise ValueError(f"Unexpected primary task counts: {task_counts}")

    scheme_values: dict[str, list[float]] = {
        RECOMMENDED_SCHEME: [
            aggregate(percentile_by_suite, index, equal_suite=True)
            for index in range(len(labels))
        ],
        "percentile_equal_task": [
            aggregate(percentile_by_suite, index, equal_suite=False)
            for index in range(len(labels))
        ],
        "zscore_equal_suite": [
            aggregate(zscore_by_suite, index, equal_suite=True)
            for index in range(len(labels))
        ],
        "zscore_equal_task": [
            aggregate(zscore_by_suite, index, equal_suite=False)
            for index in range(len(labels))
        ],
        "percentile_accuracy_only_equal_suite": [
            aggregate(
                percentile_by_suite, index, equal_suite=True,
                included_suites=SUITES[:-1],
            )
            for index in range(len(labels))
        ],
    }

    sensitivity_rows: list[dict[str, object]] = []
    scheme_ranks: dict[str, list[float]] = {}
    for scheme in SCHEMES:
        ranks = average_ranks(scheme_values[scheme], descending=True)
        scheme_ranks[scheme] = ranks
        for index, label in enumerate(labels):
            model = metadata[label]
            sensitivity_rows.append({
                "model_id": model_ids[index], "model": label,
                "architecture": model["architecture"], "tokenizer": model["tokenizer"],
                "scale": model["scale"], "scheme": scheme,
                "score": scheme_values[scheme][index], "rank": ranks[index],
            })

    suite_rows: list[dict[str, object]] = []
    for suite in SUITES:
        for index, label in enumerate(labels):
            percentiles = percentile_by_suite[suite][index]
            model = metadata[label]
            suite_rows.append({
                "model_id": model_ids[index], "model": label,
                "architecture": model["architecture"], "tokenizer": model["tokenizer"],
                "scale": model["scale"], "suite": suite,
                "task_count": len(percentiles),
                "mean_percentile": fmean(percentiles),
                "mean_zscore": fmean(zscore_by_suite[suite][index]),
                "mean_rank": fmean(task_rank_by_suite[suite][index]),
                "percentile_sd": pstdev(percentiles),
            })

    rng = random.Random(args.seed)
    bootstrap_scores = [[] for _ in labels]
    bootstrap_ranks = [[] for _ in labels]
    top_shares = [0.0] * len(labels)
    for _ in range(args.bootstrap_replicates):
        replicate = [0.0] * len(labels)
        for suite in SUITES:
            task_count = task_counts[suite]
            sampled = [rng.randrange(task_count) for _ in range(task_count)]
            for index in range(len(labels)):
                task_values = percentile_by_suite[suite][index]
                replicate[index] += fmean(task_values[position] for position in sampled) / len(SUITES)
        ranks = average_ranks(replicate, descending=True)
        best = max(replicate)
        winners = [index for index, value in enumerate(replicate)
                   if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)]
        for index in range(len(labels)):
            bootstrap_scores[index].append(replicate[index])
            bootstrap_ranks[index].append(ranks[index])
        for index in winners:
            top_shares[index] += 1.0 / len(winners)

    summary_rows: list[dict[str, object]] = []
    all_percentiles = [
        [item for suite in SUITES for item in percentile_by_suite[suite][index]]
        for index in range(len(labels))
    ]
    for index, label in enumerate(labels):
        model = metadata[label]
        percentiles = all_percentiles[index]
        suite_win_rates = [fmean(fractional_wins_by_suite[suite][index]) for suite in SUITES]
        ranks_across_schemes = [scheme_ranks[scheme][index] for scheme in SCHEMES]
        summary_rows.append({
            "model_id": model_ids[index], "model": label,
            "architecture": model["architecture"], "tokenizer": model["tokenizer"],
            "scale": model["scale"],
            "composite_percentile": scheme_values[RECOMMENDED_SCHEME][index],
            "composite_rank": scheme_ranks[RECOMMENDED_SCHEME][index],
            "bootstrap_ci_low": quantile(bootstrap_scores[index], 0.025),
            "bootstrap_ci_high": quantile(bootstrap_scores[index], 0.975),
            "bootstrap_mean_rank": fmean(bootstrap_ranks[index]),
            "bootstrap_rank_low": quantile(bootstrap_ranks[index], 0.025),
            "bootstrap_rank_high": quantile(bootstrap_ranks[index], 0.975),
            "bootstrap_top_probability": top_shares[index] / args.bootstrap_replicates,
            "task_percentile_sd": pstdev(percentiles),
            "task_percentile_iqr": quantile(percentiles, 0.75) - quantile(percentiles, 0.25),
            "equal_suite_fractional_win_rate": fmean(suite_win_rates),
            "mean_task_rank_equal_suite": fmean(
                fmean(task_rank_by_suite[suite][index]) for suite in SUITES
            ),
            "sensitivity_rank_min": min(ranks_across_schemes),
            "sensitivity_rank_max": max(ranks_across_schemes),
        })
    summary_rows.sort(key=lambda row: float(row["composite_rank"]))

    agreement_rows: list[dict[str, object]] = []
    for first in SCHEMES:
        for second in SCHEMES:
            x = scheme_ranks[first]
            y = scheme_ranks[second]
            xmean, ymean = fmean(x), fmean(y)
            numerator = sum((a - xmean) * (b - ymean) for a, b in zip(x, y))
            denominator = math.sqrt(
                sum((a - xmean) ** 2 for a in x) * sum((b - ymean) ** 2 for b in y)
            )
            agreement_rows.append({
                "scheme_a": first, "scheme_b": second,
                "spearman_rank_correlation": numerator / denominator if denominator else 1.0,
            })

    output = args.output_dir
    write_csv(output / "composite_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(output / "composite_sensitivity.csv", sensitivity_rows, list(sensitivity_rows[0]))
    write_csv(output / "composite_suite_scores.csv", suite_rows, list(suite_rows[0]))
    write_csv(output / "composite_method_agreement.csv", agreement_rows, list(agreement_rows[0]))
    methodology = {
        "primary_scheme": RECOMMENDED_SCHEME,
        "suite_order": list(SUITES),
        "task_counts": task_counts,
        "model_count": len(labels),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "degenerate_tasks_assigned_zero_zscore": degenerate_tasks,
        "interpretation": (
            "Bootstrap intervals quantify sensitivity to resampling the observed tasks within suites; "
            "they do not represent uncertainty over training seeds or future benchmark populations."
        ),
    }
    (output / "composite_methodology.json").write_text(
        json.dumps(methodology, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Cross-task composite analysis", "",
        "## Prespecified primary summary", "",
        "Each task is converted to a within-task percentile rank across the registered models. "
        "Tasks are averaged within each evaluation suite, and the four suite means are then "
        "weighted equally. This avoids directly averaging accuracy, F1, MCC, and correlation.", "",
        "| Rank | Model | Composite percentile | 95% task-resampling interval | Task percentile SD | Sensitivity rank range |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        markdown.append(
            f"| {float(row['composite_rank']):.0f} | {row['model']} | "
            f"{float(row['composite_percentile']):.2f} | "
            f"{float(row['bootstrap_ci_low']):.2f}–{float(row['bootstrap_ci_high']):.2f} | "
            f"{float(row['task_percentile_sd']):.2f} | "
            f"{float(row['sensitivity_rank_min']):.0f}–{float(row['sensitivity_rank_max']):.0f} |"
        )
    markdown.extend([
        "", "## Interpretation safeguards", "",
        f"- The composite is relative to the current {len(labels)}-model comparison set; it is not an absolute accuracy score.",
        "- The interval resamples observed tasks within suites. It measures sensitivity to task composition, not training-seed uncertainty or conventional statistical significance.",
        "- Task percentile SD describes consistency of relative placement across tasks; lower is more consistent, not necessarily more accurate.",
        "- One trained seed is available per condition, so small ordering differences must not be described as seed-robust effects.",
        "- Rankings from all five aggregation definitions are retained in `composite_sensitivity.csv`; conclusions should emphasize models that remain strong across definitions.",
        "", "## Reproducibility", "",
        f"The task bootstrap used `{args.bootstrap_replicates}` paired replicates with seed `{args.seed}`. "
        "See `composite_methodology.json` for the machine-readable specification.", "",
    ])
    (output / "composite_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Models: {len(labels)}")
    print(f"Primary tasks: {len(rows)} across {len(SUITES)} suites")
    print(f"Bootstrap replicates: {args.bootstrap_replicates} (seed={args.seed})")
    print(f"Degenerate z-score tasks: {len(degenerate_tasks)}")
    print(f"Top model: {summary_rows[0]['model']} ({summary_rows[0]['composite_percentile']:.2f})")
    print(f"Output directory: {output.resolve()}")


if __name__ == "__main__":
    main()
