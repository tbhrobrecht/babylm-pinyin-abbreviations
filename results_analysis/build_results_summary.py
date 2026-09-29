#!/usr/bin/env python3
"""Build reproducible comparison tables for all Mandarin BabyLM models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


OFFICIAL_ZERO = (
    "winogrande_zh_mubench", "xstorycloze_zh_mubench",
    "hellaswag_zh_mubench", "xcomps_zh", "zhoblimp",
    "global_piqa_parallel_zh", "global_piqa_nonparallel_zh",
)
OFFICIAL_FINE = (
    "pos", "arc", "belebele", "bmlama", "include", "mnli", "sib200",
    "truthfulqa", "xnli",
)
SUITES = (
    "BabyLM official Chinese", "Chinese zero-shot",
    "Chinese fine-tune", "CogBench",
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    architecture: str
    tokenizer: str
    scale: str
    cohort: str
    official_format: str
    official_path: Path
    chinese_path: Path
    config_path: Path
    model_path: Path | None = None
    metadata_path: Path | None = None


@dataclass(frozen=True)
class Contrast:
    name: str
    category: str
    minuend: str
    subtrahend: str


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=here / "models.json")
    parser.add_argument("--output", type=Path, default=here / "processed")
    parser.add_argument("--repository-root", type=Path, default=here.parent)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_registry(path: Path, root: Path) -> list[ModelSpec]:
    def optional(value: str | None) -> Path | None:
        return (root / value).resolve() if value else None

    models = []
    for item in read_json(path)["models"]:
        official = item["official"]
        models.append(ModelSpec(
            model_id=item["model_id"], label=item["label"],
            architecture=item["architecture"], tokenizer=item["tokenizer"],
            scale=item["scale"], cohort=item["cohort"],
            official_format=official["format"],
            official_path=(root / official["path"]).resolve(),
            chinese_path=(root / item["chinese_path"]).resolve(),
            config_path=(root / item["config_path"]).resolve(),
            model_path=optional(item.get("model_path")),
            metadata_path=optional(item.get("metadata_path")),
        ))
    if len({m.model_id for m in models}) != len(models):
        raise ValueError("Model registry IDs must be unique")
    if len({m.label for m in models}) != len(models):
        raise ValueError("Model registry labels must be unique")
    return models


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def source_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def base_row(model: ModelSpec) -> dict[str, str]:
    return {
        "model_id": model.model_id, "model": model.label,
        "architecture": model.architecture, "tokenizer": model.tokenizer,
        "scale": model.scale, "cohort": model.cohort,
    }


def metric_row(
    model: ModelSpec, root: Path, path: Path, *, suite: str, task: str,
    subtask: str, section: str, metric: str, score: float, primary: bool,
    temperature: float | str = "", n_result_files: int | str = "",
    fast: bool | str = "",
) -> dict[str, Any]:
    return {
        **base_row(model), "suite": suite, "task": task, "subtask": subtask,
        "section": section, "metric": metric, "score": float(score),
        "primary_metric": primary, "status": "locally_scored",
        "temperature": temperature, "n_result_files": n_result_files,
        "fast": fast, "source_file": source_path(path, root),
    }


def flatten_score(value: Any) -> tuple[str, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "aggregate", float(value)
    if isinstance(value, dict):
        values = [(str(k), float(v)) for k, v in value.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(values) == 1:
            return values[0]
    raise ValueError(f"Expected one numeric score, received {value!r}")


def load_official_submission(model: ModelSpec, root: Path) -> list[dict[str, Any]]:
    rows = []
    for task, value in read_json(model.official_path).items():
        subtask, score = flatten_score(value)
        rows.append(metric_row(
            model, root, model.official_path, suite=SUITES[0], task=task,
            subtask=subtask, section="submission", metric="score", score=score,
            primary=True,
        ))
    return rows


def official_model_dir(root: Path, section: str, model_id: str) -> Path:
    section_root = root / section
    matches = [p for p in section_root.iterdir()
               if p.is_dir() and p.name.split("__")[-1] == model_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one {section} directory for {model_id}; found {len(matches)}")
    return matches[0]


def load_official_raw(model: ModelSpec, root: Path) -> list[dict[str, Any]]:
    rows = []
    zero_dir = official_model_dir(model.official_path, "zeroshot", model.model_id)
    task_data: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(zero_dir.glob("results_*.json")):
        for task, result in read_json(path).get("results", {}).items():
            task_data[task] = (result, path)
    for task in OFFICIAL_ZERO:
        if task not in task_data:
            raise ValueError(f"Missing official zero-shot result for {model.model_id}/{task}")
        result, path = task_data[task]
        key = "acc_norm,none" if task.startswith("global_piqa") else "acc,none"
        rows.append(metric_row(
            model, root, path, suite=SUITES[0], task=task, subtask=task,
            section="submission", metric="score", score=result[key], primary=True,
        ))

    fine_root = model.official_path / "finetune" / model.model_id
    for task in OFFICIAL_FINE:
        if task == "pos":
            preferred = fine_root / "pos" / "zh" / "eval_results.json"
            fallback = fine_root / "pos" / "eval_results.json"
            path = preferred if preferred.is_file() else fallback
        else:
            path = fine_root / "zh" / task / "eval_results.json"
        result = read_json(path)
        rows.append(metric_row(
            model, root, path, suite=SUITES[0], task=task, subtask="zh",
            section="submission", metric="score", score=result["eval_accuracy"],
            primary=True,
        ))
    return rows


def load_official(model: ModelSpec, root: Path) -> list[dict[str, Any]]:
    if model.official_format == "submission":
        rows = load_official_submission(model, root)
    elif model.official_format == "raw":
        rows = load_official_raw(model, root)
    else:
        raise ValueError(f"Unknown official format: {model.official_format}")
    found = {r["task"] for r in rows}
    expected = set(OFFICIAL_ZERO) | set(OFFICIAL_FINE)
    if found != expected:
        raise ValueError(
            f"Official task mismatch for {model.model_id}: "
            f"missing={sorted(expected-found)}, extra={sorted(found-expected)}"
        )
    return rows


def parse_zero_report(path: Path, root: Path, model: ModelSpec, task: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^TEMPERATURE:\s*([0-9.]+)", text, re.MULTILINE)
    temperature: float | str = float(match.group(1)) if match else ""
    rows = []
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            section = line[4:].strip().lower().replace(" ", "_")
            continue
        if line.startswith("TEMPERATURE:"):
            continue
        if section == "average_accuracy" and re.fullmatch(r"-?[0-9.]+", line):
            rows.append(metric_row(
                model, root, path, suite=SUITES[1], task=task,
                subtask="aggregate", section=section, metric="accuracy",
                score=float(line) / 100.0, primary=True, temperature=temperature,
            ))
            continue
        item = re.fullmatch(r"([^:]+):\s*(-?[0-9.]+)", line)
        if item and section:
            rows.append(metric_row(
                model, root, path, suite=SUITES[1], task=task,
                subtask=item.group(1).strip(), section=section, metric="accuracy",
                score=float(item.group(2)) / 100.0, primary=False,
                temperature=temperature,
            ))
    if sum(bool(r["primary_metric"]) for r in rows) != 1:
        raise ValueError(f"Expected one average accuracy in {path}")
    return rows


def load_chinese(model: ModelSpec, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    model_root = model.chinese_path / "main"
    for task in ("zhoblimp", "hanzi_structure", "hanzi_pinyin"):
        report = model_root / "zero_shot" / "causal" / task / task / "best_temperature_report.txt"
        rows.extend(parse_zero_report(report, root, model, task))
    for task in ("afqmc", "ocnli", "tnews", "cluewsc2020"):
        report = model_root / "finetune" / task / "results.txt"
        parsed = re.findall(
            r"^([A-Za-z0-9_]+):\s*(-?[0-9.eE+\-]+)",
            report.read_text(encoding="utf-8"), re.MULTILINE,
        )
        if not parsed:
            raise ValueError(f"No metrics found in {report}")
        for metric, value in parsed:
            rows.append(metric_row(
                model, root, report, suite=SUITES[2], task=task, subtask="zh",
                section="evaluation", metric=metric.lower(), score=float(value),
                primary=metric.lower() == "accuracy",
            ))
    for task in ("word_fmri", "fmri"):
        matches = list((model_root / "cogbench" / task).glob("*report.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected one CogBench report for {model.model_id}/{task}")
        path = matches[0]
        report = read_json(path)
        for metric in ("mean", "min", "max"):
            rows.append(metric_row(
                model, root, path, suite=SUITES[3], task=task,
                subtask="aggregate", section="aggregate", metric=metric,
                score=report[metric], primary=metric == "mean",
                n_result_files=int(report["n_result_files"]), fast=bool(report["fast"]),
            ))
        for item in report["files"]:
            item_path = Path(item["file"])
            details.append({
                **base_row(model), "task": task,
                "region": item_path.parent.name if task == "fmri" else "",
                "subject": item_path.stem.replace("_score", "").replace("_average", ""),
                "result_file": item_path.name, "score": float(item["value"]),
                "source_file": source_path(path, root),
            })
    return rows, details


def safetensors_metadata(path: Path, root: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    return {
        "stored_tensor_parameters": sum(math.prod(t["shape"]) for t in tensors),
        "tensor_count": len(tensors), "model_file_bytes": path.stat().st_size,
        "model_file": source_path(path, root),
    }


def load_metadata(models: list[ModelSpec], root: Path) -> list[dict[str, Any]]:
    cache: dict[Path, Any] = {}
    rows = []
    for model in models:
        config = read_json(model.config_path)
        if model.model_path:
            stored = safetensors_metadata(model.model_path / "model.safetensors", root)
        elif model.metadata_path:
            cache.setdefault(model.metadata_path, read_json(model.metadata_path))
            stored = cache[model.metadata_path][model.model_id]
        else:
            raise ValueError(f"No model metadata for {model.model_id}")
        rows.append({
            **base_row(model),
            "stored_tensor_parameters": int(stored["stored_tensor_parameters"]),
            "tensor_count": int(stored["tensor_count"]),
            "model_file_bytes": int(stored["model_file_bytes"]),
            "dtype": config.get("dtype", config.get("torch_dtype", "")),
            "vocab_size": config.get("vocab_size", ""),
            "hidden_size": config.get("hidden_size", config.get("n_embd", "")),
            "num_hidden_layers": config.get("num_hidden_layers", config.get("n_layer", "")),
            "num_attention_heads": config.get("num_attention_heads", config.get("n_head", "")),
            "intermediate_size": config.get("intermediate_size", ""),
            "max_position_embeddings": config.get("max_position_embeddings", ""),
            "tie_word_embeddings": config.get("tie_word_embeddings", ""),
            "source_file": source_path(model.config_path, root),
            "model_file": stored["model_file"],
        })
    return rows


def verify_manifest(path: Path) -> tuple[int, list[str]]:
    checked = 0
    errors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        source = path.parent / relative.removeprefix("./")
        if not source.is_file():
            errors.append(f"missing: {source}")
        elif hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            errors.append(f"hash mismatch: {source}")
        checked += source.is_file()
    return checked, errors


def load_legacy_hidden(models: list[ModelSpec], root: Path) -> list[dict[str, Any]]:
    """Preserve prediction-only coverage from the original compact snapshot."""
    by_id = {model.model_id: model for model in models}
    path = root / "results_analysis/raw_snapshot/hidden_task_manifest.json"
    rows = []
    for item in read_json(path):
        model = by_id[item["model_id"]]
        rows.append({**base_row(model), **item})
    return rows


def comparison_key(row: dict[str, Any]) -> str:
    return " | ".join(str(row[k]) for k in ("suite", "task", "subtask", "metric"))


def make_wide(primary: list[dict[str, Any]], models: list[ModelSpec]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    labels = [m.label for m in models]
    for row in primary:
        key = comparison_key(row)
        record = grouped.setdefault(key, {
            "comparison_key": key, "suite": row["suite"], "task": row["task"],
            "subtask": row["subtask"], "metric": row["metric"],
        })
        if row["model"] in record:
            raise ValueError(f"Duplicate primary metric for {row['model']}: {key}")
        record[row["model"]] = row["score"]
    for row in grouped.values():
        missing = [label for label in labels if label not in row]
        if missing:
            raise ValueError(f"Missing models for {row['comparison_key']}: {missing}")
        best = max(float(row[label]) for label in labels)
        row["best_score"] = best
        row["winner"] = "; ".join(
            label for label in labels
            if math.isclose(float(row[label]), best, rel_tol=0, abs_tol=1e-12)
        )
    return sorted(grouped.values(), key=lambda r: (r["suite"], r["task"], r["metric"]))


def build_contrasts(models: list[ModelSpec]) -> list[Contrast]:
    lookup = {(m.scale, m.architecture, m.tokenizer): m for m in models}
    contrasts: list[Contrast] = []

    def add(name: str, category: str, plus: tuple[str, str, str], minus: tuple[str, str, str]) -> None:
        contrasts.append(Contrast(name, category, lookup[plus].label, lookup[minus].label))

    for architecture in ("GPT2", "Qwen2"):
        for tokenizer in ("Hybrid", "BPE"):
            add(f"{architecture} {tokenizer}: 100M - 30M", "scale",
                ("100M", architecture, tokenizer), ("30M", architecture, tokenizer))
        for scale in ("30M", "100M"):
            add(f"{scale} {architecture}: Hybrid - BPE", "baseline_tokenizer",
                (scale, architecture, "Hybrid"), (scale, architecture, "BPE"))
        add(f"30M {architecture}: Atomic cross - within", "boundary_policy",
            ("30M", architecture, "Atomic BPE cross"),
            ("30M", architecture, "Atomic BPE within"))
        for tokenizer in ("Atomic BPE within", "Atomic BPE cross"):
            add(f"30M {architecture}: {tokenizer} - BPE", "atomic_vs_bpe",
                ("30M", architecture, tokenizer), ("30M", architecture, "BPE"))
    for scale, tokenizer in (
        ("30M", "Hybrid"), ("30M", "BPE"), ("100M", "Hybrid"),
        ("100M", "BPE"), ("30M", "Atomic BPE within"),
        ("30M", "Atomic BPE cross"),
    ):
        add(f"{scale} {tokenizer}: Qwen2 - GPT2", "architecture",
            (scale, "Qwen2", tokenizer), (scale, "GPT2", tokenizer))
    return contrasts


def make_effects(wide: list[dict[str, Any]], contrasts: list[Contrast]) -> list[dict[str, Any]]:
    return [{
        "comparison_key": row["comparison_key"], "suite": row["suite"],
        "task": row["task"], "metric": row["metric"],
        "contrast": contrast.name, "category": contrast.category,
        "minuend": contrast.minuend, "subtrahend": contrast.subtrahend,
        "difference": float(row[contrast.minuend]) - float(row[contrast.subtrahend]),
    } for row in wide for contrast in contrasts]


def make_suite_averages(wide: list[dict[str, Any]], models: list[ModelSpec]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        for suite in SUITES:
            matches = [row for row in wide if row["suite"] == suite]
            rows.append({
                **base_row(model), "suite": suite, "task_count": len(matches),
                "mean_score": fmean(float(row[model.label]) for row in matches),
            })
    return rows


def make_contrast_summary(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in effects:
        key = (row["suite"], row["contrast"], row["category"],
               row["minuend"], row["subtrahend"])
        grouped.setdefault(key, []).append(float(row["difference"]))
    return [{
        "suite": key[0], "contrast": key[1], "category": key[2],
        "minuend": key[3], "subtrahend": key[4], "task_count": len(values),
        "mean_difference": fmean(values),
    } for key, values in sorted(grouped.items())]


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |",
              "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(output)


def build_summary(
    output: Path, models: list[ModelSpec], metrics: list[dict[str, Any]],
    wide: list[dict[str, Any]], metadata: list[dict[str, Any]],
    averages: list[dict[str, Any]], contrasts: list[dict[str, Any]], checked: int,
) -> None:
    mean = {(r["model"], r["suite"]): r["mean_score"] for r in averages}
    cohorts = list(dict.fromkeys(m.cohort for m in models))
    lines = [
        "# Mandarin BabyLM 12-model comparison", "", "## Validation and scope", "",
        f"- Compared `{len(models)}` models across `{len(wide)}` comparable primary metrics.",
        f"- Parsed `{len(metrics)}` total metric rows, including supplementary metrics.",
        f"- Verified `{checked}` legacy snapshot files against `SOURCE_MANIFEST.sha256`.",
        "- Raw official outputs use evaluator collation rules: Global PIQA uses normalized accuracy; other zero-shot tasks use accuracy.",
        "- Server-scored tasks are excluded from official local scores; Chinese-pipeline Hanzi scores remain a separate local suite.",
        "", "## Model metadata", "",
        markdown_table(
            ["Model", "Scale", "Architecture", "Tokenizer", "Stored parameters", "Layers", "Hidden", "Vocab"],
            ([r["model"], r["scale"], r["architecture"], r["tokenizer"],
              f"{r['stored_tensor_parameters']:,}", r["num_hidden_layers"],
              r["hidden_size"], r["vocab_size"]] for r in metadata),
        ), "", "## Suite-level descriptive averages", "",
        markdown_table(
            ["Model", "Official (16)", "Chinese zero-shot (3)",
             "Chinese fine-tune (4)", "CogBench (2)"],
            ([m.label, *(f"{mean[(m.label, suite)]:.4f}" for suite in SUITES)] for m in models),
        ), "",
        "These are unweighted descriptive averages within each suite; do not average across suites because their metrics and scales differ.",
        "", "## Selected mean paired contrasts", "",
        markdown_table(
            ["Suite", "Contrast", "Mean difference"],
            ([r["suite"], r["contrast"], f"{r['mean_difference']:+.4f}"]
             for r in contrasts if r["category"] in {"scale", "boundary_policy"}),
        ), "", "Positive values favor the condition before the minus sign.",
        "", "## Official task scores by cohort", "",
    ]
    official = [row for row in wide if row["suite"] == SUITES[0]]
    for cohort in cohorts:
        labels = [m.label for m in models if m.cohort == cohort]
        lines.extend([
            f"### {cohort}", "",
            markdown_table(
                ["Task", *labels, "Best in cohort"],
                ([r["task"], *(f"{float(r[label]):.4f}" for label in labels),
                  "; ".join(label for label in labels
                            if math.isclose(float(r[label]), max(float(r[x]) for x in labels),
                                            rel_tol=0, abs_tol=1e-12))]
                 for r in official),
            ), "",
        ])
    lines.extend([
        "## Generated files", "",
        "- `metrics_long.csv`: every parsed metric with scale and cohort metadata.",
        "- `primary_scores_wide.csv`: all 12 models side by side for every primary task.",
        "- `suite_averages.csv`: descriptive per-model means within each suite.",
        "- `pairwise_effects.csv`: task-level size, tokenizer, boundary, and architecture contrasts.",
        "- `contrast_summary.csv`: mean paired contrast by suite.",
        "- `cogbench_detail.csv`: subject/region-level CogBench results.",
        "- `model_metadata.csv`: configuration and exact stored tensor counts.",
        "- `hidden_tasks.csv`: archival prediction-only coverage for the original four models.",
        "", "## Interpretation cautions", "",
        "1. Do not combine accuracy, F1, MCC, and fMRI correlation into one score.",
        "2. Suite averages weight tasks equally, not examples equally.",
        "3. Aggregate files do not provide confidence intervals.",
        "4. One seed exists per condition, so differences are not seed-robust evidence.",
        "5. 30M/100M are experiment families; exact stored parameter counts appear above.", "",
    ])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    output = args.output.resolve()
    models = load_registry(args.registry.resolve(), root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = root / "results_analysis/raw_snapshot/SOURCE_MANIFEST.sha256"
    checked, errors = verify_manifest(manifest)
    if errors:
        raise RuntimeError("Snapshot integrity failure:\n" + "\n".join(errors))

    metrics: list[dict[str, Any]] = []
    cogbench: list[dict[str, Any]] = []
    for model in models:
        metrics.extend(load_official(model, root))
        model_metrics, model_cogbench = load_chinese(model, root)
        metrics.extend(model_metrics)
        cogbench.extend(model_cogbench)
    metadata = load_metadata(models, root)
    hidden = load_legacy_hidden(models, root)
    for row in metrics:
        if not math.isfinite(float(row["score"])):
            raise ValueError(f"Non-finite metric: {row}")
    wide = make_wide([r for r in metrics if r["primary_metric"]], models)
    effects = make_effects(wide, build_contrasts(models))
    averages = make_suite_averages(wide, models)
    contrast_summary = make_contrast_summary(effects)
    labels = [m.label for m in models]

    write_csv(output / "metrics_long.csv", metrics,
              ["model_id", "model", "architecture", "tokenizer", "scale", "cohort",
               "suite", "task", "subtask", "section", "metric", "score",
               "primary_metric", "status", "temperature", "n_result_files", "fast", "source_file"])
    write_csv(output / "primary_scores_wide.csv", wide,
              ["comparison_key", "suite", "task", "subtask", "metric", *labels,
               "best_score", "winner"])
    write_csv(output / "suite_averages.csv", averages,
              ["model_id", "model", "architecture", "tokenizer", "scale", "cohort",
               "suite", "task_count", "mean_score"])
    write_csv(output / "pairwise_effects.csv", effects,
              ["comparison_key", "suite", "task", "metric", "contrast", "category",
               "minuend", "subtrahend", "difference"])
    write_csv(output / "contrast_summary.csv", contrast_summary,
              ["suite", "contrast", "category", "minuend", "subtrahend",
               "task_count", "mean_difference"])
    write_csv(output / "cogbench_detail.csv", cogbench,
              ["model_id", "model", "architecture", "tokenizer", "scale", "cohort",
               "task", "region", "subject", "result_file", "score", "source_file"])
    write_csv(output / "model_metadata.csv", metadata,
              ["model_id", "model", "architecture", "tokenizer", "scale", "cohort",
               "stored_tensor_parameters", "tensor_count", "model_file_bytes", "dtype",
               "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
               "intermediate_size", "max_position_embeddings", "tie_word_embeddings",
               "source_file", "model_file"])
    write_csv(output / "hidden_tasks.csv", hidden,
              ["model_id", "model", "architecture", "tokenizer", "scale", "cohort",
               "task", "status", "leaf_count", "numeric_count", "all_numeric_finite",
               "source_file"])
    build_summary(output, models, metrics, wide, metadata, averages, contrast_summary, checked)
    print(f"Models compared: {len(models)}")
    print(f"Verified legacy source files: {checked}")
    print(f"Scored metric rows: {len(metrics)}")
    print(f"Primary comparison rows: {len(wide)}")
    print(f"Task-level contrasts: {len(effects)}")
    print(f"CogBench detail rows: {len(cogbench)}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
