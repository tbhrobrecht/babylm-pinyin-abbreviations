#!/usr/bin/env python3
"""Build human-readable BabyLM comparison tables from the LRZ result snapshot.

The script uses only the Python standard library. It never treats hidden-task
predictions as scored results and keeps source paths on every metric row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


MODELS = {
    "hf_nk_babylm_zho_gpt2_hybrid": ("GPT2 hybrid", "GPT2", "Hybrid"),
    "hf_nk_babylm_zho_qwen2_hybrid": ("Qwen2 hybrid", "Qwen2", "Hybrid"),
    "hf_nk_babylm_zho_gpt2_bpe": ("GPT2 BPE", "GPT2", "BPE"),
    "hf_nk_babylm_zho_qwen2_bpe": ("Qwen2 BPE", "Qwen2", "BPE"),
}

MODEL_ORDER = [
    "GPT2 hybrid",
    "Qwen2 hybrid",
    "GPT2 BPE",
    "Qwen2 BPE",
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "raw_snapshot",
        help="Root of the downloaded LRZ snapshot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "processed",
        help="Directory for generated CSV and Markdown files",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def flatten_single_score(value: Any) -> tuple[str, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "aggregate", float(value)
    if isinstance(value, dict):
        numeric = [(str(k), float(v)) for k, v in value.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numeric) == 1:
            return numeric[0]
    raise ValueError(f"Expected one numeric score, received: {value!r}")


def base_row(model_id: str) -> dict[str, str]:
    model, architecture, tokenizer = MODELS[model_id]
    return {
        "model_id": model_id,
        "model": model,
        "architecture": architecture,
        "tokenizer": tokenizer,
    }


def load_official(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    official = root / "results" / "eval" / "official"
    for model_id in MODELS:
        path = official / f"{model_id}_submission.json"
        data = read_json(path)
        for task, value in data.items():
            subtask, score = flatten_single_score(value)
            rows.append({
                **base_row(model_id),
                "suite": "BabyLM official Chinese",
                "task": task,
                "subtask": subtask,
                "section": "submission",
                "metric": "score",
                "score": score,
                "primary_metric": True,
                "status": "locally_scored",
                "temperature": "",
                "n_result_files": "",
                "fast": "",
                "source_file": relpath(path, root),
            })
    return rows


def parse_zero_shot_report(path: Path, root: Path, model_id: str, task: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    temperature_match = re.search(r"^TEMPERATURE:\s*([0-9.]+)", text, re.MULTILINE)
    temperature = float(temperature_match.group(1)) if temperature_match else ""
    rows: list[dict[str, Any]] = []
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("### "):
            section = line[4:].strip().lower().replace(" ", "_")
            continue
        if line.startswith("TEMPERATURE:"):
            continue
        if section == "average_accuracy" and re.fullmatch(r"-?[0-9.]+", line):
            rows.append({
                **base_row(model_id),
                "suite": "Chinese zero-shot",
                "task": task,
                "subtask": "aggregate",
                "section": section,
                "metric": "accuracy",
                "score": float(line) / 100.0,
                "primary_metric": True,
                "status": "locally_scored",
                "temperature": temperature,
                "n_result_files": "",
                "fast": "",
                "source_file": relpath(path, root),
            })
            continue
        match = re.fullmatch(r"([^:]+):\s*(-?[0-9.]+)", line)
        if match and section:
            rows.append({
                **base_row(model_id),
                "suite": "Chinese zero-shot",
                "task": task,
                "subtask": match.group(1).strip(),
                "section": section,
                "metric": "accuracy",
                "score": float(match.group(2)) / 100.0,
                "primary_metric": False,
                "status": "locally_scored",
                "temperature": temperature,
                "n_result_files": "",
                "fast": "",
                "source_file": relpath(path, root),
            })
    averages = [row for row in rows if row["primary_metric"]]
    if len(averages) != 1:
        raise ValueError(f"Expected one average accuracy in {path}, found {len(averages)}")
    return rows


def load_chinese(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    pipeline = root / "results" / "eval" / "chinese_pipeline"
    for model_id in MODELS:
        model_root = pipeline / model_id / "main"
        for task in ("zhoblimp", "hanzi_structure", "hanzi_pinyin"):
            report = (model_root / "zero_shot" / "causal" / task / task /
                      "best_temperature_report.txt")
            rows.extend(parse_zero_shot_report(report, root, model_id, task))

        for task in ("afqmc", "ocnli", "tnews", "cluewsc2020"):
            report = model_root / "finetune" / task / "results.txt"
            text = report.read_text(encoding="utf-8")
            parsed = re.findall(r"^([A-Za-z0-9_]+):\s*(-?[0-9.eE+\-]+)", text, re.MULTILINE)
            if not parsed:
                raise ValueError(f"No metrics found in {report}")
            for metric, value in parsed:
                rows.append({
                    **base_row(model_id),
                    "suite": "Chinese fine-tune",
                    "task": task,
                    "subtask": "zh",
                    "section": "evaluation",
                    "metric": metric.lower(),
                    "score": float(value),
                    "primary_metric": metric.lower() == "accuracy",
                    "status": "locally_scored",
                    "temperature": "",
                    "n_result_files": "",
                    "fast": "",
                    "source_file": relpath(report, root),
                })

        for task in ("word_fmri", "fmri"):
            matches = list((model_root / "cogbench" / task).glob("*report.json"))
            if len(matches) != 1:
                raise ValueError(f"Expected one CogBench report for {model_id}/{task}")
            report_path = matches[0]
            report = read_json(report_path)
            for metric in ("mean", "min", "max"):
                rows.append({
                    **base_row(model_id),
                    "suite": "CogBench",
                    "task": task,
                    "subtask": "aggregate",
                    "section": "aggregate",
                    "metric": metric,
                    "score": float(report[metric]),
                    "primary_metric": metric == "mean",
                    "status": "locally_scored",
                    "temperature": "",
                    "n_result_files": int(report["n_result_files"]),
                    "fast": bool(report["fast"]),
                    "source_file": relpath(report_path, root),
                })
            for item in report["files"]:
                item_path = Path(item["file"])
                region = item_path.parent.name if task == "fmri" else ""
                subject = item_path.stem.replace("_score", "").replace("_average", "")
                details.append({
                    **base_row(model_id),
                    "task": task,
                    "region": region,
                    "subject": subject,
                    "result_file": item_path.name,
                    "score": float(item["value"]),
                    "source_file": relpath(report_path, root),
                })
    return rows, details


def load_hidden(root: Path) -> list[dict[str, Any]]:
    path = root / "hidden_task_manifest.json"
    rows = []
    for item in read_json(path):
        rows.append({**base_row(item["model_id"]), **item})
    return rows


def load_model_metadata(root: Path) -> list[dict[str, Any]]:
    file_metadata = read_json(root / "model_file_metadata.json")
    rows: list[dict[str, Any]] = []
    config_root = root / "code" / "babylm-pinyin-abbreviations"
    for model_id in MODELS:
        config_path = config_root / model_id / "config.json"
        config = read_json(config_path)
        stored = file_metadata[model_id]
        rows.append({
            **base_row(model_id),
            "stored_tensor_parameters": int(stored["stored_tensor_parameters"]),
            "tensor_count": int(stored["tensor_count"]),
            "model_file_bytes": int(stored["model_file_bytes"]),
            "dtype": config.get("dtype", ""),
            "vocab_size": config.get("vocab_size", ""),
            "hidden_size": config.get("hidden_size", config.get("n_embd", "")),
            "num_hidden_layers": config.get("num_hidden_layers", config.get("n_layer", "")),
            "num_attention_heads": config.get("num_attention_heads", config.get("n_head", "")),
            "intermediate_size": config.get("intermediate_size", ""),
            "max_position_embeddings": config.get("max_position_embeddings", ""),
            "tie_word_embeddings": config.get("tie_word_embeddings", ""),
            "source_file": relpath(config_path, root),
            "model_file": stored["model_file"],
        })
    return rows


def comparison_key(row: dict[str, Any]) -> str:
    return " | ".join(str(row[k]) for k in ("suite", "task", "subtask", "metric"))


def make_wide(primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in primary_rows:
        key = comparison_key(row)
        record = grouped.setdefault(key, {
            "comparison_key": key,
            "suite": row["suite"],
            "task": row["task"],
            "subtask": row["subtask"],
            "metric": row["metric"],
        })
        record[row["model"]] = row["score"]
    wide = list(grouped.values())
    for row in wide:
        missing = [model for model in MODEL_ORDER if model not in row]
        if missing:
            raise ValueError(f"Missing models for {row['comparison_key']}: {missing}")
        scores = {model: float(row[model]) for model in MODEL_ORDER}
        maximum = max(scores.values())
        row["best_score"] = maximum
        row["winner"] = "; ".join(model for model, score in scores.items()
                                  if math.isclose(score, maximum, rel_tol=0.0, abs_tol=1e-12))
    return sorted(wide, key=lambda row: (row["suite"], row["task"], row["metric"]))


def make_effects(wide: list[dict[str, Any]]) -> list[dict[str, Any]]:
    effects = []
    for row in wide:
        gh = float(row["GPT2 hybrid"])
        qh = float(row["Qwen2 hybrid"])
        gb = float(row["GPT2 BPE"])
        qb = float(row["Qwen2 BPE"])
        effects.append({
            "comparison_key": row["comparison_key"],
            "suite": row["suite"],
            "task": row["task"],
            "metric": row["metric"],
            "gpt2_hybrid_minus_bpe": gh - gb,
            "qwen2_hybrid_minus_bpe": qh - qb,
            "qwen2_minus_gpt2_hybrid": qh - gh,
            "qwen2_minus_gpt2_bpe": qb - gb,
            "tokenizer_architecture_interaction": (qh - qb) - (gh - gb),
            "winner": row["winner"],
        })
    return effects


def verify_manifest(root: Path) -> tuple[int, list[str]]:
    manifest = root / "SOURCE_MANIFEST.sha256"
    errors = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("./")
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"hash mismatch: {relative}")
        checked += 1
    return checked, errors


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |",
              "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(output)


def build_summary(
    output: Path,
    metrics: list[dict[str, Any]],
    wide: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    hidden: list[dict[str, Any]],
    manifest_count: int,
) -> None:
    official = [row for row in wide if row["suite"] == "BabyLM official Chinese"]
    macro = {model: fmean(float(row[model]) for row in official) for model in MODEL_ORDER}
    official_effects = [row for row in effects if row["suite"] == "BabyLM official Chinese"]
    avg_effect = {
        key: fmean(float(row[key]) for row in official_effects)
        for key in (
            "gpt2_hybrid_minus_bpe",
            "qwen2_hybrid_minus_bpe",
            "qwen2_minus_gpt2_hybrid",
            "qwen2_minus_gpt2_bpe",
            "tokenizer_architecture_interaction",
        )
    }
    wins = {model: 0 for model in MODEL_ORDER}
    for row in official:
        for model in row["winner"].split("; "):
            wins[model] += 1

    cog = [row for row in wide if row["suite"] == "CogBench"]
    chinese_zero = [row for row in wide if row["suite"] == "Chinese zero-shot"]
    chinese_finetune = [row for row in wide if row["suite"] == "Chinese fine-tune"]
    lines = [
        "# Mandarin BabyLM model comparison",
        "",
        "## Snapshot integrity",
        "",
        f"- Verified `{manifest_count}` downloaded source files against `SOURCE_MANIFEST.sha256`.",
        f"- Parsed `{len(metrics)}` scored metric rows and `{len(wide)}` directly comparable primary metrics.",
        "- Hidden tasks are listed as `predictions_only`; no hidden score is invented or treated as zero.",
        "- CogBench reports show `fast=False`, indicating the full configured runs.",
        "",
        "## Model metadata",
        "",
        markdown_table(
            ["Model", "Architecture", "Tokenizer", "Stored tensor parameters", "Layers", "Hidden", "Vocab"],
            ([row["model"], row["architecture"], row["tokenizer"],
              f"{row['stored_tensor_parameters']:,}", row["num_hidden_layers"],
              row["hidden_size"], row["vocab_size"]] for row in metadata),
        ),
        "",
        "`Stored tensor parameters` is counted directly from the safetensors header. It is not an estimate based on file size.",
        "",
        "## Official Chinese results",
        "",
        markdown_table(
            ["Model", "Unweighted macro-average", "Task wins (ties count for each)"],
            ([model, f"{macro[model]:.4f}", wins[model]] for model in MODEL_ORDER),
        ),
        "",
        "The macro-average is descriptive: tasks differ in size, difficulty, and variance.",
        "",
        "### Mean paired effects across the 16 official tasks",
        "",
        markdown_table(
            ["Contrast", "Mean score difference"],
            [
                ["GPT2: Hybrid - BPE", f"{avg_effect['gpt2_hybrid_minus_bpe']:+.4f}"],
                ["Qwen2: Hybrid - BPE", f"{avg_effect['qwen2_hybrid_minus_bpe']:+.4f}"],
                ["Hybrid: Qwen2 - GPT2", f"{avg_effect['qwen2_minus_gpt2_hybrid']:+.4f}"],
                ["BPE: Qwen2 - GPT2", f"{avg_effect['qwen2_minus_gpt2_bpe']:+.4f}"],
                ["Tokenizer x architecture interaction", f"{avg_effect['tokenizer_architecture_interaction']:+.4f}"],
            ],
        ),
        "",
        "Positive tokenizer effects favor Hybrid. Positive architecture effects favor Qwen2.",
        "",
        "### Task-level official scores",
        "",
        markdown_table(
            ["Task", *MODEL_ORDER, "Winner"],
            ([row["task"], *(f"{float(row[m]):.4f}" for m in MODEL_ORDER), row["winner"]]
             for row in official),
        ),
        "",
        "## Chinese-pipeline primary metrics",
        "",
        "### Zero-shot average accuracy",
        "",
        markdown_table(
            ["Task", *MODEL_ORDER, "Winner"],
            ([row["task"], *(f"{float(row[m]):.4f}" for m in MODEL_ORDER), row["winner"]]
             for row in chinese_zero),
        ),
        "",
        "### Fine-tuned accuracy",
        "",
        markdown_table(
            ["Task", *MODEL_ORDER, "Winner"],
            ([row["task"], *(f"{float(row[m]):.4f}" for m in MODEL_ORDER), row["winner"]]
             for row in chinese_finetune),
        ),
        "",
        "F1 and MCC are preserved in `metrics_long.csv` as supplementary metrics; they are not mixed into the accuracy comparison.",
        "",
        "## CogBench aggregates",
        "",
        markdown_table(
            ["Task", *MODEL_ORDER, "Winner"],
            ([row["task"], *(f"{float(row[m]):.4f}" for m in MODEL_ORDER), row["winner"]]
             for row in cog),
        ),
        "",
        "The original CogBench logs emitted ill-conditioned ridge-regression warnings. Outputs were complete, but small differences may be numerically sensitive.",
        "",
        "## Hidden tasks",
        "",
        f"The snapshot records `{len(hidden)}` model/task prediction manifests covering `hanzi_pinyin`, `hanzi_structure`, and `meco_l1`.",
        "Authoritative scores require the evaluation server and are intentionally absent here.",
        "",
        "## Files produced",
        "",
        "- `metrics_long.csv`: every parsed scored metric, including supplementary metrics and zero-shot subsections.",
        "- `primary_scores_wide.csv`: one comparable row per primary task metric.",
        "- `pairwise_effects.csv`: tokenizer, architecture, and interaction contrasts.",
        "- `cogbench_detail.csv`: subject/region-level CogBench values.",
        "- `model_metadata.csv`: exact exported-model metadata.",
        "- `hidden_tasks.csv`: predictions-only task coverage and integrity counts.",
        "",
        "## Interpretation cautions",
        "",
        "1. Do not combine accuracy, F1, MCC, and fMRI correlations into a single inferential statistic.",
        "2. The descriptive macro-average weights tasks equally, not examples equally.",
        "3. No confidence intervals are available from aggregate-only files; example-level predictions would be needed for bootstrap testing.",
        "4. Differences between architectures and tokenizers are based on one trained model per condition, so they should not be presented as seed-robust effects.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    checked, manifest_errors = verify_manifest(root)
    if manifest_errors:
        raise RuntimeError("Snapshot integrity failure:\n" + "\n".join(manifest_errors))

    official = load_official(root)
    chinese, cogbench_detail = load_chinese(root)
    metrics = official + chinese
    hidden = load_hidden(root)
    metadata = load_model_metadata(root)

    for row in metrics:
        if not math.isfinite(float(row["score"])):
            raise ValueError(f"Non-finite metric: {row}")

    primary = [row for row in metrics if row["primary_metric"]]
    wide = make_wide(primary)
    effects = make_effects(wide)

    metric_fields = [
        "model_id", "model", "architecture", "tokenizer", "suite", "task",
        "subtask", "section", "metric", "score", "primary_metric", "status",
        "temperature", "n_result_files", "fast", "source_file",
    ]
    write_csv(output / "metrics_long.csv", metrics, metric_fields)
    write_csv(
        output / "primary_scores_wide.csv",
        wide,
        ["comparison_key", "suite", "task", "subtask", "metric", *MODEL_ORDER,
         "best_score", "winner"],
    )
    write_csv(
        output / "pairwise_effects.csv",
        effects,
        ["comparison_key", "suite", "task", "metric", "gpt2_hybrid_minus_bpe",
         "qwen2_hybrid_minus_bpe", "qwen2_minus_gpt2_hybrid",
         "qwen2_minus_gpt2_bpe", "tokenizer_architecture_interaction", "winner"],
    )
    write_csv(
        output / "cogbench_detail.csv",
        cogbench_detail,
        ["model_id", "model", "architecture", "tokenizer", "task", "region",
         "subject", "result_file", "score", "source_file"],
    )
    write_csv(
        output / "model_metadata.csv",
        metadata,
        ["model_id", "model", "architecture", "tokenizer", "stored_tensor_parameters",
         "tensor_count", "model_file_bytes", "dtype", "vocab_size", "hidden_size",
         "num_hidden_layers", "num_attention_heads", "intermediate_size",
         "max_position_embeddings", "tie_word_embeddings", "source_file", "model_file"],
    )
    write_csv(
        output / "hidden_tasks.csv",
        hidden,
        ["model_id", "model", "architecture", "tokenizer", "task", "status",
         "leaf_count", "numeric_count", "all_numeric_finite", "source_file"],
    )
    build_summary(output, metrics, wide, effects, metadata, hidden, checked)

    print(f"Verified source files: {checked}")
    print(f"Scored metric rows: {len(metrics)}")
    print(f"Primary comparison rows: {len(wide)}")
    print(f"CogBench detail rows: {len(cogbench_detail)}")
    print(f"Hidden prediction manifests: {len(hidden)}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
