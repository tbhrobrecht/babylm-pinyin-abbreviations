#!/usr/bin/env python3
"""Upload one or more completed models and their evaluations to Hugging Face.

The command is deliberately conservative:

* dry-run is the default;
* repositories are created one at a time;
* local files are never deleted;
* every repository receives raw evaluation outputs and compact processed CSVs;
* the remote ``model.safetensors`` SHA-256 is verified after upload.

Authenticate first with ``hf auth login``. See ``--help`` for examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results_analysis/models.json"
PROCESSED = ROOT / "results_analysis/processed"
DEFAULT_NAMESPACE = "timorobrecht"


@dataclass(frozen=True)
class UploadComponent:
    local_path: Path
    remote_path: str
    kind: str  # file or folder


@dataclass(frozen=True)
class ModelUpload:
    model_id: str
    label: str
    repo_id: str
    model_dir: Path
    components: tuple[UploadComponent, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", default=[],
        help="Registry model_id or label to upload; repeat for multiple models.",
    )
    parser.add_argument("--all", action="store_true", help="Upload all completed models serially.")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument(
        "--execute", action="store_true",
        help="Create public repositories and upload. Without this flag, print a dry-run plan.",
    )
    return parser.parse_args()


def read_registry() -> list[dict[str, object]]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]


def resolve_repo_name(model_id: str) -> str:
    return model_id.removeprefix("hf_")


def resolve_model_dir(model: dict[str, object]) -> Path:
    model_path = model.get("model_path")
    if model_path:
        path = ROOT / str(model_path)
    else:
        path = ROOT / str(model["model_id"])
    required = ("README.md", "config.json", "model.safetensors", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete Hugging Face export at {path}: missing {missing}")
    return path


def matching_directory(parent: Path, model_id: str) -> Path:
    matches = [path for path in parent.iterdir()
               if path.is_dir() and path.name.split("__")[-1] == model_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one evaluation directory for {model_id} in {parent}; found {len(matches)}")
    return matches[0]


def raw_evaluation_components(model: dict[str, object]) -> list[UploadComponent]:
    model_id = str(model["model_id"])
    official = model["official"]
    assert isinstance(official, dict)
    official_root = ROOT / str(official["path"])
    components: list[UploadComponent] = []
    if official["format"] == "submission":
        components.append(UploadComponent(
            official_root, "evaluation_results/official/submission.json", "file",
        ))
    elif official["format"] == "raw":
        components.extend((
            UploadComponent(
                matching_directory(official_root / "zeroshot", model_id),
                "evaluation_results/official/zeroshot", "folder",
            ),
            UploadComponent(
                official_root / "finetune" / model_id,
                "evaluation_results/official/finetune", "folder",
            ),
        ))
        meco_root = official_root / "meco"
        if meco_root.is_dir():
            components.append(UploadComponent(
                matching_directory(meco_root, model_id),
                "evaluation_results/official/meco", "folder",
            ))
    else:
        raise ValueError(f"Unknown official evaluation format for {model_id}: {official['format']}")

    chinese = ROOT / str(model["chinese_path"])
    if not chinese.is_dir():
        raise ValueError(f"Missing Chinese evaluation directory for {model_id}: {chinese}")
    components.append(UploadComponent(
        chinese, "evaluation_results/chinese_pipeline", "folder",
    ))
    for component in components:
        if not component.local_path.exists():
            raise ValueError(f"Missing upload component: {component.local_path}")
    return components


def filter_csv(source: Path, destination: Path, model_id: str) -> int:
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "model_id" not in (reader.fieldnames or []):
            raise ValueError(f"Expected model_id column in {source}")
        rows = [row for row in reader if row["model_id"] == model_id]
        fields = list(reader.fieldnames or [])
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def extract_primary_scores(destination: Path, label: str) -> int:
    source = PROCESSED / "primary_scores_wide.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if label not in (reader.fieldnames or []):
            raise ValueError(f"Model label {label!r} is absent from {source}")
        fields = ("comparison_key", "suite", "task", "subtask", "metric", "score")
        rows = [{**{field: row[field] for field in fields[:-1]}, "score": row[label]}
                for row in reader]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_processed_bundle(model: dict[str, object], parent: Path) -> Path:
    model_id = str(model["model_id"])
    label = str(model["label"])
    bundle = parent / model_id / "processed"
    bundle.mkdir(parents=True)
    counts = {
        "metrics_long.csv": filter_csv(PROCESSED / "metrics_long.csv", bundle / "metrics_long.csv", model_id),
        "suite_averages.csv": filter_csv(PROCESSED / "suite_averages.csv", bundle / "suite_averages.csv", model_id),
        "cogbench_detail.csv": filter_csv(PROCESSED / "cogbench_detail.csv", bundle / "cogbench_detail.csv", model_id),
        "model_metadata.csv": filter_csv(PROCESSED / "model_metadata.csv", bundle / "model_metadata.csv", model_id),
        "primary_scores.csv": extract_primary_scores(bundle / "primary_scores.csv", label),
    }
    manifest = {
        "model_id": model_id,
        "label": label,
        "architecture": model["architecture"],
        "tokenizer": model["tokenizer"],
        "scale": model["scale"],
        "cohort": model["cohort"],
        "row_counts": counts,
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (bundle.parent / "README.md").write_text(
        "# Evaluation results\n\n"
        "This directory contains the raw official BabyLM and Chinese-pipeline "
        "evaluation outputs for this model. `processed/` contains compact, "
        "model-specific tables generated by `results_analysis/build_results_summary.py`.\n\n"
        "Suite averages are descriptive and should not be combined across suites "
        "because their metrics differ. CogBench values are correlation-based.\n",
        encoding="utf-8",
    )
    return bundle.parent


def build_plan(model: dict[str, object], namespace: str, staging_root: Path) -> ModelUpload:
    model_id = str(model["model_id"])
    model_dir = resolve_model_dir(model)
    evaluation_bundle = make_processed_bundle(model, staging_root)
    components = [
        UploadComponent(model_dir, ".", "folder"),
        *raw_evaluation_components(model),
        UploadComponent(evaluation_bundle / "processed", "evaluation_results/processed", "folder"),
        UploadComponent(evaluation_bundle / "README.md", "evaluation_results/README.md", "file"),
    ]
    return ModelUpload(
        model_id=model_id,
        label=str(model["label"]),
        repo_id=f"{namespace}/{resolve_repo_name(model_id)}",
        model_dir=model_dir,
        components=tuple(components),
    )


def iter_files(component: UploadComponent) -> Iterable[tuple[Path, str]]:
    if component.kind == "file":
        yield component.local_path, component.remote_path
        return
    prefix = "" if component.remote_path == "." else component.remote_path.rstrip("/") + "/"
    for path in component.local_path.rglob("*"):
        if path.is_file() and ".cache" not in path.parts:
            yield path, prefix + path.relative_to(component.local_path).as_posix()


def print_plan(plan: ModelUpload) -> None:
    files = [item for component in plan.components for item in iter_files(component)]
    total = sum(path.stat().st_size for path, _ in files)
    print(f"{plan.model_id} -> https://huggingface.co/{plan.repo_id}")
    print(f"  files: {len(files)}")
    print(f"  bytes: {total}")
    print(f"  model weight sha256: {sha256(plan.model_dir / 'model.safetensors')}")
    for component in plan.components:
        print(f"  {component.kind}: {component.local_path} -> {component.remote_path}")


def upload(plan: ModelUpload) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    identity = api.whoami()
    if identity.get("name") != plan.repo_id.split("/", 1)[0]:
        raise RuntimeError(
            f"Authenticated as {identity.get('name')!r}, expected {plan.repo_id.split('/', 1)[0]!r}"
        )
    api.create_repo(plan.repo_id, repo_type="model", private=False, exist_ok=True)
    for component in plan.components:
        message = f"Upload {component.remote_path} for {plan.model_id}"
        if component.kind == "folder":
            api.upload_folder(
                repo_id=plan.repo_id,
                repo_type="model",
                folder_path=component.local_path,
                path_in_repo=None if component.remote_path == "." else component.remote_path,
                commit_message=message,
                ignore_patterns=[".cache/**", "**/.cache/**"],
            )
        else:
            api.upload_file(
                repo_id=plan.repo_id,
                repo_type="model",
                path_or_fileobj=component.local_path,
                path_in_repo=component.remote_path,
                commit_message=message,
            )

    expected = {remote for component in plan.components for _, remote in iter_files(component)}
    info = api.model_info(plan.repo_id, files_metadata=True)
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    missing = sorted(expected - set(siblings))
    if missing:
        raise RuntimeError(f"Remote verification failed; missing files: {missing[:10]}")
    remote_weight = siblings["model.safetensors"]
    remote_sha = remote_weight.lfs.sha256 if remote_weight.lfs else None
    local_sha = sha256(plan.model_dir / "model.safetensors")
    if remote_sha != local_sha:
        raise RuntimeError(f"Weight checksum mismatch: local={local_sha}, remote={remote_sha}")
    print(f"VERIFIED {plan.repo_id} ({len(expected)} files, weight sha256={local_sha})")


def select_models(models: list[dict[str, object]], requested: list[str], all_models: bool) -> list[dict[str, object]]:
    if all_models:
        if requested:
            raise ValueError("Use either --all or --model, not both")
        return models
    if not requested:
        raise ValueError("Specify at least one --model, or use --all")
    lookup = {str(model[key]): model for model in models for key in ("model_id", "label")}
    unknown = [value for value in requested if value not in lookup]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    selected = [lookup[value] for value in requested]
    if len({str(model["model_id"]) for model in selected}) != len(selected):
        raise ValueError("The same model was selected more than once")
    return selected


def main() -> None:
    args = parse_args()
    models = select_models(read_registry(), args.model, args.all)
    with tempfile.TemporaryDirectory(prefix="babylm-hf-upload-", dir="/tmp") as staging:
        plans = [build_plan(model, args.namespace, Path(staging)) for model in models]
        for plan in plans:
            print_plan(plan)
            if args.execute:
                upload(plan)
            else:
                print("  DRY RUN: add --execute to upload")


if __name__ == "__main__":
    main()
