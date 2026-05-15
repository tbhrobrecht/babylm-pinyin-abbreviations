"""Build a bucketed tone/length probability matrix from tone statistics JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TONE_BUCKETS = {
    "1": "tone_1",
    "2": "tone_2",
    "4": "tone_4",
    "3": "tone_3_or_5",
    "5": "tone_3_or_5",
    "blank": "tone_3_or_5",
}
TONE_ORDER = ("tone_1", "tone_2", "tone_4", "tone_3_or_5")

LENGTH_ORDER = ("length_1_or_2", "length_3", "length_4_to_6")


def load_stats(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def length_bucket(length: int) -> str:
    if length in {1, 2}:
        return "length_1_or_2"
    if length == 3:
        return "length_3"
    if length in {4, 5, 6}:
        return "length_4_to_6"
    raise ValueError(f"Unsupported pinyin length: {length}")


def bucketed_marginals(stats: dict[str, Any]) -> tuple[Counter[str], Counter[str], int]:
    tone_counts: Counter[str] = Counter()
    for tone, values in stats.get("tones", {}).items():
        bucket = TONE_BUCKETS.get(str(tone))
        if bucket is None:
            raise ValueError(f"Unsupported tone bucket: {tone}")
        tone_counts[bucket] += int(values["count"])

    length_counts: Counter[str] = Counter()
    for values in stats.get("lengths", []):
        bucket = length_bucket(int(values["length"]))
        length_counts[bucket] += int(values["count"])

    total = int(stats.get("total_unique_cases") or sum(tone_counts.values()))
    return tone_counts, length_counts, total


def exact_matrix_from_cases(stats: dict[str, Any]) -> tuple[dict[str, dict[str, int]], int]:
    matrix = {tone: {length: 0 for length in LENGTH_ORDER} for tone in TONE_ORDER}
    for case in stats["unique_cases"]:
        tone_bucket = TONE_BUCKETS.get(str(case["tone"]))
        if tone_bucket is None:
            raise ValueError(f"Unsupported tone bucket: {case['tone']}")
        len_bucket = length_bucket(int(case["length"]))
        matrix[tone_bucket][len_bucket] += 1
    total = sum(sum(row.values()) for row in matrix.values())
    return matrix, total


def estimated_matrix_from_marginals(
    tone_counts: Counter[str],
    length_counts: Counter[str],
    total: int,
) -> dict[str, dict[str, float]]:
    if total == 0:
        return {tone: {length: 0.0 for length in LENGTH_ORDER} for tone in TONE_ORDER}

    return {
        tone: {
            length: tone_counts[tone] * length_counts[length] / total
            for length in LENGTH_ORDER
        }
        for tone in TONE_ORDER
    }


def matrix_payload(stats: dict[str, Any]) -> dict[str, Any]:
    if "unique_cases" in stats:
        counts, total = exact_matrix_from_cases(stats)
        method = "exact_from_unique_cases"
    else:
        tone_counts, length_counts, total = bucketed_marginals(stats)
        counts = estimated_matrix_from_marginals(tone_counts, length_counts, total)
        method = "estimated_from_marginals_assuming_independence"

    probabilities = {
        tone: {
            length: round((counts[tone][length] / total) if total else 0.0, 6)
            for length in LENGTH_ORDER
        }
        for tone in TONE_ORDER
    }

    percentages = {
        tone: {
            length: round(probabilities[tone][length] * 100, 4)
            for length in LENGTH_ORDER
        }
        for tone in TONE_ORDER
    }

    return {
        "method": method,
        "total_unique_cases": total,
        "tone_buckets": list(TONE_ORDER),
        "length_buckets": list(LENGTH_ORDER),
        "counts": counts,
        "probabilities": probabilities,
        "percentages": percentages,
    }


def print_cli_matrix(payload: dict[str, Any]) -> None:
    print(f"method: {payload['method']}")
    print(f"total_unique_cases: {payload['total_unique_cases']}")
    print()

    headers = ["tone \\ length", *LENGTH_ORDER]
    rows = []
    for tone in TONE_ORDER:
        row = [tone]
        for length in LENGTH_ORDER:
            row.append(f"{payload['percentages'][tone][length]:.4f}%")
        rows.append(row)

    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a bucketed tone/length probability matrix."
    )
    parser.add_argument("--input", type=Path, default=Path("data/10k_statistics.jsonl"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = matrix_payload(load_stats(args.input))
    print_cli_matrix(payload)

    if args.output:
        write_output(args.output, payload)
        print(f"\nWrote matrix JSON to {args.output}")


if __name__ == "__main__":
    main()
