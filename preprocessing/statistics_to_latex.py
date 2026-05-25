"""Render pinyin tone/length summary statistics as a publication-ready LaTeX table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TONE_LABELS = {
    "1": "Tone 1",
    "2": "Tone 2",
    "3": "Tone 3",
    "4": "Tone 4",
    "blank": "Neutral/unmarked",
}
TONE_ORDER = ("1", "2", "3", "4", "blank")


def load_stats(path: Path) -> dict[str, Any]:
    """Load the JSON summary payload written by ``tone_statistics.py``."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid statistics JSON in {path}: {exc}") from exc


def latex_escape(text: str) -> str:
    """Escape free text inserted into LaTeX prose fields."""
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
    return "".join(replacements.get(char, char) for char in text)


def validate_stats(stats: dict[str, Any]) -> int:
    """Validate required summary fields and return the number of unique cases."""
    try:
        total = int(stats["total_unique_cases"])
        tones = stats["tones"]
        lengths = stats["lengths"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Statistics payload is missing required summary fields.") from exc

    missing_tones = [tone for tone in TONE_ORDER if tone not in tones]
    if missing_tones:
        raise ValueError(f"Missing tone categories: {', '.join(missing_tones)}")

    tone_total = sum(int(tones[tone]["count"]) for tone in TONE_ORDER)
    length_total = sum(int(item["count"]) for item in lengths)
    if tone_total != total or length_total != total:
        raise ValueError(
            "Statistics totals do not agree: "
            f"reported={total}, tones={tone_total}, lengths={length_total}."
        )
    return total


def format_row(
    tone_label: str = "",
    tone_count: str = "",
    tone_percent: str = "",
    length_label: str = "",
    length_count: str = "",
    length_percent: str = "",
) -> str:
    return (
        f"{tone_label} & {tone_count} & {tone_percent} & "
        f"{length_label} & {length_count} & {length_percent} \\\\"
    )


def render_table(
    stats: dict[str, Any],
    caption: str,
    label: str,
) -> str:
    """Return a standalone LaTeX table environment for summary statistics."""
    total = validate_stats(stats)
    count_format = f"{len(str(total))}.0"
    tone_rows = [
        (
            TONE_LABELS[tone],
            str(int(stats["tones"][tone]["count"])),
            f'{float(stats["tones"][tone]["percentage"]):.2f}',
        )
        for tone in TONE_ORDER
    ]
    length_rows = [
        (
            str(int(item["length"])),
            str(int(item["count"])),
            f'{float(item["percentage"]):.2f}',
        )
        for item in sorted(stats["lengths"], key=lambda item: int(item["length"]))
    ]

    rows: list[str] = []
    data_row_count = max(len(tone_rows), len(length_rows))
    for index in range(data_row_count):
        tone = tone_rows[index] if index < len(tone_rows) else ("", "", "")
        length = length_rows[index] if index < len(length_rows) else ("", "", "")
        rows.append(format_row(*tone, *length))
    rows.append(r"\addlinespace")
    rows.append(format_row("Total", str(total), "100.00", "Total", str(total), "100.00"))

    body = "\n".join(rows)
    return f"""% Requires \\usepackage{{booktabs}}
% Requires \\usepackage{{siunitx}}
\\begin{{table}}[t]
\\centering
\\caption{{{latex_escape(caption)}}}
\\label{{{label}}}
\\begin{{tabular}}{{
    l
    S[table-format={count_format}]
    S[table-format=3.2]
    @{{\\hspace{{1.75em}}}}
    l
    S[table-format={count_format}]
    S[table-format=3.2]
}}
\\toprule
\\multicolumn{{3}}{{c}}{{Tone category}} &
\\multicolumn{{3}}{{c}}{{Syllable length (letters)}} \\\\
\\cmidrule(lr){{1-3}} \\cmidrule(lr){{4-6}}
{{Category}} & {{Count}} & {{Percent (\\%)}} &
{{Length}} & {{Count}} & {{Percent (\\%)}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\vspace{{2pt}}

\\begin{{minipage}}{{0.95\\linewidth}}
\\footnotesize\\emph{{Note.}} Percentages are calculated over {total:,} unique
character--contextual-pinyin cases; syllable length excludes the final tone
digit. Neutral/unmarked denotes readings without an explicit tone digit.
\\end{{minipage}}
\\end{{table}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pinyin summary statistics JSON to a LaTeX table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/10k_statistics.jsonl"),
        help="Summary statistics JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/10k_statistics_table.tex"),
        help="Path for the generated LaTeX table.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Distribution of tone categories and syllable lengths among unique "
            "contextual pinyin cases in the 10k BabyLM-Zho sample."
        ),
    )
    parser.add_argument("--label", default="tab:10k-pinyin-statistics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = render_table(load_stats(args.input), args.caption, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8", newline="\n")
    print(f"Wrote LaTeX table to {args.output}")


if __name__ == "__main__":
    main()
