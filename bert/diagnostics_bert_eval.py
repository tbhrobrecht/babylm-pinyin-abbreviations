"""Diagnostics for comparing causal and MLM minimal-pair scoring."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_REPO = PROJECT_ROOT.parent / "chinese-babylm-eval-pipeline-1"
if str(EVAL_REPO) not in sys.path:
    sys.path.insert(0, str(EVAL_REPO))

from evaluation_pipeline.sentence_zero_shot.read_files import decode  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_items(data_path: Path, task: str, limit: int) -> list[dict]:
    items = []
    for path in sorted(data_path.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    items.append(decode(line, path, task, full_sentence_scores=False, images=None))
                    if len(items) >= limit:
                        return items
    return items


def content_positions(input_ids: list[int], tokenizer) -> list[int]:
    special_ids = set(tokenizer.all_special_ids)
    return [index for index, token_id in enumerate(input_ids) if token_id not in special_ids]


@torch.no_grad()
def mlm_scores(model, tokenizer, sentence: str) -> tuple[float, float, int]:
    encoded = tokenizer(sentence, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    positions = content_positions(input_ids, tokenizer)
    if not positions:
        return 0.0, 0.0, 0

    batch = []
    targets = []
    for position in positions:
        masked = list(input_ids)
        targets.append(masked[position])
        masked[position] = tokenizer.mask_token_id
        batch.append(torch.tensor(masked, dtype=torch.long))

    tokens = torch.nn.utils.rnn.pad_sequence(
        batch,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(DEVICE)
    attention_mask = tokens.ne(tokenizer.pad_token_id).long()
    logits = model(input_ids=tokens, attention_mask=attention_mask).logits
    row_ids = torch.arange(len(positions), device=DEVICE)
    col_ids = torch.tensor(positions, device=DEVICE)
    target_ids = torch.tensor(targets, device=DEVICE)
    token_log_probs = F.log_softmax(logits[row_ids, col_ids], dim=-1)
    selected = token_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    summed = float(selected.sum().item())
    return summed, summed / len(positions), len(positions)


@torch.no_grad()
def causal_scores(model, tokenizer, sentence: str) -> tuple[float, float, int]:
    encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(DEVICE)
    if input_ids.shape[1] < 2:
        return 0.0, 0.0, 0
    attention_mask = encoded["attention_mask"].to(DEVICE)
    logits = model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1]).logits
    targets = input_ids[:, 1:]
    token_log_probs = F.log_softmax(logits, dim=-1)
    selected = token_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    summed = float(selected.sum().item())
    count = int(targets.numel())
    return summed, summed / count, count


def evaluate_pairwise(items: list[dict], scorer) -> dict:
    correct_sum = 0
    correct_mean = 0
    length_ties = 0
    length_bad_longer = 0
    margins_sum = []
    margins_mean = []
    by_uid = Counter()
    by_uid_correct_sum = Counter()
    by_uid_correct_mean = Counter()

    for item in items:
        sentences = item["sentences"]
        label = int(item["label"])
        scores = [scorer(sentence) for sentence in sentences]
        sums = [score[0] for score in scores]
        means = [score[1] for score in scores]
        lengths = [score[2] for score in scores]
        pred_sum = max(range(len(sums)), key=lambda index: sums[index])
        pred_mean = max(range(len(means)), key=lambda index: means[index])
        other = 1 - label
        correct_sum += int(pred_sum == label)
        correct_mean += int(pred_mean == label)
        length_ties += int(lengths[label] == lengths[other])
        length_bad_longer += int(lengths[other] > lengths[label])
        margins_sum.append(sums[label] - sums[other])
        margins_mean.append(means[label] - means[other])
        uid = item["UID"]
        by_uid[uid] += 1
        by_uid_correct_sum[uid] += int(pred_sum == label)
        by_uid_correct_mean[uid] += int(pred_mean == label)

    n = max(1, len(items))
    worst = []
    for uid, total in by_uid.items():
        worst.append(
            (
                by_uid_correct_sum[uid] / total,
                by_uid_correct_mean[uid] / total,
                total,
                uid,
            )
        )
    worst.sort()
    return {
        "n": len(items),
        "sum_accuracy": correct_sum / n,
        "mean_accuracy": correct_mean / n,
        "length_tie_rate": length_ties / n,
        "bad_longer_rate": length_bad_longer / n,
        "avg_sum_margin": sum(margins_sum) / n,
        "avg_mean_margin": sum(margins_mean) / n,
        "worst_uids": worst[:10],
    }


def print_report(name: str, result: dict) -> None:
    print(f"\n{name}")
    print(f"items={result['n']}")
    print(f"sum_accuracy={result['sum_accuracy']:.3f}")
    print(f"mean_accuracy={result['mean_accuracy']:.3f}")
    print(f"length_tie_rate={result['length_tie_rate']:.3f}")
    print(f"bad_candidate_longer_rate={result['bad_longer_rate']:.3f}")
    print(f"avg_sum_margin={result['avg_sum_margin']:.3f}")
    print(f"avg_mean_margin={result['avg_mean_margin']:.3f}")
    print("worst_uids=sum_acc mean_acc n uid")
    for sum_acc, mean_acc, total, uid in result["worst_uids"]:
        print(f"{sum_acc:.2f} {mean_acc:.2f} {total} {uid}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--task", default="zhoblimp")
    parser.add_argument("--bert-model", type=Path, default=Path("hf_full_chinese_bert"))
    parser.add_argument("--gpt-model", type=Path, default=Path("hf_models/chinese_outdated/hf_full_chinese_gpu2"))
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()

    items = read_items(args.data_path, args.task, args.limit)
    print(f"Loaded {len(items)} items from {args.data_path}")

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)
    bert_model = AutoModelForMaskedLM.from_pretrained(args.bert_model, trust_remote_code=True).to(DEVICE)
    bert_model.eval()
    print_report(
        "BERT MLM pseudo-likelihood",
        evaluate_pairwise(items, lambda sentence: mlm_scores(bert_model, bert_tokenizer, sentence)),
    )

    if args.gpt_model.exists():
        gpt_tokenizer = AutoTokenizer.from_pretrained(args.gpt_model, trust_remote_code=True)
        gpt_model = AutoModelForCausalLM.from_pretrained(args.gpt_model, trust_remote_code=True).to(DEVICE)
        gpt_model.eval()
        print_report(
            "GPT causal likelihood",
            evaluate_pairwise(items, lambda sentence: causal_scores(gpt_model, gpt_tokenizer, sentence)),
        )


if __name__ == "__main__":
    main()
