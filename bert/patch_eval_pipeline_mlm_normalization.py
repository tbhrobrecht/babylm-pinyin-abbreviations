"""Patch the local Chinese BabyLM eval pipeline with MLM score normalization."""

from __future__ import annotations

from pathlib import Path


EVAL_REPO = Path(r"C:\Users\Timo\VS Code Projects\chinese-babylm-eval-pipeline-1")
RUN_PATH = EVAL_REPO / "evaluation_pipeline" / "sentence_zero_shot" / "run.py"
COMPUTE_PATH = EVAL_REPO / "evaluation_pipeline" / "sentence_zero_shot" / "compute_results.py"


def patch_run() -> None:
    text = RUN_PATH.read_text(encoding="utf-8")
    if "score_normalization" not in text:
        text = text.replace(
            '    parser.add_argument("--non_causal_batch_size", default=64, type=int, help="Mini-batch size to process each batch of inputs involving masked tokens")',
            '    parser.add_argument("--non_causal_batch_size", default=64, type=int, help="Mini-batch size to process each batch of inputs involving masked tokens")\n'
            '    parser.add_argument("--score_normalization", choices=["sum", "mean"], default="sum", help="Aggregate token log-probabilities with a sum or per-token mean. Mean is useful for MLM pseudo-likelihood when candidates tokenize to different lengths.")',
        )
        text = text.replace(
            '    args.output_path = args.output_dir / args.model_name / revision_name / "zero_shot" / args.backend / args.task / dataset',
            '    backend_dir = args.backend if args.score_normalization == "sum" else f"{args.backend}_{args.score_normalization}"\n'
            '    args.output_path = args.output_dir / args.model_name / revision_name / "zero_shot" / backend_dir / args.task / dataset',
        )
        RUN_PATH.write_text(text, encoding="utf-8", newline="\n")


def patch_compute_results() -> None:
    text = COMPUTE_PATH.read_text(encoding="utf-8")
    if "aggregate_token_log_probs" not in text:
        text = text.replace(
            "DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')\n\n\n",
            "DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')\n\n\n"
            "def aggregate_token_log_probs(values: torch.Tensor, args: argparse.ArgumentParser) -> float:\n"
            '    """Aggregate token log-probs for one candidate sentence/completion."""\n'
            "    if values.numel() == 0:\n"
            "        return 0.0\n"
            '    if getattr(args, "score_normalization", "sum") == "mean":\n'
            "        return torch.mean(values).item()\n"
            "    return torch.sum(values).item()\n\n\n",
        )
        text = text.replace(
            "summed_log_probs.append(torch.sum(concat_temp_log_probs[start_idx:end_idx]).item())",
            "summed_log_probs.append(aggregate_token_log_probs(concat_temp_log_probs[start_idx:end_idx], args))",
        )
        COMPUTE_PATH.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    patch_run()
    patch_compute_results()
    print(f"patched {RUN_PATH}")
    print(f"patched {COMPUTE_PATH}")


if __name__ == "__main__":
    main()
