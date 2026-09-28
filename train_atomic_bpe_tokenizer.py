"""Train BPE over indivisible two-character pinyin-code syllable atoms."""

from __future__ import annotations

import argparse
import json
import shutil
import string
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from hf.tokenization_atomic_bpe_pinyin_code import BOUNDARY_POLICIES, CROSS_WORD, WITHIN_WORD
from preprocessing.encoding import encoded_word_atoms, is_encoded_run, split_encoded_words
from train_hybrid_tokenizer import (
    FIXED_SPECIAL_TOKENS,
    atomic_tokens,
    collect_counts,
    ordered_vocab,
    resolve_min_preserved_frequency,
    surface_char_tokens,
    write_json,
)
from train_sentencepiece import SPECIAL_TOKENS


METADATA_NAME = "atomic_bpe_tokenizer_metadata.json"
MERGES_NAME = "atomic_bpe_merges.json"
PRIVATE_USE_START = 0xE000


def require_tokenizers():
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install with `py -m pip install tokenizers`."
        ) from exc
    return Tokenizer, BPE, WhitespaceSplit, BpeTrainer


def atom_symbol_maps(atoms: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    if len(atoms) > 0xF8FF - PRIVATE_USE_START + 1:
        raise ValueError("Atomic alphabet is too large for the BMP private-use mapping")
    atom_to_symbol = {
        atom: chr(PRIVATE_USE_START + index) for index, atom in enumerate(atoms)
    }
    return atom_to_symbol, {symbol: atom for atom, symbol in atom_to_symbol.items()}


def map_encoded_sequence(sequence: str, atom_to_symbol: dict[str, str]) -> str:
    pieces: list[str] = []
    for atom in encoded_word_atoms(sequence):
        try:
            pieces.append(atom_to_symbol[atom])
        except KeyError as exc:
            raise ValueError(f"Encoded atom {atom!r} is outside the configured alphabet") from exc
    return "".join(pieces)


def iter_mapped_lines(
    input_paths: Iterable[Path],
    boundary_policy: str,
    atom_to_symbol: dict[str, str],
) -> Iterable[str]:
    """Yield encoded-only training lines with policy boundaries represented by spaces."""
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                sequences: list[str] = []
                for item in line.split():
                    if not is_encoded_run(item):
                        continue
                    units = split_encoded_words(item) if boundary_policy == WITHIN_WORD else [item]
                    sequences.extend(map_encoded_sequence(unit, atom_to_symbol) for unit in units)
                if sequences:
                    yield " ".join(sequences)


def unmap_piece(piece: str, symbol_to_atom: dict[str, str]) -> str:
    try:
        return "".join(symbol_to_atom[symbol] for symbol in piece)
    except KeyError as exc:
        raise ValueError(f"Unexpected non-atomic symbol in learned BPE piece {piece!r}") from exc


def train_atom_backend(
    input_paths: list[Path],
    boundary_policy: str,
    atoms: list[str],
    learned_budget: int,
    min_frequency: int,
) -> tuple[list[str], list[tuple[str, str]]]:
    Tokenizer, BPE, WhitespaceSplit, BpeTrainer = require_tokenizers()
    atom_to_symbol, symbol_to_atom = atom_symbol_maps(atoms)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = BpeTrainer(
        vocab_size=len(atoms) + learned_budget + 1,
        min_frequency=min_frequency,
        limit_alphabet=len(atoms),
        show_progress=True,
        special_tokens=["<unk>"],
        initial_alphabet=list(atom_to_symbol.values()),
    )
    tokenizer.train_from_iterator(
        iter_mapped_lines(input_paths, boundary_policy, atom_to_symbol),
        trainer=trainer,
    )

    with tempfile.TemporaryDirectory(prefix="atomic-bpe-") as temp_dir:
        vocab_path, merges_path = tokenizer.model.save(temp_dir, "backend")
        backend_vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        merge_lines = Path(merges_path).read_text(encoding="utf-8").splitlines()

    learned_pieces = [
        unmap_piece(piece, symbol_to_atom)
        for piece, _piece_id in sorted(backend_vocab.items(), key=lambda item: item[1])
        if piece != "<unk>" and len(piece) > 1
    ]
    merges: list[tuple[str, str]] = []
    for line in merge_lines:
        if not line or line.startswith("#"):
            continue
        left, right = line.split()
        merges.append((unmap_piece(left, symbol_to_atom), unmap_piece(right, symbol_to_atom)))
    return learned_pieces, merges


def build_atomic_bpe(args: argparse.Namespace) -> tuple[dict[str, int], dict, list[tuple[str, str]]]:
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive")
    if args.min_frequency <= 0:
        raise ValueError("--min-frequency must be positive")
    min_preserved_frequency = resolve_min_preserved_frequency(args)
    atoms = atomic_tokens(args.initial_alphabet, args.digits)
    encoded_counts, _special_counts, preserved_counts, stats = collect_counts(
        args.input,
        permissive=args.permissive,
        max_invalid_examples=args.max_invalid_examples,
    )
    del encoded_counts
    selected_preserved = sorted(
        token
        for token, count in preserved_counts.items()
        if len(token) > 1 and count >= min_preserved_frequency
    )
    base_tokens = FIXED_SPECIAL_TOKENS + SPECIAL_TOKENS + surface_char_tokens() + selected_preserved + atoms
    base_vocab = ordered_vocab(base_tokens)
    if len(base_vocab) > args.vocab_size:
        raise ValueError(
            f"--vocab-size {args.vocab_size} is smaller than the required base "
            f"vocabulary ({len(base_vocab)} tokens)"
        )
    learned_budget = args.vocab_size - len(base_vocab)
    learned_pieces, merges = train_atom_backend(
        args.input,
        args.boundary_policy,
        atoms,
        learned_budget,
        args.min_frequency,
    )
    vocab = ordered_vocab([*base_vocab, *learned_pieces])
    metadata = {
        "format": "babylm-pinyin-code-atomic-bpe-v1",
        "boundary_policy": args.boundary_policy,
        "target_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "minimum_merge_frequency": args.min_frequency,
        "minimum_preserved_frequency": min_preserved_frequency,
        "permissive": args.permissive,
        "corpus_paths": [str(path) for path in args.input],
        "atomic_alphabet": args.initial_alphabet,
        "atomic_digits": args.digits,
        "number_of_atomic_tokens": len(atoms),
        "number_of_learned_pieces": len(learned_pieces),
        "number_of_merges": len(merges),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "statistics": stats,
    }
    return vocab, metadata, merges


def write_tokenizer_files(
    output_dir: Path,
    vocab: dict[str, int],
    metadata: dict,
    merges: list[tuple[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "vocab.json", vocab)
    write_json(output_dir / METADATA_NAME, metadata)
    write_json(
        output_dir / MERGES_NAME,
        {
            "format": "babylm-pinyin-code-atomic-bpe-v1",
            "boundary_policy": metadata["boundary_policy"],
            "merges": [list(pair) for pair in merges],
        },
    )
    write_json(
        output_dir / "special_tokens_map.json",
        {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "mask_token": "<mask>",
        },
    )
    write_json(
        output_dir / "tokenizer_config.json",
        {
            "tokenizer_class": "AtomicBPEPinyinCodeTokenizer",
            "auto_map": {
                "AutoTokenizer": [
                    "tokenization_atomic_bpe_pinyin_code.AtomicBPEPinyinCodeTokenizer",
                    None,
                ]
            },
            "model_max_length": 1000000000000000019884624838656,
            "add_bos_token": False,
            "add_eos_token": False,
            "strict_validation": not bool(metadata.get("permissive", False)),
            "readable_decode": False,
            "boundary_policy": metadata["boundary_policy"],
            "atomic_alphabet": metadata["atomic_alphabet"],
            "atomic_digits": metadata["atomic_digits"],
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "mask_token": "<mask>",
        },
    )
    hf_dir = Path(__file__).resolve().parent / "hf"
    for module_name in (
        "tokenization_atomic_bpe_pinyin_code.py",
        "tokenization_hybrid_pinyin_code.py",
        "tokenization_pinyin_code.py",
    ):
        shutil.copy2(hf_dir / module_name, output_dir / module_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BPE whose indivisible units are complete pinyin-code syllables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("data/processed/10k_babylm_zho.txt")],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-policy", choices=BOUNDARY_POLICIES, required=True)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--min-preserved-frequency", type=int, default=20)
    parser.add_argument("--initial-alphabet", default=string.ascii_uppercase + string.ascii_lowercase)
    parser.add_argument("--digits", default=string.digits)
    parser.add_argument("--permissive", action="store_true")
    parser.add_argument("--max-invalid-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab, metadata, merges = build_atomic_bpe(args)
    write_tokenizer_files(args.output_dir, vocab, metadata, merges)
    print(
        f"Wrote {metadata['boundary_policy']} atomic BPE tokenizer to {args.output_dir}: "
        f"vocab={len(vocab):,}, atoms={metadata['number_of_atomic_tokens']:,}, "
        f"learned={metadata['number_of_learned_pieces']:,}, merges={len(merges):,}"
    )


if __name__ == "__main__":
    main()
