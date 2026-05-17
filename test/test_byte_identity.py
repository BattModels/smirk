"""Byte-identity contract for the shared pre-tokenization refactor.

Step 1.2 of the ``vocab-tokenizer-clms`` study extracts ``compute_alphabet``
and ``tokenize_words`` out of ``GpeTrainer`` into a shared module that both the
BPE trainer and (later) a Unigram-LM sibling trainer call. The refactor is a
pure code move: it must not change a single trained artifact.

This test pins that contract. It trains ``GpeTrainer`` through the public
``smirk.train_gpe`` entry point across the four ``merge_brackets`` x
``split_structure`` boundary configs (the matrix already exercised by
``test_smirk_gpe.py``) and two vocab sizes, and asserts the serialized
tokenizer is **byte-identical** to a golden captured from stock smirk v0.2.0.

The two vocab sizes exercise both training-loop exit conditions:
``256`` binds the ``vocab_size`` cutoff for the ``split_structure=False``
configs and exhausts the merge queue for the others; ``4096`` exhausts all.

Regenerate the goldens by running this file as a script::

    uv run python test/test_byte_identity.py

Regenerate **only** against stock smirk v0.2.0 (tag ``v0.2.0`` / branch
``vtc-main`` before the refactor) — never against refactored code, or the
test becomes circular.
"""

import json
from pathlib import Path

import pytest
import smirk

TEST_DIR = Path(__file__).parent
GOLDEN_DIR = TEST_DIR / "golden"
SMILES_FILE = str(TEST_DIR / "smiles.txt")

# (merge_brackets, split_structure) — the four boundary configs.
CONFIGS = [
    (False, False),
    (True, True),
    (True, False),
    (False, True),
]
VOCAB_SIZES = [256, 4096]


def golden_path(merge_brackets: bool, split_structure: bool, vocab_size: int) -> Path:
    """Path of the golden artifact for one (config, size) cell."""
    return GOLDEN_DIR / (
        f"gpe_mb{int(merge_brackets)}_ss{int(split_structure)}_v{vocab_size}.json"
    )


def train_to_str(merge_brackets: bool, split_structure: bool, vocab_size: int) -> str:
    """Train a GPE tokenizer and return its serialized JSON form.

    A trailing newline is appended so the golden files on disk are
    well-formed POSIX text files (and survive the ``end-of-file-fixer``
    pre-commit hook); the JSON artifact itself is everything before it.
    """
    tok = smirk.train_gpe(
        [SMILES_FILE],
        merge_brackets=merge_brackets,
        split_structure=split_structure,
        vocab_size=vocab_size,
    )
    return tok.to_str() + "\n"


@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
@pytest.mark.parametrize("merge_brackets,split_structure", CONFIGS)
def test_bpe_artifacts_byte_identical_to_stock(
    merge_brackets: bool, split_structure: bool, vocab_size: int
) -> None:
    """Refactored GpeTrainer reproduces stock v0.2.0 artifacts byte-for-byte."""
    golden_file = golden_path(merge_brackets, split_structure, vocab_size)
    assert golden_file.is_file(), (
        f"missing golden {golden_file.name}; "
        "regenerate with `uv run python test/test_byte_identity.py` on stock v0.2.0"
    )
    golden = golden_file.read_text()
    actual = train_to_str(merge_brackets, split_structure, vocab_size)

    if actual != golden:
        # Surface a structural diff before the raw byte assertion: if only the
        # JSON key order drifted, the model vocab/merges still match.
        a_model = json.loads(actual)["model"]
        g_model = json.loads(golden)["model"]
        assert a_model["vocab"] == g_model["vocab"], "trained vocab diverged from stock"
        assert a_model["merges"] == g_model["merges"], "merge order diverged from stock"
    assert (
        actual == golden
    ), "serialized tokenizer is not byte-identical to stock v0.2.0"


def test_training_is_deterministic() -> None:
    """A second training run on identical input yields identical artifacts."""
    first = train_to_str(False, True, 4096)
    second = train_to_str(False, True, 4096)
    assert first == second


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(exist_ok=True)
    for size in VOCAB_SIZES:
        for mb, ss in CONFIGS:
            text = train_to_str(mb, ss, size)
            out = golden_path(mb, ss, size)
            out.write_text(text)
            print(f"wrote {out.relative_to(TEST_DIR.parent)} ({len(text)} bytes)")
