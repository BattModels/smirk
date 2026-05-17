"""Scaffold-instrumentation log contract for Step 1.5 of vocab-tokenizer-clms.

``GpeTrainer`` is instrumented (``main.tex`` §3.2) to stream a per-merge-step
JSONL log when ``scaffold_log_path`` is set: a header line, then one record
per committed merge carrying that merge's selected-candidate frequency and the
running standalone frequency of the tokens it touched (delta-encoded — only
the left, right and merged token, since no other token's standalone frequency
changes).

The instrumentation is logging-only: a tokenizer trained with the log enabled
is byte-identical to one trained without it. This test pins the log's shape
and that byte-identity across the four ``merge_brackets`` x ``split_structure``
boundary configs. ``test_byte_identity.py`` separately pins the default
(``scaffold_log_path=None``) path against stock smirk v0.2.0.
"""

import json
from pathlib import Path

import pytest
import smirk

TEST_DIR = Path(__file__).parent
SMILES_FILE = str(TEST_DIR / "smiles.txt")

# (merge_brackets, split_structure) — the four boundary configs.
CONFIGS = [
    (False, False),
    (True, True),
    (True, False),
    (False, True),
]
VOCAB_SIZE = 512

RECORD_KEYS = {"step", "pair", "new_id", "new_token", "candidate_freq", "standalone"}


def _train(
    merge_brackets: bool, split_structure: bool, scaffold_log_path: str | None = None
) -> smirk.SmirkTokenizerFast:
    """Train a GPE tokenizer, optionally streaming a scaffold log."""
    return smirk.train_gpe(
        [SMILES_FILE],
        merge_brackets=merge_brackets,
        split_structure=split_structure,
        vocab_size=VOCAB_SIZE,
        scaffold_log_path=scaffold_log_path,
    )


@pytest.mark.parametrize("merge_brackets,split_structure", CONFIGS)
def test_scaffold_log_is_well_formed(
    merge_brackets: bool, split_structure: bool, tmp_path: Path
) -> None:
    """The log is a JSON header line followed by one record per committed merge."""
    log = tmp_path / "scaffold.jsonl"
    tok = _train(merge_brackets, split_structure, str(log))
    assert log.is_file()

    lines = log.read_text().splitlines()
    header = json.loads(lines[0])
    assert header["format"] == "smirk-scaffold-log/v1"
    assert header["merge_brackets"] == merge_brackets
    assert header["vocab_size"] == VOCAB_SIZE
    assert len(header["base_alphabet"]) >= 1

    records = [json.loads(line) for line in lines[1:]]
    n_merges = len(json.loads(tok.to_str())["model"]["merges"])
    assert len(records) == n_merges

    for step, rec in enumerate(records):
        assert RECORD_KEYS <= set(rec)
        assert rec["step"] == step
        assert len(rec["pair"]) == 2
        assert rec["candidate_freq"] >= 1
        # Delta-encoded: the left, right and merged token — 2 on a self-pair.
        assert 2 <= len(rec["standalone"]) <= 3


@pytest.mark.parametrize("merge_brackets,split_structure", CONFIGS)
def test_scaffold_log_does_not_change_the_tokenizer(
    merge_brackets: bool, split_structure: bool, tmp_path: Path
) -> None:
    """A tokenizer trained with the log enabled is byte-identical to one without."""
    log = tmp_path / "scaffold.jsonl"
    with_log = _train(merge_brackets, split_structure, str(log))
    without_log = _train(merge_brackets, split_structure, None)
    assert with_log.to_str() == without_log.to_str()
