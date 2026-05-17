"""Tests for the Unigram-LM tokenizer arm (``smirk.train_unigram``).

The Unigram sibling of ``test_smirk_gpe.py``. Exercises the four
``merge_brackets`` x ``split_structure`` boundary configs, encode/decode
round-trips, the no-Private-Use-Area-in-artifacts guarantee, and Unigram
piece-set determinism — for the Unigram arm the *piece set* is what is
reproducible across retraining, not the byte-exact artifact
(``vocab-tokenizer-clms`` study, §3.2 determinism amendment).
"""

from pathlib import Path

import pytest
import smirk

from .test_tokenize_smiles import smile_strings  # noqa

SMILE_TEST_FILE = Path(__file__).parent.joinpath("smiles.txt")

# (merge_brackets, split_structure) — the four boundary configs.
CONFIGS = [
    {"merge_brackets": False, "split_structure": False},
    {"merge_brackets": True, "split_structure": True},
    {"merge_brackets": True, "split_structure": False},
    {"merge_brackets": False, "split_structure": True},
]

# The Basic Multilingual Plane Private Use Area block.
PUA_LO, PUA_HI = 0xE000, 0xF8FF


def _piece_set(**config) -> list[str]:
    tok = smirk.train_unigram([str(SMILE_TEST_FILE)], vocab_size=256, **config)
    return sorted(tok.get_vocab().keys())


@pytest.fixture(scope="session", params=CONFIGS)
def trained(request):
    return smirk.train_unigram([str(SMILE_TEST_FILE)], vocab_size=256, **request.param)


@pytest.mark.usefixtures("smile_strings")
def test_train_unigram_round_trips(trained, smile_strings):  # noqa F811
    code = trained(smile_strings)
    decode = trained.batch_decode(code["input_ids"])
    assert decode == smile_strings


def test_trained_vocab_exceeds_the_atomic_baseline(trained):
    assert trained.vocab_size > smirk.SmirkTokenizerFast().vocab_size
    assert "[PAD]" in trained.get_vocab()


def test_no_pua_codepoints_in_artifact(trained):
    # §3.2: the PUA bridge is internal to training — no Private Use Area
    # codepoint may surface in a saved tokenizer artifact.
    blob = trained.to_str()
    assert not any(PUA_LO <= ord(c) <= PUA_HI for c in blob)
    for token in trained.get_vocab():
        assert not any(PUA_LO <= ord(c) <= PUA_HI for c in token), token


def test_vocab_size_counts_base_plus_pieces():
    # vocab_size (mirrored from GpeTrainer) counts the base alphabet plus
    # multi-glyph pieces. On the MB / no-split config the toy corpus yields
    # enough multi-glyph pieces to hit the target exactly.
    tok = smirk.train_unigram(
        [str(SMILE_TEST_FILE)],
        vocab_size=256,
        merge_brackets=True,
        split_structure=False,
    )
    assert tok.vocab_size == 256


@pytest.mark.parametrize("config", CONFIGS)
def test_training_is_piece_set_deterministic(config):
    # §3.2 determinism amendment: the Unigram piece *set* is reproducible
    # across retraining on identical input (ids / scores may permute).
    assert _piece_set(**config) == _piece_set(**config)


def test_max_piece_length_caps_multi_glyph_pieces():
    # main.tex §3.2: max_piece_length is a settable knob. A one-glyph cap
    # leaves HuggingFace no multi-glyph pieces to seed, so the trained
    # vocabulary is strictly smaller than an uncapped run at the same target.
    capped = smirk.train_unigram(
        [str(SMILE_TEST_FILE)], vocab_size=256, max_piece_length=1
    )
    uncapped = smirk.train_unigram(
        [str(SMILE_TEST_FILE)], vocab_size=256, max_piece_length=128
    )
    assert capped.vocab_size < uncapped.vocab_size


def test_seed_size_knob_is_accepted():
    # main.tex §3.2: seed_size is a settable knob (the Phase-2.5 seed-cap
    # spot-check). A reduced cap still trains a vocabulary past the baseline.
    tok = smirk.train_unigram([str(SMILE_TEST_FILE)], vocab_size=256, seed_size=2048)
    assert tok.vocab_size > smirk.SmirkTokenizerFast().vocab_size
