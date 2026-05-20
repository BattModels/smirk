"""Layer-B chunker binding (vocab-tokenizer-clms Step 3.2).

Read-only Python access to the shared structural pre-tokenizer used by both
trainers. Tests pin the chunk boundaries that the §3.2 preregistration relies
on; matching coverage and offset invariants are asserted so the binding can
be consumed at inference time without surprises.
"""

from importlib.resources import files

import pytest
from smirk import SmirkTokenizerFast
from smirk.smirk import SmirkTokenizer


@pytest.fixture
def rs_tokenizer():
    vocab_file = files("smirk").joinpath("vocab_smiles.json")
    assert vocab_file.is_file()
    return SmirkTokenizer.from_vocab(str(vocab_file))


@pytest.fixture
def fast_tokenizer():
    return SmirkTokenizerFast()


# Pinned chunk boundaries (mirror src/pre_tokenizers/mod.rs::test_split_structure).
EXPECTED_CHUNKS = {
    "CC": [("CC", (0, 2))],
    "C.C": [("C", (0, 1)), (".", (1, 2)), ("C", (2, 3))],
    "C(C)": [("C", (0, 1)), ("(", (1, 2)), ("C", (2, 3)), (")", (3, 4))],
    "C/C": [("C", (0, 1)), ("/", (1, 2)), ("C", (2, 3))],
    r"C\C": [("C", (0, 1)), ("\\", (1, 2)), ("C", (2, 3))],
    "C[13C]": [("C", (0, 1)), ("[13C]", (1, 6))],
    "C%10ccccc%10C": [
        ("C", (0, 1)),
        ("%10", (1, 4)),
        ("ccccc", (4, 9)),
        ("%10", (9, 12)),
        ("C", (12, 13)),
    ],
    "C1ccccc1C": [
        ("C", (0, 1)),
        ("1", (1, 2)),
        ("ccccc", (2, 7)),
        ("1", (7, 8)),
        ("C", (8, 9)),
    ],
    "[NH4+]": [("[NH4+]", (0, 6))],
    "CC(=O)O": [
        ("CC", (0, 2)),
        ("(", (2, 3)),
        ("=O", (3, 5)),
        (")", (5, 6)),
        ("O", (6, 7)),
    ],
}


@pytest.mark.parametrize(("smi", "expected"), list(EXPECTED_CHUNKS.items()))
def test_pretokenize_layer_b_rust(rs_tokenizer, smi, expected):
    chunks = rs_tokenizer.pretokenize_layer_b(smi)
    assert [(c, tuple(o)) for c, o in chunks] == expected


@pytest.mark.parametrize(("smi", "expected"), list(EXPECTED_CHUNKS.items()))
def test_pretokenize_layer_b_fast(fast_tokenizer, smi, expected):
    chunks = fast_tokenizer.pretokenize_layer_b(smi)
    assert [(c, tuple(o)) for c, o in chunks] == expected


@pytest.mark.parametrize("smi", list(EXPECTED_CHUNKS))
def test_layer_b_spans_cover_input(rs_tokenizer, smi):
    chunks = rs_tokenizer.pretokenize_layer_b(smi)
    # Reconstruction by chunk-string concatenation matches the input.
    assert "".join(c for c, _ in chunks) == smi
    # Offsets are non-overlapping and cover [0, len(smi)).
    spans = [tuple(o) for _, o in chunks]
    assert spans[0][0] == 0
    assert spans[-1][1] == len(smi)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start


def test_layer_b_empty_input(rs_tokenizer):
    assert rs_tokenizer.pretokenize_layer_b("") == []


def test_layer_b_independent_of_model(fast_tokenizer):
    # Same chunks for the stock baseline and a freshly-constructed instance —
    # the Layer-B chunker is read-only and ignores the underlying model.
    other = SmirkTokenizerFast()
    smi = "CCOc1ccc(C(=O)c2ccccc2)cc1"
    assert fast_tokenizer.pretokenize_layer_b(smi) == other.pretokenize_layer_b(smi)
