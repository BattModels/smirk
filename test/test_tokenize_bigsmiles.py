import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import smirk
from smirk.smirk import SmirkTokenizer


def _assert_pretokenize(
    tokenizer: SmirkTokenizer, text: str, expected_tokens: list[str]
) -> None:
    assert tokenizer.pretokenize(text) == expected_tokens


def _tokens(spec: str) -> list[str]:
    return spec.split()


def _smi_fixture(filename: str) -> list[tuple[str, str]]:
    path = Path(__file__).with_name(filename)
    return [
        (f"{filename}:{idx}", line)
        for idx, line in enumerate(path.read_text().splitlines(), start=1)
        if line and not line.startswith("#")
    ]


def _bigsmiles_fixture() -> list[tuple[str, str]]:
    return _smi_fixture("bigsmiles.smi")


def _bigsmiles_csv_fixture() -> list[tuple[str, str]]:
    path = Path(__file__).with_name("bigsmiles.csv")
    data_lines = [
        (idx, line)
        for idx, line in enumerate(path.read_text().splitlines(), start=1)
        if line and not line.startswith("#")
    ]
    reader = csv.reader(line for _, line in data_lines)
    header = next(reader)
    bigsmiles_index = header.index("BigSMILES")
    return [
        (f"bigsmiles.csv:{line_no}", row[bigsmiles_index])
        for (line_no, _), row in zip(data_lines[1:], reader)
        if row[bigsmiles_index]
    ]


INLINE_ROUNDTRIP_BIGSMILES = [
    "{[$]CC[$]}",
    "{[$]CC[$],[$]C(C)C[$]}",
    "{[<]CC[>]}",
    "[$1]",
    "[<2]",
    "[]",
    "{[]CC[$]}",
    "{[$]CC[$];C[$],[$]C}",
    "CC{[$]CC[$]}CC",
    "{[$]CC(c1ccccc1)[$]}",
    "{[>]CCCCCC(=O)[<],[>]NCCCCCCN[<]}",
    "{[$]CC[$]}{[$]CC(C)[$]}",
]


def _roundtrip_fixtures() -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "inline",
            [
                (f"inline:{idx}", text)
                for idx, text in enumerate(INLINE_ROUNDTRIP_BIGSMILES, start=1)
            ],
        ),
        ("bigsmiles.smi", _bigsmiles_fixture()),
        ("opensmiles.smi", _smi_fixture("opensmiles.smi")),
        ("bigsmiles.csv", _bigsmiles_csv_fixture()),
    ]


UNDEFINED_FRAGMENT_PLACEHOLDER_CASES = [
    ("{[][$]CC(C)([#R])[$][]}", ["#R"]),
    ("C([#Arm])([#Arm])([#Arm])[#Arm]", ["#Arm", "#Arm", "#Arm", "#Arm"]),
    (
        "{[][<][#A][#R][#A][<],[>][#B][#R']([#B][>])([#B][>])[#B][>][]}",
        ["#A", "#R", "#A", "#B", "#R'", "#B", "#B", "#B"],
    ),
    (
        "{[][<][#A][#R][#A][<],[>][#B][#R']([#B][>])([#B][>])[#B][>];"
        "[>][#E1],[<][#E2][]}",
        ["#A", "#R", "#A", "#B", "#R'", "#B", "#B", "#B", "#E1", "#E2"],
    ),
    (
        "{[][>]COC(=O){[$][$]COC[$][$]}C(=O)OC[>],c1([<])cc([#L]2)cc([#L]3)c1."
        "c4([<])cc([#L]5)cc([#L]6)c4.c7([<])cc([#L]8)cc([#L]9)c7."
        "C%10([<])cc([#L]%11)cc([#L]%12)c%10.[Pd++]258%11.[Pd++]369%12}",
        ["#L", "#L", "#L", "#L", "#L", "#L", "#L", "#L"],
    ),
]

BARE_LABEL_BIGSMILES_CASES = [
    (
        "A([<1[Inner]1])R(A[<1[Inner]1])(B[>1[Inner]2])B[>1[Inner]2]",
        7,
    ),
    (
        "A([<1[<1]1])R(A[<1[<1]1])(B[>1[>1]2])B[>1[>1]2]",
        3,
    ),
    (
        "A([$1[Inner]1])R(A'[$1[Inner]1])(A[$1[Inner]2])A'[$1[Inner]2]",
        9,
    ),
    (
        "A([$1[$1]1])R(A'[$1[$1]1])(A[$1[$1]2])A'[$1[$1]2]",
        5,
    ),
    (
        "A([$1[$1]1])R(A'[$1[$2]1])(A[$1[$1]2])A'[$1[$2]2]",
        5,
    ),
    (
        "A([$1[<1]1])R(A'[$1[>1]1])(A[$1[<1]2])A'[$1[>1]2]",
        5,
    ),
]
BARE_LABEL_BIGSMILES_WITH_DEFINITIONS = [
    (
        "A([<1[Inner]1])R(A[<1[Inner]1])(B[>1[Inner]2])B[>1[Inner]2]."
        "{#A=C}.{#R=C}.{#B=C}.{#Inner=<}"
    ),
    "A([<1[<1]1])R(A[<1[<1]1])(B[>1[>1]2])B[>1[>1]2].{#A=C}.{#R=C}.{#B=C}",
    (
        "A([$1[Inner]1])R(A'[$1[Inner]1])(A[$1[Inner]2])A'[$1[Inner]2]."
        "{#A=C}.{#A'=C}.{#R=C}.{#Inner=$}"
    ),
    "A([$1[$1]1])R(A'[$1[$1]1])(A[$1[$1]2])A'[$1[$1]2].{#A=C}.{#A'=C}.{#R=C}",
    "A([$1[$1]1])R(A'[$1[$2]1])(A[$1[$1]2])A'[$1[$2]2].{#A=C}.{#A'=C}.{#R=C}",
    "A([$1[<1]1])R(A'[$1[>1]1])(A[$1[<1]2])A'[$1[>1]2].{#A=C}.{#A'=C}.{#R=C}",
]
EXPECTED_UNKNOWN_BIGSMILES_TEXTS = (
    {text for text, _ in UNDEFINED_FRAGMENT_PLACEHOLDER_CASES}
    | {text for text, _ in BARE_LABEL_BIGSMILES_CASES}
    | set(BARE_LABEL_BIGSMILES_WITH_DEFINITIONS)
)
NON_EXACT_ROUNDTRIP_TEXTS = EXPECTED_UNKNOWN_BIGSMILES_TEXTS | {"[Cu++]"}


def _is_lossless_roundtrip_text(text: str) -> bool:
    return ".{#" not in text and text not in NON_EXACT_ROUNDTRIP_TEXTS


@pytest.fixture
def bigsmiles_tokenizer() -> SmirkTokenizer:
    return SmirkTokenizer(bigsmiles=True)


@pytest.fixture
def smiles_tokenizer() -> SmirkTokenizer:
    return SmirkTokenizer(bigsmiles=False)


ROUNDTRIP_FIXTURES = _roundtrip_fixtures()


@pytest.mark.parametrize(
    ("fixture_name", "fixture_rows"),
    ROUNDTRIP_FIXTURES,
    ids=[name for name, _ in ROUNDTRIP_FIXTURES],
)
def test_bigsmiles_roundtrip_batch_decode(fixture_name, fixture_rows):
    bigsmirk = smirk.SmirkBigSmilesFast()
    bigsmiles_batch = [text for _, text in fixture_rows]
    encoded = bigsmirk(bigsmiles_batch, add_special_tokens=False)
    decoded = bigsmirk.batch_decode(encoded["input_ids"], skip_special_tokens=True)
    itemwise_decoded = [
        bigsmirk.decode(ids, skip_special_tokens=True) for ids in encoded["input_ids"]
    ]
    exact_failures = [
        f"{source}: expected {text!r}, got {decoded_text!r}"
        for (source, text), decoded_text in zip(fixture_rows, decoded)
        if _is_lossless_roundtrip_text(text) and decoded_text != text
    ]

    assert decoded == itemwise_decoded
    assert len(decoded) == len(bigsmiles_batch)
    assert not exact_failures, (
        f"{fixture_name} exact roundtrip mismatches:\n" + "\n".join(exact_failures)
    )


def test_bigsmiles_fixture_has_no_unknown_tokens():
    bigsmirk = smirk.SmirkBigSmilesFast()
    failures = []

    for line_no, text in _bigsmiles_fixture():
        if text in EXPECTED_UNKNOWN_BIGSMILES_TEXTS:
            continue

        tokens = bigsmirk.tokenize(text, add_special_tokens=False)
        if bigsmirk.unk_token in tokens:
            failures.append(f"line {line_no}: {text}")

    assert not failures, "unknown tokens in BigSMILES fixtures:\n" + "\n".join(failures)


def _unknown_spans(bigsmirk: smirk.SmirkBigSmilesFast, text: str) -> list[str]:
    encoding = bigsmirk._tokenizer.encode(text, add_special_tokens=False)
    unk_token_id = bigsmirk._tokenizer.token_to_id(bigsmirk.unk_token)
    return [
        text[start:end]
        for token_id, (start, end) in zip(encoding["input_ids"], encoding["offsets"])
        if token_id == unk_token_id
    ]


@pytest.mark.parametrize(
    ("text", "unknown_spans"),
    UNDEFINED_FRAGMENT_PLACEHOLDER_CASES,
)
def test_bigsmiles_undefined_fragment_placeholders_return_unknowns(text, unknown_spans):
    bigsmirk = smirk.SmirkBigSmilesFast()
    tokens = bigsmirk.tokenize(text, add_special_tokens=False)
    actual_unknown_spans = _unknown_spans(bigsmirk, text)

    assert actual_unknown_spans == unknown_spans
    assert tokens.count(bigsmirk.unk_token) == len(actual_unknown_spans)


@pytest.mark.parametrize(
    ("text", "unknown_count"),
    BARE_LABEL_BIGSMILES_CASES,
)
def test_bigsmiles_bare_labels_return_unknowns(text, unknown_count):
    bigsmirk = smirk.SmirkBigSmilesFast()
    tokens = bigsmirk.tokenize(text, add_special_tokens=False)

    assert tokens.count(bigsmirk.unk_token) == unknown_count


@pytest.mark.parametrize(
    ("text", "expected_tokens"),
    [
        ("OC[C@@H]", _tokens("O C [ C @@ H ]")),
        ("C[C@H](N)C(=O)O", _tokens("C [ C @ H ] ( N ) C ( = O ) O")),
    ],
)
def test_smiles_tokens_match_between_modes(
    bigsmiles_tokenizer, smiles_tokenizer, text, expected_tokens
):
    _assert_pretokenize(bigsmiles_tokenizer, text, expected_tokens)
    _assert_pretokenize(smiles_tokenizer, text, expected_tokens)


@pytest.mark.parametrize(
    "bigsmiles,expected_type",
    [
        (True, "BigSmirkPreTokenizer"),
        (False, None),
    ],
)
def test_tokenizer_serialize_pretokenizer_type(bigsmiles, expected_type):
    tokenizer = SmirkTokenizer(bigsmiles=bigsmiles)
    config = json.loads(tokenizer.to_str())
    assert "pre_tokenizer" in config

    if expected_type is None:
        assert "type" not in config["pre_tokenizer"]
        assert "bigsmiles_version" not in config["pre_tokenizer"]
    else:
        assert config["pre_tokenizer"].get("type") == expected_type
        assert config["pre_tokenizer"].get("bigsmiles_version") == "1.1"


@pytest.mark.parametrize(
    "text",
    [
        "{[$]CC[$]}",
        "{[<]CC[>]}",
        "{[]CC[$]}",
        "{[$]CC[$],[$]C(C)C[$]}",
    ],
)
def test_bigsmiles_tokenizer_save_load(bigsmiles_tokenizer, text):
    with NamedTemporaryFile("w", suffix=".json", delete=False) as file:
        bigsmiles_tokenizer.save(file.name)
        with open(file.name) as saved:
            config = json.load(saved)
        loaded = SmirkTokenizer.from_file(file.name)

    assert config["pre_tokenizer"].get("bigsmiles_version") == "1.1"
    original_splits = bigsmiles_tokenizer.pretokenize(text)
    loaded_splits = loaded.pretokenize(text)
    assert original_splits == loaded_splits
