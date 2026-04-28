import json
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


@pytest.fixture
def bigsmiles_tokenizer() -> SmirkTokenizer:
    return SmirkTokenizer(bigsmiles=True)


@pytest.fixture
def smiles_tokenizer() -> SmirkTokenizer:
    return SmirkTokenizer(bigsmiles=False)


@pytest.mark.parametrize(
    "bigsmiles_batch",
    [
        ["{[$]CC[$]}"],
        ["{[$]CC[$],[$]C(C)C[$]}", "{[<]CC[>]}"],
        ["[$1]", "[<2]", "[]", "{[]CC[$]}", "{[$]CC[$];C[$],[$]C}"],
        ["CC{[$]CC[$]}CC", "{[$]CC(c1ccccc1)[$]}"],
        ["{[>]CCCCCC(=O)[<],[>]NCCCCCCN[<]}", "{[$]CC[$]}{[$]CC(C)[$]}"],
    ],
)
def test_bigsmiles_roundtrip_batch_decode(bigsmiles_batch):
    bigsmirk = smirk.SmirkBigSmilesFast()
    encoded = bigsmirk(bigsmiles_batch, add_special_tokens=False)
    decoded = bigsmirk.batch_decode(encoded["input_ids"], skip_special_tokens=True)
    assert decoded == bigsmiles_batch


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
