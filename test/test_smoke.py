from pathlib import Path

from smirk import SmirkBigSmilesFast, SmirkTokenizerFast

TEST_DIR = Path(__file__).parent


def test_smirk():
    tok = SmirkTokenizerFast()
    mask_token_id = tok.mask_token_id
    assert mask_token_id is not None
    with open(TEST_DIR.joinpath("smiles.txt")) as file:
        for line in file:
            if line.startswith("#"):
                continue
            out = tok(line.strip())
            assert mask_token_id not in out["input_ids"]


def test_bigsmirk():
    tok = SmirkBigSmilesFast()
    mask_token_id = tok.mask_token_id
    assert mask_token_id is not None
    with open(TEST_DIR.joinpath("bigsmiles.smi")) as file:
        for line in file:
            if line.startswith("#"):
                continue
            out = tok(line.strip())
            assert mask_token_id not in out["input_ids"]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
