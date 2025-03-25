import inspect
import pickle
from collections.abc import Mapping
from random import shuffle
from tempfile import TemporaryDirectory

import pytest
import smirk
from parameterized import parameterized_class
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    is_torch_available,
)

from .test_smirk_gpe import SMILE_TEST_FILE


def get_smirk_gpe():
    return smirk.train_gpe([str(SMILE_TEST_FILE)])


def transformers_testname(cls, idx, input_dict):
    return f"{cls.__name__}_{input_dict['name']}"


def require_torch(test_case):
    return pytest.mark.skipif(not is_torch_available(), reason="Torch not available")(
        test_case
    )


@parameterized_class(
    [
        {
            "name": "smirk",
            "tokenizer": smirk.SmirkTokenizerFast(),
        },
        {
            "name": "smirk-gpe",
            "tokenizer": get_smirk_gpe(),
        },
    ],
    class_name_func=transformers_testname,
)
class TestTransformers:
    name: str
    tokenizer: PreTrainedTokenizerBase
    tokenizer_class = smirk.SmirkTokenizerFast

    def get_tokenizer(self, *args, **kwargs):
        with TemporaryDirectory() as tmpdirname:
            self.tokenizer.save_pretrained(tmpdirname)
            return self.tokenizer_class.from_pretrained(tmpdirname, *args, **kwargs)

    def get_batch(self):
        batch = [
            "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
            "O=C(O)[C@@H]2N3C(=O)[C@@H](NC(=O)[C@@H](c1ccc(O)cc1)N)[C@H]3SC2(C)C",
            "CN3[C@H]1CC[C@@H]3C[C@@H](C1)OC(=O)C(CO)c2ccccc2",
        ]
        shuffle(batch)
        return batch

    def test_model_input_names_signature(self):
        assert self.tokenizer.model_input_names[0] == "input_ids"

    def test_has_init_kwargs(self):
        tokenizer = self.tokenizer
        assert hasattr(tokenizer, "init_kwargs")

    def test_max_model_length(self):
        tokenizer = self.tokenizer
        assert hasattr(tokenizer, "model_max_length")
        assert tokenizer.model_max_length > 0

    def test_auto_tokenizer(self):
        tokenizer = self.get_tokenizer()
        with TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            tok = AutoTokenizer.from_pretrained(tmpdir)
            assert isinstance(tok, self.tokenizer_class)
            assert isinstance(tok, PreTrainedTokenizerBase)
            assert tok.to_str() == tokenizer.to_str()

    def test_save_and_load_tokenizer(self, tokenizer=None):
        tokenizer = tokenizer or self.get_tokenizer()
        with TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            state = tokenizer.to_str()
            del tokenizer
            tok2 = self.tokenizer_class.from_pretrained(tmpdir)
        assert tok2.to_str() == state
        return tok2

    def test_rust_tokenizer_signature(self):
        signature = inspect.signature(self.tokenizer_class)
        assert "tokenizer_file" in signature.parameters
        assert signature.parameters["tokenizer_file"].default is None

    def test_tokenizer_fast_store_full_signature(self):
        signature = inspect.signature(self.tokenizer_class)
        tokenizer = self.tokenizer

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty and parameter_name not in [
                "vocab_file",
                "merges_file",
                "tokenizer_file",
            ]:
                assert parameter_name in tokenizer.init_kwargs

    def test_tokenize_special_tokens(self):
        tokenizer = self.get_tokenizer()
        SPECIAL_TOKEN_1 = "[SPECIAL_TOKEN_1]"
        SPECIAL_TOKEN_2 = "[SPECIAL_TOKEN_2]"
        tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [SPECIAL_TOKEN_2]},
            replace_additional_special_tokens=False,
        )
        token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
        token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)
        assert len(token_1) == 1
        assert len(token_2) == 1
        assert token_1[0] == SPECIAL_TOKEN_1
        assert token_2[0] == SPECIAL_TOKEN_2

        # Check that tokens persist after saving and loading
        tok2 = self.test_save_and_load_tokenizer(tokenizer)
        assert tok2.tokenize(SPECIAL_TOKEN_1) == [SPECIAL_TOKEN_1]
        assert tok2.tokenize(SPECIAL_TOKEN_2) == [SPECIAL_TOKEN_2]

    def test_pickle(self):
        tokenizer = self.get_tokenizer()
        bin = pickle.dumps(tokenizer)
        state = tokenizer.to_str()
        del tokenizer
        tokenizer_new = pickle.loads(bin)
        assert tokenizer_new.to_str() == state

    def test_tokenizers_common_properties(self):
        tokenizer = self.get_tokenizer()
        attributes_list = [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ]
        for attr in attributes_list:
            assert hasattr(tokenizer, attr)
            assert hasattr(tokenizer, attr + "_id")

        assert hasattr(tokenizer, "additional_special_tokens")
        assert hasattr(tokenizer, "additional_special_tokens_ids")

        attributes_list = [
            "model_max_length",
            "init_inputs",
            "init_kwargs",
        ]
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            attributes_list += [
                "added_tokens_encoder",
                "added_tokens_decoder",
            ]
        for attr in attributes_list:
            assert hasattr(tokenizer, attr)

    def test_internal_consistency(self):
        if self.name == "smirk-gpe":
            pytest.xfail(
                "smirk-gpe doesn't have a 1:1 mapping between token strings and token ids"
            )

        text = self.get_batch()[0]
        tokenizer = self.get_tokenizer()
        tokens = tokenizer.tokenize(text)
        assert isinstance(tokens, list) and all(isinstance(t, str) for t in tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        ids_encoded = tokenizer.encode(text, add_special_tokens=False)
        assert ids == ids_encoded
        tokens_2 = tokenizer.convert_ids_to_tokens(ids)
        assert isinstance(tokens_2, list) and all(isinstance(t, str) for t in tokens_2)
        assert tokens_2 == tokens
        text_out = tokenizer.decode(ids)
        assert text_out == text

    def test_token_type_ids(self):
        text = self.get_batch()[0]
        out = self.tokenizer(text, return_token_type_ids=True)
        assert 0 in out["token_type_ids"]

    def test_special_tokens_mask(self):
        text = self.get_batch()[0]
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        encode_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        enc_w_special = encode_dict["input_ids"]
        special_tokens_mask = encode_dict["special_tokens_mask"]
        assert len(special_tokens_mask) == len(enc_w_special)
        filtered = [
            x for i, x in enumerate(enc_w_special) if not special_tokens_mask[i]
        ]
        assert filtered == enc

    def test_get_vocab(self):
        tokenizer = self.tokenizer
        vocab = tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(tokenizer) >= len(vocab)
        id_to_token = [
            tokenizer.convert_ids_to_tokens(id) for id in range(len(tokenizer))
        ]
        # Smirk-gpe maps identical "glyphs" to different ids
        assert len(tokenizer) == len(id_to_token)
        assert len(vocab) <= len(id_to_token)

        # Add token
        tokenizer.add_tokens(["ajdkfsfsk"])
        id_to_token = [
            tokenizer.convert_ids_to_tokens(id) for id in range(len(tokenizer))
        ]
        assert len(tokenizer.get_vocab()) <= len(id_to_token)
        assert len(tokenizer) == len(id_to_token)

    def test_conversion_reversible(self):
        tokenizer = self.tokenizer
        vocab = tokenizer.get_vocab()
        for token, id in vocab.items():
            if token == tokenizer.unk_token:
                continue
            assert token == tokenizer.convert_ids_to_tokens(id)
            assert id == tokenizer.convert_tokens_to_ids(token)

    def test_call(self):
        tokenizer = self.tokenizer
        batch = self.get_batch()

        # Test not batched
        enc_1 = tokenizer.encode_plus(batch[0])
        enc_2 = tokenizer(batch[0])
        assert enc_1 == enc_2

        # Test batched
        enc_1 = tokenizer.batch_encode_plus(batch)
        enc_2 = tokenizer(batch)
        assert enc_1 == enc_2

    @require_torch
    def test_torch_encode(self):
        import torch

        self.check_return_tensors("pt", torch.Tensor)

    def test_numpy_encode(self):
        import numpy

        self.check_return_tensors("np", numpy.ndarray)

    def check_return_tensors(self, return_tensors, tensor_type):
        tokenizer = self.get_tokenizer()
        batch = self.get_batch()

        enc = tokenizer(batch[0])
        # Check for single sequence
        assert all(not isinstance(x, tensor_type) for x in enc.values())
        enc = tokenizer(batch[0], return_tensors=return_tensors)
        assert all(isinstance(x, tensor_type) for x in enc.values())
        enc = tokenizer.encode(batch[0], return_tensors=return_tensors)
        assert isinstance(enc, tensor_type)

        # Check for batch
        enc = tokenizer(batch)
        assert all(not isinstance(x, tensor_type) for x in enc.values())
        enc = tokenizer(
            batch,
            return_tensors=return_tensors,
            truncation=True,
            max_length=3,
        )
        assert all(isinstance(x, tensor_type) for x in enc.values())

    def test_padding_max_length(self):
        tokenizer = self.get_tokenizer()
        batch = self.get_batch()

        base_enc = tokenizer.encode(batch[0], padding=False)
        assert tokenizer.pad_token_id not in base_enc
        max_length = len(base_enc) + 10
        padded_enc = tokenizer.encode(
            batch[0], padding="max_length", max_length=max_length
        )
        assert len(padded_enc) == max_length
        assert padded_enc == (base_enc + [tokenizer.pad_token_id] * 10)

        # Repeat for a batch
        base_enc = tokenizer.batch_encode_plus(batch, padding=False)
        max_length = max(len(x) for x in base_enc["input_ids"]) + 10
        padded_batch = tokenizer.batch_encode_plus(
            batch, padding="max_length", max_length=max_length
        )
        assert all(len(x) == max_length for x in padded_batch["input_ids"])

    def test_padding_multiple_of(self):
        tokenizer = self.get_tokenizer()
        empty = tokenizer("", padding=True, pad_to_multiple_of=8)
        assert all(len(v) % 8 == 0 for v in empty.values())
        batch_enc = tokenizer(self.get_batch(), padding="longest", pad_to_multiple_of=8)
        for k, v in batch_enc.items():
            assert all(len(x) % 8 == 0 for x in v)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_padding_side(self, side):
        tokenizer = self.get_tokenizer()
        batch = self.get_batch()
        enc = tokenizer.batch_encode_plus(batch, padding=False)
        assert not all(len(enc["input_ids"][0]) == len(x) for x in enc["input_ids"])
        max_length = max(len(x) for x in enc["input_ids"])

        # Change padding side
        tokenizer.padding_side = side
        padded = tokenizer.batch_encode_plus(batch, padding=True, padding_side=side)
        pad_token_id = tokenizer.pad_token_id
        for bdx in range(len(batch)):
            ids = padded["input_ids"][bdx]
            ref_ids = enc["input_ids"][bdx]
            n = max_length - len(ref_ids)
            if side == "left":
                assert ids == (n * [pad_token_id] + ref_ids)
            else:
                assert ids == (ref_ids + n * [pad_token_id])

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_padding_attention(self, side):
        tokenizer = self.get_tokenizer()
        batch = self.get_batch()
        tokenizer.padding_side = side
        enc = tokenizer.batch_encode_plus(
            batch, padding=False, return_attention_mask=True
        )
        max_length = max(len(x) for x in enc["input_ids"])
        padded = tokenizer.batch_encode_plus(
            batch, padding=True, padding_side=side, return_attention_mask=True
        )
        pad_token_id = tokenizer.pad_token_id
        for bdx in range(len(batch)):
            ids = padded["input_ids"][bdx]
            atten = padded["attention_mask"][bdx]
            ref_ids = enc["input_ids"][bdx]
            ref_attn = enc["attention_mask"][bdx]
            n = max_length - len(ref_ids)
            if side == "left":
                assert ids == (n * [pad_token_id] + ref_ids)
                assert atten == (n * [0] + ref_attn)
            else:
                assert ids == (ref_ids + n * [pad_token_id])
                assert atten == (ref_attn + n * [0])

    def test_truncation(self):
        tokenizer = self.get_tokenizer()
        text = self.get_batch()[0]

        enc = tokenizer.encode(text, add_special_tokens=False)
        trunc_len = len(enc) - 3
        assert trunc_len > 0
        tokenizer.truncation_side = "right"
        trunc = tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=trunc_len
        )
        assert len(trunc) == trunc_len
        assert trunc == enc[:trunc_len]

        tokenizer.truncation_side = "left"
        trunc = tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=trunc_len
        )
        assert len(trunc) == trunc_len
        assert trunc == enc[-trunc_len:]

    def test_batch_encode_plus_batch_sequence_length(self):
        tokenizer = self.tokenizer
        batch = self.get_batch()
        for text in batch:
            assert tokenizer.pad_token not in text

        enc = [tokenizer.encode_plus(text) for text in batch]
        enc_batch = tokenizer.batch_encode_plus(batch, padding=False)
        assert isinstance(enc_batch, Mapping) and all(
            isinstance(x, list) for x in enc_batch.values()
        )
        assert enc == dol2lod(enc_batch)

        max_length = max([len(x["input_ids"]) for x in enc])
        enc_padded = [
            tokenizer.encode_plus(text, max_length=max_length, padding="max_length")
            for text in batch
        ]
        env_batch_padded = tokenizer.batch_encode_plus(batch, padding=True)
        assert enc_padded == dol2lod(env_batch_padded)

        # Check that longest doesn't change with max length
        enc_batch_padded_2 = tokenizer.batch_encode_plus(
            batch, max_length=max_length + 10, padding="longest"
        )
        assert dol2lod(enc_batch_padded_2) == dol2lod(env_batch_padded)

        # Check that no padding is doesn't change with max length
        enc_batch_no_padding = tokenizer.batch_encode_plus(batch, padding=False)
        enc_batch_no_padding_2 = tokenizer.batch_encode_plus(
            batch, max_length=max_length + 10, padding=False
        )
        assert dol2lod(enc_batch_no_padding) == dol2lod(enc_batch_no_padding_2)

    def test_convert_tokens_to_string_format(self):
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize("CC[C@@H](C)C", add_special_tokens=False)
        for token in tokens:
            assert isinstance(token, str)
        string = tokenizer.convert_tokens_to_string(tokens)
        assert isinstance(string, str)

    def test_offsets_mapping(self):
        tokenizer = self.tokenizer
        text = self.get_batch()[0]
        tokens_with_offsets = tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            add_special_tokens=True,
        )
        added_tokens = tokenizer.num_special_tokens_to_add(False)
        offsets = tokens_with_offsets["offset_mapping"]
        assert len(offsets) == len(tokens_with_offsets["input_ids"])
        assert sum(tokens_with_offsets["special_tokens_mask"]) == added_tokens


def dol2lod(d: dict[str, list]) -> list[dict]:
    "Convert dict of lists to list of dicts"
    return [dict(zip(d, t)) for t in zip(*d.values())]
