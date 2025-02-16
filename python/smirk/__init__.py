import os
from importlib.resources import files
from typing import List, Optional, Union

from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import (
    AddedToken,
    PaddingStrategy,
    TextInput,
    TruncationStrategy,
)
from transformers.tokenization_utils_fast import TOKENIZER_FILE
from transformers.utils import add_code_sample_docstrings

from . import smirk as rs_smirk

SPECIAL_TOKENS = {
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
}
""" Default Special tokens used by the :py:class:`SmirkTokenizerFast`
and :py:func:`SmirkSelfiesFast` tokenizers.
"""


class SmirkTokenizerFast(PreTrainedTokenizerBase):
    def __init__(self, tokenizer_file: Optional[os.PathLike] = None, **kwargs):
        """
        A Chemically-Complete Tokenizer for OpenSMILES

        :param os.PathLike vocab_file: Path to a JSON encoded vocabulary mapping tokens to ids.
        :param tokenizer_file: Path to a JSON serialize SmirkTokenizerFast tokenizers
        :param str template: A :py:class:`post-processing template<tokenizers.processors.TemplateProcessing>`.
            Defaults to no post-processing. For a BERT-like template, use: :code:`[CLS] $0 [SEP]`.
        :param bool add_special_tokens: If true, adds the :py:data:`default special tokens<SPECIAL_TOKENS>` to the vocabulary.
            Defaults to true.
        :param kwargs: Additional kwargs are passed to :py:class:`transformers.PreTrainedTokenizerFast`
        """
        # Create SmirkTokenizer
        default_vocab_file = files("smirk").joinpath("vocab_smiles.json")
        if tokenizer := kwargs.pop("tokenizer", None):
            tokenizer = tokenizer
        elif tokenizer_file:
            tokenizer = rs_smirk.SmirkTokenizer.from_file(str(tokenizer_file))
            kwargs["tokenizer_file"] = str(tokenizer_file)
        elif vocab_file := kwargs.pop("vocab_file", default_vocab_file):
            tokenizer = rs_smirk.SmirkTokenizer.from_vocab(str(vocab_file))
            kwargs["vocab_file"] = str(vocab_file)
        else:
            tokenizer = rs_smirk.SmirkTokenizer()

        self._tokenizer = tokenizer
        super().__init__(**kwargs)

        # Check for unsupported features
        assert self.split_special_tokens is False
        assert self.chat_template is None

        if kwargs.pop("add_special_tokens", True):
            self.add_special_tokens(SPECIAL_TOKENS)

        if template := kwargs.get("template", None):
            tokenizer.post_processor = template

    def __len__(self) -> int:
        """Size of the full vocabulary including added tokens"""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def __repr__(self):
        return self.__class__.__name__

    def is_fast(self):
        return True

    def to_str(self) -> str:
        """Serialize the tokenizer into a JSON string"""
        return self._tokenizer.to_str()

    def get_vocab(self) -> dict[str, int]:
        """Returns the vocabulary of the tokenzier as a dictionary"""
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> dict[str, int]:
        """The tokenzier's vocabulary"""
        return self.get_vocab()

    @property
    def vocab_size(self):
        """The size of the vocabulary without the added tokens"""
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    @property
    def added_tokens_decoder(self) -> dict[int, AddedToken]:
        """Mapping from id to :py:class:`AddedToken` for the added tokens"""
        return {
            id: AddedToken(content)
            for id, content in self._tokenizer.get_added_tokens_decoder().items()
        }

    @property
    def added_tokens_encoder(self) -> dict[str, int]:
        """Mapping from added token to token id"""
        return {
            content: id
            for id, content in self._tokenizer.get_added_tokens_decoder().items()
        }

    def convert_ids_to_tokens(
        self, index: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        """Decode a token id, or a list of ids, into their token(s)"""
        if isinstance(index, int):
            return self._tokenizer.id_to_token(index)
        return [self._tokenizer.id_to_token(i) for i in index]

    def convert_tokens_to_ids(
        self, token: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        """Convert a token, or list of tokens, into their token id(s)"""
        if isinstance(token, str):
            return self._tokenizer.token_to_id(token)
        return [self._tokenizer.token_to_id(t) for t in token]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Joins a list of tokens into a string."""
        return "".join(tokens)

    def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
        special_tokens: bool = False,
    ) -> int:
        # Normalize to AddedTokens
        new_tokens = [
            (
                AddedToken(token, special=special_tokens)
                if isinstance(token, str)
                else token
            )
            for token in new_tokens
        ]
        return self._tokenizer.add_tokens(new_tokens)

    def batch_decode_plus(self, ids, **kwargs) -> list[str]:
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)
        return self._tokenizer.decode_batch(
            ids, skip_special_tokens=skip_special_tokens
        )

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return len(self.build_inputs_with_special_tokens([], [] if pair else None))

    def __check_encode_kwargs(self, kwargs):
        assert (
            kwargs.pop("return_overflowing_tokens", False) is False
        ), "Not implemented"
        assert kwargs.pop("split_special_tokens", False) is False, "Not implemented"
        assert kwargs.pop("is_split_into_words", False) is False, "Not implemented"

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: List[TextInput],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = False,
        return_attention_mask: Optional[bool] = True,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        self.__check_encode_kwargs(kwargs)

        # Set the tokenizer's padding and truncation strategy
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        encoding = self._tokenizer.encode_batch(
            batch_text_or_text_pairs, add_special_tokens=add_special_tokens
        )

        return self._convert_encoding(
            encoding,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            return_tensors=return_tensors,
        )

    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
    ):
        """Configure truncation and padding options for the tokenizer"""
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            self._tokenizer.no_truncation()
        else:
            target = {
                "strategy": truncation_strategy.value,
                "max_length": max_length or 512,
                "stride": stride,
                "direction": self.truncation_side,
            }
            target = {k: v for k, v in target.items() if v is not None}
            self._tokenizer.with_truncation(**target)

        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            self._tokenizer.no_padding()
        else:
            length = (
                max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            )
            target = {
                "length": length,
                "pad_to_multiple_of": pad_to_multiple_of,
                "direction": self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_type_id": self.pad_token_type_id,
            }
            target = {k: v for k, v in target.items() if v is not None}
            self._tokenizer.with_padding(**target)

    def _encode_plus(
        self,
        text: Union[TextInput],
        text_pair: Optional = None,
        return_tensors: Optional[bool] = None,
        **kwargs,
    ) -> BatchEncoding:
        assert text_pair is None, "Not implemented"
        batched_output = self._batch_encode_plus(
            [text],
            return_tensors=return_tensors,
            **kwargs,
        )
        if return_tensors is None:
            return BatchEncoding(
                {k: v[0] for k, v in batched_output.items()},
                batched_output.encodings,
                n_sequences=1,
            )
        return batched_output

    def _convert_encoding(
        self,
        encoding: List[dict],
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding:
        # Convert encoding to dict
        data = {"input_ids": [x["input_ids"] for x in encoding]}
        if return_token_type_ids:
            data["token_type_ids"] = [x["token_type_ids"] for x in encoding]
        if return_attention_mask or return_attention_mask is None:
            data["attention_mask"] = [x["attention_mask"] for x in encoding]
        if return_special_tokens_mask:
            data["special_tokens_mask"] = [x["special_tokens_mask"] for x in encoding]
        if return_offsets_mapping:
            data["offset_mapping"] = [x["offsets"] for x in encoding]
        if return_length:
            data["length"] = [len(x["input_ids"]) for x in encoding]
        batch = BatchEncoding(
            data,
            encoding,
            n_sequences=len(encoding),
            tensor_type=return_tensors,
        )
        return batch

    def _decode(self, token_ids, **kwargs):
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def tokenize(self, text: str, add_special_tokens=False) -> list[str]:
        """Converts a string into a sequence of tokens"""
        return self._tokenizer.tokenize(text, add_special_tokens)

    @property
    def post_processor(self):
        """
        Returns the JSON serialization of the post-processor

        .. tip:: This is not part of :py:class:`transformers.PreTrainedTokenizerFast`'s API.

        """
        return self._tokenizer.post_processor

    def _save_pretrained(
        self,
        save_directory,
        file_names,
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> tuple[str]:
        assert legacy_format is None or not legacy_format
        tokenizer_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE,
        )
        self._tokenizer.save(tokenizer_file)
        return file_names + (tokenizer_file,)


# Register with AutoTokenizer
AutoTokenizer.register("SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast)


def SmirkSelfiesFast(
    vocab_file: Optional[os.PathLike] = None, unk_token="[UNK]", **kwargs
) -> PreTrainedTokenizerFast:
    """Instantiate a Chemically-Consistent tokenizer for SELFIES

    :param os.PathLike vocab_file: The vocabulary for the :py:class:`tokenizers.models.WordLevel` model.
        The default vocabulary includes all possible SELFIES tokens
    :param str unk_token: The unknown token to be used.
    :param bool add_special_tokens: if true, adds the :py:data:`default special tokens<SPECIAL_TOKENS>`
        to the vocabulary. Defaults to true.
    :param kwargs: Additional kwargs are passed to :py:class:`transformers.PreTrainedTokenizerFast`

    """
    import json
    from importlib.resources import files

    from tokenizers import Regex, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.normalizers import Strip
    from tokenizers.pre_tokenizers import Sequence, Split

    vocab_file = vocab_file or files("smirk").joinpath("vocab_selfies.json")
    assert isinstance(vocab_file, os.PathLike)
    with open(vocab_file, "r") as fid:
        vocab = json.load(fid)

    tok = Tokenizer(WordLevel(vocab, unk_token))
    # Regex generated using `opt/build_vocab.py -f smiles -t regex`
    regex = (
        r"Branch|Ring|A[c|g|l|m|r|s|t|u]|B[a|e|h|i|k|r]?|"
        r"C[a|d|e|f|l|m|n|o|r|s|u]?|D[b|s|y]|E[r|s|u]|F[e|l|m|r]?|"
        r"G[a|d|e]|H[e|f|g|o|s]?|I[n|r]?|Kr?|L[a|i|r|u|v]|"
        r"M[c|d|g|n|o|t]|N[a|b|d|e|h|i|o|p]?|O[g|s]?|P[a|b|d|m|o|r|t|u]?|"
        r"R[a|b|e|f|g|h|n|u]|S[b|c|e|g|i|m|n|r]?|T[a|b|c|e|h|i|l|m|s]|"
        r"U|V|W|Xe|Yb?|Z[n|r]|[\.\-=\#\$:/\\\+\-]|\d|@|@@"
    )

    tok.pre_tokenizer = Sequence(
        [
            Split(Regex(r"\[|]"), behavior="removed"),  # Strip Brackets
            Split(Regex(regex), behavior="isolated"),  # Tokenize
        ]
    )
    tok.normalizer = Strip()
    add_special_tokens = kwargs.pop("add_special_tokens", True)
    tok_tf = PreTrainedTokenizerFast(tokenizer_object=tok, **kwargs)

    if add_special_tokens:
        tok_tf.add_special_tokens(SPECIAL_TOKENS)

    return tok_tf


def train_gpe(
    files: list[str],
    ref: Optional[SmirkTokenizerFast] = None,
    min_frequency: int = 0,
    vocab_size: int = 1024,
    merge_brackets: bool = False,
    split_structure: bool = True,
) -> SmirkTokenizerFast:
    """
    Train a Smirk-GPE Tokenizer from a corpus of SMILES encodings.

    :param files: List of files containing the corpus to train the tokenizer on
    :param ref: The initial tokenizer to start from when training. Defaults to `SmirkTokenizerFast()`
        This determines the initial vocabulary and identity of the unknown token
    :param min_frequency: Minimum count for a pair to be considered for a merge
    :param vocab_size: The target size of the final vocabulary
    :param merge_brackets: If true, merges with brackets (`[` or `]`) are allowed
    :param split_structure: If true, will split SMILES encoding on structural elements,
        before considering merges (i.e. merges across structural elements are not allowed)
    """
    ref = ref or SmirkTokenizerFast()
    tokenizer = ref._tokenizer.train(
        files,
        min_frequency=min_frequency,
        vocab_size=vocab_size,
        merge_brackets=merge_brackets,
        split_structure=split_structure,
    )
    return SmirkTokenizerFast(tokenizer=tokenizer)
