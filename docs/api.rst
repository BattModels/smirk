API Reference
=============

.. autoclass:: smirk.SmirkTokenizerFast
   :special-members:
   :members:

.. py:method:: smirk.SmirkTokenizerFast.__call__(text: str, ...)

   Primary method for tokenizing text

   .. seealso:: :py:meth:`transformers.PreTrainedTokenizerBase.__call__` for the 🤗 documentation

   .. attention:: The following features are not supported:

      * Passing in pairs of text. i.e. :code:`text_pair` or related arguments.

      * Does not support :code:`split_special_tokens`.

      * Pre-tokenizer inputs, i.e. :code:`is_split_into_words`

      * Returning overflowing tokens (i.e. :code:`return_overflowing_tokens`)

.. py:method:: smirk.SmirkTokenizerFast.decode(list[int] | Tensor, skip_special_tokens: bool = False, ...) -> str

   Primary method for decoding token ids back into text.

   .. seealso:: :py:meth:`transformers.PreTrainedTokenizerBase.decode` for the 🤗 documentation

.. py:method:: smirk.SmirkTokenizerFast.batch_decode(list[list[int]] | Tensor, skip_special_tokens: bool = False, ...) -> list[str]

   Primary method for decoding batches of token ids back into text.

   .. seealso:: :py:meth:`transformers.PreTrainedTokenizerBase.batch_decode` for the 🤗 documentation

.. autofunction:: smirk.SmirkSelfiesFast

.. autodata:: smirk.SPECIAL_TOKENS

.. autofunction:: smirk.train_gpe

Command Line Utility
--------------------
.. argparse::
    :module: smirk.cli
    :func: __cli_parser
