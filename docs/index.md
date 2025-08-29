# Smirk: A Tokenizer for OpenSMILES

<div align="center" display="flex" >

![GitHub License](https://img.shields.io/github/license/BattModels/smirk)
<a href="https://arxiv.org/abs/2409.15370">![arXiv:2409.15370](https://img.shields.io/badge/cs.LG-2409.15370-b31b1b?style=flat&amp;logo=arxiv&amp;logoColor=red)</a>

</div>

Smirk is a chemistry-specific tokenizer that provides complete coverage of the [OpenSMILES]
specification, that is built using Rust 🦀 and [HuggingFace's tokenizers][Tokenizers] 🤗.
Installation is easy, and Smirk works out-of-the-box with the [HuggingFace] ecosystem.

Check out [Getting Started](smirk_demo.ipynb) to see `smirk` in action, or [read the paper][paper] to learn
about tokenization for molecular foundation models.

## Why Smirk?

Molecular Foundation Models are demonstrating impressive performance, but current models use tokenizers
that fail to represent *all* of chemistry, inherently limiting their performance.
`smirk` fixes this by tokenizing SMILES encodings all the way down to their constituent elements.
Enabling complete coverage of [OpenSMILES] with a vocabulary of 167 tokens.

[OpenSMILES]: http://opensmiles.org/
[paper]: https://arxiv.org/abs/2409.15370
[HuggingFace]: https://huggingface.co/docs
[Tokenizers]: https://huggingface.co/docs/tokenizers


```{eval-rst}
.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   smirk_demo
   api
   developer
   changelog
```
