# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/BattModels/smirk)

### Added

- Started a changelog ([#2](https://github.com/BattModels/smirk/pull/2))

### Changed

- Bumped PyO3, tokenizers and dict_derive dependencies ([#2](https://github.com/BattModels/smirk/pull/2))
- Switched to uv for CI/pre-commit workflows ([#2](https://github.com/BattModels/smirk/pull/2))

### Breaking

- Increased minimum python version to 3.9 ([#2](https://github.com/BattModels/smirk/pull/2))

### Fixed

- Mark version as dynamic in pyproject ([#2](https://github.com/BattModels/smirk/pull/2))
- The vocab for `SmirkSelfiesFast` can now be set by passing a `vocab_file` ([#3](https://github.com/BattModels/smirk/pull/3))
- The default unknown token for the rust `SmirkTokenzier` is now `[UNK]` matching the python default ([#3](https://github.com/BattModels/smirk/pull/3))

### Removed

- Renamed `SmirkSelfiesFast` `vocab` parameter to `vocab_file` ([#3](https://github.com/BattModels/smirk/pull/3))
- Default for `--split-structure` is now `True` for `smirk.cli` and `train_gpe` ([#3](https://github.com/BattModels/smirk/pull/3))
- Moved GPE training from a method (`SmirkTokenizerFast.train`) to a function (`smirk.train_gpe`) ([#3](https://github.com/BattModels/smirk/pull/3))

## [v0.1.1](https://github.com/BattModels/smirk/tree/v0.1.1) - 2024-12-09

Preprint v2 posted: [arXiv:2409.15370v2](https://arxiv.org/abs/2409.15370v2)

### Added

- Added support for post-processing templates to `SmirkTokenizerFast` ([#1](https://github.com/BattModels/smirk/pull/1))
- Registered smirk with transformer's AutoTokenizer ([#1](https://github.com/BattModels/smirk/pull/1))
- Added `vocab`, `convert_ids_to_tokens` and `convert_tokens_to_ids` methods ([#1](https://github.com/BattModels/smirk/pull/1))
- Added support for truncating and padding during tokenization ([#1](https://github.com/BattModels/smirk/pull/1))

### Fixed

- Fixed CI to install test dependencies ([#1](https://github.com/BattModels/smirk/pull/1))

## [v0.1.0](https://github.com/BattModels/smirk/tree/v0.1.0) - 2024-09-11

Preprint posted: [arXiv:2409.15370v1](https://arxiv.org/abs/2409.15370v1)

### Added

- Initial tagged version of smirk
