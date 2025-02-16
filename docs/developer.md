# Developer Documentation

## Installing from Source

1. Install [rust]
2. Pip install `pip install --no-binary smirk`

## Development Environment

1. Install [uv]
2. Build smirk: `uv run maturin develop`
3. Use it: `uv run python -c 'import smirk'`

[rust]: https://www.rust-lang.org/tools/install
[uv]: https://docs.astral.sh/uv/

## Running Tests

To test at the rust level: `cargo test`. To run the python tests:

```shell
uv run maturin develop     # Ensure the rust library is up to date
uv run pytest              # Runs the testsuite
```

## Building Documentation

To build the documentation:
```shell
uv run --group docs sphinx-autobuild docs build/html
```

### Editing Notebooks

1. Launch the notebook in an isolated environment:
```shell
uv run --isolated --no-default-groups --with jupyter,pip jupyter notebook
```
2. Make some changes and save the notebook
3. **Purge notebook outputs** `uv run pre-commit run --all`
4. Commit your changes

```{important}
Please avoid committing notebooks with saved cell outputs
```

## Linting

:All in one: `uv run pre-commit run --all`
:Rust:
   Check syntax with `cargo check` and formatting with `cargo fmt`
:Python:
   Check syntax with `ruff --fix --select=I` and formatting with `ruff format`
