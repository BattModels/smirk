import argparse

from . import train_gpe


def __cli_parser():
    p = argparse.ArgumentParser(
        "python -m smirk.cli",
        description="Train a smirk-gpe tokenizer from a corpus of SMILES encodings",
    )
    p.add_argument("files", nargs="+")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument(
        "--merge-brackets",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Allow merges with bracket (`[` or `]`) tokens",
    )
    p.add_argument(
        "--split-structure",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Split SMILES on structure before training",
    )
    p.add_argument(
        "-o",
        "--output",
        default=".",
        type=str,
        help="directory where trained smirk-gpe model is saved",
    )
    return p


def cli(argv=None):
    parser = __cli_parser()
    args = parser.parse_args(argv)
    tok_gpe = train_gpe(
        args.files,
        vocab_size=args.vocab_size,
        merge_brackets=args.merge_brackets,
        split_structure=args.split_structure,
    )
    tok_gpe.save_pretrained(args.output)


if __name__ == "__main__":
    cli()
