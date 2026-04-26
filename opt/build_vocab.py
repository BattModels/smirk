#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import defaultdict

# fmt: off
ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
# fmt: on

ALIPHATIC_ORGANIC = ["B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
AROMATIC_ORGANIC = ["b", "c", "n", "o", "s", "p"]
AROMATIC_SYMBOLS = ["b", "c", "n", "o", "p", "s", "se", "as"]
CHIRAL = ["@", "@@"]
CHIRAL_CONFIG = ["TH", "AL", "SP", "TB", "OH"]
BONDS = [".", "-", "=", "#", "$", ":", "/", "\\"]
DIGITS = [str(x) for x in range(10)]
BIGSMILES_EXTRA_TOKENS = ["{", "}", ",", ";", "<", ">"]
BIGSMILES_LABEL = r"[A-Z][A-Za-z0-9']*"


def build_smiles_alphabet():
    vocab = set()

    # Organic Subset
    vocab.update(ALIPHATIC_ORGANIC)
    vocab.update(AROMATIC_ORGANIC)

    # Bracket Atoms
    vocab.update(["[", "]"])  # brackets
    vocab.update(DIGITS)
    vocab.update(ELEMENT_SYMBOLS)
    vocab.update(AROMATIC_SYMBOLS)
    vocab.add("*")
    vocab.update(CHIRAL)
    vocab.update([f"@{s}" for s in CHIRAL_CONFIG])

    # Charge
    vocab.update(["+", "-"])

    # Bonds and Chains
    vocab.update(BONDS)
    vocab.add("%")  # Ring bond
    vocab.update(["(", ")"])  # Branches
    vocab.add(".")  # non-bond

    return vocab


def const_str(name, regex, comment=None, public=False, separator="|", wrapper=None):
    out = f"const {name}: &'static str ="
    if isinstance(regex, list):
        if wrapper:
            out += f" {wrapper}!(concat!(\n"
        else:
            out += " concat!(\n"
        for idx, r in enumerate(regex):
            out += f'    r"{r}'
            if idx < len(regex) - 1:
                out += separator
            out += '",\n'

        if wrapper:
            out += "));"
        else:
            out += ");"
    else:
        out += f' r"{regex}";'

    if public:
        out = "pub " + out

    if comment:
        out = f"// {comment}\n{out}"
    return out


def merge_tokens(tokens):
    branches = defaultdict(set)
    for token in tokens:
        assert len(token) in [1, 2]
        if len(token) == 1:
            branches[token[0]] |= {None}
        else:
            branches[token[0]] |= set(token[1])

    out = []
    for leader, tail in branches.items():
        if None in tail:
            tail -= {None}
            if len(tail) == 0:
                cr = leader
            elif len(tail) == 1:
                cr = f"{leader}{tail.pop()}?"
            else:
                cr = f"{leader}[{'|'.join(sorted(tail))}]?"
        else:
            if len(tail) == 1:
                cr = f"{leader}{tail.pop()}"
            else:
                cr = f"{leader}[{'|'.join(sorted(tail))}]"

        out.append(cr)
    return sorted(out)


def merge_tokens_grouped(tokens):
    branches = defaultdict(set)
    for token in tokens:
        assert len(token) in [1, 2]
        if len(token) == 1:
            branches[token[0]] |= {None}
        else:
            branches[token[0]].add(token[1])

    out = []
    for leader, tail in branches.items():
        if None in tail:
            tail -= {None}
            if len(tail) == 0:
                cr = leader
            elif len(tail) == 1:
                cr = f"{leader}{tail.pop()}?"
            else:
                cr = f"{leader}(?:{'|'.join(sorted(tail))})?"
        else:
            if len(tail) == 1:
                cr = f"{leader}{tail.pop()}"
            else:
                cr = f"{leader}(?:{'|'.join(sorted(tail))})"

        out.append(cr)
    return sorted(out)


def match_chars(chars: list[str]):
    """Combine chars into a regex: `[chars]`, adding escapes as needed"""
    return "[" + re.escape("".join(chars)) + "]"


def build_smiles_pretokenizer():
    print(
        const_str(
            "MATCH_OUTER",
            [
                "|".join(merge_tokens(ALIPHATIC_ORGANIC)),
                "|".join(merge_tokens(AROMATIC_ORGANIC)),
                r"\*",  # Wildcard
                match_chars(BONDS),
                r"\d|%",  # Ring Bond
                r"\(|\)",  # Branches
                r"\[.*?]",
            ],
            public=True,
        )
    )
    print(
        const_str(
            "BRACKETED_SYMBOL",
            [
                *merge_tokens(ELEMENT_SYMBOLS),
                *merge_tokens(AROMATIC_SYMBOLS),
                r"\*",
            ],
        )
    )
    chiral = "|".join(merge_tokens(CHIRAL_CONFIG))
    print(const_str("CHIRAL", f"@(?:@|{chiral})?"))


def build_bigsmiles_pretokenizer():
    print(
        const_str(
            "BRACKETED_SYMBOL",
            [
                *merge_tokens_grouped(ELEMENT_SYMBOLS),
                *merge_tokens_grouped(AROMATIC_SYMBOLS),
                r"\*",
            ],
        )
    )
    chiral = "|".join(merge_tokens_grouped(CHIRAL_CONFIG))
    print(const_str("CHIRAL", f"@(?:@|{chiral})?"))
    print(
        const_str(
            "MATCH_OUTER_BIGSMILES",
            [
                "|".join(merge_tokens(ALIPHATIC_ORGANIC)),  # organic subset elements
                "|".join(merge_tokens(AROMATIC_ORGANIC)),  # aromatic organic subset
                r"\*",  # wildcard
                match_chars(BONDS),  # bonds
                r"\d|%",  # ring closures
                r"\(|\)",  # branches
                r"\{|\}",  # stochastic object delimiters
                r",|;",  # repeat unit separator and end group separator
                BIGSMILES_LABEL,  # fragment and abstract spec labels
                r"\[(?:[^\[\]]+|\[[^\[\]]*\])*\]",  # bracketed atoms/descriptors
            ],
            public=True,
        )
    )
    print(
        const_str(
            "MATCH_INNER_BIGSMILES",
            [
                r"^(?:",
                r"",
                r"|",
                r"(\$|<|>)(\d+)?",
                r"|",
                rf"(\$|<|>)(\d+)?(\[)(\$|<|>|{BIGSMILES_LABEL})(\d+)?(\])(\d+)",
                r"|",
                r"(#)([!-~]+)",
                r"|",
                r"(\d+)?",
                r"({BRACKETED_SYMBOL})",
                r"(?:({CHIRAL})(\d{{1,2}})?)?",
                r"(?:(H)(\d)?)?",
                r"(?:([+-]{{1,2}})(\d{{1,2}})?)?",
                r"(?:(:)(\d+))?",
                r")$",
            ],
            public=True,
            separator="",
            wrapper="formatcp",
        )
    )
    print(
        const_str(
            "BONDING_DESCRIPTOR",
            [
                r"(\$|<|>)",  # descriptor type
                r"(\d+)?",  # optional index
            ],
            public=True,
            separator="",
        )
    )
    print(
        const_str(
            "LADDER_BONDING_DESCRIPTOR",
            [
                r"(\$|<|>)",  # outer descriptor type
                r"(\d+)?",  # outer descriptor id
                rf"(\[)(\$|<|>|{BIGSMILES_LABEL})(\d+)?(\])",  # inner descriptor
                r"(\d+)",  # group id
            ],
            public=True,
            separator="",
        )
    )
    print(const_str("FRAGMENT_REFERENCE", r"(#)([!-~]+)", public=True))


def build_selfies_pretokenizer():
    print(
        "|".join(
            [
                "Branch",
                "Ring",
                *merge_tokens(ELEMENT_SYMBOLS),
                match_chars([*BONDS, "+", "-"]),
                r"\d",
                *CHIRAL,
            ]
        )
    )


def build_selfies_alphabet():
    vocab = set()
    vocab.update(ELEMENT_SYMBOLS)
    vocab.update(DIGITS)
    vocab.update(BONDS)
    vocab.update(CHIRAL)
    vocab.update(["+", "-", "Branch", "Ring"])

    return vocab


def build_bigsmiles_alphabet():
    vocab = build_smiles_alphabet()
    vocab.update(BIGSMILES_EXTRA_TOKENS)

    return vocab


def build_vocab(tokens: set):
    tokens = ["[UNK]", *sorted(tokens)]
    return {token: id for id, token in enumerate(tokens)}


def build_bigsmiles_vocab():
    vocab = build_vocab(build_smiles_alphabet())
    for token in BIGSMILES_EXTRA_TOKENS:
        vocab[token] = len(vocab)
    return vocab


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-t", "--type", choices=["vocab", "regex"], default="vocab")
    p.add_argument(
        "-f", "--format", choices=["smiles", "bigsmiles", "selfies"], default="smiles"
    )
    p.add_argument("output", type=argparse.FileType("w"), default=sys.stdout, nargs="?")
    args = p.parse_args()

    if args.type == "vocab":
        if args.format == "bigsmiles":
            vocab = build_bigsmiles_vocab()
        else:
            if args.format == "smiles":
                alphabet = build_smiles_alphabet()
            elif args.format == "selfies":
                alphabet = build_selfies_alphabet()
            else:
                # Argparse should catch this sooner
                raise RuntimeError("Unknown format", args.format)

            # Convert enumerated glyphs to a vocab
            vocab = build_vocab(alphabet)

        json.dump(vocab, args.output, indent=4)
        args.output.write("\n")

    elif args.type == "regex":
        if args.format == "smiles":
            build_smiles_pretokenizer()
        elif args.format == "bigsmiles":
            build_bigsmiles_pretokenizer()
        elif args.format == "selfies":
            build_selfies_pretokenizer()
        else:
            # Argparse should catch this sooner
            raise RuntimeError("Unknown format", args.format)
