use super::split_bigsmiles::{MATCH_INNER_BIGSMILES, MATCH_OUTER_BIGSMILES};
use once_cell::sync::Lazy;
use regex::{Match, Regex};
use serde::de::Visitor;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use std::fmt;
use tokenizers::tokenizer::pattern::Pattern;
use tokenizers::tokenizer::{
    Offsets, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};

#[derive(Clone)]
pub struct BigSmirkPreTokenizer {
    outer: Regex,
    inner: Regex,
}

impl BigSmirkPreTokenizer {
    pub fn new(outer: &str, inner: &str) -> Self {
        Self {
            outer: Regex::new(&outer).unwrap(),
            inner: Regex::new(&inner).unwrap(),
        }
    }

    pub fn split(&self, text: &String) -> Vec<String> {
        self.find_matches(text)
            .unwrap()
            .into_iter()
            .map(|(offset, _)| text.get(offset.0..offset.1).unwrap().to_owned())
            .filter(|tok| !tok.is_empty())
            .collect()
    }
}

impl Default for BigSmirkPreTokenizer {
    fn default() -> Self {
        BigSmirkPreTokenizer::new(MATCH_OUTER_BIGSMILES, MATCH_INNER_BIGSMILES)
    }
}

impl PartialEq for BigSmirkPreTokenizer {
    fn eq(&self, other: &Self) -> bool {
        self.outer.as_str() == other.outer.as_str() && self.inner.as_str() == other.inner.as_str()
    }
}

impl fmt::Debug for BigSmirkPreTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BigSmirkPreTokenizer")
            .field("outer", &format_args!("'{}'", &self.outer.as_str()))
            .field("inner", &format_args!("'{}'", &self.inner.as_str()))
            .finish()
    }
}

impl Serialize for BigSmirkPreTokenizer {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("BigSmirkPreTokenizer", 3)?;
        state.serialize_field("type", "BigSmirkPreTokenizer")?;
        state.serialize_field("outer", self.outer.as_str())?;
        state.serialize_field("inner", self.inner.as_str())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for BigSmirkPreTokenizer {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "BigSmirkPreTokenizer",
            &["type", "outer", "inner"],
            BigSmirkPreTokenizerVisitor,
        )
    }
}

struct BigSmirkPreTokenizerVisitor;
impl<'de> Visitor<'de> for BigSmirkPreTokenizerVisitor {
    type Value = BigSmirkPreTokenizer;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "struct BigSmirkPreTokenizer with type field")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut outer: Option<String> = None;
        let mut inner: Option<String> = None;
        let mut type_field: Option<String> = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "type" => {
                    type_field = Some(map.next_value()?);
                }
                "outer" => {
                    if let Some(x) = map.next_value()? {
                        outer = Some(x);
                    }
                }
                "inner" => {
                    if let Some(x) = map.next_value()? {
                        inner = Some(x);
                    }
                }
                _ => {
                    let _: serde::de::IgnoredAny = map.next_value()?;
                }
            }
        }
        match type_field.as_deref() {
            Some("BigSmirkPreTokenizer") => {}
            _ => {
                return Err(serde::de::Error::custom(
                    "Missing or invalid type field for BigSmirkPreTokenizer",
                ));
            }
        }
        Ok(BigSmirkPreTokenizer::new(
            outer.expect("Missing `outer`").as_str(),
            inner.expect("Missing `inner`").as_str(),
        ))
    }
}

impl PreTokenizer for BigSmirkPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(self.to_owned(), SplitDelimiterBehavior::Isolated))
    }
}

fn append_split(splits: &mut Vec<(Offsets, bool)>, prev: &mut usize, m: Match, offset: usize) {
    let start = m.start() + offset;
    let end = m.end() + offset;
    if *prev != start {
        splits.push(((*prev, start), false));
    }
    splits.push(((start, end), true));
    *prev = end;
}

impl Pattern for BigSmirkPreTokenizer {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let mut splits = Vec::with_capacity(inside.len());
        let mut prev = 0;
        let n_inner_groups = self.inner.captures_len();
        static IS_NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\d+$").unwrap());
        static IS_BONDING_DESC: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[\$<>]$").unwrap());
        for m_outer in self.outer.find_iter(inside) {
            // Check for Brackets
            if m_outer.as_str().starts_with("[") {
                // Record opening [
                splits.push(((m_outer.start(), m_outer.start() + 1), true));
                prev = m_outer.start() + 1;

                // Record contents between brackets
                let bracketed = &inside[(m_outer.start() + 1)..(m_outer.end() - 1)];

                // Try to match with inner pattern
                if let Some(capture) = self.inner.captures(&bracketed) {
                    // Unpack bracketed atoms
                    for i in 1..n_inner_groups {
                        if let Some(m) = capture.get(i) {
                            let matched_str = m.as_str();
                            if matched_str.is_empty() {
                                continue;
                            }
                            if IS_NUMBER.is_match(matched_str) {
                                // Tokenize numbers as digits
                                for d in m.range() {
                                    let s = d + m_outer.start() + 1;
                                    splits.push(((s, s + 1), true));
                                    prev = s + 1;
                                }
                            } else if IS_BONDING_DESC.is_match(matched_str) {
                                // Bonding descriptor ($, <, >) - keep as single token
                                append_split(&mut splits, &mut prev, m, m_outer.start() + 1)
                            } else {
                                append_split(&mut splits, &mut prev, m, m_outer.start() + 1)
                            }
                        }
                    }
                }

                // Check for trailing unmatched characters within the brackets
                if prev != (m_outer.end() - 1) {
                    splits.push(((prev, m_outer.end() - 1), false));
                    prev = m_outer.end() - 1;
                }

                // Record closing ]
                assert!(m_outer.as_str().ends_with("]"));
                splits.push(((prev, m_outer.end()), true));
                prev = m_outer.end();
            } else {
                append_split(&mut splits, &mut prev, m_outer, 0);
            }
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false));
        }
        Ok(splits)
    }
}

#[cfg(test)]
pub mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;
    use crate::test_utils::check_serde;
    use tokenizers::tokenizer::{OffsetReferential, OffsetType};

    #[test]
    fn serialize_default() {
        let default = BigSmirkPreTokenizer::default();
        check_serde(&default);
    }

    #[test]
    fn serialize_pretok() {
        let pretok = BigSmirkPreTokenizer::new(r".|\[.*?]", ".");
        check_serde(&pretok);
    }

    fn all_matches(tok: &BigSmirkPreTokenizer, bigsmiles: &str) -> bool {
        let splits = tok.find_matches(bigsmiles).unwrap();
        splits.into_iter().all(|(_s, m)| m)
    }

    fn get_matched_pretokens(tok: &BigSmirkPreTokenizer, bigsmiles: &str) -> Vec<String> {
        tok.find_matches(bigsmiles)
            .unwrap()
            .into_iter()
            .filter(|(_, m)| *m)
            .map(|(o, _)| bigsmiles[o.0..o.1].into())
            .collect()
    }

    fn get_split_tokens(tok: &BigSmirkPreTokenizer, bigsmiles: &str) -> Vec<String> {
        let mut bigsmiles = PreTokenizedString::from(bigsmiles);
        tok.pre_tokenize(&mut bigsmiles).unwrap();
        bigsmiles
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect()
    }

    #[test]
    fn test_standard_smiles_basic() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "OC[C@@H]"),
            ["O", "C", "[", "C", "@@", "H", "]"]
        );
        assert_eq!(
            get_split_tokens(&pretok, "OC[C@@H][OH]"),
            ["O", "C", "[", "C", "@@", "H", "]", "[", "O", "H", "]"]
        );
    }

    #[test]
    fn test_standard_smiles_chirality() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(get_split_tokens(&pretok, "[C@]"), ["[", "C", "@", "]"]);
        assert_eq!(
            get_split_tokens(&pretok, "[C@H]"),
            ["[", "C", "@", "H", "]"]
        );
        assert_eq!(get_split_tokens(&pretok, "[C@@]"), ["[", "C", "@@", "]"]);
        assert_eq!(
            get_split_tokens(&pretok, "[Fe@TB3+3]"),
            ["[", "Fe", "@TB", "3", "+", "3", "]"]
        );
    }

    #[test]
    fn test_standard_smiles_isotopes_charges() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "[16C]"),
            ["[", "1", "6", "C", "]"]
        );
        assert_eq!(
            get_split_tokens(&pretok, "[C+12]"),
            ["[", "C", "+", "1", "2", "]"]
        );
        assert_eq!(
            get_split_tokens(&pretok, "[CH4:200]"),
            ["[", "C", "H", "4", ":", "2", "0", "0", "]"]
        );
    }

    #[test]
    fn test_standard_smiles_rings_bonds() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "C1CCCC2C1CCCC2"),
            ["C", "1", "C", "C", "C", "C", "2", "C", "1", "C", "C", "C", "C", "2"]
        );
        assert_eq!(get_split_tokens(&pretok, "C%12"), ["C", "%", "1", "2"]);
        assert_eq!(
            get_split_tokens(&pretok, "F/C=C/F"),
            ["F", "/", "C", "=", "C", "/", "F"]
        );
        assert_eq!(
            get_split_tokens(&pretok, r"F/C=C\F"),
            ["F", "/", "C", "=", "C", "\\", "F"]
        );
    }

    #[test]
    fn test_standard_smiles_complex() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "[Na+].[Cl-]"),
            ["[", "Na", "+", "]", ".", "[", "Cl", "-", "]"]
        );
        assert_eq!(get_split_tokens(&pretok, "CC-O"), ["C", "C", "-", "O"]);
        assert_eq!(
            get_split_tokens(&pretok, "O=C=O"),
            ["O", "=", "C", "=", "O"]
        );
        assert_eq!(get_split_tokens(&pretok, "C#N"), ["C", "#", "N"]);
        assert_eq!(
            get_split_tokens(&pretok, "c1ccccc1"),
            ["c", "1", "c", "c", "c", "c", "c", "1"]
        );
        assert_eq!(
            get_split_tokens(&pretok, "FC(Br)(Cl)F"),
            ["F", "C", "(", "Br", ")", "(", "Cl", ")", "F"]
        );
        assert!(all_matches(
            &pretok,
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1"
        ));
    }

    #[test]
    fn test_stochastic_object_simple() {
        let pretok = BigSmirkPreTokenizer::default();
        // Simple polymer repeat unit with AA-type bonding
        assert_eq!(
            get_split_tokens(&pretok, "{[$]CC[$]}"),
            ["{", "[", "$", "]", "C", "C", "[", "$", "]", "}"]
        );
    }

    #[test]
    fn test_stochastic_object_multiple_units() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "{[$]CC[$],[$]C(C)C[$]}"),
            [
                "{", "[", "$", "]", "C", "C", "[", "$", "]", ",", "[", "$", "]", "C", "(", "C",
                ")", "C", "[", "$", "]", "}"
            ]
        );
    }

    #[test]
    fn test_ab_type_descriptors() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "{[<]CC[>]}"),
            ["{", "[", "<", "]", "C", "C", "[", ">", "]", "}"]
        );
        assert_eq!(
            get_split_tokens(&pretok, "{[>]CCCCCC(=O)[<],[>]NCCCCCCN[<]}"),
            [
                "{", "[", ">", "]", "C", "C", "C", "C", "C", "C", "(", "=", "O", ")", "[", "<",
                "]", ",", "[", ">", "]", "N", "C", "C", "C", "C", "C", "C", "N", "[", "<", "]",
                "}"
            ]
        );
    }

    #[test]
    fn test_indexed_bonding_descriptors() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(get_split_tokens(&pretok, "[$1]"), ["[", "$", "1", "]"]);
        assert_eq!(get_split_tokens(&pretok, "[$2]"), ["[", "$", "2", "]"]);

        assert_eq!(get_split_tokens(&pretok, "[<1]"), ["[", "<", "1", "]"]);
        assert_eq!(get_split_tokens(&pretok, "[>1]"), ["[", ">", "1", "]"]);

        assert_eq!(
            get_split_tokens(&pretok, "{[$1]CC[$1],[$2]C(C)C[$2]}"),
            [
                "{", "[", "$", "1", "]", "C", "C", "[", "$", "1", "]", ",", "[", "$", "2", "]",
                "C", "(", "C", ")", "C", "[", "$", "2", "]", "}"
            ]
        );
    }

    #[test]
    fn test_ladder_descriptors() {
        let pretok = BigSmirkPreTokenizer::default();

        assert_eq!(
            get_split_tokens(&pretok, "[<1[<1]1]"),
            ["[", "<", "1", "[", "<", "1", "]", "1", "]"]
        );
        assert_eq!(
            get_split_tokens(&pretok, "[$1[$2]3]"),
            ["[", "$", "1", "[", "$", "2", "]", "3", "]"]
        );

        assert_eq!(
            get_split_tokens(&pretok, "{[<1[<1]1]CC[>1[>1]1]}"),
            [
                "{", "[", "<", "1", "[", "<", "1", "]", "1", "]", "C", "C", "[", ">", "1", "[",
                ">", "1", "]", "1", "]", "}"
            ]
        );
    }

    #[test]
    fn test_external_bond_order_with_descriptors() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "C=[$2]"),
            ["C", "=", "[", "$", "2", "]"]
        );
        assert_eq!(
            get_split_tokens(&pretok, r"C/[>1]"),
            ["C", "/", "[", ">", "1", "]"]
        );
        assert_eq!(
            get_split_tokens(&pretok, r"C\[<1]"),
            ["C", "\\", "[", "<", "1", "]"]
        );
    }

    #[test]
    fn test_empty_terminal() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(get_split_tokens(&pretok, "[]"), ["[", "]"]);

        assert_eq!(
            get_split_tokens(&pretok, "{[]CC[$]}"),
            ["{", "[", "]", "C", "C", "[", "$", "]", "}"]
        );

        assert_eq!(
            get_split_tokens(&pretok, "{[]CC[]}"),
            ["{", "[", "]", "C", "C", "[", "]", "}"]
        );
    }

    #[test]
    fn test_end_groups_semicolon() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "{[$]CC[$];[H][$],[$]O}"),
            [
                "{", "[", "$", "]", "C", "C", "[", "$", "]", ";", "[", "H", "]", "[", "$", "]",
                ",", "[", "$", "]", "O", "}"
            ]
        );
        assert_eq!(
            get_split_tokens(&pretok, "{[$]CC[$];C[$],[$]C}"),
            [
                "{", "[", "$", "]", "C", "C", "[", "$", "]", ";", "C", "[", "$", "]", ",", "[",
                "$", "]", "C", "}"
            ]
        );
    }

    #[test]
    fn test_block_copolymer() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "{[$]CC[$]}{[$]CC(C)[$]}"),
            [
                "{", "[", "$", "]", "C", "C", "[", "$", "]", "}", "{", "[", "$", "]", "C", "C",
                "(", "C", ")", "[", "$", "]", "}"
            ]
        );
        assert_eq!(
            get_split_tokens(&pretok, "CC{[$]CC[$]}CC"),
            ["C", "C", "{", "[", "$", "]", "C", "C", "[", "$", "]", "}", "C", "C"]
        );
    }

    #[test]
    fn test_graft_copolymer_nested() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "{[$]CC(C{[<]CC[>]})[$]}"),
            [
                "{", "[", "$", "]", "C", "C", "(", "C", "{", "[", "<", "]", "C", "C", "[", ">",
                "]", "}", ")", "[", "$", "]", "}"
            ]
        );
        assert_eq!(
            get_split_tokens(&pretok, "{[$]CC{[<]C[>]}[$]}"),
            [
                "{", "[", "$", "]", "C", "C", "{", "[", "<", "]", "C", "[", ">", "]", "}", "[",
                "$", "]", "}"
            ]
        );
    }

    #[test]
    fn test_fragment_reference() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(get_split_tokens(&pretok, "[#PEG]"), ["[", "#", "PEG", "]"]);
        assert_eq!(
            get_split_tokens(&pretok, "[#Styrene]"),
            ["[", "#", "Styrene", "]"]
        );
        assert_eq!(get_split_tokens(&pretok, "[#+]"), ["[", "#", "+", "]"]);
        assert_eq!(
            get_split_tokens(&pretok, "[#PEG-1]"),
            ["[", "#", "PEG-1", "]"]
        );
        assert_eq!(get_split_tokens(&pretok, "[#A]"), ["[", "#", "A", "]"]);
        assert_eq!(
            get_split_tokens(&pretok, "{[$][#Styrene][$]}"),
            ["{", "[", "$", "]", "[", "#", "Styrene", "]", "[", "$", "]", "}"]
        );
    }

    #[test]
    fn test_reject_invalid_bracket_symbol_forms() {
        let pretok = BigSmirkPreTokenizer::default();
        assert!(!all_matches(&pretok, "[B|]"));
        assert!(!all_matches(&pretok, "[C@@Hextra]"));
        assert!(!all_matches(&pretok, "[$=]"));
        assert!(!all_matches(&pretok, "[>#]"));
        assert!(!all_matches(&pretok, "[$/]"));
        assert!(!all_matches(&pretok, r"[$\]"));
        assert!(!all_matches(&pretok, "[#PEG 1]"));
    }

    #[test]
    fn test_mixed_smiles_bigsmiles() {
        let pretok = BigSmirkPreTokenizer::default();
        assert_eq!(
            get_split_tokens(&pretok, "CCCC{[$]CC(c1ccccc1)[$]}CCCC"),
            [
                "C", "C", "C", "C", "{", "[", "$", "]", "C", "C", "(", "c", "1", "c", "c", "c",
                "c", "c", "1", ")", "[", "$", "]", "}", "C", "C", "C", "C"
            ]
        );
        assert_eq!(
            get_split_tokens(&pretok, "CCCC{[$]CC[$]}CCCC.NCC"),
            [
                "C", "C", "C", "C", "{", "[", "$", "]", "C", "C", "[", "$", "]", "}", "C", "C",
                "C", "C", ".", "N", "C", "C"
            ]
        );
    }

    #[test]
    fn test_opensmiles_spec() {
        let pretok = BigSmirkPreTokenizer::default();
        let mut opensmiles_examples = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        opensmiles_examples.push("test");
        opensmiles_examples.push("opensmiles.smi");
        let examples = fs::read_to_string(opensmiles_examples.as_path())
            .expect("failed to open opensmiles.smi");
        for line in examples.lines().filter(|x| !x.starts_with("#")) {
            dbg!(&line);
            assert!(all_matches(&pretok, line));
        }
    }
}
