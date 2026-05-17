//! Shared trainer-input preparation for the GPE (BPE) and Unigram-LM trainers.
//!
//! [`compute_alphabet`] and [`tokenize_words`] were originally private methods
//! of [`crate::gpe::GpeTrainer`]. They are lifted here unchanged so a
//! Unigram-LM sibling trainer can drive the identical Layer A/B/C
//! pre-tokenization-to-glyph-id pipeline (`vocab-tokenizer-clms` study, §3.2).
//! [`Word`] — the glyph-id sequence both trainers consume — lives here with
//! them; the BPE merge *driver* (`Merge`, `count_pairs`, `do_train`) stays in
//! [`crate::gpe`].

use either::Either;
use std::collections::{HashMap, HashSet};
use std::slice::Windows;

use crate::gpe::GPE;

type Pair = (u32, u32);

/// A pre-tokenized word: a sequence of glyph ids.
#[derive(PartialEq, Debug)]
pub struct Word {
    glyphs: Vec<u32>,
}

impl Word {
    pub fn windows(&self, size: usize) -> Windows<'_, u32> {
        self.glyphs.windows(size)
    }

    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    // Replace a Pair of tokens with a new token (id)
    // Return a HashMap of pair => ±n for the impact of this merge on pair_counts
    pub fn merge(&mut self, pair: Pair, id: u32) -> HashMap<Pair, i64> {
        let mut changes: HashMap<Pair, i64> = HashMap::new();
        let mut ldx = 0;
        let word = &mut self.glyphs;
        for rdx in 1..word.len() {
            let cur_pair = (word[ldx], word[rdx]);
            if cur_pair == pair {
                *changes.entry(cur_pair).or_insert(0) -= 1;
                if ldx != 0 {
                    // Update Left-side Pair Count
                    *changes.entry((word[ldx - 1], word[ldx])).or_insert(0) -= 1;
                    *changes.entry((word[ldx - 1], id)).or_insert(0) += 1;
                }
                if rdx + 1 < word.len() {
                    // Update Right-side Pair Count
                    *changes.entry((word[rdx], word[rdx + 1])).or_insert(0) -= 1;
                    *changes.entry((id, word[rdx + 1])).or_insert(0) += 1;

                    // Move over a glyph across the ldx - rdx gap
                    word[ldx + 1] = word[rdx + 1];
                }
                word[ldx] = id;
            } else {
                ldx += 1;
                word[ldx] = word[rdx];
            }
        }
        word.drain(ldx + 1..word.len());
        changes
    }
}

impl FromIterator<u32> for Word {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let glyphs: Vec<u32> = iter.into_iter().collect();
        Word { glyphs }
    }
}

/// Compute the initial alphabet and limit it if relevant
pub fn compute_alphabet(
    model: &GPE,
    wc: &HashMap<String, u64>,
    initial_alphabet: &HashSet<String>,
    limit_alphabet: Option<usize>,
    w2id: &mut HashMap<String, u32>,
    id2w: &mut Vec<String>,
) {
    let mut alphabet: HashMap<String, usize> = HashMap::new();
    for (word, count) in wc {
        for glyph in model.tokenize.split(word) {
            alphabet
                .entry(glyph)
                .and_modify(|c| *c += *count as usize)
                .or_insert(*count as usize);
        }
    }

    // Add the initial alphabet
    initial_alphabet.iter().for_each(|glyph| {
        alphabet
            .entry(glyph.to_owned())
            .and_modify(|c| *c = std::usize::MAX)
            .or_insert(std::usize::MAX);
    });

    // Sort the alphabet and populate w2id and id2w
    let mut alphabet = alphabet.into_iter().collect::<Vec<_>>();

    // Truncate alphabet, if required, by removing the most uncommon glyphs
    if let Some(limit) = limit_alphabet {
        if alphabet.len() > limit {
            let n_remove = alphabet.len() - limit;
            alphabet.sort_unstable_by_key(|k| k.1);
            alphabet.drain(..n_remove);
        }
    }

    // Sort for determinism
    alphabet.sort_unstable_by_key(|k| k.0.to_owned());
    for (glyph, _) in alphabet {
        if !w2id.contains_key(&glyph) {
            id2w.push(glyph.clone());
            w2id.insert(glyph, (id2w.len() - 1) as u32);
        }
    }
}

/// Split each corpus word into a sequence of glyph ids.
///
/// When `merge_brackets` is false the `[` / `]` glyphs are dropped from the
/// training stream (Layer C); when true they are retained as merge candidates.
pub fn tokenize_words(
    model: &GPE,
    wc: &HashMap<String, u64>,
    merge_brackets: bool,
    w2id: &mut HashMap<String, u32>,
    id2w: &mut Vec<String>,
) -> (Vec<Word>, Vec<i64>) {
    let mut words: Vec<Word> = Vec::with_capacity(wc.len());
    let mut counts: Vec<i64> = Vec::with_capacity(wc.len());
    for (word, count) in wc {
        counts.push(*count as i64);
        let token_iter = model.tokenize.split(word).into_iter();

        let iter = if !merge_brackets {
            Either::Left(token_iter.filter(|s| !(s == "[" || s == "]")))
        } else {
            Either::Right(token_iter)
        };

        let symbol_ids = iter.map(|symbol| {
            w2id.get(&symbol)
                .map(|v| v.to_owned())
                .or_else(|| {
                    let id = id2w.len() as u32;
                    id2w.push(symbol.to_string());
                    w2id.insert(symbol, id);
                    Some(id)
                })
                .unwrap()
        });
        words.push(symbol_ids.collect());
    }
    (words, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_change() {
        let mut word = Word {
            glyphs: [0, 1, 3, 4, 5].to_vec(),
        };
        let changes = word.merge((1, 3), 6);
        assert_eq!(word.glyphs, [0, 6, 4, 5]);
        let expect: HashMap<Pair, i64> = HashMap::from([
            ((0, 1), -1),
            ((1, 3), -1),
            ((0, 6), 1),
            ((3, 4), -1),
            ((6, 4), 1),
        ]);
        assert_eq!(changes, expect);
    }

    #[test]
    fn test_double_merge() {
        let mut word = Word {
            glyphs: [0, 1, 3, 1, 3].to_vec(),
        };
        let changes = word.merge((1, 3), 6);
        assert_eq!(word.glyphs, [0, 6, 6]);
        let expect: HashMap<Pair, i64> = HashMap::from([
            ((0, 1), -1),
            ((1, 3), -2),
            ((3, 1), -1),
            ((0, 6), 1),
            ((6, 6), 1),
            ((6, 1), 0),
        ]);
        assert_eq!(changes, expect);
    }

    #[test]
    fn test_merge_nochange() {
        let mut word = Word {
            glyphs: [0, 1, 3, 4, 1, 3, 5].to_vec(),
        };
        let changes = &word.merge((1, 7), 6);
        assert_eq!(word.glyphs, [0, 1, 3, 4, 1, 3, 5]);
        assert!(changes.len() == 0);
    }

    /// `compute_alphabet` drops the least-frequent glyphs when over the limit.
    ///
    /// Exercises the `limit_alphabet` truncation branch, which the public
    /// `train_gpe` path never sets and so the byte-identity test cannot reach.
    #[test]
    fn compute_alphabet_respects_limit() {
        let model = GPE::default();
        // Glyph frequencies (count x occurrences per word):
        //   C = 6 + 4 + 1 = 11, O = 3, S = 2, N = 1.
        let wc: HashMap<String, u64> = HashMap::from([
            ("CCO".to_string(), 3),
            ("CCS".to_string(), 2),
            ("CN".to_string(), 1),
        ]);
        let mut w2id: HashMap<String, u32> = HashMap::new();
        let mut id2w: Vec<String> = Vec::new();

        compute_alphabet(&model, &wc, &HashSet::new(), Some(2), &mut w2id, &mut id2w);

        // The two rarest glyphs (N, S) are dropped; survivors are id-ordered
        // by glyph string.
        assert_eq!(id2w, vec!["C".to_string(), "O".to_string()]);
        assert_eq!(w2id.get("C"), Some(&0));
        assert_eq!(w2id.get("O"), Some(&1));
        assert_eq!(w2id.len(), 2);
    }
}
