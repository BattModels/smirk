use derive_builder::Builder;
use macro_rules_attribute::derive;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use tokenizers::parallelism::{MaybeParallelBridge, MaybeParallelRefIterator};
use tokenizers::{AddedToken, Result, Trainer};

use super::model::GPE;
use crate::shared::{compute_alphabet, tokenize_words, Word};

type Pair = (u32, u32);

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    pos: HashSet<usize>,
}
impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.count != other.count {
            return self.count.cmp(&other.count);
        }

        // Resolve ties in favor of smaller pairs
        other.pair.cmp(&self.pair)
    }
}

/// Header line of a scaffold-instrumentation log (Step 1.5, main.tex §3.2).
///
/// One header object, followed by one [`ScaffoldRecord`] line per committed
/// merge. `base_alphabet` is the pre-merge vocabulary as `[id, glyph]` pairs
/// in id order, so the log reconstructs every token's surface form on its own.
#[derive(Serialize)]
struct ScaffoldHeader<'a> {
    format: &'a str,
    min_frequency: u64,
    vocab_size: usize,
    merge_brackets: bool,
    limit_alphabet: Option<usize>,
    base_alphabet: Vec<(u32, &'a str)>,
}

/// One committed-merge record of a scaffold-instrumentation log (main.tex §3.2).
///
/// `candidate_freq` is the selected merge candidate's frequency at this step.
/// `standalone` carries the running standalone frequency of the (left, right,
/// merged) tokens this merge touched — no other token's standalone frequency
/// changes, so the log is delta-encoded and replays into the full vector.
#[derive(Serialize)]
struct ScaffoldRecord<'a> {
    step: usize,
    pair: (u32, u32),
    new_id: u32,
    new_token: &'a str,
    candidate_freq: u64,
    standalone: Vec<(u32, i64)>,
}

// Glyph Pair Encoding - BPE but supports multi-character "glyphs"
#[derive(Builder, Debug, Deserialize, Serialize, Clone)]
#[builder(default)]
pub struct GpeTrainer {
    // the min frequency of a pair to produce a merge operation
    pub min_frequency: u64,
    // the target vocabulary size
    pub vocab_size: usize,
    // the initial alphabet
    pub alphabet: HashSet<String>,
    // limit the size of the initial alphabet
    pub limit_alphabet: Option<usize>,
    // Special tokens to include in the vocab
    pub special_tokens: Vec<AddedToken>,
    // Should bracket ([, ]) be candidates for merges
    pub merge_brackets: bool,
    // Internal Map for tracking word counts
    word_counts: HashMap<String, u64>,
    // Step 1.5 scaffold instrumentation (vocab-tokenizer-clms study, main.tex
    // §3.2): when `Some`, `do_train` streams a per-merge-step JSONL log to this
    // path. `None` (the default) leaves training byte-identical to stock
    // output — the instrumentation is logging-only and never alters a merge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scaffold_log_path: Option<PathBuf>,
}

impl Default for GpeTrainer {
    fn default() -> Self {
        Self {
            min_frequency: 0,
            vocab_size: 1024,
            alphabet: HashSet::new(),
            limit_alphabet: None,
            special_tokens: Vec::new(),
            merge_brackets: false,
            word_counts: HashMap::new(),
            scaffold_log_path: None,
        }
    }
}

impl GpeTrainer {
    pub fn builder() -> GpeTrainerBuilder {
        GpeTrainerBuilder::default()
    }

    #[allow(dead_code)]
    fn new(min_frequency: u64, vocab_size: usize, alphabet: HashSet<String>) -> Self {
        Self {
            min_frequency,
            vocab_size,
            alphabet,
            ..Default::default()
        }
    }

    ///
    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(
        &self,
        model: &GPE,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        if !w2id.contains_key(&model.unk_token) {
            id2w.push(model.unk_token.to_owned());
            w2id.insert(model.unk_token.to_owned(), (id2w.len() - 1) as u32);
        }
        for token in &self.special_tokens {
            if !w2id.contains_key(&token.content) {
                id2w.push(token.content.to_owned());
                w2id.insert(token.content.to_owned(), (id2w.len() - 1) as u32);
            }
        }
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[i64],
    ) -> (HashMap<Pair, i64>, HashMap<Pair, HashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .filter(|(_, word)| word.len() >= 2) // Words shorter than 2 have no pair
            .map(|(i, word)| {
                let mut pair_count = HashMap::new();
                let mut where_to_update: HashMap<Pair, HashSet<usize>> = HashMap::new();

                for token_pair in word.windows(2) {
                    let pair: Pair = (*token_pair.get(0).unwrap(), *token_pair.get(1).unwrap());
                    let count = counts[i];
                    pair_count
                        .entry(pair)
                        .and_modify(|c| *c += count)
                        .or_insert(count);
                    where_to_update
                        .entry(pair)
                        .and_modify(|s| {
                            s.insert(i);
                        })
                        .or_insert_with(|| {
                            let mut s = HashSet::new();
                            s.insert(i);
                            s
                        });
                }
                (pair_count, where_to_update)
            })
            .reduce(
                || (HashMap::new(), HashMap::new()),
                |(mut pair_count, mut where_to_update), (pc, p2w)| {
                    for (pair, count) in pc {
                        pair_count
                            .entry(pair)
                            .and_modify(|c| *c += count)
                            .or_insert(count);
                        let words = p2w.get(&pair).unwrap();
                        where_to_update
                            .entry(pair)
                            .and_modify(|s| {
                                words.iter().for_each(|w| {
                                    s.insert(*w);
                                });
                            })
                            .or_insert(words.clone());
                    }
                    (pair_count, where_to_update)
                },
            )
    }

    pub fn do_train(
        &self,
        word_counts: &HashMap<String, u64>,
        model: &mut GPE,
    ) -> Result<Vec<AddedToken>> {
        // Setup initial alphabet
        let mut word_to_id: HashMap<String, u32> = HashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<String> = Vec::with_capacity(self.vocab_size);
        self.add_special_tokens(&model, &mut word_to_id, &mut id_to_word);
        compute_alphabet(
            &model.tokenize,
            word_counts,
            &self.alphabet,
            self.limit_alphabet,
            &mut word_to_id,
            &mut id_to_word,
        );

        // Save vocab without merges
        let vocab = word_to_id.to_owned();

        // Tokenize words, returning word_counts => (Vec, Vec)
        let (words, counts) = tokenize_words(
            &model.tokenize,
            word_counts,
            self.merge_brackets,
            &mut word_to_id,
            &mut id_to_word,
        );
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts);

        // --- Step 1.5 scaffold instrumentation (logging-only; main.tex §3.2) ---
        // When `scaffold_log_path` is set, stream a per-merge-step JSONL log:
        // the running standalone frequency of every token a merge touches and
        // that merge's selected-candidate frequency. This only reads training
        // state and writes a side file — it never alters merge selection, so
        // BPE artifacts stay byte-identical to stock output.
        let mut scaffold_writer = match &self.scaffold_log_path {
            Some(path) => {
                let file = File::create(path)
                    .map_err(|e| format!("scaffold_log_path {}: {e}", path.display()))?;
                Some(BufWriter::new(file))
            }
            None => None,
        };
        // Running standalone frequency per token id (current segmentation).
        let mut standalone: HashMap<u32, i64> = HashMap::new();
        if let Some(writer) = scaffold_writer.as_mut() {
            for (i, word) in words.iter().enumerate() {
                for &glyph in word.glyphs() {
                    *standalone.entry(glyph).or_insert(0) += counts[i];
                }
            }
            let base_alphabet: Vec<(u32, &str)> = id_to_word
                .iter()
                .enumerate()
                .map(|(id, token)| (id as u32, token.as_str()))
                .collect();
            let header = ScaffoldHeader {
                format: "smirk-scaffold-log/v1",
                min_frequency: self.min_frequency,
                vocab_size: self.vocab_size,
                merge_brackets: self.merge_brackets,
                limit_alphabet: self.limit_alphabet,
                base_alphabet,
            };
            serde_json::to_writer(&mut *writer, &header)?;
            writer.write_all(b"\n")?;
        }

        // Build a priority queue of merges
        let mut queue = BinaryHeap::new();
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            queue.push(Merge {
                pair,
                count: count.try_into().unwrap(),
                pos,
            });
        });

        let mut merges: Vec<Pair> = Vec::new();
        loop {
            // Stop if the vocab is large enough, or we have no merges left
            if (word_to_id.len() >= self.vocab_size) || queue.is_empty() {
                break;
            }

            // Pop a pair from the queue
            let mut top = queue.pop().unwrap();

            if top.count != pair_counts[&top.pair] as u64 {
                // Previous merge reduced the count, update
                top.count = pair_counts[&top.pair] as u64;
                if top.count != 0 {
                    queue.push(top);
                }
                continue;
            }

            // Check if pair meets threshold for merge
            if top.count < 1 || top.count < self.min_frequency {
                continue;
            }

            // Create a new token for the most frequently occurring pair
            let left_token = &id_to_word[top.pair.0 as usize];
            let right_token = &id_to_word[top.pair.1 as usize];
            let new_token = format!("{}{}", left_token, right_token);
            id_to_word.push(new_token.clone());
            let new_token_id = (id_to_word.len() - 1) as u32;
            word_to_id.insert(new_token, new_token_id);
            merges.push(top.pair);

            // Update words with new token, recording which pairs to update
            let changes = top
                .pos
                .maybe_par_iter()
                .map(|&i| {
                    let word = &words[i] as *const _ as *mut Word;
                    unsafe { ((*word).merge(top.pair, new_token_id), i) }
                })
                .collect::<Vec<_>>();

            // --- scaffold instrumentation: log this committed merge ---
            // `new_token_id` is brand-new, so each of its occurrences in the
            // just-merged words was created here; counting them gives the
            // exact number of (left, right) pairs collapsed. `top.count` is
            // the selected candidate's post-recount frequency.
            if let Some(writer) = scaffold_writer.as_mut() {
                let (left, right) = top.pair;
                let created: i64 = top
                    .pos
                    .iter()
                    .map(|&i| {
                        let occ = words[i]
                            .glyphs()
                            .iter()
                            .filter(|&&g| g == new_token_id)
                            .count() as i64;
                        occ * counts[i]
                    })
                    .sum();
                standalone.insert(new_token_id, created);
                // Each collapsed pair consumes one `left` and one `right`
                // (two `left`s when left == right) and creates the new token.
                let touched = if left == right {
                    *standalone.entry(left).or_insert(0) -= 2 * created;
                    vec![(left, standalone[&left]), (new_token_id, created)]
                } else {
                    *standalone.entry(left).or_insert(0) -= created;
                    *standalone.entry(right).or_insert(0) -= created;
                    vec![
                        (left, standalone[&left]),
                        (right, standalone[&right]),
                        (new_token_id, created),
                    ]
                };
                let record = ScaffoldRecord {
                    step: merges.len() - 1,
                    pair: top.pair,
                    new_id: new_token_id,
                    new_token: &id_to_word[new_token_id as usize],
                    candidate_freq: top.count,
                    standalone: touched,
                };
                serde_json::to_writer(&mut *writer, &record)?;
                writer.write_all(b"\n")?;
            }

            // Update pair_counts with changes
            for (change, iw) in changes {
                // Update pair_count
                let word_count = counts[iw];
                change.iter().for_each(|(pair, delta)| {
                    let count = *delta * (word_count as i64);
                    let _ = *pair_counts
                        .entry(*pair)
                        .and_modify(|c| *c += count)
                        .or_insert(count);

                    // If count is positive, may have a new word to update
                    if count > 0 {
                        where_to_update
                            .entry(*pair)
                            .and_modify(|s| {
                                s.insert(iw);
                            })
                            .or_insert({
                                let mut s = HashSet::new();
                                s.insert(iw);
                                s
                            });
                    }
                });
            }
            // Update the queue with new pairs
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair] as u64;
                queue.push(Merge { pair, count, pos });
            });
        }

        // Flush the scaffold log, surfacing any deferred write error.
        if let Some(mut writer) = scaffold_writer {
            writer.flush()?;
        }

        // Update Model
        model.with_vocab_and_merges(vocab, merges);
        Ok(self.special_tokens.clone())
    }
}

impl Trainer for GpeTrainer {
    type Model = GPE;

    // Don't use this, use `GPETrainer.train_from_files` directly
    fn train(&self, model: &mut GPE) -> Result<Vec<AddedToken>> {
        self.do_train(&self.word_counts, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        false
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<HashMap<String, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = HashMap::new();
                for word in words {
                    map.entry(word).and_modify(|c| *c += 1).or_insert(1);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            );

        self.word_counts = words?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wrapper::PreTokenizerWrapper;
    use crate::{gpe::GPE, pre_tokenizers::split_structure};
    use std::path::{Path, PathBuf};
    use tokenizers::Model;
    use tokenizers::{
        normalizers::Strip, DecoderWrapper, PostProcessorWrapper, TokenizerBuilder, TokenizerImpl,
    };

    #[test]
    fn test_trainer() {
        let word_counts: HashMap<String, u64> = [
            ("C", 4),
            ("CSCCSCCS", 2),
            ("CCSC", 1),
            ("[C@H]", 2),
            ("(", 3),
            (")", 4),
            ("CS", 3),
        ]
        .into_iter()
        .map(|(s, c)| (s.into(), c))
        .collect();
        let trainer = GpeTrainer::default();
        let mut model = GPE::default();
        assert_eq!(model.unk_token, "[UNK]");
        let _ = trainer.do_train(&word_counts, &mut model);

        let expected_vocab: HashMap<String, u32> = [
            ("[UNK]", 0),
            ("(", 1),
            (")", 2),
            ("@", 3),
            ("C", 4),
            ("H", 5),
            ("S", 6),
            ("[", 7),
            ("]", 8),
        ]
        .into_iter()
        .map(|(s, c)| (s.into(), c))
        .collect();
        let expected_merges: Vec<Pair> =
            [(4, 6), (4, 9), (3, 5), (4, 11), (9, 10), (13, 10), (10, 4)].into();
        assert_eq!(model.vocab, expected_vocab);
        assert_eq!(model.merges, expected_merges);
        assert_eq!(model.get_vocab_size(), 16);
        assert_eq!(model.id_to_token(9).unwrap(), "CS");
        assert_eq!(model.id_to_token(10).unwrap(), "CCS");
        assert_eq!(model.id_to_token(11).unwrap(), "@H");
        assert_eq!(model.id_to_token(12).unwrap(), "C@H");
        assert_eq!(model.id_to_token(13).unwrap(), "CSCCS");
        assert_eq!(model.id_to_token(14).unwrap(), "CSCCSCCS");
        assert_eq!(model.id_to_token(15).unwrap(), "CCSC");
    }

    #[test]
    fn test_tokenizer() {
        let mut tokenizer: TokenizerImpl<
            GPE,
            Strip,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = TokenizerBuilder::default()
            .with_model(GPE::default())
            .with_pre_tokenizer(Some(split_structure().into()))
            .build()
            .unwrap();
        let mut trainer = GpeTrainer::default();
        let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/smiles.txt");
        let files: Vec<String> = vec![test_file.to_string_lossy().into()];
        let _ = tokenizer.train_from_files(&mut trainer, files).unwrap();
        assert!(tokenizer
            .get_vocab(true)
            .contains_key(&tokenizer.get_model().unk_token))
    }

    // --- Step 1.5 scaffold instrumentation ---

    /// The `test_trainer` corpus, reused by the scaffold-instrumentation tests.
    fn scaffold_corpus() -> HashMap<String, u64> {
        [
            ("C", 4),
            ("CSCCSCCS", 2),
            ("CCSC", 1),
            ("[C@H]", 2),
            ("(", 3),
            (")", 4),
            ("CS", 3),
        ]
        .into_iter()
        .map(|(s, c)| (s.into(), c))
        .collect()
    }

    /// Train with a scaffold log written to `path`, returning the trained model.
    fn train_with_scaffold_log(word_counts: &HashMap<String, u64>, path: &Path) -> GPE {
        let trainer = GpeTrainer::builder()
            .scaffold_log_path(Some(path.to_path_buf()))
            .build()
            .unwrap();
        let mut model = GPE::default();
        trainer.do_train(word_counts, &mut model).unwrap();
        model
    }

    /// Parse a scaffold log into its (header, per-merge records).
    fn read_scaffold_log(path: &Path) -> (serde_json::Value, Vec<serde_json::Value>) {
        let text = std::fs::read_to_string(path).unwrap();
        let mut lines = text.lines();
        let header: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        let records = lines.map(|l| serde_json::from_str(l).unwrap()).collect();
        (header, records)
    }

    #[test]
    fn scaffold_log_does_not_change_the_model() {
        let word_counts = scaffold_corpus();
        let mut stock = GPE::default();
        GpeTrainer::default()
            .do_train(&word_counts, &mut stock)
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("scaffold.jsonl");
        let instrumented = train_with_scaffold_log(&word_counts, &log);

        assert_eq!(instrumented.vocab, stock.vocab);
        assert_eq!(instrumented.merges, stock.merges);
        assert!(log.is_file());
    }

    #[test]
    fn scaffold_log_has_one_record_per_merge() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("scaffold.jsonl");
        let model = train_with_scaffold_log(&scaffold_corpus(), &log);
        let (header, records) = read_scaffold_log(&log);

        assert_eq!(header["format"], "smirk-scaffold-log/v1");
        assert_eq!(records.len(), model.merges.len());
        for (step, rec) in records.iter().enumerate() {
            assert_eq!(rec["step"].as_u64().unwrap() as usize, step);
            let pair = &model.merges[step];
            assert_eq!(rec["pair"][0].as_u64().unwrap() as u32, pair.0);
            assert_eq!(rec["pair"][1].as_u64().unwrap() as u32, pair.1);
            assert!(rec["candidate_freq"].as_u64().unwrap() >= 1);
        }
    }

    #[test]
    fn scaffold_log_records_running_standalone_frequency() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("scaffold.jsonl");
        train_with_scaffold_log(&scaffold_corpus(), &log);
        let (_, records) = read_scaffold_log(&log);

        // First merge is (C=4, S=6) -> "CS" (id 9). C occurs 22x and S 10x in
        // the corpus segmentation; the merge collapses all 10 (C,S) pairs.
        let first = &records[0];
        assert_eq!(first["pair"], serde_json::json!([4, 6]));
        assert_eq!(first["new_id"], 9);
        assert_eq!(first["new_token"], "CS");
        assert_eq!(first["candidate_freq"], 10);
        assert_eq!(
            first["standalone"],
            serde_json::json!([[4, 12], [6, 0], [9, 10]])
        );
    }

    #[test]
    fn scaffold_log_handles_a_self_pair_merge() {
        // (S,S) is the only initial pair; S has id 1, the merged token id 2.
        let word_counts: HashMap<String, u64> = [("SS", 5), ("SSSS", 3)]
            .into_iter()
            .map(|(s, c)| (s.into(), c))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("scaffold.jsonl");
        train_with_scaffold_log(&word_counts, &log);
        let (_, records) = read_scaffold_log(&log);

        // candidate_freq counts windows: 5*1 + 3*3 = 14. The non-overlapping
        // merge collapses 5*1 + 3*2 = 11 occurrences, each consuming two S, so
        // standalone[S] = 5*2 + 3*4 - 2*11 = 0.
        let first = &records[0];
        assert_eq!(first["pair"], serde_json::json!([1, 1]));
        assert_eq!(first["candidate_freq"], 14);
        assert_eq!(first["standalone"], serde_json::json!([[1, 0], [2, 11]]));
    }

    #[test]
    fn no_scaffold_log_without_a_path() {
        // The default trainer has scaffold_log_path = None: training succeeds
        // and the instrumentation stays inert.
        let mut model = GPE::default();
        assert!(GpeTrainer::default()
            .do_train(&scaffold_corpus(), &mut model)
            .is_ok());
    }
}
