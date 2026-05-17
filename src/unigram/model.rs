use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tokenizers::{
    Model, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Result, Token,
};

use super::trainer::UnigramTrainer;
use crate::pre_tokenizers::SmirkPreTokenizer;

/// One Unigram-LM vocabulary piece: a Layer-A glyph-string sequence and its
/// unigram log-probability score.
///
/// Pieces are stored as explicit glyph sequences (not concatenated strings) so
/// segmentation is unambiguous regardless of whether a glyph string is itself a
/// concatenation of others.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnigramPiece {
    /// The Layer-A glyphs composing this piece.
    pub glyphs: Vec<String>,
    /// Unigram log-probability score.
    pub score: f64,
}

impl UnigramPiece {
    /// The piece's surface form — its glyphs concatenated.
    pub fn token(&self) -> String {
        self.glyphs.concat()
    }
}

/// A glyph-aware Unigram-LM tokenization model — the Unigram arm's analogue of
/// [`crate::gpe::GPE`].
///
/// Layer-A pre-tokenizes a SMILES string into glyphs, then segments the glyph
/// sequence by Viterbi argmax over the piece vocabulary (`vocab-tokenizer-clms`
/// study, §3.2). Pieces are plain glyph strings — no Private Use Area
/// codepoints ever reach a saved artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct UnigramModel {
    /// Unknown-token surface form.
    unk_token: String,
    /// Layer-A pre-tokenizer (glyph splitting), persisted with the model.
    pub tokenize: SmirkPreTokenizer,
    /// Vocabulary pieces in vocab-id order (id = index); excludes unk.
    pieces: Vec<UnigramPiece>,
    // ---- derived on construction, not serialized ----
    /// Glyph sequence -> piece id, for Viterbi lookup.
    by_glyphs: HashMap<Vec<String>, u32>,
    /// Surface form -> id (pieces and unk).
    token_to_id: HashMap<String, u32>,
    /// Longest piece length in glyphs (Viterbi span bound).
    max_piece_len: usize,
    /// Score charged for an unknown glyph during Viterbi.
    unk_score: f64,
}

impl UnigramModel {
    /// Build a model from its unk token, Layer-A pre-tokenizer, and pieces.
    pub fn new(unk_token: String, tokenize: SmirkPreTokenizer, pieces: Vec<UnigramPiece>) -> Self {
        let mut by_glyphs = HashMap::with_capacity(pieces.len());
        let mut token_to_id = HashMap::with_capacity(pieces.len() + 1);
        let mut max_piece_len = 1usize;
        for (id, piece) in pieces.iter().enumerate() {
            by_glyphs.insert(piece.glyphs.clone(), id as u32);
            token_to_id.insert(piece.token(), id as u32);
            max_piece_len = max_piece_len.max(piece.glyphs.len());
        }
        let unk_id = pieces.len() as u32;
        token_to_id.insert(unk_token.clone(), unk_id);
        let unk_score = pieces.iter().map(|p| p.score).fold(f64::INFINITY, f64::min);
        let unk_score = if unk_score.is_finite() {
            unk_score
        } else {
            0.0
        };
        Self {
            unk_token,
            tokenize,
            pieces,
            by_glyphs,
            token_to_id,
            max_piece_len,
            unk_score,
        }
    }

    /// Replace the piece vocabulary (used by [`UnigramTrainer`] after training).
    pub fn with_pieces(&mut self, pieces: Vec<UnigramPiece>) {
        *self = Self::new(self.unk_token.clone(), self.tokenize.clone(), pieces);
    }

    /// The vocabulary id reserved for the unknown token.
    fn unk_id(&self) -> u32 {
        self.pieces.len() as u32
    }

    /// Layer-A split: glyph surface forms with their byte offsets.
    fn glyph_splits(&self, sequence: &str) -> Result<Vec<(String, (usize, usize))>> {
        let mut pts = PreTokenizedString::from(sequence);
        self.tokenize.pre_tokenize(&mut pts)?;
        Ok(pts
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .iter()
            .map(|(s, o, _)| (s.to_string(), *o))
            .collect())
    }
}

impl Default for UnigramModel {
    fn default() -> Self {
        Self::new(
            "[UNK]".to_string(),
            SmirkPreTokenizer::default(),
            Vec::new(),
        )
    }
}

impl Model for UnigramModel {
    type Trainer = UnigramTrainer;

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }
        let glyphs = self.glyph_splits(sequence)?;
        let n = glyphs.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Viterbi argmax over the glyph sequence (subword sampling disabled).
        // best[k] = (total log-prob, segment start, token id) covering glyphs[0..k].
        let span = self.max_piece_len.max(1);
        let mut best: Vec<(f64, usize, u32)> = vec![(f64::NEG_INFINITY, 0, 0); n + 1];
        best[0] = (0.0, 0, 0);
        for k in 1..=n {
            for start in k.saturating_sub(span)..k {
                if best[start].0 == f64::NEG_INFINITY {
                    continue;
                }
                let piece: Vec<String> = glyphs[start..k].iter().map(|(g, _)| g.clone()).collect();
                let (score, id) = match self.by_glyphs.get(&piece) {
                    Some(&id) => (self.pieces[id as usize].score, id),
                    // Only a single out-of-vocabulary glyph falls through to unk.
                    None if k - start == 1 => (self.unk_score, self.unk_id()),
                    None => continue,
                };
                let total = best[start].0 + score;
                if total > best[k].0 {
                    best[k] = (total, start, id);
                }
            }
        }

        // Backtrack into segments, then emit tokens with merged offsets.
        let mut segments: Vec<(usize, usize, u32)> = Vec::new();
        let mut k = n;
        while k > 0 {
            let (_, start, id) = best[k];
            segments.push((start, k, id));
            k = start;
        }
        segments.reverse();
        Ok(segments
            .into_iter()
            .map(|(start, end, id)| Token {
                id,
                value: self
                    .id_to_token(id)
                    .unwrap_or_else(|| self.unk_token.clone()),
                offsets: (glyphs[start].1 .0, glyphs[end - 1].1 .1),
            })
            .collect())
    }

    fn get_trainer(&self) -> UnigramTrainer {
        UnigramTrainer::default()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_id.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.pieces.len() + 1
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        if id == self.unk_id() {
            Some(self.unk_token.clone())
        } else {
            self.pieces.get(id as usize).map(|p| p.token())
        }
    }

    fn save(&self, folder: &Path, prefix: Option<&str>) -> Result<Vec<PathBuf>> {
        let name = match prefix {
            Some(prefix) => format!("{prefix}-unigram.json"),
            None => "unigram.json".to_string(),
        };
        let path = folder.join(name);
        let file = File::create(&path)?;
        serde_json::to_writer(file, self)?;
        Ok(vec![path])
    }
}

impl Serialize for UnigramModel {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("Unigram", 4)?;
        s.serialize_field("type", "Unigram")?;
        s.serialize_field("unk_token", &self.unk_token)?;
        s.serialize_field("tokenize", &self.tokenize)?;
        s.serialize_field("vocab", &self.pieces)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for UnigramModel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_struct(
            "Unigram",
            &["type", "unk_token", "tokenize", "vocab"],
            UnigramVisitor,
        )
    }
}

struct UnigramVisitor;

impl<'de> Visitor<'de> for UnigramVisitor {
    type Value = UnigramModel;

    fn expecting(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "struct UnigramModel")
    }

    fn visit_map<V: MapAccess<'de>>(
        self,
        mut map: V,
    ) -> std::result::Result<UnigramModel, V::Error> {
        let mut unk_token: Option<String> = None;
        let mut tokenize: Option<SmirkPreTokenizer> = None;
        let mut pieces: Option<Vec<UnigramPiece>> = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "unk_token" => unk_token = Some(map.next_value()?),
                "tokenize" => tokenize = Some(map.next_value()?),
                "vocab" => pieces = Some(map.next_value()?),
                "type" => {
                    let tag: String = map.next_value()?;
                    if tag != "Unigram" {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(&tag),
                            &"Unigram",
                        ));
                    }
                }
                _ => {
                    let _: IgnoredAny = map.next_value()?;
                }
            }
        }
        Ok(UnigramModel::new(
            unk_token.unwrap_or_else(|| "[UNK]".to_string()),
            tokenize.unwrap_or_default(),
            pieces.unwrap_or_default(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Model;

    fn piece(glyphs: &[&str], score: f64) -> UnigramPiece {
        UnigramPiece {
            glyphs: glyphs.iter().map(|g| g.to_string()).collect(),
            score,
        }
    }

    /// A tiny model: base glyphs C/O plus the multi-glyph piece "CC".
    fn toy_model() -> UnigramModel {
        UnigramModel::new(
            "[UNK]".to_string(),
            SmirkPreTokenizer::default(),
            vec![
                piece(&["C"], -2.0),
                piece(&["O"], -3.0),
                piece(&["C", "C"], -1.0),
            ],
        )
    }

    #[test]
    fn viterbi_prefers_the_higher_scoring_segmentation() {
        let model = toy_model();
        // "CC" as one piece (-1.0) beats "C"+"C" (-4.0).
        let tokens = model.tokenize("CC").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, "CC");
        assert_eq!(tokens[0].offsets, (0, 2));
    }

    #[test]
    fn viterbi_round_trips_the_surface_form() {
        let model = toy_model();
        for smiles in ["CCO", "OCC", "CC", "O"] {
            let decoded: String = model
                .tokenize(smiles)
                .unwrap()
                .iter()
                .map(|t| t.value.clone())
                .collect();
            assert_eq!(decoded, smiles);
        }
    }

    #[test]
    fn vocab_size_counts_pieces_plus_unk() {
        let model = toy_model();
        assert_eq!(model.get_vocab_size(), 4);
        assert_eq!(model.token_to_id("CC"), Some(2));
        assert_eq!(model.token_to_id("[UNK]"), Some(3));
        assert_eq!(model.id_to_token(3).as_deref(), Some("[UNK]"));
    }

    #[test]
    fn serde_round_trips() {
        let model = toy_model();
        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("\"type\":\"Unigram\""));
        let loaded: UnigramModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model, loaded);
    }
}
