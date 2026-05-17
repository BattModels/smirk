//! Unigram-LM tokenizer arm — a glyph-aware Unigram model and its trainer.
//!
//! Mirrors `src/gpe/` (the BPE arm). [`UnigramModel`] holds a glyph-piece
//! vocabulary and segments by Viterbi argmax; [`UnigramTrainer`] mirrors
//! `GpeTrainer`'s knobs and shared Layer A/B/C front-end, delegating the
//! EM/pruning loop to HuggingFace's `UnigramTrainer` through the
//! [`crate::pua`] bridge (`vocab-tokenizer-clms` study, §3.2).

mod model;
mod trainer;

pub use model::UnigramModel;
pub use trainer::UnigramTrainer;
