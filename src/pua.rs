//! Internal PUA (Private Use Area) bridge for the Unigram-LM trainer.
//!
//! HuggingFace's `UnigramTrainer` operates on Unicode `char` slices (its
//! seed-substring enumeration is a suffix array over a character stream);
//! Smirk's glyphs are `u32` ids. This module bridges the two by mapping each
//! glyph id `g` to a single Unicode Private Use Area codepoint `U+E000 + g`,
//! so a glyph-id sequence becomes a `String` HuggingFace can train on.
//!
//! The mapping is purely internal to training: PUA codepoints never appear in
//! a saved [`crate::unigram::UnigramModel`] or any serialized artifact
//! (`vocab-tokenizer-clms` study, §3.2). [`decode_piece`] turns a trained PUA
//! piece back into glyph ids; the trainer must only hand it genuine PUA
//! pieces (HuggingFace's `<UNK>` / special-token entries are not PUA strings).

/// First Private Use Area codepoint (`U+E000`).
const PUA_BASE: u32 = 0xE000;

/// Size of the Basic Multilingual Plane PUA block (`U+E000`..=`U+F8FF`).
const PUA_LEN: u32 = 0xF8FF - PUA_BASE + 1; // 6400

/// Map a single glyph id to its PUA codepoint.
///
/// # Panics
/// If `glyph` does not fit the BMP PUA block (`glyph >= 6400`). Smirk base
/// alphabets are ~165 glyphs, far inside the block.
pub fn encode_glyph(glyph: u32) -> char {
    assert!(
        glyph < PUA_LEN,
        "glyph id {glyph} exceeds the {PUA_LEN}-codepoint PUA block"
    );
    // PUA_BASE + glyph lands in U+E000..=U+F8FF, all valid Unicode scalars.
    char::from_u32(PUA_BASE + glyph).expect("PUA codepoint is a valid char")
}

/// Map a PUA codepoint back to its glyph id.
///
/// # Panics
/// If `c` is not a PUA codepoint this bridge produces.
pub fn decode_glyph(c: char) -> u32 {
    let cp = c as u32;
    assert!(
        (PUA_BASE..PUA_BASE + PUA_LEN).contains(&cp),
        "char U+{cp:04X} is not a bridge PUA codepoint"
    );
    cp - PUA_BASE
}

/// Encode a glyph-id sequence (one Smirk word / Layer-B chunk) as a PUA string.
///
/// Feeding one chunk per HuggingFace `Sentence` keeps its `\0`-separated
/// suffix-array enumeration from spanning two Layer-B chunks (§3.2).
pub fn encode_word(glyphs: &[u32]) -> String {
    glyphs.iter().map(|&g| encode_glyph(g)).collect()
}

/// Decode a trained PUA piece back into its glyph-id sequence.
pub fn decode_piece(piece: &str) -> Vec<u32> {
    piece.chars().map(decode_glyph).collect()
}

/// Whether every codepoint in `s` falls in this bridge's PUA block.
pub fn is_pua(s: &str) -> bool {
    s.chars()
        .all(|c| (PUA_BASE..PUA_BASE + PUA_LEN).contains(&(c as u32)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_round_trips_through_pua() {
        for g in [0u32, 1, 42, 164, 165, 6399] {
            assert_eq!(decode_glyph(encode_glyph(g)), g);
        }
    }

    #[test]
    fn word_round_trips() {
        let glyphs = vec![0u32, 5, 5, 130, 7];
        let encoded = encode_word(&glyphs);
        assert_eq!(encoded.chars().count(), glyphs.len());
        assert_eq!(decode_piece(&encoded), glyphs);
    }

    #[test]
    fn encoded_codepoints_are_all_in_the_pua_block() {
        let encoded = encode_word(&[0, 100, 6399]);
        assert!(is_pua(&encoded));
        assert!(!is_pua("CC"));
        assert!(!is_pua("<UNK>"));
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn encode_glyph_rejects_out_of_range() {
        encode_glyph(6400);
    }
}
