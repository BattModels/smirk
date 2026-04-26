use const_format::formatcp;

const BRACKETED_SYMBOL: &'static str = concat!(
    r"A(?:c|g|l|m|r|s|t|u)|",
    r"B(?:a|e|h|i|k|r)?|",
    r"C(?:a|d|e|f|l|m|n|o|r|s|u)?|",
    r"D(?:b|s|y)|",
    r"E(?:r|s|u)|",
    r"F(?:e|l|m|r)?|",
    r"G(?:a|d|e)|",
    r"H(?:e|f|g|o|s)?|",
    r"I(?:n|r)?|",
    r"Kr?|",
    r"L(?:a|i|r|u|v)|",
    r"M(?:c|d|g|n|o|t)|",
    r"N(?:a|b|d|e|h|i|o|p)?|",
    r"O(?:g|s)?|",
    r"P(?:a|b|d|m|o|r|t|u)?|",
    r"R(?:a|b|e|f|g|h|n|u)|",
    r"S(?:b|c|e|g|i|m|n|r)?|",
    r"T(?:a|b|c|e|h|i|l|m|s)|",
    r"U|",
    r"V|",
    r"W|",
    r"Xe|",
    r"Yb?|",
    r"Z(?:n|r)|",
    r"as|",
    r"b|",
    r"c|",
    r"n|",
    r"o|",
    r"p|",
    r"se?|",
    r"\*",
);

const CHIRAL: &'static str = r"@(?:@|AL|OH|SP|T(?:B|H))?";

pub const MATCH_OUTER_BIGSMILES: &'static str = concat!(
    r"Br?|Cl?|F|I|N|O|P|S|", // organic subset elements
    r"b|c|n|o|p|s|",         // Aromatic organic subset
    r"\*|",                  // Wildcard
    r"[\.\-=\#\$:/\\]|",     // Bonds
    r"\d|%|",                // Ring closures
    r"\(|\)|",
    r"\{|\}|",                         // Stochastic object delimiters
    r",|;|",                           // Repeat unit separator and end group separator
    r"[A-Z][A-Za-z0-9']*|",            // Fragment and abstract spec labels
    r"\[(?:[^\[\]]+|\[[^\[\]]*\])*\]", // Bracketed atoms/descriptors
);

pub const MATCH_INNER_BIGSMILES: &'static str = formatcp!(concat!(
    r"^(?:",
    r"",
    r"|",
    r"(\$|<|>)(\d+)?",
    r"|",
    r"(\$|<|>)(\d+)?(\[)(\$|<|>|[A-Z][A-Za-z0-9']*)(\d+)?(\])(\d+)",
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
));

pub const BONDING_DESCRIPTOR: &'static str = concat!(
    r"(\$|<|>)", // Descriptor type
    r"(\d+)?",   // Optional index
);

pub const LADDER_BONDING_DESCRIPTOR: &'static str = concat!(
    r"(\$|<|>)",                                  // Outer descriptor type
    r"(\d+)?",                                    // Outer descriptor id
    r"(\[)(\$|<|>|[A-Z][A-Za-z0-9']*)(\d+)?(\])", // Inner descriptor
    r"(\d+)",                                     // Group id
);

pub const FRAGMENT_REFERENCE: &'static str = r"(#)([!-~]+)";
