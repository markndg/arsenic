use unicode_segmentation::UnicodeSegmentation;

use crate::types::{MorphologyMetrics, ResponseType};

pub struct MorphologyAnalyser;

impl MorphologyAnalyser {
    pub fn analyse(text: &str, token_count: usize) -> MorphologyMetrics {
        let word_count = text.split_whitespace().count();
        let sentence_count = Self::sentence_count(text);
        let paragraph_count = Self::paragraph_count(text);
        let has_lists = Self::has_lists(text);
        let has_headers = Self::has_headers(text);
        let has_code_blocks = Self::has_code_blocks(text);
        let has_caveats = Self::has_caveats(text);
        let mut metrics = MorphologyMetrics {
            token_count,
            word_count,
            sentence_count,
            paragraph_count,
            has_lists,
            has_headers,
            has_code_blocks,
            has_caveats,
            response_type: ResponseType::SingleLine,
        };
        metrics.response_type = Self::detect_response_type(&metrics, text);
        metrics
    }

    fn detect_response_type(metrics: &MorphologyMetrics, text: &str) -> ResponseType {
        if metrics.has_lists || metrics.has_headers || metrics.has_code_blocks {
            return ResponseType::Structured;
        }
        let s = metrics.sentence_count;
        let p = metrics.paragraph_count;
        if p > 1 {
            return ResponseType::MultiParagraph;
        }
        if s >= 5 {
            ResponseType::LongParagraph
        } else if s >= 2 {
            ResponseType::ShortParagraph
        } else if text.lines().count() <= 1 && !text.contains('\n') {
            ResponseType::SingleLine
        } else {
            ResponseType::ShortParagraph
        }
    }

    pub fn has_caveats(text: &str) -> bool {
        let lower = text.to_lowercase();
        [
            "might not",
            "may not",
            "cannot guarantee",
            "not necessarily",
            "i'm not sure",
            "unclear",
            "it depends",
            "typically",
            "generally speaking",
        ]
        .iter()
        .any(|p| lower.contains(p))
    }

    pub fn has_code_blocks(text: &str) -> bool {
        text.contains("```")
            || (text.contains("    ") && text.lines().any(|l| l.starts_with("    ")))
    }

    pub fn has_headers(text: &str) -> bool {
        text.lines().any(|l| {
            let t = l.trim_start();
            t.starts_with('#')
                || (t.len() < 80
                    && t.ends_with(':')
                    && t.split_whitespace().count() <= 8)
        })
    }

    pub fn has_lists(text: &str) -> bool {
        text.lines().any(|l| {
            let t = l.trim_start();
            t.starts_with("- ")
                || t.starts_with("* ")
                || t.starts_with("1.")
                || (t
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
                    && t.contains(". "))
        })
    }

    pub fn paragraph_count(text: &str) -> usize {
        let paras: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();
        paras.len().max(1)
    }

    pub fn sentence_count(text: &str) -> usize {
        let s: Vec<&str> = text.unicode_sentences().collect();
        if s.is_empty() {
            1
        } else {
            s.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_line_classifies_correctly() {
        let m = MorphologyAnalyser::analyse("Paris.", 2);
        assert!(matches!(m.response_type, ResponseType::SingleLine));
        assert!(!m.has_lists);
        assert!(!m.has_headers);
        assert!(!m.has_code_blocks);
    }

    #[test]
    fn structured_wins_over_paragraph_classification() {
        // Long multi-paragraph text BUT with a list — should be Structured.
        let text = "Intro paragraph here.\n\nNext paragraph.\n\n- bullet one\n- bullet two";
        let m = MorphologyAnalyser::analyse(text, 20);
        assert!(matches!(m.response_type, ResponseType::Structured));
        assert!(m.has_lists);
    }

    #[test]
    fn multi_paragraph_classification() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let m = MorphologyAnalyser::analyse(text, 10);
        assert!(matches!(m.response_type, ResponseType::MultiParagraph));
        assert_eq!(m.paragraph_count, 3);
    }

    #[test]
    fn long_vs_short_paragraph_threshold() {
        let short = "One. Two.";
        let m_short = MorphologyAnalyser::analyse(short, 4);
        assert!(matches!(m_short.response_type, ResponseType::ShortParagraph));

        let long = "One. Two. Three. Four. Five. Six.";
        let m_long = MorphologyAnalyser::analyse(long, 6);
        assert!(matches!(m_long.response_type, ResponseType::LongParagraph));
    }

    #[test]
    fn detects_fenced_and_indented_code_blocks() {
        assert!(MorphologyAnalyser::has_code_blocks("here:\n```rust\nfn x() {}\n```"));
        assert!(MorphologyAnalyser::has_code_blocks("prose\n    let x = 1;\n"));
        assert!(!MorphologyAnalyser::has_code_blocks("just prose."));
    }

    #[test]
    fn detects_headers_with_hash_and_short_colon_lines() {
        assert!(MorphologyAnalyser::has_headers("# Title\n\nbody"));
        assert!(MorphologyAnalyser::has_headers("Overview:\nText follows."));
        // 9 words ending in colon: too long → not a header.
        assert!(!MorphologyAnalyser::has_headers(
            "This is a quite long sentence ending with a colon:"
        ));
    }

    #[test]
    fn detects_lists_with_dash_star_and_numbered() {
        assert!(MorphologyAnalyser::has_lists("- one\n- two"));
        assert!(MorphologyAnalyser::has_lists("* one\n* two"));
        assert!(MorphologyAnalyser::has_lists("1. one\n2. two"));
        assert!(!MorphologyAnalyser::has_lists("just prose with a -dash inside"));
    }

    #[test]
    fn detects_caveat_phrases() {
        assert!(MorphologyAnalyser::has_caveats(
            "This might not work in every case."
        ));
        assert!(MorphologyAnalyser::has_caveats("I'm not sure about that."));
        assert!(!MorphologyAnalyser::has_caveats("The answer is 42."));
    }

    #[test]
    fn empty_text_yields_one_paragraph_one_sentence() {
        let m = MorphologyAnalyser::analyse("", 0);
        assert_eq!(m.paragraph_count, 1);
        assert_eq!(m.sentence_count, 1);
        assert_eq!(m.word_count, 0);
    }
}
