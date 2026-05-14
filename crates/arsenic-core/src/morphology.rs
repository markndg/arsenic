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
