use regex::Regex;
use std::sync::LazyLock;

use crate::types::ToneMetrics;

static HEDGE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(might|could|perhaps|possibly|it depends|generally|typically|usually|often|sometimes|in some cases|it's worth noting|however|that said|on the other hand)\b",
    )
    .expect("valid regex")
});

static CONTRACTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(i'm|you're|we're|they're|it's|that's|what's|there's|here's|don't|doesn't|didn't|won't|wouldn't|couldn't|shouldn't|can't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|i've|you've|we've|they've|i'll|you'll|we'll|they'll|i'd|you'd|we'd|they'd)\b",
    )
    .expect("valid regex")
});

static PASSIVE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(am|is|are|was|were|been|being)\s+(\w+ed|\w+en)\b",
    )
    .expect("valid regex")
});

pub struct ToneAnalyser;

impl ToneAnalyser {
    pub fn hedge_word_count(text: &str) -> usize {
        HEDGE_RE.find_iter(text).count()
    }

    pub fn contraction_count(text: &str) -> usize {
        CONTRACTION_RE.find_iter(text).count()
    }

    pub fn passive_voice_ratio(text: &str) -> f64 {
        let sentences = sentence_split_simple(text);
        if sentences.is_empty() {
            return 0.0;
        }
        let passive = sentences
            .iter()
            .filter(|s| PASSIVE_RE.is_match(s))
            .count();
        passive as f64 / sentences.len() as f64
    }

    pub fn formality_score(text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let n = words.len().max(1);
        let avg_word_len: f64 = words
            .iter()
            .map(|w| w.chars().filter(|c| c.is_alphabetic()).count() as f64)
            .sum::<f64>()
            / n as f64;
        let contraction_rate = Self::contraction_count(text) as f64 / n as f64;
        let hedge_rate = Self::hedge_word_count(text) as f64 / n as f64;
        let sentences = sentence_split_simple(text);
        let avg_sentence_len = if sentences.is_empty() {
            0.0
        } else {
            words.len() as f64 / sentences.len() as f64
        };
        let mut score = 0.35 * (avg_word_len / 12.0).min(1.0);
        score += 0.25 * (avg_sentence_len / 25.0).min(1.0);
        score += 0.2 * (1.0 - contraction_rate.min(1.0));
        score += 0.2 * (1.0 - hedge_rate.min(1.0));
        score.clamp(0.0, 1.0)
    }

    pub fn assertiveness_score(text: &str) -> f64 {
        let words = text.split_whitespace().count().max(1);
        let hedge = Self::hedge_word_count(text) as f64;
        let density = hedge / words as f64;
        (1.0 - (density * 5.0).min(1.0)).clamp(0.0, 1.0)
    }

    pub fn analyse(text: &str) -> ToneMetrics {
        let sentences = sentence_split_simple(text);
        let word_count = text.split_whitespace().count().max(1);
        let avg_sentence_len = if sentences.is_empty() {
            word_count as f64
        } else {
            word_count as f64 / sentences.len() as f64
        };
        ToneMetrics {
            formality_score: Self::formality_score(text),
            assertiveness_score: Self::assertiveness_score(text),
            hedge_word_count: Self::hedge_word_count(text),
            contraction_count: Self::contraction_count(text),
            average_sentence_length: avg_sentence_len,
            passive_voice_ratio: Self::passive_voice_ratio(text),
        }
    }
}

fn sentence_split_simple(text: &str) -> Vec<String> {
    text.split(|c| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}
