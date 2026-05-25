use std::collections::HashMap;

/// Offline semantic similarity using weighted token overlap (cosine on a simple bag-of-words).
/// BGE-small / Candle integration is not wired here yet; this is the v1 default scorer.
/// When disabled (`arsenic compare --no-semantic`), [`SemanticAnalyser::cosine_similarity`]
/// returns `1.0` immediately and does not tokenize. `ComparisonEngine` also short-circuits before calling this.
/// Bounded to \[0, 1\].
pub struct SemanticAnalyser {
    enabled: bool,
}

impl SemanticAnalyser {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn cosine_similarity(&self, v1: &str, v2: &str) -> anyhow::Result<f64> {
        if !self.enabled {
            return Ok(1.0);
        }
        Ok(token_cosine(v1, v2))
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty() && t.len() > 1)
        .map(|t| t.to_string())
        .collect()
}

fn token_cosine(a: &str, b: &str) -> f64 {
    let ta = tokenize(a);
    let tb = tokenize(b);
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let mut freq_a: HashMap<String, f64> = HashMap::new();
    for t in &ta {
        *freq_a.entry(t.clone()).or_insert(0.0) += 1.0;
    }
    let mut freq_b: HashMap<String, f64> = HashMap::new();
    for t in &tb {
        *freq_b.entry(t.clone()).or_insert(0.0) += 1.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (k, va) in &freq_a {
        na += va * va;
        if let Some(vb) = freq_b.get(k) {
            dot += va * vb;
        }
    }
    for vb in freq_b.values() {
        nb += vb * vb;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-9);
    (dot / denom).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_analyser_returns_one_without_tokenising() {
        let a = SemanticAnalyser::new(false);
        let sim = a.cosine_similarity("anything", "completely different").unwrap();
        assert_eq!(sim, 1.0);
        assert!(!a.is_enabled());
    }

    #[test]
    fn identical_text_scores_one() {
        let a = SemanticAnalyser::new(true);
        let sim = a
            .cosine_similarity(
                "Paris is the capital of France.",
                "Paris is the capital of France.",
            )
            .unwrap();
        assert!((sim - 1.0).abs() < 1e-9, "got {sim}");
    }

    #[test]
    fn disjoint_token_sets_score_zero() {
        let a = SemanticAnalyser::new(true);
        let sim = a
            .cosine_similarity("apple banana cherry", "violin trumpet drum")
            .unwrap();
        assert!(sim.abs() < 1e-9, "expected ~0, got {sim}");
    }

    #[test]
    fn partial_overlap_scores_between_zero_and_one() {
        let a = SemanticAnalyser::new(true);
        let sim = a
            .cosine_similarity(
                "The capital of France is Paris.",
                "Paris is a beautiful city in France.",
            )
            .unwrap();
        assert!(sim > 0.0 && sim < 1.0, "expected (0,1), got {sim}");
    }

    #[test]
    fn empty_texts_score_one() {
        let a = SemanticAnalyser::new(true);
        assert_eq!(a.cosine_similarity("", "").unwrap(), 1.0);
    }

    #[test]
    fn one_empty_one_nonempty_scores_zero() {
        let a = SemanticAnalyser::new(true);
        assert_eq!(a.cosine_similarity("", "hello world").unwrap(), 0.0);
        assert_eq!(a.cosine_similarity("hello world", "").unwrap(), 0.0);
    }

    #[test]
    fn tokenisation_is_case_insensitive() {
        let a = SemanticAnalyser::new(true);
        let sim = a
            .cosine_similarity("Paris IS the capital", "paris is THE capital")
            .unwrap();
        assert!((sim - 1.0).abs() < 1e-9, "expected 1.0, got {sim}");
    }

    #[test]
    fn single_char_tokens_are_filtered_out() {
        // tokenize() drops 1-char tokens; "a b c" becomes empty → empty-vs-empty = 1.0
        let a = SemanticAnalyser::new(true);
        let sim = a.cosine_similarity("a b c", "x y z").unwrap();
        assert_eq!(sim, 1.0);
    }
}
