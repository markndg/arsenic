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
