//! Deterministic sentence embeddings (hash bag) used for claim matching and
//! sentence-level semantic scoring when Candle/BGE is disabled or unavailable.
//! With `--features semantic` and a valid model directory, `candle_embed` can
//! replace this path at runtime (see `ComparisonEngine` wiring).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const HASH_DIM: usize = 256;

pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}

pub fn hash_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; HASH_DIM];
    for w in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        if w.len() < 2 {
            continue;
        }
        let mut h = DefaultHasher::new();
        w.hash(&mut h);
        let idx = (h.finish() as usize) % HASH_DIM;
        v[idx] += 1.0;
    }
    l2_normalize(&v)
}

pub fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
    }
    dot.clamp(-1.0, 1.0)
}

pub fn embed_batch_hash(texts: &[String]) -> Vec<Vec<f32>> {
    texts.iter().map(|t| hash_embed(t)).collect()
}

/// v2 sentence-level similarity: for each v1 sentence, best match on v2, weighted by density.
pub fn weighted_sentence_similarity(text1: &str, text2: &str) -> f64 {
    use unicode_segmentation::UnicodeSegmentation;
    let s1: Vec<&str> = text1
        .unicode_sentences()
        .map(|s| s.trim())
        .filter(|s| s.len() > 3)
        .collect();
    let s2: Vec<String> = text2
        .unicode_sentences()
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() > 3)
        .collect();
    if s1.is_empty() && s2.is_empty() {
        return 1.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }
    let em2: Vec<Vec<f32>> = s2.iter().map(|s| hash_embed(s)).collect();
    let mut num = 0.0;
    let mut den = 0.0;
    for s in &s1 {
        let e1 = hash_embed(s);
        let w = sentence_weight(s);
        let best = em2
            .iter()
            .map(|e2| cosine_f32(&e1, e2))
            .fold(0.0f64, f64::max);
        num += best * w;
        den += w;
    }
    if den < 1e-9 {
        return 0.0;
    }
    (num / den).clamp(0.0, 1.0)
}

fn sentence_weight(s: &str) -> f64 {
    (0.15 + s.split_whitespace().count() as f64 / 120.0).min(1.2)
}
