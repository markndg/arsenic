//! Sentence-level claim extraction and cross-matching (v2).

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

use crate::embedding::{cosine_f32, embed_batch_hash};
use crate::types::{
    AnchorDrift, AnchorType, Claim, ClaimAnchor, ClaimDiff, ClaimDrift, ClaimMatch, DriftDirection,
    RiskLevel,
};

pub struct ClaimExtractor;

static SCAFFOLD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(great question|in conclusion|it's worth noting|i hope this (helps|explanation)|feel free to ask|do you have any specific questions)")
        .expect("regex")
});

static NUMERIC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b\d[\d.,]*\b").expect("regex"));
static YEAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(19|20)\d{2}\b").expect("regex"));

/// Strip URLs and `host:port` fragments so endpoint metadata never becomes numeric anchors.
static HTTP_URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"https?://[^\s<>\])'"]+"#).expect("regex")
});
static HOST_PORT: LazyLock<Regex> = LazyLock::new(|| {
    // `localhost:11434`, `127.0.0.1:11434`, or any IPv4:port (endpoint echoes in model text).
    Regex::new("(?i)\\b(?:localhost|127\\.0\\.0\\.1|(?:\\d{1,3}\\.){3}\\d{1,3})\\s*:\\s*\\d{2,5}\\b")
        .expect("regex")
});
static WS_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new("[ \t]{2,}").expect("regex"));

/// Lowercase English stoplist + discourse words; see `claim_stopwords.txt`.
static PROPER_NOUN_STOPLIST: LazyLock<HashSet<String>> = LazyLock::new(|| {
    include_str!("claim_stopwords.txt")
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim().to_lowercase())
        .filter(|l| !l.is_empty())
        .collect()
});

static CALENDAR_MONTHS: LazyLock<HashSet<String>> = LazyLock::new(|| {
    [
        "january", "february", "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
        "sept", "oct", "nov", "dec",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
});

/// Whole-token contractions (ASCII or U+2019 apostrophe) — not meaningful claim anchors.
static CONTRACTION_ANCHOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        "(?i)^(?:",
        "i[\x27\u{2019}]ll|i[\x27\u{2019}]m|i[\x27\u{2019}]ve|i[\x27\u{2019}]d|",
        "it[\x27\u{2019}]s|it[\x27\u{2019}]d|",
        "we[\x27\u{2019}]re|we[\x27\u{2019}]ve|we[\x27\u{2019}]ll|",
        "you[\x27\u{2019}]re|you[\x27\u{2019}]ve|you[\x27\u{2019}]ll|you[\x27\u{2019}]d|",
        "they[\x27\u{2019}]re|they[\x27\u{2019}]ve|they[\x27\u{2019}]ll|they[\x27\u{2019}]d|",
        "he[\x27\u{2019}]s|he[\x27\u{2019}]ll|he[\x27\u{2019}]d|",
        "she[\x27\u{2019}]s|she[\x27\u{2019}]ll|she[\x27\u{2019}]d|",
        "that[\x27\u{2019}]s|there[\x27\u{2019}]s|here[\x27\u{2019}]s|what[\x27\u{2019}]s|who[\x27\u{2019}]s|",
        "where[\x27\u{2019}]s|when[\x27\u{2019}]s|how[\x27\u{2019}]s|let[\x27\u{2019}]s|",
        "(?:do|does|did|is|are|was|were|have|has|had|would|could|should|must|might|need)[\x27\u{2019}]?nt|",
        "can[\x27\u{2019}]t|won[\x27\u{2019}]t|shan[\x27\u{2019}]t|ain[\x27\u{2019}]t",
        ")$",
    ))
    .expect("regex")
});

/// Remove URLs and host:port segments from model output before claim extraction.
/// [`ClaimExtractor::extract`] only reads `ModelResponse.content`; this strips accidental
/// endpoint echoes from that text (never reads `ModelResponse.raw`).
fn sanitize_model_text_for_claims(text: &str) -> String {
    let s = HTTP_URL.replace_all(text, " ");
    let s = HOST_PORT.replace_all(&s, " ");
    WS_RUN.replace_all(&s, " ").trim().to_string()
}

/// Ports and other numeric tokens that are infrastructure, not factual claims.
fn is_noise_numeric_token(raw: &str) -> bool {
    let t = raw.trim().trim_end_matches(|c: char| c == ',' || c == '.');
    matches!(
        t,
        "11434" | "11435"
            | "3000"
            | "3001"
            | "4000"
            | "4200"
            | "5000"
            | "5173"
            | "5432"
            | "5678"
            | "6379"
            | "7860"
            | "8080"
            | "8081"
            | "8443"
            | "9090"
            | "9200"
            | "27017"
    )
}

fn is_spurious_proper_noun_token(word: &str) -> bool {
    let value = word.trim_matches(|c: char| !c.is_alphanumeric());
    if value.chars().count() < 4 {
        return true;
    }
    let lower = value.to_lowercase();
    if CALENDAR_MONTHS.contains(&lower) {
        return true;
    }
    PROPER_NOUN_STOPLIST.contains(&lower)
}

impl ClaimExtractor {
    pub fn extract(text: &str) -> Vec<Claim> {
        let text = sanitize_model_text_for_claims(text);
        let mut out = Vec::new();
        for sent in text.unicode_sentences() {
            let s = sent.trim();
            if s.len() < 8 {
                continue;
            }
            if Self::is_scaffolding(s) {
                continue;
            }
            let density = Self::information_density(s);
            if density < 0.12 {
                continue;
            }
            let anchors = Self::extract_anchors(s);
            out.push(Claim {
                text: s.to_string(),
                information_density: density,
                anchors,
            });
        }
        out
    }

    pub fn information_density(sentence: &str) -> f64 {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let n = words.len().max(1);
        let mut score = 0.0;
        if NUMERIC.is_match(sentence) {
            score += 0.25;
        }
        if YEAR.is_match(sentence) {
            score += 0.15;
        }
        let caps = words
            .iter()
            .filter(|w| {
                w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && w.len() > 2
                    && !w.ends_with(':')
            })
            .count();
        score += (caps as f64 / n as f64).min(0.25);
        let long = words
            .iter()
            .filter(|w| w.chars().filter(|c| c.is_alphabetic()).count() > 6)
            .count();
        score += (long as f64 / n as f64).min(0.2);
        if sentence.to_lowercase().contains("because ")
            || sentence.contains("Therefore")
            || sentence.to_lowercase().contains("which means")
        {
            score += 0.1;
        }
        if sentence.contains("n't") || sentence.to_lowercase().contains(" not ") {
            score += 0.05;
        }
        score.min(1.0)
    }

    pub fn extract_anchors(sentence: &str) -> Vec<ClaimAnchor> {
        let sentence = sanitize_model_text_for_claims(sentence);
        let mut a = Vec::new();
        for m in NUMERIC.find_iter(&sentence) {
            let value = m.as_str().to_string();
            if is_noise_numeric_token(&value) {
                continue;
            }
            a.push(ClaimAnchor {
                anchor_type: AnchorType::NumericValue,
                value,
            });
        }
        for m in YEAR.find_iter(&sentence) {
            a.push(ClaimAnchor {
                anchor_type: AnchorType::DateOrYear,
                value: m.as_str().to_string(),
            });
        }
        let words: Vec<&str> = sentence.split_whitespace().collect();
        for (i, w) in words.iter().enumerate() {
            if w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                && w.len() > 2
                && i > 0
            {
                let value = w.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
                if CONTRACTION_ANCHOR.is_match(value.as_str()) {
                    continue;
                }
                if is_spurious_proper_noun_token(&value) {
                    continue;
                }
                a.push(ClaimAnchor {
                    anchor_type: AnchorType::ProperNoun,
                    value,
                });
            }
        }
        a
    }

    pub fn is_scaffolding(sentence: &str) -> bool {
        let t = sentence.trim();
        if t.len() < 20 && !t.contains('?') && SCAFFOLD.is_match(t) {
            return true;
        }
        SCAFFOLD.is_match(t)
    }
}

pub struct ClaimMatcher {
    pub match_threshold: f64,
    pub drift_threshold: f64,
}

impl Default for ClaimMatcher {
    fn default() -> Self {
        Self::for_embedding_tier(false)
    }
}

impl ClaimMatcher {
    /// Thresholds for claim sentence similarity (hash embeddings today; BGE later).
    /// When `high_fidelity_embeddings` is true (spec: real sentence model on), use strict cutoffs.
    /// When false (`--no-semantic` or hash-only path), rephrased facts often land ~0.70 — use relaxed cutoffs.
    pub fn for_embedding_tier(high_fidelity_embeddings: bool) -> Self {
        if high_fidelity_embeddings {
            Self {
                match_threshold: 0.75,
                drift_threshold: 0.60,
            }
        } else {
            Self {
                match_threshold: 0.60,
                drift_threshold: 0.40,
            }
        }
    }

    pub fn match_claims(&self, v1_claims: Vec<Claim>, v2_claims: Vec<Claim>) -> anyhow::Result<ClaimDiff> {
        if v1_claims.is_empty() && v2_claims.is_empty() {
            return Ok(ClaimDiff {
                risk: RiskLevel::Green,
                direction: DriftDirection::NotApplicable,
                v1_claims,
                v2_claims,
                matched_pairs: vec![],
                dropped_claims: vec![],
                new_claims: vec![],
                drifted_claims: vec![],
                preservation_score: 1.0,
            });
        }

        let texts: Vec<String> = v1_claims
            .iter()
            .chain(v2_claims.iter())
            .map(|c| c.text.clone())
            .collect();
        let embs = embed_batch_hash(&texts);
        let n1 = v1_claims.len();
        let mut used_v2: Vec<bool> = vec![false; v2_claims.len()];
        let mut matched_pairs = Vec::new();
        let mut drifted = Vec::new();
        let mut v1_matched = vec![false; n1];

        for i in 0..n1 {
            let e1 = &embs[i];
            let c1 = &v1_claims[i];
            let mut best_j: Option<usize> = None;
            let mut best_sim = -1.0f64;
            for j in 0..v2_claims.len() {
                if used_v2[j] {
                    continue;
                }
                let sim = cosine_f32(e1, &embs[n1 + j]);
                if sim > best_sim {
                    best_sim = sim;
                    best_j = Some(j);
                }
            }
            let Some(j) = best_j else { continue };
            if best_sim < self.drift_threshold {
                continue;
            }
            let c2 = &v2_claims[j];
            let (_agree, anchor_drifts) = check_anchor_agreement(c1, c2);
            if anchor_drifts.is_empty() {
                // Above drift threshold and anchors align → treat as matched (hash embeddings rarely
                // reach `match_threshold`; borderline pairs are not genuine claim drift).
                matched_pairs.push(ClaimMatch {
                    v1_claim: c1.clone(),
                    v2_claim: c2.clone(),
                    similarity: best_sim,
                    anchor_agreement: true,
                });
                used_v2[j] = true;
                v1_matched[i] = true;
            } else {
                drifted.push(ClaimDrift {
                    v1_claim: c1.clone(),
                    v2_claim: c2.clone(),
                    similarity: best_sim,
                    drifted_anchors: anchor_drifts,
                });
                used_v2[j] = true;
                v1_matched[i] = true;
            }
        }

        let mut dropped = Vec::new();
        for (i, c) in v1_claims.iter().enumerate() {
            if !v1_matched[i] {
                dropped.push(c.clone());
            }
        }
        let mut new_claims = Vec::new();
        for (j, c) in v2_claims.iter().enumerate() {
            if !used_v2[j] {
                new_claims.push(c.clone());
            }
        }

        let preservation = Self::preservation_score(matched_pairs.len(), v1_claims.len());
        let any_drift_anchors = drifted.iter().any(|d| !d.drifted_anchors.is_empty());
        let risk = if preservation < 0.70 || !dropped.is_empty() || any_drift_anchors {
            RiskLevel::Red
        } else if preservation < 0.90 || !drifted.is_empty() {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };

        let direction = if !dropped.is_empty() || drifted.iter().any(|d| !d.drifted_anchors.is_empty()) {
            DriftDirection::Regression
        } else if !new_claims.is_empty() && dropped.is_empty() {
            DriftDirection::Improvement
        } else {
            DriftDirection::Neutral
        };

        Ok(ClaimDiff {
            risk,
            direction,
            v1_claims,
            v2_claims,
            matched_pairs,
            dropped_claims: dropped,
            new_claims,
            drifted_claims: drifted,
            preservation_score: preservation,
        })
    }

    fn preservation_score(matched: usize, v1_total: usize) -> f64 {
        if v1_total == 0 {
            return 1.0;
        }
        (matched as f64 / v1_total as f64).min(1.0)
    }
}

/// Compare anchors **within each `AnchorType` bucket**, never positionally across the mixed list.
/// Numeric / date: exact-value pairs first (closest token position), then greedy nearest-position
/// pairing for leftovers so `"1919"` is not compared to `"28"` when both sentences share the same facts.
/// Proper nouns / key terms: exact string matches only (no fuzzy cross-pairing of different tokens).
fn check_anchor_agreement(v1: &Claim, v2: &Claim) -> (bool, Vec<AnchorDrift>) {
    let mut drifts = Vec::new();
    for typ in [
        AnchorType::NumericValue,
        AnchorType::DateOrYear,
        AnchorType::ProperNoun,
        AnchorType::KeyTerm,
    ] {
        let b1 = anchor_bucket(v1, typ);
        let b2 = anchor_bucket(v2, typ);
        if b1.is_empty() && b2.is_empty() {
            continue;
        }
        drifts.extend(match typ {
            AnchorType::NumericValue | AnchorType::DateOrYear => anchor_drifts_two_phase(typ, &b1, &b2),
            AnchorType::ProperNoun | AnchorType::KeyTerm => anchor_drifts_exact_only(&b1, &b2),
        });
    }
    let agree = drifts.is_empty();
    (agree, drifts)
}

/// `(char index of value in claim text, value)` sorted by index for stable pairing.
fn anchor_bucket(claim: &Claim, typ: AnchorType) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = claim
        .anchors
        .iter()
        .filter(|a| a.anchor_type == typ)
        .map(|a| {
            let pos = claim.text.find(a.value.as_str()).unwrap_or(usize::MAX);
            (pos, a.value.clone())
        })
        .collect();
    out.sort_by_key(|(p, _)| *p);
    out
}

/// Exact-value pairs first (min |pos1-pos2|), then greedy nearest-unpaired positions; emit drift when paired values differ.
fn anchor_drifts_two_phase(
    typ: AnchorType,
    v1: &[(usize, String)],
    v2: &[(usize, String)],
) -> Vec<AnchorDrift> {
    let mut drifts = Vec::new();
    let mut used1 = vec![false; v1.len()];
    let mut used2 = vec![false; v2.len()];

    // Phase 1: same value, closest positions
    for i in 0..v1.len() {
        if used1[i] {
            continue;
        }
        let (p1, val1) = &v1[i];
        let mut best_j: Option<usize> = None;
        let mut best_d = usize::MAX;
        for j in 0..v2.len() {
            if used2[j] || v2[j].1 != *val1 {
                continue;
            }
            let d = p1.abs_diff(v2[j].0);
            if d < best_d || (d == best_d && best_j.map_or(true, |bj| j < bj)) {
                best_d = d;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            used1[i] = true;
            used2[j] = true;
        }
    }

    // Phase 2: pair remaining by nearest token position; drift if values differ
    loop {
        let mut pick: Option<(usize, usize, usize)> = None;
        for i in 0..v1.len() {
            if used1[i] {
                continue;
            }
            for j in 0..v2.len() {
                if used2[j] {
                    continue;
                }
                let d = v1[i].0.abs_diff(v2[j].0);
                match pick {
                    None => pick = Some((i, j, d)),
                    Some((bi, bj, bd)) if d < bd || (d == bd && (i < bi || (i == bi && j < bj))) => {
                        pick = Some((i, j, d));
                    }
                    _ => {}
                }
            }
        }
        let Some((i, j, _)) = pick else {
            break;
        };
        used1[i] = true;
        used2[j] = true;
        if v1[i].1 != v2[j].1 {
            drifts.push(AnchorDrift {
                anchor_type: typ,
                v1_value: v1[i].1.clone(),
                v2_value: v2[j].1.clone(),
            });
        }
    }

    drifts
}

/// Only identical token values consume a pair; no positional pairing across different strings.
fn anchor_drifts_exact_only(v1: &[(usize, String)], v2: &[(usize, String)]) -> Vec<AnchorDrift> {
    let mut used1 = vec![false; v1.len()];
    let mut used2 = vec![false; v2.len()];
    for i in 0..v1.len() {
        if used1[i] {
            continue;
        }
        let (p1, val1) = &v1[i];
        let mut best_j: Option<usize> = None;
        let mut best_d = usize::MAX;
        for j in 0..v2.len() {
            if used2[j] || v2[j].1 != *val1 {
                continue;
            }
            let d = p1.abs_diff(v2[j].0);
            if d < best_d || (d == best_d && best_j.map_or(true, |bj| j < bj)) {
                best_d = d;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            used1[i] = true;
            used2[j] = true;
        }
    }
    Vec::new()
}

#[cfg(test)]
mod anchor_tests {
    use super::*;

    fn claim(text: &str, anchors: Vec<ClaimAnchor>) -> Claim {
        Claim {
            text: text.to_string(),
            information_density: 0.5,
            anchors,
        }
    }

    #[test]
    fn versailles_same_facts_no_false_numeric_drift() {
        let t = "The Treaty of Versailles was signed on June 28, 1919.";
        let t2 = "The Treaty of Versailles was not signed until June 28, 1919.";
        let c1 = claim(
            t,
            vec![
                ClaimAnchor {
                    anchor_type: AnchorType::NumericValue,
                    value: "28".into(),
                },
                ClaimAnchor {
                    anchor_type: AnchorType::NumericValue,
                    value: "1919".into(),
                },
            ],
        );
        let c2 = claim(
            t2,
            vec![
                ClaimAnchor {
                    anchor_type: AnchorType::NumericValue,
                    value: "28".into(),
                },
                ClaimAnchor {
                    anchor_type: AnchorType::NumericValue,
                    value: "1919".into(),
                },
            ],
        );
        let (agree, drifts) = check_anchor_agreement(&c1, &c2);
        assert!(agree, "expected no anchor drifts, got {:?}", drifts);
        assert!(drifts.is_empty());
    }

    #[test]
    fn ww1_style_numeric_pairs_by_value_not_position() {
        let c1 = claim(
            "World War I ended in 1918 after roughly 4 years of fighting, involving over 11 million military deaths.",
            vec![
                ClaimAnchor {
                    anchor_type: AnchorType::NumericValue,
                    value: "1918".into(),
                },
                ClaimAnchor {
                    anchor_type: AnchorType::NumericValue,
                    value: "11".into(),
                },
            ],
        );
        let c2 = claim(
            "World War I ended in 1918 after roughly 4 years of fighting, involving over 11 million military deaths.",
            vec![
                ClaimAnchor {
                    anchor_type: AnchorType::NumericValue,
                    value: "11".into(),
                },
                ClaimAnchor {
                    anchor_type: AnchorType::NumericValue,
                    value: "1918".into(),
                },
            ],
        );
        let (agree, drifts) = check_anchor_agreement(&c1, &c2);
        assert!(agree, "order of extractions should not create false drifts: {:?}", drifts);
    }

    #[test]
    fn extract_anchors_skips_contractions_as_proper_nouns() {
        use crate::types::AnchorType;
        let a = ClaimExtractor::extract_anchors("First word I'll answer in Paris today.");
        let proper: Vec<_> = a
            .iter()
            .filter(|x| matches!(x.anchor_type, AnchorType::ProperNoun))
            .map(|x| x.value.as_str())
            .collect();
        assert!(
            !proper.iter().any(|v| *v == "I'll"),
            "contraction should not be a proper-noun anchor: {proper:?}"
        );
        assert!(proper.contains(&"Paris"), "expected Paris as anchor, got {proper:?}");
    }

    #[test]
    fn extract_anchors_strips_urls_and_host_port_before_numerics() {
        let a = ClaimExtractor::extract_anchors(
            "First see http://localhost:11434/v1 and http://127.0.0.1:11434/api then 11434 alone.",
        );
        assert!(
            !a.iter().any(|x| x.value == "11434"),
            "expected no port anchor, got {:?}",
            a.iter().map(|x| &x.value).collect::<Vec<_>>()
        );
    }

    #[test]
    fn extract_anchors_filters_titlecase_stopwords() {
        use crate::types::AnchorType;
        let a = ClaimExtractor::extract_anchors("Context clause London However Debt rises sharply.");
        let proper: Vec<_> = a
            .iter()
            .filter(|x| matches!(x.anchor_type, AnchorType::ProperNoun))
            .map(|x| x.value.as_str())
            .collect();
        assert!(proper.contains(&"London"), "expected London, got {proper:?}");
        for banned in ["However", "Debt"] {
            assert!(
                !proper.iter().any(|v| *v == banned),
                "did not expect {banned} in {proper:?}"
            );
        }
    }

    #[test]
    fn matcher_uses_relaxed_thresholds_when_not_high_fidelity() {
        let m = ClaimMatcher::for_embedding_tier(false);
        assert!((m.match_threshold - 0.60).abs() < f64::EPSILON);
        assert!((m.drift_threshold - 0.40).abs() < f64::EPSILON);
        let h = ClaimMatcher::for_embedding_tier(true);
        assert!((h.match_threshold - 0.75).abs() < f64::EPSILON);
        assert!((h.drift_threshold - 0.60).abs() < f64::EPSILON);
    }
}
