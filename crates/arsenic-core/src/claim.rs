//! Sentence-level claim extraction and cross-matching (v2).

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

use crate::embedding::{cosine_f32, embed_batch_hash};
use crate::types::{
    AnchorDrift, AnchorType, Claim, ClaimAnchor, ClaimDiff, ClaimDrift, ClaimMatch, ClaimMatchKind,
    ClaimMateriality, DriftDirection, Probe, RiskLevel,
};

pub struct ClaimExtractor;

static SCAFFOLD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(great question|in conclusion|it's worth noting|i hope this (helps|explanation)|feel free to ask|do you have any specific questions)")
        .expect("regex")
});

static NUMERIC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b\d[\d.,]*\b").expect("regex"));
static YEAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(19|20)\d{2}\b").expect("regex"));

/// Strip URLs and `host:port` fragments so endpoint metadata never becomes numeric anchors.
static HTTP_URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>\])'"]+"#).expect("regex"));
static HOST_PORT: LazyLock<Regex> = LazyLock::new(|| {
    // `localhost:11434`, `127.0.0.1:11434`, or any IPv4:port (endpoint echoes in model text).
    Regex::new(
        "(?i)\\b(?:localhost|127\\.0\\.0\\.1|(?:\\d{1,3}\\.){3}\\d{1,3})\\s*:\\s*\\d{2,5}\\b",
    )
    .expect("regex")
});
static WS_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new("[ \t]{2,}").expect("regex"));

static MD_BOLD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*\*([^*]+)\*\*").expect("regex"));
static MD_ITALIC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*([^*]+)\*").expect("regex"));
static MD_CODE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"`([^`]+)`").expect("regex"));
static MD_HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^#{1,6}\s+").expect("regex"));
static ORDERED_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*\d+[\.)]\s+").expect("regex"));
static BULLET_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*[-*•]\s+").expect("regex"));
static CODE_FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^```\w*\s*$").expect("regex"));
static CONVO_SCAFFOLD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(if you want|let me know|feel free|i can also|would you like|happy to help)")
        .expect("regex")
});

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
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
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

/// Strip markdown/list scaffolding so formatting changes do not appear as claim loss.
pub fn normalize_line_for_claims(line: &str) -> String {
    if CODE_FENCE.is_match(line.trim()) {
        return String::new();
    }
    let mut l = line.to_string();
    l = MD_HEADING.replace(&l, "").to_string();
    l = ORDERED_MARKER.replace(&l, "").to_string();
    l = BULLET_MARKER.replace(&l, "").to_string();
    l = MD_BOLD.replace_all(&l, "$1").to_string();
    l = MD_ITALIC.replace_all(&l, "$1").to_string();
    l = MD_CODE.replace_all(&l, "$1").to_string();
    l.trim().to_string()
}

pub fn normalize_text_for_claims(text: &str) -> String {
    sanitize_model_text_for_claims(text)
        .lines()
        .map(normalize_line_for_claims)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_anchor_value(raw: &str) -> String {
    raw.trim()
        .trim_matches(|c: char| c == '*' || c == '`' || c == ',')
        .to_lowercase()
}

fn anchor_values_materially_differ(v1: &str, v2: &str) -> bool {
    normalize_anchor_value(v1) != normalize_anchor_value(v2)
}

pub fn classify_claim_materiality(text: &str, anchors: &[ClaimAnchor]) -> ClaimMateriality {
    let t = text.trim();
    if t.is_empty() {
        return ClaimMateriality::Formatting;
    }
    if ClaimExtractor::is_scaffolding(t) || CONVO_SCAFFOLD.is_match(t) {
        return ClaimMateriality::Scaffolding;
    }
    if is_json_or_field_fragment(t) || is_heading_label_only(t) {
        return ClaimMateriality::Formatting;
    }
    if anchors.iter().any(|a| {
        matches!(
            a.anchor_type,
            AnchorType::NumericValue | AnchorType::DateOrYear
        )
    }) {
        return ClaimMateriality::Material;
    }
    if anchors
        .iter()
        .any(|a| matches!(a.anchor_type, AnchorType::ProperNoun))
        && ClaimExtractor::information_density(t) >= 0.25
    {
        return ClaimMateriality::Material;
    }
    if t.len() < 24 && !t.contains('?') {
        return ClaimMateriality::Formatting;
    }
    ClaimMateriality::Material
}

fn is_json_or_field_fragment(text: &str) -> bool {
    let t = text.trim();
    t.starts_with('{')
        || (t.starts_with('"') && t.contains(':'))
        || t.starts_with("\"band\"")
        || t.starts_with("\"mark\"")
        || t.starts_with("\"criteria_met\"")
}

fn is_heading_label_only(text: &str) -> bool {
    let t = text.trim();
    if t.starts_with("---") {
        return true;
    }
    let words: Vec<_> = t.split_whitespace().collect();
    words.len() <= 4
        && !NUMERIC.is_match(t)
        && !t.contains('?')
        && t.chars().filter(|c| c.is_alphabetic()).count() < 30
        && (t.contains("Introduction")
            || t.contains("Conclusion")
            || t.starts_with("Step ")
            || t.starts_with("Method "))
}

/// Ports and other numeric tokens that are infrastructure, not factual claims.
fn is_noise_numeric_token(raw: &str) -> bool {
    let t = raw.trim().trim_end_matches([',', '.']);
    matches!(
        t,
        "11434"
            | "11435"
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

static CHEM_ELEMENT_RUN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:[A-Z][a-z]?\d*){2,}$").expect("regex"));

static KNOWN_UPPERCASE_ACRONYMS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "REST", "JSON", "HTML", "HTTP", "HTTPS", "API", "SQL", "URI", "URL", "TLS", "SSL", "GPU",
        "CPU", "RAM", "AWS", "GCP", "CLI", "GPT", "LLM", "OCR", "PDF", "XML", "YAML", "TOML",
        "JWT", "SSH", "DNS", "TCP", "UDP", "IDE", "SDK", "SaaS", "iOS", "macOS",
    ]
    .into_iter()
    .collect()
});

/// Whether a capitalised token is too generic to use as a mutation anchor or required value.
pub fn is_spurious_anchor_value(word: &str) -> bool {
    if looks_like_chemistry_formula(word) || is_noise_uppercase_abbrev(word) {
        return true;
    }
    is_spurious_proper_noun_token(word)
}

fn has_subscript_digit(s: &str) -> bool {
    s.chars().any(|c| {
        matches!(c, '₀' | '₁' | '₂' | '₃' | '₄' | '₅' | '₆' | '₇' | '₈' | '₉')
            || ('\u{2080}'..='\u{2089}').contains(&c)
    })
}

fn looks_like_chemistry_formula(word: &str) -> bool {
    let value = word.trim();
    if value.is_empty() {
        return false;
    }
    if has_subscript_digit(value) {
        return true;
    }
    let alnum: String = value
        .chars()
        .filter(|c| c.is_alphanumeric() || has_subscript_digit(&c.to_string()))
        .collect();
    if has_subscript_digit(&alnum) {
        return true;
    }
    if !(alnum.chars().any(|c| c.is_ascii_lowercase()) || alnum.chars().any(|c| c.is_ascii_digit()))
    {
        return false;
    }
    if CHEM_ELEMENT_RUN.is_match(&alnum) {
        return true;
    }
    let letters: String = alnum.chars().filter(|c| c.is_alphabetic()).collect();
    // Hill-style inorganic tokens (e.g. NaOH, NaCl).
    if (3..=8).contains(&letters.len())
        && letters.chars().any(|c| c.is_ascii_uppercase())
        && letters.chars().any(|c| c.is_ascii_lowercase())
    {
        let upper = letters.chars().filter(|c| c.is_ascii_uppercase()).count();
        let lower = letters.chars().filter(|c| c.is_ascii_lowercase()).count();
        if upper >= 2 && lower >= 1 {
            return true;
        }
    }
    false
}

fn is_noise_uppercase_abbrev(word: &str) -> bool {
    let value = word.trim_matches(|c: char| c.is_alphanumeric());
    if value.is_empty() {
        return false;
    }
    if value.len() < 3 && value.chars().all(|c| c.is_ascii_uppercase()) {
        return true;
    }
    if (2..=4).contains(&value.len())
        && value.chars().all(|c| c.is_ascii_uppercase())
        && !KNOWN_UPPERCASE_ACRONYMS.contains(value)
    {
        return true;
    }
    false
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
        let text = normalize_text_for_claims(text);
        let mut out = Vec::new();
        for sent in text.unicode_sentences() {
            let s = sent.trim();
            if s.len() < 8 {
                continue;
            }
            if Self::is_scaffolding(s) || CONVO_SCAFFOLD.is_match(s) {
                continue;
            }
            if is_heading_label_only(s) {
                continue;
            }
            let density = Self::information_density(s);
            if density < 0.12 {
                continue;
            }
            let anchors = Self::extract_anchors(s);
            if classify_claim_materiality(s, &anchors) == ClaimMateriality::Formatting {
                continue;
            }
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
            if w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && w.len() > 2 && i > 0 {
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

    pub fn match_claims(
        &self,
        v1_claims: Vec<Claim>,
        v2_claims: Vec<Claim>,
        probe: &Probe,
    ) -> anyhow::Result<ClaimDiff> {
        let category = probe.category;
        let preservation_threshold = category.preservation_threshold();
        let preservation_amber = category.preservation_amber_threshold();
        let dropped_force_red = category.dropped_claims_force_red()
            && probe.claim_anchor_policy != crate::types::ClaimAnchorPolicy::Lenient;

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
                preservation_threshold,
                material_preservation_score: 1.0,
                has_material_anchor_drift: false,
                material_dropped_claims: vec![],
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
            let material_drifts: Vec<_> = anchor_drifts
                .iter()
                .filter(|d| anchor_values_materially_differ(&d.v1_value, &d.v2_value))
                .cloned()
                .collect();
            if material_drifts.is_empty() {
                let match_kind = if best_sim >= self.match_threshold {
                    ClaimMatchKind::Semantic
                } else {
                    ClaimMatchKind::AnchorOnly
                };
                matched_pairs.push(ClaimMatch {
                    v1_claim: c1.clone(),
                    v2_claim: c2.clone(),
                    similarity: best_sim,
                    anchor_agreement: true,
                    match_kind,
                });
                used_v2[j] = true;
                v1_matched[i] = true;
            } else if classify_claim_materiality(&c1.text, &c1.anchors)
                == ClaimMateriality::Material
            {
                drifted.push(ClaimDrift {
                    v1_claim: c1.clone(),
                    v2_claim: c2.clone(),
                    similarity: best_sim,
                    drifted_anchors: material_drifts,
                });
                used_v2[j] = true;
                v1_matched[i] = true;
            } else {
                matched_pairs.push(ClaimMatch {
                    v1_claim: c1.clone(),
                    v2_claim: c2.clone(),
                    similarity: best_sim,
                    anchor_agreement: false,
                    match_kind: ClaimMatchKind::AnchorOnly,
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

        let material_v1: Vec<_> = v1_claims
            .iter()
            .filter(|c| {
                classify_claim_materiality(&c.text, &c.anchors) == ClaimMateriality::Material
            })
            .collect();
        let material_matched = matched_pairs
            .iter()
            .filter(|m| {
                classify_claim_materiality(&m.v1_claim.text, &m.v1_claim.anchors)
                    == ClaimMateriality::Material
                    && (m.match_kind == ClaimMatchKind::Semantic
                        || (m.match_kind == ClaimMatchKind::AnchorOnly
                            && m.anchor_agreement
                            && m.similarity >= self.drift_threshold))
            })
            .count();
        let material_preservation =
            Self::preservation_score(material_matched, material_v1.len().max(1));

        let material_dropped: Vec<Claim> = dropped
            .iter()
            .filter(|c| {
                classify_claim_materiality(&c.text, &c.anchors) == ClaimMateriality::Material
            })
            .cloned()
            .collect();

        let has_material_anchor_drift = drifted.iter().any(|d| {
            !d.drifted_anchors.is_empty()
                && classify_claim_materiality(&d.v1_claim.text, &d.v1_claim.anchors)
                    == ClaimMateriality::Material
        });

        let any_drift_anchors = has_material_anchor_drift;
        let risk = if material_preservation < preservation_threshold
            || (dropped_force_red && !material_dropped.is_empty())
            || any_drift_anchors
        {
            RiskLevel::Red
        } else if preservation < preservation_amber
            || (!material_dropped.is_empty() && !dropped_force_red)
            || !drifted.is_empty()
        {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };

        let direction = if !material_dropped.is_empty() || any_drift_anchors {
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
            preservation_threshold,
            material_preservation_score: material_preservation,
            has_material_anchor_drift,
            material_dropped_claims: material_dropped,
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
            AnchorType::NumericValue | AnchorType::DateOrYear => {
                anchor_drifts_two_phase(typ, &b1, &b2)
            }
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
                    Some((bi, bj, bd))
                        if d < bd || (d == bd && (i < bi || (i == bi && j < bj))) =>
                    {
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
    use crate::types::ProbeCategory;
    use crate::types::{ClaimAnchorPolicy, PresentationDriftPolicy, Probe, ProbeSource};

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
        assert!(
            agree,
            "order of extractions should not create false drifts: {:?}",
            drifts
        );
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
            !proper.contains(&"I'll"),
            "contraction should not be a proper-noun anchor: {proper:?}"
        );
        assert!(
            proper.contains(&"Paris"),
            "expected Paris as anchor, got {proper:?}"
        );
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
        let a =
            ClaimExtractor::extract_anchors("Context clause London However Debt rises sharply.");
        let proper: Vec<_> = a
            .iter()
            .filter(|x| matches!(x.anchor_type, AnchorType::ProperNoun))
            .map(|x| x.value.as_str())
            .collect();
        assert!(
            proper.contains(&"London"),
            "expected London, got {proper:?}"
        );
        for banned in ["However", "Debt"] {
            assert!(
                !proper.contains(&banned),
                "did not expect {banned} in {proper:?}"
            );
        }
    }

    #[test]
    fn preservation_thresholds_by_category() {
        assert!((ProbeCategory::Factual.preservation_threshold() - 0.70).abs() < f64::EPSILON);
        assert!((ProbeCategory::Schema.preservation_threshold() - 0.70).abs() < f64::EPSILON);
        assert!((ProbeCategory::Instruction.preservation_threshold() - 0.70).abs() < f64::EPSILON);
        assert!((ProbeCategory::Semantic.preservation_threshold() - 0.50).abs() < f64::EPSILON);
        assert!((ProbeCategory::Tone.preservation_threshold() - 0.50).abs() < f64::EPSILON);
        assert!(ProbeCategory::Factual.dropped_claims_force_red());
        assert!(!ProbeCategory::Tone.dropped_claims_force_red());
    }

    #[test]
    fn chemistry_formulae_are_spurious_anchors() {
        for w in ["C₁₇H₃₃COOK", "NaOH", "CO₂", "NaCl"] {
            assert!(
                is_spurious_anchor_value(w),
                "expected chemistry token {w} to be filtered"
            );
        }
        assert!(!is_spurious_anchor_value("REST"));
    }

    #[test]
    fn section_header_tokens_are_spurious_anchors() {
        for w in [
            "Select",
            "Choose",
            "Static",
            "System",
            "Management",
            "Experience",
            "Requirements",
            "Existing",
            "Social",
            "Tool",
            "Innovation",
            "Communication",
            "Adaptation",
            "Considerations",
            "Impact",
            "Future",
            "Potential",
        ] {
            assert!(is_spurious_anchor_value(w), "expected {w} to be filtered");
        }
    }

    #[test]
    fn bold_paris_matches_plain_paris() {
        let v1 = ClaimExtractor::extract("The capital of France is Paris.");
        let v2 = ClaimExtractor::extract("The capital of France is **Paris**.");
        let probe = Probe {
            id: uuid::Uuid::new_v4(),
            name: "capitals".into(),
            category: ProbeCategory::Factual,
            prompt: "capital?".into(),
            system_prompt: None,
            known_answer: Some("Paris".into()),
            expected_schema: None,
            instructions: vec![],
            tags: vec![],
            source: ProbeSource::Standard,
            expected_verbosity: None,
            expected_tone: None,
            refusal_expectation: None,
            mutation_hint: None,
            custom_assertions: vec![],
            format_sensitive: false,
            structure_sensitive: false,
            claim_anchor_policy: ClaimAnchorPolicy::Balanced,
            presentation_drift: PresentationDriftPolicy::Review,
            latency_slo_ms: None,
        };
        let diff = ClaimMatcher::default()
            .match_claims(v1, v2, &probe)
            .expect("match");
        assert!(diff.material_preservation_score >= 0.99);
        assert!(!diff.has_material_anchor_drift);
    }

    #[test]
    fn numeric_bold_matches_plain() {
        let v1 = ClaimExtractor::extract("The answer is 136.");
        let v2 = ClaimExtractor::extract("The answer is **136**.");
        let probe = Probe {
            id: uuid::Uuid::new_v4(),
            name: "num".into(),
            category: ProbeCategory::Factual,
            prompt: "n?".into(),
            system_prompt: None,
            known_answer: None,
            expected_schema: None,
            instructions: vec![],
            tags: vec![],
            source: ProbeSource::Standard,
            expected_verbosity: None,
            expected_tone: None,
            refusal_expectation: None,
            mutation_hint: None,
            custom_assertions: vec![],
            format_sensitive: false,
            structure_sensitive: false,
            claim_anchor_policy: ClaimAnchorPolicy::Balanced,
            presentation_drift: PresentationDriftPolicy::Review,
            latency_slo_ms: None,
        };
        let diff = ClaimMatcher::default()
            .match_claims(v1, v2, &probe)
            .expect("match");
        assert!(diff.material_preservation_score >= 0.99);
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
