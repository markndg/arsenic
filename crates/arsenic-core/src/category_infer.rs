//! Infer [`ProbeCategory`] for single-prompt reconcile runs.

use crate::types::ProbeCategory;

/// Classify a user prompt so claim preservation thresholds match intent.
pub fn infer_probe_category(prompt: &str) -> ProbeCategory {
    let lower = prompt.to_lowercase();

    if lower.contains("json")
        || lower.contains("return a")
        || lower.contains("output as")
        || lower.contains("structured")
    {
        return ProbeCategory::Schema;
    }

    if (lower.starts_with("what")
        || lower.starts_with("who")
        || lower.starts_with("when")
        || lower.starts_with("where")
        || lower.starts_with("how many")
        || lower.starts_with("how much"))
        && prompt.len() < 100
    {
        return ProbeCategory::Factual;
    }

    if lower.contains("respond in")
        || lower.contains("format:")
        || lower.contains("bullet")
        || lower.contains("numbered list")
        || lower.contains("no more than")
        || lower.contains("exactly")
    {
        return ProbeCategory::Instruction;
    }

    if lower.contains("should i")
        || lower.contains("is it ok")
        || lower.contains("controversial")
        || lower.contains("illegal")
    {
        return ProbeCategory::Refusal;
    }

    ProbeCategory::Semantic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_schema_from_json_request() {
        assert!(matches!(
            infer_probe_category("Return a JSON object with name and age"),
            ProbeCategory::Schema
        ));
    }

    #[test]
    fn infers_factual_from_short_what_question() {
        assert!(matches!(
            infer_probe_category("What is the capital of France?"),
            ProbeCategory::Factual
        ));
    }

    #[test]
    fn infers_instruction_from_format_hint() {
        assert!(matches!(
            infer_probe_category("List three items in a numbered list format:"),
            ProbeCategory::Instruction
        ));
    }

    #[test]
    fn infers_refusal_from_sensitive_phrasing() {
        assert!(matches!(
            infer_probe_category("Should I do something illegal?"),
            ProbeCategory::Refusal
        ));
    }

    #[test]
    fn defaults_to_semantic_for_open_ended() {
        assert!(matches!(
            infer_probe_category("Explain what APIs are to a junior developer"),
            ProbeCategory::Semantic
        ));
    }
}
