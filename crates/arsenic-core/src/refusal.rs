use regex::Regex;
use std::sync::LazyLock;

use crate::types::{FinishReason, ModelResponse};

static REFUSAL_PATTERNS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(i can't|i cannot|i'm not able to|i won't|i'm unable to|i don't think i should|i'm going to decline|i am not able to|as an ai|i'm sorry,? i can't)",
    )
    .expect("valid regex")
});

pub struct RefusalDetector;

impl RefusalDetector {
    pub fn is_refusal(response: &ModelResponse) -> bool {
        if matches!(response.finish_reason, FinishReason::Refusal) {
            return true;
        }
        REFUSAL_PATTERNS.is_match(&response.content)
    }
}
