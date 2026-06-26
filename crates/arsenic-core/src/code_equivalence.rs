//! Code-body extraction and equivalence — separates markdown presentation from executable semantics.

use crate::types::Probe;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeEquivalence {
    Exact,
    EquivalentFormatting,
    Different,
}

impl From<CodeEquivalence> for crate::types::CodeEquivalenceLevel {
    fn from(value: CodeEquivalence) -> Self {
        match value {
            CodeEquivalence::Exact => Self::Exact,
            CodeEquivalence::EquivalentFormatting => Self::EquivalentFormatting,
            CodeEquivalence::Different => Self::Different,
        }
    }
}

pub fn build_code_equivalence_diff(
    probe: &Probe,
    v1: &str,
    v2: &str,
) -> Option<crate::types::CodeEquivalenceDiff> {
    if !is_code_context(probe, v1, v2) {
        return None;
    }
    let equivalence = compare_code_equivalence(v1, v2).into();
    Some(crate::types::CodeEquivalenceDiff {
        equivalence,
        applies: true,
    })
}

pub fn code_formatting_only(dims: &crate::types::ProbeDimensions) -> bool {
    dims.code_equivalence.as_ref().is_some_and(|c| {
        c.applies
            && matches!(
                c.equivalence,
                crate::types::CodeEquivalenceLevel::Exact
                    | crate::types::CodeEquivalenceLevel::EquivalentFormatting
            )
    })
}

/// Whether this probe expects code output.
pub fn is_code_probe(probe: &Probe) -> bool {
    probe
        .tags
        .iter()
        .any(|t| t == "code-generation" || t.starts_with("code-"))
        || probe.expected_schema.is_some()
        || prompt_expects_code(&probe.prompt)
}

fn prompt_expects_code(prompt: &str) -> bool {
    let lower = prompt.to_lowercase();
    lower.contains("return only the sql")
        || lower.contains("return only the code")
        || lower.contains("return only the function")
        || lower.contains("return only valid json")
        || lower.contains("return only the assignment")
        || lower.contains("return only a ")
        || lower.contains("just the pattern")
        || (lower.contains("write a") && lower.contains("no explanation"))
}

/// Heuristic: content looks like code even without markdown fences.
pub fn looks_like_code_content(text: &str) -> bool {
    let body = extract_code_body(text);
    let s = body.trim();
    if s.is_empty() {
        return false;
    }
    if s.contains("```") {
        return true;
    }
    if s.starts_with('{') || s.starts_with('[') {
        return true;
    }
    let lower = s.to_lowercase();
    [
        "select ",
        "from ",
        "def ",
        "class ",
        "const ",
        "function ",
        "import ",
        "=>",
        "fn ",
        "public ",
        "return ",
        "var ",
        "let ",
    ]
    .iter()
    .any(|sig| lower.contains(sig))
}

pub fn is_code_context(probe: &Probe, v1: &str, v2: &str) -> bool {
    is_code_probe(probe) || looks_like_code_content(v1) || looks_like_code_content(v2)
}

/// Strip surrounding markdown fences and an optional language tag line.
pub fn extract_code_body(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(inner) = extract_single_fenced_block(trimmed) {
        return inner;
    }
    trimmed.to_string()
}

fn extract_single_fenced_block(text: &str) -> Option<String> {
    let start = text.find("```")?;
    let after_open = &text[start + 3..];
    let after_open = after_open.strip_prefix('\n').unwrap_or(after_open);
    let lang_end = after_open.find('\n').unwrap_or(0);
    let first_line = after_open[..lang_end].trim();
    let body_start = if first_line.is_empty()
        || is_language_tag(first_line)
        || first_line
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        if first_line.is_empty() {
            0
        } else {
            lang_end + 1
        }
    } else {
        0
    };
    let rest = &after_open[body_start..];
    let close = rest.find("```")?;
    Some(rest[..close].trim_end().to_string())
}

fn is_language_tag(tag: &str) -> bool {
    matches!(
        tag.to_lowercase().as_str(),
        "sql"
            | "python"
            | "py"
            | "javascript"
            | "js"
            | "json"
            | "typescript"
            | "ts"
            | "rust"
            | "go"
            | "java"
            | "c"
            | "cpp"
            | "bash"
            | "sh"
            | "html"
            | "css"
            | "yaml"
            | "yml"
            | "xml"
            | "markdown"
            | "md"
    )
}

pub fn compare_code_equivalence(v1: &str, v2: &str) -> CodeEquivalence {
    let b1 = extract_code_body(v1);
    let b2 = extract_code_body(v2);

    if b1 == b2 {
        return CodeEquivalence::Exact;
    }

    let n1 = normalize_code_formatting(&b1);
    let n2 = normalize_code_formatting(&b2);
    if n1 == n2 {
        return CodeEquivalence::EquivalentFormatting;
    }

    let t1 = code_tokens(&n1);
    let t2 = code_tokens(&n2);
    if !t1.is_empty() && t1 == t2 {
        return CodeEquivalence::EquivalentFormatting;
    }

    CodeEquivalence::Different
}

/// Normalize formatting-only differences: line endings, trailing space, blank lines, dedent.
pub fn normalize_code_formatting(code: &str) -> String {
    let normalized = code.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines: Vec<String> = normalized
        .lines()
        .map(|l| l.trim_end().to_string())
        .collect();

    while lines.first().is_some_and(|l| l.trim().is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }

    lines = collapse_blank_lines(lines);
    lines = dedent_lines(lines);
    lines.join("\n")
}

fn collapse_blank_lines(lines: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut prev_blank = false;
    for line in lines {
        let blank = line.trim().is_empty();
        if blank {
            if !prev_blank {
                out.push(String::new());
            }
            prev_blank = true;
        } else {
            out.push(line);
            prev_blank = false;
        }
    }
    out
}

fn dedent_lines(lines: Vec<String>) -> Vec<String> {
    let min_indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| *c == ' ' || *c == '\t').count())
        .min()
        .unwrap_or(0);
    if min_indent == 0 {
        return lines;
    }
    lines
        .into_iter()
        .map(|l| {
            if l.trim().is_empty() {
                l
            } else {
                l.chars().skip(min_indent).collect()
            }
        })
        .collect()
}

fn code_tokens(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        } else if "(){}[];,.*=<>!+-/\\|&|^~?:\"'`".contains(ch) {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Prose outside fenced code blocks — used for "code only" instruction checks.
pub fn non_code_prose(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.contains("```") {
        let mut rest = trimmed;
        let mut prose = String::new();
        while let Some(start) = rest.find("```") {
            prose.push_str(rest[..start].trim());
            if !prose.is_empty() && !prose.ends_with('\n') {
                prose.push('\n');
            }
            rest = &rest[start + 3..];
            rest = rest.strip_prefix('\n').unwrap_or(rest);
            if let Some(lang_end) = rest.find('\n') {
                let tag = rest[..lang_end].trim();
                if !tag.is_empty()
                    && (is_language_tag(tag)
                        || tag
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
                {
                    rest = &rest[lang_end + 1..];
                }
            }
            if let Some(close) = rest.find("```") {
                rest = &rest[close + 3..];
            } else {
                return prose.trim().to_string();
            }
        }
        prose.push_str(rest.trim());
        return prose.trim().to_string();
    }
    if looks_like_code_content(trimmed) && is_mostly_code(trimmed) {
        return String::new();
    }
    trimmed.to_string()
}

fn is_mostly_code(text: &str) -> bool {
    let body = extract_code_body(text);
    let code_lines = body.lines().filter(|l| !l.trim().is_empty()).count();
    let prose_indicators = [
        "here is",
        "here's",
        "this code",
        "the following",
        "explanation",
    ];
    let lower = text.to_lowercase();
    code_lines >= 1 && !prose_indicators.iter().any(|p| lower.contains(p))
}

pub fn prompt_requires_code_only(prompt: &str) -> bool {
    let lower = prompt.to_lowercase();
    lower.contains("return only")
        || lower.contains("no explanation")
        || lower.contains("only the function")
        || lower.contains("only the sql")
        || lower.contains("only the code")
        || lower.contains("just the ")
}

pub fn formatting_blocks(probe: &Probe) -> bool {
    probe.format_sensitive
        || probe.structure_sensitive
        || probe.presentation_drift == crate::types::PresentationDriftPolicy::Blocking
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fenced_sql_vs_plain_sql_is_equivalent() {
        let v1 = "```sql\nSELECT users.name, orders.total\nFROM users\nJOIN orders ON users.id = orders.user_id;\n```";
        let v2 = "SELECT users.name, orders.total\nFROM users\nJOIN orders ON users.id = orders.user_id;";
        assert!(matches!(
            compare_code_equivalence(v1, v2),
            CodeEquivalence::Exact | CodeEquivalence::EquivalentFormatting
        ));
    }

    #[test]
    fn fenced_python_vs_plain_python() {
        let v1 = "```python\nprint(\"Hello\")\n```";
        let v2 = "print(\"Hello\")";
        assert!(matches!(
            compare_code_equivalence(v1, v2),
            CodeEquivalence::Exact | CodeEquivalence::EquivalentFormatting
        ));
    }

    #[test]
    fn fenced_json_vs_plain_json() {
        let v1 = "```json\n{\"a\":1}\n```";
        let v2 = "{\"a\":1}";
        assert!(matches!(
            compare_code_equivalence(v1, v2),
            CodeEquivalence::Exact | CodeEquivalence::EquivalentFormatting
        ));
    }

    #[test]
    fn whitespace_only_sql_equivalent() {
        let v1 = "SELECT *\nFROM users;";
        let v2 = "SELECT * FROM users;";
        assert_eq!(
            compare_code_equivalence(v1, v2),
            CodeEquivalence::EquivalentFormatting
        );
    }

    #[test]
    fn indentation_only_python_equivalent() {
        let v1 = "def f():\n    return 1";
        let v2 = "def f():\n        return 1";
        assert_eq!(
            compare_code_equivalence(v1, v2),
            CodeEquivalence::EquivalentFormatting
        );
    }

    #[test]
    fn language_tag_change_equivalent() {
        let v1 = "```python\nx = 1\n```";
        let v2 = "```py\nx = 1\n```";
        assert!(matches!(
            compare_code_equivalence(v1, v2),
            CodeEquivalence::Exact | CodeEquivalence::EquivalentFormatting
        ));
    }

    #[test]
    fn semantic_sql_join_change_is_different() {
        let v1 = "SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id";
        let v2 = "SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id";
        assert_eq!(compare_code_equivalence(v1, v2), CodeEquivalence::Different);
    }

    #[test]
    fn semantic_operator_change_is_different() {
        let v1 = "SELECT * FROM t WHERE x >= 5";
        let v2 = "SELECT * FROM t WHERE x > 5";
        assert_eq!(compare_code_equivalence(v1, v2), CodeEquivalence::Different);
    }

    #[test]
    fn semantic_python_change_is_different() {
        let v1 = "return x + 1";
        let v2 = "return x + 2";
        assert_eq!(compare_code_equivalence(v1, v2), CodeEquivalence::Different);
    }

    #[test]
    fn non_code_prose_empty_for_plain_code() {
        assert!(non_code_prose("SELECT 1;").trim().is_empty());
        assert!(non_code_prose("```sql\nSELECT 1;\n```").trim().is_empty());
    }

    #[test]
    fn non_code_prose_detects_explanation() {
        let v = "```python\nprint(1)\n```\n\nHere is the code you asked for.";
        assert!(non_code_prose(v).contains("Here is"));
    }
}
