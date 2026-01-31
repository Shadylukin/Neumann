// SPDX-License-Identifier: MIT OR Apache-2.0
//! Input validation for shell commands.

use rustyline::validate::{ValidationContext, ValidationResult, Validator};

/// Validates Neumann shell input.
#[derive(Debug, Default)]
pub struct NeumannValidator;

#[allow(clippy::unused_self)]
impl NeumannValidator {
    /// Creates a new validator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Validates the input line.
    fn validate_input(&self, input: &str) -> ValidationResult {
        let trimmed = input.trim();

        // Empty input is valid (will be handled as no-op)
        if trimmed.is_empty() {
            return ValidationResult::Valid(None);
        }

        // Check for unclosed strings
        if has_unclosed_string(trimmed) {
            return ValidationResult::Incomplete;
        }

        // Check for unclosed brackets/parens
        if has_unclosed_brackets(trimmed) {
            return ValidationResult::Incomplete;
        }

        // Check for trailing operators that suggest continuation
        if needs_continuation(trimmed) {
            return ValidationResult::Incomplete;
        }

        ValidationResult::Valid(None)
    }
}

/// Checks if the input has unclosed string literals.
fn has_unclosed_string(input: &str) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    let mut prev_char = '\0';

    for c in input.chars() {
        match c {
            '\'' if !in_double && prev_char != '\\' => in_single = !in_single,
            '"' if !in_single && prev_char != '\\' => in_double = !in_double,
            _ => {},
        }
        prev_char = c;
    }

    in_single || in_double
}

/// Checks if the input has unclosed brackets or parentheses.
fn has_unclosed_brackets(input: &str) -> bool {
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;
    let mut in_string = false;
    let mut string_char = '"';

    for c in input.chars() {
        if in_string {
            if c == string_char {
                in_string = false;
            }
            continue;
        }

        match c {
            '\'' | '"' => {
                in_string = true;
                string_char = c;
            },
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            '{' => brace_depth += 1,
            '}' => brace_depth -= 1,
            _ => {},
        }
    }

    paren_depth > 0 || bracket_depth > 0 || brace_depth > 0
}

/// Checks if the input ends with an operator suggesting continuation.
fn needs_continuation(input: &str) -> bool {
    let trimmed = input.trim_end();
    trimmed.ends_with(',')
        || trimmed.ends_with("AND")
        || trimmed.ends_with("OR")
        || trimmed.ends_with("->")
        || trimmed.ends_with(':')
}

impl Validator for NeumannValidator {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        Ok(self.validate_input(ctx.input()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input_valid() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("");
        assert!(matches!(result, ValidationResult::Valid(_)));
    }

    #[test]
    fn test_complete_query_valid() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("SELECT * FROM users");
        assert!(matches!(result, ValidationResult::Valid(_)));
    }

    #[test]
    fn test_unclosed_single_quote() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("SELECT * FROM 'users");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn test_unclosed_double_quote() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("SELECT * FROM \"users");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn test_closed_quotes_valid() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("EMBED STORE 'key' [0.1, 0.2]");
        assert!(matches!(result, ValidationResult::Valid(_)));
    }

    #[test]
    fn test_unclosed_paren() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("CREATE TABLE users (id INT");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn test_unclosed_bracket() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("EMBED STORE 'key' [0.1, 0.2");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn test_unclosed_brace() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("NODE CREATE person {name: 'Alice'");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn test_trailing_comma() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("INSERT INTO users VALUES (1,");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn test_trailing_and() {
        let validator = NeumannValidator::new();
        let result = validator.validate_input("SELECT * FROM users WHERE id = 1 AND");
        assert!(matches!(result, ValidationResult::Incomplete));
    }

    #[test]
    fn test_has_unclosed_string() {
        assert!(has_unclosed_string("'hello"));
        assert!(has_unclosed_string("\"hello"));
        assert!(!has_unclosed_string("'hello'"));
        assert!(!has_unclosed_string("\"hello\""));
    }

    #[test]
    fn test_has_unclosed_brackets() {
        assert!(has_unclosed_brackets("("));
        assert!(has_unclosed_brackets("["));
        assert!(has_unclosed_brackets("{"));
        assert!(!has_unclosed_brackets("()"));
        assert!(!has_unclosed_brackets("[]"));
        assert!(!has_unclosed_brackets("{}"));
    }

    #[test]
    fn test_needs_continuation() {
        assert!(needs_continuation("SELECT *,"));
        assert!(needs_continuation("WHERE x AND"));
        assert!(needs_continuation("EDGE CREATE 1 ->"));
        assert!(!needs_continuation("SELECT * FROM users"));
    }
}
