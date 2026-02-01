// SPDX-License-Identifier: MIT OR Apache-2.0
//! Input handling for the shell (completion, highlighting, validation).

mod completer;
mod highlighter;
mod validator;

pub use completer::NeumannCompleter;
pub use highlighter::NeumannHighlighter;
pub use validator::NeumannValidator;

use crate::style::Theme;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::Helper;

/// Combined helper providing completion, highlighting, hints, and validation.
pub struct NeumannHelper {
    completer: NeumannCompleter,
    highlighter: NeumannHighlighter,
    validator: NeumannValidator,
}

impl NeumannHelper {
    /// Creates a new helper with the given theme.
    #[must_use]
    pub fn new(theme: Theme) -> Self {
        Self {
            completer: NeumannCompleter::new(),
            highlighter: NeumannHighlighter::new(theme),
            validator: NeumannValidator::new(),
        }
    }

    /// Updates the list of available tables for completion.
    pub fn set_tables(&mut self, tables: Vec<String>) {
        self.completer.set_tables(tables);
    }
}

impl Default for NeumannHelper {
    fn default() -> Self {
        Self::new(Theme::auto())
    }
}

impl Helper for NeumannHelper {}

impl Completer for NeumannHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        self.completer.complete(line, pos, ctx)
    }
}

impl Highlighter for NeumannHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> std::borrow::Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize, kind: CmdKind) -> bool {
        self.highlighter.highlight_char(line, pos, kind)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> std::borrow::Cow<'b, str> {
        self.highlighter.highlight_prompt(prompt, default)
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> std::borrow::Cow<'h, str> {
        self.highlighter.highlight_hint(hint)
    }

    fn highlight_candidate<'c>(
        &self,
        candidate: &'c str,
        completion: rustyline::CompletionType,
    ) -> std::borrow::Cow<'c, str> {
        self.highlighter.highlight_candidate(candidate, completion)
    }
}

impl Hinter for NeumannHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        None
    }
}

impl Validator for NeumannHelper {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        self.validator.validate(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helper_creation() {
        let _helper = NeumannHelper::new(Theme::plain());
    }

    #[test]
    fn test_helper_default() {
        let _helper = NeumannHelper::default();
    }

    #[test]
    fn test_helper_set_tables() {
        let mut helper = NeumannHelper::new(Theme::plain());
        helper.set_tables(vec!["users".to_string(), "orders".to_string()]);
    }

    #[test]
    fn test_helper_completer_trait() {
        let helper = NeumannHelper::new(Theme::plain());
        let history = rustyline::history::DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);
        let result = helper.complete("SELECT ", 7, &ctx);
        assert!(result.is_ok());
        let (start, completions) = result.unwrap();
        assert_eq!(start, 7);
        assert!(!completions.is_empty());
    }

    #[test]
    fn test_helper_highlighter_trait() {
        let helper = NeumannHelper::new(Theme::plain());

        // Test highlight
        let highlighted = helper.highlight("SELECT * FROM users", 0);
        assert!(!highlighted.is_empty());

        // Test highlight_char
        let needs_highlight = helper.highlight_char("SELECT", 0, CmdKind::MoveCursor);
        // Result depends on implementation
        let _ = needs_highlight;

        // Test highlight_prompt
        let prompt = helper.highlight_prompt("neumann> ", false);
        assert!(!prompt.is_empty());

        // Test highlight_hint
        let hint = helper.highlight_hint("users");
        assert!(!hint.is_empty());

        // Test highlight_candidate
        let candidate = helper.highlight_candidate("SELECT", rustyline::CompletionType::List);
        assert!(!candidate.is_empty());
    }

    #[test]
    fn test_helper_hinter_trait() {
        let helper = NeumannHelper::new(Theme::plain());
        let history = rustyline::history::DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);
        let hint = helper.hint("SELECT", 6, &ctx);
        assert!(hint.is_none()); // Hints are disabled
    }

    #[test]
    fn test_helper_with_different_themes() {
        let themes = [
            Theme::plain(),
            Theme::dark(),
            Theme::light(),
            Theme::auto(),
        ];
        for theme in themes {
            let helper = NeumannHelper::new(theme);
            let highlighted = helper.highlight("SELECT", 0);
            assert!(!highlighted.is_empty());
        }
    }

    #[test]
    fn test_helper_complete_partial() {
        let helper = NeumannHelper::new(Theme::plain());
        let history = rustyline::history::DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);

        // Complete partial command
        let result = helper.complete("SEL", 3, &ctx);
        assert!(result.is_ok());
        let (start, completions) = result.unwrap();
        assert_eq!(start, 0);
        assert!(completions.iter().any(|p| p.display == "SELECT"));
    }

    #[test]
    fn test_helper_complete_with_tables() {
        let mut helper = NeumannHelper::new(Theme::plain());
        helper.set_tables(vec!["users".to_string(), "orders".to_string()]);

        let history = rustyline::history::DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);
        let result = helper.complete("SELECT * FROM ", 14, &ctx);
        assert!(result.is_ok());
        let (_, completions) = result.unwrap();
        assert!(completions.iter().any(|p| p.display == "users"));
        assert!(completions.iter().any(|p| p.display == "orders"));
    }

    #[test]
    fn test_helper_highlight_keywords() {
        let helper = NeumannHelper::new(Theme::plain());

        // Various SQL keywords
        let _h1 = helper.highlight("INSERT INTO users VALUES (1)", 0);
        let _h2 = helper.highlight("UPDATE users SET name = 'x'", 0);
        let _h3 = helper.highlight("DELETE FROM users WHERE id = 1", 0);
        let _h4 = helper.highlight("CREATE TABLE test (id INT)", 0);
        let _h5 = helper.highlight("DROP TABLE test", 0);
    }
}
