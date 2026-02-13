// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Error types and diagnostics for the Neumann parser.
//!
//! Provides rich error messages with source context, including:
//! - Precise source locations (line and column)
//! - Error context with source snippets
//! - Helpful suggestions and expected tokens

use std::fmt;

use crate::{
    span::{get_line, line_col, Span},
    token::TokenKind,
};

/// Result type for parser operations.
pub type ParseResult<T> = Result<T, ParseError>;

/// A parse error with source location.
#[derive(Clone, Debug)]
pub struct ParseError {
    /// The kind of error.
    pub kind: ParseErrorKind,
    /// Where the error occurred.
    pub span: Span,
    /// Optional help message.
    pub help: Option<String>,
}

impl ParseError {
    /// Creates a new parse error.
    #[must_use]
    pub const fn new(kind: ParseErrorKind, span: Span) -> Self {
        Self {
            kind,
            span,
            help: None,
        }
    }

    /// Creates an "unexpected token" error.
    pub fn unexpected(found: TokenKind, span: Span, expected: impl Into<String>) -> Self {
        Self::new(
            ParseErrorKind::UnexpectedToken {
                found,
                expected: expected.into(),
            },
            span,
        )
    }

    /// Creates an "unexpected EOF" error.
    pub fn unexpected_eof(span: Span, expected: impl Into<String>) -> Self {
        Self::new(
            ParseErrorKind::UnexpectedEof {
                expected: expected.into(),
            },
            span,
        )
    }

    /// Creates an "invalid syntax" error.
    pub fn invalid(message: impl Into<String>, span: Span) -> Self {
        Self::new(ParseErrorKind::InvalidSyntax(message.into()), span)
    }

    /// Adds a help message to the error.
    #[must_use]
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Formats the error with source context.
    #[must_use]
    #[allow(clippy::format_push_string)]
    pub fn format_with_source(&self, source: &str) -> String {
        let (line, col) = line_col(source, self.span.start);
        let line_text = get_line(source, self.span.start);

        let kind = &self.kind;
        let mut result = format!("error: {kind}\n");
        result.push_str(&format!("  --> line {line}:{col}\n"));
        result.push_str("   |\n");
        result.push_str(&format!("{line:3} | {line_text}\n"));
        result.push_str("   | ");

        // Add carets under the error
        for _ in 0..(col - 1) {
            result.push(' ');
        }
        let len = self.span.len().max(1) as usize;
        let caret_len = len.min(line_text.len().saturating_sub(col - 1));
        for _ in 0..caret_len.max(1) {
            result.push('^');
        }
        result.push('\n');

        if let Some(help) = &self.help {
            result.push_str(&format!("   = help: {help}\n"));
        }

        result
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.kind, self.span)
    }
}

impl std::error::Error for ParseError {}

/// Parse error kinds.
#[derive(Clone, Debug)]
pub enum ParseErrorKind {
    /// Unexpected token
    UnexpectedToken {
        /// The token that was found.
        found: TokenKind,
        /// Description of what was expected.
        expected: String,
    },
    /// Unexpected end of input
    UnexpectedEof {
        /// Description of what was expected.
        expected: String,
    },
    /// Invalid syntax
    InvalidSyntax(String),
    /// Invalid number
    InvalidNumber(String),
    /// Unterminated string
    UnterminatedString,
    /// Unknown keyword or command
    UnknownCommand(String),
    /// Duplicate column
    DuplicateColumn(String),
    /// Invalid escape sequence
    InvalidEscape(char),
    /// Expression too deeply nested
    TooDeep,
    /// Custom error
    Custom(String),
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedToken { found, expected } => {
                write!(f, "unexpected {found}, expected {expected}")
            },
            Self::UnexpectedEof { expected } => {
                write!(f, "unexpected end of input, expected {expected}")
            },
            Self::InvalidSyntax(msg) | Self::Custom(msg) => write!(f, "{msg}"),
            Self::InvalidNumber(msg) => write!(f, "invalid number: {msg}"),
            Self::UnterminatedString => write!(f, "unterminated string literal"),
            Self::UnknownCommand(cmd) => write!(f, "unknown command: {cmd}"),
            Self::DuplicateColumn(col) => write!(f, "duplicate column: {col}"),
            Self::InvalidEscape(c) => write!(f, "invalid escape sequence: \\{c}"),
            Self::TooDeep => write!(f, "expression nesting too deep"),
        }
    }
}

/// A collection of parse errors.
#[derive(Clone, Debug, Default)]
pub struct Errors {
    errors: Vec<ParseError>,
}

impl Errors {
    /// Creates an empty error collection.
    #[must_use]
    pub const fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Adds an error to the collection.
    pub fn push(&mut self, error: ParseError) {
        self.errors.push(error);
    }

    /// Returns true if there are no errors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns the number of errors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.errors.len()
    }

    /// Returns an iterator over the errors.
    pub fn iter(&self) -> impl Iterator<Item = &ParseError> {
        self.errors.iter()
    }

    /// Consumes the collection and returns the errors.
    #[must_use]
    pub fn into_vec(self) -> Vec<ParseError> {
        self.errors
    }

    /// Formats all errors with source context.
    #[must_use]
    pub fn format_with_source(&self, source: &str) -> String {
        self.errors
            .iter()
            .map(|e| e.format_with_source(source))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl IntoIterator for Errors {
    type Item = ParseError;
    type IntoIter = std::vec::IntoIter<ParseError>;

    fn into_iter(self) -> Self::IntoIter {
        self.errors.into_iter()
    }
}

impl<'a> IntoIterator for &'a Errors {
    type Item = &'a ParseError;
    type IntoIter = std::slice::Iter<'a, ParseError>;

    fn into_iter(self) -> Self::IntoIter {
        self.errors.iter()
    }
}

impl Extend<ParseError> for Errors {
    fn extend<T: IntoIterator<Item = ParseError>>(&mut self, iter: T) {
        self.errors.extend(iter);
    }
}

impl FromIterator<ParseError> for Errors {
    fn from_iter<T: IntoIterator<Item = ParseError>>(iter: T) -> Self {
        Self {
            errors: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_unexpected() {
        let err = ParseError::unexpected(
            TokenKind::Integer(42),
            Span::from_offsets(0, 2),
            "identifier",
        );
        assert!(matches!(err.kind, ParseErrorKind::UnexpectedToken { .. }));
        assert_eq!(err.span, Span::from_offsets(0, 2));
    }

    #[test]
    fn test_parse_error_unexpected_eof() {
        let err = ParseError::unexpected_eof(Span::point(crate::span::BytePos(10)), "expression");
        assert!(matches!(err.kind, ParseErrorKind::UnexpectedEof { .. }));
    }

    #[test]
    fn test_parse_error_invalid() {
        let err = ParseError::invalid("bad syntax", Span::from_offsets(5, 10));
        assert!(matches!(err.kind, ParseErrorKind::InvalidSyntax(_)));
    }

    #[test]
    fn test_parse_error_with_help() {
        let err = ParseError::invalid("missing semicolon", Span::from_offsets(0, 5))
            .with_help("add a semicolon at the end");
        assert_eq!(err.help, Some("add a semicolon at the end".to_string()));
    }

    #[test]
    fn test_parse_error_display() {
        let err =
            ParseError::unexpected(TokenKind::Comma, Span::from_offsets(10, 11), "column name");
        let s = format!("{}", err);
        assert!(s.contains("unexpected"));
        assert!(s.contains(","));
        assert!(s.contains("column name"));
    }

    #[test]
    fn test_format_with_source() {
        let source = "SELECT * FORM users";
        let err = ParseError::invalid("expected FROM, found FORM", Span::from_offsets(9, 13));
        let formatted = err.format_with_source(source);

        assert!(formatted.contains("error:"));
        assert!(formatted.contains("line 1:10"));
        assert!(formatted.contains("SELECT * FORM users"));
        assert!(formatted.contains("^^^^"));
    }

    #[test]
    fn test_format_with_source_multiline() {
        let source = "SELECT *\nFORM users\nWHERE id = 1";
        let err = ParseError::invalid("expected FROM", Span::from_offsets(9, 13));
        let formatted = err.format_with_source(source);

        assert!(formatted.contains("line 2:1"));
        assert!(formatted.contains("FORM users"));
    }

    #[test]
    fn test_format_with_help() {
        let source = "SELCT * FROM users";
        let err = ParseError::invalid("unknown keyword SELCT", Span::from_offsets(0, 5))
            .with_help("did you mean SELECT?");
        let formatted = err.format_with_source(source);

        assert!(formatted.contains("help: did you mean SELECT?"));
    }

    #[test]
    fn test_errors_collection() {
        let mut errors = Errors::new();
        assert!(errors.is_empty());
        assert_eq!(errors.len(), 0);

        errors.push(ParseError::invalid("error 1", Span::from_offsets(0, 5)));
        errors.push(ParseError::invalid("error 2", Span::from_offsets(10, 15)));

        assert!(!errors.is_empty());
        assert_eq!(errors.len(), 2);

        let mut count = 0;
        for _ in &errors {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_errors_into_vec() {
        let mut errors = Errors::new();
        errors.push(ParseError::invalid("error 1", Span::from_offsets(0, 5)));
        errors.push(ParseError::invalid("error 2", Span::from_offsets(10, 15)));

        let vec = errors.into_vec();
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_errors_extend() {
        let mut errors = Errors::new();
        errors.extend(vec![
            ParseError::invalid("error 1", Span::from_offsets(0, 5)),
            ParseError::invalid("error 2", Span::from_offsets(10, 15)),
        ]);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_errors_from_iter() {
        let errors: Errors = vec![
            ParseError::invalid("error 1", Span::from_offsets(0, 5)),
            ParseError::invalid("error 2", Span::from_offsets(10, 15)),
        ]
        .into_iter()
        .collect();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_errors_format_with_source() {
        let source = "SELECT * FROM";
        let mut errors = Errors::new();
        errors.push(ParseError::invalid("error 1", Span::from_offsets(0, 6)));
        errors.push(ParseError::invalid("error 2", Span::from_offsets(7, 8)));

        let formatted = errors.format_with_source(source);
        assert!(formatted.contains("error 1"));
        assert!(formatted.contains("error 2"));
    }

    #[test]
    fn test_parse_error_kind_display() {
        assert!(format!("{}", ParseErrorKind::UnterminatedString).contains("unterminated"));
        assert!(format!("{}", ParseErrorKind::TooDeep).contains("deep"));
        assert!(format!("{}", ParseErrorKind::InvalidNumber("bad".to_string())).contains("bad"));
        assert!(format!("{}", ParseErrorKind::UnknownCommand("FOO".to_string())).contains("FOO"));
        assert!(format!("{}", ParseErrorKind::DuplicateColumn("id".to_string())).contains("id"));
        assert!(format!("{}", ParseErrorKind::InvalidEscape('x')).contains("\\x"));
        assert!(format!("{}", ParseErrorKind::Custom("custom".to_string())).contains("custom"));
    }

    #[test]
    fn test_errors_into_iterator() {
        let mut errors = Errors::new();
        errors.push(ParseError::invalid("error 1", Span::from_offsets(0, 5)));
        errors.push(ParseError::invalid("error 2", Span::from_offsets(10, 15)));

        let mut count = 0;
        for _ in errors {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_zero_length_span_caret() {
        let source = "SELECT FROM";
        let err = ParseError::invalid("expected expression", Span::point(crate::span::BytePos(7)));
        let formatted = err.format_with_source(source);
        // Should still show at least one caret
        assert!(formatted.contains("^"));
    }

    #[test]
    fn test_errors_iter() {
        let mut errors = Errors::new();
        errors.push(ParseError::invalid("error 1", Span::from_offsets(0, 5)));
        errors.push(ParseError::invalid("error 2", Span::from_offsets(10, 15)));

        // Explicitly test iter() method
        let mut count = 0;
        for err in errors.iter() {
            assert!(matches!(err.kind, ParseErrorKind::InvalidSyntax(_)));
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_unexpected_eof_display() {
        let kind = ParseErrorKind::UnexpectedEof {
            expected: "identifier".to_string(),
        };
        let displayed = format!("{}", kind);
        assert!(displayed.contains("unexpected end of input"));
        assert!(displayed.contains("identifier"));
    }
}
