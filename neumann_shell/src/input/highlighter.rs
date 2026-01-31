// SPDX-License-Identifier: MIT OR Apache-2.0
//! Syntax highlighting for shell input.

use crate::style::Theme;
use owo_colors::OwoColorize;
use rustyline::highlight::{CmdKind, Highlighter};
use std::borrow::Cow;

/// Syntax highlighter for Neumann shell.
#[derive(Debug)]
pub struct NeumannHighlighter {
    theme: Theme,
}

impl NeumannHighlighter {
    /// Creates a new highlighter with the given theme.
    #[must_use]
    pub const fn new(theme: Theme) -> Self {
        Self { theme }
    }

    /// Highlights the input line.
    fn highlight_line(&self, line: &str) -> String {
        let mut result = String::with_capacity(line.len() * 2);
        let mut in_string = false;
        let mut string_char = '"';
        let mut current_word = String::new();

        for c in line.chars() {
            if in_string {
                current_word.push(c);
                if c == string_char {
                    // End of string
                    result.push_str(&current_word.style(self.theme.string).to_string());
                    current_word.clear();
                    in_string = false;
                }
            } else if c == '\'' || c == '"' {
                // Flush current word
                if !current_word.is_empty() {
                    result.push_str(&self.style_word(&current_word));
                    current_word.clear();
                }
                // Start string
                in_string = true;
                string_char = c;
                current_word.push(c);
            } else if c.is_whitespace() || c == ',' || c == '(' || c == ')' || c == '[' || c == ']'
            {
                // Flush current word
                if !current_word.is_empty() {
                    result.push_str(&self.style_word(&current_word));
                    current_word.clear();
                }
                // Style brackets/parens
                if c == '(' || c == ')' || c == '[' || c == ']' {
                    result.push_str(&c.style(self.theme.muted).to_string());
                } else {
                    result.push(c);
                }
            } else {
                current_word.push(c);
            }
        }

        // Flush remaining
        if !current_word.is_empty() {
            if in_string {
                result.push_str(&current_word.style(self.theme.string).to_string());
            } else {
                result.push_str(&self.style_word(&current_word));
            }
        }

        result
    }

    /// Styles a word based on its type.
    fn style_word(&self, word: &str) -> String {
        let upper = word.to_uppercase();

        // SQL/Query keywords
        if is_keyword(&upper) {
            return word.style(self.theme.keyword).to_string();
        }

        // Types
        if is_type(&upper) {
            return word.style(self.theme.label).to_string();
        }

        // Numbers
        if word.parse::<f64>().is_ok() {
            return word.style(self.theme.number).to_string();
        }

        // NULL/TRUE/FALSE
        if upper == "NULL" {
            return word.style(self.theme.null).to_string();
        }
        if upper == "TRUE" || upper == "FALSE" {
            return word.style(self.theme.keyword).to_string();
        }

        // Operators
        if is_operator(word) {
            return word.style(self.theme.muted).to_string();
        }

        // Default
        word.to_string()
    }
}

/// Checks if a word is a SQL/query keyword.
#[allow(clippy::too_many_lines)]
fn is_keyword(word: &str) -> bool {
    matches!(
        word,
        "SELECT"
            | "INSERT"
            | "UPDATE"
            | "DELETE"
            | "CREATE"
            | "DROP"
            | "TABLE"
            | "FROM"
            | "WHERE"
            | "INTO"
            | "VALUES"
            | "SET"
            | "AND"
            | "OR"
            | "NOT"
            | "IN"
            | "LIKE"
            | "BETWEEN"
            | "IS"
            | "ORDER"
            | "BY"
            | "ASC"
            | "DESC"
            | "LIMIT"
            | "OFFSET"
            | "JOIN"
            | "ON"
            | "AS"
            | "DISTINCT"
            | "COUNT"
            | "SUM"
            | "AVG"
            | "MIN"
            | "MAX"
            | "GROUP"
            | "HAVING"
            | "NODE"
            | "EDGE"
            | "GRAPH"
            | "ALGORITHM"
            | "PATTERN"
            | "MATCH"
            | "PATH"
            | "NEIGHBORS"
            | "OUTGOING"
            | "INCOMING"
            | "BOTH"
            | "EMBED"
            | "STORE"
            | "GET"
            | "BUILD"
            | "INDEX"
            | "SIMILAR"
            | "COSINE"
            | "EUCLIDEAN"
            | "DOT_PRODUCT"
            | "ENTITY"
            | "CONNECT"
            | "BATCH"
            | "FIND"
            | "BLOB"
            | "BLOBS"
            | "PUT"
            | "LINK"
            | "UNLINK"
            | "TAG"
            | "UNTAG"
            | "VERIFY"
            | "GC"
            | "REPAIR"
            | "META"
            | "VAULT"
            | "INIT"
            | "IDENTITY"
            | "ROTATE"
            | "GRANT"
            | "REVOKE"
            | "CACHE"
            | "CLEAR"
            | "EVICT"
            | "SEMANTIC"
            | "CHECKPOINT"
            | "CHECKPOINTS"
            | "ROLLBACK"
            | "TO"
            | "CHAIN"
            | "BEGIN"
            | "COMMIT"
            | "TRANSACTION"
            | "HEIGHT"
            | "TIP"
            | "BLOCK"
            | "HISTORY"
            | "DRIFT"
            | "CLUSTER"
            | "DISCONNECT"
            | "STATUS"
            | "NODES"
            | "LEADER"
            | "DESCRIBE"
            | "SHOW"
            | "TABLES"
            | "EMBEDDINGS"
            | "VECTOR"
            | "CODEBOOK"
            | "GLOBAL"
            | "LOCAL"
            | "SAVE"
            | "LOAD"
            | "COMPRESSED"
            | "RECOVER"
            | "WAL"
            | "TRUNCATE"
            | "STATS"
            | "INFO"
            | "LIST"
            | "FOR"
            | "TAGS"
            | "CHUNK"
            | "FULL"
            | "THRESHOLD"
            | "EMBEDDING"
            | "RESPONSE"
            | "CONSTRAINT"
            | "UNIQUE"
            | "EXISTS"
            | "AGGREGATE"
            | "PAGERANK"
            | "BETWEENNESS"
            | "CLOSENESS"
            | "EIGENVECTOR"
            | "LOUVAIN"
            | "LABEL_PROPAGATION"
            | "DAMPING"
            | "ITERATIONS"
            | "ANALYZE"
            | "TRANSITIONS"
    )
}

/// Checks if a word is a type name.
fn is_type(word: &str) -> bool {
    matches!(
        word,
        "INT" | "INTEGER" | "TEXT" | "STRING" | "FLOAT" | "BOOL" | "BOOLEAN" | "BYTES" | "JSON"
    )
}

/// Checks if a word is an operator.
fn is_operator(word: &str) -> bool {
    matches!(
        word,
        "=" | "!=" | "<>" | "<" | ">" | "<=" | ">=" | "+" | "-" | "*" | "/" | "->" | ":"
    )
}

impl Highlighter for NeumannHighlighter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        Cow::Owned(self.highlight_line(line))
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        // Always re-highlight on char input
        true
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Owned(prompt.style(self.theme.info).to_string())
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(hint.style(self.theme.muted).to_string())
    }

    fn highlight_candidate<'c>(
        &self,
        candidate: &'c str,
        _completion: rustyline::CompletionType,
    ) -> Cow<'c, str> {
        Cow::Borrowed(candidate)
    }
}

impl Default for NeumannHighlighter {
    fn default() -> Self {
        Self::new(Theme::auto())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_keyword() {
        let highlighter = NeumannHighlighter::new(Theme::plain());
        let result = highlighter.highlight_line("SELECT");
        assert!(result.contains("SELECT"));
    }

    #[test]
    fn test_highlight_string() {
        let highlighter = NeumannHighlighter::new(Theme::plain());
        let result = highlighter.highlight_line("'hello'");
        assert!(result.contains("hello"));
    }

    #[test]
    fn test_highlight_number() {
        let highlighter = NeumannHighlighter::new(Theme::plain());
        let result = highlighter.highlight_line("123");
        assert!(result.contains("123"));
    }

    #[test]
    fn test_highlight_full_query() {
        let highlighter = NeumannHighlighter::new(Theme::plain());
        let result = highlighter.highlight_line("SELECT * FROM users WHERE id = 1");
        assert!(result.contains("SELECT"));
        assert!(result.contains("FROM"));
        assert!(result.contains("users"));
        assert!(result.contains("WHERE"));
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("SELECT"));
        assert!(is_keyword("NODE"));
        assert!(is_keyword("EMBED"));
        assert!(!is_keyword("foo"));
    }

    #[test]
    fn test_is_type() {
        assert!(is_type("INT"));
        assert!(is_type("TEXT"));
        assert!(!is_type("SELECT"));
    }

    #[test]
    fn test_is_operator() {
        assert!(is_operator("="));
        assert!(is_operator("->"));
        assert!(!is_operator("SELECT"));
    }
}
