// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tab completion for shell commands.

use rustyline::completion::{Completer, Pair};
use rustyline::Context;

/// Command completer for Neumann shell.
#[derive(Debug, Default)]
pub struct NeumannCompleter {
    /// Dynamic table names (populated at runtime).
    tables: Vec<String>,
}

#[allow(clippy::unused_self)]
impl NeumannCompleter {
    /// Creates a new completer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the list of available tables.
    pub fn set_tables(&mut self, tables: Vec<String>) {
        self.tables = tables;
    }

    /// Checks if any word matches the keyword (case-insensitive).
    fn contains_keyword(words: &[&str], keyword: &str) -> bool {
        words.iter().any(|w| w.eq_ignore_ascii_case(keyword))
    }

    /// Returns completions for the given input.
    fn complete_input(&self, line: &str, pos: usize) -> Vec<Pair> {
        let input = &line[..pos];
        let words: Vec<&str> = input.split_whitespace().collect();

        if words.is_empty() {
            return self.complete_top_level("");
        }

        let last_word = words.last().map_or("", |w| *w);
        let prefix = if input.ends_with(' ') { "" } else { last_word };

        // Determine completion context based on previous words
        match words.first().map(|s| s.to_uppercase()).as_deref() {
            Some("SELECT") => self.complete_select(&words, prefix),
            Some("INSERT") => self.complete_insert(&words, prefix),
            Some("UPDATE") => self.complete_update(&words, prefix),
            Some("DELETE") => self.complete_delete(&words, prefix),
            Some("CREATE") => self.complete_create(&words, prefix),
            Some("DROP") => self.complete_drop(&words, prefix),
            Some("NODE") => self.complete_node(&words, prefix),
            Some("EDGE") => self.complete_edge(&words, prefix),
            Some("GRAPH") => self.complete_graph(&words, prefix),
            Some("EMBED") => self.complete_embed(&words, prefix),
            Some("SIMILAR") => self.complete_similar(&words, prefix),
            Some("ENTITY") => self.complete_entity(&words, prefix),
            Some("BLOB") => self.complete_blob(&words, prefix),
            Some("VAULT") => self.complete_vault(&words, prefix),
            Some("CACHE") => self.complete_cache(&words, prefix),
            Some("CHAIN") => self.complete_chain(&words, prefix),
            Some("CLUSTER") => self.complete_cluster(&words, prefix),
            Some("DESCRIBE") => self.complete_describe(&words, prefix),
            Some("SHOW") => self.complete_show(&words, prefix),
            _ => {
                if input.ends_with(' ') || words.len() == 1 {
                    self.complete_top_level(prefix)
                } else {
                    Vec::new()
                }
            },
        }
    }

    /// Top-level command completions.
    fn complete_top_level(&self, prefix: &str) -> Vec<Pair> {
        let commands = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "NODE",
            "EDGE",
            "GRAPH",
            "EMBED",
            "SIMILAR",
            "ENTITY",
            "FIND",
            "BLOB",
            "BLOBS",
            "VAULT",
            "CACHE",
            "CHECKPOINT",
            "CHECKPOINTS",
            "ROLLBACK",
            "CHAIN",
            "CLUSTER",
            "DESCRIBE",
            "SHOW",
            "SAVE",
            "LOAD",
            "help",
            "exit",
            "quit",
            "tables",
            "clear",
        ];
        Self::filter_completions(&commands, prefix)
    }

    fn complete_select(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if Self::contains_keyword(words, "FROM") {
            // After FROM, suggest table names
            Self::filter_completions_from_strings(&self.tables, prefix)
        } else if words.len() == 2 || (words.len() == 1 && prefix.is_empty()) {
            // After SELECT, suggest * or FROM
            Self::filter_completions(&["*", "FROM"], prefix)
        } else {
            Self::filter_completions(&["FROM", "WHERE", "LIMIT", "ORDER"], prefix)
        }
    }

    fn complete_insert(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if Self::contains_keyword(words, "INTO") {
            if !Self::contains_keyword(words, "VALUES") {
                // After table name, suggest VALUES
                if words.len() > 3 {
                    return Self::filter_completions(&["VALUES"], prefix);
                }
                // After INTO, suggest table names
                return Self::filter_completions_from_strings(&self.tables, prefix);
            }
        } else {
            return Self::filter_completions(&["INTO"], prefix);
        }
        Vec::new()
    }

    fn complete_update(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if Self::contains_keyword(words, "SET") {
            Self::filter_completions(&["WHERE"], prefix)
        } else if words.len() == 2 {
            Self::filter_completions_from_strings(&self.tables, prefix)
        } else {
            Self::filter_completions(&["SET"], prefix)
        }
    }

    fn complete_delete(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if Self::contains_keyword(words, "FROM") {
            if words.len() > 3 {
                return Self::filter_completions(&["WHERE"], prefix);
            }
            Self::filter_completions_from_strings(&self.tables, prefix)
        } else {
            Self::filter_completions(&["FROM"], prefix)
        }
    }

    fn complete_create(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(&["TABLE"], prefix)
        } else {
            Vec::new()
        }
    }

    fn complete_drop(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if Self::contains_keyword(words, "TABLE") {
            Self::filter_completions_from_strings(&self.tables, prefix)
        } else {
            Self::filter_completions(&["TABLE"], prefix)
        }
    }

    fn complete_node(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(&["CREATE", "LIST", "GET", "DELETE"], prefix)
        } else {
            Vec::new()
        }
    }

    fn complete_edge(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(&["CREATE", "LIST", "GET", "DELETE"], prefix)
        } else {
            Vec::new()
        }
    }

    fn complete_graph(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if Self::contains_keyword(words, "ALGORITHM") {
            Self::filter_completions(
                &[
                    "PAGERANK",
                    "BETWEENNESS",
                    "CLOSENESS",
                    "EIGENVECTOR",
                    "LOUVAIN",
                    "LABEL_PROPAGATION",
                ],
                prefix,
            )
        } else if Self::contains_keyword(words, "PATTERN") {
            Self::filter_completions(&["MATCH", "COUNT", "EXISTS"], prefix)
        } else if Self::contains_keyword(words, "BATCH") {
            Self::filter_completions(&["CREATE", "DELETE", "UPDATE"], prefix)
        } else if Self::contains_keyword(words, "CONSTRAINT") {
            Self::filter_completions(&["CREATE", "DROP", "LIST"], prefix)
        } else if Self::contains_keyword(words, "INDEX") {
            Self::filter_completions(&["CREATE", "DROP", "SHOW"], prefix)
        } else if Self::contains_keyword(words, "AGGREGATE") {
            Self::filter_completions(&["COUNT", "SUM", "AVG", "MIN", "MAX"], prefix)
        } else if words.len() <= 2 {
            Self::filter_completions(
                &[
                    "ALGORITHM",
                    "PATTERN",
                    "BATCH",
                    "CONSTRAINT",
                    "INDEX",
                    "AGGREGATE",
                ],
                prefix,
            )
        } else {
            Vec::new()
        }
    }

    fn complete_embed(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(&["STORE", "GET", "DELETE", "BUILD", "BATCH"], prefix)
        } else if Self::contains_keyword(words, "BUILD") {
            Self::filter_completions(&["INDEX"], prefix)
        } else {
            Vec::new()
        }
    }

    fn complete_similar(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.iter().any(|w| w.starts_with('[')) {
            // After vector, suggest metric or LIMIT
            Self::filter_completions(&["COSINE", "EUCLIDEAN", "DOT_PRODUCT", "LIMIT"], prefix)
        } else if words.len() > 2 {
            Self::filter_completions(&["COSINE", "EUCLIDEAN", "DOT_PRODUCT", "LIMIT"], prefix)
        } else {
            Vec::new()
        }
    }

    fn complete_entity(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(&["CREATE", "GET", "CONNECT", "BATCH"], prefix)
        } else {
            Vec::new()
        }
    }

    fn complete_blob(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(
                &[
                    "INIT", "PUT", "GET", "DELETE", "INFO", "LINK", "UNLINK", "LINKS", "TAG",
                    "UNTAG", "VERIFY", "GC", "REPAIR", "STATS", "META",
                ],
                prefix,
            )
        } else if Self::contains_keyword(words, "GC") {
            Self::filter_completions(&["FULL"], prefix)
        } else if Self::contains_keyword(words, "META") {
            Self::filter_completions(&["SET", "GET"], prefix)
        } else {
            Vec::new()
        }
    }

    fn complete_vault(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(
                &[
                    "INIT", "IDENTITY", "SET", "GET", "DELETE", "LIST", "ROTATE", "GRANT", "REVOKE",
                ],
                prefix,
            )
        } else {
            Vec::new()
        }
    }

    fn complete_cache(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if Self::contains_keyword(words, "SEMANTIC") {
            Self::filter_completions(&["GET", "PUT"], prefix)
        } else if words.len() <= 2 {
            Self::filter_completions(
                &["INIT", "STATS", "CLEAR", "EVICT", "GET", "PUT", "SEMANTIC"],
                prefix,
            )
        } else {
            Vec::new()
        }
    }

    fn complete_chain(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(
                &[
                    "HEIGHT", "TIP", "BLOCK", "VERIFY", "HISTORY", "SIMILAR", "DRIFT",
                ],
                prefix,
            )
        } else {
            Vec::new()
        }
    }

    fn complete_cluster(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(
                &["CONNECT", "DISCONNECT", "STATUS", "NODES", "LEADER"],
                prefix,
            )
        } else {
            Vec::new()
        }
    }

    fn complete_describe(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if Self::contains_keyword(words, "TABLE") {
            Self::filter_completions_from_strings(&self.tables, prefix)
        } else if words.len() <= 2 {
            Self::filter_completions(&["TABLE", "NODE", "EDGE"], prefix)
        } else {
            Vec::new()
        }
    }

    fn complete_show(&self, words: &[&str], prefix: &str) -> Vec<Pair> {
        if words.len() <= 2 {
            Self::filter_completions(&["TABLES", "EMBEDDINGS", "VECTOR", "CODEBOOK"], prefix)
        } else if Self::contains_keyword(words, "CODEBOOK") {
            Self::filter_completions(&["GLOBAL", "LOCAL"], prefix)
        } else if Self::contains_keyword(words, "VECTOR") {
            Self::filter_completions(&["INDEX"], prefix)
        } else {
            Vec::new()
        }
    }

    /// Filters completions by prefix.
    fn filter_completions(options: &[&str], prefix: &str) -> Vec<Pair> {
        let prefix_upper = prefix.to_uppercase();
        options
            .iter()
            .filter(|opt| opt.to_uppercase().starts_with(&prefix_upper))
            .map(|opt| Pair {
                display: (*opt).to_string(),
                replacement: (*opt).to_string(),
            })
            .collect()
    }

    /// Filters completions from a string vector.
    fn filter_completions_from_strings(options: &[String], prefix: &str) -> Vec<Pair> {
        let prefix_lower = prefix.to_lowercase();
        options
            .iter()
            .filter(|opt| opt.to_lowercase().starts_with(&prefix_lower))
            .map(|opt| Pair {
                display: opt.clone(),
                replacement: opt.clone(),
            })
            .collect()
    }
}

impl Completer for NeumannCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let completions = self.complete_input(line, pos);

        // Calculate the start position for replacement
        let start = line[..pos]
            .rfind(|c: char| c.is_whitespace())
            .map_or(0, |i| i + 1);

        Ok((start, completions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_empty() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_input("", 0);
        assert!(!completions.is_empty());
        assert!(completions.iter().any(|p| p.display == "SELECT"));
    }

    #[test]
    fn test_complete_partial_select() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_input("SEL", 3);
        assert!(completions.iter().any(|p| p.display == "SELECT"));
    }

    #[test]
    fn test_complete_after_select() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_input("SELECT ", 7);
        assert!(completions.iter().any(|p| p.display == "*"));
        assert!(completions.iter().any(|p| p.display == "FROM"));
    }

    #[test]
    fn test_complete_node_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_input("NODE ", 5);
        assert!(completions.iter().any(|p| p.display == "CREATE"));
        assert!(completions.iter().any(|p| p.display == "LIST"));
        assert!(completions.iter().any(|p| p.display == "GET"));
        assert!(completions.iter().any(|p| p.display == "DELETE"));
    }

    #[test]
    fn test_complete_graph_algorithm() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_input("GRAPH ALGORITHM ", 16);
        assert!(completions.iter().any(|p| p.display == "PAGERANK"));
        assert!(completions.iter().any(|p| p.display == "LOUVAIN"));
    }

    #[test]
    fn test_complete_with_tables() {
        let mut completer = NeumannCompleter::new();
        completer.set_tables(vec!["users".to_string(), "orders".to_string()]);
        let completions = completer.complete_input("SELECT * FROM ", 14);
        assert!(completions.iter().any(|p| p.display == "users"));
        assert!(completions.iter().any(|p| p.display == "orders"));
    }

    #[test]
    fn test_complete_case_insensitive() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_input("sel", 3);
        assert!(completions.iter().any(|p| p.display == "SELECT"));
    }
}
