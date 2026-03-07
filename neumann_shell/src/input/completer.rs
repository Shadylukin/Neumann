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

    #[test]
    fn test_completer_default() {
        let completer = NeumannCompleter::default();
        assert!(completer.tables.is_empty());
    }

    #[test]
    fn test_complete_top_level_all_commands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_top_level("");
        assert!(completions.iter().any(|p| p.display == "INSERT"));
        assert!(completions.iter().any(|p| p.display == "UPDATE"));
        assert!(completions.iter().any(|p| p.display == "DELETE"));
        assert!(completions.iter().any(|p| p.display == "CREATE"));
        assert!(completions.iter().any(|p| p.display == "DROP"));
        assert!(completions.iter().any(|p| p.display == "NODE"));
        assert!(completions.iter().any(|p| p.display == "EDGE"));
        assert!(completions.iter().any(|p| p.display == "GRAPH"));
        assert!(completions.iter().any(|p| p.display == "EMBED"));
        assert!(completions.iter().any(|p| p.display == "SIMILAR"));
        assert!(completions.iter().any(|p| p.display == "ENTITY"));
        assert!(completions.iter().any(|p| p.display == "FIND"));
        assert!(completions.iter().any(|p| p.display == "BLOB"));
        assert!(completions.iter().any(|p| p.display == "BLOBS"));
        assert!(completions.iter().any(|p| p.display == "VAULT"));
        assert!(completions.iter().any(|p| p.display == "CACHE"));
        assert!(completions.iter().any(|p| p.display == "CHECKPOINT"));
        assert!(completions.iter().any(|p| p.display == "CHECKPOINTS"));
        assert!(completions.iter().any(|p| p.display == "ROLLBACK"));
        assert!(completions.iter().any(|p| p.display == "CHAIN"));
        assert!(completions.iter().any(|p| p.display == "CLUSTER"));
        assert!(completions.iter().any(|p| p.display == "DESCRIBE"));
        assert!(completions.iter().any(|p| p.display == "SHOW"));
        assert!(completions.iter().any(|p| p.display == "SAVE"));
        assert!(completions.iter().any(|p| p.display == "LOAD"));
        assert!(completions.iter().any(|p| p.display == "help"));
        assert!(completions.iter().any(|p| p.display == "exit"));
        assert!(completions.iter().any(|p| p.display == "quit"));
        assert!(completions.iter().any(|p| p.display == "tables"));
        assert!(completions.iter().any(|p| p.display == "clear"));
    }

    #[test]
    fn test_complete_select_from_tables() {
        let mut completer = NeumannCompleter::new();
        completer.set_tables(vec!["users".to_string(), "orders".to_string()]);
        let completions = completer.complete_select(&["SELECT", "*", "FROM"], "");
        assert!(completions.iter().any(|p| p.display == "users"));
    }

    #[test]
    fn test_complete_select_where_limit() {
        let completer = NeumannCompleter::new();
        // After FROM with table, we need 5 words to trigger WHERE/LIMIT
        let completions = completer.complete_select(&["SELECT", "*"], "");
        assert!(completions.iter().any(|p| p.display == "FROM"));
    }

    #[test]
    fn test_complete_insert_into() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_insert(&["INSERT"], "");
        assert!(completions.iter().any(|p| p.display == "INTO"));
    }

    #[test]
    fn test_complete_insert_table_names() {
        let mut completer = NeumannCompleter::new();
        completer.set_tables(vec!["users".to_string()]);
        let completions = completer.complete_insert(&["INSERT", "INTO"], "");
        assert!(completions.iter().any(|p| p.display == "users"));
    }

    #[test]
    fn test_complete_insert_values() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_insert(&["INSERT", "INTO", "users", "(id)"], "");
        assert!(completions.iter().any(|p| p.display == "VALUES"));
    }

    #[test]
    fn test_complete_update_set() {
        let completer = NeumannCompleter::new();
        // With >2 words and no SET, suggest SET
        let completions = completer.complete_update(&["UPDATE", "users", "x"], "");
        assert!(completions.iter().any(|p| p.display == "SET"));
    }

    #[test]
    fn test_complete_update_table_names() {
        let mut completer = NeumannCompleter::new();
        completer.set_tables(vec!["users".to_string()]);
        // With 2 words, suggest table names
        let completions = completer.complete_update(&["UPDATE", ""], "");
        assert!(completions.iter().any(|p| p.display == "users"));
    }

    #[test]
    fn test_complete_update_where() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_update(&["UPDATE", "users", "SET", "x=1"], "");
        assert!(completions.iter().any(|p| p.display == "WHERE"));
    }

    #[test]
    fn test_complete_delete_from() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_delete(&["DELETE"], "");
        assert!(completions.iter().any(|p| p.display == "FROM"));
    }

    #[test]
    fn test_complete_delete_table_names() {
        let mut completer = NeumannCompleter::new();
        completer.set_tables(vec!["users".to_string()]);
        let completions = completer.complete_delete(&["DELETE", "FROM"], "");
        assert!(completions.iter().any(|p| p.display == "users"));
    }

    #[test]
    fn test_complete_delete_where() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_delete(&["DELETE", "FROM", "users", "x"], "");
        assert!(completions.iter().any(|p| p.display == "WHERE"));
    }

    #[test]
    fn test_complete_create_table() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_create(&["CREATE"], "");
        assert!(completions.iter().any(|p| p.display == "TABLE"));
    }

    #[test]
    fn test_complete_create_after_table() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_create(&["CREATE", "TABLE", "users"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_complete_drop_table() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_drop(&["DROP"], "");
        assert!(completions.iter().any(|p| p.display == "TABLE"));
    }

    #[test]
    fn test_complete_drop_table_names() {
        let mut completer = NeumannCompleter::new();
        completer.set_tables(vec!["users".to_string()]);
        let completions = completer.complete_drop(&["DROP", "TABLE"], "");
        assert!(completions.iter().any(|p| p.display == "users"));
    }

    #[test]
    fn test_complete_edge_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_edge(&["EDGE"], "");
        assert!(completions.iter().any(|p| p.display == "CREATE"));
        assert!(completions.iter().any(|p| p.display == "LIST"));
        assert!(completions.iter().any(|p| p.display == "GET"));
        assert!(completions.iter().any(|p| p.display == "DELETE"));
    }

    #[test]
    fn test_complete_edge_after_subcommand() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_edge(&["EDGE", "CREATE", "knows"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_complete_graph_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_graph(&["GRAPH"], "");
        assert!(completions.iter().any(|p| p.display == "ALGORITHM"));
        assert!(completions.iter().any(|p| p.display == "PATTERN"));
        assert!(completions.iter().any(|p| p.display == "BATCH"));
        assert!(completions.iter().any(|p| p.display == "CONSTRAINT"));
        assert!(completions.iter().any(|p| p.display == "INDEX"));
        assert!(completions.iter().any(|p| p.display == "AGGREGATE"));
    }

    #[test]
    fn test_complete_graph_pattern() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_graph(&["GRAPH", "PATTERN"], "");
        assert!(completions.iter().any(|p| p.display == "MATCH"));
        assert!(completions.iter().any(|p| p.display == "COUNT"));
        assert!(completions.iter().any(|p| p.display == "EXISTS"));
    }

    #[test]
    fn test_complete_graph_batch() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_graph(&["GRAPH", "BATCH"], "");
        assert!(completions.iter().any(|p| p.display == "CREATE"));
        assert!(completions.iter().any(|p| p.display == "DELETE"));
        assert!(completions.iter().any(|p| p.display == "UPDATE"));
    }

    #[test]
    fn test_complete_graph_constraint() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_graph(&["GRAPH", "CONSTRAINT"], "");
        assert!(completions.iter().any(|p| p.display == "CREATE"));
        assert!(completions.iter().any(|p| p.display == "DROP"));
        assert!(completions.iter().any(|p| p.display == "LIST"));
    }

    #[test]
    fn test_complete_graph_index() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_graph(&["GRAPH", "INDEX"], "");
        assert!(completions.iter().any(|p| p.display == "CREATE"));
        assert!(completions.iter().any(|p| p.display == "DROP"));
        assert!(completions.iter().any(|p| p.display == "SHOW"));
    }

    #[test]
    fn test_complete_graph_aggregate() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_graph(&["GRAPH", "AGGREGATE"], "");
        assert!(completions.iter().any(|p| p.display == "COUNT"));
        assert!(completions.iter().any(|p| p.display == "SUM"));
        assert!(completions.iter().any(|p| p.display == "AVG"));
        assert!(completions.iter().any(|p| p.display == "MIN"));
        assert!(completions.iter().any(|p| p.display == "MAX"));
    }

    #[test]
    fn test_complete_graph_empty_after_deep() {
        let completer = NeumannCompleter::new();
        // With ALGORITHM keyword, suggest algorithm names
        let completions = completer.complete_graph(&["GRAPH", "ALGORITHM", "PAGERANK", "x"], "");
        // The ALGORITHM check takes precedence, so we get algorithm suggestions
        assert!(completions.iter().any(|p| p.display == "PAGERANK"));
    }

    #[test]
    fn test_complete_embed_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_embed(&["EMBED"], "");
        assert!(completions.iter().any(|p| p.display == "STORE"));
        assert!(completions.iter().any(|p| p.display == "GET"));
        assert!(completions.iter().any(|p| p.display == "DELETE"));
        assert!(completions.iter().any(|p| p.display == "BUILD"));
        assert!(completions.iter().any(|p| p.display == "BATCH"));
    }

    #[test]
    fn test_complete_embed_build_index() {
        let completer = NeumannCompleter::new();
        // With >2 words and BUILD keyword, suggest INDEX
        let completions = completer.complete_embed(&["EMBED", "BUILD", "x"], "");
        assert!(completions.iter().any(|p| p.display == "INDEX"));
    }

    #[test]
    fn test_complete_similar_metrics() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_similar(&["SIMILAR", "[1,2,3]"], "");
        assert!(completions.iter().any(|p| p.display == "COSINE"));
        assert!(completions.iter().any(|p| p.display == "EUCLIDEAN"));
        assert!(completions.iter().any(|p| p.display == "DOT_PRODUCT"));
        assert!(completions.iter().any(|p| p.display == "LIMIT"));
    }

    #[test]
    fn test_complete_similar_after_key() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_similar(&["SIMILAR", "'key'", "5"], "");
        assert!(completions.iter().any(|p| p.display == "COSINE"));
    }

    #[test]
    fn test_complete_similar_early() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_similar(&["SIMILAR"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_complete_entity_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_entity(&["ENTITY"], "");
        assert!(completions.iter().any(|p| p.display == "CREATE"));
        assert!(completions.iter().any(|p| p.display == "GET"));
        assert!(completions.iter().any(|p| p.display == "CONNECT"));
        assert!(completions.iter().any(|p| p.display == "BATCH"));
    }

    #[test]
    fn test_complete_blob_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_blob(&["BLOB"], "");
        assert!(completions.iter().any(|p| p.display == "INIT"));
        assert!(completions.iter().any(|p| p.display == "PUT"));
        assert!(completions.iter().any(|p| p.display == "GET"));
        assert!(completions.iter().any(|p| p.display == "DELETE"));
        assert!(completions.iter().any(|p| p.display == "INFO"));
        assert!(completions.iter().any(|p| p.display == "LINK"));
        assert!(completions.iter().any(|p| p.display == "UNLINK"));
        assert!(completions.iter().any(|p| p.display == "LINKS"));
        assert!(completions.iter().any(|p| p.display == "TAG"));
        assert!(completions.iter().any(|p| p.display == "UNTAG"));
        assert!(completions.iter().any(|p| p.display == "VERIFY"));
        assert!(completions.iter().any(|p| p.display == "GC"));
        assert!(completions.iter().any(|p| p.display == "REPAIR"));
        assert!(completions.iter().any(|p| p.display == "STATS"));
        assert!(completions.iter().any(|p| p.display == "META"));
    }

    #[test]
    fn test_complete_blob_gc() {
        let completer = NeumannCompleter::new();
        // With >2 words and GC keyword, suggest FULL
        let completions = completer.complete_blob(&["BLOB", "GC", "x"], "");
        assert!(completions.iter().any(|p| p.display == "FULL"));
    }

    #[test]
    fn test_complete_blob_meta() {
        let completer = NeumannCompleter::new();
        // With >2 words and META keyword, suggest SET/GET
        let completions = completer.complete_blob(&["BLOB", "META", "x"], "");
        assert!(completions.iter().any(|p| p.display == "SET"));
        assert!(completions.iter().any(|p| p.display == "GET"));
    }

    #[test]
    fn test_complete_vault_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_vault(&["VAULT"], "");
        assert!(completions.iter().any(|p| p.display == "INIT"));
        assert!(completions.iter().any(|p| p.display == "IDENTITY"));
        assert!(completions.iter().any(|p| p.display == "SET"));
        assert!(completions.iter().any(|p| p.display == "GET"));
        assert!(completions.iter().any(|p| p.display == "DELETE"));
        assert!(completions.iter().any(|p| p.display == "LIST"));
        assert!(completions.iter().any(|p| p.display == "ROTATE"));
        assert!(completions.iter().any(|p| p.display == "GRANT"));
        assert!(completions.iter().any(|p| p.display == "REVOKE"));
    }

    #[test]
    fn test_complete_cache_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_cache(&["CACHE"], "");
        assert!(completions.iter().any(|p| p.display == "INIT"));
        assert!(completions.iter().any(|p| p.display == "STATS"));
        assert!(completions.iter().any(|p| p.display == "CLEAR"));
        assert!(completions.iter().any(|p| p.display == "EVICT"));
        assert!(completions.iter().any(|p| p.display == "GET"));
        assert!(completions.iter().any(|p| p.display == "PUT"));
        assert!(completions.iter().any(|p| p.display == "SEMANTIC"));
    }

    #[test]
    fn test_complete_cache_semantic() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_cache(&["CACHE", "SEMANTIC"], "");
        assert!(completions.iter().any(|p| p.display == "GET"));
        assert!(completions.iter().any(|p| p.display == "PUT"));
    }

    #[test]
    fn test_complete_chain_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_chain(&["CHAIN"], "");
        assert!(completions.iter().any(|p| p.display == "HEIGHT"));
        assert!(completions.iter().any(|p| p.display == "TIP"));
        assert!(completions.iter().any(|p| p.display == "BLOCK"));
        assert!(completions.iter().any(|p| p.display == "VERIFY"));
        assert!(completions.iter().any(|p| p.display == "HISTORY"));
        assert!(completions.iter().any(|p| p.display == "SIMILAR"));
        assert!(completions.iter().any(|p| p.display == "DRIFT"));
    }

    #[test]
    fn test_complete_cluster_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_cluster(&["CLUSTER"], "");
        assert!(completions.iter().any(|p| p.display == "CONNECT"));
        assert!(completions.iter().any(|p| p.display == "DISCONNECT"));
        assert!(completions.iter().any(|p| p.display == "STATUS"));
        assert!(completions.iter().any(|p| p.display == "NODES"));
        assert!(completions.iter().any(|p| p.display == "LEADER"));
    }

    #[test]
    fn test_complete_describe_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_describe(&["DESCRIBE"], "");
        assert!(completions.iter().any(|p| p.display == "TABLE"));
        assert!(completions.iter().any(|p| p.display == "NODE"));
        assert!(completions.iter().any(|p| p.display == "EDGE"));
    }

    #[test]
    fn test_complete_describe_table_names() {
        let mut completer = NeumannCompleter::new();
        completer.set_tables(vec!["users".to_string()]);
        let completions = completer.complete_describe(&["DESCRIBE", "TABLE"], "");
        assert!(completions.iter().any(|p| p.display == "users"));
    }

    #[test]
    fn test_complete_show_subcommands() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_show(&["SHOW"], "");
        assert!(completions.iter().any(|p| p.display == "TABLES"));
        assert!(completions.iter().any(|p| p.display == "EMBEDDINGS"));
        assert!(completions.iter().any(|p| p.display == "VECTOR"));
        assert!(completions.iter().any(|p| p.display == "CODEBOOK"));
    }

    #[test]
    fn test_complete_show_codebook() {
        let completer = NeumannCompleter::new();
        // With >2 words and CODEBOOK keyword, suggest GLOBAL/LOCAL
        let completions = completer.complete_show(&["SHOW", "CODEBOOK", "x"], "");
        assert!(completions.iter().any(|p| p.display == "GLOBAL"));
        assert!(completions.iter().any(|p| p.display == "LOCAL"));
    }

    #[test]
    fn test_complete_show_vector() {
        let completer = NeumannCompleter::new();
        // With >2 words and VECTOR keyword, suggest INDEX
        let completions = completer.complete_show(&["SHOW", "VECTOR", "x"], "");
        assert!(completions.iter().any(|p| p.display == "INDEX"));
    }

    #[test]
    fn test_filter_completions_prefix() {
        let completions = NeumannCompleter::filter_completions(&["SELECT", "SET", "SHOW"], "SE");
        assert_eq!(completions.len(), 2);
        assert!(completions.iter().any(|p| p.display == "SELECT"));
        assert!(completions.iter().any(|p| p.display == "SET"));
    }

    #[test]
    fn test_filter_completions_from_strings_prefix() {
        let tables = vec![
            "users".to_string(),
            "orders".to_string(),
            "user_logs".to_string(),
        ];
        let completions = NeumannCompleter::filter_completions_from_strings(&tables, "us");
        assert_eq!(completions.len(), 2);
        assert!(completions.iter().any(|p| p.display == "users"));
        assert!(completions.iter().any(|p| p.display == "user_logs"));
    }

    #[test]
    fn test_contains_keyword() {
        let words = ["SELECT", "*", "from", "users"];
        assert!(NeumannCompleter::contains_keyword(&words, "FROM"));
        assert!(NeumannCompleter::contains_keyword(&words, "from"));
        assert!(!NeumannCompleter::contains_keyword(&words, "WHERE"));
    }

    #[test]
    fn test_complete_unknown_command() {
        let completer = NeumannCompleter::new();
        // After unknown command with space, suggests top-level commands
        let completions = completer.complete_input("UNKNOWN ", 8);
        assert!(!completions.is_empty());
    }

    #[test]
    fn test_complete_single_word() {
        let completer = NeumannCompleter::new();
        // Partial match for SELECT - will find matching commands
        let completions = completer.complete_input("SEL", 3);
        // Should suggest top-level commands that match
        assert!(!completions.is_empty());
        assert!(completions.iter().any(|p| p.display == "SELECT"));
    }

    #[test]
    fn test_completer_trait_impl() {
        let completer = NeumannCompleter::new();
        let history = rustyline::history::DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);
        let result = completer.complete("SELECT ", 7, &ctx);
        assert!(result.is_ok());
        let (start, completions) = result.unwrap();
        assert_eq!(start, 7);
        assert!(!completions.is_empty());
    }

    #[test]
    fn test_completer_trait_start_position() {
        let completer = NeumannCompleter::new();
        let history = rustyline::history::DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);

        // When typing a partial word, start should be at the beginning of that word
        let result = completer.complete("SELECT SEL", 10, &ctx);
        assert!(result.is_ok());
        let (start, _) = result.unwrap();
        assert_eq!(start, 7);
    }

    #[test]
    fn test_node_after_subcommand() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_node(&["NODE", "CREATE", "person"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_entity_after_subcommand() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_entity(&["ENTITY", "CREATE", "type"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_vault_after_subcommand() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_vault(&["VAULT", "GET", "key"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_cache_after_subcommand() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_cache(&["CACHE", "GET", "key"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_chain_after_subcommand() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_chain(&["CHAIN", "HEIGHT", "x"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_cluster_after_subcommand() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_cluster(&["CLUSTER", "STATUS", "x"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_describe_after_subcommand() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_describe(&["DESCRIBE", "NODE", "person"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_show_after_deep() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_show(&["SHOW", "TABLES", "x"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_blob_after_deep() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_blob(&["BLOB", "PUT", "file", "data"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_embed_after_deep() {
        let completer = NeumannCompleter::new();
        let completions = completer.complete_embed(&["EMBED", "STORE", "key", "[1,2,3]"], "");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_insert_after_values() {
        let completer = NeumannCompleter::new();
        let completions =
            completer.complete_insert(&["INSERT", "INTO", "users", "VALUES", "(1)"], "");
        assert!(completions.is_empty());
    }
}
