//! Full-text search index for string properties.
//!
//! Provides tokenization and inverted index for text search capabilities.

#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::needless_range_loop)]

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;

use crate::{GraphEngine, IndexTarget, Node, PropertyValue, Result};

/// Configuration for full-text index.
#[derive(Debug, Clone)]
pub struct FullTextConfig {
    /// Minimum token length to index.
    pub min_token_length: usize,
    /// Whether to convert tokens to lowercase.
    pub case_insensitive: bool,
    /// Stop words to exclude from indexing.
    pub stop_words: HashSet<String>,
}

impl Default for FullTextConfig {
    fn default() -> Self {
        Self {
            min_token_length: 2,
            case_insensitive: true,
            stop_words: default_stop_words(),
        }
    }
}

impl FullTextConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn min_token_length(mut self, len: usize) -> Self {
        self.min_token_length = len;
        self
    }

    #[must_use]
    pub const fn case_insensitive(mut self, value: bool) -> Self {
        self.case_insensitive = value;
        self
    }

    #[must_use]
    pub fn stop_words(mut self, words: HashSet<String>) -> Self {
        self.stop_words = words;
        self
    }
}

fn default_stop_words() -> HashSet<String> {
    [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "is", "it",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Full-text search index using an inverted index structure.
pub struct FullTextIndex {
    /// Property being indexed.
    property: String,
    /// Target (Node or Edge).
    target: IndexTarget,
    /// Inverted index: token -> set of entity IDs.
    index: RwLock<HashMap<String, HashSet<u64>>>,
    /// Forward index: entity ID -> set of tokens (for updates/deletes).
    forward: RwLock<HashMap<u64, HashSet<String>>>,
    /// Configuration.
    config: FullTextConfig,
}

impl FullTextIndex {
    pub fn new(property: impl Into<String>, target: IndexTarget, config: FullTextConfig) -> Self {
        Self {
            property: property.into(),
            target,
            index: RwLock::new(HashMap::new()),
            forward: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Tokenize a string into searchable tokens.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() >= self.config.min_token_length)
            .map(|s| {
                if self.config.case_insensitive {
                    s.to_lowercase()
                } else {
                    s.to_string()
                }
            })
            .filter(|s| !self.config.stop_words.contains(s))
            .collect()
    }

    /// Index an entity.
    pub fn index_entity(&self, id: u64, text: &str) {
        let tokens: HashSet<String> = self.tokenize(text).into_iter().collect();

        let mut index = self.index.write();
        let mut forward = self.forward.write();

        // Remove old tokens if entity was previously indexed
        if let Some(old_tokens) = forward.remove(&id) {
            for token in old_tokens {
                if let Some(ids) = index.get_mut(&token) {
                    ids.remove(&id);
                    if ids.is_empty() {
                        index.remove(&token);
                    }
                }
            }
        }

        // Add new tokens
        for token in &tokens {
            index.entry(token.clone()).or_default().insert(id);
        }
        forward.insert(id, tokens);
    }

    /// Remove an entity from the index.
    pub fn remove_entity(&self, id: u64) {
        let mut index = self.index.write();
        let mut forward = self.forward.write();

        if let Some(tokens) = forward.remove(&id) {
            for token in tokens {
                if let Some(ids) = index.get_mut(&token) {
                    ids.remove(&id);
                    if ids.is_empty() {
                        index.remove(&token);
                    }
                }
            }
        }
    }

    /// Search for entities matching a query.
    pub fn search(&self, query: &str) -> Vec<u64> {
        let tokens = self.tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let index = self.index.read();

        // Find entities that contain ALL query tokens (AND semantics)
        let mut result: Option<HashSet<u64>> = None;

        for token in &tokens {
            if let Some(ids) = index.get(token) {
                result = Some(match result {
                    Some(r) => r.intersection(ids).copied().collect(),
                    None => ids.clone(),
                });
            } else {
                // Token not found, no results
                return Vec::new();
            }
        }

        result.map_or_else(Vec::new, |r| r.into_iter().collect())
    }

    /// Search with prefix matching.
    pub fn search_prefix(&self, prefix: &str) -> Vec<u64> {
        let prefix = if self.config.case_insensitive {
            prefix.to_lowercase()
        } else {
            prefix.to_string()
        };

        let index = self.index.read();
        let mut result = HashSet::new();

        for (token, ids) in index.iter() {
            if token.starts_with(&prefix) {
                result.extend(ids);
            }
        }

        result.into_iter().collect()
    }

    /// Search with fuzzy matching (Levenshtein distance).
    pub fn search_fuzzy(&self, query: &str, max_distance: usize) -> Vec<u64> {
        let query = if self.config.case_insensitive {
            query.to_lowercase()
        } else {
            query.to_string()
        };

        let index = self.index.read();
        let mut result = HashSet::new();

        for (token, ids) in index.iter() {
            if levenshtein_distance(token, &query) <= max_distance {
                result.extend(ids);
            }
        }

        result.into_iter().collect()
    }

    #[must_use]
    pub fn property(&self) -> &str {
        &self.property
    }

    #[must_use]
    pub const fn target(&self) -> IndexTarget {
        self.target
    }
}

/// Calculate Levenshtein distance between two strings.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut matrix = vec![vec![0; b_len + 1]; a_len + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(a_len + 1) {
        row[0] = i;
    }
    for j in 0..=b_len {
        matrix[0][j] = j;
    }

    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);
        }
    }

    matrix[a_len][b_len]
}

impl GraphEngine {
    /// Create a full-text index on a node property.
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails.
    pub fn create_fulltext_index(&self, property: &str) -> Result<()> {
        self.create_fulltext_index_with_config(property, FullTextConfig::default())
    }

    /// Create a full-text index with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails.
    pub fn create_fulltext_index_with_config(
        &self,
        property: &str,
        config: FullTextConfig,
    ) -> Result<()> {
        let index = FullTextIndex::new(property, IndexTarget::Node, config);

        // Index all existing nodes
        let nodes = self.get_all_node_ids()?;
        for node_id in nodes {
            if let Ok(node) = self.get_node(node_id) {
                if let Some(PropertyValue::String(text)) = node.properties.get(property) {
                    index.index_entity(node_id, text);
                }
            }
        }

        self.fulltext_indexes
            .write()
            .insert(property.to_string(), index);
        Ok(())
    }

    /// Perform a full-text search.
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn fulltext_search(&self, property: &str, query: &str) -> Result<Vec<Node>> {
        let indexes = self.fulltext_indexes.read();
        let index = indexes
            .get(property)
            .ok_or_else(|| crate::GraphError::IndexNotFound {
                target: "fulltext".to_string(),
                property: property.to_string(),
            })?;

        let ids = index.search(query);
        ids.into_iter().map(|id| self.get_node(id)).collect()
    }

    /// Perform a prefix-based full-text search.
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn fulltext_search_prefix(&self, property: &str, prefix: &str) -> Result<Vec<Node>> {
        let indexes = self.fulltext_indexes.read();
        let index = indexes
            .get(property)
            .ok_or_else(|| crate::GraphError::IndexNotFound {
                target: "fulltext".to_string(),
                property: property.to_string(),
            })?;

        let ids = index.search_prefix(prefix);
        ids.into_iter().map(|id| self.get_node(id)).collect()
    }

    /// Perform a fuzzy full-text search.
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn fulltext_search_fuzzy(
        &self,
        property: &str,
        query: &str,
        max_distance: usize,
    ) -> Result<Vec<Node>> {
        let indexes = self.fulltext_indexes.read();
        let index = indexes
            .get(property)
            .ok_or_else(|| crate::GraphError::IndexNotFound {
                target: "fulltext".to_string(),
                property: property.to_string(),
            })?;

        let ids = index.search_fuzzy(query, max_distance);
        ids.into_iter().map(|id| self.get_node(id)).collect()
    }

    /// Check if a full-text index exists.
    #[must_use]
    pub fn has_fulltext_index(&self, property: &str) -> bool {
        self.fulltext_indexes.read().contains_key(property)
    }

    /// Drop a full-text index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn drop_fulltext_index(&self, property: &str) -> Result<()> {
        self.fulltext_indexes
            .write()
            .remove(property)
            .map(|_| ())
            .ok_or_else(|| crate::GraphError::IndexNotFound {
                target: "fulltext".to_string(),
                property: property.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let index = FullTextIndex::new("test", IndexTarget::Node, FullTextConfig::default());
        let tokens = index.tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Stop words should be excluded
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_fulltext_search() {
        let index = FullTextIndex::new("content", IndexTarget::Node, FullTextConfig::default());

        index.index_entity(1, "The quick brown fox jumps over the lazy dog");
        index.index_entity(2, "A quick brown cat sleeps all day");
        index.index_entity(3, "The lazy dog wakes up");

        // Search for "quick"
        let results = index.search("quick");
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(!results.contains(&3));

        // Search for "lazy dog" (AND semantics)
        let results = index.search("lazy dog");
        assert!(results.contains(&1));
        assert!(results.contains(&3));
        assert!(!results.contains(&2));
    }

    #[test]
    fn test_prefix_search() {
        let index = FullTextIndex::new("content", IndexTarget::Node, FullTextConfig::default());

        index.index_entity(1, "programming is fun");
        index.index_entity(2, "program execution");
        index.index_entity(3, "professional development");

        let results = index.search_prefix("prog");
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(!results.contains(&3));

        let results = index.search_prefix("pro");
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(results.contains(&3));
    }

    #[test]
    fn test_fuzzy_search() {
        let index = FullTextIndex::new("content", IndexTarget::Node, FullTextConfig::default());

        index.index_entity(1, "hello world");
        index.index_entity(2, "hallo welt");

        // Fuzzy search with distance 1
        let results = index.search_fuzzy("helo", 1);
        assert!(results.contains(&1)); // "hello" is distance 1 from "helo"

        // Fuzzy search with distance 2
        let results = index.search_fuzzy("helo", 2);
        assert!(results.contains(&1));
        assert!(results.contains(&2)); // "hallo" is distance 2 from "helo"
    }

    #[test]
    fn test_remove_entity() {
        let index = FullTextIndex::new("content", IndexTarget::Node, FullTextConfig::default());

        index.index_entity(1, "hello world");
        index.index_entity(2, "hello universe");

        let results = index.search("hello");
        assert_eq!(results.len(), 2);

        index.remove_entity(1);

        let results = index.search("hello");
        assert_eq!(results.len(), 1);
        assert!(results.contains(&2));
    }

    #[test]
    fn test_update_entity() {
        let index = FullTextIndex::new("content", IndexTarget::Node, FullTextConfig::default());

        index.index_entity(1, "hello world");
        assert!(!index.search("hello").is_empty());
        assert!(index.search("goodbye").is_empty());

        // Update the entity
        index.index_entity(1, "goodbye world");
        assert!(index.search("hello").is_empty());
        assert!(!index.search("goodbye").is_empty());
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("abc", "ab"), 1);
        assert_eq!(levenshtein_distance("abc", "abcd"), 1);
        assert_eq!(levenshtein_distance("abc", "adc"), 1);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_graph_engine_fulltext() {
        let engine = GraphEngine::new();

        let n1 = engine
            .create_node(
                "Doc",
                [(
                    "content".to_string(),
                    PropertyValue::String("rust programming language".to_string()),
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let n2 = engine
            .create_node(
                "Doc",
                [(
                    "content".to_string(),
                    PropertyValue::String("python programming guide".to_string()),
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let _n3 = engine
            .create_node(
                "Doc",
                [(
                    "content".to_string(),
                    PropertyValue::String("database management systems".to_string()),
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine.create_fulltext_index("content").unwrap();

        let results = engine.fulltext_search("content", "programming").unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<u64> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&n1));
        assert!(ids.contains(&n2));
    }

    #[test]
    fn test_fulltext_config_builder() {
        let config = FullTextConfig::new()
            .min_token_length(3)
            .case_insensitive(false)
            .stop_words(["foo", "bar"].into_iter().map(String::from).collect());

        assert_eq!(config.min_token_length, 3);
        assert!(!config.case_insensitive);
        assert!(config.stop_words.contains("foo"));
        assert!(config.stop_words.contains("bar"));
    }

    #[test]
    fn test_fulltext_index_property_and_target() {
        let index = FullTextIndex::new("name", IndexTarget::Node, FullTextConfig::default());
        assert_eq!(index.property(), "name");
        assert_eq!(index.target(), IndexTarget::Node);
    }

    #[test]
    fn test_tokenize_case_sensitive() {
        let config = FullTextConfig::new().case_insensitive(false);
        let index = FullTextIndex::new("test", IndexTarget::Node, config);
        let tokens = index.tokenize("Hello World");
        assert!(tokens.contains(&"Hello".to_string()));
        assert!(tokens.contains(&"World".to_string()));
        assert!(!tokens.contains(&"hello".to_string()));
    }

    #[test]
    fn test_tokenize_min_length() {
        let config = FullTextConfig::new().min_token_length(5);
        let index = FullTextIndex::new("test", IndexTarget::Node, config);
        let tokens = index.tokenize("Hi there everyone");
        assert!(!tokens.contains(&"hi".to_string()));
        assert!(tokens.contains(&"there".to_string()));
        assert!(tokens.contains(&"everyone".to_string()));
    }

    #[test]
    fn test_search_empty_query() {
        let index = FullTextIndex::new("test", IndexTarget::Node, FullTextConfig::default());
        index.index_entity(1, "hello world");
        let results = index.search("");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_no_match() {
        let index = FullTextIndex::new("test", IndexTarget::Node, FullTextConfig::default());
        index.index_entity(1, "hello world");
        let results = index.search("xyz");
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_entity() {
        let index = FullTextIndex::new("test", IndexTarget::Node, FullTextConfig::default());
        index.remove_entity(999); // Should not panic
    }

    #[test]
    fn test_graph_engine_fulltext_prefix() {
        let engine = GraphEngine::new();

        engine
            .create_node(
                "Doc",
                [(
                    "title".to_string(),
                    PropertyValue::String("programming tutorial".to_string()),
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine
            .create_node(
                "Doc",
                [(
                    "title".to_string(),
                    PropertyValue::String("program guide".to_string()),
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine.create_fulltext_index("title").unwrap();

        let results = engine.fulltext_search_prefix("title", "prog").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_graph_engine_fulltext_fuzzy() {
        let engine = GraphEngine::new();

        engine
            .create_node(
                "Doc",
                [(
                    "title".to_string(),
                    PropertyValue::String("hello world".to_string()),
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        engine.create_fulltext_index("title").unwrap();

        let results = engine.fulltext_search_fuzzy("title", "helo", 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_graph_engine_has_and_drop_fulltext_index() {
        let engine = GraphEngine::new();

        assert!(!engine.has_fulltext_index("content"));
        engine.create_fulltext_index("content").unwrap();
        assert!(engine.has_fulltext_index("content"));
        engine.drop_fulltext_index("content").unwrap();
        assert!(!engine.has_fulltext_index("content"));
    }

    #[test]
    fn test_graph_engine_fulltext_index_not_found() {
        let engine = GraphEngine::new();
        let result = engine.fulltext_search("nonexistent", "query");
        assert!(result.is_err());

        let result = engine.fulltext_search_prefix("nonexistent", "query");
        assert!(result.is_err());

        let result = engine.fulltext_search_fuzzy("nonexistent", "query", 1);
        assert!(result.is_err());

        let result = engine.drop_fulltext_index("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_engine_fulltext_with_config() {
        let engine = GraphEngine::new();

        engine
            .create_node(
                "Doc",
                [(
                    "body".to_string(),
                    PropertyValue::String("HELLO WORLD".to_string()),
                )]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let config = FullTextConfig::new().case_insensitive(false);
        engine
            .create_fulltext_index_with_config("body", config)
            .unwrap();

        // Case-sensitive search
        let results = engine.fulltext_search("body", "HELLO").unwrap();
        assert_eq!(results.len(), 1);

        let results = engine.fulltext_search("body", "hello").unwrap();
        assert!(results.is_empty());
    }
}
