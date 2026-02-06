// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! TF-IDF embedding generator using feature hashing.

use std::collections::{HashMap, HashSet};

/// Default embedding dimension.
pub const DEFAULT_DIMENSION: usize = 128;

/// Common English stopwords to filter out.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "you", "your", "this",
    "but", "have", "had", "been", "not", "they", "we", "their", "or", "which", "if", "can", "do",
    "into", "no", "so", "than", "when", "what", "all", "more", "some", "other", "most", "also",
    "about", "such", "only", "see",
];

/// TF-IDF based embedding generator.
#[derive(Debug)]
pub struct TfIdfEmbedder {
    /// Number of dimensions in the output embedding.
    dimension: usize,
    /// Document frequency for each term (how many docs contain the term).
    doc_freq: HashMap<String, usize>,
    /// Total number of documents in the corpus.
    doc_count: usize,
    /// Stopwords set for fast lookup.
    stopwords: HashSet<String>,
}

impl Default for TfIdfEmbedder {
    fn default() -> Self {
        Self::new(DEFAULT_DIMENSION)
    }
}

impl TfIdfEmbedder {
    /// Create a new embedder with the specified dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            doc_freq: HashMap::new(),
            doc_count: 0,
            stopwords: STOPWORDS.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Tokenize text into lowercase words, removing stopwords and punctuation.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| {
                let len = w.len();
                len > 1 && len < 50 && !self.stopwords.contains(*w)
            })
            .map(String::from)
            .collect()
    }

    /// Build vocabulary from a corpus of documents.
    pub fn fit(&mut self, documents: &[&str]) {
        self.doc_freq.clear();
        self.doc_count = documents.len();

        for doc in documents {
            // Get unique terms in this document
            let terms: HashSet<String> = self.tokenize(doc).into_iter().collect();
            for term in terms {
                *self.doc_freq.entry(term).or_insert(0) += 1;
            }
        }
    }

    /// Compute IDF (inverse document frequency) for a term.
    fn idf(&self, term: &str) -> f32 {
        let df = self.doc_freq.get(term).copied().unwrap_or(0);
        if df == 0 || self.doc_count == 0 {
            return 0.0;
        }
        // Standard IDF formula: log(N / df)
        ((self.doc_count as f32) / (df as f32)).ln()
    }

    /// Hash a term to a dimension index using FNV-1a hash.
    fn hash_term(&self, term: &str) -> usize {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        for byte in term.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3); // FNV prime
        }
        (hash as usize) % self.dimension
    }

    /// Generate an embedding for a document.
    #[must_use]
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0_f32; self.dimension];
        let tokens = self.tokenize(text);

        if tokens.is_empty() {
            return embedding;
        }

        // Compute term frequencies
        let mut tf: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0) += 1;
        }

        let total_terms = tokens.len() as f32;

        // Compute TF-IDF and hash to embedding
        for (term, count) in tf {
            let tf_val = (count as f32) / total_terms;
            let idf_val = self.idf(&term);
            let tfidf = tf_val * idf_val;

            if tfidf > 0.0 {
                let idx = self.hash_term(&term);
                embedding[idx] += tfidf;
            }
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }

    /// Format an embedding as a query string for EMBED command.
    #[must_use]
    pub fn format_embedding(embedding: &[f32]) -> String {
        embedding
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let embedder = TfIdfEmbedder::new(64);
        let tokens = embedder.tokenize("Hello, World! This is a test.");
        // "a", "is", "this" are stopwords
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(!tokens.contains(&"this".to_string()));
    }

    #[test]
    fn test_fit_updates_doc_freq() {
        let mut embedder = TfIdfEmbedder::new(64);
        embedder.fit(&["hello world", "world again", "hello test"]);

        assert_eq!(embedder.doc_count, 3);
        assert_eq!(*embedder.doc_freq.get("world").unwrap(), 2);
        assert_eq!(*embedder.doc_freq.get("hello").unwrap(), 2);
        assert_eq!(*embedder.doc_freq.get("again").unwrap(), 1);
    }

    #[test]
    fn test_embed_produces_correct_dimension() {
        let mut embedder = TfIdfEmbedder::new(128);
        embedder.fit(&["hello world", "world again"]);
        let embedding = embedder.embed("hello world");
        assert_eq!(embedding.len(), 128);
    }

    #[test]
    fn test_embed_is_normalized() {
        let mut embedder = TfIdfEmbedder::new(128);
        embedder.fit(&["hello world", "world again", "test content"]);
        let embedding = embedder.embed("hello world test");

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Allow small floating-point error
        assert!((norm - 1.0).abs() < 0.001 || norm == 0.0);
    }

    #[test]
    fn test_embed_empty_returns_zeros() {
        let embedder = TfIdfEmbedder::new(64);
        let embedding = embedder.embed("");
        assert!(embedding.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_similar_docs_have_similar_embeddings() {
        let mut embedder = TfIdfEmbedder::new(128);
        embedder.fit(&[
            "rust programming language systems",
            "python programming language scripting",
            "cooking recipes food kitchen",
        ]);

        let rust_emb = embedder.embed("rust programming language");
        let python_emb = embedder.embed("python programming language");
        let cooking_emb = embedder.embed("cooking recipes");

        // Cosine similarity (embeddings are normalized, so dot product = cosine)
        let rust_python: f32 = rust_emb.iter().zip(&python_emb).map(|(a, b)| a * b).sum();
        let rust_cooking: f32 = rust_emb.iter().zip(&cooking_emb).map(|(a, b)| a * b).sum();

        // Programming languages should be more similar to each other than to cooking
        assert!(rust_python > rust_cooking);
    }

    #[test]
    fn test_format_embedding() {
        let embedding = vec![0.1, 0.2, 0.3];
        let formatted = TfIdfEmbedder::format_embedding(&embedding);
        assert_eq!(formatted, "0.100000, 0.200000, 0.300000");
    }

    #[test]
    fn test_hash_term_is_deterministic() {
        let embedder = TfIdfEmbedder::new(128);
        let idx1 = embedder.hash_term("hello");
        let idx2 = embedder.hash_term("hello");
        assert_eq!(idx1, idx2);
        assert!(idx1 < 128);
    }

    #[test]
    fn test_dimension() {
        let embedder = TfIdfEmbedder::new(256);
        assert_eq!(embedder.dimension(), 256);
    }

    #[test]
    fn test_default() {
        let embedder = TfIdfEmbedder::default();
        assert_eq!(embedder.dimension(), DEFAULT_DIMENSION);
    }
}
