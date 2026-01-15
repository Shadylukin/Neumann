//! Corpus builder for training data.
//!
//! Downloads from:
//! - Dolly-15k instruction-following dataset
//! - TinyStories for narrative understanding

use std::{collections::HashMap, fs, path::Path};

use serde::Deserialize;

pub type CorpusResult<T> = Result<T, CorpusError>;

#[derive(Debug)]
pub enum CorpusError {
    Io(std::io::Error),
    EmptyInput(String),
    DownloadFailed(String),
}

impl From<std::io::Error> for CorpusError {
    fn from(e: std::io::Error) -> Self {
        CorpusError::Io(e)
    }
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorpusError::Io(e) => write!(f, "IO error: {}", e),
            CorpusError::EmptyInput(s) => write!(f, "Empty input: {}", s),
            CorpusError::DownloadFailed(s) => write!(f, "Download failed: {}", s),
        }
    }
}

impl std::error::Error for CorpusError {}

#[derive(Debug, Deserialize)]
struct DollyRecord {
    instruction: String,
    context: Option<String>,
    response: String,
}

#[derive(Debug, Clone)]
pub struct CorpusConfig {
    pub max_chars: usize,
    pub dolly_limit: usize,
    pub stories_chars: usize,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        Self {
            max_chars: 2_000_000,
            dolly_limit: 5000,
            stories_chars: 1_000_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextCorpus {
    pub text: String,
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: HashMap<usize, char>,
    pub vocab_size: usize,
}

impl TextCorpus {
    pub fn load_or_build(path: &Path, config: &CorpusConfig) -> CorpusResult<Self> {
        let text = if path.exists() {
            println!("Loading corpus: {}", path.display());
            fs::read_to_string(path)?
        } else {
            println!("Building corpus...");
            let text = build_corpus(config)?;
            fs::write(path, &text)?;
            println!("Saved: {} ({} chars)", path.display(), text.len());
            text
        };
        Self::from_text(text)
    }

    pub fn from_text(text: String) -> CorpusResult<Self> {
        if text.is_empty() {
            return Err(CorpusError::EmptyInput("Empty corpus".into()));
        }

        let mut chars: Vec<char> = text
            .chars()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort();

        let char_to_idx: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let idx_to_char: HashMap<usize, char> =
            chars.iter().enumerate().map(|(i, &c)| (i, c)).collect();
        let vocab_size = chars.len();

        println!("Corpus: {} chars | Vocab: {}", text.len(), vocab_size);

        Ok(Self {
            text,
            char_to_idx,
            idx_to_char,
            vocab_size,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| *self.char_to_idx.get(&c).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .map(|&i| *self.idx_to_char.get(&i).unwrap_or(&'?'))
            .collect()
    }

    pub fn create_sequences(&self, seq_len: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let encoded = self.encode(&self.text);
        (0..encoded.len().saturating_sub(seq_len + 1))
            .map(|i| {
                (
                    encoded[i..i + seq_len].to_vec(),
                    encoded[i + 1..i + seq_len + 1].to_vec(),
                )
            })
            .collect()
    }
}

fn build_corpus(config: &CorpusConfig) -> CorpusResult<String> {
    let mut parts: Vec<String> = Vec::new();

    println!("  Downloading Dolly-15k...");
    let dolly = download_dolly(config.dolly_limit)?;
    println!("    {} examples", dolly.matches("---").count());
    parts.push(dolly);

    println!("  Downloading TinyStories...");
    let stories = download_tinystories(config.stories_chars)?;
    println!("    {} chars", stories.len());
    parts.push(stories);

    use rand::seq::SliceRandom;
    parts.shuffle(&mut rand::thread_rng());

    let text: String = parts.concat().chars().take(config.max_chars).collect();
    println!("  Total: {} chars", text.len());
    Ok(text)
}

fn http_client() -> CorpusResult<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .user_agent("SeedModel/1.0")
        .build()
        .map_err(|e| CorpusError::DownloadFailed(format!("HTTP error: {}", e)))
}

fn download_dolly(limit: usize) -> CorpusResult<String> {
    let url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl";
    let text = http_client()?
        .get(url)
        .send()
        .map_err(|e| CorpusError::DownloadFailed(format!("Dolly download failed: {}", e)))?
        .text()
        .map_err(|e| CorpusError::DownloadFailed(format!("Dolly read failed: {}", e)))?;

    let mut corpus = String::new();
    for line in text.lines().take(limit) {
        if let Ok(r) = serde_json::from_str::<DollyRecord>(line) {
            let q = r.instruction.trim();
            let a = r.response.trim();
            if q.is_empty() || a.is_empty() {
                continue;
            }

            match r
                .context
                .as_ref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                Some(ctx) => corpus.push_str(&format!(
                    "CONTEXT:\n{ctx}\n\nQUESTION:\n{q}\n\nANSWER:\n{a}\n\n---\n\n"
                )),
                None => corpus.push_str(&format!("QUESTION:\n{q}\n\nANSWER:\n{a}\n\n---\n\n")),
            }
        }
    }

    if corpus.is_empty() {
        return Err(CorpusError::DownloadFailed(
            "No Dolly records parsed".into(),
        ));
    }
    Ok(corpus)
}

fn download_tinystories(char_limit: usize) -> CorpusResult<String> {
    let url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt";
    let response = http_client()?
        .get(url)
        .send()
        .map_err(|e| CorpusError::DownloadFailed(format!("TinyStories download failed: {}", e)))?;

    if !response.status().is_success() {
        return Err(CorpusError::DownloadFailed(format!(
            "TinyStories HTTP {}",
            response.status()
        )));
    }

    let text = response
        .text()
        .map_err(|e| CorpusError::DownloadFailed(format!("TinyStories read failed: {}", e)))?;

    if text.is_empty() {
        return Err(CorpusError::DownloadFailed(
            "TinyStories returned empty".into(),
        ));
    }

    Ok(text.chars().take(char_limit).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_from_text() {
        let text = "QUESTION: test\nANSWER: response".to_string();
        let corpus = TextCorpus::from_text(text.clone()).unwrap();
        assert!(corpus.vocab_size > 0);
        assert_eq!(corpus.decode(&corpus.encode(&text)), text);
    }

    #[test]
    fn test_create_sequences() {
        let corpus = TextCorpus::from_text("abcdefghij".to_string()).unwrap();
        let sequences = corpus.create_sequences(3);
        assert!(!sequences.is_empty());
        assert_eq!(sequences[0].0.len(), 3);
        assert_eq!(sequences[0].1.len(), 3);
    }
}
