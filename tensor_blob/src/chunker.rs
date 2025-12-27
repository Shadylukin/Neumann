use sha2::{Digest, Sha256};

/// A content-addressed chunk of data.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Content hash in format "sha256:{hex}".
    pub hash: String,
    /// Raw chunk data.
    pub data: Vec<u8>,
    /// Size of the chunk in bytes.
    pub size: usize,
}

impl Chunk {
    pub fn new(data: Vec<u8>) -> Self {
        let hash = compute_hash(&data);
        let size = data.len();
        Self { hash, data, size }
    }

    pub fn key(&self) -> String {
        format!("_blob:chunk:{}", self.hash)
    }
}

/// Chunker for splitting data into content-addressable chunks.
pub struct Chunker {
    chunk_size: usize,
}

impl Chunker {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Split data into chunks.
    pub fn chunk<'a>(&'a self, data: &'a [u8]) -> impl Iterator<Item = Chunk> + 'a {
        data.chunks(self.chunk_size).map(|chunk_data| {
            let hash = compute_hash(chunk_data);
            Chunk {
                hash,
                data: chunk_data.to_vec(),
                size: chunk_data.len(),
            }
        })
    }

    /// Count how many chunks data would produce without allocating.
    pub fn chunk_count(&self, data_len: usize) -> usize {
        if data_len == 0 {
            0
        } else {
            data_len.div_ceil(self.chunk_size)
        }
    }
}

/// Compute SHA-256 hash of data.
pub fn compute_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("sha256:{:x}", result)
}

/// Compute SHA-256 hash of multiple data segments.
pub fn compute_hash_streaming<'a>(segments: impl Iterator<Item = &'a [u8]>) -> String {
    let mut hasher = Sha256::new();
    for segment in segments {
        hasher.update(segment);
    }
    let result = hasher.finalize();
    format!("sha256:{:x}", result)
}

/// A streaming hasher for computing checksums incrementally.
pub struct StreamingHasher {
    hasher: Sha256,
}

impl Default for StreamingHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingHasher {
    pub fn new() -> Self {
        Self {
            hasher: Sha256::new(),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    pub fn finalize(self) -> String {
        let result = self.hasher.finalize();
        format!("sha256:{:x}", result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hash() {
        let data = b"hello world";
        let hash = compute_hash(data);
        // SHA-256 of "hello world"
        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), 7 + 64); // "sha256:" + 64 hex chars
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let data = b"test data";
        let hash1 = compute_hash(data);
        let hash2 = compute_hash(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_compute_hash_different_data() {
        let hash1 = compute_hash(b"data1");
        let hash2 = compute_hash(b"data2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_compute_hash_empty() {
        let hash = compute_hash(b"");
        assert!(hash.starts_with("sha256:"));
    }

    #[test]
    fn test_chunker_single_chunk() {
        let chunker = Chunker::new(1024);
        let data = vec![0u8; 100];
        let chunks: Vec<_> = chunker.chunk(&data).collect();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].size, 100);
        assert_eq!(chunks[0].data, data);
    }

    #[test]
    fn test_chunker_multiple_chunks() {
        let chunker = Chunker::new(100);
        let data = vec![0u8; 250];
        let chunks: Vec<_> = chunker.chunk(&data).collect();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].size, 100);
        assert_eq!(chunks[1].size, 100);
        assert_eq!(chunks[2].size, 50);
    }

    #[test]
    fn test_chunker_exact_multiple() {
        let chunker = Chunker::new(100);
        let data = vec![0u8; 300];
        let chunks: Vec<_> = chunker.chunk(&data).collect();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].size, 100);
        assert_eq!(chunks[1].size, 100);
        assert_eq!(chunks[2].size, 100);
    }

    #[test]
    fn test_chunker_empty_data() {
        let chunker = Chunker::new(100);
        let data: Vec<u8> = vec![];
        let chunks: Vec<_> = chunker.chunk(&data).collect();

        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_chunk_count() {
        let chunker = Chunker::new(100);
        assert_eq!(chunker.chunk_count(0), 0);
        assert_eq!(chunker.chunk_count(1), 1);
        assert_eq!(chunker.chunk_count(100), 1);
        assert_eq!(chunker.chunk_count(101), 2);
        assert_eq!(chunker.chunk_count(200), 2);
        assert_eq!(chunker.chunk_count(250), 3);
    }

    #[test]
    fn test_chunk_key() {
        let chunk = Chunk::new(vec![1, 2, 3]);
        assert!(chunk.key().starts_with("_blob:chunk:sha256:"));
    }

    #[test]
    fn test_streaming_hasher() {
        let mut hasher = StreamingHasher::new();
        hasher.update(b"hello ");
        hasher.update(b"world");
        let hash = hasher.finalize();

        let direct_hash = compute_hash(b"hello world");
        assert_eq!(hash, direct_hash);
    }

    #[test]
    fn test_compute_hash_streaming() {
        let segments = vec![b"hello ".as_slice(), b"world".as_slice()];
        let hash = compute_hash_streaming(segments.into_iter());

        let direct_hash = compute_hash(b"hello world");
        assert_eq!(hash, direct_hash);
    }

    #[test]
    fn test_chunk_content_addressing() {
        let chunker = Chunker::new(100);

        // Same content should produce same hash
        let data1 = vec![42u8; 100];
        let data2 = vec![42u8; 100];

        let chunks1: Vec<_> = chunker.chunk(&data1).collect();
        let chunks2: Vec<_> = chunker.chunk(&data2).collect();

        assert_eq!(chunks1[0].hash, chunks2[0].hash);
    }

    #[test]
    fn test_chunk_different_content() {
        let chunker = Chunker::new(100);

        let data1 = vec![1u8; 100];
        let data2 = vec![2u8; 100];

        let chunks1: Vec<_> = chunker.chunk(&data1).collect();
        let chunks2: Vec<_> = chunker.chunk(&data2).collect();

        assert_ne!(chunks1[0].hash, chunks2[0].hash);
    }
}
