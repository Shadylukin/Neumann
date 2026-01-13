use std::time::{SystemTime, UNIX_EPOCH};

use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{
    chunker::{Chunk, Chunker, StreamingHasher},
    error::{BlobError, Result},
    gc::increment_chunk_refs,
    metadata::PutOptions,
};

/// Internal state shared between BlobWriter and the store.
pub(crate) struct WriteState {
    pub artifact_id: String,
    pub filename: String,
    pub content_type: String,
    pub created_by: String,
    pub linked_to: Vec<String>,
    pub tags: Vec<String>,
    pub custom_metadata: std::collections::HashMap<String, String>,
    pub embedding: Option<(Vec<f32>, String)>,
}

/// Streaming writer for uploading artifacts.
pub struct BlobWriter {
    store: TensorStore,
    chunker: Chunker,
    state: WriteState,
    chunks: Vec<String>,
    total_size: usize,
    hasher: StreamingHasher,
    buffer: Vec<u8>,
}

impl BlobWriter {
    pub(crate) fn new(
        store: TensorStore,
        chunk_size: usize,
        artifact_id: String,
        filename: String,
        options: PutOptions,
        default_content_type: &str,
    ) -> Self {
        Self {
            store,
            chunker: Chunker::new(chunk_size),
            state: WriteState {
                artifact_id,
                filename,
                content_type: options
                    .content_type
                    .unwrap_or_else(|| default_content_type.to_string()),
                created_by: options.created_by.unwrap_or_default(),
                linked_to: options.linked_to,
                tags: options.tags,
                custom_metadata: options.metadata,
                embedding: options.embedding,
            },
            chunks: Vec::new(),
            total_size: 0,
            hasher: StreamingHasher::new(),
            buffer: Vec::new(),
        }
    }

    /// Write data to the artifact. Data is chunked and stored incrementally.
    pub async fn write(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        // Update the full-file hash
        self.hasher.update(data);
        self.total_size += data.len();

        // Add to buffer
        self.buffer.extend_from_slice(data);

        // Process complete chunks
        while self.buffer.len() >= self.chunker.chunk_size() {
            let chunk_data: Vec<u8> = self.buffer.drain(..self.chunker.chunk_size()).collect();
            let chunk = Chunk::new(chunk_data);
            self.store_chunk(chunk).await?;
        }

        Ok(())
    }

    /// Store a chunk, handling deduplication.
    async fn store_chunk(&mut self, chunk: Chunk) -> Result<()> {
        let chunk_key = chunk.key();

        // Check if chunk already exists (deduplication)
        if self.store.exists(&chunk_key) {
            // Increment reference count
            increment_chunk_refs(&self.store, &chunk_key)?;
        } else {
            // Store new chunk
            let mut tensor = TensorData::new();
            tensor.set(
                "_type",
                TensorValue::Scalar(ScalarValue::String("blob_chunk".to_string())),
            );
            tensor.set("_data", TensorValue::Scalar(ScalarValue::Bytes(chunk.data)));
            tensor.set(
                "_size",
                TensorValue::Scalar(ScalarValue::Int(chunk.size as i64)),
            );
            tensor.set("_refs", TensorValue::Scalar(ScalarValue::Int(1)));
            tensor.set(
                "_created",
                TensorValue::Scalar(ScalarValue::Int(current_timestamp() as i64)),
            );

            self.store.put(&chunk_key, tensor)?;
        }

        self.chunks.push(chunk_key);
        Ok(())
    }

    /// Finalize the artifact and return its ID.
    pub async fn finish(mut self) -> Result<String> {
        // Flush remaining buffer
        if !self.buffer.is_empty() {
            let chunk = Chunk::new(std::mem::take(&mut self.buffer));
            self.store_chunk(chunk).await?;
        }

        let checksum = self.hasher.finalize();
        let now = current_timestamp();

        // Build metadata tensor
        let mut tensor = TensorData::new();
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("blob_artifact".to_string())),
        );
        tensor.set(
            "_id",
            TensorValue::Scalar(ScalarValue::String(self.state.artifact_id.clone())),
        );
        tensor.set(
            "_filename",
            TensorValue::Scalar(ScalarValue::String(self.state.filename)),
        );
        tensor.set(
            "_content_type",
            TensorValue::Scalar(ScalarValue::String(self.state.content_type)),
        );
        tensor.set(
            "_size",
            TensorValue::Scalar(ScalarValue::Int(self.total_size as i64)),
        );
        tensor.set(
            "_checksum",
            TensorValue::Scalar(ScalarValue::String(checksum)),
        );
        tensor.set(
            "_chunk_size",
            TensorValue::Scalar(ScalarValue::Int(self.chunker.chunk_size() as i64)),
        );
        tensor.set(
            "_chunk_count",
            TensorValue::Scalar(ScalarValue::Int(self.chunks.len() as i64)),
        );
        tensor.set("_chunks", TensorValue::Pointers(self.chunks));
        tensor.set(
            "_created",
            TensorValue::Scalar(ScalarValue::Int(now as i64)),
        );
        tensor.set(
            "_modified",
            TensorValue::Scalar(ScalarValue::Int(now as i64)),
        );
        tensor.set(
            "_created_by",
            TensorValue::Scalar(ScalarValue::String(self.state.created_by)),
        );

        if !self.state.linked_to.is_empty() {
            tensor.set("_linked_to", TensorValue::Pointers(self.state.linked_to));
        }

        if !self.state.tags.is_empty() {
            tensor.set(
                "_tags",
                TensorValue::Pointers(
                    self.state
                        .tags
                        .into_iter()
                        .map(|t| format!("tag:{t}"))
                        .collect(),
                ),
            );
        }

        // Custom metadata
        for (key, value) in self.state.custom_metadata {
            tensor.set(
                format!("_meta:{key}"),
                TensorValue::Scalar(ScalarValue::String(value)),
            );
        }

        // Embedding with sparse detection
        if let Some((embedding, model)) = self.state.embedding {
            use tensor_store::SparseVector;
            let storage = if crate::streaming::should_use_sparse(&embedding) {
                TensorValue::Sparse(SparseVector::from_dense(&embedding))
            } else {
                TensorValue::Vector(embedding)
            };
            tensor.set("_embedding", storage);
            tensor.set(
                "_embedded_model",
                TensorValue::Scalar(ScalarValue::String(model)),
            );
        }

        let meta_key = format!("_blob:meta:{}", self.state.artifact_id);
        self.store.put(&meta_key, tensor)?;

        Ok(self.state.artifact_id)
    }

    /// Get the current total size written.
    pub fn bytes_written(&self) -> usize {
        self.total_size
    }

    /// Get the number of chunks written so far.
    pub fn chunks_written(&self) -> usize {
        self.chunks.len()
    }
}

/// Streaming reader for downloading artifacts.
pub struct BlobReader {
    store: TensorStore,
    chunks: Vec<String>,
    current_chunk: usize,
    current_data: Option<Vec<u8>>,
    current_offset: usize,
    total_size: usize,
    bytes_read: usize,
    checksum: String,
}

impl BlobReader {
    pub(crate) fn new(store: TensorStore, artifact_id: &str) -> Result<Self> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = store
            .get(&meta_key)
            .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

        let chunks = get_pointers(&tensor, "_chunks")
            .ok_or_else(|| BlobError::NotFound(format!("chunks for {artifact_id}")))?;
        let total_size = get_int(&tensor, "_size").unwrap_or(0) as usize;
        let checksum = get_string(&tensor, "_checksum").unwrap_or_default();

        Ok(Self {
            store,
            chunks,
            current_chunk: 0,
            current_data: None,
            current_offset: 0,
            total_size,
            bytes_read: 0,
            checksum,
        })
    }

    /// Read the next chunk. Returns None when all chunks have been read.
    pub async fn next_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        if self.current_chunk >= self.chunks.len() {
            return Ok(None);
        }

        let chunk_key = &self.chunks[self.current_chunk];
        let tensor = self
            .store
            .get(chunk_key)
            .map_err(|_| BlobError::ChunkMissing(chunk_key.clone()))?;

        let data = get_bytes(&tensor, "_data")
            .ok_or_else(|| BlobError::ChunkMissing(chunk_key.clone()))?;

        self.current_chunk += 1;
        self.bytes_read += data.len();

        Ok(Some(data))
    }

    /// Read all remaining data into a single buffer.
    pub async fn read_all(&mut self) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(self.total_size);

        while let Some(chunk) = self.next_chunk().await? {
            result.extend(chunk);
        }

        Ok(result)
    }

    /// Read into a buffer, returning number of bytes read.
    pub async fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        // Load chunk if needed
        if self.current_data.is_none()
            || self.current_offset >= self.current_data.as_ref().map(|d| d.len()).unwrap_or(0)
        {
            match self.next_chunk().await? {
                Some(data) => {
                    self.current_data = Some(data);
                    self.current_offset = 0;
                },
                None => return Ok(0),
            }
        }

        let data = self.current_data.as_ref().unwrap();
        let remaining = &data[self.current_offset..];
        let to_copy = remaining.len().min(buf.len());

        buf[..to_copy].copy_from_slice(&remaining[..to_copy]);
        self.current_offset += to_copy;

        Ok(to_copy)
    }

    /// Verify the artifact checksum.
    pub async fn verify(&mut self) -> Result<bool> {
        let mut hasher = StreamingHasher::new();

        // Reset to start
        self.current_chunk = 0;
        self.bytes_read = 0;

        while let Some(chunk) = self.next_chunk().await? {
            hasher.update(&chunk);
        }

        let actual = hasher.finalize();
        Ok(actual == self.checksum)
    }

    /// Get the expected checksum.
    pub fn checksum(&self) -> &str {
        &self.checksum
    }

    /// Get total artifact size.
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get bytes read so far.
    pub fn bytes_read(&self) -> usize {
        self.bytes_read
    }

    /// Get the number of chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

// Helper functions

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(crate) fn get_int(tensor: &TensorData, field: &str) -> Option<i64> {
    match tensor.get(field) {
        Some(TensorValue::Scalar(ScalarValue::Int(i))) => Some(*i),
        _ => None,
    }
}

pub(crate) fn get_string(tensor: &TensorData, field: &str) -> Option<String> {
    match tensor.get(field) {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => Some(s.clone()),
        _ => None,
    }
}

pub(crate) fn get_bytes(tensor: &TensorData, field: &str) -> Option<Vec<u8>> {
    match tensor.get(field) {
        Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => Some(b.clone()),
        _ => None,
    }
}

pub(crate) fn get_pointers(tensor: &TensorData, field: &str) -> Option<Vec<String>> {
    match tensor.get(field) {
        Some(TensorValue::Pointers(p)) => Some(p.clone()),
        _ => None,
    }
}

#[cfg(feature = "vector")]
pub(crate) fn get_vector(tensor: &TensorData, field: &str) -> Option<Vec<f32>> {
    match tensor.get(field) {
        Some(TensorValue::Vector(v)) => Some(v.clone()),
        Some(TensorValue::Sparse(s)) => Some(s.to_dense()),
        _ => None,
    }
}

/// Check if a vector should use sparse storage (50% threshold).
pub(crate) fn should_use_sparse(vector: &[f32]) -> bool {
    if vector.is_empty() {
        return false;
    }
    let nnz = vector.iter().filter(|&&v| v.abs() > 1e-6).count();
    // For 0.5 threshold: sparse if nnz <= len/2, i.e., nnz*2 <= len
    nnz * 2 <= vector.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::PutOptions;

    fn create_test_store() -> TensorStore {
        TensorStore::new()
    }

    #[tokio::test]
    async fn test_blob_writer_small_file() {
        let store = create_test_store();
        let mut writer = BlobWriter::new(
            store.clone(),
            1024,
            "test-artifact".to_string(),
            "test.txt".to_string(),
            PutOptions::default(),
            "text/plain",
        );

        writer.write(b"hello world").await.unwrap();
        let artifact_id = writer.finish().await.unwrap();

        assert_eq!(artifact_id, "test-artifact");

        // Verify metadata was stored
        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = store.get(&meta_key).unwrap();
        assert_eq!(
            get_string(&tensor, "_filename"),
            Some("test.txt".to_string())
        );
        assert_eq!(get_int(&tensor, "_size"), Some(11));
        assert_eq!(get_int(&tensor, "_chunk_count"), Some(1));
    }

    #[tokio::test]
    async fn test_blob_writer_multi_chunk() {
        let store = create_test_store();
        let chunk_size = 10;
        let mut writer = BlobWriter::new(
            store.clone(),
            chunk_size,
            "multi-chunk".to_string(),
            "data.bin".to_string(),
            PutOptions::default(),
            "application/octet-stream",
        );

        // Write 25 bytes = 3 chunks (10 + 10 + 5)
        writer.write(&[0u8; 25]).await.unwrap();
        let artifact_id = writer.finish().await.unwrap();

        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = store.get(&meta_key).unwrap();
        assert_eq!(get_int(&tensor, "_chunk_count"), Some(3));
        assert_eq!(get_int(&tensor, "_size"), Some(25));
    }

    #[tokio::test]
    async fn test_blob_writer_incremental_write() {
        let store = create_test_store();
        let mut writer = BlobWriter::new(
            store.clone(),
            10,
            "incremental".to_string(),
            "data.bin".to_string(),
            PutOptions::default(),
            "application/octet-stream",
        );

        // Write in small increments
        writer.write(&[1, 2, 3]).await.unwrap();
        writer.write(&[4, 5, 6]).await.unwrap();
        writer.write(&[7, 8, 9, 10, 11, 12]).await.unwrap();

        let artifact_id = writer.finish().await.unwrap();

        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = store.get(&meta_key).unwrap();
        assert_eq!(get_int(&tensor, "_size"), Some(12));
    }

    #[tokio::test]
    async fn test_blob_writer_with_options() {
        let store = create_test_store();
        let options = PutOptions::new()
            .with_content_type("application/pdf")
            .with_created_by("user:alice")
            .with_link("task:123")
            .with_tag("quarterly")
            .with_meta("author", "Alice");

        let mut writer = BlobWriter::new(
            store.clone(),
            1024,
            "with-options".to_string(),
            "report.pdf".to_string(),
            options,
            "application/octet-stream",
        );

        writer.write(b"PDF content").await.unwrap();
        let artifact_id = writer.finish().await.unwrap();

        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = store.get(&meta_key).unwrap();
        assert_eq!(
            get_string(&tensor, "_content_type"),
            Some("application/pdf".to_string())
        );
        assert_eq!(
            get_string(&tensor, "_created_by"),
            Some("user:alice".to_string())
        );
        assert_eq!(
            get_string(&tensor, "_meta:author"),
            Some("Alice".to_string())
        );
    }

    #[tokio::test]
    async fn test_blob_reader_small_file() {
        let store = create_test_store();

        // First write
        let mut writer = BlobWriter::new(
            store.clone(),
            1024,
            "read-test".to_string(),
            "test.txt".to_string(),
            PutOptions::default(),
            "text/plain",
        );
        writer.write(b"hello world").await.unwrap();
        writer.finish().await.unwrap();

        // Then read
        let mut reader = BlobReader::new(store, "read-test").unwrap();
        let data = reader.read_all().await.unwrap();

        assert_eq!(data, b"hello world");
        assert_eq!(reader.bytes_read(), 11);
    }

    #[tokio::test]
    async fn test_blob_reader_multi_chunk() {
        let store = create_test_store();
        let data = vec![42u8; 25];

        // Write
        let mut writer = BlobWriter::new(
            store.clone(),
            10,
            "multi-read".to_string(),
            "data.bin".to_string(),
            PutOptions::default(),
            "application/octet-stream",
        );
        writer.write(&data).await.unwrap();
        writer.finish().await.unwrap();

        // Read
        let mut reader = BlobReader::new(store, "multi-read").unwrap();
        let result = reader.read_all().await.unwrap();

        assert_eq!(result, data);
        assert_eq!(reader.chunk_count(), 3);
    }

    #[tokio::test]
    async fn test_blob_reader_chunk_by_chunk() {
        let store = create_test_store();

        // Write 30 bytes in 10-byte chunks
        let mut writer = BlobWriter::new(
            store.clone(),
            10,
            "chunk-read".to_string(),
            "data.bin".to_string(),
            PutOptions::default(),
            "application/octet-stream",
        );
        writer.write(&[1u8; 30]).await.unwrap();
        writer.finish().await.unwrap();

        // Read chunk by chunk
        let mut reader = BlobReader::new(store, "chunk-read").unwrap();
        let chunk1 = reader.next_chunk().await.unwrap().unwrap();
        let chunk2 = reader.next_chunk().await.unwrap().unwrap();
        let chunk3 = reader.next_chunk().await.unwrap().unwrap();
        let chunk4 = reader.next_chunk().await.unwrap();

        assert_eq!(chunk1.len(), 10);
        assert_eq!(chunk2.len(), 10);
        assert_eq!(chunk3.len(), 10);
        assert!(chunk4.is_none());
    }

    #[tokio::test]
    async fn test_blob_reader_verify() {
        let store = create_test_store();
        let data = b"verification test data";

        // Write
        let mut writer = BlobWriter::new(
            store.clone(),
            1024,
            "verify-test".to_string(),
            "test.txt".to_string(),
            PutOptions::default(),
            "text/plain",
        );
        writer.write(data).await.unwrap();
        writer.finish().await.unwrap();

        // Verify
        let mut reader = BlobReader::new(store, "verify-test").unwrap();
        let valid = reader.verify().await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_blob_reader_not_found() {
        let store = create_test_store();
        let result = BlobReader::new(store, "nonexistent");
        assert!(matches!(result, Err(BlobError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_deduplication() {
        let store = create_test_store();
        let data = vec![42u8; 10];

        // Write same data twice
        let mut writer1 = BlobWriter::new(
            store.clone(),
            10,
            "dedup-1".to_string(),
            "file1.bin".to_string(),
            PutOptions::default(),
            "application/octet-stream",
        );
        writer1.write(&data).await.unwrap();
        writer1.finish().await.unwrap();

        let mut writer2 = BlobWriter::new(
            store.clone(),
            10,
            "dedup-2".to_string(),
            "file2.bin".to_string(),
            PutOptions::default(),
            "application/octet-stream",
        );
        writer2.write(&data).await.unwrap();
        writer2.finish().await.unwrap();

        // Count chunks - should only be 1 due to deduplication
        let chunk_count = store.scan("_blob:chunk:").len();
        assert_eq!(chunk_count, 1);

        // But the chunk should have ref count of 2
        let chunks = store.scan("_blob:chunk:");
        let chunk = store.get(&chunks[0]).unwrap();
        assert_eq!(get_int(&chunk, "_refs"), Some(2));
    }
}
