// SPDX-License-Identifier: MIT OR Apache-2.0
//! Length-delimited message framing for TCP transport.
//!
//! Wire format v1 (legacy):
//! ```text
//! +------------------+------------------+
//! | Length (4B BE)   | Payload (bincode)|
//! +------------------+------------------+
//! ```
//!
//! Wire format v2 (with compression support):
//! ```text
//! +------------------+-------+------------------+
//! | Length (4B BE)   | Flags | Payload          |
//! +------------------+-------+------------------+
//!                     1 byte
//! ```
//!
//! - Length is a 4-byte big-endian u32 (includes flags byte + payload)
//! - Flags byte: bit 0 = compressed (1 = LZ4)
//! - Payload is bincode-serialized Message (possibly compressed)

use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::time::timeout;

use super::compression::{self, CompressionConfig, CompressionMethod};
use super::error::{TcpError, TcpResult};
use crate::network::Message;

/// Saturating conversion from `Duration` milliseconds to `u64`.
fn timeout_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// Convert a byte length to a 4-byte big-endian length prefix.
fn length_prefix(len: usize, max: usize) -> TcpResult<[u8; 4]> {
    let n = u32::try_from(len).map_err(|_| TcpError::MessageTooLarge {
        size: len,
        max_size: max,
    })?;
    Ok(n.to_be_bytes())
}

/// Length-delimited codec for message framing.
#[derive(Clone)]
pub struct LengthDelimitedCodec {
    max_frame_length: usize,
    compression: CompressionConfig,
    compress_enabled: bool,
}

impl LengthDelimitedCodec {
    #[must_use]
    pub fn new(max_frame_length: usize) -> Self {
        Self {
            max_frame_length,
            compression: CompressionConfig::default(),
            compress_enabled: false,
        }
    }

    #[must_use]
    pub const fn with_compression(max_frame_length: usize, compression: CompressionConfig) -> Self {
        Self {
            max_frame_length,
            compression,
            compress_enabled: false,
        }
    }

    pub fn set_compression_enabled(&mut self, enabled: bool) {
        self.compress_enabled = enabled && self.compression.enabled;
    }

    #[must_use]
    pub const fn compression_enabled(&self) -> bool {
        self.compress_enabled
    }

    #[must_use]
    pub const fn compression_config(&self) -> &CompressionConfig {
        &self.compression
    }

    /// # Errors
    ///
    /// Returns `MessageTooLarge` if the serialized payload exceeds the max frame length.
    pub fn encode(&self, msg: &Message) -> TcpResult<Vec<u8>> {
        let payload = bitcode::serialize(msg)?;

        if payload.len() > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: payload.len(),
                max_size: self.max_frame_length,
            });
        }

        let header = length_prefix(payload.len(), self.max_frame_length)?;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&header);
        frame.extend_from_slice(&payload);

        Ok(frame)
    }

    /// # Errors
    ///
    /// Returns `MessageTooLarge` if the payload exceeds the max frame length,
    /// or a deserialization error if decoding fails.
    pub fn decode_payload(&self, payload: &[u8]) -> TcpResult<Message> {
        if payload.len() > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: payload.len(),
                max_size: self.max_frame_length,
            });
        }

        let msg: Message = bitcode::deserialize(payload)?;
        Ok(msg)
    }

    /// # Errors
    ///
    /// Returns `MessageTooLarge` if the frame (flags + payload) exceeds the max frame length.
    pub fn encode_v2(&self, msg: &Message) -> TcpResult<Vec<u8>> {
        let serialized = bitcode::serialize(msg)?;

        let (payload, flags) =
            if self.compress_enabled && serialized.len() >= self.compression.min_size {
                let compressed = compression::compress(&serialized, self.compression.method);
                if compression::is_beneficial(serialized.len(), compressed.len()) {
                    (
                        compressed,
                        compression::frame_flags(self.compression.method),
                    )
                } else {
                    (serialized, compression::flags::NONE)
                }
            } else {
                (serialized, compression::flags::NONE)
            };

        let frame_content_len = 1 + payload.len();
        if frame_content_len > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: frame_content_len,
                max_size: self.max_frame_length,
            });
        }

        let header = length_prefix(frame_content_len, self.max_frame_length)?;
        let mut frame = Vec::with_capacity(4 + frame_content_len);
        frame.extend_from_slice(&header);
        frame.push(flags);
        frame.extend_from_slice(&payload);

        Ok(frame)
    }

    /// # Errors
    ///
    /// Returns `InvalidFrame` for empty payloads, `MessageTooLarge` if the
    /// decompressed data exceeds the max frame length.
    pub fn decode_payload_v2(&self, payload: &[u8]) -> TcpResult<Message> {
        if payload.is_empty() {
            return Err(TcpError::InvalidFrame("empty v2 payload".to_string()));
        }

        let flags = payload[0];
        let data = &payload[1..];

        let method = compression::method_from_flags(flags)?;
        let decompressed = if method == CompressionMethod::None {
            data.to_vec()
        } else {
            compression::decompress(data, method)?
        };

        if decompressed.len() > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: decompressed.len(),
                max_size: self.max_frame_length,
            });
        }

        let msg: Message = bitcode::deserialize(&decompressed)?;
        Ok(msg)
    }

    /// Read a v1 frame. Returns `None` on graceful connection close.
    ///
    /// # Errors
    ///
    /// Returns `MessageTooLarge`, `InvalidFrame`, or an I/O error.
    pub async fn read_frame<R>(&self, reader: &mut R) -> TcpResult<Option<Message>>
    where
        R: AsyncRead + Unpin,
    {
        let mut length_buf = [0u8; 4];
        match reader.read_exact(&mut length_buf).await {
            Ok(_) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            },
            Err(e) => return Err(e.into()),
        }

        let length = u32::from_be_bytes(length_buf) as usize;

        if length > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: length,
                max_size: self.max_frame_length,
            });
        }

        if length == 0 {
            return Err(TcpError::InvalidFrame("zero-length frame".to_string()));
        }

        let mut payload = vec![0u8; length];
        reader.read_exact(&mut payload).await?;

        let msg = self.decode_payload(&payload)?;
        Ok(Some(msg))
    }

    /// # Errors
    ///
    /// Returns a serialization or I/O error.
    pub async fn write_frame<W>(&self, writer: &mut W, msg: &Message) -> TcpResult<()>
    where
        W: AsyncWrite + Unpin,
    {
        let frame = self.encode(msg)?;
        writer.write_all(&frame).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Read a v1 frame with timeout. Returns `None` on graceful connection close.
    ///
    /// # Errors
    ///
    /// Returns `Timeout`, `MessageTooLarge`, `InvalidFrame`, or an I/O error.
    pub async fn read_frame_with_timeout<R>(
        &self,
        reader: &mut R,
        io_timeout: Duration,
    ) -> TcpResult<Option<Message>>
    where
        R: AsyncRead + Unpin,
    {
        let mut length_buf = [0u8; 4];
        match timeout(io_timeout, reader.read_exact(&mut length_buf)).await {
            Ok(Ok(_)) => {},
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            },
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                return Err(TcpError::Timeout {
                    operation: "read length",
                    timeout_ms: timeout_ms(io_timeout),
                })
            },
        }

        let length = u32::from_be_bytes(length_buf) as usize;

        if length > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: length,
                max_size: self.max_frame_length,
            });
        }

        if length == 0 {
            return Err(TcpError::InvalidFrame("zero-length frame".to_string()));
        }

        let mut payload = vec![0u8; length];
        timeout(io_timeout, reader.read_exact(&mut payload))
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "read payload",
                timeout_ms: timeout_ms(io_timeout),
            })??;

        let msg = self.decode_payload(&payload)?;
        Ok(Some(msg))
    }

    /// Write a v1 frame with timeout.
    ///
    /// # Errors
    ///
    /// Returns `Timeout`, serialization, or I/O error.
    pub async fn write_frame_with_timeout<W>(
        &self,
        writer: &mut W,
        msg: &Message,
        io_timeout: Duration,
    ) -> TcpResult<()>
    where
        W: AsyncWrite + Unpin,
    {
        let frame = self.encode(msg)?;
        timeout(io_timeout, writer.write_all(&frame))
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "write frame",
                timeout_ms: timeout_ms(io_timeout),
            })??;
        timeout(io_timeout, writer.flush())
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "flush",
                timeout_ms: timeout_ms(io_timeout),
            })??;
        Ok(())
    }

    /// Read a v2 frame. Returns `None` on graceful connection close.
    ///
    /// # Errors
    ///
    /// Returns `MessageTooLarge`, `InvalidFrame`, or an I/O error.
    pub async fn read_frame_v2<R>(&self, reader: &mut R) -> TcpResult<Option<Message>>
    where
        R: AsyncRead + Unpin,
    {
        let mut length_buf = [0u8; 4];
        match reader.read_exact(&mut length_buf).await {
            Ok(_) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            },
            Err(e) => return Err(e.into()),
        }

        let length = u32::from_be_bytes(length_buf) as usize;

        if length > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: length,
                max_size: self.max_frame_length,
            });
        }

        if length == 0 {
            return Err(TcpError::InvalidFrame("zero-length v2 frame".to_string()));
        }

        let mut payload = vec![0u8; length];
        reader.read_exact(&mut payload).await?;

        let msg = self.decode_payload_v2(&payload)?;
        Ok(Some(msg))
    }

    /// # Errors
    ///
    /// Returns a serialization or I/O error.
    pub async fn write_frame_v2<W>(&self, writer: &mut W, msg: &Message) -> TcpResult<()>
    where
        W: AsyncWrite + Unpin,
    {
        let frame = self.encode_v2(msg)?;
        writer.write_all(&frame).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Read a v2 frame with timeout. Returns `None` on graceful connection close.
    ///
    /// # Errors
    ///
    /// Returns `Timeout`, `MessageTooLarge`, `InvalidFrame`, or an I/O error.
    pub async fn read_frame_v2_with_timeout<R>(
        &self,
        reader: &mut R,
        io_timeout: Duration,
    ) -> TcpResult<Option<Message>>
    where
        R: AsyncRead + Unpin,
    {
        let mut length_buf = [0u8; 4];
        match timeout(io_timeout, reader.read_exact(&mut length_buf)).await {
            Ok(Ok(_)) => {},
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            },
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                return Err(TcpError::Timeout {
                    operation: "read length",
                    timeout_ms: timeout_ms(io_timeout),
                })
            },
        }

        let length = u32::from_be_bytes(length_buf) as usize;

        if length > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: length,
                max_size: self.max_frame_length,
            });
        }

        if length == 0 {
            return Err(TcpError::InvalidFrame("zero-length v2 frame".to_string()));
        }

        let mut payload = vec![0u8; length];
        timeout(io_timeout, reader.read_exact(&mut payload))
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "read payload",
                timeout_ms: timeout_ms(io_timeout),
            })??;

        let msg = self.decode_payload_v2(&payload)?;
        Ok(Some(msg))
    }

    /// Write a v2 frame with timeout.
    ///
    /// # Errors
    ///
    /// Returns `Timeout`, serialization, or I/O error.
    pub async fn write_frame_v2_with_timeout<W>(
        &self,
        writer: &mut W,
        msg: &Message,
        io_timeout: Duration,
    ) -> TcpResult<()>
    where
        W: AsyncWrite + Unpin,
    {
        let frame = self.encode_v2(msg)?;
        timeout(io_timeout, writer.write_all(&frame))
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "write frame",
                timeout_ms: timeout_ms(io_timeout),
            })??;
        timeout(io_timeout, writer.flush())
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "flush",
                timeout_ms: timeout_ms(io_timeout),
            })??;
        Ok(())
    }

    #[must_use]
    pub const fn max_frame_length(&self) -> usize {
        self.max_frame_length
    }
}

impl Default for LengthDelimitedCodec {
    fn default() -> Self {
        Self::new(16 * 1024 * 1024) // 16 MB default
    }
}

/// Handshake message sent when connecting.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Handshake {
    /// Sender's node ID.
    pub node_id: String,
    /// Protocol version.
    pub protocol_version: u32,
    /// Optional capabilities.
    pub capabilities: Vec<String>,
}

impl Handshake {
    /// Current protocol version.
    ///
    /// - v1: Original length-prefixed bincode
    /// - v2: Added flags byte for compression support
    pub const PROTOCOL_VERSION: u32 = 2;

    /// Minimum supported protocol version for backward compatibility.
    pub const MIN_PROTOCOL_VERSION: u32 = 1;

    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            protocol_version: Self::PROTOCOL_VERSION,
            capabilities: vec![],
        }
    }

    #[must_use]
    pub fn with_capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }

    #[must_use]
    pub fn with_compression(self) -> Self {
        self.with_capability(super::compression::COMPRESSION_CAPABILITY)
    }

    #[must_use]
    pub fn supports_compression(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == super::compression::COMPRESSION_CAPABILITY)
    }

    #[must_use]
    pub const fn is_v2(&self) -> bool {
        self.protocol_version >= 2
    }

    #[must_use]
    pub fn compression_negotiated(&self, peer: &Self) -> bool {
        self.is_v2() && peer.is_v2() && self.supports_compression() && peer.supports_compression()
    }

    /// # Errors
    ///
    /// Returns a serialization error if encoding fails.
    pub fn encode(&self) -> TcpResult<Vec<u8>> {
        let payload = bitcode::serialize(self)?;
        let header = length_prefix(payload.len(), usize::MAX)?;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&header);
        frame.extend_from_slice(&payload);
        Ok(frame)
    }

    /// # Errors
    ///
    /// Returns `HandshakeFailed` if the payload is too large, deserialization
    /// fails, or the protocol version is unsupported.
    pub async fn read_from<R>(reader: &mut R, max_size: usize) -> TcpResult<Self>
    where
        R: AsyncRead + Unpin,
    {
        let mut length_buf = [0u8; 4];
        reader.read_exact(&mut length_buf).await?;
        let length = u32::from_be_bytes(length_buf) as usize;

        if length > max_size {
            return Err(TcpError::HandshakeFailed(format!(
                "handshake too large: {length} bytes"
            )));
        }

        let mut payload = vec![0u8; length];
        reader.read_exact(&mut payload).await?;

        let handshake: Self =
            bitcode::deserialize(&payload).map_err(|e| TcpError::HandshakeFailed(e.to_string()))?;

        if handshake.protocol_version < Self::MIN_PROTOCOL_VERSION
            || handshake.protocol_version > Self::PROTOCOL_VERSION
        {
            return Err(TcpError::HandshakeFailed(format!(
                "unsupported protocol version: {} (supported: {}-{})",
                handshake.protocol_version,
                Self::MIN_PROTOCOL_VERSION,
                Self::PROTOCOL_VERSION
            )));
        }

        Ok(handshake)
    }

    /// # Errors
    ///
    /// Returns a serialization or I/O error.
    pub async fn write_to<W>(&self, writer: &mut W) -> TcpResult<()>
    where
        W: AsyncWrite + Unpin,
    {
        let frame = self.encode()?;
        writer.write_all(&frame).await?;
        writer.flush().await?;
        Ok(())
    }

    /// # Errors
    ///
    /// Returns `Timeout`, `HandshakeFailed`, or an I/O error.
    pub async fn read_from_with_timeout<R>(
        reader: &mut R,
        max_size: usize,
        io_timeout: Duration,
    ) -> TcpResult<Self>
    where
        R: AsyncRead + Unpin,
    {
        let mut length_buf = [0u8; 4];
        timeout(io_timeout, reader.read_exact(&mut length_buf))
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "handshake read length",
                timeout_ms: timeout_ms(io_timeout),
            })??;
        let length = u32::from_be_bytes(length_buf) as usize;

        if length > max_size {
            return Err(TcpError::HandshakeFailed(format!(
                "handshake too large: {length} bytes"
            )));
        }

        let mut payload = vec![0u8; length];
        timeout(io_timeout, reader.read_exact(&mut payload))
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "handshake read payload",
                timeout_ms: timeout_ms(io_timeout),
            })??;

        let handshake: Self =
            bitcode::deserialize(&payload).map_err(|e| TcpError::HandshakeFailed(e.to_string()))?;

        if handshake.protocol_version < Self::MIN_PROTOCOL_VERSION
            || handshake.protocol_version > Self::PROTOCOL_VERSION
        {
            return Err(TcpError::HandshakeFailed(format!(
                "unsupported protocol version: {} (supported: {}-{})",
                handshake.protocol_version,
                Self::MIN_PROTOCOL_VERSION,
                Self::PROTOCOL_VERSION
            )));
        }

        Ok(handshake)
    }

    /// # Errors
    ///
    /// Returns `Timeout`, serialization, or I/O error.
    pub async fn write_to_with_timeout<W>(
        &self,
        writer: &mut W,
        io_timeout: Duration,
    ) -> TcpResult<()>
    where
        W: AsyncWrite + Unpin,
    {
        let frame = self.encode()?;
        timeout(io_timeout, writer.write_all(&frame))
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "handshake write",
                timeout_ms: timeout_ms(io_timeout),
            })??;
        timeout(io_timeout, writer.flush())
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "handshake flush",
                timeout_ms: timeout_ms(io_timeout),
            })??;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor};
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use tensor_store::SparseVector;
    use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

    use super::*;
    use crate::network::{Message, QueryResponse, RequestVote};

    struct ErrorReader;

    impl AsyncRead for ErrorReader {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            Poll::Ready(Err(io::Error::other("boom")))
        }
    }

    struct PendingReader {
        data: Vec<u8>,
        pos: usize,
        stall_after: usize,
    }

    impl PendingReader {
        fn new(data: Vec<u8>, stall_after: usize) -> Self {
            Self {
                data,
                pos: 0,
                stall_after,
            }
        }
    }

    impl AsyncRead for PendingReader {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            let me = self.get_mut();
            if me.pos >= me.stall_after {
                return Poll::Pending;
            }

            let available = &me.data[me.pos..];
            let to_copy = available.len().min(buf.remaining());
            buf.put_slice(&available[..to_copy]);
            me.pos += to_copy;
            Poll::Ready(Ok(()))
        }
    }

    enum PendingMode {
        Write,
        Flush,
    }

    struct PendingWriter {
        mode: PendingMode,
    }

    impl PendingWriter {
        fn new(mode: PendingMode) -> Self {
            Self { mode }
        }
    }

    impl AsyncWrite for PendingWriter {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            match self.mode {
                PendingMode::Write => Poll::Pending,
                PendingMode::Flush => Poll::Ready(Ok(buf.len())),
            }
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            match self.mode {
                PendingMode::Write => Poll::Ready(Ok(())),
                PendingMode::Flush => Poll::Pending,
            }
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[test]
    fn test_encode_decode() {
        let codec = LengthDelimitedCodec::new(1024 * 1024);

        let msg = Message::RequestVote(RequestVote {
            term: 1,
            candidate_id: "node1".to_string(),
            last_log_index: 10,
            last_log_term: 1,
            state_embedding: SparseVector::from_dense(&[0.1, 0.2, 0.3]),
        });

        let encoded = codec.encode(&msg).unwrap();
        assert!(encoded.len() > 4);

        // Extract length
        let length = u32::from_be_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        assert_eq!(length, encoded.len() - 4);

        // Decode payload
        let decoded = codec.decode_payload(&encoded[4..]).unwrap();
        if let Message::RequestVote(rv) = decoded {
            assert_eq!(rv.term, 1);
            assert_eq!(rv.candidate_id, "node1");
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_message_too_large() {
        let codec = LengthDelimitedCodec::new(100);

        let msg = Message::RequestVote(RequestVote {
            term: 1,
            candidate_id: "node1".to_string(),
            last_log_index: 10,
            last_log_term: 1,
            state_embedding: SparseVector::from_dense(&vec![1.0; 100]), /* Large embedding with
                                                                         * non-zero values */
        });

        let result = codec.encode(&msg);
        assert!(matches!(result, Err(TcpError::MessageTooLarge { .. })));
    }

    #[tokio::test]
    async fn test_read_write_frame() {
        let codec = LengthDelimitedCodec::new(1024 * 1024);

        let msg = Message::Ping { term: 42 };

        // Write to buffer
        let mut buffer = Vec::new();
        codec.write_frame(&mut buffer, &msg).await.unwrap();

        // Read from buffer
        let mut cursor = Cursor::new(buffer);
        let decoded = codec.read_frame(&mut cursor).await.unwrap().unwrap();

        if let Message::Ping { term } = decoded {
            assert_eq!(term, 42);
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_handshake_encode_decode() {
        let handshake = Handshake::new("node1")
            .with_capability("tls")
            .with_capability("compression");

        let encoded = handshake.encode().unwrap();
        assert!(encoded.len() > 4);

        // Decode
        let length = u32::from_be_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        let decoded: Handshake = bitcode::deserialize(&encoded[4..4 + length]).unwrap();

        assert_eq!(decoded.node_id, "node1");
        assert_eq!(decoded.protocol_version, Handshake::PROTOCOL_VERSION);
        assert_eq!(decoded.capabilities.len(), 2);
    }

    #[tokio::test]
    async fn test_handshake_read_write() {
        let handshake = Handshake::new("test_node");

        // Write to buffer
        let mut buffer = Vec::new();
        handshake.write_to(&mut buffer).await.unwrap();

        // Read from buffer
        let mut cursor = Cursor::new(buffer);
        let decoded = Handshake::read_from(&mut cursor, 1024).await.unwrap();

        assert_eq!(decoded.node_id, "test_node");
    }

    #[test]
    fn test_codec_default() {
        let codec = LengthDelimitedCodec::default();
        assert_eq!(codec.max_frame_length(), 16 * 1024 * 1024);
    }

    #[test]
    fn test_max_frame_length() {
        let codec = LengthDelimitedCodec::new(4096);
        assert_eq!(codec.max_frame_length(), 4096);
    }

    #[test]
    fn test_decode_payload_too_large() {
        let codec = LengthDelimitedCodec::new(10);
        let large_payload = vec![0u8; 100];

        let result = codec.decode_payload(&large_payload);
        assert!(matches!(
            result,
            Err(TcpError::MessageTooLarge {
                size: 100,
                max_size: 10
            })
        ));
    }

    #[test]
    fn test_encode_v2_uncompressed_when_not_beneficial() {
        let config = CompressionConfig::default()
            .with_method(CompressionMethod::None)
            .with_min_size(1);
        let mut codec = LengthDelimitedCodec::with_compression(1024, config);
        codec.set_compression_enabled(true);

        let msg = Message::Ping { term: 1 };
        let frame = codec.encode_v2(&msg).unwrap();
        assert_eq!(frame[4], compression::flags::NONE);
    }

    #[test]
    fn test_encode_v2_message_too_large() {
        let codec = LengthDelimitedCodec::new(8);
        let msg = Message::QueryResponse(QueryResponse {
            query_id: 1,
            shard_id: 0,
            result: vec![1u8; 32],
            execution_time_us: 0,
            success: true,
            error: None,
        });

        let result = codec.encode_v2(&msg);
        assert!(matches!(result, Err(TcpError::MessageTooLarge { .. })));
    }

    #[test]
    fn test_decode_v2_decompressed_too_large() {
        let codec = LengthDelimitedCodec::new(10);
        let original = vec![42u8; 64];
        let compressed = compression::compress(&original, CompressionMethod::Lz4);

        let mut payload = Vec::with_capacity(1 + compressed.len());
        payload.push(compression::flags::LZ4);
        payload.extend_from_slice(&compressed);

        let result = codec.decode_payload_v2(&payload);
        assert!(matches!(
            result,
            Err(TcpError::MessageTooLarge {
                size: 64,
                max_size: 10
            })
        ));
    }

    #[tokio::test]
    async fn test_read_frame_zero_length() {
        let codec = LengthDelimitedCodec::new(1024);

        // Create frame with zero length
        let zero_length: [u8; 4] = 0u32.to_be_bytes();
        let mut cursor = Cursor::new(zero_length.to_vec());

        let result = codec.read_frame(&mut cursor).await;
        assert!(matches!(result, Err(TcpError::InvalidFrame(_))));
    }

    #[tokio::test]
    async fn test_read_frame_too_large() {
        let codec = LengthDelimitedCodec::new(100);

        // Create frame with length exceeding max
        let large_length: [u8; 4] = 1000u32.to_be_bytes();
        let mut cursor = Cursor::new(large_length.to_vec());

        let result = codec.read_frame(&mut cursor).await;
        assert!(matches!(
            result,
            Err(TcpError::MessageTooLarge {
                size: 1000,
                max_size: 100
            })
        ));
    }

    #[tokio::test]
    async fn test_read_frame_connection_closed() {
        let codec = LengthDelimitedCodec::new(1024);

        // Empty buffer simulates connection closed
        let mut cursor = Cursor::new(Vec::new());

        let result = codec.read_frame(&mut cursor).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_read_frame_error_non_eof() {
        let codec = LengthDelimitedCodec::new(1024);
        let mut reader = ErrorReader;

        let result = codec.read_frame(&mut reader).await;
        assert!(matches!(result, Err(TcpError::Io(_))));
    }

    #[tokio::test]
    async fn test_handshake_too_large() {
        // Create a frame with length exceeding max_size
        let large_length: [u8; 4] = 5000u32.to_be_bytes();
        let mut cursor = Cursor::new(large_length.to_vec());

        let result = Handshake::read_from(&mut cursor, 1024).await;
        assert!(matches!(result, Err(TcpError::HandshakeFailed(_))));
        if let Err(TcpError::HandshakeFailed(msg)) = result {
            assert!(msg.contains("too large"));
        }
    }

    #[tokio::test]
    async fn test_handshake_wrong_protocol_version() {
        // Create handshake with unsupported version
        let handshake = Handshake {
            node_id: "test".to_string(),
            protocol_version: 999, // Unsupported version
            capabilities: vec![],
        };

        let payload = bitcode::serialize(&handshake).unwrap();
        let length = payload.len() as u32;

        let mut frame = Vec::new();
        frame.extend_from_slice(&length.to_be_bytes());
        frame.extend_from_slice(&payload);

        let mut cursor = Cursor::new(frame);
        let result = Handshake::read_from(&mut cursor, 1024).await;
        assert!(matches!(result, Err(TcpError::HandshakeFailed(_))));
        if let Err(TcpError::HandshakeFailed(msg)) = result {
            assert!(msg.contains("unsupported protocol version"));
        }
    }

    #[test]
    fn test_handshake_debug() {
        let handshake = Handshake::new("node1");
        let debug = format!("{:?}", handshake);
        assert!(debug.contains("Handshake"));
        assert!(debug.contains("node1"));
    }

    #[test]
    fn test_handshake_clone() {
        let handshake = Handshake::new("node1").with_capability("tls");
        let cloned = handshake.clone();
        assert_eq!(cloned.node_id, "node1");
        assert_eq!(cloned.capabilities.len(), 1);
    }

    // === Timeout method tests ===

    #[tokio::test]
    async fn test_read_frame_with_timeout_success() {
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let msg = Message::Ping { term: 42 };

        // Write to buffer
        let mut buffer = Vec::new();
        codec.write_frame(&mut buffer, &msg).await.unwrap();

        // Read with timeout (should succeed immediately)
        let mut cursor = Cursor::new(buffer);
        let decoded = codec
            .read_frame_with_timeout(&mut cursor, Duration::from_secs(5))
            .await
            .unwrap()
            .unwrap();

        if let Message::Ping { term } = decoded {
            assert_eq!(term, 42);
        } else {
            panic!("wrong message type");
        }
    }

    #[tokio::test]
    async fn test_read_frame_with_timeout_connection_closed() {
        let codec = LengthDelimitedCodec::new(1024);

        // Empty buffer simulates connection closed
        let mut cursor = Cursor::new(Vec::new());

        let result = codec
            .read_frame_with_timeout(&mut cursor, Duration::from_secs(1))
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_read_frame_with_timeout_error_non_eof() {
        let codec = LengthDelimitedCodec::new(1024);
        let mut reader = ErrorReader;

        let result = codec
            .read_frame_with_timeout(&mut reader, Duration::from_millis(10))
            .await;
        assert!(matches!(result, Err(TcpError::Io(_))));
    }

    #[tokio::test]
    async fn test_read_frame_with_timeout_payload_timeout() {
        let codec = LengthDelimitedCodec::new(1024);
        let length: [u8; 4] = 16u32.to_be_bytes();
        let mut reader = PendingReader::new(length.to_vec(), 4);

        let result = codec
            .read_frame_with_timeout(&mut reader, Duration::from_millis(10))
            .await;

        assert!(matches!(
            result,
            Err(TcpError::Timeout {
                operation: "read payload",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_read_frame_with_timeout_zero_length() {
        let codec = LengthDelimitedCodec::new(1024);

        // Create frame with zero length
        let zero_length: [u8; 4] = 0u32.to_be_bytes();
        let mut cursor = Cursor::new(zero_length.to_vec());

        let result = codec
            .read_frame_with_timeout(&mut cursor, Duration::from_secs(1))
            .await;
        assert!(matches!(result, Err(TcpError::InvalidFrame(_))));
    }

    #[tokio::test]
    async fn test_read_frame_with_timeout_too_large() {
        let codec = LengthDelimitedCodec::new(100);

        // Create frame with length exceeding max
        let large_length: [u8; 4] = 1000u32.to_be_bytes();
        let mut cursor = Cursor::new(large_length.to_vec());

        let result = codec
            .read_frame_with_timeout(&mut cursor, Duration::from_secs(1))
            .await;
        assert!(matches!(
            result,
            Err(TcpError::MessageTooLarge {
                size: 1000,
                max_size: 100
            })
        ));
    }

    #[tokio::test]
    async fn test_write_frame_with_timeout_write_timeout() {
        let codec = LengthDelimitedCodec::new(1024);
        let msg = Message::Ping { term: 7 };
        let mut writer = PendingWriter::new(PendingMode::Write);

        let result = codec
            .write_frame_with_timeout(&mut writer, &msg, Duration::from_millis(10))
            .await;

        assert!(matches!(
            result,
            Err(TcpError::Timeout {
                operation: "write frame",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_write_frame_with_timeout_flush_timeout() {
        let codec = LengthDelimitedCodec::new(1024);
        let msg = Message::Ping { term: 8 };
        let mut writer = PendingWriter::new(PendingMode::Flush);

        let result = codec
            .write_frame_with_timeout(&mut writer, &msg, Duration::from_millis(10))
            .await;

        assert!(matches!(
            result,
            Err(TcpError::Timeout {
                operation: "flush",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_write_frame_with_timeout_success() {
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let msg = Message::Ping { term: 99 };

        // Write with timeout (should succeed immediately)
        let mut buffer = Vec::new();
        codec
            .write_frame_with_timeout(&mut buffer, &msg, Duration::from_secs(5))
            .await
            .unwrap();

        // Verify by reading back
        let mut cursor = Cursor::new(buffer);
        let decoded = codec.read_frame(&mut cursor).await.unwrap().unwrap();

        if let Message::Ping { term } = decoded {
            assert_eq!(term, 99);
        } else {
            panic!("wrong message type");
        }
    }

    #[tokio::test]
    async fn test_handshake_read_with_timeout_success() {
        let handshake = Handshake::new("test_node").with_capability("test");

        // Write to buffer
        let mut buffer = Vec::new();
        handshake.write_to(&mut buffer).await.unwrap();

        // Read with timeout (should succeed immediately)
        let mut cursor = Cursor::new(buffer);
        let decoded = Handshake::read_from_with_timeout(&mut cursor, 1024, Duration::from_secs(5))
            .await
            .unwrap();

        assert_eq!(decoded.node_id, "test_node");
        assert_eq!(decoded.capabilities.len(), 1);
    }

    #[tokio::test]
    async fn test_handshake_read_with_timeout_too_large() {
        // Create a frame with length exceeding max_size
        let large_length: [u8; 4] = 5000u32.to_be_bytes();
        let mut cursor = Cursor::new(large_length.to_vec());

        let result =
            Handshake::read_from_with_timeout(&mut cursor, 1024, Duration::from_secs(1)).await;
        assert!(matches!(result, Err(TcpError::HandshakeFailed(_))));
        if let Err(TcpError::HandshakeFailed(msg)) = result {
            assert!(msg.contains("too large"));
        }
    }

    #[tokio::test]
    async fn test_handshake_read_with_timeout_wrong_version() {
        // Create handshake with unsupported version
        let handshake = Handshake {
            node_id: "test".to_string(),
            protocol_version: 999, // Unsupported version
            capabilities: vec![],
        };

        let payload = bitcode::serialize(&handshake).unwrap();
        let length = payload.len() as u32;

        let mut frame = Vec::new();
        frame.extend_from_slice(&length.to_be_bytes());
        frame.extend_from_slice(&payload);

        let mut cursor = Cursor::new(frame);
        let result =
            Handshake::read_from_with_timeout(&mut cursor, 1024, Duration::from_secs(1)).await;
        assert!(matches!(result, Err(TcpError::HandshakeFailed(_))));
        if let Err(TcpError::HandshakeFailed(msg)) = result {
            assert!(msg.contains("unsupported protocol version"));
        }
    }

    #[tokio::test]
    async fn test_handshake_write_with_timeout_success() {
        let handshake = Handshake::new("timeout_node");

        // Write with timeout (should succeed immediately)
        let mut buffer = Vec::new();
        handshake
            .write_to_with_timeout(&mut buffer, Duration::from_secs(5))
            .await
            .unwrap();

        // Verify by reading back
        let mut cursor = Cursor::new(buffer);
        let decoded = Handshake::read_from(&mut cursor, 1024).await.unwrap();

        assert_eq!(decoded.node_id, "timeout_node");
    }

    #[tokio::test]
    async fn test_read_frame_with_timeout_actual_timeout() {
        use std::pin::Pin;
        use std::task::{Context, Poll};
        use tokio::io::ReadBuf;

        // Create a reader that never returns
        struct NeverReader;

        impl AsyncRead for NeverReader {
            fn poll_read(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                _buf: &mut ReadBuf<'_>,
            ) -> Poll<std::io::Result<()>> {
                Poll::Pending
            }
        }

        let codec = LengthDelimitedCodec::new(1024);
        let mut reader = NeverReader;

        // Should timeout quickly
        let result = codec
            .read_frame_with_timeout(&mut reader, Duration::from_millis(10))
            .await;

        assert!(matches!(
            result,
            Err(TcpError::Timeout {
                operation: "read length",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_handshake_read_with_timeout_actual_timeout() {
        use std::pin::Pin;
        use std::task::{Context, Poll};
        use tokio::io::ReadBuf;

        // Create a reader that never returns
        struct NeverReader;

        impl AsyncRead for NeverReader {
            fn poll_read(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                _buf: &mut ReadBuf<'_>,
            ) -> Poll<std::io::Result<()>> {
                Poll::Pending
            }
        }

        let mut reader = NeverReader;

        // Should timeout quickly
        let result =
            Handshake::read_from_with_timeout(&mut reader, 1024, Duration::from_millis(10)).await;

        assert!(matches!(
            result,
            Err(TcpError::Timeout {
                operation: "handshake read length",
                ..
            })
        ));
    }

    // === V2 frame format tests ===

    #[tokio::test]
    async fn test_v2_encode_decode_uncompressed() {
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let msg = Message::Ping { term: 42 };

        // Encode with v2 format
        let frame = codec.encode_v2(&msg).unwrap();

        // Verify structure: length (4) + flags (1) + payload
        let length = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        assert_eq!(length, frame.len() - 4); // length includes flags + payload
        assert_eq!(frame[4], 0x00); // flags = uncompressed

        // Decode
        let payload = &frame[4..];
        let decoded = codec.decode_payload_v2(payload).unwrap();

        if let Message::Ping { term } = decoded {
            assert_eq!(term, 42);
        } else {
            panic!("wrong message type");
        }
    }

    #[tokio::test]
    async fn test_v2_encode_decode_compressed() {
        use super::compression::CompressionConfig;
        use crate::network::RequestVote;
        use tensor_store::SparseVector;

        let mut codec =
            LengthDelimitedCodec::with_compression(1024 * 1024, CompressionConfig::default());
        codec.set_compression_enabled(true);

        // Create a message with repeating data that compresses well
        // Use a large sparse vector with repeated values
        let mut dense = vec![0.0f32; 1000];
        for i in (0..1000).step_by(2) {
            dense[i] = 0.5; // Repeated value compresses well
        }
        let embedding = SparseVector::from_dense(&dense);

        let msg = Message::RequestVote(RequestVote {
            term: 1,
            candidate_id: "a".repeat(500), // Highly compressible
            last_log_index: 100,
            last_log_term: 1,
            state_embedding: embedding,
        });

        // Encode with v2 format
        let frame = codec.encode_v2(&msg).unwrap();

        // The flags byte should indicate compression
        let flags = frame[4];
        assert_eq!(flags, 0x01, "expected LZ4 compression flag");

        // Decode
        let payload = &frame[4..];
        let decoded = codec.decode_payload_v2(payload).unwrap();

        if let Message::RequestVote(rv) = decoded {
            assert_eq!(rv.term, 1);
            assert_eq!(rv.candidate_id, "a".repeat(500));
            assert_eq!(rv.last_log_index, 100);
        } else {
            panic!("wrong message type");
        }
    }

    #[tokio::test]
    async fn test_v2_read_write_frame() {
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let msg = Message::Ping { term: 99 };

        // Write v2 frame
        let mut buffer = Vec::new();
        codec.write_frame_v2(&mut buffer, &msg).await.unwrap();

        // Read v2 frame
        let mut cursor = Cursor::new(buffer);
        let decoded = codec.read_frame_v2(&mut cursor).await.unwrap().unwrap();

        if let Message::Ping { term } = decoded {
            assert_eq!(term, 99);
        } else {
            panic!("wrong message type");
        }
    }

    #[tokio::test]
    async fn test_v2_read_frame_connection_closed() {
        let codec = LengthDelimitedCodec::new(1024);
        let mut cursor = Cursor::new(Vec::new());

        let result = codec.read_frame_v2(&mut cursor).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_v2_read_frame_error_non_eof() {
        let codec = LengthDelimitedCodec::new(1024);
        let mut reader = ErrorReader;

        let result = codec.read_frame_v2(&mut reader).await;
        assert!(matches!(result, Err(TcpError::Io(_))));
    }

    #[tokio::test]
    async fn test_v2_read_frame_zero_length() {
        let codec = LengthDelimitedCodec::new(1024);
        let zero_length: [u8; 4] = 0u32.to_be_bytes();
        let mut cursor = Cursor::new(zero_length.to_vec());

        let result = codec.read_frame_v2(&mut cursor).await;
        assert!(matches!(result, Err(TcpError::InvalidFrame(_))));
    }

    #[tokio::test]
    async fn test_v2_read_frame_too_large() {
        let codec = LengthDelimitedCodec::new(8);
        let large_length: [u8; 4] = 128u32.to_be_bytes();
        let mut cursor = Cursor::new(large_length.to_vec());

        let result = codec.read_frame_v2(&mut cursor).await;
        assert!(matches!(
            result,
            Err(TcpError::MessageTooLarge {
                size: 128,
                max_size: 8
            })
        ));
    }

    #[tokio::test]
    async fn test_v2_read_write_with_timeout() {
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let msg = Message::Ping { term: 77 };

        // Write v2 frame with timeout
        let mut buffer = Vec::new();
        codec
            .write_frame_v2_with_timeout(&mut buffer, &msg, Duration::from_secs(5))
            .await
            .unwrap();

        // Read v2 frame with timeout
        let mut cursor = Cursor::new(buffer);
        let decoded = codec
            .read_frame_v2_with_timeout(&mut cursor, Duration::from_secs(5))
            .await
            .unwrap()
            .unwrap();

        if let Message::Ping { term } = decoded {
            assert_eq!(term, 77);
        } else {
            panic!("wrong message type");
        }
    }

    #[tokio::test]
    async fn test_v2_read_with_timeout_connection_closed() {
        let codec = LengthDelimitedCodec::new(1024);
        let mut cursor = Cursor::new(Vec::new());

        let result = codec
            .read_frame_v2_with_timeout(&mut cursor, Duration::from_millis(10))
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_v2_read_with_timeout_error_non_eof() {
        let codec = LengthDelimitedCodec::new(1024);
        let mut reader = ErrorReader;

        let result = codec
            .read_frame_v2_with_timeout(&mut reader, Duration::from_millis(10))
            .await;
        assert!(matches!(result, Err(TcpError::Io(_))));
    }

    #[tokio::test]
    async fn test_v2_read_with_timeout_actual_timeout() {
        struct NeverReader;

        impl AsyncRead for NeverReader {
            fn poll_read(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                _buf: &mut ReadBuf<'_>,
            ) -> Poll<io::Result<()>> {
                Poll::Pending
            }
        }

        let codec = LengthDelimitedCodec::new(1024);
        let mut reader = NeverReader;

        let result = codec
            .read_frame_v2_with_timeout(&mut reader, Duration::from_millis(10))
            .await;

        assert!(matches!(
            result,
            Err(TcpError::Timeout {
                operation: "read length",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_v2_read_with_timeout_zero_length() {
        let codec = LengthDelimitedCodec::new(1024);
        let zero_length: [u8; 4] = 0u32.to_be_bytes();
        let mut cursor = Cursor::new(zero_length.to_vec());

        let result = codec
            .read_frame_v2_with_timeout(&mut cursor, Duration::from_millis(10))
            .await;

        assert!(matches!(result, Err(TcpError::InvalidFrame(_))));
    }

    #[tokio::test]
    async fn test_v2_read_with_timeout_too_large() {
        let codec = LengthDelimitedCodec::new(8);
        let large_length: [u8; 4] = 128u32.to_be_bytes();
        let mut cursor = Cursor::new(large_length.to_vec());

        let result = codec
            .read_frame_v2_with_timeout(&mut cursor, Duration::from_millis(10))
            .await;

        assert!(matches!(
            result,
            Err(TcpError::MessageTooLarge {
                size: 128,
                max_size: 8
            })
        ));
    }

    #[tokio::test]
    async fn test_v2_read_with_timeout_payload_timeout() {
        let codec = LengthDelimitedCodec::new(1024);
        let length: [u8; 4] = 16u32.to_be_bytes();
        let mut reader = PendingReader::new(length.to_vec(), 4);

        let result = codec
            .read_frame_v2_with_timeout(&mut reader, Duration::from_millis(10))
            .await;

        assert!(matches!(
            result,
            Err(TcpError::Timeout {
                operation: "read payload",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_write_frame_v2_with_timeout_write_timeout() {
        let codec = LengthDelimitedCodec::new(1024);
        let msg = Message::Ping { term: 11 };
        let mut writer = PendingWriter::new(PendingMode::Write);

        let result = codec
            .write_frame_v2_with_timeout(&mut writer, &msg, Duration::from_millis(10))
            .await;

        assert!(matches!(
            result,
            Err(TcpError::Timeout {
                operation: "write frame",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_write_frame_v2_with_timeout_flush_timeout() {
        let codec = LengthDelimitedCodec::new(1024);
        let msg = Message::Ping { term: 12 };
        let mut writer = PendingWriter::new(PendingMode::Flush);

        let result = codec
            .write_frame_v2_with_timeout(&mut writer, &msg, Duration::from_millis(10))
            .await;

        assert!(matches!(
            result,
            Err(TcpError::Timeout {
                operation: "flush",
                ..
            })
        ));
    }

    #[test]
    fn test_v2_decode_empty_payload() {
        let codec = LengthDelimitedCodec::new(1024);
        let result = codec.decode_payload_v2(&[]);
        assert!(matches!(result, Err(TcpError::InvalidFrame(_))));
    }

    // === Handshake compression capability tests ===

    #[test]
    fn test_handshake_with_compression() {
        let handshake = Handshake::new("node1").with_compression();
        assert!(handshake.supports_compression());
        assert!(handshake.is_v2());
    }

    #[test]
    fn test_handshake_compression_negotiated() {
        let h1 = Handshake::new("node1").with_compression();
        let h2 = Handshake::new("node2").with_compression();

        assert!(h1.compression_negotiated(&h2));
        assert!(h2.compression_negotiated(&h1));
    }

    #[test]
    fn test_handshake_compression_not_negotiated_without_capability() {
        let h1 = Handshake::new("node1").with_compression();
        let h2 = Handshake::new("node2"); // No compression capability

        assert!(!h1.compression_negotiated(&h2));
        assert!(!h2.compression_negotiated(&h1));
    }

    #[test]
    fn test_handshake_compression_not_negotiated_v1_peer() {
        let h1 = Handshake::new("node1").with_compression();
        let h2 = Handshake {
            node_id: "node2".to_string(),
            protocol_version: 1, // v1 peer
            capabilities: vec!["compression".to_string()],
        };

        // Even if v1 peer has compression capability, it can't use v2 framing
        assert!(!h2.is_v2());
        assert!(!h1.compression_negotiated(&h2));
    }

    #[test]
    fn test_codec_compression_config() {
        use super::compression::{CompressionConfig, CompressionMethod};

        let config = CompressionConfig::default().with_method(CompressionMethod::Lz4);
        let codec = LengthDelimitedCodec::with_compression(1024, config);

        assert!(!codec.compression_enabled()); // Disabled by default until negotiated
        assert_eq!(codec.compression_config().method, CompressionMethod::Lz4);
    }

    #[test]
    fn test_codec_set_compression_enabled() {
        use super::compression::CompressionConfig;

        let mut codec = LengthDelimitedCodec::with_compression(1024, CompressionConfig::default());
        assert!(!codec.compression_enabled());

        codec.set_compression_enabled(true);
        assert!(codec.compression_enabled());

        codec.set_compression_enabled(false);
        assert!(!codec.compression_enabled());
    }

    #[test]
    fn test_handshake_protocol_version() {
        let handshake = Handshake::new("node1");
        assert_eq!(handshake.protocol_version, Handshake::PROTOCOL_VERSION);
        assert_eq!(handshake.protocol_version, 2);
    }

    #[test]
    fn test_handshake_min_protocol_version() {
        assert_eq!(Handshake::MIN_PROTOCOL_VERSION, 1);
    }

    #[test]
    fn test_codec_encode_message_too_large() {
        // Create a codec with very small max frame size
        let codec = LengthDelimitedCodec::new(10);

        // Create a message that will exceed the max frame size after serialization
        let large_message = Message::Gossip(crate::gossip::GossipMessage::Sync {
            sender: "node1".to_string(),
            states: vec![crate::gossip::GossipNodeState::with_wall_time(
                "node1".to_string(),
                crate::membership::NodeHealth::Healthy,
                100,
                1,
                200,
            )],
            sender_time: 100,
        });

        let result = codec.encode(&large_message);
        assert!(result.is_err());
        if let Err(TcpError::MessageTooLarge { size, max_size }) = result {
            assert!(size > max_size);
        } else {
            panic!("Expected MessageTooLarge error");
        }
    }

    #[test]
    fn test_codec_compression_disabled_encodes_normally() {
        use super::compression::{CompressionConfig, CompressionMethod};

        let config = CompressionConfig::default().with_method(CompressionMethod::Lz4);
        let codec = LengthDelimitedCodec::with_compression(65535, config);

        // Compression should be disabled by default (needs negotiation)
        assert!(!codec.compression_enabled());

        // Encoding should work without compression
        let msg = Message::Gossip(crate::gossip::GossipMessage::Sync {
            sender: "test".to_string(),
            states: vec![],
            sender_time: 0,
        });

        let result = codec.encode(&msg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_handshake_supports_compression_method() {
        let handshake = Handshake::new("node1");
        // Default should not support any compression
        assert!(!handshake.supports_compression());
    }

    #[test]
    fn test_handshake_with_compression_support() {
        let handshake = Handshake::new("node1").with_compression();
        assert!(handshake.supports_compression());
    }
}
