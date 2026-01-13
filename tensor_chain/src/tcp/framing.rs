//! Length-delimited message framing for TCP transport.
//!
//! Wire format:
//! ```text
//! +------------------+------------------+
//! | Length (4B BE)   | Payload (bincode)|
//! +------------------+------------------+
//! ```
//!
//! - Length is a 4-byte big-endian u32
//! - Payload is bincode-serialized Message

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use super::error::{TcpError, TcpResult};
use crate::network::Message;

/// Length-delimited codec for message framing.
pub struct LengthDelimitedCodec {
    max_frame_length: usize,
}

impl LengthDelimitedCodec {
    /// Create a new codec with the given maximum frame length.
    pub fn new(max_frame_length: usize) -> Self {
        Self { max_frame_length }
    }

    /// Encode a message to bytes with length prefix.
    pub fn encode(&self, msg: &Message) -> TcpResult<Vec<u8>> {
        let payload = bincode::serialize(msg)?;

        if payload.len() > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: payload.len(),
                max_size: self.max_frame_length,
            });
        }

        let length = payload.len() as u32;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&length.to_be_bytes());
        frame.extend_from_slice(&payload);

        Ok(frame)
    }

    /// Decode a message from bytes (without length prefix).
    pub fn decode_payload(&self, payload: &[u8]) -> TcpResult<Message> {
        if payload.len() > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: payload.len(),
                max_size: self.max_frame_length,
            });
        }

        let msg: Message = bincode::deserialize(payload)?;
        Ok(msg)
    }

    /// Read a frame from an async reader.
    ///
    /// Returns None if the connection was closed gracefully.
    pub async fn read_frame<R>(&self, reader: &mut R) -> TcpResult<Option<Message>>
    where
        R: AsyncRead + Unpin,
    {
        // Read length prefix
        let mut length_buf = [0u8; 4];
        match reader.read_exact(&mut length_buf).await {
            Ok(_) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None); // Connection closed
            },
            Err(e) => return Err(e.into()),
        }

        let length = u32::from_be_bytes(length_buf) as usize;

        // Validate length
        if length > self.max_frame_length {
            return Err(TcpError::MessageTooLarge {
                size: length,
                max_size: self.max_frame_length,
            });
        }

        if length == 0 {
            return Err(TcpError::InvalidFrame("zero-length frame".to_string()));
        }

        // Read payload
        let mut payload = vec![0u8; length];
        reader.read_exact(&mut payload).await?;

        // Decode message
        let msg = self.decode_payload(&payload)?;
        Ok(Some(msg))
    }

    /// Write a frame to an async writer.
    pub async fn write_frame<W>(&self, writer: &mut W, msg: &Message) -> TcpResult<()>
    where
        W: AsyncWrite + Unpin,
    {
        let frame = self.encode(msg)?;
        writer.write_all(&frame).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Get the maximum frame length.
    pub fn max_frame_length(&self) -> usize {
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
    pub const PROTOCOL_VERSION: u32 = 1;

    /// Create a new handshake message.
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            protocol_version: Self::PROTOCOL_VERSION,
            capabilities: vec![],
        }
    }

    /// Add a capability.
    pub fn with_capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }

    /// Encode handshake to bytes.
    pub fn encode(&self) -> TcpResult<Vec<u8>> {
        let payload = bincode::serialize(self)?;
        let length = payload.len() as u32;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&length.to_be_bytes());
        frame.extend_from_slice(&payload);
        Ok(frame)
    }

    /// Read handshake from an async reader.
    pub async fn read_from<R>(reader: &mut R, max_size: usize) -> TcpResult<Self>
    where
        R: AsyncRead + Unpin,
    {
        // Read length prefix
        let mut length_buf = [0u8; 4];
        reader.read_exact(&mut length_buf).await?;
        let length = u32::from_be_bytes(length_buf) as usize;

        if length > max_size {
            return Err(TcpError::HandshakeFailed(format!(
                "handshake too large: {} bytes",
                length
            )));
        }

        // Read payload
        let mut payload = vec![0u8; length];
        reader.read_exact(&mut payload).await?;

        // Decode
        let handshake: Handshake =
            bincode::deserialize(&payload).map_err(|e| TcpError::HandshakeFailed(e.to_string()))?;

        // Validate protocol version
        if handshake.protocol_version != Self::PROTOCOL_VERSION {
            return Err(TcpError::HandshakeFailed(format!(
                "protocol version mismatch: expected {}, got {}",
                Self::PROTOCOL_VERSION,
                handshake.protocol_version
            )));
        }

        Ok(handshake)
    }

    /// Write handshake to an async writer.
    pub async fn write_to<W>(&self, writer: &mut W) -> TcpResult<()>
    where
        W: AsyncWrite + Unpin,
    {
        let frame = self.encode()?;
        writer.write_all(&frame).await?;
        writer.flush().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use tensor_store::SparseVector;

    use super::*;
    use crate::network::{Message, RequestVote};

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
        let decoded: Handshake = bincode::deserialize(&encoded[4..4 + length]).unwrap();

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
        // Create handshake with wrong version
        let handshake = Handshake {
            node_id: "test".to_string(),
            protocol_version: 999, // Wrong version
            capabilities: vec![],
        };

        let payload = bincode::serialize(&handshake).unwrap();
        let length = payload.len() as u32;

        let mut frame = Vec::new();
        frame.extend_from_slice(&length.to_be_bytes());
        frame.extend_from_slice(&payload);

        let mut cursor = Cursor::new(frame);
        let result = Handshake::read_from(&mut cursor, 1024).await;
        assert!(matches!(result, Err(TcpError::HandshakeFailed(_))));
        if let Err(TcpError::HandshakeFailed(msg)) = result {
            assert!(msg.contains("version mismatch"));
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
}
