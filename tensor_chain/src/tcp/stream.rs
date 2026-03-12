// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stream abstraction for TCP and TLS connections.
//!
//! Provides type aliases for dynamic dispatch over different stream types,
//! allowing the transport layer to work with both plain TCP and TLS connections.

use tokio::io::{AsyncRead, AsyncWrite};

/// Combined async read/write trait for bidirectional streams.
pub trait AsyncStream: AsyncRead + AsyncWrite + Unpin + Send + Sync {}

/// Blanket implementation for any type that satisfies the bounds.
impl<T: AsyncRead + AsyncWrite + Unpin + Send + Sync> AsyncStream for T {}

/// Boxed bidirectional stream for dynamic dispatch.
pub type DynStream = Box<dyn AsyncStream>;

/// Boxed async reader for dynamic dispatch.
pub type DynRead = Box<dyn AsyncRead + Unpin + Send + Sync>;

/// Boxed async writer for dynamic dispatch.
pub type DynWrite = Box<dyn AsyncWrite + Unpin + Send + Sync>;

/// Helper to split any async stream into boxed read/write halves.
pub fn split_stream<S>(stream: S) -> (DynRead, DynWrite)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + Sync + 'static,
{
    let (read, write) = tokio::io::split(stream);
    (Box::new(read), Box::new(write))
}

/// Helper to box a stream into a `DynStream`.
pub fn box_stream<S>(stream: S) -> DynStream
where
    S: AsyncStream + 'static,
{
    Box::new(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyn_types_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DynRead>();
        assert_send::<DynWrite>();
    }

    #[tokio::test]
    async fn test_split_stream() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Create a duplex stream for testing
        let (client, mut server) = tokio::io::duplex(1024);

        let (mut read, mut write) = split_stream(client);

        // Write from server
        server.write_all(b"hello").await.unwrap();
        server.flush().await.unwrap();

        // Read on client
        let mut buf = [0u8; 5];
        read.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"hello");

        // Write from client
        write.write_all(b"world").await.unwrap();
        write.flush().await.unwrap();

        // Read on server
        let mut buf = [0u8; 5];
        server.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"world");
    }
}
