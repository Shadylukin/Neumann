// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! TCP network proxy for fault injection in multi-process Jepsen tests.
//!
//! Each proxy sits between two cluster nodes, forwarding TCP traffic
//! bidirectionally. Fault injection is controlled via atomic flags:
//! partition (drop all bytes), delay (add latency), or heal (restore).
//!
//! No sudo required — the proxy operates entirely in userspace.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use tokio::task::JoinHandle;

/// Control flags for a single proxy link.
pub struct ProxyControl {
    /// When true, drop all traffic (simulates network partition).
    drop_all: AtomicBool,
    /// Added latency in milliseconds per forwarded chunk.
    delay_ms: AtomicU64,
    /// When false, the proxy shuts down.
    running: AtomicBool,
}

impl ProxyControl {
    const fn new() -> Self {
        Self {
            drop_all: AtomicBool::new(false),
            delay_ms: AtomicU64::new(0),
            running: AtomicBool::new(true),
        }
    }
}

/// A TCP proxy that forwards traffic between two addresses with fault injection.
pub struct NetworkProxy {
    listen_addr: SocketAddr,
    target_addr: SocketAddr,
    control: Arc<ProxyControl>,
    shutdown_tx: broadcast::Sender<()>,
    handle: Option<JoinHandle<()>>,
}

impl NetworkProxy {
    /// Create and start a new proxy forwarding from `listen_addr` to `target_addr`.
    ///
    /// # Errors
    ///
    /// Returns an error if the proxy cannot bind to `listen_addr`.
    pub async fn start(listen_addr: SocketAddr, target_addr: SocketAddr) -> std::io::Result<Self> {
        let control = Arc::new(ProxyControl::new());
        let (shutdown_tx, _) = broadcast::channel(1);

        let listener = TcpListener::bind(listen_addr).await?;
        let actual_addr = listener.local_addr()?;

        let ctrl = control.clone();
        let mut shutdown_rx = shutdown_tx.subscribe();
        let tgt = target_addr;

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok((inbound, _)) => {
                                if !ctrl.running.load(Ordering::Relaxed) {
                                    break;
                                }
                                let ctrl_clone = ctrl.clone();
                                tokio::spawn(Self::handle_connection(inbound, tgt, ctrl_clone));
                            }
                            Err(_) => break,
                        }
                    }
                    _ = shutdown_rx.recv() => break,
                }
            }
        });

        Ok(Self {
            listen_addr: actual_addr,
            target_addr,
            control,
            shutdown_tx,
            handle: Some(handle),
        })
    }

    async fn handle_connection(
        mut inbound: TcpStream,
        target: SocketAddr,
        control: Arc<ProxyControl>,
    ) {
        let Ok(mut outbound) = TcpStream::connect(target).await else {
            return;
        };

        let (mut in_read, mut in_write) = inbound.split();
        let (mut out_read, mut out_write) = outbound.split();

        let ctrl_a = control.clone();
        let ctrl_b = control;

        let client_to_server = async move {
            let mut buf = [0u8; 8192];
            loop {
                if !ctrl_a.running.load(Ordering::Relaxed) {
                    break;
                }
                match in_read.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if ctrl_a.drop_all.load(Ordering::Relaxed) {
                            continue;
                        }
                        let delay = ctrl_a.delay_ms.load(Ordering::Relaxed);
                        if delay > 0 {
                            tokio::time::sleep(Duration::from_millis(delay)).await;
                        }
                        if out_write.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                    },
                }
            }
        };

        let server_to_client = async move {
            let mut buf = [0u8; 8192];
            loop {
                if !ctrl_b.running.load(Ordering::Relaxed) {
                    break;
                }
                match out_read.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if ctrl_b.drop_all.load(Ordering::Relaxed) {
                            continue;
                        }
                        let delay = ctrl_b.delay_ms.load(Ordering::Relaxed);
                        if delay > 0 {
                            tokio::time::sleep(Duration::from_millis(delay)).await;
                        }
                        if in_write.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                    },
                }
            }
        };

        tokio::select! {
            () = client_to_server => {}
            () = server_to_client => {}
        }
    }

    /// Enable partition: drop all traffic through this proxy.
    pub fn partition(&self) {
        self.control.drop_all.store(true, Ordering::Release);
    }

    /// Disable partition: allow traffic to flow again.
    pub fn heal(&self) {
        self.control.drop_all.store(false, Ordering::Release);
    }

    /// Add latency to each forwarded chunk.
    pub fn slow(&self, ms: u64) {
        self.control.delay_ms.store(ms, Ordering::Release);
    }

    /// Check whether this proxy is currently partitioned.
    #[must_use]
    pub fn is_partitioned(&self) -> bool {
        self.control.drop_all.load(Ordering::Acquire)
    }

    /// The address this proxy is listening on.
    #[must_use]
    pub const fn listen_addr(&self) -> SocketAddr {
        self.listen_addr
    }

    /// The target address this proxy forwards to.
    #[must_use]
    pub const fn target_addr(&self) -> SocketAddr {
        self.target_addr
    }

    /// Shut down the proxy, stopping all traffic forwarding.
    pub async fn shutdown(&mut self) {
        self.control.running.store(false, Ordering::Release);
        let _ = self.shutdown_tx.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for NetworkProxy {
    fn drop(&mut self) {
        self.control.running.store(false, Ordering::Release);
        let _ = self.shutdown_tx.send(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proxy_forwards_data() {
        let echo_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let echo_addr = echo_listener.local_addr().unwrap();

        // Simple echo server
        tokio::spawn(async move {
            if let Ok((mut stream, _)) = echo_listener.accept().await {
                let mut buf = [0u8; 1024];
                if let Ok(n) = stream.read(&mut buf).await {
                    let _ = stream.write_all(&buf[..n]).await;
                }
            }
        });

        let mut proxy = NetworkProxy::start("127.0.0.1:0".parse().unwrap(), echo_addr)
            .await
            .unwrap();

        let mut client = TcpStream::connect(proxy.listen_addr()).await.unwrap();
        client.write_all(b"hello").await.unwrap();

        let mut buf = [0u8; 1024];
        let n = client.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"hello");

        proxy.shutdown().await;
    }

    #[tokio::test]
    async fn test_proxy_partition_blocks_traffic() {
        let echo_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let echo_addr = echo_listener.local_addr().unwrap();

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = echo_listener.accept().await {
                    tokio::spawn(async move {
                        let mut buf = [0u8; 1024];
                        while let Ok(n) = stream.read(&mut buf).await {
                            if n == 0 {
                                break;
                            }
                            let _ = stream.write_all(&buf[..n]).await;
                        }
                    });
                }
            }
        });

        let mut proxy = NetworkProxy::start("127.0.0.1:0".parse().unwrap(), echo_addr)
            .await
            .unwrap();

        // Partition: data should be dropped
        proxy.partition();
        assert!(proxy.is_partitioned());

        let mut client = TcpStream::connect(proxy.listen_addr()).await.unwrap();
        client.write_all(b"hello").await.unwrap();

        // With partition active, read should time out
        let mut buf = [0u8; 1024];
        let result = tokio::time::timeout(Duration::from_millis(200), client.read(&mut buf)).await;
        assert!(
            result.is_err(),
            "read should have timed out during partition"
        );

        proxy.shutdown().await;
    }

    #[tokio::test]
    async fn test_proxy_heal_restores_traffic() {
        let echo_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let echo_addr = echo_listener.local_addr().unwrap();

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = echo_listener.accept().await {
                    tokio::spawn(async move {
                        let mut buf = [0u8; 1024];
                        while let Ok(n) = stream.read(&mut buf).await {
                            if n == 0 {
                                break;
                            }
                            let _ = stream.write_all(&buf[..n]).await;
                        }
                    });
                }
            }
        });

        let mut proxy = NetworkProxy::start("127.0.0.1:0".parse().unwrap(), echo_addr)
            .await
            .unwrap();

        // Partition then heal
        proxy.partition();
        proxy.heal();
        assert!(!proxy.is_partitioned());

        // After healing, new connections should work
        let mut client = TcpStream::connect(proxy.listen_addr()).await.unwrap();
        client.write_all(b"world").await.unwrap();

        let mut buf = [0u8; 1024];
        let n = tokio::time::timeout(Duration::from_secs(2), client.read(&mut buf))
            .await
            .expect("read should not timeout after heal")
            .unwrap();
        assert_eq!(&buf[..n], b"world");

        proxy.shutdown().await;
    }
}
