// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Process management for multi-process Jepsen tests.
//!
//! `NodeProcess` wraps a single `neumann_server` process. `ProcessCluster`
//! manages a set of nodes plus TCP proxies for fault injection.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use crate::network_proxy::NetworkProxy;

/// A single server process in the Jepsen cluster.
pub struct NodeProcess {
    child: Option<Child>,
    node_id: String,
    grpc_port: u16,
    #[allow(dead_code)]
    raft_port: u16,
    #[allow(dead_code)]
    data_dir: PathBuf,
    log_path: PathBuf,
    env_vars: HashMap<String, String>,
    binary: PathBuf,
}

impl NodeProcess {
    /// Spawn a new server process.
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot be started.
    pub fn spawn(
        binary: &Path,
        node_id: &str,
        grpc_port: u16,
        raft_port: u16,
        peers_str: &str,
        data_dir: &Path,
    ) -> std::io::Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let log_path = data_dir.join(format!("{node_id}.log"));

        let mut env_vars = HashMap::new();
        env_vars.insert(
            "NEUMANN_BIND_ADDR".to_string(),
            format!("127.0.0.1:{grpc_port}"),
        );
        env_vars.insert("NEUMANN_CLUSTER_NODE_ID".to_string(), node_id.to_string());
        env_vars.insert(
            "NEUMANN_CLUSTER_BIND_ADDR".to_string(),
            format!("127.0.0.1:{raft_port}"),
        );
        env_vars.insert("NEUMANN_CLUSTER_PEERS".to_string(), peers_str.to_string());
        env_vars.insert(
            "NEUMANN_DATA_DIR".to_string(),
            data_dir.to_string_lossy().to_string(),
        );
        // Disable features not needed for cluster tests
        env_vars.insert("NEUMANN_ENABLE_GRPC_WEB".to_string(), "false".to_string());
        env_vars.insert("NEUMANN_ENABLE_REFLECTION".to_string(), "false".to_string());
        env_vars.insert("RUST_LOG".to_string(), "warn".to_string());

        let log_file = std::fs::File::create(&log_path)?;
        let stderr_file = log_file.try_clone()?;

        let child = Command::new(binary)
            .envs(&env_vars)
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(stderr_file))
            .spawn()?;

        Ok(Self {
            child: Some(child),
            node_id: node_id.to_string(),
            grpc_port,
            raft_port,
            data_dir: data_dir.to_path_buf(),
            log_path,
            env_vars,
            binary: binary.to_path_buf(),
        })
    }

    /// Kill the process with SIGKILL (immediate, no cleanup).
    pub fn kill(&mut self) {
        if let Some(ref child) = self.child {
            #[allow(clippy::cast_possible_wrap)]
            let pid = child.id() as i32;
            // SIGKILL = 9: immediate termination, no cleanup
            #[allow(unsafe_code)]
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
            // Reap the zombie
            if let Some(ref mut child) = self.child {
                let _ = child.wait();
            }
            self.child = None;
        }
    }

    /// Pause the process with SIGSTOP (freeze).
    pub fn pause(&self) {
        if let Some(ref child) = self.child {
            #[allow(clippy::cast_possible_wrap)]
            let pid = child.id() as i32;
            #[allow(unsafe_code)]
            unsafe {
                libc::kill(pid, libc::SIGSTOP);
            }
        }
    }

    /// Resume the process with SIGCONT (unfreeze).
    pub fn resume(&self) {
        if let Some(ref child) = self.child {
            #[allow(clippy::cast_possible_wrap)]
            let pid = child.id() as i32;
            #[allow(unsafe_code)]
            unsafe {
                libc::kill(pid, libc::SIGCONT);
            }
        }
    }

    /// Restart the process with the same configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot be restarted.
    pub fn restart(&mut self) -> std::io::Result<()> {
        self.kill();
        std::thread::sleep(Duration::from_millis(100));

        let log_file = std::fs::File::create(&self.log_path)?;
        let stderr_file = log_file.try_clone()?;

        let child = Command::new(&self.binary)
            .envs(&self.env_vars)
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(stderr_file))
            .spawn()?;

        self.child = Some(child);
        Ok(())
    }

    /// Check if the process is still alive.
    #[must_use]
    pub fn is_alive(&mut self) -> bool {
        self.child
            .as_mut()
            .is_some_and(|child| child.try_wait().is_ok_and(|s| s.is_none()))
    }

    /// Wait for the server to become healthy by polling the gRPC endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the server does not become healthy within the timeout.
    pub async fn wait_healthy(&mut self, timeout: Duration) -> std::io::Result<()> {
        let port = self.grpc_port;
        let addr = format!("http://127.0.0.1:{port}");
        let deadline = tokio::time::Instant::now() + timeout;
        let mut interval = tokio::time::interval(Duration::from_millis(200));

        while tokio::time::Instant::now() < deadline {
            interval.tick().await;

            // Try connecting with the client
            let connect_result = tokio::time::timeout(
                Duration::from_millis(500),
                neumann_client::NeumannClient::connect(&addr).build(),
            )
            .await;

            if let Ok(Ok(client)) = connect_result {
                let query_result =
                    tokio::time::timeout(Duration::from_millis(500), client.execute("SELECT 1"))
                        .await;
                if query_result.is_ok() {
                    return Ok(());
                }
            }
        }

        let node_id = &self.node_id;
        Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!("node {node_id} did not become healthy within {timeout:?}"),
        ))
    }

    /// Get the node ID.
    #[must_use]
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get the gRPC port.
    #[must_use]
    pub const fn grpc_port(&self) -> u16 {
        self.grpc_port
    }

    /// Get the gRPC address.
    #[must_use]
    pub fn grpc_addr(&self) -> String {
        let port = self.grpc_port;
        format!("http://127.0.0.1:{port}")
    }

    /// Get the log file path for debugging.
    #[must_use]
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }
}

impl Drop for NodeProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Port assignment for a node in the cluster.
#[derive(Debug, Clone)]
pub struct NodePorts {
    pub grpc_port: u16,
    pub raft_port: u16,
}

/// A multi-node cluster with TCP proxies for fault injection.
pub struct ProcessCluster {
    nodes: Vec<NodeProcess>,
    proxies: Vec<NetworkProxy>,
    base_dir: PathBuf,
    server_binary: PathBuf,
    node_ports: Vec<NodePorts>,
    /// Proxy index layout: for N nodes, proxy[i*N + j] connects node i -> node j.
    /// `proxy_indices[i][j]` gives the proxy index (if i != j).
    proxy_indices: Vec<Vec<Option<usize>>>,
}

impl ProcessCluster {
    /// Compute port assignments and create the cluster (does not start it).
    ///
    /// # Arguments
    /// * `node_count` - Number of nodes in the cluster (typically 3 or 5)
    /// * `base_port` - Base port for assignments (gRPC = base + i, Raft = base + 100 + i)
    /// * `server_binary` - Path to the `neumann_server` binary
    /// * `base_dir` - Directory for node data and logs
    #[must_use]
    pub fn new(
        node_count: usize,
        base_port: u16,
        server_binary: PathBuf,
        base_dir: PathBuf,
    ) -> Self {
        let node_ports: Vec<NodePorts> = (0..node_count)
            .map(|i| {
                #[allow(clippy::cast_possible_truncation)]
                let offset = i as u16;
                NodePorts {
                    grpc_port: base_port + offset,
                    raft_port: base_port + 100 + offset,
                }
            })
            .collect();

        let proxy_indices = vec![vec![None; node_count]; node_count];

        Self {
            nodes: Vec::new(),
            proxies: Vec::new(),
            base_dir,
            server_binary,
            node_ports,
            proxy_indices,
        }
    }

    /// Start all proxies and nodes, waiting for health.
    ///
    /// Each node's peer list points to proxy addresses (not direct addresses),
    /// enabling fault injection via the proxies.
    ///
    /// # Errors
    ///
    /// Returns an error if any proxy or node fails to start.
    #[allow(clippy::needless_range_loop)] // 2D proxy matrix uses both i,j as indices
    pub async fn start(&mut self) -> std::io::Result<()> {
        let n = self.node_ports.len();

        // Start proxies: one per directional link (i -> j)
        // Proxy for "node j's view of node i" listens on a dynamic port
        // and forwards to node i's real Raft address.
        let mut proxy_listen_addrs: Vec<Vec<Option<SocketAddr>>> = vec![vec![None; n]; n];

        let mut proxy_idx = 0;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let raft_port = self.node_ports[i].raft_port;
                let target_addr: SocketAddr = format!("127.0.0.1:{raft_port}").parse().unwrap();
                let listen_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

                let proxy = NetworkProxy::start(listen_addr, target_addr).await?;
                let actual_listen = proxy.listen_addr();
                proxy_listen_addrs[j][i] = Some(actual_listen);
                self.proxy_indices[j][i] = Some(proxy_idx);
                self.proxies.push(proxy);
                proxy_idx += 1;
            }
        }

        // Start nodes with proxy-based peer addresses
        for i in 0..n {
            let node_id = format!("node-{i}");
            let data_dir = self.base_dir.join(&node_id);

            let mut peer_parts = Vec::new();
            for j in 0..n {
                if i == j {
                    continue;
                }
                if let Some(proxy_addr) = proxy_listen_addrs[i][j] {
                    peer_parts.push(format!("node-{j}={proxy_addr}"));
                }
            }
            let peers_str = peer_parts.join(",");

            let node = NodeProcess::spawn(
                &self.server_binary,
                &node_id,
                self.node_ports[i].grpc_port,
                self.node_ports[i].raft_port,
                &peers_str,
                &data_dir,
            )?;

            self.nodes.push(node);
        }

        // Wait for all nodes to become healthy
        for node in &mut self.nodes {
            node.wait_healthy(Duration::from_secs(30)).await?;
        }

        Ok(())
    }

    /// Kill a specific node.
    pub fn kill_node(&mut self, idx: usize) {
        if idx < self.nodes.len() {
            self.nodes[idx].kill();
        }
    }

    /// Restart a specific node.
    ///
    /// # Errors
    ///
    /// Returns an error if the node cannot be restarted.
    pub fn restart_node(&mut self, idx: usize) -> std::io::Result<()> {
        if idx < self.nodes.len() {
            self.nodes[idx].restart()
        } else {
            Ok(())
        }
    }

    /// Wait for a restarted node to become healthy.
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not become healthy.
    pub async fn wait_node_healthy(&mut self, idx: usize) -> std::io::Result<()> {
        if idx < self.nodes.len() {
            self.nodes[idx].wait_healthy(Duration::from_secs(30)).await
        } else {
            Ok(())
        }
    }

    /// Pause a specific node (SIGSTOP).
    pub fn pause_node(&mut self, idx: usize) {
        if idx < self.nodes.len() {
            self.nodes[idx].pause();
        }
    }

    /// Resume a specific node (SIGCONT).
    pub fn resume_node(&mut self, idx: usize) {
        if idx < self.nodes.len() {
            self.nodes[idx].resume();
        }
    }

    /// Look up the proxy index for the directional link from node `from` to node `to`.
    fn proxy_for(&self, from: usize, to: usize) -> Option<&NetworkProxy> {
        let idx = *self.proxy_indices.get(from)?.get(to)?;
        idx.and_then(|i| self.proxies.get(i))
    }

    /// Partition two nodes (both directions).
    pub fn partition(&self, a: usize, b: usize) {
        if let Some(proxy) = self.proxy_for(a, b) {
            proxy.partition();
        }
        if let Some(proxy) = self.proxy_for(b, a) {
            proxy.partition();
        }
    }

    /// Heal partition between two nodes (both directions).
    pub fn heal(&self, a: usize, b: usize) {
        if let Some(proxy) = self.proxy_for(a, b) {
            proxy.heal();
        }
        if let Some(proxy) = self.proxy_for(b, a) {
            proxy.heal();
        }
    }

    /// Heal all partitions.
    pub fn heal_all(&self) {
        for proxy in &self.proxies {
            proxy.heal();
        }
    }

    /// Get the gRPC address for a specific node.
    #[must_use]
    pub fn grpc_addr(&self, idx: usize) -> String {
        let port = self.node_ports[idx].grpc_port;
        format!("http://127.0.0.1:{port}")
    }

    /// Get the number of nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_ports.len()
    }

    /// Check if a node is alive.
    pub fn is_node_alive(&mut self, idx: usize) -> bool {
        if idx < self.nodes.len() {
            self.nodes[idx].is_alive()
        } else {
            false
        }
    }

    /// Get log path for a node (useful for debugging failures).
    #[must_use]
    pub fn node_log_path(&self, idx: usize) -> Option<&Path> {
        self.nodes.get(idx).map(NodeProcess::log_path)
    }

    /// Shut down all nodes and proxies.
    pub async fn shutdown(&mut self) {
        for node in &mut self.nodes {
            node.kill();
        }
        for proxy in &mut self.proxies {
            proxy.shutdown().await;
        }
    }
}

impl Drop for ProcessCluster {
    fn drop(&mut self) {
        for node in &mut self.nodes {
            node.kill();
        }
    }
}
