// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Docker-based Jepsen cluster management.
//!
//! Manages a cluster of `neumann_server` containers via Docker Compose for
//! Jepsen-grade fault injection testing. Supports kernel-level network faults
//! (iptables, tc netem), process crashes (docker kill), and disk faults.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

/// Error type for Docker operations.
#[derive(Debug)]
pub enum DockerError {
    Io(std::io::Error),
    CommandFailed { cmd: String, stderr: String },
    Timeout(String),
}

impl std::fmt::Display for DockerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::CommandFailed { cmd, stderr } => {
                write!(f, "command failed: {cmd}\nstderr: {stderr}")
            },
            Self::Timeout(msg) => write!(f, "timeout: {msg}"),
        }
    }
}

impl std::error::Error for DockerError {}

impl From<std::io::Error> for DockerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A Docker-based Jepsen test cluster.
///
/// Manages container lifecycle, network faults, and disk faults via the
/// `docker` CLI. No external crate dependencies.
pub struct DockerCluster {
    compose_file: PathBuf,
    node_count: usize,
    container_names: Vec<String>,
    grpc_addrs: Vec<String>,
    project_name: String,
}

impl DockerCluster {
    /// Start a cluster with the given number of nodes (3 or 5).
    ///
    /// Selects the appropriate compose file and brings up all containers.
    /// The Docker image must be pre-built (`docker build --target jepsen -t neumann:jepsen .`).
    ///
    /// # Errors
    ///
    /// Returns an error if Docker commands fail.
    pub fn start(node_count: usize) -> Result<Self, DockerError> {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf();

        let compose_file = if node_count <= 3 {
            workspace_root.join("docker-compose.jepsen.yml")
        } else {
            workspace_root.join("docker-compose.jepsen-5node.yml")
        };

        let actual_count = if node_count <= 3 { 3 } else { 5 };

        let container_names: Vec<String> = (0..actual_count)
            .map(|i| format!("neumann-node-{i}"))
            .collect();

        let grpc_addrs: Vec<String> = (0..actual_count)
            .map(|i| {
                let port = 19200 + i;
                format!("http://127.0.0.1:{port}")
            })
            .collect();

        let project_name = format!("jepsen-{}", std::process::id());

        let cluster = Self {
            compose_file,
            node_count: actual_count,
            container_names,
            grpc_addrs,
            project_name,
        };

        // Bring up all containers
        cluster.compose_up()?;

        Ok(cluster)
    }

    /// Bring up the cluster via docker compose.
    fn compose_up(&self) -> Result<(), DockerError> {
        let compose_file = self.compose_file.to_string_lossy().to_string();
        let project = &self.project_name;

        docker_command(&[
            "compose",
            "-f",
            &compose_file,
            "-p",
            project,
            "up",
            "-d",
            "--wait",
        ])?;

        Ok(())
    }

    /// Wait for all nodes to become healthy via gRPC health check.
    ///
    /// # Errors
    ///
    /// Returns an error if any node does not become healthy within the timeout.
    pub async fn wait_healthy(&self, timeout: Duration) -> Result<(), DockerError> {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut interval = tokio::time::interval(Duration::from_millis(500));

        for (idx, addr) in self.grpc_addrs.iter().enumerate() {
            loop {
                if tokio::time::Instant::now() >= deadline {
                    return Err(DockerError::Timeout(format!(
                        "node-{idx} did not become healthy within {timeout:?}"
                    )));
                }
                interval.tick().await;

                let connect_result = tokio::time::timeout(
                    Duration::from_millis(1000),
                    neumann_client::NeumannClient::connect(addr).build(),
                )
                .await;

                if let Ok(Ok(client)) = connect_result {
                    let query_result = tokio::time::timeout(
                        Duration::from_millis(1000),
                        client.execute("SELECT 1"),
                    )
                    .await;
                    if query_result.is_ok() {
                        eprintln!("[docker] node-{idx} healthy");
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Shut down the cluster and remove volumes.
    ///
    /// # Errors
    ///
    /// Returns an error if compose down fails.
    pub fn shutdown(&self) -> Result<(), DockerError> {
        let compose_file = self.compose_file.to_string_lossy().to_string();
        let project = &self.project_name;

        let _ = docker_command(&["compose", "-f", &compose_file, "-p", project, "down", "-v"]);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Process fault injection
    // -----------------------------------------------------------------------

    /// Kill a node immediately (SIGKILL).
    ///
    /// # Errors
    ///
    /// Returns an error if the docker command fails.
    pub fn kill_node(&self, idx: usize) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] killing {container}");
        docker_command(&["kill", "--signal=KILL", container])?;
        Ok(())
    }

    /// Stop a node gracefully.
    ///
    /// # Errors
    ///
    /// Returns an error if the docker command fails.
    pub fn stop_node(&self, idx: usize) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] stopping {container}");
        docker_command(&["stop", container])?;
        Ok(())
    }

    /// Start a previously stopped/killed node (WAL recovery restart).
    ///
    /// # Errors
    ///
    /// Returns an error if the docker command fails.
    pub fn start_node(&self, idx: usize) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] starting {container}");
        docker_command(&["start", container])?;
        Ok(())
    }

    /// Pause a node (freeze all processes in the container).
    ///
    /// # Errors
    ///
    /// Returns an error if the docker command fails.
    pub fn pause_node(&self, idx: usize) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] pausing {container}");
        docker_command(&["pause", container])?;
        Ok(())
    }

    /// Resume a paused node.
    ///
    /// # Errors
    ///
    /// Returns an error if the docker command fails.
    pub fn resume_node(&self, idx: usize) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] resuming {container}");
        docker_command(&["unpause", container])?;
        Ok(())
    }

    /// Wait for a single node to become healthy.
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not respond within the timeout.
    pub async fn wait_node_healthy(
        &self,
        idx: usize,
        timeout: Duration,
    ) -> Result<(), DockerError> {
        let addr = &self.grpc_addrs[idx];
        let deadline = tokio::time::Instant::now() + timeout;
        let mut interval = tokio::time::interval(Duration::from_millis(500));

        loop {
            if tokio::time::Instant::now() >= deadline {
                return Err(DockerError::Timeout(format!(
                    "node-{idx} did not become healthy within {timeout:?}"
                )));
            }
            interval.tick().await;

            let connect_result = tokio::time::timeout(
                Duration::from_millis(1000),
                neumann_client::NeumannClient::connect(addr).build(),
            )
            .await;

            if let Ok(Ok(client)) = connect_result {
                let query_result =
                    tokio::time::timeout(Duration::from_millis(1000), client.execute("SELECT 1"))
                        .await;
                if query_result.is_ok() {
                    return Ok(());
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Network fault injection (kernel-level via docker exec)
    // -----------------------------------------------------------------------

    /// Partition two nodes using iptables (bidirectional).
    ///
    /// # Errors
    ///
    /// Returns an error if iptables commands fail.
    pub fn partition_node(&self, a: usize, b: usize) -> Result<(), DockerError> {
        let container_a = &self.container_names[a];
        let container_b = &self.container_names[b];

        let ip_a = resolve_container_ip(container_a)?;
        let ip_b = resolve_container_ip(container_b)?;

        eprintln!("[docker] partitioning node-{a} ({ip_a}) <-> node-{b} ({ip_b})");

        // Block traffic on A destined for B
        docker_exec(
            container_a,
            &format!("iptables -A INPUT -s {ip_b} -j DROP && iptables -A OUTPUT -d {ip_b} -j DROP"),
        )?;

        // Block traffic on B destined for A
        docker_exec(
            container_b,
            &format!("iptables -A INPUT -s {ip_a} -j DROP && iptables -A OUTPUT -d {ip_a} -j DROP"),
        )?;

        Ok(())
    }

    /// Heal a partition between two nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if iptables commands fail.
    pub fn heal_partition(&self, a: usize, b: usize) -> Result<(), DockerError> {
        let container_a = &self.container_names[a];
        let container_b = &self.container_names[b];

        eprintln!("[docker] healing partition node-{a} <-> node-{b}");

        docker_exec(container_a, "iptables -F")?;
        docker_exec(container_b, "iptables -F")?;

        Ok(())
    }

    /// Partition one node from all others.
    ///
    /// # Errors
    ///
    /// Returns an error if iptables commands fail.
    pub fn partition_from_all(&self, idx: usize) -> Result<(), DockerError> {
        eprintln!("[docker] partitioning node-{idx} from all");
        for other in 0..self.node_count {
            if other != idx {
                self.partition_node(idx, other)?;
            }
        }
        Ok(())
    }

    /// Heal all network partitions on all nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if iptables flush fails on any node.
    pub fn heal_all(&self) -> Result<(), DockerError> {
        eprintln!("[docker] healing all partitions");
        for container in &self.container_names {
            let _ = docker_exec(container, "iptables -F");
            let _ = docker_exec(container, "tc qdisc del dev eth0 root 2>/dev/null || true");
        }
        Ok(())
    }

    /// Add network latency to a node using tc netem.
    ///
    /// # Errors
    ///
    /// Returns an error if tc commands fail.
    pub fn add_latency(&self, idx: usize, ms: u32) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] adding {ms}ms latency to node-{idx}");
        docker_exec(
            container,
            &format!("tc qdisc replace dev eth0 root netem delay {ms}ms"),
        )?;
        Ok(())
    }

    /// Add packet loss to a node using tc netem.
    ///
    /// # Errors
    ///
    /// Returns an error if tc commands fail.
    pub fn add_packet_loss(&self, idx: usize, pct: u32) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] adding {pct}% packet loss to node-{idx}");
        docker_exec(
            container,
            &format!("tc qdisc replace dev eth0 root netem loss {pct}%"),
        )?;
        Ok(())
    }

    /// Clear all network faults (iptables + tc) on a node.
    ///
    /// # Errors
    ///
    /// Returns an error if the commands fail.
    pub fn clear_network_faults(&self, idx: usize) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] clearing network faults on node-{idx}");
        docker_exec(container, "iptables -F")?;
        let _ = docker_exec(container, "tc qdisc del dev eth0 root 2>/dev/null || true");
        Ok(())
    }

    /// Create an asymmetric partition: block traffic from A to B but not B to A.
    ///
    /// # Errors
    ///
    /// Returns an error if iptables commands fail.
    pub fn asymmetric_partition(&self, from: usize, to: usize) -> Result<(), DockerError> {
        let container_from = &self.container_names[from];
        let container_to = &self.container_names[to];
        let ip_to = resolve_container_ip(container_to)?;

        eprintln!("[docker] asymmetric partition: node-{from} -> node-{to} blocked");
        docker_exec(
            container_from,
            &format!("iptables -A OUTPUT -d {ip_to} -j DROP"),
        )?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Disk fault injection
    // -----------------------------------------------------------------------

    /// Fill the data volume with a large file to simulate disk full.
    ///
    /// # Errors
    ///
    /// Returns an error if the dd command fails unexpectedly.
    pub fn fill_disk(&self, idx: usize) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] filling disk on node-{idx}");
        // dd will fail when disk is full; we treat that as expected
        let _ = docker_exec(
            container,
            "dd if=/dev/zero of=/var/lib/neumann/filler bs=1M count=100 2>/dev/null || true",
        );
        Ok(())
    }

    /// Remove the filler file to free disk space.
    ///
    /// # Errors
    ///
    /// Returns an error if the rm command fails.
    pub fn clear_disk(&self, idx: usize) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] clearing disk on node-{idx}");
        docker_exec(container, "rm -f /var/lib/neumann/filler")?;
        Ok(())
    }

    /// Clear WAL files on a node (for testing recovery from empty state).
    ///
    /// # Errors
    ///
    /// Returns an error if the rm command fails.
    pub fn clear_wal(&self, idx: usize) -> Result<(), DockerError> {
        let container = &self.container_names[idx];
        eprintln!("[docker] clearing WAL on node-{idx}");
        docker_exec(container, "rm -f /var/lib/neumann/*.wal")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /// Get the gRPC address for a node.
    #[must_use]
    pub fn grpc_addr(&self, idx: usize) -> &str {
        &self.grpc_addrs[idx]
    }

    /// Get the number of nodes in the cluster.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Collect logs from a specific container.
    ///
    /// # Errors
    ///
    /// Returns an error if docker logs fails.
    pub fn collect_logs(&self, idx: usize) -> Result<String, DockerError> {
        let container = &self.container_names[idx];
        docker_command_output(&["logs", "--tail=200", container])
    }
}

impl Drop for DockerCluster {
    fn drop(&mut self) {
        let compose_file = self.compose_file.to_string_lossy().to_string();
        let project = &self.project_name;

        let _ = Command::new("docker")
            .args(["compose", "-f", &compose_file, "-p", project, "down", "-v"])
            .output();
    }
}

/// Run a docker command and check for success.
fn docker_command(args: &[&str]) -> Result<(), DockerError> {
    let output = Command::new("docker").args(args).output()?;

    if output.status.success() {
        Ok(())
    } else {
        Err(DockerError::CommandFailed {
            cmd: format!("docker {}", args.join(" ")),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        })
    }
}

/// Run a docker command and return stdout.
fn docker_command_output(args: &[&str]) -> Result<String, DockerError> {
    let output = Command::new("docker").args(args).output()?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(DockerError::CommandFailed {
            cmd: format!("docker {}", args.join(" ")),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        })
    }
}

/// Execute a command inside a container.
fn docker_exec(container: &str, cmd: &str) -> Result<String, DockerError> {
    docker_command_output(&["exec", container, "bash", "-c", cmd])
}

/// Resolve the container IP address on the Docker network.
fn resolve_container_ip(container: &str) -> Result<String, DockerError> {
    let ip = docker_command_output(&[
        "inspect",
        "-f",
        "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
        container,
    ])?;
    let ip = ip.trim().to_string();
    if ip.is_empty() {
        return Err(DockerError::CommandFailed {
            cmd: format!("resolve IP for {container}"),
            stderr: "empty IP address".to_string(),
        });
    }
    Ok(ip)
}
