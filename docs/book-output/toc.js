// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">Getting Started</li><li class="chapter-item expanded "><a href="getting-started/installation.html"><strong aria-hidden="true">1.</strong> Installation</a></li><li class="chapter-item expanded "><a href="getting-started/quick-start.html"><strong aria-hidden="true">2.</strong> Quick Start</a></li><li class="chapter-item expanded "><a href="getting-started/building-from-source.html"><strong aria-hidden="true">3.</strong> Building from Source</a></li><li class="chapter-item expanded affix "><li class="part-title">Architecture</li><li class="chapter-item expanded "><a href="architecture/overview.html"><strong aria-hidden="true">4.</strong> Overview</a></li><li class="chapter-item expanded "><a href="architecture/tensor-store.html"><strong aria-hidden="true">5.</strong> Tensor Store</a></li><li class="chapter-item expanded "><a href="architecture/relational-engine.html"><strong aria-hidden="true">6.</strong> Relational Engine</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="architecture/relational-transactions.html"><strong aria-hidden="true">6.1.</strong> Transactions</a></li></ol></li><li class="chapter-item expanded "><a href="architecture/graph-engine.html"><strong aria-hidden="true">7.</strong> Graph Engine</a></li><li class="chapter-item expanded "><a href="architecture/vector-engine.html"><strong aria-hidden="true">8.</strong> Vector Engine</a></li><li class="chapter-item expanded "><a href="architecture/tensor-compress.html"><strong aria-hidden="true">9.</strong> Tensor Compress</a></li><li class="chapter-item expanded "><a href="architecture/tensor-vault.html"><strong aria-hidden="true">10.</strong> Tensor Vault</a></li><li class="chapter-item expanded "><a href="architecture/tensor-cache.html"><strong aria-hidden="true">11.</strong> Tensor Cache</a></li><li class="chapter-item expanded "><a href="architecture/tensor-blob.html"><strong aria-hidden="true">12.</strong> Tensor Blob</a></li><li class="chapter-item expanded "><a href="architecture/tensor-checkpoint.html"><strong aria-hidden="true">13.</strong> Tensor Checkpoint</a></li><li class="chapter-item expanded "><a href="architecture/tensor-unified.html"><strong aria-hidden="true">14.</strong> Tensor Unified</a></li><li class="chapter-item expanded "><a href="architecture/tensor-chain.html"><strong aria-hidden="true">15.</strong> Tensor Chain</a></li><li class="chapter-item expanded "><a href="architecture/neumann-parser.html"><strong aria-hidden="true">16.</strong> Neumann Parser</a></li><li class="chapter-item expanded "><a href="architecture/query-router.html"><strong aria-hidden="true">17.</strong> Query Router</a></li><li class="chapter-item expanded "><a href="architecture/neumann-shell.html"><strong aria-hidden="true">18.</strong> Neumann Shell</a></li><li class="chapter-item expanded "><a href="architecture/neumann-server.html"><strong aria-hidden="true">19.</strong> Neumann Server</a></li><li class="chapter-item expanded "><a href="architecture/neumann-client.html"><strong aria-hidden="true">20.</strong> Neumann Client</a></li><li class="chapter-item expanded "><a href="architecture/neumann-ts.html"><strong aria-hidden="true">21.</strong> TypeScript SDK</a></li><li class="chapter-item expanded "><a href="architecture/neumann-py.html"><strong aria-hidden="true">22.</strong> Python SDK</a></li><li class="chapter-item expanded "><a href="architecture/tcp-transport.html"><strong aria-hidden="true">23.</strong> TCP Transport</a></li><li class="chapter-item expanded "><a href="architecture/snapshot-streaming.html"><strong aria-hidden="true">24.</strong> Snapshot Streaming</a></li><li class="chapter-item expanded "><a href="architecture/transaction-workspace.html"><strong aria-hidden="true">25.</strong> Transaction Workspace</a></li><li class="chapter-item expanded affix "><li class="part-title">Concepts</li><li class="chapter-item expanded "><a href="concepts/tensor-data-model.html"><strong aria-hidden="true">26.</strong> Tensor Data Model</a></li><li class="chapter-item expanded "><a href="concepts/sparse-vectors.html"><strong aria-hidden="true">27.</strong> Sparse Vectors</a></li><li class="chapter-item expanded "><a href="concepts/semantic-operations.html"><strong aria-hidden="true">28.</strong> Semantic Operations</a></li><li class="chapter-item expanded "><a href="concepts/distributed-transactions.html"><strong aria-hidden="true">29.</strong> Distributed Transactions</a></li><li class="chapter-item expanded "><a href="concepts/consensus-protocols.html"><strong aria-hidden="true">30.</strong> Consensus Protocols</a></li><li class="chapter-item expanded "><a href="concepts/embedding-state.html"><strong aria-hidden="true">31.</strong> Embedding State Machine</a></li><li class="chapter-item expanded "><a href="concepts/codebook-manager.html"><strong aria-hidden="true">32.</strong> Codebook Manager</a></li><li class="chapter-item expanded affix "><li class="part-title">Tutorials</li><li class="chapter-item expanded "><a href="tutorials/worked-examples.html"><strong aria-hidden="true">33.</strong> Worked Examples</a></li><li class="chapter-item expanded affix "><li class="part-title">Operations</li><li class="chapter-item expanded "><a href="operations/deployment.html"><strong aria-hidden="true">34.</strong> Deployment</a></li><li class="chapter-item expanded "><a href="operations/configuration.html"><strong aria-hidden="true">35.</strong> Configuration</a></li><li class="chapter-item expanded "><a href="operations/monitoring.html"><strong aria-hidden="true">36.</strong> Monitoring</a></li><li class="chapter-item expanded "><a href="operations/troubleshooting.html"><strong aria-hidden="true">37.</strong> Troubleshooting</a></li><li class="chapter-item expanded "><a href="operations/example-configurations.html"><strong aria-hidden="true">38.</strong> Example Configurations</a></li><li class="chapter-item expanded "><a href="operations/runbooks/index.html"><strong aria-hidden="true">39.</strong> Runbooks</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="operations/runbooks/node-management.html"><strong aria-hidden="true">39.1.</strong> Node Management</a></li><li class="chapter-item "><a href="operations/runbooks/cluster-upgrade.html"><strong aria-hidden="true">39.2.</strong> Cluster Upgrade</a></li><li class="chapter-item "><a href="operations/runbooks/leader-election.html"><strong aria-hidden="true">39.3.</strong> Leader Election</a></li><li class="chapter-item "><a href="operations/runbooks/split-brain.html"><strong aria-hidden="true">39.4.</strong> Split-Brain Recovery</a></li><li class="chapter-item "><a href="operations/runbooks/node-recovery.html"><strong aria-hidden="true">39.5.</strong> Node Recovery</a></li><li class="chapter-item "><a href="operations/runbooks/backup-restore.html"><strong aria-hidden="true">39.6.</strong> Backup and Restore</a></li><li class="chapter-item "><a href="operations/runbooks/capacity-planning.html"><strong aria-hidden="true">39.7.</strong> Capacity Planning</a></li><li class="chapter-item "><a href="operations/runbooks/deadlock-resolution.html"><strong aria-hidden="true">39.8.</strong> Deadlock Resolution</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">Performance</li><li class="chapter-item expanded "><a href="benchmarks/index.html"><strong aria-hidden="true">40.</strong> Benchmarks</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="benchmarks/tensor-store.html"><strong aria-hidden="true">40.1.</strong> tensor_store</a></li><li class="chapter-item "><a href="benchmarks/relational-engine.html"><strong aria-hidden="true">40.2.</strong> relational_engine</a></li><li class="chapter-item "><a href="benchmarks/graph-engine.html"><strong aria-hidden="true">40.3.</strong> graph_engine</a></li><li class="chapter-item "><a href="benchmarks/vector-engine.html"><strong aria-hidden="true">40.4.</strong> vector_engine</a></li><li class="chapter-item "><a href="benchmarks/tensor-compress.html"><strong aria-hidden="true">40.5.</strong> tensor_compress</a></li><li class="chapter-item "><a href="benchmarks/tensor-vault.html"><strong aria-hidden="true">40.6.</strong> tensor_vault</a></li><li class="chapter-item "><a href="benchmarks/tensor-cache.html"><strong aria-hidden="true">40.7.</strong> tensor_cache</a></li><li class="chapter-item "><a href="benchmarks/tensor-blob.html"><strong aria-hidden="true">40.8.</strong> tensor_blob</a></li><li class="chapter-item "><a href="benchmarks/tensor-chain.html"><strong aria-hidden="true">40.9.</strong> tensor_chain</a></li><li class="chapter-item "><a href="benchmarks/neumann-parser.html"><strong aria-hidden="true">40.10.</strong> neumann_parser</a></li><li class="chapter-item "><a href="benchmarks/query-router.html"><strong aria-hidden="true">40.11.</strong> query_router</a></li></ol></li><li class="chapter-item expanded "><a href="stress-tests/index.html"><strong aria-hidden="true">41.</strong> Stress Tests</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="stress-tests/hnsw.html"><strong aria-hidden="true">41.1.</strong> HNSW Stress</a></li><li class="chapter-item "><a href="stress-tests/tiered-store.html"><strong aria-hidden="true">41.2.</strong> TieredStore Stress</a></li><li class="chapter-item "><a href="stress-tests/mixed-workload.html"><strong aria-hidden="true">41.3.</strong> Mixed Workload</a></li></ol></li><li class="chapter-item expanded "><a href="integration-tests/index.html"><strong aria-hidden="true">42.</strong> Integration Tests</a></li><li class="chapter-item expanded affix "><li class="part-title">Contributing</li><li class="chapter-item expanded "><a href="contributing/code-style.html"><strong aria-hidden="true">43.</strong> Code Style</a></li><li class="chapter-item expanded "><a href="contributing/testing.html"><strong aria-hidden="true">44.</strong> Testing</a></li><li class="chapter-item expanded "><a href="contributing/documentation.html"><strong aria-hidden="true">45.</strong> Documentation</a></li><li class="chapter-item expanded "><a href="contributing/fuzzing.html"><strong aria-hidden="true">46.</strong> Fuzzing</a></li><li class="chapter-item expanded affix "><li class="spacer"></li><li class="chapter-item expanded affix "><a href="api-reference.html">API Reference</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
