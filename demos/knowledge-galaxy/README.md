# Knowledge Galaxy

Interactive 3D visualization of ~50K CS papers powered by Neumann's unified infrastructure.

## Architecture

- **galaxy-server** (Rust): Serves the Neumann database via Web (port 9000) and REST (port 8080) APIs
- **Frontend** (Three.js + TypeScript): 3D particle visualization with bloom, search, and spatial selection
- **Pipeline** (Python): Fetches papers from OpenAlex, generates embeddings and UMAP projections, loads into Neumann

## Port Contract

| Service | Port | Purpose |
|---------|------|---------|
| Web | 9000 | `/api/galaxy` - read-only query endpoint |
| REST | 8080 | `/collections/galaxy/spatial3d/*` - 3D spatial operations |
| Frontend | 5173 | Vite dev server |
| gRPC | 9200 | Standard Neumann gRPC (not used by frontend) |

## Quick Start

### 1. Run the data pipeline

```bash
cd pipeline
pip install -r requirements.txt

# Fetch ~50K CS papers from OpenAlex
python fetch_papers.py --max-papers 50000

# Generate embeddings (requires GPU recommended)
python embed_papers.py

# Project to 3D coordinates
python project_umap.py

# Load into Neumann database
python load_neumann.py --output ../galaxy.db
```

### 2. Start the server

```bash
# From demos/knowledge-galaxy/
cargo run --release -- --db galaxy.db --web-port 9000 --rest-port 8080
```

### 3. Load 3D spatial data (after server starts)

```bash
cd pipeline
python load_spatial.py --rest-url http://localhost:8080
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Features

- 50K papers as glowing particles in 3D space (UMAP projection)
- Category colors: AI (blue), ML (cyan), CV (emerald), NLP (amber), SE (violet), DB (teal), Systems (red), Theory (purple)
- Particle size scaled by citation count
- Full-text search via hero query (FIND NODE)
- Citation edges on click (CONNECTED TO)
- Spatial region selection (shift+drag)
- Bloom post-processing and constellation lines
- 60fps with Three.js Points + custom shaders

## Hero Query Examples

```sql
-- Search by similarity
FIND NODE SIMILAR TO 'paper:W2100837269' LIMIT 100

-- Filter by category
FIND NODE WHERE category = 'AI' LIMIT 100

-- Combined search
FIND NODE WHERE category = 'ML' SIMILAR TO 'paper:W2100837269' CONNECTED TO 'paper:W2034567890' LIMIT 50
```

## Development

The demo binary is excluded from the workspace (not in `members`, in `exclude`).
Build with: `cargo build -p knowledge-galaxy --release`

The 3D spatial index is ephemeral - it lives in memory and must be reloaded via `load_spatial.py` on each server restart.
