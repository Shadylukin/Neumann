import { Galaxy } from './galaxy.ts';
import { ParticleSystem } from './particles.ts';
import type { PaperData } from './particles.ts';
import { ConnectionRenderer } from './connections.ts';
import { SearchBar } from './search.ts';
import { Tooltip } from './tooltip.ts';
import { SpatialSelector } from './spatial.ts';
import { queryGalaxy } from './api.ts';
import { categoryColor } from './colors.ts';

// DOM elements for stats overlay
const statTotal = document.getElementById('stat-total') as HTMLElement;
const statVisible = document.getElementById('stat-visible') as HTMLElement;
const statLatency = document.getElementById('stat-latency') as HTMLElement;
const statFps = document.getElementById('stat-fps') as HTMLElement;
const loadingOverlay = document.getElementById(
  'loading-overlay',
) as HTMLElement;

// Pipeline stores coordinates offset by +500 to avoid negative numbers
// (the Neumann parser doesn't handle negative literals in properties).
const COORD_OFFSET = 500;

/** Convert raw API items to PaperData with 3D positions.
 *  Items are pre-flattened by flattenResponse() in api.ts, so fields
 *  like title, category, year are direct properties. The node key
 *  comes from the `label` field (graph node label) or `_id`. */
function toPaperData(items: Record<string, unknown>[]): PaperData[] {
  return items.map((item, i) => {
    const nodeId = String(item['label'] ?? item['id'] ?? item['_id'] ?? `p${i}`);
    const hasCoords = item['x'] != null;

    return {
      id: nodeId,
      x: hasCoords ? Number(item['x']) - COORD_OFFSET : (Math.random() - 0.5) * 600,
      y: hasCoords ? Number(item['y']) - COORD_OFFSET : (Math.random() - 0.5) * 600,
      z: hasCoords ? Number(item['z']) - COORD_OFFSET : (Math.random() - 0.5) * 600,
      category: String(item['category'] ?? item['topic'] ?? 'Other'),
      cited_by_count: Number(item['cited_by_count'] ?? item['citations'] ?? 0),
      title: item['title'] ? String(item['title']) : undefined,
      authors: item['authors'] ? String(item['authors']) : undefined,
      year: item['year'] ? Number(item['year']) : undefined,
      abstract: item['abstract'] ? String(item['abstract']) : undefined,
    };
  });
}

/** Generate synthetic demo data when the backend is not available. */
function generateDemoData(count: number): PaperData[] {
  const categories = [
    'AI',
    'ML',
    'CV',
    'NLP',
    'SE',
    'DB',
    'Systems',
    'Theory',
  ];
  const papers: PaperData[] = [];

  for (let i = 0; i < count; i++) {
    const cat = categories[Math.floor(Math.random() * categories.length)];

    // Cluster by category with some spread
    const catIdx = categories.indexOf(cat);
    const angle = (catIdx / categories.length) * Math.PI * 2;
    const radius = 150 + Math.random() * 100;

    papers.push({
      id: `paper:${i}`,
      x: Math.cos(angle) * radius + (Math.random() - 0.5) * 120,
      y: (Math.random() - 0.5) * 300,
      z: Math.sin(angle) * radius + (Math.random() - 0.5) * 120,
      category: cat,
      cited_by_count: Math.floor(Math.random() * 500),
      title: `Research Paper #${i} on ${cat}`,
      authors: 'Demo Author et al.',
      year: 2015 + Math.floor(Math.random() * 11),
      abstract: `This paper explores advances in ${cat} with novel approaches to fundamental problems in the field.`,
    });
  }

  return papers;
}

async function loadPapers(): Promise<PaperData[]> {
  try {
    // Load graph nodes (from quick_load.py), fall back to relational table
    let response = await queryGalaxy('NODE LIST LIMIT 50000');
    if (response.error || response.items.length === 0) {
      response = await queryGalaxy('SELECT * FROM papers LIMIT 50000');
    }
    if (response.error) {
      console.warn('Galaxy API returned error, using demo data:', response.error);
      return generateDemoData(5000);
    }
    if (response.items.length === 0) {
      console.warn('No papers returned, using demo data');
      return generateDemoData(5000);
    }
    return toPaperData(response.items);
  } catch {
    console.warn('Could not reach Galaxy API, using demo data');
    return generateDemoData(5000);
  }
}

async function main(): Promise<void> {
  // Load paper data
  const papers = await loadPapers();

  // Create the galaxy renderer
  const galaxy = new Galaxy();

  // Create particle system
  const particleSystem = new ParticleSystem(papers);
  galaxy.addParticles(particleSystem);

  // Create connection renderer
  const connections = new ConnectionRenderer();
  galaxy.getScene().add(connections.getObject());
  galaxy.onAnimate((time) => connections.update(time));

  // Track visible and highlighted paper count
  let visibleCount = papers.length;
  let highlightedIds: Set<string> = new Set();

  // Search bar
  const searchBar = new SearchBar((ids, latencyMs) => {
    statLatency.textContent = `${latencyMs}ms`;

    if (ids.length === 0) {
      // Clear highlight
      highlightedIds = new Set();
      particleSystem.highlight(highlightedIds);
      visibleCount = papers.length;
      connections.clear();
    } else {
      highlightedIds = new Set(ids);
      particleSystem.highlight(highlightedIds);
      visibleCount = ids.length;

      // Fly to the centroid of results
      let cx = 0,
        cy = 0,
        cz = 0,
        n = 0;
      for (const id of ids) {
        const pos = particleSystem.getPosition(id);
        if (pos) {
          cx += pos.x;
          cy += pos.y;
          cz += pos.z;
          n++;
        }
      }
      if (n > 0) {
        galaxy.flyTo(cx / n, cy / n, cz / n);
      }
    }
  });
  searchBar.setPapers(papers);

  // Tooltip with click-to-show-edges
  const tooltip = new Tooltip(
    galaxy.getCamera(),
    galaxy.getRenderer().domElement,
    (paperId) => {
      // On click, show citation edges from this paper
      const sourcePos = particleSystem.getPosition(paperId);
      if (!sourcePos) return;

      // Query for connected papers
      queryGalaxy(`FIND NODE WHERE cited_by = '${paperId}' LIMIT 50`)
        .then((response) => {
          if (response.error || response.items.length === 0) {
            connections.clear();
            return;
          }

          const targetPositions: import('three').Vector3[] = [];
          for (const item of response.items) {
            const targetId = String(item['id'] ?? item['_id'] ?? '');
            const pos = particleSystem.getPosition(targetId);
            if (pos) targetPositions.push(pos);
          }

          const paper = particleSystem.getPaper(
            particleSystem.getIndex(paperId) ?? 0,
          );
          const color = paper
            ? categoryColor(paper.category)
            : categoryColor('Other');

          connections.showEdges(sourcePos, targetPositions, color);
        })
        .catch((err) => {
          console.error('Failed to load citation edges:', err);
          connections.clear();
        });
    },
  );
  tooltip.bind(particleSystem);

  // Spatial selector (shift+drag)
  const _spatialSelector = new SpatialSelector(
    galaxy.getCamera(),
    galaxy.getScene(),
    galaxy.getRenderer().domElement,
    (ids, count) => {
      if (count > 0) {
        highlightedIds = new Set(ids);
        particleSystem.highlight(highlightedIds);
        visibleCount = count;
      } else {
        highlightedIds = new Set();
        particleSystem.highlight(highlightedIds);
        visibleCount = papers.length;
      }
    },
  );
  _spatialSelector.enable();

  // Update stats overlay each second
  setInterval(() => {
    statTotal.textContent = String(papers.length);
    statVisible.textContent = String(visibleCount);
    statFps.textContent = String(galaxy.getFps());
  }, 500);

  // Hide loading overlay
  loadingOverlay.classList.add('fade-out');
  setTimeout(() => {
    loadingOverlay.style.display = 'none';
  }, 800);
}

main().catch((err) => {
  console.error('Failed to initialize Knowledge Galaxy:', err);
  loadingOverlay.querySelector('.loading-text')!.textContent =
    'Failed to load. Check console for details.';
});
