import { queryGalaxy } from './api.ts';
import type { PaperData } from './particles.ts';

export type SearchResultCallback = (ids: string[], latencyMs: number) => void;

/** Known category short labels for category-based search. */
const CATEGORIES = new Set([
  'ai', 'ml', 'cv', 'nlp', 'se', 'db', 'systems', 'theory', 'other',
]);

export class SearchBar {
  private input: HTMLInputElement;
  private button: HTMLButtonElement;
  private resultsEl: HTMLElement;
  private onResults: SearchResultCallback;
  private searching = false;
  private papers: PaperData[] = [];

  constructor(onResults: SearchResultCallback) {
    this.onResults = onResults;

    this.input = document.getElementById('search-input') as HTMLInputElement;
    this.button = document.getElementById('search-btn') as HTMLButtonElement;
    this.resultsEl = document.getElementById(
      'search-results',
    ) as HTMLElement;

    this.button.addEventListener('click', () => this.doSearch());
    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.doSearch();
    });
  }

  /** Set the paper data for client-side title search. */
  setPapers(papers: PaperData[]): void {
    this.papers = papers;
  }

  private async doSearch(): Promise<void> {
    const term = this.input.value.trim();
    if (!term || this.searching) return;

    this.searching = true;
    this.button.textContent = '...';
    this.resultsEl.textContent = 'Searching...';

    const start = performance.now();

    try {
      let ids: string[];

      if (CATEGORIES.has(term.toLowerCase())) {
        // Category search: use server-side FIND NODE WHERE
        const cat = term.toUpperCase() === 'NLP' ? 'NLP' : term.charAt(0).toUpperCase() + term.slice(1);
        const response = await queryGalaxy(
          `FIND NODE WHERE category = '${cat}' LIMIT 500`,
        );
        ids = response.items
          .map((item) => String(item['label'] ?? item['id'] ?? item['_id'] ?? ''))
          .filter((id) => id.length > 0);
      } else {
        // Client-side title search (fast, all papers already loaded)
        const lower = term.toLowerCase();
        ids = this.papers
          .filter((p) => p.title?.toLowerCase().includes(lower))
          .map((p) => p.id);
      }

      const latencyMs = Math.round(performance.now() - start);
      this.resultsEl.textContent = `${ids.length} results (${latencyMs}ms)`;
      this.onResults(ids, latencyMs);
    } catch (err) {
      const latencyMs = Math.round(performance.now() - start);
      this.resultsEl.textContent = `Network error`;
      console.error('Search failed:', err);
      this.onResults([], latencyMs);
    } finally {
      this.searching = false;
      this.button.textContent = 'Search';
    }
  }

  /** Clear the search state and restore all particles. */
  clear(): void {
    this.input.value = '';
    this.resultsEl.textContent = '';
    this.onResults([], 0);
  }
}
