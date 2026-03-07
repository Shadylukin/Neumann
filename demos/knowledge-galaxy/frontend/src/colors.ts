import * as THREE from 'three';

/** Category hex color palette. */
export const CATEGORY_COLORS: Record<string, string> = {
  AI: '#4488ff',
  ML: '#00ddff',
  CV: '#00cc88',
  NLP: '#ffaa00',
  SE: '#aa44ff',
  DB: '#00bbaa',
  Systems: '#ff4444',
  Theory: '#8844cc',
  Other: '#aaaacc',
};

/** Pre-built Three.js Color objects keyed by category. */
export const CATEGORY_THREE_COLORS: Record<string, THREE.Color> = {};
for (const [cat, hex] of Object.entries(CATEGORY_COLORS)) {
  CATEGORY_THREE_COLORS[cat] = new THREE.Color(hex);
}

/** Resolve a category string to a Three.js Color, falling back to Other. */
export function categoryColor(category: string): THREE.Color {
  return CATEGORY_THREE_COLORS[category] ?? CATEGORY_THREE_COLORS['Other'];
}
