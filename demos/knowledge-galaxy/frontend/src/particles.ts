import * as THREE from 'three';
import { categoryColor } from './colors.ts';

/** Raw paper data passed in from the API layer. */
export interface PaperData {
  id: string;
  x: number;
  y: number;
  z: number;
  category: string;
  cited_by_count: number;
  title?: string;
  authors?: string;
  year?: number;
  abstract?: string;
}

// Inline shaders to avoid async file loading at runtime.
const VERTEX_SHADER = /* glsl */ `
attribute vec3 aColor;
attribute float aSize;
attribute float aPhase;

uniform float uTime;
uniform float uPixelRatio;

varying vec3 vColor;
varying float vAlpha;

void main() {
    vColor = aColor;

    float pulse = 1.0 + 0.15 * sin(uTime * 1.5 + aPhase * 6.2831);

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    float dist = -mvPosition.z;
    gl_PointSize = aSize * pulse * uPixelRatio * (300.0 / dist);

    float fade = smoothstep(4000.0, 500.0, dist);
    vAlpha = fade * (0.7 + 0.3 * pulse);

    gl_Position = projectionMatrix * mvPosition;
}
`;

const FRAGMENT_SHADER = /* glsl */ `
varying vec3 vColor;
varying float vAlpha;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;

    float glow = 1.0 - smoothstep(0.0, 0.5, dist);
    float core = smoothstep(0.2, 0.0, dist) * 0.6;

    vec3 color = vColor * (glow + core) + vec3(core * 0.3);
    float alpha = glow * vAlpha;

    gl_FragColor = vec4(color, alpha);
}
`;

const BASE_SIZE = 6.0;
const MIN_SIZE = 2.0;
const MAX_SIZE = 20.0;
const ENTRY_DURATION = 1.5; // seconds for staggered fade-in

export class ParticleSystem {
  private points: THREE.Points;
  private material: THREE.ShaderMaterial;
  private papers: PaperData[];
  private idToIndex: Map<string, number>;
  private baseColors: Float32Array;
  private entryStart: number;

  constructor(papers: PaperData[]) {
    this.papers = papers;
    this.idToIndex = new Map();
    this.entryStart = performance.now() / 1000;

    const count = papers.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    const phases = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const p = papers[i];
      this.idToIndex.set(p.id, i);

      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;

      const color = categoryColor(p.category);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;

      // Size scaled by citation count, clamped
      const raw = BASE_SIZE * (1 + Math.log(1 + p.cited_by_count) * 0.5);
      sizes[i] = Math.min(MAX_SIZE, Math.max(MIN_SIZE, raw));

      phases[i] = Math.random();
    }

    // Keep a copy for highlight/restore
    this.baseColors = new Float32Array(colors);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));

    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: {
        uTime: { value: 0.0 },
        uPixelRatio: { value: window.devicePixelRatio },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geometry, this.material);
  }

  /** Update time uniform and entry animation. */
  update(time: number): void {
    this.material.uniforms['uTime'].value = time;

    // Staggered entry animation: particles fade in from the center
    const elapsed = time - this.entryStart;
    if (elapsed < ENTRY_DURATION) {
      const progress = elapsed / ENTRY_DURATION;
      const geom = this.points.geometry;
      const positions = geom.getAttribute('position') as THREE.BufferAttribute;
      const sizes = geom.getAttribute('aSize') as THREE.BufferAttribute;

      for (let i = 0; i < this.papers.length; i++) {
        const stagger = i / this.papers.length;
        const t = Math.min(1, Math.max(0, (progress - stagger * 0.5) / 0.5));
        const ease = t * t * (3 - 2 * t); // smoothstep

        const p = this.papers[i];
        positions.setXYZ(i, p.x * ease, p.y * ease, p.z * ease);

        const raw =
          BASE_SIZE * (1 + Math.log(1 + p.cited_by_count) * 0.5);
        sizes.setX(i, Math.min(MAX_SIZE, Math.max(MIN_SIZE, raw)) * ease);
      }

      positions.needsUpdate = true;
      sizes.needsUpdate = true;
    }
  }

  /** Highlight specific papers; dim everything else. */
  highlight(ids: Set<string>): void {
    const colorAttr = this.points.geometry.getAttribute(
      'aColor',
    ) as THREE.BufferAttribute;
    const arr = colorAttr.array as Float32Array;

    if (ids.size === 0) {
      // Restore all
      arr.set(this.baseColors);
    } else {
      for (let i = 0; i < this.papers.length; i++) {
        if (ids.has(this.papers[i].id)) {
          // Full brightness
          arr[i * 3] = this.baseColors[i * 3];
          arr[i * 3 + 1] = this.baseColors[i * 3 + 1];
          arr[i * 3 + 2] = this.baseColors[i * 3 + 2];
        } else {
          // Dim to 15% opacity
          arr[i * 3] = this.baseColors[i * 3] * 0.15;
          arr[i * 3 + 1] = this.baseColors[i * 3 + 1] * 0.15;
          arr[i * 3 + 2] = this.baseColors[i * 3 + 2] * 0.15;
        }
      }
    }

    colorAttr.needsUpdate = true;
  }

  /** Get the index of a paper by ID. */
  getIndex(id: string): number | undefined {
    return this.idToIndex.get(id);
  }

  /** Get paper data by index. */
  getPaper(index: number): PaperData | undefined {
    return this.papers[index];
  }

  /** Get total paper count. */
  get count(): number {
    return this.papers.length;
  }

  /** Get the Three.js Points object for adding to a scene. */
  getObject(): THREE.Points {
    return this.points;
  }

  /** Get position of a paper by ID. */
  getPosition(id: string): THREE.Vector3 | null {
    const idx = this.idToIndex.get(id);
    if (idx === undefined) return null;
    const pos = this.points.geometry.getAttribute(
      'position',
    ) as THREE.BufferAttribute;
    return new THREE.Vector3(
      pos.getX(idx),
      pos.getY(idx),
      pos.getZ(idx),
    );
  }
}
