import * as THREE from 'three';
import type { ParticleSystem, PaperData } from './particles.ts';
import { CATEGORY_COLORS } from './colors.ts';

export type TooltipClickCallback = (paperId: string) => void;

export class Tooltip {
  private el: HTMLElement;
  private titleEl: HTMLElement;
  private badgeEl: HTMLElement;
  private metaEl: HTMLElement;
  private abstractEl: HTMLElement;
  private raycaster: THREE.Raycaster;
  private mouse: THREE.Vector2;
  private camera: THREE.PerspectiveCamera;
  private particles: ParticleSystem | null = null;
  private onClick: TooltipClickCallback;
  private hoveredPaper: PaperData | null = null;
  private visible = false;

  constructor(
    camera: THREE.PerspectiveCamera,
    domElement: HTMLElement,
    onClick: TooltipClickCallback,
  ) {
    this.camera = camera;
    this.onClick = onClick;
    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points = { threshold: 8 };
    this.mouse = new THREE.Vector2();

    this.el = document.getElementById('tooltip') as HTMLElement;
    this.titleEl = this.el.querySelector('.tooltip-title') as HTMLElement;
    this.badgeEl = this.el.querySelector('.tooltip-badge') as HTMLElement;
    this.metaEl = this.el.querySelector('.tooltip-meta') as HTMLElement;
    this.abstractEl = this.el.querySelector(
      '.tooltip-abstract',
    ) as HTMLElement;

    domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
    domElement.addEventListener('click', () => this.onMouseClick());
  }

  /** Bind a particle system for raycasting. */
  bind(particles: ParticleSystem): void {
    this.particles = particles;
  }

  private onMouseMove(event: MouseEvent): void {
    this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    if (!this.particles) return;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersections = this.raycaster.intersectObject(
      this.particles.getObject(),
    );

    if (intersections.length > 0) {
      const idx = intersections[0].index;
      if (idx !== undefined) {
        const paper = this.particles.getPaper(idx);
        if (paper) {
          this.show(paper, event.clientX, event.clientY);
          return;
        }
      }
    }

    this.hide();
  }

  private onMouseClick(): void {
    if (this.hoveredPaper) {
      this.onClick(this.hoveredPaper.id);
    }
  }

  private show(paper: PaperData, mouseX: number, mouseY: number): void {
    this.hoveredPaper = paper;
    this.visible = true;

    this.titleEl.textContent = paper.title ?? paper.id;

    // Category badge
    const catColor = CATEGORY_COLORS[paper.category] ?? CATEGORY_COLORS['Other'];
    this.badgeEl.textContent = paper.category;
    this.badgeEl.style.background = catColor + '33';
    this.badgeEl.style.color = catColor;
    this.badgeEl.style.display = 'inline-block';

    // Meta line
    const parts: string[] = [];
    if (paper.authors) parts.push(paper.authors);
    if (paper.year) parts.push(String(paper.year));
    parts.push(`${paper.cited_by_count} citations`);
    this.metaEl.textContent = parts.join(' / ');

    // Abstract
    if (paper.abstract) {
      this.abstractEl.textContent = paper.abstract;
      this.abstractEl.style.display = 'block';
    } else {
      this.abstractEl.style.display = 'none';
    }

    // Position near cursor, keep within viewport
    const pad = 16;
    let left = mouseX + pad;
    let top = mouseY + pad;

    if (left + 320 > window.innerWidth) {
      left = mouseX - 320 - pad;
    }
    if (top + 200 > window.innerHeight) {
      top = mouseY - 200 - pad;
    }

    this.el.style.left = `${left}px`;
    this.el.style.top = `${top}px`;
    this.el.style.display = 'block';
  }

  private hide(): void {
    if (!this.visible) return;
    this.visible = false;
    this.hoveredPaper = null;
    this.el.style.display = 'none';
  }
}
