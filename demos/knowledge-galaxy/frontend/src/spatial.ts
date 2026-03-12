import * as THREE from 'three';
import { spatialRegion3D } from './api.ts';

export type SpatialSelectCallback = (ids: string[], count: number) => void;

export class SpatialSelector {
  private camera: THREE.PerspectiveCamera;
  private scene: THREE.Scene;
  private domElement: HTMLElement;
  private onSelect: SpatialSelectCallback;

  private enabled = false;
  private dragging = false;
  private startScreen = new THREE.Vector2();
  private endScreen = new THREE.Vector2();
  private wireframe: THREE.LineSegments | null = null;

  constructor(
    camera: THREE.PerspectiveCamera,
    scene: THREE.Scene,
    domElement: HTMLElement,
    onSelect: SpatialSelectCallback,
  ) {
    this.camera = camera;
    this.scene = scene;
    this.domElement = domElement;
    this.onSelect = onSelect;

    this.domElement.addEventListener('mousedown', (e) =>
      this.onMouseDown(e),
    );
    this.domElement.addEventListener('mousemove', (e) =>
      this.onMouseMove(e),
    );
    this.domElement.addEventListener('mouseup', (e) => this.onMouseUp(e));
  }

  enable(): void {
    this.enabled = true;
    this.domElement.style.cursor = 'crosshair';
  }

  disable(): void {
    this.enabled = false;
    this.dragging = false;
    this.domElement.style.cursor = '';
    this.removeWireframe();
  }

  private onMouseDown(event: MouseEvent): void {
    if (!this.enabled || !event.shiftKey) return;
    this.dragging = true;
    this.startScreen.set(event.clientX, event.clientY);
    this.endScreen.set(event.clientX, event.clientY);
    event.preventDefault();
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.dragging) return;
    this.endScreen.set(event.clientX, event.clientY);
    this.updateWireframe();
  }

  private async onMouseUp(event: MouseEvent): Promise<void> {
    if (!this.dragging) return;
    this.dragging = false;
    this.endScreen.set(event.clientX, event.clientY);

    // Convert screen rectangle to world-space bounding box
    const corners = this.screenRectToWorld();
    if (!corners) {
      this.removeWireframe();
      return;
    }

    const { min, max } = corners;

    try {
      const results = await spatialRegion3D(
        [min.x, min.y, min.z],
        [max.x, max.y, max.z],
      );

      const ids = results.map((r) => r.key);
      this.onSelect(ids, ids.length);
    } catch (err) {
      console.error('Spatial region query failed:', err);
      this.onSelect([], 0);
    }

    // Clean up wireframe after a short delay
    setTimeout(() => this.removeWireframe(), 500);
  }

  /** Project screen rectangle corners into world space at two depths to form a box. */
  private screenRectToWorld(): {
    min: THREE.Vector3;
    max: THREE.Vector3;
  } | null {
    const x1 = Math.min(this.startScreen.x, this.endScreen.x);
    const x2 = Math.max(this.startScreen.x, this.endScreen.x);
    const y1 = Math.min(this.startScreen.y, this.endScreen.y);
    const y2 = Math.max(this.startScreen.y, this.endScreen.y);

    // Require minimum drag distance
    if (Math.abs(x2 - x1) < 10 || Math.abs(y2 - y1) < 10) return null;

    const toNDC = (sx: number, sy: number): THREE.Vector2 =>
      new THREE.Vector2(
        (sx / window.innerWidth) * 2 - 1,
        -(sy / window.innerHeight) * 2 + 1,
      );

    const ndc1 = toNDC(x1, y1);
    const ndc2 = toNDC(x2, y2);

    // Unproject at near and far planes to get world-space extents
    const nearZ = 0.2;
    const farZ = 0.8;

    const corners = [
      new THREE.Vector3(ndc1.x, ndc1.y, nearZ).unproject(this.camera),
      new THREE.Vector3(ndc2.x, ndc2.y, nearZ).unproject(this.camera),
      new THREE.Vector3(ndc1.x, ndc1.y, farZ).unproject(this.camera),
      new THREE.Vector3(ndc2.x, ndc2.y, farZ).unproject(this.camera),
    ];

    const min = corners[0].clone();
    const max = corners[0].clone();
    for (const c of corners) {
      min.min(c);
      max.max(c);
    }

    return { min, max };
  }

  /** Draw or update a wireframe box in the scene. */
  private updateWireframe(): void {
    this.removeWireframe();

    const corners = this.screenRectToWorld();
    if (!corners) return;

    const { min, max } = corners;
    const size = max.clone().sub(min);
    const center = min.clone().add(size.clone().multiplyScalar(0.5));

    const boxGeom = new THREE.BoxGeometry(size.x, size.y, size.z);
    const edges = new THREE.EdgesGeometry(boxGeom);
    const material = new THREE.LineBasicMaterial({
      color: 0x4488ff,
      transparent: true,
      opacity: 0.5,
    });

    this.wireframe = new THREE.LineSegments(edges, material);
    this.wireframe.position.copy(center);
    this.scene.add(this.wireframe);
  }

  private removeWireframe(): void {
    if (this.wireframe) {
      this.scene.remove(this.wireframe);
      this.wireframe.geometry.dispose();
      if (this.wireframe.material instanceof THREE.Material) {
        this.wireframe.material.dispose();
      }
      this.wireframe = null;
    }
  }
}
