import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import type { ParticleSystem } from './particles.ts';

const STAR_COUNT = 2000;
const STAR_SPREAD = 3000;

export class Galaxy {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private composer: EffectComposer;
  private clock: THREE.Clock;
  private particleSystems: ParticleSystem[] = [];
  private animationCallbacks: Array<(time: number) => void> = [];
  private fpsFrames = 0;
  private fpsLastTime = 0;
  private currentFps = 0;

  constructor() {
    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    document.body.appendChild(this.renderer.domElement);

    // Scene
    this.scene = new THREE.Scene();

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      5000,
    );
    this.camera.position.set(0, 0, 800);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.5;
    this.controls.minDistance = 50;
    this.controls.maxDistance = 3000;

    // Post-processing
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      1.5, // strength
      0.8, // radius
      0.2, // threshold
    );
    this.composer.addPass(bloomPass);

    // Star field background
    this.createStarField();

    // Clock
    this.clock = new THREE.Clock();
    this.fpsLastTime = performance.now();

    // Window resize
    window.addEventListener('resize', () => this.resize());

    // Start render loop
    this.animate();
  }

  private createStarField(): void {
    const positions = new Float32Array(STAR_COUNT * 3);
    for (let i = 0; i < STAR_COUNT; i++) {
      positions[i * 3] = (Math.random() - 0.5) * STAR_SPREAD * 2;
      positions[i * 3 + 1] = (Math.random() - 0.5) * STAR_SPREAD * 2;
      positions[i * 3 + 2] = (Math.random() - 0.5) * STAR_SPREAD * 2;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(positions, 3),
    );

    const material = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 1.5,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const stars = new THREE.Points(geometry, material);
    this.scene.add(stars);
  }

  private animate(): void {
    requestAnimationFrame(() => this.animate());

    const elapsed = this.clock.getElapsedTime();

    this.controls.update();

    // Update particle systems
    for (const ps of this.particleSystems) {
      ps.update(elapsed);
    }

    // Run registered callbacks
    for (const cb of this.animationCallbacks) {
      cb(elapsed);
    }

    this.composer.render();

    // FPS counter
    this.fpsFrames++;
    const now = performance.now();
    if (now - this.fpsLastTime >= 1000) {
      this.currentFps = this.fpsFrames;
      this.fpsFrames = 0;
      this.fpsLastTime = now;
    }
  }

  /** Add a particle system to the scene and animation loop. */
  addParticles(system: ParticleSystem): void {
    this.particleSystems.push(system);
    this.scene.add(system.getObject());
  }

  /** Register a callback to run each frame. */
  onAnimate(callback: (time: number) => void): void {
    this.animationCallbacks.push(callback);
  }

  /** Smoothly fly the camera to a target position. */
  flyTo(x: number, y: number, z: number): void {
    const target = new THREE.Vector3(x, y, z);
    const cameraOffset = target
      .clone()
      .add(new THREE.Vector3(0, 0, 200));

    // Disable auto-rotate during fly
    this.controls.autoRotate = false;

    const startPos = this.camera.position.clone();
    const startTarget = this.controls.target.clone();
    const startTime = performance.now();
    const duration = 1500; // ms

    const flyStep = (): void => {
      const t = Math.min(1, (performance.now() - startTime) / duration);
      const ease = t * t * (3 - 2 * t); // smoothstep

      this.camera.position.lerpVectors(startPos, cameraOffset, ease);
      this.controls.target.lerpVectors(startTarget, target, ease);
      this.controls.update();

      if (t < 1) {
        requestAnimationFrame(flyStep);
      } else {
        this.controls.autoRotate = true;
      }
    };

    requestAnimationFrame(flyStep);
  }

  /** Handle window resize. */
  resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
  }

  getCamera(): THREE.PerspectiveCamera {
    return this.camera;
  }

  getScene(): THREE.Scene {
    return this.scene;
  }

  getRenderer(): THREE.WebGLRenderer {
    return this.renderer;
  }

  getControls(): OrbitControls {
    return this.controls;
  }

  getFps(): number {
    return this.currentFps;
  }
}
