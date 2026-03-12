import * as THREE from 'three';

const CONNECTION_VERTEX = /* glsl */ `
attribute vec3 aLineColor;
attribute float aAlpha;

uniform float uTime;
uniform float uDashOffset;

varying vec3 vLineColor;
varying float vLineAlpha;
varying float vDash;

void main() {
    vLineColor = aLineColor;
    vLineAlpha = aAlpha;

    // Compute a dash pattern based on position along the line
    vDash = position.x * 0.05 + uTime * 2.0 + uDashOffset;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const CONNECTION_FRAGMENT = /* glsl */ `
varying vec3 vLineColor;
varying float vLineAlpha;
varying float vDash;

void main() {
    // Animated dash pattern
    float dash = smoothstep(0.3, 0.5, fract(vDash));
    float alpha = vLineAlpha * (0.3 + 0.7 * dash);

    gl_FragColor = vec4(vLineColor, alpha);
}
`;

export class ConnectionRenderer {
  private group: THREE.Group;
  private material: THREE.ShaderMaterial;

  constructor() {
    this.group = new THREE.Group();

    this.material = new THREE.ShaderMaterial({
      vertexShader: CONNECTION_VERTEX,
      fragmentShader: CONNECTION_FRAGMENT,
      uniforms: {
        uTime: { value: 0 },
        uDashOffset: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
  }

  /** Draw edges from a source position to an array of target positions. */
  showEdges(
    sourcePos: THREE.Vector3,
    targetPositions: THREE.Vector3[],
    color: THREE.Color,
  ): void {
    this.clear();

    if (targetPositions.length === 0) return;

    const segmentCount = targetPositions.length;
    const positions = new Float32Array(segmentCount * 6); // 2 vertices per segment, 3 components each
    const colors = new Float32Array(segmentCount * 6);
    const alphas = new Float32Array(segmentCount * 2);

    for (let i = 0; i < segmentCount; i++) {
      const target = targetPositions[i];
      const base = i * 6;

      // Source vertex
      positions[base] = sourcePos.x;
      positions[base + 1] = sourcePos.y;
      positions[base + 2] = sourcePos.z;

      // Target vertex
      positions[base + 3] = target.x;
      positions[base + 4] = target.y;
      positions[base + 5] = target.z;

      // Colors for both vertices
      colors[base] = color.r;
      colors[base + 1] = color.g;
      colors[base + 2] = color.b;
      colors[base + 3] = color.r * 0.5;
      colors[base + 4] = color.g * 0.5;
      colors[base + 5] = color.b * 0.5;

      // Alpha: bright at source, dimmer at target
      alphas[i * 2] = 0.8;
      alphas[i * 2 + 1] = 0.3;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(positions, 3),
    );
    geometry.setAttribute(
      'aLineColor',
      new THREE.BufferAttribute(colors, 3),
    );
    geometry.setAttribute(
      'aAlpha',
      new THREE.BufferAttribute(alphas, 1),
    );

    const lines = new THREE.LineSegments(geometry, this.material.clone());
    this.group.add(lines);
  }

  /** Remove all edges. */
  clear(): void {
    while (this.group.children.length > 0) {
      const child = this.group.children[0];
      if (child instanceof THREE.LineSegments) {
        child.geometry.dispose();
        if (child.material instanceof THREE.Material) {
          child.material.dispose();
        }
      }
      this.group.remove(child);
    }
  }

  /** Update time uniform for dash animation. */
  update(time: number): void {
    for (const child of this.group.children) {
      if (child instanceof THREE.LineSegments) {
        const mat = child.material as THREE.ShaderMaterial;
        mat.uniforms['uTime'].value = time;
      }
    }
  }

  /** Get the group to add to the scene. */
  getObject(): THREE.Group {
    return this.group;
  }
}
