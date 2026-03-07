// Particle vertex shader -- Knowledge Galaxy
// Attributes: aColor, aSize, aPhase
// Uniforms: uTime, uPixelRatio
// Varyings: vColor, vAlpha

attribute vec3 aColor;
attribute float aSize;
attribute float aPhase;

uniform float uTime;
uniform float uPixelRatio;

varying vec3 vColor;
varying float vAlpha;

void main() {
    vColor = aColor;

    // Pulse animation driven by per-particle phase offset
    float pulse = 1.0 + 0.15 * sin(uTime * 1.5 + aPhase * 6.2831);

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    // Perspective size scaling
    float dist = -mvPosition.z;
    gl_PointSize = aSize * pulse * uPixelRatio * (300.0 / dist);

    // Distance-based alpha fade
    float fade = smoothstep(4000.0, 500.0, dist);
    vAlpha = fade * (0.7 + 0.3 * pulse);

    gl_Position = projectionMatrix * mvPosition;
}
