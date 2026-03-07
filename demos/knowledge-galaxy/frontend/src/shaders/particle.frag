// Particle fragment shader -- Knowledge Galaxy
// Circular particle with soft glow and core brightening

varying vec3 vColor;
varying float vAlpha;

void main() {
    // Distance from center of point sprite (0..1 range)
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    // Discard pixels outside the circle
    if (dist > 0.5) discard;

    // Soft outer glow
    float glow = 1.0 - smoothstep(0.0, 0.5, dist);

    // Core brightening -- inner 20% gets extra brightness
    float core = smoothstep(0.2, 0.0, dist) * 0.6;

    vec3 color = vColor * (glow + core) + vec3(core * 0.3);
    float alpha = glow * vAlpha;

    gl_FragColor = vec4(color, alpha);
}
