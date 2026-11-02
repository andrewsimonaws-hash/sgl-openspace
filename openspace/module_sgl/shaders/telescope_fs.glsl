/*****************************************************************************************
 * SGL module: the view from a telescope in the solar gravitational lens image plane.    *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#version __CONTEXT__

// Port of layer_c_proto/ring_view.py (lensed_planet + render). All angles are
// in units of the Einstein angle theta_E, so values stay near 1.

in vec2 vs_st;
out vec4 out_color;

uniform sampler2D planetTexture;  // image-plane texture: x right, y up, disk fills it
uniform vec2 telescopeOffset;     // telescope position, in planet-image radii
uniform float kappa;              // drawn planet radius, in theta_E
uniform float fieldOfView;        // half-width of the view, in theta_E
uniform float sunRadius;          // solar radius, in theta_E
uniform float occulterRadius;     // coronagraph radius, in solar radii
uniform float coronaStrength;
uniform float gain;
uniform bool eyepiece;
uniform float resolution;

const vec3 CoronaColor = vec3(1.0, 0.82, 0.6);
const vec3 RimColor = vec3(0.78);

vec3 lensedPlanet(vec2 theta) {
  // Point-mass lens equation: where did the ray arriving from direction theta come from?
  vec2 beta = theta - theta / dot(theta, theta);
  // Position on the planet's disk, measured from its apparent centre
  vec2 q = (beta - kappa * telescopeOffset) / kappa;
  if (dot(q, q) > 1.0) {
    return vec3(0.0);
  }
  // The image plane holds the planet inverted: sky offset q is image-plane point -q
  vec2 uv = (-q + 1.0) * 0.5;
  return texture(planetTexture, uv).rgb;
}

vec3 sky(vec2 theta) {
  float r = length(theta) / sunRadius;   // in solar radii
  if (r <= occulterRadius) {
    return vec3(0.0);
  }
  vec3 color = lensedPlanet(theta);
  if (coronaStrength > 0.0) {
    color += coronaStrength * pow(max(r, 1.0), -3.0) * CoronaColor;
  }
  return color;
}

void main() {
  vec2 p = vs_st * 2.0 - 1.0;              // -1..1 across the view, y up
  float radius = length(p);
  if (eyepiece && radius > 1.0) {
    out_color = vec4(0.0);
    return;
  }

  // 2x2 supersampling, as in the Python reference
  vec3 color = vec3(0.0);
  float px = 1.0 / resolution;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      vec2 sub = vec2(float(i) - 0.5, float(j) - 0.5) * px;
      color += sky((p + sub) * fieldOfView);
    }
  }
  color = clamp(color * 0.25 * gain, 0.0, 1.0);

  if (eyepiece) {
    float rim = smoothstep(0.975, 0.985, radius);
    bool tick = (abs(p.x) < 0.004 && abs(p.y) > 0.88) ||
                (abs(p.y) < 0.004 && abs(p.x) > 0.88);
    if (tick) {
      color = RimColor;
    }
    color = mix(color, RimColor, rim);
  }
  out_color = vec4(color, 1.0);
}
