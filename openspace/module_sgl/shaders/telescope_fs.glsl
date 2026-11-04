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
uniform float occulterShade;      // brightness of the occulting disk (0 = black)
uniform float coronaStrength;
uniform float gain;
uniform bool eyepiece;
uniform float resolution;

const float Pi = 3.14159265358979;
const float SolarEquator = 0.35;  // position angle of the solar equator, radians
const int RampSize = 12;
const vec3 CoronaRamp[RampSize] = vec3[RampSize](
  vec3(0.000, 0.000, 0.000),
  vec3(0.158, 0.072, 0.033),
  vec3(0.337, 0.139, 0.045),
  vec3(0.503, 0.222, 0.055),
  vec3(0.633, 0.308, 0.074),
  vec3(0.749, 0.396, 0.100),
  vec3(0.858, 0.490, 0.142),
  vec3(0.929, 0.588, 0.223),
  vec3(0.965, 0.696, 0.328),
  vec3(0.975, 0.813, 0.446),
  vec3(0.984, 0.922, 0.606),
  vec3(0.992, 1.000, 0.937)
);
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

// 1 inside |v| < halfWidth, 0 outside, blended over a pixel so the edge doesn't alias
float band(float v, float halfWidth, float pixel) {
  return 1.0 - smoothstep(halfWidth - 0.5 * pixel, halfWidth + 0.5 * pixel, abs(v));
}

float hash(float i, float j) {
  return fract(sin(i * 12.9898 + j * 78.233) * 43758.5453);
}

// Smooth 2D value noise that repeats every `period` cells in x
float noise(float x, float y, float period) {
  float ix = floor(x);
  float iy = floor(y);
  float ux = fract(x);
  float uy = fract(y);
  ux = ux * ux * (3.0 - 2.0 * ux);
  uy = uy * uy * (3.0 - 2.0 * uy);
  float x0 = mod(ix, period);
  float x1 = mod(ix + 1.0, period);
  return mix(mix(hash(x0, iy), hash(x1, iy), ux),
             mix(hash(x0, iy + 1.0), hash(x1, iy + 1.0), ux), uy);
}

// Thin bright threads with dark gaps, zero mean
float ridged(float n) {
  float v = 1.0 - abs(2.0 * n - 1.0);
  return v * v - 0.36;
}

// Filaments, stretched along the radius: long in ln r, narrow in angle
float filaments(float foot, float lr) {
  float t = foot / (2.0 * Pi);
  float s = noise(t * 23.0, lr * 2.0, 23.0) - 0.5;
  s += 0.7 * (noise(t * 47.0, lr * 3.0 + 7.0, 47.0) - 0.5);
  s += 0.7 * ridged(noise(t * 97.0, lr * 4.0 + 14.0, 97.0));
  s += 0.5 * ridged(noise(t * 191.0, lr * 6.0 + 21.0, 191.0));
  s += 0.35 * ridged(noise(t * 383.0, lr * 8.0 + 28.0, 383.0));
  return s / 0.3;
}

// An eclipse photograph's colour at each display brightness: dark brown, orange, then cream
vec3 coronaRamp(float value) {
  float x = clamp(value, 0.0, 1.0) * float(RampSize - 1);
  int i = min(int(x), RampSize - 2);
  return mix(CoronaRamp[i], CoronaRamp[i + 1], x - float(i));
}

float streamer(float foot, float center, float width, float height) {
  float d = mod(foot - center + Pi, 2.0 * Pi) - Pi;
  return height * exp(-0.5 * d * d / (width * width));
}

// A stylized solar-minimum eclipse corona, for orientation only (the real one outshines
// the ring by about 10^5). Its radial profile is Baumbach's (1937) mean eclipse corona,
// with a gamma so it reads as the eye sees it. The reference renders and the GPU check
// run without it.
vec3 corona(vec2 theta, float r) {
  r = max(r, 1.0);
  float lr = log(r);
  float psi = atan(theta.y, theta.x) - SolarEquator;
  // Follow the field lines: rays fan out over the poles and converge over the equator
  float foot = psi + 0.5 * lr * sin(2.0 * psi);
  // Two helmet streamers and two faint pseudostreamers, narrowing with height
  float s = streamer(foot, 0.0, 0.3665, 1.0);
  s = max(s, streamer(foot, 3.2812, 0.3142, 0.9));
  s = max(s, streamer(foot, 1.0123, 0.1396, 0.35));
  s = max(s, streamer(foot, 4.1539, 0.1571, 0.3));
  float fil = filaments(foot, lr);
  // The r^-17 and r^-7 terms carry the structure; the r^-2.5 dust floor has none
  float k = 2.565 * pow(r, -17.0) * (0.55 + 0.45 * s) +
            1.425 * pow(r, -7.0) * (0.06 + 0.94 * s);
  float linear = (k * exp(0.9 * fil) + 0.0532 * pow(r, -2.5)) / 3.123;
  // A thin bright ring just outside the occulter, barely touched by the filaments
  float rim = 0.25 * exp(-max(r - occulterRadius, 0.0) / 0.014) * (1.0 + 0.12 * tanh(fil));
  return coronaRamp(coronaStrength * pow(linear, 0.45) + rim);
}

vec3 sky(vec2 theta) {
  float r = length(theta) / sunRadius;   // in solar radii
  // The occulting disk covers the sky, its edge blended over a pixel (in solar radii)
  float pixel = 2.0 * fieldOfView / (resolution * sunRadius);
  float covered = band(r, occulterRadius - 0.5 * pixel, pixel);
  vec3 color = vec3(0.0);
  if (covered < 1.0) {
    color = lensedPlanet(theta);
    if (coronaStrength > 0.0) {
      color += corona(theta, r);
    }
  }
  return mix(color, vec3(occulterShade), covered);
}

void main() {
  vec2 p = vs_st * 2.0 - 1.0;              // -1..1 across the view, y up
  float radius = length(p);
  float pixel = 2.0 / resolution;          // one pixel, in p
  if (eyepiece && radius > 1.0 + pixel) {
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

  float alpha = 1.0;
  if (eyepiece) {
    float rim = smoothstep(0.975, 0.985, radius);
    // Tick marks at the four compass points
    float tick = max(band(p.x, 0.004, pixel) * smoothstep(0.88 - pixel, 0.88, abs(p.y)),
                     band(p.y, 0.004, pixel) * smoothstep(0.88 - pixel, 0.88, abs(p.x)));
    color = mix(color, RimColor, max(tick, rim));
    // The rim's outer edge fades out over a pixel instead of being cut off
    alpha = clamp((1.0 - radius) / pixel + 0.5, 0.0, 1.0);
  }
  out_color = vec4(color, alpha);
}
