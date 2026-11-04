/*****************************************************************************************
 * SGL module: the Sun's corona as seen from close up, for the opening's pass by the Sun. *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#include "fragment.glsl"

// The corona's white light, integrated along each pixel's line of sight from the camera:
// Saito's (1970) electron density, averaged over latitude, scattering sunlight by Thomson
// scattering from a limb-darkened Sun (Billings 1966). All distances are in solar radii.
// A stylized pattern of streamers and filaments goes on top, fixed to the Sun's directions and
// sampled at each sightline's closest point, where most of its light comes from.

in vec2 vs_ndc;

uniform sampler2D loops;          // the loops' summed brightness
uniform vec3 sunView;             // the Sun's center in view space
uniform mat4 inverseProjection;
uniform vec2 viewportOrigin;
uniform mat3 frameToView;         // SGL_Frame axes in view space
uniform vec3 solarPole;           // the Sun's north pole in SGL_Frame
uniform float structure;          // 0 = the smooth computed glow, 1 = the full pattern
uniform float strength;
uniform float fade;

const int Steps = 64;
const float LimbDarkening = 0.63;   // at 550 nm
const float MaxAngle = 1.5359;      // 88 degrees along the sightline
const float Gamma = 0.45;           // eye-like tone curve
// The integral at 1.02 solar radii from 4.1 away, so that brightness there is about 1
const float Normalization = 3.49576;
const float FilamentStrength = 0.8;
const float Pi = 3.14159265358979;

// An eclipse photograph's colour at each display brightness: dark brown, orange, then cream
const int RampSize = 12;
const vec3 Ramp[RampSize] = vec3[RampSize](
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

vec3 ramp(float value) {
  float x = clamp(value, 0.0, 1.0) * float(RampSize - 1);
  int i = min(int(x), RampSize - 2);
  return mix(Ramp[i], Ramp[i + 1], x - float(i));
}

float density(float r) {
  return 2.1012 * pow(r, -16.0) + 0.6084 * pow(r, -6.0) + 0.00502 * pow(r, -2.5);
}

// Thomson scattering by an electron at r; sin2chi is the sine squared of the angle between
// the radial direction and the line of sight
float scattering(float r, float sin2chi) {
  float s = 1.0 / max(r, 1.00001);
  float s2 = s * s;
  float c2 = 1.0 - s2;
  float c = sqrt(c2);
  float lg = log((1.0 + s) / c);
  float A = c * s2;
  float B = -(1.0 - 3.0 * s2 - (c2 / s) * (1.0 + 3.0 * s2) * lg) / 8.0;
  float C = 4.0 / 3.0 - c - c * c2 / 3.0;
  float D = (5.0 + s2 - (c2 / s) * (5.0 - s2) * lg) / 8.0;
  float u = LimbDarkening;
  return 2.0 * ((1.0 - u) * C + u * D) - sin2chi * ((1.0 - u) * A + u * B);
}

float hash(vec3 c) {
  return fract(sin(c.x * 12.9898 + c.y * 78.233 + c.z * 37.719) * 43758.5453);
}

float noise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = x - i;
  vec3 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i), hash(i + vec3(1, 0, 0)), u.x),
        mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), u.x), u.y),
    mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), u.x),
        mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), u.x), u.y),
    u.z
  );
}

// Thin bright threads with dark gaps, zero mean
float ridged(float n) {
  float v = 1.0 - abs(2.0 * n - 1.0);
  return v * v - 0.36;
}

// Value noise at `cycles` per turn around the Sun, drifting slowly with height
float layer(vec3 direction, float cycles, float height) {
  return noise(direction * cycles / (2.0 * Pi) + vec3(0.6, 0.64, 0.48) * height);
}

Fragment getFragment() {
  vec4 h = inverseProjection * vec4(vs_ndc, 1.0, 1.0);
  vec3 ray = normalize(h.xyz / h.w);

  // Closest approach of this sightline to the Sun's center
  float along = dot(ray, sunView);
  vec3 closest = ray * along - sunView;
  float p = length(closest);

  // The globe covers everything inside its limb, blended over a pixel
  float visible = 1.0;
  if (along > 0.0) {
    float pixel = max(fwidth(p), 1e-5);
    visible = smoothstep(1.0 - pixel, 1.0 + pixel, p);
  }
  if (visible <= 0.0) {
    discard;
  }

  // Integrate with s = p tan(a) along the sightline, from the camera outward
  float impact = max(p, 0.5);
  float a0 = atan(-along, impact);
  float da = max(MaxAngle - a0, 0.0) / float(Steps);
  float integral = 0.0;
  for (int i = 0; i < Steps; i++) {
    float a = a0 + (float(i) + 0.5) * da;
    float ca = cos(a);
    float r = impact / ca;
    integral += density(r) * scattering(r, ca * ca) * impact / (ca * ca) * da;
  }

  // The pattern, at the closest point's direction on the Sun. Field lines fan out up to the
  // source surface (2.5 solar radii) and are radial past it, so the height stops there.
  vec3 n = transpose(frameToView) * closest / max(p, 1e-5);
  float sinLat = clamp(dot(n, solarPole), -1.0, 1.0);
  float lat = asin(sinLat);
  float lr = log(clamp(p, 1.0, 2.5));
  float foot = lat + 0.5 * lr * sin(2.0 * lat);
  vec3 toEquator = n - sinLat * solarPole;
  toEquator /= max(length(toEquator), 1e-5);
  vec3 dir = cos(foot) * toEquator + sin(foot) * solarPole;

  float fil = layer(dir, 23.0, lr * 2.0) - 0.5;
  fil += 0.7 * (layer(dir, 47.0, lr * 3.0 + 7.0) - 0.5);
  fil += 0.7 * ridged(layer(dir, 97.0, lr * 4.0 + 14.0));
  fil += 0.5 * ridged(layer(dir, 191.0, lr * 6.0 + 21.0));
  fil += 0.35 * ridged(layer(dir, 383.0, lr * 8.0 + 28.0));
  fil /= 0.3;

  // Streamers: a belt about the equator, broken up in longitude by slow noise, so that some
  // sectors are bright and others open into dark coronal holes
  float slow = 0.6 * noise(dir * 1.3 + 3.1) + 0.4 * noise(dir * 2.9 + 8.3);
  float belt = exp(-pow(foot / 0.5236, 2.0));
  float midLat = abs(foot) - 1.0821;
  float side = 0.5 * exp(-midLat * midLat / 0.02467) * clamp((slow - 0.5) / 0.15, 0.0, 1.0);
  float s = clamp(belt * (0.25 + 1.5 * slow) + side, 0.0, 1.3);

  // Nearly round at the limb, as eclipse photographs show, with the pattern growing with height
  float high = smoothstep(1.0, 1.4, p);
  float shape = mix(0.75 + 0.25 * s, 0.04 + 0.96 * s, high);
  shape = mix(1.0, shape, structure);
  // Far out, sightlines cross many threads, and their contrast averages away
  float threads = pow(1.0 - clamp((p - 1.5) / 0.9, 0.0, 1.0), 1.5);
  float linear = integral / Normalization * shape;
  linear *= exp(structure * FilamentStrength * fil * threads);
  float glow = strength * pow(max(linear, 0.0), Gamma);

  // The loops are brighter where the region is active, threaded by the same filaments
  float loop = texelFetch(loops, ivec2(gl_FragCoord.xy - viewportOrigin), 0).r;
  loop *= mix(1.0, 0.2 + 1.3 * min(s, 1.0), structure) * exp(structure * 0.5 * fil);
  vec3 color = ramp(glow + loop);

  Fragment frag;
  frag.color = vec4(color, visible * fade);
  frag.depth = 0.0;
  frag.gPosition = vec4(0.0, 0.0, 0.0, 1.0);
  frag.gNormal = vec4(0.0, 0.0, 0.0, 1.0);
  return frag;
}
