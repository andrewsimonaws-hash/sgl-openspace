/*****************************************************************************************
 * SGL module: the Sun's corona as seen from close up, for the opening's pass by the Sun. *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#version __CONTEXT__

layout(location = 0) in vec4 in_vertex;   // position (solar radii), weight

out Vertex {
  vec3 viewPosition;
  float sigma;          // the strand's half-width, in solar radii
  float peak;           // its brightness along the middle
} vs_out;

uniform vec3 sunView;
uniform mat3 frameToView;
uniform mat4 projection;
uniform float screenHeight;
uniform float strength;

const float BaseSigma = 0.00233;

void main() {
  vec3 view = frameToView * in_vertex.xyz + sunView;
  float height = length(in_vertex.xyz) - 1.0;

  // Strands widen toward their feet, and the dense plasma sits low
  float widen = 1.0 + 3.5 * exp(-height / 0.05);
  float sigma = BaseSigma * widen;
  float peak = strength * 0.1047 * in_vertex.w * (0.25 + exp(-height / 0.13)) / widen;

  // Never thinner than a pixel, keeping the strand's total light so it can't flicker
  float pixel = 2.0 * abs(view.z) / (projection[1][1] * screenHeight);
  float drawn = max(sigma, 0.7 * pixel);

  vs_out.viewPosition = view;
  vs_out.sigma = drawn;
  vs_out.peak = peak * sigma / drawn;
}
