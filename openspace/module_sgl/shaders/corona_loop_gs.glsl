/*****************************************************************************************
 * SGL module: the Sun's corona as seen from close up, for the opening's pass by the Sun. *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#version __CONTEXT__

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

in Vertex {
  vec3 viewPosition;
  float sigma;
  float peak;
} gs_in[];

out float gs_offset;    // across the strand, in widths (sigma)
out float gs_peak;

uniform mat4 projection;

const float Extent = 3.0;   // widths out to where the strand has faded to nothing

void main() {
  vec3 p0 = gs_in[0].viewPosition;
  vec3 p1 = gs_in[1].viewPosition;
  // Across the segment and facing the camera, so a strand keeps its width from any angle
  vec3 across = cross(p1 - p0, p0 + p1);
  float len = length(across);
  if (len <= 0.0) {
    return;
  }
  across /= len;

  for (int i = 0; i < 2; i++) {
    vec3 p = gs_in[i].viewPosition;
    float reach = Extent * gs_in[i].sigma;
    for (int side = -1; side <= 1; side += 2) {
      gs_offset = float(side) * Extent;
      gs_peak = gs_in[i].peak;
      gl_Position = projection * vec4(p + float(side) * reach * across, 1.0);
      EmitVertex();
    }
  }
  EndPrimitive();
}
