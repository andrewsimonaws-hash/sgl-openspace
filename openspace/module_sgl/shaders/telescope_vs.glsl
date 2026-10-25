/*****************************************************************************************
 * SGL module: the view from a telescope in the solar gravitational lens image plane.    *
 * Portfolio project, Andrew Simon. Built against OpenSpace v0.22.0 (MIT license).       *
 ****************************************************************************************/

#version __CONTEXT__

// One triangle that covers the whole target; vs_st runs 0..1 across the visible square.
out vec2 vs_st;

void main() {
  vec2 corner = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vs_st = corner;
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
