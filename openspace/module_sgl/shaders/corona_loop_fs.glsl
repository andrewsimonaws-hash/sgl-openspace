/*****************************************************************************************
 * SGL module: the Sun's corona as seen from close up, for the opening's pass by the Sun. *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#version __CONTEXT__

in float gs_offset;
in float gs_peak;

out float out_value;

void main() {
  out_value = gs_peak * exp(-0.5 * gs_offset * gs_offset);
}
