# Solar gravitational lens (SGL) module: the telescope view from the SGL image plane.
# Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.
# Built against OpenSpace v0.22.0.

include(${PROJECT_SOURCE_DIR}/support/cmake/module_definition.cmake)

set(HEADER_FILES
  sglmodule.h
  rendering/screenspacesgltelescope.h
  rendering/renderablesglcorona.h
)
source_group("Header Files" FILES ${HEADER_FILES})

set(SOURCE_FILES
  sglmodule.cpp
  rendering/screenspacesgltelescope.cpp
  rendering/renderablesglcorona.cpp
)
source_group("Source Files" FILES ${SOURCE_FILES})

set(SHADER_FILES
  shaders/telescope_vs.glsl
  shaders/telescope_fs.glsl
  shaders/corona_glow_vs.glsl
  shaders/corona_glow_fs.glsl
  shaders/corona_loop_vs.glsl
  shaders/corona_loop_gs.glsl
  shaders/corona_loop_fs.glsl
)
source_group("Shader Files" FILES ${SHADER_FILES})

create_new_module(
  "Sgl"
  sgl_module
  ${HEADER_FILES} ${SOURCE_FILES} ${SHADER_FILES}
)
