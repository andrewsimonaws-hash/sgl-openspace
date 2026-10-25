# Solar gravitational lens (SGL) module: the telescope view from the SGL image plane.
# Portfolio project, Andrew Simon. Built against OpenSpace v0.22.0.

include(${PROJECT_SOURCE_DIR}/support/cmake/module_definition.cmake)

set(HEADER_FILES
  sglmodule.h
  rendering/screenspacesgltelescope.h
)
source_group("Header Files" FILES ${HEADER_FILES})

set(SOURCE_FILES
  sglmodule.cpp
  rendering/screenspacesgltelescope.cpp
)
source_group("Source Files" FILES ${SOURCE_FILES})

set(SHADER_FILES
  shaders/telescope_vs.glsl
  shaders/telescope_fs.glsl
)
source_group("Shader Files" FILES ${SHADER_FILES})

create_new_module(
  "Sgl"
  sgl_module
  ${HEADER_FILES} ${SOURCE_FILES} ${SHADER_FILES}
)
