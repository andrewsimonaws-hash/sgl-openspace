/*****************************************************************************************
 * SGL module: the Sun's corona as seen from close up, for the opening's pass by the Sun. *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#ifndef __OPENSPACE_MODULE_SGL___RENDERABLESGLCORONA___H__
#define __OPENSPACE_MODULE_SGL___RENDERABLESGLCORONA___H__

#include <openspace/rendering/renderable.h>

#include <openspace/documentation/documentation.h>
#include <openspace/properties/scalar/floatproperty.h>
#include <ghoul/glm.h>
#include <ghoul/opengl/ghoul_gl.h>
#include <filesystem>
#include <memory>

namespace ghoul::opengl { class ProgramObject; }

namespace openspace {

class RenderableSglCorona : public Renderable {
public:
    explicit RenderableSglCorona(const ghoul::Dictionary& dictionary);

    void initializeGL() override;
    void deinitializeGL() override;
    void render(const RenderData& data, RendererTasks& rendererTask) override;

    static openspace::Documentation Documentation();

private:
    void resizeLoopBuffer(int width, int height);

    std::filesystem::path _loopFile;
    FloatProperty _glowStrength;
    FloatProperty _loopStrength;
    FloatProperty _structure;

    std::unique_ptr<ghoul::opengl::ProgramObject> _loopProgram;
    std::unique_ptr<ghoul::opengl::ProgramObject> _glowProgram;
    GLuint _loopVao = 0;
    GLuint _loopVbo = 0;
    GLuint _emptyVao = 0;
    GLuint _loopFbo = 0;
    GLuint _loopTexture = 0;
    glm::vec3 _solarPole = glm::vec3(0.f, 0.f, 1.f);
    int _nVertices = 0;
    int _bufferWidth = 0;
    int _bufferHeight = 0;
};

} // namespace openspace

#endif // __OPENSPACE_MODULE_SGL___RENDERABLESGLCORONA___H__
