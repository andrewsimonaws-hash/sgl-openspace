/*****************************************************************************************
 * SGL module: the Sun's corona as seen from close up, for the opening's pass by the Sun. *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#include <modules/sgl/rendering/renderablesglcorona.h>

#include <openspace/engine/globals.h>
#include <openspace/rendering/renderengine.h>
#include <openspace/util/updatestructures.h>
#include <ghoul/filesystem/filesystem.h>
#include <ghoul/glm.h>
#include <ghoul/logging/logmanager.h>
#include <ghoul/misc/dictionary.h>
#include <ghoul/opengl/programobject.h>
#include <ghoul/opengl/textureunit.h>
#include <array>
#include <format>
#include <fstream>
#include <optional>
#include <vector>

namespace {
    constexpr std::string_view _loggerCat = "RenderableSglCorona";
    constexpr double SolarRadius = 6.957e8;    // metres, as the Sun's globe is drawn

    using namespace openspace;

    constexpr Property::PropertyInfo GlowStrengthInfo = {
        "GlowStrength",
        "Glow strength",
        "The display brightness of the smooth glow, just above the Sun's surface. The real "
        "corona is about a million times fainter than the disk, so this is an exposure "
        "choice.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo LoopStrengthInfo = {
        "LoopStrength",
        "Loop strength",
        "The brightness of the coronal loops, relative to the smooth glow.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo StructureInfo = {
        "Structure",
        "Structure",
        "How much of the stylized streamers and filaments to show: 0 gives the smooth, "
        "computed glow, and 1 the full pattern.",
        Property::Visibility::User
    };

    // The corona around the Sun for a close pass. A glow is integrated along each pixel's
    // line of sight from the camera, using a measured electron density and Thomson
    // scattering, then given a stylized pattern of streamers and filaments fixed to the
    // Sun. Coronal loops are drawn as 3D curves over it. All of it comes from the real
    // camera and Sun positions, so the glow meets the Sun's limb from any viewpoint.
    struct [[codegen::Dictionary(RenderableSglCorona)]] Parameters {
        // float32: the solar pole (x, y, z, 0), then vertex pairs for the coronal loops,
        // each x, y, z (solar radii) and weight
        std::filesystem::path loopFile;

        // [[codegen::verbatim(GlowStrengthInfo.description)]]
        std::optional<float> glowStrength;

        // [[codegen::verbatim(LoopStrengthInfo.description)]]
        std::optional<float> loopStrength;

        // [[codegen::verbatim(StructureInfo.description)]]
        std::optional<float> structure;
    };
#include "renderablesglcorona_codegen.cpp"
} // namespace

namespace openspace {

Documentation RenderableSglCorona::Documentation() {
    return codegen::doc<Parameters>("sgl_renderable_corona");
}

RenderableSglCorona::RenderableSglCorona(const ghoul::Dictionary& dictionary)
    : Renderable(dictionary, { .automaticallyUpdateRenderBin = false })
    , _glowStrength(GlowStrengthInfo, 0.8f, 0.f, 2.f)
    , _loopStrength(LoopStrengthInfo, 1.f, 0.f, 5.f)
    , _structure(StructureInfo, 0.9f, 0.f, 1.f)
{
    const Parameters p = codegen::bake<Parameters>(dictionary);
    _loopFile = p.loopFile;
    _glowStrength = p.glowStrength.value_or(_glowStrength);
    _loopStrength = p.loopStrength.value_or(_loopStrength);
    _structure = p.structure.value_or(_structure);
    addProperty(Fadeable::_opacity);
    addProperty(_glowStrength);
    addProperty(_loopStrength);
    addProperty(_structure);

    setBoundingSphere(5.0 * SolarRadius);
    // Blended over the deferred scene, like the additive plane this replaces
    setRenderBin(RenderBin::PreDeferredTransparent);
}

void RenderableSglCorona::initializeGL() {
    _loopProgram = ghoul::opengl::ProgramObject::Build(
        "SglCoronaLoops",
        absPath("${MODULE_SGL}/shaders/corona_loop_vs.glsl"),
        absPath("${MODULE_SGL}/shaders/corona_loop_fs.glsl"),
        absPath("${MODULE_SGL}/shaders/corona_loop_gs.glsl")
    );
    _glowProgram = global::renderEngine->buildRenderProgram(
        "SglCoronaGlow",
        absPath("${MODULE_SGL}/shaders/corona_glow_vs.glsl"),
        absPath("${MODULE_SGL}/shaders/corona_glow_fs.glsl")
    );

    std::ifstream file(absPath(_loopFile), std::ios::binary | std::ios::ate);
    if (!file) {
        LERROR(std::format("Could not open loop file '{}'", _loopFile));
    }
    const std::streamsize size = file ? static_cast<std::streamsize>(file.tellg()) : 0;
    std::vector<float> vertices(static_cast<size_t>(size) / sizeof(float));
    if (file) {
        file.seekg(0);
        file.read(
            reinterpret_cast<char*>(vertices.data()),
            static_cast<std::streamsize>(vertices.size() * sizeof(float))
        );
    }
    if (vertices.size() >= 4) {
        _solarPole = glm::vec3(vertices[0], vertices[1], vertices[2]);
        vertices.erase(vertices.begin(), vertices.begin() + 4);
    }
    _nVertices = static_cast<int>(vertices.size() / 4);

    glCreateVertexArrays(1, &_emptyVao);
    glCreateVertexArrays(1, &_loopVao);
    glCreateBuffers(1, &_loopVbo);
    glNamedBufferStorage(
        _loopVbo,
        vertices.size() * sizeof(float),
        vertices.empty() ? nullptr : vertices.data(),
        GL_NONE_BIT
    );
    glVertexArrayVertexBuffer(_loopVao, 0, _loopVbo, 0, 4 * sizeof(float));
    glEnableVertexArrayAttrib(_loopVao, 0);
    glVertexArrayAttribFormat(_loopVao, 0, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_loopVao, 0, 0);

    glCreateFramebuffers(1, &_loopFbo);
}

void RenderableSglCorona::deinitializeGL() {
    glDeleteFramebuffers(1, &_loopFbo);
    glDeleteTextures(1, &_loopTexture);
    glDeleteBuffers(1, &_loopVbo);
    glDeleteVertexArrays(1, &_loopVao);
    glDeleteVertexArrays(1, &_emptyVao);
    _loopTexture = 0;
    _loopProgram = nullptr;
    if (_glowProgram) {
        global::renderEngine->removeRenderProgram(_glowProgram.get());
        _glowProgram = nullptr;
    }
}

void RenderableSglCorona::resizeLoopBuffer(int width, int height) {
    if (width == _bufferWidth && height == _bufferHeight) {
        return;
    }
    glDeleteTextures(1, &_loopTexture);
    glCreateTextures(GL_TEXTURE_2D, 1, &_loopTexture);
    glTextureStorage2D(_loopTexture, 1, GL_R16F, width, height);
    glTextureParameteri(_loopTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(_loopTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glNamedFramebufferTexture(_loopFbo, GL_COLOR_ATTACHMENT0, _loopTexture, 0);
    glNamedFramebufferDrawBuffer(_loopFbo, GL_COLOR_ATTACHMENT0);
    _bufferWidth = width;
    _bufferHeight = height;
}

void RenderableSglCorona::render(const RenderData& data, RendererTasks&) {
    const float fade = opacity();
    if (fade <= 0.f || !_loopProgram || !_glowProgram) {
        return;
    }

    // The Sun and the frame's axes in view space, in solar radii. The camera's view scale
    // is divided out: it changes nothing on screen, and these go to single-precision floats.
    const glm::dmat4 view = data.camera.combinedViewMatrix();
    const double viewScale = glm::length(glm::dvec3(view[0]));
    const glm::dvec3 sunView = glm::dvec3(
        view * glm::dvec4(data.modelTransform.translation, 1.0)
    ) / (viewScale * SolarRadius);
    const glm::dmat3 frameToView =
        glm::dmat3(view) * glm::dmat3(data.modelTransform.rotation) / viewScale;
    const glm::mat4 projection = glm::mat4(data.camera.projectionMatrix());

    GLint previousFbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFbo);
    std::array<GLint, 4> viewport;
    glGetIntegerv(GL_VIEWPORT, viewport.data());
    resizeLoopBuffer(viewport[2], viewport[3]);

    // First the loops' summed brightness, offscreen: the colour ramp below has to apply to
    // the total, so that overlapping strands climb from brown through orange to cream.
    GLboolean depthTest = GL_FALSE;
    glGetBooleanv(GL_DEPTH_TEST, &depthTest);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);

    const std::array<GLfloat, 4> black = { 0.f, 0.f, 0.f, 0.f };
    glClearNamedFramebufferfv(_loopFbo, GL_COLOR, 0, black.data());
    glBindFramebuffer(GL_FRAMEBUFFER, _loopFbo);
    glViewport(0, 0, viewport[2], viewport[3]);
    glBlendFunc(GL_ONE, GL_ONE);

    _loopProgram->activate();
    _loopProgram->setUniform("sunView", glm::vec3(sunView));
    _loopProgram->setUniform("frameToView", glm::mat3(frameToView));
    _loopProgram->setUniform("projection", projection);
    _loopProgram->setUniform("screenHeight", static_cast<float>(viewport[3]));
    _loopProgram->setUniform("strength", _loopStrength.value());
    glBindVertexArray(_loopVao);
    glDrawArrays(GL_LINES, 0, _nVertices);
    _loopProgram->deactivate();

    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previousFbo));
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

    // Then the glow over the whole view, with the loops added and the ramp applied
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    _glowProgram->activate();
    ghoul::opengl::TextureUnit unit;
    unit.bind(_loopTexture);
    _glowProgram->setUniform("loops", unit);
    _glowProgram->setUniform("sunView", glm::vec3(sunView));
    _glowProgram->setUniform("inverseProjection", glm::inverse(projection));
    _glowProgram->setUniform(
        "viewportOrigin",
        glm::vec2(static_cast<float>(viewport[0]), static_cast<float>(viewport[1]))
    );
    _glowProgram->setUniform("frameToView", glm::mat3(frameToView));
    _glowProgram->setUniform("solarPole", _solarPole);
    _glowProgram->setUniform("structure", _structure.value());
    _glowProgram->setUniform("strength", _glowStrength.value());
    _glowProgram->setUniform("fade", fade);
    glBindVertexArray(_emptyVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    _glowProgram->deactivate();

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);
    if (depthTest) {
        glEnable(GL_DEPTH_TEST);
    }
}

} // namespace openspace
