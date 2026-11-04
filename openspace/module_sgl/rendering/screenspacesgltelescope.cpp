/*****************************************************************************************
 * SGL module: the view from a telescope in the solar gravitational lens image plane.    *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#include <modules/sgl/rendering/screenspacesgltelescope.h>

#include <openspace/documentation/documentation.h>
#include <openspace/engine/globals.h>
#include <openspace/rendering/renderengine.h>
#include <openspace/scene/scene.h>
#include <openspace/scene/scenegraphnode.h>
#include <ghoul/filesystem/filesystem.h>
#include <ghoul/io/texture/texturereader.h>
#include <ghoul/io/texture/texturewriter.h>
#include <ghoul/logging/logmanager.h>
#include <ghoul/misc/dictionary.h>
#include <ghoul/opengl/framebufferobject.h>
#include <ghoul/opengl/programobject.h>
#include <ghoul/opengl/texture.h>
#include <ghoul/opengl/textureunit.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <optional>

namespace {
    constexpr std::string_view _loggerCat = "ScreenSpaceSglTelescope";

    // 2GM/c^2 for the Sun, and the solar radius, in metres
    constexpr double SchwarzschildRadius = 2953.34;
    constexpr double SolarRadius = 6.957e8;
    constexpr double Parsec = 3.0856775814913673e16;

    // Drawn planet radius in units of the Einstein angle, for the three planet sizes
    constexpr std::array<double, 3> Kappas = { 0.02, 0.05, 0.12 };

    using namespace openspace;

    constexpr Property::PropertyInfo PlanetTextureInfo = {
        "PlanetTexture",
        "Planet texture",
        "The planet as it appears in the image plane (x right, y up), with the disk "
        "filling the image.",
        Property::Visibility::AdvancedUser
    };

    constexpr Property::PropertyInfo TelescopeNodeInfo = {
        "TelescopeNode",
        "Telescope node",
        "The scene graph node of the spacecraft whose view is shown. Its position in the "
        "image plane sets where the planet appears relative to the Sun. If empty, the "
        "manual offset is used instead.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo ImagePlaneNodeInfo = {
        "ImagePlaneNode",
        "Image plane node",
        "The scene graph node at the centre of the planet's image, whose local x and y "
        "axes span the image plane.",
        Property::Visibility::AdvancedUser
    };

    constexpr Property::PropertyInfo SunNodeInfo = {
        "SunNode",
        "Sun node",
        "The scene graph node at the centre of the Sun. The distance from it to the "
        "image plane sets the Einstein angle and the Sun's apparent size.",
        Property::Visibility::AdvancedUser
    };

    constexpr Property::PropertyInfo ManualOffsetInfo = {
        "ManualOffset",
        "Manual telescope offset",
        "Telescope position in the image plane, in units of the planet image's radius, "
        "used when no telescope node is set.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo TelescopeOffsetInfo = {
        "TelescopeOffset",
        "Telescope offset",
        "The telescope position currently used, in units of the planet image's radius.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo PlanetSizeInfo = {
        "PlanetSize",
        "Planet size",
        "How much the planet is magnified so it can be seen. The real planet is about "
        "three millionths of an arcsecond across, far below what any telescope resolves.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo MagnificationInfo = {
        "Magnification",
        "Magnification",
        "How many times larger than real the planet is drawn at the current planet size.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo PlanetRadiusInfo = {
        "PlanetRadius",
        "Planet radius",
        "The target planet's radius in metres.",
        Property::Visibility::AdvancedUser
    };

    constexpr Property::PropertyInfo TargetDistanceInfo = {
        "TargetDistance",
        "Target distance",
        "The distance from the Sun to the target planet in metres.",
        Property::Visibility::AdvancedUser
    };

    constexpr Property::PropertyInfo FieldOfViewInfo = {
        "FieldOfView",
        "Field of view",
        "Half the width of the view, in units of the Einstein ring's radius.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo OcculterRadiusInfo = {
        "OcculterRadius",
        "Occulter radius",
        "Radius of the coronagraph occulter in solar radii. The Einstein ring sits at "
        "about 1.09 solar radii at 650 AU, so the occulter must stay below that.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo OcculterShadeInfo = {
        "OcculterShade",
        "Occulter shade",
        "Brightness of the coronagraph's occulting disk, so it shows against the sky. "
        "0 draws it black, as the reference renders do.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo CoronaStrengthInfo = {
        "CoronaStrength",
        "Corona strength",
        "Brightness of a stylized eclipse-like corona around the occulter: a glow at the "
        "limb and fading rays. The real corona is far brighter than the ring; this is "
        "for orientation only.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo GainInfo = {
        "Gain",
        "Display gain",
        "Multiplies the planet texture's brightness. The default of 1 shows a texture "
        "that already carries its display gain as it is.",
        Property::Visibility::AdvancedUser
    };

    constexpr Property::PropertyInfo EyepieceInfo = {
        "Eyepiece",
        "Eyepiece mask",
        "Draws the view in a circular eyepiece with a rim and reticle ticks.",
        Property::Visibility::User
    };

    constexpr Property::PropertyInfo ResolutionInfo = {
        "Resolution",
        "Resolution",
        "Size in pixels of the square texture the view is drawn into.",
        Property::Visibility::AdvancedUser
    };

    constexpr Property::PropertyInfo SaveFramePathInfo = {
        "SaveFramePath",
        "Save frame path",
        "Where 'Save frame' writes the current view as a PNG.",
        Property::Visibility::Developer
    };

    constexpr Property::PropertyInfo SaveFrameInfo = {
        "SaveFrame",
        "Save frame",
        "Writes the current view to the save-frame path, for checking against the "
        "Python reference.",
        Property::Visibility::Developer
    };

    // The view from a telescope in the solar gravitational lens image plane: the target
    // planet lensed into an Einstein ring around a coronagraph occulter.
    struct [[codegen::Dictionary(ScreenSpaceSglTelescope)]] Parameters {
        std::optional<std::string> identifier [[codegen::identifier()]];

        // [[codegen::verbatim(PlanetTextureInfo.description)]]
        std::filesystem::path planetTexture;

        // [[codegen::verbatim(TelescopeNodeInfo.description)]]
        std::optional<std::string> telescopeNode;

        // [[codegen::verbatim(ImagePlaneNodeInfo.description)]]
        std::string imagePlaneNode [[codegen::identifier()]];

        // [[codegen::verbatim(SunNodeInfo.description)]]
        std::optional<std::string> sunNode [[codegen::identifier()]];

        // [[codegen::verbatim(ManualOffsetInfo.description)]]
        std::optional<glm::vec2> manualOffset;

        // [[codegen::verbatim(PlanetSizeInfo.description)]] 0 = thin, 1 = medium,
        // 2 = bold.
        std::optional<int> planetSize [[codegen::inrange(0, 2)]];

        // [[codegen::verbatim(PlanetRadiusInfo.description)]]
        std::optional<double> planetRadius [[codegen::greater(0.0)]];

        // [[codegen::verbatim(TargetDistanceInfo.description)]]
        std::optional<double> targetDistance [[codegen::greater(0.0)]];

        // [[codegen::verbatim(FieldOfViewInfo.description)]]
        std::optional<float> fieldOfView [[codegen::inrange(1.05f, 4.f)]];

        // [[codegen::verbatim(OcculterRadiusInfo.description)]]
        std::optional<float> occulterRadius [[codegen::inrange(0.f, 1.08f)]];

        // [[codegen::verbatim(OcculterShadeInfo.description)]]
        std::optional<float> occulterShade [[codegen::inrange(0.f, 1.f)]];

        // [[codegen::verbatim(CoronaStrengthInfo.description)]]
        std::optional<float> coronaStrength [[codegen::greaterequal(0.f)]];

        // [[codegen::verbatim(GainInfo.description)]]
        std::optional<float> gain [[codegen::greater(0.f)]];

        // [[codegen::verbatim(EyepieceInfo.description)]]
        std::optional<bool> eyepiece;

        // [[codegen::verbatim(ResolutionInfo.description)]]
        std::optional<int> resolution [[codegen::inrange(128, 4096)]];
    };
} // namespace
#include "screenspacesgltelescope_codegen.cpp"

namespace openspace {

Documentation ScreenSpaceSglTelescope::Documentation() {
    return codegen::doc<Parameters>(
        "sgl_screenspace_telescope",
        ScreenSpaceRenderable::Documentation()
    );
}

ScreenSpaceSglTelescope::ScreenSpaceSglTelescope(const ghoul::Dictionary& dictionary)
    : ScreenSpaceRenderable(dictionary)
    , _planetTexturePath(PlanetTextureInfo)
    , _telescopeNode(TelescopeNodeInfo)
    , _imagePlaneNode(ImagePlaneNodeInfo)
    , _sunNode(SunNodeInfo, "SunCenter")
    , _manualOffset(ManualOffsetInfo, glm::vec2(0.f), glm::vec2(-3.f), glm::vec2(3.f))
    , _telescopeOffset(
        TelescopeOffsetInfo, glm::vec2(0.f), glm::vec2(-1e6f), glm::vec2(1e6f)
    )
    , _planetSize(PlanetSizeInfo)
    , _magnification(MagnificationInfo, 0.0, 0.0, 1e12)
    , _planetRadius(PlanetRadiusInfo, 6.3781e6, 1.0, 1e10)
    , _targetDistance(TargetDistanceInfo, 30.0 * Parsec, 1e12, 1e22)
    , _fieldOfView(FieldOfViewInfo, 1.45f, 1.05f, 4.f)
    , _occulterRadius(OcculterRadiusInfo, 1.02f, 0.f, 1.08f)
    , _occulterShade(OcculterShadeInfo, 0.f, 0.f, 1.f)
    , _coronaStrength(CoronaStrengthInfo, 0.06f, 0.f, 1.f)
    , _gain(GainInfo, 1.f, 0.1f, 10.f)
    , _eyepiece(EyepieceInfo, true)
    , _resolution(ResolutionInfo, 1024, 128, 4096)
    , _saveFramePath(SaveFramePathInfo, "${USER}/sgl_telescope_frame.png")
    , _saveFrame(SaveFrameInfo)
{
    const Parameters p = codegen::bake<Parameters>(dictionary);

    std::string identifier = p.identifier.value_or("ScreenSpaceSglTelescope");
    setIdentifier(makeUniqueIdentifier(std::move(identifier)));

    _planetTexturePath = p.planetTexture.string();
    _planetTexturePath.onChange([this]() { _textureIsDirty = true; });
    addProperty(_planetTexturePath);

    _telescopeNode = p.telescopeNode.value_or(_telescopeNode);
    addProperty(_telescopeNode);

    _imagePlaneNode = p.imagePlaneNode;
    addProperty(_imagePlaneNode);

    _sunNode = p.sunNode.value_or(_sunNode);
    addProperty(_sunNode);

    _manualOffset = p.manualOffset.value_or(_manualOffset);
    addProperty(_manualOffset);

    _telescopeOffset.setReadOnly(true);
    addProperty(_telescopeOffset);

    _planetSize.addOptions({
        { 0, "Thin (kappa 0.02)" },
        { 1, "Medium (kappa 0.05)" },
        { 2, "Bold (kappa 0.12)" }
    });
    _planetSize = p.planetSize.value_or(1);
    addProperty(_planetSize);

    _magnification.setReadOnly(true);
    addProperty(_magnification);

    _planetRadius = p.planetRadius.value_or(_planetRadius);
    addProperty(_planetRadius);

    _targetDistance = p.targetDistance.value_or(_targetDistance);
    addProperty(_targetDistance);

    _fieldOfView = p.fieldOfView.value_or(_fieldOfView);
    addProperty(_fieldOfView);

    _occulterRadius = p.occulterRadius.value_or(_occulterRadius);
    addProperty(_occulterRadius);

    _occulterShade = p.occulterShade.value_or(_occulterShade);
    addProperty(_occulterShade);

    _coronaStrength = p.coronaStrength.value_or(_coronaStrength);
    addProperty(_coronaStrength);

    _gain = p.gain.value_or(_gain);
    addProperty(_gain);

    _eyepiece = p.eyepiece.value_or(_eyepiece);
    addProperty(_eyepiece);

    _resolution = p.resolution.value_or(_resolution);
    _resolution.onChange([this]() { _framebufferIsDirty = true; });
    addProperty(_resolution);

    addProperty(_saveFramePath);
    _saveFrame.onChange([this]() { _shouldSaveFrame = true; });
    addProperty(_saveFrame);

    _objectSize = glm::ivec2(_resolution);
}

ScreenSpaceSglTelescope::~ScreenSpaceSglTelescope() {}

void ScreenSpaceSglTelescope::initializeGL() {
    ScreenSpaceRenderable::initializeGL();

    _program = ghoul::opengl::ProgramObject::Build(
        "ScreenSpaceSglTelescope",
        absPath("${MODULE_SGL}/shaders/telescope_vs.glsl"),
        absPath("${MODULE_SGL}/shaders/telescope_fs.glsl")
    );
    // The full-screen triangle is generated in the vertex shader; OpenGL still needs a
    // vertex array object bound to draw.
    glCreateVertexArrays(1, &_vao);
}

void ScreenSpaceSglTelescope::deinitializeGL() {
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;
    _program = nullptr;
    _planetTexture = nullptr;
    _framebuffer = nullptr;
    _viewTexture = nullptr;

    ScreenSpaceRenderable::deinitializeGL();
}

void ScreenSpaceSglTelescope::update() {
    if (_textureIsDirty && !_planetTexturePath.value().empty()) {
        _planetTexture = ghoul::io::texture::loadTexture(
            absPath(_planetTexturePath),
            2,
            ghoul::opengl::Texture::SamplerInit{
                .filter = ghoul::opengl::Texture::FilterMode::Linear,
                .wrapping = ghoul::opengl::Texture::WrappingMode::ClampToEdge
            }
        );
        _textureIsDirty = false;
    }

    // A fisheye or other multi-viewport window calls render() once per viewport, but the
    // view only changes between frames, so it is ray-traced once per update()
    _viewIsDirty = true;

    Scene* scene = global::renderEngine->scene();
    if (!scene) {
        return;
    }
    const SceneGraphNode* plane = scene->sceneGraphNode(_imagePlaneNode.value());
    const SceneGraphNode* sun = scene->sceneGraphNode(_sunNode.value());
    const SceneGraphNode* telescope = _telescopeNode.value().empty() ?
        nullptr :
        scene->sceneGraphNode(_telescopeNode.value());

    // Say once which named node is missing, instead of silently showing a stale view
    std::string missing;
    const auto report = [&missing](const std::string& entry) {
        missing += missing.empty() ? entry : ", " + entry;
    };
    if (!plane) {
        report(std::format("ImagePlaneNode '{}'", _imagePlaneNode.value()));
    }
    if (!sun) {
        report(std::format("SunNode '{}'", _sunNode.value()));
    }
    if (!telescope && !_telescopeNode.value().empty()) {
        report(std::format(
            "TelescopeNode '{}'{}",
            _telescopeNode.value(),
            (plane && sun) ? " (using ManualOffset)" : ""
        ));
    }
    if (missing != _missingNodes) {
        if (!missing.empty()) {
            LWARNING(std::format("Scene graph node(s) not found: {}", missing));
        }
        _missingNodes = missing;
    }
    if (!plane || !sun) {
        return;
    }

    // Everything the shader needs is a ratio of angles, so it works in units of the
    // Einstein angle and stays near 1 even though the real angles are 1e-5 to 1e-11 rad
    const glm::dvec3 planePosition = plane->worldPosition();
    const double z = glm::length(planePosition - sun->worldPosition());
    const double thetaE = std::sqrt(2.0 * SchwarzschildRadius / z);
    _sunRadius = static_cast<float>(SolarRadius / z / thetaE);
    const double imageRadius = _planetRadius * z / _targetDistance;

    glm::vec2 offset = _manualOffset;
    if (telescope) {
        // Into the image plane's own frame: x and y span the plane
        const glm::dvec3 local = glm::transpose(plane->worldRotationMatrix()) *
            (telescope->worldPosition() - planePosition);
        offset = glm::vec2(local.x / imageRadius, local.y / imageRadius);
    }
    _telescopeOffset = offset;

    const double planetAngle = _planetRadius / _targetDistance;
    _magnification = Kappas[_planetSize.value()] * thetaE / planetAngle;
}

void ScreenSpaceSglTelescope::render(const RenderData& renderData) {
    if (!_program || !_planetTexture) {
        return;
    }
    if (_framebufferIsDirty) {
        createFramebuffer();
        _viewIsDirty = true;
    }
    if (_viewIsDirty) {
        renderView();
        _viewIsDirty = false;
    }

    if (_shouldSaveFrame) {
        saveFrame();
        _shouldSaveFrame = false;
    }

    ScreenSpaceRenderable::render(renderData);
}

void ScreenSpaceSglTelescope::createFramebuffer() {
    const unsigned int res = static_cast<unsigned int>(_resolution.value());

    _framebuffer = std::make_unique<ghoul::opengl::FramebufferObject>();
    _framebuffer->activate();
    _viewTexture = std::make_unique<ghoul::opengl::Texture>(
        ghoul::opengl::Texture::FormatInit {
            .dimensions = glm::uvec3(res, res, 1),
            .type = GL_TEXTURE_2D,
            .format = ghoul::opengl::Texture::Format::RGBA,
            .dataType = GL_UNSIGNED_BYTE
        },
        ghoul::opengl::Texture::SamplerInit {
            .filter = ghoul::opengl::Texture::FilterMode::Linear,
            .wrapping = ghoul::opengl::Texture::WrappingMode::ClampToEdge
        }
    );
    _framebuffer->attachTexture(_viewTexture.get(), GL_COLOR_ATTACHMENT0);
    ghoul::opengl::FramebufferObject::deactivate();

    _objectSize = glm::ivec2(_resolution);
    _framebufferIsDirty = false;
}

void ScreenSpaceSglTelescope::renderView() {
    std::array<GLint, 4> viewport;
    glGetIntegerv(GL_VIEWPORT, viewport.data());
    const GLint defaultFbo = ghoul::opengl::FramebufferObject::getActiveObject();
    const GLboolean blendWasOn = glIsEnabled(GL_BLEND);
    const GLboolean depthWasOn = glIsEnabled(GL_DEPTH_TEST);

    _framebuffer->activate();
    glViewport(0, 0, _resolution, _resolution);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    _program->activate();
    ghoul::opengl::TextureUnit unit;
    unit.bind(*_planetTexture);
    _program->setUniform("planetTexture", unit);
    _program->setUniform("telescopeOffset", _telescopeOffset.value());
    _program->setUniform("kappa", static_cast<float>(Kappas[_planetSize.value()]));
    _program->setUniform("fieldOfView", _fieldOfView.value());
    _program->setUniform("sunRadius", _sunRadius);
    _program->setUniform("occulterRadius", _occulterRadius.value());
    _program->setUniform("occulterShade", _occulterShade.value());
    _program->setUniform("coronaStrength", _coronaStrength.value());
    _program->setUniform("gain", _gain.value());
    _program->setUniform("eyepiece", _eyepiece.value());
    _program->setUniform("resolution", static_cast<float>(_resolution.value()));

    glBindVertexArray(_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    _program->deactivate();

    if (blendWasOn) {
        glEnable(GL_BLEND);
    }
    if (depthWasOn) {
        glEnable(GL_DEPTH_TEST);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, defaultFbo);
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
}

void ScreenSpaceSglTelescope::saveFrame() {
    _viewTexture->downloadTexture();
    const std::filesystem::path path = absPath(_saveFramePath);
    ghoul::io::texture::saveTexture(*_viewTexture, path.string());
    LINFO(std::format("Saved telescope view to '{}'", path));
}

void ScreenSpaceSglTelescope::bindTexture(ghoul::opengl::TextureUnit& unit) {
    if (_viewTexture) {
        unit.bind(*_viewTexture);
    }
}

} // namespace openspace
