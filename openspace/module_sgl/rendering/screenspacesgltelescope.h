/*****************************************************************************************
 * SGL module: the view from a telescope in the solar gravitational lens image plane.    *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#ifndef __OPENSPACE_MODULE_SGL___SCREENSPACESGLTELESCOPE___H__
#define __OPENSPACE_MODULE_SGL___SCREENSPACESGLTELESCOPE___H__

#include <openspace/rendering/screenspacerenderable.h>

#include <openspace/properties/misc/optionproperty.h>
#include <openspace/properties/misc/stringproperty.h>
#include <openspace/properties/misc/triggerproperty.h>
#include <openspace/properties/scalar/boolproperty.h>
#include <openspace/properties/scalar/doubleproperty.h>
#include <openspace/properties/scalar/floatproperty.h>
#include <openspace/properties/scalar/intproperty.h>
#include <openspace/properties/vector/vec2property.h>
#include <ghoul/opengl/ghoul_gl.h>
#include <memory>
#include <string>

namespace ghoul::opengl {
    class FramebufferObject;
    class ProgramObject;
    class Texture;
} // namespace ghoul::opengl

namespace openspace {

/**
 * What a telescope in the solar gravitational lens image plane sees when it looks back at
 * the Sun: the target planet lensed into an Einstein ring around a coronagraph occulter.
 *
 * Every pixel is ray-traced through the point-mass lens equation,
 * beta = theta - theta/|theta|^2 (angles in units of the Einstein angle), and reads the
 * planet texture where the ray came from.
 * The planet is magnified so it can be seen at all; the factor is shown in the
 * `Magnification` property. The view is drawn into an offscreen texture and shown as a
 * screen-space plane, with an optional circular eyepiece mask.
 *
 * The math is a port of layer_c_proto/ring_view.py, which was checked against
 * the wave-optics model (Toth & Turyshev 2021) before this port.
 */
class ScreenSpaceSglTelescope : public ScreenSpaceRenderable {
public:
    explicit ScreenSpaceSglTelescope(const ghoul::Dictionary& dictionary);
    ~ScreenSpaceSglTelescope() override;

    void initializeGL() override;
    void deinitializeGL() override;
    void update() override;
    void render(const RenderData& renderData) override;

    static openspace::Documentation Documentation();

private:
    void bindTexture(ghoul::opengl::TextureUnit& unit) override;
    void createFramebuffer();
    void renderView();
    void saveFrame();

    StringProperty _planetTexturePath;
    StringProperty _telescopeNode;
    StringProperty _imagePlaneNode;
    StringProperty _sunNode;
    Vec2Property _manualOffset;
    Vec2Property _telescopeOffset;
    OptionProperty _planetSize;
    DoubleProperty _magnification;
    DoubleProperty _planetRadius;
    DoubleProperty _targetDistance;
    FloatProperty _fieldOfView;
    FloatProperty _occulterRadius;
    FloatProperty _occulterShade;
    FloatProperty _coronaStrength;
    FloatProperty _gain;
    BoolProperty _eyepiece;
    IntProperty _resolution;
    StringProperty _saveFramePath;
    TriggerProperty _saveFrame;

    // Solar radius in units of the Einstein angle at the image plane's distance
    float _sunRadius = 0.918f;
    bool _framebufferIsDirty = true;
    bool _textureIsDirty = true;
    bool _viewIsDirty = true;
    bool _shouldSaveFrame = false;
    // The node names last reported missing, so each problem is logged once
    std::string _missingNodes;

    std::unique_ptr<ghoul::opengl::Texture> _planetTexture;
    std::unique_ptr<ghoul::opengl::FramebufferObject> _framebuffer;
    std::unique_ptr<ghoul::opengl::Texture> _viewTexture;
    std::unique_ptr<ghoul::opengl::ProgramObject> _program;
    GLuint _vao = 0;
};

} // namespace openspace

#endif // __OPENSPACE_MODULE_SGL___SCREENSPACESGLTELESCOPE___H__
