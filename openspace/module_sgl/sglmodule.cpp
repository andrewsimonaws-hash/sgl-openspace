/*****************************************************************************************
 * SGL module: the view from a telescope in the solar gravitational lens image plane.    *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
 ****************************************************************************************/

#include <modules/sgl/sglmodule.h>

#include <modules/sgl/rendering/renderablesglcorona.h>
#include <modules/sgl/rendering/screenspacesgltelescope.h>
#include <openspace/documentation/documentation.h>
#include <openspace/rendering/renderable.h>
#include <openspace/rendering/screenspacerenderable.h>
#include <openspace/util/factorymanager.h>
#include <ghoul/misc/assert.h>
#include <ghoul/misc/templatefactory.h>

namespace openspace {

SglModule::SglModule() : OpenSpaceModule(Name) {}

void SglModule::internalInitialize(const ghoul::Dictionary&) {
    ghoul::TemplateFactory<ScreenSpaceRenderable>* fSsRenderable =
        FactoryManager::ref().factory<ScreenSpaceRenderable>();
    ghoul_assert(fSsRenderable, "ScreenSpaceRenderable factory was not created");

    fSsRenderable->registerClass<ScreenSpaceSglTelescope>("ScreenSpaceSglTelescope");

    ghoul::TemplateFactory<Renderable>* fRenderable =
        FactoryManager::ref().factory<Renderable>();
    ghoul_assert(fRenderable, "Renderable factory was not created");

    fRenderable->registerClass<RenderableSglCorona>("RenderableSglCorona");
}

std::vector<Documentation> SglModule::documentations() const {
    return {
        ScreenSpaceSglTelescope::Documentation(),
        RenderableSglCorona::Documentation()
    };
}

} // namespace openspace
