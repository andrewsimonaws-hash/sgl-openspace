/*****************************************************************************************
 * SGL module: the view from a telescope in the solar gravitational lens image plane.    *
 * Portfolio project, Andrew Simon. Built against OpenSpace v0.22.0 (MIT license).       *
 ****************************************************************************************/

#include <modules/sgl/sglmodule.h>

#include <modules/sgl/rendering/screenspacesgltelescope.h>
#include <openspace/documentation/documentation.h>
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
}

std::vector<Documentation> SglModule::documentations() const {
    return {
        ScreenSpaceSglTelescope::Documentation()
    };
}

} // namespace openspace
