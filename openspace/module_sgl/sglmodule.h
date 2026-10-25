/*****************************************************************************************
 * SGL module: the view from a telescope in the solar gravitational lens image plane.    *
 * Portfolio project, Andrew Simon. Built against OpenSpace v0.22.0 (MIT license).       *
 ****************************************************************************************/

#ifndef __OPENSPACE_MODULE_SGL___SGLMODULE___H__
#define __OPENSPACE_MODULE_SGL___SGLMODULE___H__

#include <openspace/util/openspacemodule.h>

namespace openspace {

class SglModule : public OpenSpaceModule {
public:
    constexpr static const char* Name = "Sgl";

    SglModule();

    std::vector<openspace::Documentation> documentations() const override;

protected:
    void internalInitialize(const ghoul::Dictionary&) override;
};

} // namespace openspace

#endif // __OPENSPACE_MODULE_SGL___SGLMODULE___H__
