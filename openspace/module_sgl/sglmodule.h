/*****************************************************************************************
 * SGL module: the view from a telescope in the solar gravitational lens image plane.    *
 * Copyright (c) 2026 Andrew Simon. MIT License; see LICENSE in the sgl-openspace repo.  *
 * Built against OpenSpace v0.22.0.                                                      *
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
