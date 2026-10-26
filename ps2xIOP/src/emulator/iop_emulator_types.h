#pragma once

#include <cstdint>

// Types the emulator (ported from the ATV 1 tree) shares with its callers
// that ATV 2's public ps2x/iop/iop_types.h does not have.
namespace ps2x::iop
{
    struct ModuleLoadResult
    {
        bool handled = false;
        int32_t moduleId = -1;
        int32_t startResult = -1;
    };
}
