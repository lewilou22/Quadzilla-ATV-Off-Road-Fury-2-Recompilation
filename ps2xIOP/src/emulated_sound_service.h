#pragma once

#include "iop_service.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ps2x::iop::detail
{
    // A game's own sound driver IRX run on the IOP emulator (with its SPU2),
    // instead of an HLE of its RPC protocol.
    struct EmulatedSoundDriverBindings
    {
        std::string serviceName;
        // IRX images loaded in this order, e.g. "cdrom0:\\MODULES\\LIBSD.IRX;1".
        std::vector<std::string> modules;
        // RPC servers the driver registers; calls to them run on the emulated IOP.
        std::vector<uint32_t> sids;
    };

    std::unique_ptr<IopService> createEmulatedSoundDriverService(IopHost &host, EmulatedSoundDriverBindings bindings);
}
