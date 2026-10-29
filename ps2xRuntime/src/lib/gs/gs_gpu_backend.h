#pragma once

// Opt-in hardware GS (ATV_GS_BACKEND=gpu, Windows/D3D11): draws into the race's
// full-size render targets at ATV_GS_SCALE times the PS2 resolution (default 2).
// An inner GSCpuBackend keeps GS local memory and draws everything else at 1x
// (small off-screen targets, lines/points); render targets are copied back to
// local memory only when something on the CPU side reads their pages.
//   ATV_GS_GPU_STATS=1  draws/seeds/downloads/decodes every ~5 s on stderr (=0.5: interval in s),
//                       plus each kind of CPU-routed draw and each present from local memory
//                       with the CPU-side write that made the GPU copy stale.
//   ATV_GS_GPU_CAPTURE=<file>  record this backend's command stream in the ATV_GS_CAPTURE
//                       format, and in <file>.live what it returned (readbacks, local memory at the
//                       first 4 readbacks); _SKIP, _COMMANDS, _AFTER_FALLBACKS choose the window.
//   Debug: ATV_GS_GPU_ALLCPU=1 (every draw on the CPU GS), ATV_GS_GPU_NOCACHE=1,
//          ATV_GS_GPU_TRACE=<fbp> (events of that colour target), ATV_GS_GPU_PROBE=a,b (its mean
//          colour after every batch between primitives a and b).

#include "runtime/gs/gs_backend.h"

#include <array>
#include <cstdint>
#include <memory>

// nullptr when the platform or the device does not support it.
std::unique_ptr<GSRasterBackend> gsCreateGpuBackend(uint32_t scale = 0); // 0 = ATV_GS_SCALE or 2

bool gsIsGpuBackend(const GSRasterBackend *backend);

// Replay support: the palette state a capture file starts with.
void gsGpuRestoreClutState(GSRasterBackend *backend, const std::array<uint16_t, 512> &clut,
                           const std::array<uint32_t, 2> &cbp);
