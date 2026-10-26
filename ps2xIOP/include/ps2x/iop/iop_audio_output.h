#pragma once

#include <cstdint>

namespace ps2x::iop
{
    // Sound from an emulated SPU2 (a profile that runs the game's sound IRX on
    // the IOP emulator), for the host audio thread.

    // Interleaved stereo int16 at 48 kHz. Returns the frames written: at most
    // what IOP time has made due, and 0 while no emulated SPU2 runs.
    [[nodiscard]] uint32_t renderEmulatedAudio(int16_t *stereo, uint32_t maxFrames);

    // A host stream consumes renderEmulatedAudio(). Without one the SPU2 is
    // clocked by IOP time alone, so drivers still see voices finish.
    void setEmulatedAudioConsumer(bool active);
}
