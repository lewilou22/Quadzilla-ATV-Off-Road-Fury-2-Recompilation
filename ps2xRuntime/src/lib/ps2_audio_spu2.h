#pragma once

namespace ps2_audio_spu2
{
    // Plays the emulated SPU2 (ps2x::iop::renderEmulatedAudio) on the host audio
    // device: a raylib 48 kHz stream through a WSOLA time stretcher. Needs
    // InitAudioDevice() first. ATV_AUDIO=0 keeps it off (the SPU2 then runs on
    // IOP time alone); ATV_AUDIO_DUMP=file.wav records what is played,
    // ATV_AUDIO_DUMP_RAW=file.wav the SPU2 output before the stretcher.
    bool start();
    void stop();
}
