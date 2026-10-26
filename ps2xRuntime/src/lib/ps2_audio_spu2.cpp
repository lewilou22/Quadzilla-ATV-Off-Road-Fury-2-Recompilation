#include "ps2_audio_spu2.h"
#include "ps2_audio_stretch.h"
#include "ps2_host_backend.h"
#include "ps2x/iop/iop_audio_output.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace
{
    constexpr unsigned int kSpu2SampleRate = 48000u;
    constexpr unsigned int kSpu2StreamFrames = 1024u;

    std::mutex g_streamMutex;
    AudioStream g_stream{};
    bool g_streamActive = false;
    // Used only from the raylib audio thread while the stream plays.
    ps2_audio::TimeStretcher *g_stretcher = nullptr;
    void writeWavHeader(FILE *file, uint64_t frames)
    {
        const uint32_t dataBytes = static_cast<uint32_t>(std::min<uint64_t>(frames * 4u, 0xFFFFFFF0u));
        const uint32_t riffBytes = 36u + dataBytes;
        const uint32_t byteRate = kSpu2SampleRate * 4u;
        const uint32_t fmtBytes = 16u;
        const uint16_t format = 1u, channels = 2u, blockAlign = 4u, bits = 16u;
        std::fseek(file, 0, SEEK_SET);
        std::fwrite("RIFF", 1, 4, file);
        std::fwrite(&riffBytes, 4, 1, file);
        std::fwrite("WAVEfmt ", 1, 8, file);
        std::fwrite(&fmtBytes, 4, 1, file);
        std::fwrite(&format, 2, 1, file);
        std::fwrite(&channels, 2, 1, file);
        std::fwrite(&kSpu2SampleRate, 4, 1, file);
        std::fwrite(&byteRate, 4, 1, file);
        std::fwrite(&blockAlign, 2, 1, file);
        std::fwrite(&bits, 2, 1, file);
        std::fwrite("data", 1, 4, file);
        std::fwrite(&dataBytes, 4, 1, file);
        std::fseek(file, 0, SEEK_END);
    }

    // Optional 48 kHz stereo WAV capture for offline checks.
    struct WavDump
    {
        FILE *file = nullptr;
        uint64_t frames = 0;

        void open(const char *env)
        {
            const char *path = std::getenv(env);
            if (!path || path[0] == '\0')
                return;
            file = std::fopen(path, "wb");
            frames = 0;
            if (file)
                writeWavHeader(file, 0);
        }

        void write(const int16_t *samples, uint64_t count)
        {
            if (!file || count == 0u)
                return;
            std::fwrite(samples, sizeof(int16_t) * 2u, static_cast<size_t>(count), file);
            frames += count;
            if ((frames & 0xFFFFu) < count)
            {
                writeWavHeader(file, frames);
                std::fflush(file);
            }
        }

        void close()
        {
            if (!file)
                return;
            writeWavHeader(file, frames);
            std::fclose(file);
            file = nullptr;
        }
    };

    // ATV_AUDIO_DUMP: what is played. ATV_AUDIO_DUMP_RAW: the SPU2 output
    // before the time stretcher, at IOP pace.
    WavDump g_dump;
    WavDump g_rawDump;

    void streamCallback(void *buffer, unsigned int frames)
    {
        auto *samples = static_cast<int16_t *>(buffer);
        if (g_stretcher)
            g_stretcher->process(samples, frames, [](int16_t *pcm, uint32_t count)
                                 {
                                     const uint32_t rendered = ps2x::iop::renderEmulatedAudio(pcm, count);
                                     g_rawDump.write(pcm, rendered);
                                     return rendered; });
        else
            std::memset(samples, 0, static_cast<size_t>(frames) * 2u * sizeof(int16_t));
        g_dump.write(samples, frames);
    }
}

namespace ps2_audio_spu2
{
    bool start()
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        if (g_streamActive)
            return true;
        const char *setting = std::getenv("ATV_AUDIO");
        if ((setting && setting[0] == '0') || !IsAudioDeviceReady())
            return false;

        SetAudioStreamBufferSizeDefault(static_cast<int>(kSpu2StreamFrames));
        g_stream = LoadAudioStream(kSpu2SampleRate, 16, 2);
        SetAudioStreamBufferSizeDefault(0);
        if (!IsAudioStreamValid(g_stream))
            return false;

        g_dump.open("ATV_AUDIO_DUMP");
        g_rawDump.open("ATV_AUDIO_DUMP_RAW");
        g_stretcher = new ps2_audio::TimeStretcher();
        ps2x::iop::setEmulatedAudioConsumer(true);
        SetAudioStreamCallback(g_stream, streamCallback);
        PlayAudioStream(g_stream);
        g_streamActive = true;
        return true;
    }

    void stop()
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        if (!g_streamActive)
            return;
        // Unloading detaches the stream from the mixer under raylib's audio
        // lock, so no callback is running once this returns.
        StopAudioStream(g_stream);
        UnloadAudioStream(g_stream);
        g_stream = AudioStream{};
        g_streamActive = false;
        delete g_stretcher;
        g_stretcher = nullptr;
        g_dump.close();
        g_rawDump.close();
        ps2x::iop::setEmulatedAudioConsumer(false);
    }
}
