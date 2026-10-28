#pragma once

// Opt-in texture replacement packs in PCSX2's format, and texture dumps under
// PCSX2's names (GSTextureReplacements.cpp / GSTextureCache::HashCacheKey).
//   ATV2_TEXTURE_PACK=<dir>[;<dir>...]  PNGs from <dir>/replacements, or <dir> itself; earlier dirs
//                            win. A pack's fuzzy_index.bin (convert_dds_pack.py) lets direct-colour
//                            textures whose bytes differ from PCSX2's (IPU-decoded) match by thumbnail.
//   ATV2_TEXTURE_DUMP=<dir>  every texture a draw samples, once per name.
//   ATV2_TEXTURE_PACK_MB=<n> loaded-image budget (default 2048).
//   ATV2_TEXTURE_PACK_SCALE=<n> finest level sampled, as a multiple of the GS texture's size
//                            (default 16 = whatever the pack holds; 2 trades detail for speed).
//   ATV2_TEXTURE_PACK_DEBUG=1 hash/hit/load counts every 10 s.
// Images keep PS2 alpha (0x80 = opaque), as PCSX2 dumps and loads them.
//
// Textures are hashed on the thread that submits primitives, against VRAM as
// the primitive will see it, and re-hashed only after something writes their
// pages. Band threads get each queued primitive's result by its queue index.

#include "runtime/gs/gs_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace GSTexReplace
{
    struct Level
    {
        uint32_t w = 0, h = 0;
        const uint32_t *texels = nullptr;
    };

    // RGBA8 replacement with a box-filtered mip chain down to 1x1.
    struct Image
    {
        uint32_t regionW = 0, regionH = 0; // the name's region (0 = whole TW/TH)
        uint32_t levelCount = 0;
        Level levels[16];
        std::vector<uint32_t> storage;
    };

    struct Axis
    {
        float scale = 1.0f, offset = 0.0f; // level texel = base texel * scale + offset
        int size = 1, lo = 0, hi = 0;
        bool repeat = true;
        bool pow2 = true;

        int fix(int i) const
        {
            if (!repeat)
                return std::clamp(i, lo, hi);
            if (pow2)
                return i & (size - 1);
            i %= size;
            return i < 0 ? i + size : i;
        }
    };

    // Per-primitive sampling state for one level of a replacement.
    struct Sampler
    {
        const uint32_t *texels = nullptr; // nullptr: sample the GS texture
        uint32_t pitch = 0;
        Axis u, v;
        bool linear = false;
    };

    extern const bool g_active;
    inline bool Active() { return g_active; }

    // Submit thread. owner is the GS backend whose VRAM and queue these are.
    void OnSubmit(const void *owner, const GSPrimitiveBatch &batch, const uint8_t *vram, const uint16_t *clut);
    void OnQueued(const void *owner);                                   // the submitted primitive joined the band queue
    void OnBandFlush(const void *owner);                                // band queue -> in flight
    void NoteVramWrite(const void *owner, uint32_t begin, uint32_t end); // byte range; end UINT32_MAX = everything
    void NoteTargets(const void *owner, const GSContext &ctx);          // a clear/draw writes FRAME (and ZBUF)
    void NoteClutLoad(const void *owner);

    // Band threads, around DrawBand's loop over the in-flight queue.
    void BeginBand(const void *owner);
    void SetBandDraw(uint32_t index);
    void EndBand();

    // Replacement for the primitive this thread is drawing, or nullptr.
    const Image *Current();

    // Level of img nearest one texel per pixel for this primitive, never coarser
    // than the GS texture's own size.
    uint32_t PickLevel(const Image &img, const GSPrimitiveBatch &batch);

    void Prepare(const Image &img, uint32_t level, const GSDrawState &state, Sampler &out);

    // Bilinear blend, two 8-bit channels per 32-bit lane; weights in 1/256ths.
    inline uint32_t lerpRgba(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t wx, uint32_t wy)
    {
        constexpr uint32_t kMask = 0x00FF00FFu;
        const auto lerp = [](uint32_t p, uint32_t q, uint32_t w) { return ((p * (256u - w) + q * w) >> 8) & kMask; };
        const uint32_t rb = lerp(lerp(a & kMask, b & kMask, wx), lerp(c & kMask, d & kMask, wx), wy);
        const uint32_t ga = lerp(lerp((a >> 8) & kMask, (b >> 8) & kMask, wx), lerp((c >> 8) & kMask, (d >> 8) & kMask, wx), wy);
        return rb | (ga << 8);
    }

    inline int floorToInt(float x)
    {
        const int i = static_cast<int>(x);
        return (x < static_cast<float>(i)) ? i - 1 : i;
    }

    // u, v in GS texels of the base texture (the same values the GS sampler uses).
    inline uint32_t Sample(const Sampler &s, float u, float v)
    {
        const float x = u * s.u.scale + s.u.offset;
        const float y = v * s.v.scale + s.v.offset;
        if (!s.linear)
            return s.texels[static_cast<size_t>(s.v.fix(floorToInt(y))) * s.pitch + static_cast<size_t>(s.u.fix(floorToInt(x)))];
        const float sx = x - 0.5f, sy = y - 0.5f;
        const int x0 = floorToInt(sx), y0 = floorToInt(sy);
        const uint32_t wx = static_cast<uint32_t>((sx - static_cast<float>(x0)) * 256.0f);
        const uint32_t wy = static_cast<uint32_t>((sy - static_cast<float>(y0)) * 256.0f);
        const size_t r0 = static_cast<size_t>(s.v.fix(y0)) * s.pitch;
        const size_t r1 = static_cast<size_t>(s.v.fix(y0 + 1)) * s.pitch;
        const size_t c0 = static_cast<size_t>(s.u.fix(x0)), c1 = static_cast<size_t>(s.u.fix(x0 + 1));
        return lerpRgba(s.texels[r0 + c0], s.texels[r0 + c1], s.texels[r1 + c0], s.texels[r1 + c1], std::min(wx, 256u),
                        std::min(wy, 256u));
    }
}
