#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using namespace GSInternal;

extern uint32_t g_ps2XgkickCount;
extern bool g_raceLoadStarted;

namespace
{
    constexpr uint32_t kVramBytes = 4u * 1024u * 1024u;
    constexpr uint32_t kVramPageBytes = 8192u;
    constexpr uint32_t kVramBlockBytes = 256u;
    constexpr uint32_t kBandRowShift = 3u;

    // Band assignment of the calling thread while it draws a banded batch;
    // t_bandCount == 1 means the thread draws every row.
    thread_local uint32_t t_bandIndex = 0u;
    thread_local uint32_t t_bandCount = 1u;
    thread_local int t_ownedBandRow = -1;
    // ATV_GS_CULL_VERIFY: set by DrawTriangle when a sample passes coverage.
    thread_local bool t_triangleCovered = false;
    thread_local GSMem::TexturePageCache *t_bandCache = nullptr;
    // Palette of the primitive a band is drawing (DrawBand); null = the backend's current CLUT.
    thread_local const uint16_t *t_drawClut = nullptr;

    uint64_t bandClockNs()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    // block % t_bandCount without a division for the usual power-of-two counts.
    inline uint32_t bandOfBlock(uint32_t block)
    {
        const uint32_t n = t_bandCount;
        return (n & (n - 1u)) == 0u ? (block & (n - 1u)) : block % n;
    }

    inline bool ownsBandRow(int y)
    {
        return t_bandCount <= 1u || y == t_ownedBandRow ||
               bandOfBlock(static_cast<uint32_t>(y) >> kBandRowShift) == t_bandIndex;
    }

    // static_cast<int>(std::floor(f)) and std::ceil without the CRT calls. Same
    // result for every float: out-of-range values and NaN convert to INT_MIN
    // either way (cvttss2si), and are never adjusted.
    inline int floorToInt(float f)
    {
        int i = static_cast<int>(f);
        if (f < static_cast<float>(i) && i != INT_MIN)
            --i;
        return i;
    }

    inline int ceilToInt(float f)
    {
        int i = static_cast<int>(f);
        if (f > static_cast<float>(i) && i != INT_MIN)
            ++i;
        return i;
    }

    // std::min({a, b, c}) / std::max({a, b, c}): first extreme element, as
    // std::min_element / std::max_element pick it.
    inline float min3(float a, float b, float c)
    {
        float m = a;
        if (b < m)
            m = b;
        if (c < m)
            m = c;
        return m;
    }

    inline float max3(float a, float b, float c)
    {
        float m = a;
        if (m < b)
            m = b;
        if (m < c)
            m = c;
        return m;
    }

    // Raster bounds are already clipped to the non-negative GS scissor.
    // Jump over other bands' eight-row blocks instead of testing every row.
    inline int firstBandRow(int y)
    {
        if (t_bandCount <= 1u)
            return y;
        const uint32_t block = static_cast<uint32_t>(y) >> kBandRowShift;
        const uint32_t owner = bandOfBlock(block);
        if (owner == t_bandIndex)
            return y;
        uint32_t skip = t_bandIndex + t_bandCount - owner; // < 2 * t_bandCount
        if (skip >= t_bandCount)
            skip -= t_bandCount;
        return static_cast<int>((block + skip) << kBandRowShift);
    }

    inline int nextBandRow(int y)
    {
        ++y;
        if (t_bandCount > 1u && (y & ((1u << kBandRowShift) - 1u)) == 0u)
            y += static_cast<int>((t_bandCount - 1u) << kBandRowShift);
        return y;
    }

    // False only when the calling band owns none of the rows DrawTriangle
    // would visit (same bounds: vertex y minus OFY, floor/ceil, scissor).
    inline bool bandMayDrawPrimitive(const GSPrimitiveBatch &batch)
    {
        if (t_bandCount <= 1u)
            return true;
        switch (batch.state.prim.type)
        {
        case GS_PRIM_TRIANGLE:
        case GS_PRIM_TRISTRIP:
        case GS_PRIM_TRIFAN:
            break;
        default:
            return true;
        }
        const auto &ctx = batch.state.context;
        const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        const float fy0 = batch.vertices[0].y - ofy;
        const float fy1 = batch.vertices[1].y - ofy;
        const float fy2 = batch.vertices[2].y - ofy;
        const int minY = floorToInt(min3(fy0, fy1, fy2));
        const int maxY = ceilToInt(max3(fy0, fy1, fy2));
        if (maxY < ctx.scissor.y0 || minY > ctx.scissor.y1)
            return false;
        return firstBandRow(clampInt(minY, ctx.scissor.y0, ctx.scissor.y1)) <=
               clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);
    }

    void pageGeometry(uint32_t psm, uint32_t &pageWidth, uint32_t &pageHeight)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            pageWidth = 64u;
            pageHeight = 64u;
            break;
        case GS_PSM_T8:
            pageWidth = 128u;
            pageHeight = 64u;
            break;
        case GS_PSM_T4:
            pageWidth = 128u;
            pageHeight = 128u;
            break;
        default:
            pageWidth = 64u;
            pageHeight = 32u;
            break;
        }
    }

    // Conservative byte range of a widthPx x heightPx surface; a range that
    // wraps past the end of VRAM covers everything.
    template <typename Range>
    Range surfaceRange(uint32_t beginBytes, uint32_t widthPx, uint32_t psm, uint32_t heightPx,
                       uint32_t layoutKey, uint32_t slackBytes)
    {
        uint32_t pageWidth = 64u;
        uint32_t pageHeight = 32u;
        pageGeometry(psm, pageWidth, pageHeight);
        // Page extents are powers of two.
        const uint64_t pagesPerRow = std::max<uint64_t>(
            1u, (static_cast<uint64_t>(widthPx) + pageWidth - 1u) >> std::countr_zero(pageWidth));
        const uint64_t pageRows = std::max<uint64_t>(
            1u, (static_cast<uint64_t>(heightPx) + pageHeight - 1u) >> std::countr_zero(pageHeight));
        const uint64_t begin = beginBytes;
        const uint64_t end = begin + pagesPerRow * pageRows * kVramPageBytes + slackBytes;
        Range range{};
        range.layoutKey = layoutKey;
        if (end > kVramBytes)
        {
            range.begin = 0u;
            range.end = UINT32_MAX;
        }
        else
        {
            range.begin = static_cast<uint32_t>(begin);
            range.end = static_cast<uint32_t>(end);
        }
        return range;
    }

    template <typename Range>
    bool overlaps(const Range &a, const Range &b)
    {
        return a.begin < b.end && b.begin < a.end;
    }

    // ATV_GS_CAPTURE=<file>: record the command stream one backend receives
    // during a race, for ps2_gs_replay_bench. Starts at the first
    // BeginTransfer after ATV_GS_CAPTURE_SKIP race primitives (default 3M)
    // and stops after ATV_GS_CAPTURE_COMMANDS commands (default 600k). The
    // file holds VRAM and CLUT state at the start, then tagged raw records.
    struct GSCaptureState
    {
        std::FILE *file = nullptr;
        const void *owner = nullptr;
        uint64_t skip = 3000000u;
        uint64_t remaining = 600000u;
        uint64_t racePrimitives = 0u;
        bool enabled = false;
        bool finished = false;
    };

    GSCaptureState &gsCapture()
    {
        static GSCaptureState state = []
        {
            GSCaptureState s{};
            const char *path = std::getenv("ATV_GS_CAPTURE");
            s.enabled = path && *path;
            if (const char *skip = std::getenv("ATV_GS_CAPTURE_SKIP"))
                s.skip = std::strtoull(skip, nullptr, 10);
            if (const char *count = std::getenv("ATV_GS_CAPTURE_COMMANDS"))
                s.remaining = std::strtoull(count, nullptr, 10);
            return s;
        }();
        return state;
    }

    inline bool gsCaptureActive(const void *backend)
    {
        const GSCaptureState &c = gsCapture();
        return c.file && c.owner == backend;
    }

    void gsCaptureWrite(uint8_t tag, const void *payload, size_t bytes,
                        const void *tail = nullptr, size_t tailBytes = 0u)
    {
        GSCaptureState &c = gsCapture();
        std::fwrite(&tag, 1u, 1u, c.file);
        if (bytes)
            std::fwrite(payload, 1u, bytes, c.file);
        if (tailBytes)
            std::fwrite(tail, 1u, tailBytes, c.file);
        if (--c.remaining == 0u)
        {
            const uint8_t end = GSCaptureEnd;
            std::fwrite(&end, 1u, 1u, c.file);
            std::fclose(c.file);
            c.file = nullptr;
            c.finished = true;
            std::fprintf(stderr, "[gs-capture] finished\n");
        }
    }

    float fabsQ(float q)
    {
        const float aq = std::fabs(q);
        return (aq > 1.0e-5f) ? aq : 1.0f;
    }

    u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    struct HudSlotCache
    {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<uint16_t> pixels;
    };
    static HudSlotCache s_hudSlot[4];

    extern "C" void GSCpuBackend_UpdateHudSlot(uint32_t tbp, uint32_t w, uint32_t h, const uint16_t *data)
    {
        uint32_t slot = (tbp >= 0x3480u && tbp <= 0x3a80u) ? ((tbp - 0x3480u) / 0x200u) : 0u;
        if (tbp >= 13440u && tbp <= 14976u)
            slot = (tbp - 13440u) / 512u;
        slot &= 3u;
        s_hudSlot[slot].width = w;
        s_hudSlot[slot].height = h;
        s_hudSlot[slot].pixels.assign(data, data + w * h);
    }

    uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            // PS2 GS spec: U = (U AND MINU) OR MAXU  (MINU = mask, MAXU = bias)
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        // TCC controls whether stored alpha participates, including for CT32.
        // RGB-only textures must retain vertex alpha for blending/alpha tests.
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    bool isFourBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    bool isEightBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H;
    }

    // Bilinear blend of four RGBA8 texels, all channels at once.
    uint32_t lerpTexels(uint32_t c00, uint32_t c10, uint32_t c01, uint32_t c11, float fx, float fy)
    {
        auto channels = [](uint32_t c)
        { return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(c)))); };
        const __m128 f00 = channels(c00), f10 = channels(c10), f01 = channels(c01), f11 = channels(c11);
        const __m128 vfx = _mm_set1_ps(fx);
        const __m128 top = _mm_add_ps(f00, _mm_mul_ps(_mm_sub_ps(f10, f00), vfx));
        const __m128 bottom = _mm_add_ps(f01, _mm_mul_ps(_mm_sub_ps(f11, f01), vfx));
        // Round half up by truncation: inputs are blends of 0..255, and the
        // packs clamp like clampU8.
        const __m128 blended = _mm_add_ps(_mm_add_ps(top, _mm_mul_ps(_mm_sub_ps(bottom, top), _mm_set1_ps(fy))),
                                          _mm_set1_ps(0.5f));
        const __m128i words = _mm_packus_epi32(_mm_cvttps_epi32(blended), _mm_setzero_si128());
        return static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_packus_epi16(words, words)));
    }

    using PipeReadFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
    using PipeWriteFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

    // GSCpuBackend::WritePixel with the draw state decoded once per
    // primitive. The triangle and sprite loops only visit owned rows inside
    // the scissor, so the scissor/band-row checks (and WritePixel's
    // fixed-pixel debug prints) are left out; everything else, including the
    // order of reads and writes, is the same.
    struct PixelPipe
    {
        uint8_t *vram = nullptr;
        PipeReadFunc readFrame = nullptr;
        PipeReadFunc readZ = nullptr;
        PipeWriteFunc writeFrame = nullptr;
        PipeWriteFunc writeZ = nullptr;
        // Set for byte-sized formats: one address serves the read and the
        // write of a pixel. Kind 0 = 32-bit, 1 = 24-bit (keeps the top byte),
        // 2 = 16-bit.
        GSMem::PixelAddressFunc frameAddr = nullptr;
        GSMem::PixelAddressFunc zAddr = nullptr;
        uint8_t frameKind = 0, zKind = 0;
        uint32_t fbp = 0, zbp = 0, fbw = 0, fbmsk = 0;
        uint8_t fpsm = 0;
        bool alphaTest = false;
        uint8_t atst = 0, aref = 0;
        PixelWriteMask failMask{};
        bool zmask = false;
        uint8_t ztest = 4; // TEST.ZTST, or 4 when ZTE is off
        bool abe = false, pabe = false;
        uint8_t asel = 0, bsel = 0, csel = 0, dsel = 0, fix = 0;
        bool dateRead = false, date = false, datm = false;
        bool frame16 = false, frameCT24 = false, frameCT32 = false, frameAlphaBit = false;
    };

    void setupPixelPipe(const GSDrawState &state, uint8_t *vram, const PipeReadFunc *reads,
                        const PipeWriteFunc *writes, PixelPipe &p)
    {
        const auto &ctx = state.context;
        p.vram = vram;
        p.fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
        p.zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
        p.fbw = ctx.frame.fbw >= 1u ? ctx.frame.fbw : 10u;
        p.fpsm = ctx.frame.psm;
        p.fbmsk = ctx.frame.fbmsk;
        p.readFrame = reads[ctx.frame.psm & 0x3Fu];
        p.writeFrame = writes[ctx.frame.psm & 0x3Fu];
        p.readZ = reads[ctx.zbuf.psm & 0x3Fu];
        p.writeZ = writes[ctx.zbuf.psm & 0x3Fu];
        const auto kindOf = [](uint8_t psm) -> uint8_t
        {
            switch (psm)
            {
            case GS_PSM_CT24:
            case GS_PSM_Z24:
                return 1u;
            case GS_PSM_CT16:
            case GS_PSM_CT16S:
            case GS_PSM_Z16:
            case GS_PSM_Z16S:
                return 2u;
            default:
                return 0u;
            }
        };
        p.frameAddr = GSMem::GetPixelAddressFunc(ctx.frame.psm);
        p.zAddr = GSMem::GetPixelAddressFunc(ctx.zbuf.psm);
        p.frameKind = kindOf(ctx.frame.psm & 0x3Fu);
        p.zKind = kindOf(ctx.zbuf.psm & 0x3Fu);
        const uint64_t test = ctx.test;
        p.alphaTest = (test & 0x1u) != 0u;
        p.atst = static_cast<uint8_t>((test >> 1) & 0x7u);
        p.aref = static_cast<uint8_t>((test >> 4) & 0xFFu);
        // classifyAlphaTest's result for a failing comparison: ATE on, ATST NEVER.
        p.failMask = classifyAlphaTest((test & ~0xFull) | 0x1u, 0u, p.fpsm);
        p.zmask = ctx.zbuf.zmask;
        p.ztest = ((test >> 16) & 1u) != 0u ? static_cast<uint8_t>((test >> 17) & 3u) : 4u;
        p.abe = state.prim.abe;
        p.pabe = state.pabe;
        const uint64_t alphaReg = ctx.alpha;
        p.asel = alphaReg & 3;
        p.bsel = (alphaReg >> 2) & 3;
        p.csel = (alphaReg >> 4) & 3;
        p.dsel = (alphaReg >> 6) & 3;
        p.fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);
        p.date = ((test >> 14) & 0x1u) != 0u;
        p.datm = ((test >> 15) & 0x1u) != 0u;
        p.frameCT32 = p.fpsm == GS_PSM_CT32;
        p.frameCT24 = p.fpsm == GS_PSM_CT24;
        p.frame16 = bitsPerPixel(p.fpsm) == 16;
        p.dateRead = p.date && (p.frameCT32 || p.fpsm == GS_PSM_CT16 || p.fpsm == GS_PSM_CT16S);
        p.frameAlphaBit = (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24;
    }

    __forceinline bool pipeAlphaPasses(const PixelPipe &p, uint8_t alpha)
    {
        switch (p.atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < p.aref;
        case 3:
            return alpha <= p.aref;
        case 4:
            return alpha == p.aref;
        case 5:
            return alpha >= p.aref;
        case 6:
            return alpha > p.aref;
        default:
            return alpha != p.aref;
        }
    }

    __forceinline uint32_t pipeLoad(const uint8_t *vram, uint32_t address, uint8_t kind)
    {
        if (kind == 2u)
        {
            uint16_t v;
            std::memcpy(&v, vram + address, sizeof(v));
            return v;
        }
        uint32_t v;
        std::memcpy(&v, vram + address, sizeof(v));
        return kind == 1u ? (v & 0x00FFFFFFu) : v;
    }

    __forceinline void pipeStore(uint8_t *vram, uint32_t address, uint8_t kind, uint32_t value)
    {
        if (kind == 2u)
        {
            const uint16_t v = static_cast<uint16_t>(value);
            std::memcpy(vram + address, &v, sizeof(v));
            return;
        }
        if (kind == 1u)
        {
            uint32_t old;
            std::memcpy(&old, vram + address, sizeof(old));
            value = (old & 0xFF000000u) | (value & 0x00FFFFFFu);
        }
        std::memcpy(vram + address, &value, sizeof(value));
    }

    __forceinline void writePixelFast(const PixelPipe &p, int x, int y, uint32_t z,
                                      uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        const PixelWriteMask writeMask = (!p.alphaTest || pipeAlphaPasses(p, a)) ? PixelWriteMask{} : p.failMask;
        if (!writeMask.writesFramebuffer() && (!writeMask.writeDepth || p.zmask))
            return;

        bool zpass = true;
        uint32_t zAddress = 0;
        bool haveZAddress = false;
        if (p.ztest == 0u)
        {
            zpass = false;
        }
        else if (p.ztest == 2u || p.ztest == 3u)
        {
            uint32_t storedZ;
            if (p.zAddr)
            {
                zAddress = p.zAddr(p.zbp, p.fbw, x, y);
                haveZAddress = true;
                storedZ = pipeLoad(p.vram, zAddress, p.zKind);
            }
            else
            {
                storedZ = p.readZ(p.vram, p.zbp, p.fbw, x, y);
            }
            zpass = p.ztest == 2u ? z >= storedZ : z > storedZ;
        }
        if (!zpass)
            return;

        const bool alphaBlendEnabled = p.abe && !(p.pabe && (a & 0x80u) == 0u);
        const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && p.frameCT32;
        const bool frmw = p.dateRead || (writeMask.writesFramebuffer() && ((p.fbmsk != 0) || alphaBlendEnabled || preserveDestinationAlpha));

        u32 rawFramebufferPixel = 0;
        u32 fbrgba = 0;
        uint32_t frameAddress = 0;
        bool haveFrameAddress = false;
        if (frmw)
        {
            if (p.frameAddr)
            {
                frameAddress = p.frameAddr(p.fbp, p.fbw, x, y);
                haveFrameAddress = true;
                rawFramebufferPixel = pipeLoad(p.vram, frameAddress, p.frameKind);
            }
            else
            {
                rawFramebufferPixel = p.readFrame(p.vram, p.fbp, p.fbw, x, y);
            }
            fbrgba = rawFramebufferPixel;
            if (p.frame16)
                fbrgba = Rgba5551ToRgba8888(fbrgba);
            else if (p.frameCT24)
                fbrgba |= 0x80000000u;
        }

        if (p.date)
        {
            if (p.frameCT32 && (((rawFramebufferPixel >> 31) & 0x1u) != 0u) != p.datm)
                return;
            if ((p.fpsm == GS_PSM_CT16 || p.fpsm == GS_PSM_CT16S) && (((rawFramebufferPixel >> 15) & 0x1u) != 0u) != p.datm)
                return;
        }

        if (writeMask.writesFramebuffer())
        {
            if (alphaBlendEnabled)
            {
                const uint8_t dr = fbrgba & 0xFF;
                const uint8_t dg = (fbrgba >> 8) & 0xFF;
                const uint8_t db = (fbrgba >> 16) & 0xFF;
                const uint8_t da = (fbrgba >> 24) & 0xFF;
                const auto pickRGB = [](uint8_t sel, int cs, int cd) -> int
                {
                    if (sel == 0)
                        return cs;
                    if (sel == 1)
                        return cd;
                    return 0;
                };
                const int cAlpha = (p.csel == 0) ? a : (p.csel == 1) ? da : p.fix;
                r = clampU8(((pickRGB(p.asel, r, dr) - pickRGB(p.bsel, r, dr)) * cAlpha >> 7) + pickRGB(p.dsel, r, dr));
                g = clampU8(((pickRGB(p.asel, g, dg) - pickRGB(p.bsel, g, dg)) * cAlpha >> 7) + pickRGB(p.dsel, g, dg));
                b = clampU8(((pickRGB(p.asel, b, db) - pickRGB(p.bsel, b, db)) * cAlpha >> 7) + pickRGB(p.dsel, b, db));
            }

            if (writeMask.writeAlpha && p.frameAlphaBit)
                a = static_cast<uint8_t>(a | 0x80u);

            u32 pixel = pack32(r, g, b, a);
            if (p.fbmsk != 0)
                pixel = (pixel & ~p.fbmsk) | (fbrgba & p.fbmsk);
            if (preserveDestinationAlpha)
                pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
            if (p.frame16)
                pixel = Rgba8888ToRgba5551(pixel);
            if (p.frameAddr)
                pipeStore(p.vram, haveFrameAddress ? frameAddress : p.frameAddr(p.fbp, p.fbw, x, y), p.frameKind, pixel);
            else
                p.writeFrame(p.vram, p.fbp, p.fbw, x, y, pixel);
        }

        if (writeMask.writeDepth && !p.zmask)
        {
            if (p.zAddr)
                pipeStore(p.vram, haveZAddress ? zAddress : p.zAddr(p.zbp, p.fbw, x, y), p.zKind, z);
            else
                p.writeZ(p.vram, p.zbp, p.fbw, x, y, z);
        }
    }

    // GSCpuBackend::SampleTexture/LookupCLUT with TEX0, CLAMP and TEXA
    // decoded once per primitive. Only set up (ready == true) for the texel
    // and CLUT formats the per-texel path handles without diagnostics.
    struct TexturePipe
    {
        bool ready = false;
        const uint8_t *vram = nullptr;
        GSMem::TexturePageCache *cache = nullptr;
        const uint16_t *clut = nullptr;
        GSTexaReg texa{};
        uint32_t tbp0 = 0, tbw = 0;
        uint8_t psm = 0, cpsm = 0;
        uint8_t kind = 0; // 0 direct, 1 16-bit, 2 indexed
        int texW = 1, texH = 1;
        uint8_t wrapU = 0, wrapV = 0;
        uint16_t minU = 0, maxU = 0, minV = 0, maxV = 0;
        bool fst = false, linear = false;
        bool clut16 = false, fourBit = false, csm1Eight = false;
        uint32_t clutBase = 0, clutMask = 0;
    };

    void setupTexturePipe(const GSDrawState &state, const uint8_t *vram, GSMem::TexturePageCache &cache,
                          const uint16_t *clut, TexturePipe &t)
    {
        const auto &ctx = state.context;
        const auto &tex = ctx.tex0;
        t.ready = false;
        switch (tex.psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        case GS_PSM_CT24:
        case GS_PSM_Z24:
            t.kind = 0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            t.kind = 1;
            break;
        case GS_PSM_T8:
        case GS_PSM_T8H:
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            t.kind = 2;
            if (tex.cpsm != GS_PSM_CT32 && tex.cpsm != GS_PSM_CT24 && tex.cpsm != GS_PSM_CT16 && tex.cpsm != GS_PSM_CT16S)
                return;
            break;
        default:
            return;
        }
        t.vram = vram;
        t.cache = &cache;
        t.clut = clut;
        t.texa = state.texa;
        t.tbp0 = tex.tbp0;
        t.tbw = tex.tbw;
        t.psm = tex.psm;
        t.cpsm = tex.cpsm;
        t.texW = state.textureWidth;
        t.texH = state.textureHeight;
        const uint64_t clamp = ctx.clamp;
        t.wrapU = static_cast<uint8_t>(clamp & 0x3u);
        t.wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
        t.minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
        t.maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
        t.minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
        t.maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);
        t.fst = state.prim.fst;
        t.linear = state.linearFilter;
        t.clut16 = tex.cpsm == GS_PSM_CT16 || tex.cpsm == GS_PSM_CT16S;
        t.clutBase = (static_cast<uint32_t>(tex.csa) & (t.clut16 ? 0x1Fu : 0x0Fu)) << 4u;
        t.clutMask = t.clut16 ? 0x1FFu : 0x0FFu;
        t.fourBit = isFourBitIndexedPsm(tex.psm);
        t.csm1Eight = !t.clut16 && tex.csm == 0u && isEightBitIndexedPsm(tex.psm);
        t.ready = true;
    }

    __forceinline uint32_t pipeDecodeTexel(const TexturePipe &t, u32 out)
    {
        switch (t.kind)
        {
        case 0:
            return applyTexa(t.texa, t.psm, out);
        case 1:
            return applyTexa(t.texa, t.psm, Rgba5551ToRgba8888(static_cast<u16>(out)));
        default:
        {
            const uint8_t index = static_cast<u8>(out);
            const uint32_t sourceIndex = t.fourBit ? (static_cast<uint32_t>(index) & 0x0Fu) : static_cast<uint32_t>(index);
            uint32_t clutIndex = (t.clutBase + sourceIndex) & t.clutMask;
            if (t.csm1Eight)
            {
                const uint32_t block = std::min((sourceIndex & 0xF0u) + t.clutBase, 240u);
                clutIndex = block + (sourceIndex & 0x0Fu);
            }
            switch (t.cpsm)
            {
            case GS_PSM_CT32:
                return applyTexa(t.texa, t.cpsm, static_cast<uint32_t>(t.clut[clutIndex]) | (static_cast<uint32_t>(t.clut[clutIndex + 256u]) << 16u));
            case GS_PSM_CT24:
                return applyTexa(t.texa, t.cpsm, (static_cast<uint32_t>(t.clut[clutIndex]) | (static_cast<uint32_t>(t.clut[clutIndex + 256u]) << 16u)) & 0x00FFFFFFu);
            default:
                return applyTexa(t.texa, t.cpsm, Rgba5551ToRgba8888(t.clut[clutIndex]));
            }
        }
        }
    }

    __forceinline uint32_t sampleTextureFast(const TexturePipe &t, float s, float tc, float q, uint16_t u, uint16_t v)
    {
        float texUf, texVf;
        if (t.fst)
        {
            texUf = static_cast<float>(u) / 16.0f;
            texVf = static_cast<float>(v) / 16.0f;
        }
        else
        {
            const float invQ = 1.0f / fabsQ(q);
            texUf = s * invQ * static_cast<float>(t.texW);
            texVf = tc * invQ * static_cast<float>(t.texH);
        }

        if (!t.linear)
        {
            return pipeDecodeTexel(t, GSMem::ReadTexture(*t.cache, t.vram, t.psm, t.tbp0, t.tbw,
                                                         wrapTextureCoordinate(static_cast<int>(texUf), t.texW, t.wrapU, t.minU, t.maxU),
                                                         wrapTextureCoordinate(static_cast<int>(texVf), t.texH, t.wrapV, t.minV, t.maxV)));
        }

        const float sampleU = texUf - 0.5f;
        const float sampleV = texVf - 0.5f;
        const int u0 = floorToInt(sampleU);
        const int v0 = floorToInt(sampleV);
        const float fx = sampleU - static_cast<float>(u0);
        const float fy = sampleV - static_cast<float>(v0);
        const u32 wu0 = static_cast<u32>(wrapTextureCoordinate(u0, t.texW, t.wrapU, t.minU, t.maxU));
        const u32 wu1 = static_cast<u32>(wrapTextureCoordinate(u0 + 1, t.texW, t.wrapU, t.minU, t.maxU));
        const u32 wv0 = static_cast<u32>(wrapTextureCoordinate(v0, t.texH, t.wrapV, t.minV, t.maxV));
        const u32 wv1 = static_cast<u32>(wrapTextureCoordinate(v0 + 1, t.texH, t.wrapV, t.minV, t.maxV));

        u32 raw[4] = {};
        GSMem::ReadTexture4(*t.cache, t.vram, t.psm, t.tbp0, t.tbw, {wu0, wu1, wu0, wu1}, {wv0, wv0, wv1, wv1}, raw);
        return lerpTexels(pipeDecodeTexel(t, raw[0]), pipeDecodeTexel(t, raw[1]),
                          pipeDecodeTexel(t, raw[2]), pipeDecodeTexel(t, raw[3]), fx, fy);
    }
}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        const uint32_t maghDiv = magh + 1u;
        const uint32_t pixelSpan = dw + 1u;

        // Corrupted or sign-extended display register (e.g. 0xFFFFFFFFFF... bitfields all 1s)
        if (dw == 0x0FFFu || dh == 0x07FFu || dw == 0u || dh == 0u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
            return;
        }

        // DISPLAY DW/DH are VCK counts minus one. Integer division of a 1-pixel-short
        // DW (2559 vs 2556, or DW=639 with MAGH=0 treated as width) shears the host copy.
        outWidth = pixelSpan / maghDiv;
        if (outWidth == 639u)
            outWidth = 640u;
        outHeight = dh + 1u;
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    void repairDisplayFrame(GSFrameReg &frame, uint32_t width, const GSFrameReg *contextFrames, size_t contextCount)
    {
        const uint32_t minFbw = std::max<uint32_t>(1u, (width + 63u) / 64u);
        static uint32_t s_repairLogs = 0u;
        if (s_repairLogs < 20u && frame.fbw < minFbw)
        {
            std::cerr << "[repair:frame] fbp=" << frame.fbp << " fbw=" << frame.fbw << " psm=" << (unsigned)frame.psm;
            for (size_t i = 0; contextFrames && i < contextCount; ++i)
            {
                std::cerr << " ctx" << i << "=(fbp=" << contextFrames[i].fbp << ",fbw=" << contextFrames[i].fbw << ",psm=" << (unsigned)contextFrames[i].psm << ")";
            }
            std::cerr << std::endl;
            ++s_repairLogs;
        }
        if (frame.fbw >= minFbw)
            return;
        if (contextFrames)
        {
            for (size_t i = 0; i < contextCount; ++i)
            {
                const GSFrameReg &candidate = contextFrames[i];
                if (candidate.fbp == frame.fbp && candidate.fbw >= minFbw)
                {
                    frame.fbw = candidate.fbw;
                    frame.psm = candidate.psm;
                    return;
                }
            }
        }
        // DISPFB FBW=1 with a 640-wide display wraps every 64 pixels (the swizzle/comb).
        frame.fbw = minFbw;
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    // Sign-extended 32-bit stores / bad PutDispEnv payloads produce
    // DISPLAY = 0xFFFFFFFFFFxxxxxx (DW=0xFFF, DH=0x7FF). Treating those as
    // valid makes Present prefer CRT1+DISPFB1=0 over a still-good CRT2.
    bool isCorruptDisplay(uint64_t display64)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        return dw == 0x0FFFu || dh == 0x07FFu;
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        if (isCorruptDisplay(display64))
            return false;
        if (frame.fbw == 0u)
            return false;
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
                sourceY = height - 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        source.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }

    // True when nearly every sampled pixel matches the first (cleared draw buffer).
    bool looksLikeFlatFill(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        if (pixels.empty() || width == 0u || height == 0u)
            return true;
        const uint8_t r0 = pixels[0], g0 = pixels[1], b0 = pixels[2];
        uint32_t samples = 0u, mismatches = 0u;
        for (uint32_t y = 0; y < height; y += 8u)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; x += 8u)
            {
                const uint8_t *p = row + x * 4u;
                ++samples;
                const int dr = std::abs(static_cast<int>(p[0]) - static_cast<int>(r0));
                const int dg = std::abs(static_cast<int>(p[1]) - static_cast<int>(g0));
                const int db = std::abs(static_cast<int>(p[2]) - static_cast<int>(b0));
                if (dr > 6 || dg > 6 || db > 6)
                    ++mismatches;
            }
        }
        return samples == 0u || mismatches * 20u < samples; // <5% differ → flat
    }
}

GSCpuBackend::GSCpuBackend()
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            m_readVramFuncs[i] = ReadNull;
            m_writeVramFuncs[i] = WriteNull;
            break;
        }
    }
    Reset();

    m_legacyTextureFlush = std::getenv("ATV_GS_LEGACY_TEXFLUSH") != nullptr;
    m_uploadNoFlush = std::getenv("ATV_GS_UPLOAD_FLUSH") == nullptr;
    m_profileBands = std::getenv("ATV_GS_PROFILE_BANDS") != nullptr;
    m_bandPipeline = std::getenv("ATV_GS_BAND_PIPELINE") != nullptr;
    // ATV2 default (ATV1 sets ATV_GS_BANDS=12 in play_game.bat): up to 12 raster bands, half the
    // host threads. ATV_GS_BANDS=1 rasterizes inline.
    if (const char *batch = std::getenv("ATV_GS_BATCH_SIZE"))
        m_bandBatchLimit = static_cast<size_t>(std::clamp<long>(std::strtol(batch, nullptr, 10), 16, 4096));
    const char *bands = std::getenv("ATV_GS_BANDS");
    const long requested = bands ? std::strtol(bands, nullptr, 10)
                                 : std::clamp<long>(static_cast<long>(std::thread::hardware_concurrency() / 2u), 1, 12);
    // The opt-in TBP census keeps an unsynchronized map in DrawTriangle.
    if (requested > 1 && std::getenv("ATV_TRACE_TBP_CENSUS") == nullptr)
        m_bandCount = static_cast<uint32_t>(std::min<long>(requested, 16));
}

void GSCpuBackend::StartBandThreads()
{
    for (uint32_t band = 1u; band < m_bandCount; ++band)
        m_bandCaches.push_back(std::make_unique<GSMem::TexturePageCache>());
    for (uint32_t band = m_bandPipeline ? 0u : 1u; band < m_bandCount; ++band)
        m_bandThreads.emplace_back([this, band]
                                   { BandWorker(band); });
    std::fprintf(stderr, "[gs] banded rasterization: %u bands%s\n", m_bandCount, m_bandPipeline ? " (pipelined)" : "");
}

GSCpuBackend::~GSCpuBackend()
{
    {
        std::lock_guard<std::mutex> lock(m_bandMutex);
        m_bandStop = true;
    }
    m_bandStart.notify_all();
    for (std::thread &thread : m_bandThreads)
        thread.join();
}

void GSCpuBackend::BandWorker(uint32_t band)
{
    t_bandIndex = band;
    t_bandCount = m_bandCount;
    t_bandCache = band == 0u ? nullptr : m_bandCaches[band - 1u].get();
    uint64_t seenGeneration = 0;
    for (;;)
    {
        {
            std::unique_lock<std::mutex> lock(m_bandMutex);
            m_bandStart.wait(lock, [&]
                             { return m_bandStop || m_bandGeneration != seenGeneration; });
            if (m_bandStop)
                return;
            seenGeneration = m_bandGeneration;
        }
        DrawBand(band);
        {
            std::lock_guard<std::mutex> lock(m_bandMutex);
            if (--m_bandRemaining == 0u)
                m_bandDone.notify_one();
        }
    }
}

void GSCpuBackend::DrawBand(uint32_t band)
{
    // The calling thread may service another backend with a different band
    // count. Do not reuse its last row-ownership proof across batches.
    t_ownedBandRow = -1;
    const uint64_t start = m_profileBands ? bandClockNs() : 0;
    auto &cache = band == 0u ? m_texturePageCache : *m_bandCaches[band - 1u];
    size_t index = 0, nextFlush = 0, nextClut = 0;
    t_drawClut = m_bandInFlightClutStart.data();
    for (const GSPrimitiveBatch &batch : m_bandInFlight)
    {
        if (nextFlush < m_bandInFlightTextureFlushes.size() && m_bandInFlightTextureFlushes[nextFlush] == index)
        {
            cache.Invalidate();
            ++nextFlush;
        }
        while (nextClut < m_bandInFlightClutChanges.size() && m_bandInFlightClutChanges[nextClut].first == index)
            t_drawClut = m_bandInFlightClutChanges[nextClut++].second.data();
        if (bandMayDrawPrimitive(batch))
            DrawPrimitive(batch);
        ++index;
    }
    t_drawClut = nullptr;
    if (nextFlush < m_bandInFlightTextureFlushes.size())
        cache.Invalidate(); // TEXFLUSH after the final queued primitive.
    if (m_profileBands)
        m_bandDrawNs[band] = bandClockNs() - start;
}

void GSCpuBackend::WaitBandsIdle() const
{
    if (!m_bandInFlightActive)
        return;
    const uint64_t waitStart = m_profileBands ? bandClockNs() : 0;
    {
        std::unique_lock<std::mutex> lock(m_bandMutex);
        m_bandDone.wait(lock, [&]
                        { return m_bandRemaining == 0u; });
    }
    m_bandInFlightActive = false;
    if (m_profileBands)
    {
        m_bandWaitNs += bandClockNs() - waitStart;
        for (uint32_t i = 0; i < m_bandCount; ++i)
            m_bandAccumulatedNs[i] += m_bandDrawNs[i];
    }
}

void GSCpuBackend::FlushBands(BandFlushReason reason) const
{
    WaitBandsIdle();
    if (m_bandPending.empty())
        return;
    const uint64_t start = m_profileBands ? bandClockNs() : 0;
    m_bandInFlight.swap(m_bandPending);
    m_bandInFlightTextureFlushes.swap(m_bandTextureFlushes);
    m_bandPending.clear();
    m_bandTextureFlushes.clear();
    m_bandInFlightClutStart = m_bandPendingClutStart;
    m_bandInFlightClutChanges.swap(m_bandClutChanges);
    m_bandClutChanges.clear();
    m_bandInFlightWrites.swap(m_bandWrites);
    m_bandWrites.clear();
    m_lastBandKeyValid = false;
    m_bandInFlightReads.swap(m_bandReads);
    m_bandReads.clear();
    {
        std::lock_guard<std::mutex> lock(m_bandMutex);
        m_bandRemaining = m_bandPipeline ? m_bandCount : m_bandCount - 1u;
        ++m_bandGeneration;
    }
    m_bandInFlightActive = true;
    m_bandStart.notify_all();

    if (!m_bandPipeline)
    {
        // Band 0 runs on the calling thread with the shared texture page cache.
        t_bandIndex = 0u;
        t_bandCount = m_bandCount;
        const_cast<GSCpuBackend *>(this)->DrawBand(0u);
        t_bandCount = 1u;
    }
    // Only a full batch may still be drawing when the next one is queued:
    // the calling thread goes back to queueing while bands 1..N-1 finish,
    // and everything that touches VRAM, the CLUT or the caches waits for
    // them first. ATV_GS_BAND_BATCH_WAIT=1 waits here as before session 33.
    static const bool s_batchWait = std::getenv("ATV_GS_BAND_BATCH_WAIT") != nullptr;
    if (reason != BandFlushReason::Batch || (s_batchWait && !m_bandPipeline))
        WaitBandsIdle();
    if (m_profileBands)
    {
        const uint64_t now = bandClockNs();
        if (m_bandProfileStartNs == 0)
            m_bandProfileStartNs = start;
        ++m_bandFlushCounts[static_cast<size_t>(reason)];
        m_bandPrimitiveCount += m_bandInFlight.size();
        m_bandFlushNs += now - start;
        if (now - m_bandProfileStartNs >= 10000000000ull)
        {
            static constexpr const char *names[] = {"other", "alias", "batch", "hazard", "clut", "texture", "sync", "read", "write", "upload", "clear"};
            std::fprintf(stderr, "[gs-bands] wall_ms=%.3f flush_ms=%.3f wait_ms=%.3f primitives=%llu",
                         (now - m_bandProfileStartNs) / 1e6, m_bandFlushNs / 1e6, m_bandWaitNs / 1e6,
                         static_cast<unsigned long long>(m_bandPrimitiveCount));
            for (size_t i = 0; i < m_bandFlushCounts.size(); ++i)
                std::fprintf(stderr, " %s=%llu", names[i], static_cast<unsigned long long>(m_bandFlushCounts[i]));
            for (uint32_t i = 0; i < m_bandCount; ++i)
                std::fprintf(stderr, " band%u_ms=%.3f", i, m_bandAccumulatedNs[i] / 1e6);
            std::fprintf(stderr, "\n");
            m_bandProfileStartNs = now;
            m_bandFlushNs = m_bandWaitNs = m_bandPrimitiveCount = 0;
            m_bandFlushCounts.fill(0);
            m_bandAccumulatedNs.fill(0);
        }
    }
}

void GSCpuBackend::QueueBandedPrimitive(const GSPrimitiveBatch &batch)
{
    if (m_bandThreads.empty())
        StartBandThreads();

    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    BandKey key;
    key.clamp = ctx.clamp;
    key.fbp = ctx.frame.fbp;
    key.fbw = ctx.frame.fbw;
    key.fpsm = ctx.frame.psm;
    key.rows = ctx.scissor.y1;
    key.zbp = ctx.zbuf.zbp;
    key.zpsm = ctx.zbuf.psm;
    key.tbp0 = ctx.tex0.tbp0;
    key.tbw = ctx.tex0.tbw;
    key.tpsm = ctx.tex0.psm;
    key.texW = state.textureWidth;
    key.texH = state.textureHeight;
    key.flags = (ctx.zbuf.zmask ? 1u : 0u) | (state.prim.tme ? 2u : 0u);
    if (m_lastBandKeyValid && key == m_lastBandKey && m_bandPending.size() < m_bandBatchLimit)
    {
        if (m_bandPending.empty())
        {
            m_bandPendingClutStart = m_clut;
            m_bandClutChanges.clear();
        }
        m_bandPending.push_back(batch);
        return;
    }
    const uint32_t fbw = ctx.frame.fbw >= 1u ? ctx.frame.fbw : 10u;
    const uint32_t rows = static_cast<uint32_t>(ctx.scissor.y1) + 1u;
    const BandWrite frame = surfaceRange<BandWrite>(ctx.frame.fbp * kVramPageBytes, fbw * 64u, ctx.frame.psm, rows,
                                                    (ctx.frame.fbp << 16) | (fbw << 8) | ctx.frame.psm, 0u);
    const bool writesZ = !ctx.zbuf.zmask;
    const BandWrite depth = surfaceRange<BandWrite>(ctx.zbuf.zbp * kVramPageBytes, fbw * 64u, ctx.zbuf.psm, rows,
                                                    0x80000000u | (ctx.zbuf.zbp << 16) | (fbw << 8) | ctx.zbuf.psm, 0u);
    BandWrite texture{};
    const bool textured = state.prim.tme;
    if (textured)
    {
        const uint32_t texWidth = std::max<uint32_t>(ctx.tex0.tbw * 64u, static_cast<uint32_t>(state.textureWidth));
        texture = surfaceRange<BandWrite>(ctx.tex0.tbp0 * kVramBlockBytes, texWidth, ctx.tex0.psm,
                                          std::max<uint32_t>(state.textureHeight, 1u), 0u, kVramPageBytes);
    }

    // A draw that samples its own render target sees rows it has already
    // drawn, and aliased depth/colour buffers map one pixel's writes onto
    // another pixel's rows; only a single sequential pass reproduces those.
    if ((textured && (overlaps(texture, frame) || (writesZ && overlaps(texture, depth)))) ||
        (writesZ && overlaps(frame, depth)))
    {
        FlushBands(BandFlushReason::Alias);
        DrawPrimitive(batch);
        return;
    }

    const bool batchFull = m_bandPending.size() >= m_bandBatchLimit;
    bool hazard = batchFull;
    bool samplesPendingWrite = false;
    for (const BandWrite &write : m_bandWrites)
    {
        // Sampling memory drawn earlier in the batch, or addressing it with a
        // different layout, crosses band boundaries.
        samplesPendingWrite = samplesPendingWrite || (textured && overlaps(texture, write));
        hazard = hazard ||
                 (overlaps(frame, write) && write.layoutKey != frame.layoutKey) ||
                 (writesZ && overlaps(depth, write) && write.layoutKey != depth.layoutKey);
    }
    if (hazard || samplesPendingWrite)
        FlushBands(batchFull ? BandFlushReason::Batch : BandFlushReason::Hazard);

    if (m_bandPending.empty())
    {
        m_bandPendingClutStart = m_clut;
        m_bandClutChanges.clear();
    }
    m_bandPending.push_back(batch);
    m_lastBandKey = key;
    m_lastBandKeyValid = true;
    const auto recordWrite = [&](const BandWrite &range)
    {
        for (BandWrite &write : m_bandWrites)
        {
            if (write.layoutKey == range.layoutKey && write.begin == range.begin)
            {
                write.end = std::max(write.end, range.end);
                return;
            }
        }
        m_bandWrites.push_back(range);
    };
    recordWrite(frame);
    if (writesZ)
        recordWrite(depth);

    if (textured && m_uploadNoFlush)
    {
        // REGION_CLAMP/REGION_REPEAT coordinates reach up to 1023 whatever
        // the texture size. Whole pages: the band caches copy pages.
        const uint64_t clamp = ctx.clamp;
        const uint32_t width = std::max<uint32_t>({ctx.tex0.tbw * 64u, static_cast<uint32_t>(state.textureWidth),
                                                    (clamp & 0x2u) != 0u ? 1024u : 0u});
        const uint32_t height = std::max<uint32_t>({static_cast<uint32_t>(state.textureHeight), 1u,
                                                     ((clamp >> 2) & 0x2u) != 0u ? 1024u : 0u});
        BandWrite read = surfaceRange<BandWrite>(ctx.tex0.tbp0 * kVramBlockBytes, width, ctx.tex0.psm, height, 0u,
                                                 kVramPageBytes);
        read.begin &= ~(kVramPageBytes - 1u);
        if (read.end != UINT32_MAX)
            read.end = (read.end + kVramPageBytes - 1u) & ~(kVramPageBytes - 1u);
        bool merged = false;
        for (BandWrite &pending : m_bandReads)
        {
            if (pending.begin == read.begin)
            {
                pending.end = std::max(pending.end, read.end);
                merged = true;
                break;
            }
        }
        if (!merged)
            m_bandReads.push_back(read);
    }
}

bool GSCpuBackend::InFlightTouches(const BandWrite &range) const
{
    // Pipelined bands run band 0 off-thread with the shared page cache, which
    // uploads touch: they always wait.
    if (m_bandPipeline)
        return m_bandInFlightActive;
    if (!m_bandInFlightActive)
        return false;
    for (const BandWrite &write : m_bandInFlightWrites)
        if (overlaps(range, write))
            return true;
    for (const BandWrite &read : m_bandInFlightReads)
        if (overlaps(range, read))
            return true;
    return false;
}

bool GSCpuBackend::PendingTouches(const BandWrite &range) const
{
    for (const BandWrite &write : m_bandWrites)
        if (overlaps(range, write))
            return true;
    for (const BandWrite &read : m_bandReads)
        if (overlaps(range, read))
            return true;
    return false;
}

void GSCpuBackend::MaybeStartCapture()
{
    GSCaptureState &c = gsCapture();
    if (c.owner || c.finished || !m_vram || m_vramSize < GSMem::MEMORY_SIZE || c.racePrimitives < c.skip)
        return;
    FlushBands();
    c.file = std::fopen(std::getenv("ATV_GS_CAPTURE"), "wb");
    c.owner = this;
    if (!c.file)
    {
        c.finished = true;
        std::fprintf(stderr, "[gs-capture] cannot open %s\n", std::getenv("ATV_GS_CAPTURE"));
        return;
    }
    const char magic[8] = {'G', 'S', 'C', 'A', 'P', '1', 0, 0};
    const uint32_t sizes[4] = {static_cast<uint32_t>(GSMem::MEMORY_SIZE), static_cast<uint32_t>(sizeof(GSPrimitiveBatch)),
                               static_cast<uint32_t>(sizeof(GSContext)), m_bandCount};
    std::fwrite(magic, 1u, sizeof(magic), c.file);
    std::fwrite(sizes, 1u, sizeof(sizes), c.file);
    std::fwrite(m_vram, 1u, GSMem::MEMORY_SIZE, c.file);
    std::fwrite(m_clut.data(), sizeof(m_clut[0]), m_clut.size(), c.file);
    std::fwrite(m_clutCbp.data(), sizeof(m_clutCbp[0]), m_clutCbp.size(), c.file);
    std::fprintf(stderr, "[gs-capture] started after %llu race primitives, %llu commands\n",
                 static_cast<unsigned long long>(c.racePrimitives), static_cast<unsigned long long>(c.remaining));
}

void GSCpuBackend::RestoreClutState(const std::array<uint16_t, 512> &clut, const std::array<uint32_t, 2> &cbp)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushBands();
    m_clut = clut;
    m_clutCbp = cbp;
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    if (vram && vramSize < GSMem::MEMORY_SIZE)
        throw std::invalid_argument("GS CPU backend requires at least 4 MiB of VRAM");

    std::lock_guard<std::mutex> lock(m_mutex);
    FlushBands();
    m_vram = vram;
    m_vramSize = vramSize;
    ResetUnlocked();
}

void GSCpuBackend::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushBands();
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    m_clut.fill(0u);
    m_clutCbp.fill(0u);
    m_texturePageCache.Invalidate();
    for (auto &cache : m_bandCaches)
        cache->Invalidate();
    m_transfer = {};
    m_transfer.direction = 3u;
    m_uploadRangeValid = false;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
}

namespace
{
    // ATV_TRACE_PRIM_STATS=1: every ~5 s of a race, primitives per second by
    // type, how many are degenerate, sub-pixel or outside the scissor, and how
    // many repeat one of the previous 256 primitives exactly (same vertices and
    // state) or only in screen geometry (another pass over the same triangle).
    void tracePrimStats(const GSPrimitiveBatch &batch)
    {
        static const bool s_enabled = std::getenv("ATV_TRACE_PRIM_STATS") != nullptr;
        if (!s_enabled || !g_raceLoadStarted)
            return;
        struct Stats
        {
            uint64_t total = 0, byType[8]{}, zeroArea = 0, subPixel = 0, offScissor = 0, exactRepeat = 0,
                     geoRepeat = 0, textured = 0, alphaBlend = 0;
            uint64_t geo[256]{}, full[256]{};
            uint32_t ring = 0;
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        };
        static Stats st;
        const auto mix = [](uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull; };
        const auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return static_cast<uint64_t>(u); };
        const GSDrawState &state = batch.state;
        const auto &ctx = state.context;
        const uint32_t n = std::min<uint32_t>(batch.vertexCount, 3u);
        uint64_t geo = 0xcbf29ce484222325ull;
        uint64_t full = geo;
        float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
        const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        for (uint32_t i = 0; i < n; ++i)
        {
            const GSVertex &v = batch.vertices[i];
            geo = mix(mix(mix(geo, bits(v.x)), bits(v.y)), bits(static_cast<float>(v.z)));
            full = mix(mix(mix(mix(full, bits(v.s)), bits(v.t)), (uint64_t(v.u) << 16) | v.v),
                       (uint64_t(v.r) << 24) | (uint64_t(v.g) << 16) | (uint64_t(v.b) << 8) | v.a);
            minX = std::min(minX, v.x - ofx);
            maxX = std::max(maxX, v.x - ofx);
            minY = std::min(minY, v.y - ofy);
            maxY = std::max(maxY, v.y - ofy);
        }
        full = mix(mix(mix(mix(full ^ geo, ctx.frame.fbp), ctx.tex0.tbp0), ctx.alpha), ctx.test);
        full = mix(full, (uint64_t(state.prim.type) << 8) | (state.prim.tme << 1) | state.prim.abe);

        ++st.total;
        ++st.byType[state.prim.type & 7u];
        st.textured += state.prim.tme;
        st.alphaBlend += state.prim.abe;
        if (maxX < ctx.scissor.x0 || minX > ctx.scissor.x1 + 1.0f || maxY < ctx.scissor.y0 || minY > ctx.scissor.y1 + 1.0f)
            ++st.offScissor;
        if (n == 3u && state.prim.type >= GS_PRIM_TRIANGLE && state.prim.type <= GS_PRIM_TRIFAN)
        {
            const GSVertex &a = batch.vertices[0], &b = batch.vertices[1], &c = batch.vertices[2];
            const float area = std::fabs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) * 0.5f;
            if (area == 0.0f)
                ++st.zeroArea;
            else if (area < 1.0f)
                ++st.subPixel;
        }
        bool exact = false, sameGeo = false;
        for (uint32_t i = 0; i < 256u; ++i)
        {
            exact = exact || st.full[i] == full;
            sameGeo = sameGeo || st.geo[i] == geo;
        }
        st.exactRepeat += exact;
        st.geoRepeat += !exact && sameGeo;
        st.geo[st.ring & 255u] = geo;
        st.full[st.ring & 255u] = full;
        ++st.ring;

        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - st.start).count();
        if (secs < 5.0)
            return;
        const double k = 1.0 / secs;
        std::fprintf(stderr,
                     "[prim-stats] prims/s=%.0f tri=%.0f strip=%.0f fan=%.0f sprite=%.0f line=%.0f point=%.0f "
                     "textured=%.1f%% blend=%.1f%% zeroArea=%.1f%% subPixel=%.1f%% offScissor=%.1f%% "
                     "exactRepeat=%.1f%% geoRepeat=%.1f%%\n",
                     st.total * k, st.byType[GS_PRIM_TRIANGLE] * k, st.byType[GS_PRIM_TRISTRIP] * k,
                     st.byType[GS_PRIM_TRIFAN] * k, st.byType[GS_PRIM_SPRITE] * k,
                     (st.byType[GS_PRIM_LINE] + st.byType[GS_PRIM_LINESTRIP]) * k, st.byType[GS_PRIM_POINT] * k,
                     100.0 * st.textured / st.total, 100.0 * st.alphaBlend / st.total, 100.0 * st.zeroArea / st.total,
                     100.0 * st.subPixel / st.total, 100.0 * st.offScissor / st.total,
                     100.0 * st.exactRepeat / st.total, 100.0 * st.geoRepeat / st.total);
        const uint32_t ring = st.ring;
        uint64_t keepGeo[256], keepFull[256];
        std::memcpy(keepGeo, st.geo, sizeof(keepGeo));
        std::memcpy(keepFull, st.full, sizeof(keepFull));
        st = Stats{};
        std::memcpy(st.geo, keepGeo, sizeof(keepGeo));
        std::memcpy(st.full, keepFull, sizeof(keepFull));
        st.ring = ring;
    }
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    tracePrimStats(batch);
    if (gsCapture().enabled)
    {
        if (g_raceLoadStarted && !gsCapture().owner)
            ++gsCapture().racePrimitives;
        if (gsCaptureActive(this))
            gsCaptureWrite(GSCaptureSubmit, &batch, sizeof(batch));
    }
    if (!m_vram || batch.vertexCount == 0u)
        return;
    if (m_bandCount > 1u)
        QueueBandedPrimitive(batch);
    else
        DrawPrimitive(batch);
}

void GSCpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this))
    {
        struct { GSTex0Reg tex0; GSTexClutReg texclut; } record{tex0, texclut};
        gsCaptureWrite(GSCaptureLoadClut, &record, sizeof(record));
    }
    if (!m_vram || (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm)))
        return;

    switch (tex0.cld)
    {
    case 0u:
    case 6u:
    case 7u:
        return;
    case 1u:
        break;
    case 2u:
        m_clutCbp[0] = tex0.cbp;
        break;
    case 3u:
        m_clutCbp[1] = tex0.cbp;
        break;
    case 4u:
        if (m_clutCbp[0] == tex0.cbp)
            return;
        m_clutCbp[0] = tex0.cbp;
        break;
    case 5u:
        if (m_clutCbp[1] == tex0.cbp)
            return;
        m_clutCbp[1] = tex0.cbp;
        break;
    default:
        return;
    }

    if (m_bandCount <= 1u)
    {
        LoadClutUnlocked(tex0, texclut);
        return;
    }

    // Queued and in-flight draws use their CLUT snapshots (m_bandPendingClut*, m_bandInFlightClut*),
    // so the load only has to read current VRAM (LoadClutUnlocked reads it directly): let queued or
    // in-flight draws that write the palette's source finish first. ATV_GS_CLUT_FLUSH=1 flushes on
    // every palette change as before.
    static const bool s_clutFlush = std::getenv("ATV_GS_CLUT_FLUSH") != nullptr;
    const bool csm2 = tex0.csm != 0u;
    BandWrite source{};
    if (csm2)
        source = surfaceRange<BandWrite>(tex0.cbp * kVramBlockBytes, std::max<uint32_t>(texclut.cbw, 1u) * 64u,
                                         tex0.cpsm, static_cast<uint32_t>(texclut.cov) + 1u, 0u, kVramPageBytes);
    else
    {
        // A CSM1 palette (16x16 or 8x2 entries) spans at most four consecutive blocks. The page
        // range surfaceRange gives wraps for the usual spot at the end of VRAM (CBP 0x3FFC) and
        // would overlap every queued draw.
        source.begin = tex0.cbp * kVramBlockBytes;
        source.end = std::min<uint32_t>(source.begin + 4u * kVramBlockBytes, kVramBytes);
    }
    bool flush = false;
    for (const BandWrite &write : m_bandWrites)
        flush = flush || overlaps(source, write);
    if (flush)
    {
        FlushBands(BandFlushReason::Clut);
        WaitBandsIdle();
    }
    else if (InFlightTouches(source))
        WaitBandsIdle();

    const ClutTable previous = m_clut;
    LoadClutUnlocked(tex0, texclut);
    if (m_bandPending.empty() || m_clut == previous)
        return;
    if (s_clutFlush)
    {
        const ClutTable loaded = m_clut;
        m_clut = previous;
        FlushBands(BandFlushReason::Clut);
        m_clut = loaded;
        return;
    }
    if (!m_bandClutChanges.empty() && m_bandClutChanges.back().first == m_bandPending.size())
        m_bandClutChanges.back().second = m_clut;
    else
        m_bandClutChanges.emplace_back(m_bandPending.size(), m_clut);
}

void GSCpuBackend::LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return;

    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;

    const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwoBit && !fourBit;
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;

    for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
    {
        uint32_t sourceX = 0u;
        uint32_t sourceY = 0u;
        uint32_t sourceWidth = 1u;

        if (tex0.csm == 0u)
        {
            const uint32_t sourceIndex = swizzleClutIndexCSM1(entry);
            sourceX = sourceIndex & 0x0Fu;
            sourceY = sourceIndex >> 4u;
        }
        else
        {
            sourceWidth = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
            sourceX = (static_cast<uint32_t>(texclut.cou) << 4u) + entry;
            sourceY = static_cast<uint32_t>(texclut.cov);
        }

        // Straight from VRAM, not the shared texture page cache (stale until queued TEXFLUSHes run).
        const uint32_t raw = ReadVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY);
        const uint32_t destination = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteenBit ? 0x1FFu : 0x0FFu);
        if (sixteenBit)
        {
            m_clut[destination] = static_cast<uint16_t>(raw);
        }
        else
        {
            m_clut[destination] = static_cast<uint16_t>(raw & 0xFFFFu);
            m_clut[destination + 256u] = static_cast<uint16_t>(raw >> 16u);
        }
    }
}

void GSCpuBackend::Flush()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this))
        gsCaptureWrite(GSCaptureFlush, nullptr, 0u);
    FlushBands();
}

void GSCpuBackend::TextureFlush()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this))
        gsCaptureWrite(GSCaptureTextureFlush, nullptr, 0u);
    if (m_legacyTextureFlush)
        FlushBands(BandFlushReason::Texture);
    else if (!m_bandPending.empty())
    {
        if (m_bandTextureFlushes.empty() || m_bandTextureFlushes.back() != m_bandPending.size())
            m_bandTextureFlushes.push_back(m_bandPending.size());
        return;
    }
    WaitBandsIdle();
    m_texturePageCache.Invalidate();
    for (auto &cache : m_bandCaches)
        cache->Invalidate();
}

void GSCpuBackend::Sync(GSSyncReason reason)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this))
        gsCaptureWrite(GSCaptureSync, &reason, sizeof(reason));
    FlushBands(BandFlushReason::Sync);
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this))
    {
        const uint32_t record[5] = {psm, base, bw, x, y};
        gsCaptureWrite(GSCaptureReadVram, record, sizeof(record));
    }
    FlushBands(BandFlushReason::Read);
    return ReadVramUnlocked(psm, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    if (!m_vram)
        return 0u;

    return GSMem::ReadTexture(t_bandCache ? *t_bandCache : m_texturePageCache, m_vram, psm, base, bw, x, y);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this))
    {
        const uint32_t record[6] = {psm, base, bw, x, y, value};
        gsCaptureWrite(GSCaptureWriteVram, record, sizeof(record));
    }
    FlushBands(BandFlushReason::Write);
    WriteVramUnlocked(psm, base, bw, x, y, value);
}

void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (!m_vram)
        return;
    m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value);
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this))
        gsCaptureWrite(GSCaptureSnapshot, nullptr, 0u);
    FlushBands();
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    // ATV_TRACE_GS_RECT=x0,y0,x1,y1[,afterSeconds[,maxSize]]: in-race primitives whose
    // vertices touch that framebuffer rectangle, a few lines per distinct
    // frame/texture/CLUT setup. Optionally only afterSeconds after the first race
    // draw, and only primitives whose bounding box is at most maxSize pixels.
    {
        static int s_rect[6] = {0, 0, -1, -1, 0, 100000};
        static const bool s_traceRect = []
        {
            const char *env = std::getenv("ATV_TRACE_GS_RECT");
            return env && std::sscanf(env, "%d,%d,%d,%d,%d,%d", &s_rect[0], &s_rect[1], &s_rect[2],
                                      &s_rect[3], &s_rect[4], &s_rect[5]) >= 4;
        }();
        static std::atomic<int64_t> s_rectRaceStartMs{-1};
        bool rectWindowOpen = false;
        if (s_traceRect && g_raceLoadStarted)
        {
            const int64_t nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();
            int64_t expected = -1;
            s_rectRaceStartMs.compare_exchange_strong(expected, nowMs);
            rectWindowOpen = nowMs - s_rectRaceStartMs.load() >= static_cast<int64_t>(s_rect[4]) * 1000;
        }
        if (rectWindowOpen && batch.vertexCount > 0u)
        {
            const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4);
            const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
            float minX = 1.0e9f, minY = 1.0e9f, maxX = -1.0e9f, maxY = -1.0e9f;
            for (uint32_t i = 0; i < batch.vertexCount; ++i)
            {
                minX = std::min(minX, batch.vertices[i].x - ofx);
                maxX = std::max(maxX, batch.vertices[i].x - ofx);
                minY = std::min(minY, batch.vertices[i].y - ofy);
                maxY = std::max(maxY, batch.vertices[i].y - ofy);
            }
            static std::mutex s_rectMutex;
            static std::unordered_map<uint64_t, uint32_t> s_rectKeys;
            static uint32_t s_rectLines = 0u;
            if (maxX >= s_rect[0] && minX <= s_rect[2] && maxY >= s_rect[1] && minY <= s_rect[3] &&
                maxX - minX <= static_cast<float>(s_rect[5]) && maxY - minY <= static_cast<float>(s_rect[5]))
            {
                std::lock_guard<std::mutex> lock(s_rectMutex);
                const uint64_t key = (static_cast<uint64_t>(ctx.frame.fbp & 0x1FFu) << 55) |
                                     (static_cast<uint64_t>(ctx.tex0.tbp0 & 0x3FFFu) << 41) |
                                     (static_cast<uint64_t>(ctx.tex0.cbp & 0x3FFFu) << 27) |
                                     (static_cast<uint64_t>(ctx.tex0.psm & 0x3Fu) << 21) |
                                     (static_cast<uint64_t>(state.prim.type) << 8) |
                                     (static_cast<uint64_t>(state.prim.tme) << 1) |
                                     static_cast<uint64_t>(state.prim.abe);
                uint32_t &seen = s_rectKeys[key];
                if (seen < 3u && s_rectLines < 400u)
                {
                    ++seen;
                    ++s_rectLines;
                    const GSVertex &v = batch.vertices[0];
                    std::fprintf(stderr,
                                 "[gs:rect] type=%u n=%u fbp=%u fbw=%u fpsm=0x%x zbp=%u tme=%u fst=%u abe=%u fge=%u "
                                 "tbp=0x%x tbw=%u psm=0x%x tw=%u th=%u tcc=%u tfx=%u cbp=0x%x cpsm=0x%x "
                                 "alpha=0x%llx test=0x%llx clamp=0x%llx bbox=(%.1f,%.1f)-(%.1f,%.1f) "
                                 "rgba=(%u,%u,%u,%u) z=%u q=%g st=(%g,%g) uv=(%u,%u)\n",
                                 static_cast<uint32_t>(state.prim.type), static_cast<uint32_t>(batch.vertexCount),
                                 ctx.frame.fbp, ctx.frame.fbw, static_cast<uint32_t>(ctx.frame.psm),
                                 static_cast<uint32_t>(ctx.zbuf.zbp),
                                 static_cast<uint32_t>(state.prim.tme), static_cast<uint32_t>(state.prim.fst),
                                 static_cast<uint32_t>(state.prim.abe), static_cast<uint32_t>(state.prim.fge),
                                 ctx.tex0.tbp0, static_cast<uint32_t>(ctx.tex0.tbw),
                                 static_cast<uint32_t>(ctx.tex0.psm), static_cast<uint32_t>(ctx.tex0.tw),
                                 static_cast<uint32_t>(ctx.tex0.th), static_cast<uint32_t>(ctx.tex0.tcc),
                                 static_cast<uint32_t>(ctx.tex0.tfx), ctx.tex0.cbp,
                                 static_cast<uint32_t>(ctx.tex0.cpsm),
                                 static_cast<unsigned long long>(ctx.alpha),
                                 static_cast<unsigned long long>(ctx.test),
                                 static_cast<unsigned long long>(ctx.clamp),
                                 minX, minY, maxX, maxY,
                                 static_cast<uint32_t>(v.r), static_cast<uint32_t>(v.g),
                                 static_cast<uint32_t>(v.b), static_cast<uint32_t>(v.a),
                                 static_cast<uint32_t>(v.z), v.q, v.s, v.t,
                                 static_cast<uint32_t>(v.u), static_cast<uint32_t>(v.v));
                }
            }
        }
    }
    {
        static uint32_t s_path1Logs = 0u;
        static uint32_t s_deg = 0u;
        static uint32_t s_real = 0u;
        static float s_minX = 1.0e9f;
        static float s_maxX = -1.0e9f;
        static float s_minY = 1.0e9f;
        static float s_maxY = -1.0e9f;
        // Band 0 only: every band thread updating these statics per primitive
        // bounced their cache lines between cores.
        if (g_ps2XgkickCount != 0u && state.prim.type != 6u && batch.vertexCount >= 3u && t_bandIndex == 0u)
        {
            const float dx01 = batch.vertices[0].x - batch.vertices[1].x;
            const float dy01 = batch.vertices[0].y - batch.vertices[1].y;
            const float dx02 = batch.vertices[0].x - batch.vertices[2].x;
            const float dy02 = batch.vertices[0].y - batch.vertices[2].y;
            const bool degenerate = (dx01 * dx01 + dy01 * dy01 < 0.25f) &&
                                    (dx02 * dx02 + dy02 * dy02 < 0.25f);
            {
                static uint32_t s_lastFbp = 0xFFFFFFFFu;
                static uint32_t s_lastFbw = 0xFFFFFFFFu;
                static uint32_t s_lastTme = 0xFFFFFFFFu;
                static uint32_t s_lastOfx = 0xFFFFFFFFu;
                static uint32_t s_ctxLogs = 0u;
                static uint32_t s_tmeLogs = 0u;
                static uint32_t s_dispLogs = 0u;
                const uint32_t ofx = static_cast<uint32_t>(ctx.xyoffset.ofx >> 4);
                const uint32_t ofy = static_cast<uint32_t>(ctx.xyoffset.ofy >> 4);
                const uint32_t tme = static_cast<uint32_t>(state.prim.tme);
                if (s_ctxLogs < 32u &&
                    (ctx.frame.fbp != s_lastFbp || ctx.frame.fbw != s_lastFbw ||
                     tme != s_lastTme || ofx != s_lastOfx))
                {
                    std::fprintf(stderr,
                                 "[gs:prim] ctx fbp=%u fbw=%u psm=0x%x tme=%u ctxt=%u ofx=%u ofy=%u "
                                 "scissor=(%u,%u)-(%u,%u) deg=%u real=%u\n",
                                 ctx.frame.fbp,
                                 ctx.frame.fbw,
                                 static_cast<uint32_t>(ctx.frame.psm),
                                 tme,
                                 static_cast<uint32_t>(state.prim.ctxt),
                                 ofx,
                                 ofy,
                                 static_cast<uint32_t>(ctx.scissor.x0),
                                 static_cast<uint32_t>(ctx.scissor.y0),
                                 static_cast<uint32_t>(ctx.scissor.x1),
                                 static_cast<uint32_t>(ctx.scissor.y1),
                                 s_deg,
                                 s_real);
                    s_lastFbp = ctx.frame.fbp;
                    s_lastFbw = ctx.frame.fbw;
                    s_lastTme = tme;
                    s_lastOfx = ofx;
                    ++s_ctxLogs;
                }
                if (tme != 0u && s_tmeLogs < 4u)
                {
                    std::fprintf(stderr,
                                 "[gs:prim] tme fbp=%u fbw=%u tbp0=%u ofx=%u v0=(%.1f,%.1f) "
                                 "rgba=(%u,%u,%u,%u) deg=%u\n",
                                 ctx.frame.fbp,
                                 ctx.frame.fbw,
                                 ctx.tex0.tbp0,
                                 ofx,
                                 batch.vertices[0].x,
                                 batch.vertices[0].y,
                                 static_cast<uint32_t>(batch.vertices[0].r),
                                 static_cast<uint32_t>(batch.vertices[0].g),
                                 static_cast<uint32_t>(batch.vertices[0].b),
                                 static_cast<uint32_t>(batch.vertices[0].a),
                                 degenerate ? 1u : 0u);
                    ++s_tmeLogs;
                }
                // Opt-in (ATV_TRACE_SUBVP_TEX=1): textured triangles drawn inside
                // the RiderMenu 3D preview scissor (x0=337), for the striped
                // rider/ATV skin investigation.
                static const bool s_traceSubvpTex = std::getenv("ATV_TRACE_SUBVP_TEX") != nullptr;
                static uint32_t s_subvpTexLogs = 0u;
                static uint32_t s_lastSubvpTbp = 0xFFFFFFFFu;
                const auto &t0 = ctx.tex0;
                if (s_traceSubvpTex && tme != 0u && ctx.scissor.x0 == 337u &&
                    t0.tbp0 != s_lastSubvpTbp && s_subvpTexLogs < 32u)
                {
                    s_lastSubvpTbp = t0.tbp0;
                    uint32_t textureHash = 2166136261u;
                    if (t0.psm == GS_PSM_CT16 && state.textureWidth == 64 && state.textureHeight == 64)
                    {
                        for (uint32_t ty = 0u; ty < 64u; ++ty)
                        {
                            for (uint32_t tx = 0u; tx < 64u; ++tx)
                            {
                                textureHash ^= ReadTextureVramUnlocked(t0.psm, t0.tbp0, t0.tbw, tx, ty);
                                textureHash *= 16777619u;
                            }
                        }
                    }
                    // ATV_DUMP_SUBVP_TEX=<dir>: raw sampler-view texels (u32 per texel,
                    // tw x th) for offline comparison with the guest source buffer.
                    static const char *s_subvpTexDumpDir = std::getenv("ATV_DUMP_SUBVP_TEX");
                    static uint32_t s_subvpTexDumps = 0u;
                    if (s_subvpTexDumpDir && s_subvpTexDumps < 64u)
                    {
                        const uint32_t dw = 1u << std::min<uint32_t>(t0.tw, 10u);
                        const uint32_t dh = 1u << std::min<uint32_t>(t0.th, 10u);
                        char path[512];
                        std::snprintf(path, sizeof(path), "%s/vram_%02u_tbp%04x_psm%x_tbw%u_%ux%u_%08x.bin",
                                      s_subvpTexDumpDir, s_subvpTexDumps, t0.tbp0,
                                      static_cast<uint32_t>(t0.psm), static_cast<uint32_t>(t0.tbw),
                                      dw, dh, textureHash);
                        if (FILE *f = std::fopen(path, "wb"))
                        {
                            for (uint32_t ty = 0u; ty < dh; ++ty)
                            {
                                for (uint32_t tx = 0u; tx < dw; ++tx)
                                {
                                    const uint32_t texel = ReadTextureVramUnlocked(t0.psm, t0.tbp0, t0.tbw, tx, ty);
                                    std::fwrite(&texel, sizeof(texel), 1u, f);
                                }
                            }
                            std::fclose(f);
                        }
                        ++s_subvpTexDumps;
                    }
                    std::fprintf(stderr,
                                 "[gs:subvp-tex] tbp0=0x%x tbw=%u psm=0x%x tw=%u th=%u tcc=%u tfx=%u "
                                 "cbp=0x%x cpsm=0x%x csm=%u csa=%u fst=%u type=%u hash=%08x\n",
                                 t0.tbp0, static_cast<uint32_t>(t0.tbw), static_cast<uint32_t>(t0.psm),
                                 static_cast<uint32_t>(t0.tw), static_cast<uint32_t>(t0.th),
                                 static_cast<uint32_t>(t0.tcc), static_cast<uint32_t>(t0.tfx),
                                 t0.cbp, static_cast<uint32_t>(t0.cpsm), static_cast<uint32_t>(t0.csm),
                                 static_cast<uint32_t>(t0.csa), static_cast<uint32_t>(state.prim.fst),
                                 static_cast<uint32_t>(state.prim.type), textureHash);
                    for (int vi = 0; vi < 3; ++vi)
                    {
                        const auto &vv = batch.vertices[vi];
                        std::fprintf(stderr,
                                     "[gs:subvp-tex]   v%d xy=(%.1f,%.1f) stq=(%.5f,%.5f,%.5f) s/q,t/q=(%.4f,%.4f) uv=(%u,%u) rgba=(%u,%u,%u,%u)\n",
                                     vi, vv.x, vv.y, vv.s, vv.t, vv.q,
                                     vv.q != 0.0f ? vv.s / vv.q : 0.0f, vv.q != 0.0f ? vv.t / vv.q : 0.0f,
                                     static_cast<uint32_t>(vv.u), static_cast<uint32_t>(vv.v),
                                     static_cast<uint32_t>(vv.r), static_cast<uint32_t>(vv.g),
                                     static_cast<uint32_t>(vv.b), static_cast<uint32_t>(vv.a));
                    }
                    ++s_subvpTexLogs;
                }
                if (!degenerate && ctx.frame.fbp <= 140u && ctx.frame.fbw >= 8u && s_dispLogs < 8u)
                {
                    std::fprintf(stderr,
                                 "[gs:prim] disp fbp=%u fbw=%u tme=%u ofx=%u ofy=%u "
                                 "v0=(%.1f,%.1f) v1=(%.1f,%.1f) v2=(%.1f,%.1f) "
                                 "rgba=(%u,%u,%u,%u)\n",
                                 ctx.frame.fbp,
                                 ctx.frame.fbw,
                                 tme,
                                 ofx,
                                 ofy,
                                 batch.vertices[0].x, batch.vertices[0].y,
                                 batch.vertices[1].x, batch.vertices[1].y,
                                 batch.vertices[2].x, batch.vertices[2].y,
                                 static_cast<uint32_t>(batch.vertices[0].r),
                                 static_cast<uint32_t>(batch.vertices[0].g),
                                 static_cast<uint32_t>(batch.vertices[0].b),
                                 static_cast<uint32_t>(batch.vertices[0].a));
                    ++s_dispLogs;
                }
                static uint32_t s_tme0Logs = 0u;
                if (!degenerate && tme == 0u && ctx.frame.fbp <= 140u && ctx.frame.fbw >= 8u &&
                    s_tme0Logs < 4u)
                {
                    const float triMinX = std::min(batch.vertices[0].x,
                                                   std::min(batch.vertices[1].x, batch.vertices[2].x));
                    const float triMaxX = std::max(batch.vertices[0].x,
                                                   std::max(batch.vertices[1].x, batch.vertices[2].x));
                    const float triMinY = std::min(batch.vertices[0].y,
                                                   std::min(batch.vertices[1].y, batch.vertices[2].y));
                    const float triMaxY = std::max(batch.vertices[0].y,
                                                   std::max(batch.vertices[1].y, batch.vertices[2].y));
                    std::fprintf(stderr,
                                 "[gs:prim] tme0 fbp=%u rgba=(%u,%u,%u,%u) fge=%u fog=%u "
                                 "xy=(%.1f,%.1f)-(%.1f,%.1f) v0=(%.1f,%.1f)\n",
                                 ctx.frame.fbp,
                                 static_cast<uint32_t>(batch.vertices[0].r),
                                 static_cast<uint32_t>(batch.vertices[0].g),
                                 static_cast<uint32_t>(batch.vertices[0].b),
                                 static_cast<uint32_t>(batch.vertices[0].a),
                                 static_cast<uint32_t>(state.prim.fge),
                                 static_cast<uint32_t>(batch.vertices[0].fog),
                                 triMinX, triMinY, triMaxX, triMaxY,
                                 batch.vertices[0].x, batch.vertices[0].y);
                    ++s_tme0Logs;
                }
            }
            if (degenerate)
            {
                ++s_deg;
            }
            else
            {
                ++s_real;
                for (uint32_t vi = 0u; vi < 3u; ++vi)
                {
                    s_minX = std::min(s_minX, batch.vertices[vi].x);
                    s_maxX = std::max(s_maxX, batch.vertices[vi].x);
                    s_minY = std::min(s_minY, batch.vertices[vi].y);
                    s_maxY = std::max(s_maxY, batch.vertices[vi].y);
                }
                const float spanX = s_maxX - s_minX;
                const float spanY = s_maxY - s_minY;
                static uint32_t s_spanLogs = 0u;
                if (s_spanLogs < 4u && ctx.frame.fbw >= 8u && ctx.frame.fbp <= 140u &&
                    (spanX > 80.0f || spanY > 80.0f))
                {
                    std::fprintf(stderr,
                                 "[gs:prim] span fbp=%u tme=%u fge=%u fog=%u fogcol=(%u,%u,%u) "
                                 "rgba=(%u,%u,%u,%u) xy=(%.1f,%.1f)-(%.1f,%.1f) "
                                 "v0=(%.1f,%.1f) v2=(%.1f,%.1f)\n",
                                 ctx.frame.fbp,
                                 static_cast<uint32_t>(state.prim.tme),
                                 static_cast<uint32_t>(state.prim.fge),
                                 static_cast<uint32_t>(batch.vertices[0].fog),
                                 static_cast<uint32_t>(state.fogR),
                                 static_cast<uint32_t>(state.fogG),
                                 static_cast<uint32_t>(state.fogB),
                                 static_cast<uint32_t>(batch.vertices[0].r),
                                 static_cast<uint32_t>(batch.vertices[0].g),
                                 static_cast<uint32_t>(batch.vertices[0].b),
                                 static_cast<uint32_t>(batch.vertices[0].a),
                                 s_minX, s_minY, s_maxX, s_maxY,
                                 batch.vertices[0].x, batch.vertices[0].y,
                                 batch.vertices[2].x, batch.vertices[2].y);
                    ++s_spanLogs;
                }
                if (s_path1Logs < 16u)
                {
                    std::fprintf(stderr,
                                 "[gs:prim] n=%u type=%u tme=%u fge=%u abe=%u ctxt=%u fbp=%u fbw=%u psm=0x%x "
                                 "tbp0=%u tbw=%u tpsm=0x%x tw=%u th=%u tfx=%u "
                                 "ofx=%u ofy=%u scissor=(%u,%u)-(%u,%u) test=0x%llx "
                                 "v0=(%.1f,%.1f) v1=(%.1f,%.1f) v2=(%.1f,%.1f) "
                                 "rgba0=(%u,%u,%u,%u) fog=%u fogcol=(%u,%u,%u)\n",
                                 s_path1Logs,
                                 static_cast<uint32_t>(state.prim.type),
                                 static_cast<uint32_t>(state.prim.tme),
                                 static_cast<uint32_t>(state.prim.fge),
                                 static_cast<uint32_t>(state.prim.abe),
                                 static_cast<uint32_t>(state.prim.ctxt),
                                 ctx.frame.fbp,
                                 ctx.frame.fbw,
                                 static_cast<uint32_t>(ctx.frame.psm),
                                 ctx.tex0.tbp0,
                                 static_cast<uint32_t>(ctx.tex0.tbw),
                                 static_cast<uint32_t>(ctx.tex0.psm),
                                 static_cast<uint32_t>(ctx.tex0.tw),
                                 static_cast<uint32_t>(ctx.tex0.th),
                                 static_cast<uint32_t>(ctx.tex0.tfx),
                                 static_cast<uint32_t>(ctx.xyoffset.ofx >> 4),
                                 static_cast<uint32_t>(ctx.xyoffset.ofy >> 4),
                                 static_cast<uint32_t>(ctx.scissor.x0),
                                 static_cast<uint32_t>(ctx.scissor.y0),
                                 static_cast<uint32_t>(ctx.scissor.x1),
                                 static_cast<uint32_t>(ctx.scissor.y1),
                                 static_cast<unsigned long long>(ctx.test),
                                 batch.vertices[0].x, batch.vertices[0].y,
                                 batch.vertices[1].x, batch.vertices[1].y,
                                 batch.vertices[2].x, batch.vertices[2].y,
                                 static_cast<uint32_t>(batch.vertices[0].r),
                                 static_cast<uint32_t>(batch.vertices[0].g),
                                 static_cast<uint32_t>(batch.vertices[0].b),
                                 static_cast<uint32_t>(batch.vertices[0].a),
                                 static_cast<uint32_t>(batch.vertices[0].fog),
                                 static_cast<uint32_t>(state.fogR),
                                 static_cast<uint32_t>(state.fogG),
                                 static_cast<uint32_t>(state.fogB));
                    ++s_path1Logs;
                }
            }
            const uint32_t total = s_deg + s_real;
            if (total == 256u || total == 4096u || total == 16384u || total == 65536u)
            {
                std::fprintf(stderr,
                             "[gs:prim] summary total=%u deg=%u real=%u xy=(%.1f,%.1f)-(%.1f,%.1f)\n",
                             total, s_deg, s_real, s_minX, s_minY, s_maxX, s_maxY);
            }
        }
    }
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << batch.vertices[0].x << "," << batch.vertices[0].y << ")"
                      << " uv0=(" << (batch.vertices[0].u >> 4) << "," << (batch.vertices[0].v >> 4) << ")"
                      << " stq0=(" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q << ")"
                      << " v1=(" << batch.vertices[1].x << "," << batch.vertices[1].y << ")"
                      << " uv1=(" << (batch.vertices[1].u >> 4) << "," << (batch.vertices[1].v >> 4) << ")"
                      << " stq1=(" << batch.vertices[1].s << "," << batch.vertices[1].t << "," << batch.vertices[1].q << ")"
                      << " v2=(" << batch.vertices[2].x << "," << batch.vertices[2].y << ")"
                      << " uv2=(" << (batch.vertices[2].u >> 4) << "," << (batch.vertices[2].v >> 4) << ")"
                      << " stq2=(" << batch.vertices[2].s << "," << batch.vertices[2].t << "," << batch.vertices[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(batch.vertices[0].r) << ","
                      << static_cast<uint32_t>(batch.vertices[0].g) << ","
                      << static_cast<uint32_t>(batch.vertices[0].b) << ","
                      << static_cast<uint32_t>(batch.vertices[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(batch.vertices[1].r) << ","
                      << static_cast<uint32_t>(batch.vertices[1].g) << ","
                      << static_cast<uint32_t>(batch.vertices[1].b) << ","
                      << static_cast<uint32_t>(batch.vertices[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(batch.vertices[2].r) << ","
                      << static_cast<uint32_t>(batch.vertices[2].g) << ","
                      << static_cast<uint32_t>(batch.vertices[2].b) << ","
                      << static_cast<uint32_t>(batch.vertices[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((state.prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    switch (state.prim.type)
    {
    case GS_PRIM_SPRITE:
        DrawSprite(batch);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
    {
        // ATV_GS_CULL_VERIFY=1 (with ATV_GS_NO_CULL=1): report triangles the
        // GIF-thread cull would drop that still draw a pixel.
        static const bool s_cullVerify = std::getenv("ATV_GS_CULL_VERIFY") != nullptr;
        if (!s_cullVerify)
        {
            DrawTriangle(batch);
            break;
        }
        const bool predictedEmpty = gsTriangleDrawsNothing(batch);
        t_triangleCovered = false;
        DrawTriangle(batch);
        static std::atomic<uint64_t> s_checked{0}, s_empty{0}, s_wrong{0};
        const uint64_t checked = s_checked.fetch_add(1, std::memory_order_relaxed) + 1;
        if (predictedEmpty)
            s_empty.fetch_add(1, std::memory_order_relaxed);
        if (predictedEmpty && t_triangleCovered)
        {
            const uint64_t wrong = s_wrong.fetch_add(1, std::memory_order_relaxed) + 1;
            if (wrong <= 20u)
                std::fprintf(stderr, "[gs-cull] WRONG drew a pixel: v=(%.4f,%.4f)(%.4f,%.4f)(%.4f,%.4f) ofs=%u,%u sc=%u,%u-%u,%u\n",
                             batch.vertices[0].x, batch.vertices[0].y, batch.vertices[1].x, batch.vertices[1].y,
                             batch.vertices[2].x, batch.vertices[2].y, batch.state.context.xyoffset.ofx,
                             batch.state.context.xyoffset.ofy, batch.state.context.scissor.x0,
                             batch.state.context.scissor.y0, batch.state.context.scissor.x1,
                             batch.state.context.scissor.y1);
        }
        if (checked % 4000000u == 0u)
            std::fprintf(stderr, "[gs-cull] checked=%llu wouldCull=%.1f%% wrong=%llu\n",
                         static_cast<unsigned long long>(checked), 100.0 * s_empty.load() / checked,
                         static_cast<unsigned long long>(s_wrong.load()));
        break;
    }
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        DrawLine(batch);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        const auto &ctx = state.context;
        int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
        WritePixel(state, px, py, static_cast<u32>(v.z), v.r, v.g, v.b, v.a, v.fog);
        break;
    }
    default:
        break;
    }
}

void GSCpuBackend::WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    const auto &ctx = state.context;
    if (x < ctx.scissor.x0 || x > ctx.scissor.x1 || y < ctx.scissor.y0 || y > ctx.scissor.y1)
        return;
    if (!ownsBandRow(y))
        return;

    // Fog completely disabled for crisp direct rendering
    (void)fog;

    const u32 fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const u32 fbw = ctx.frame.fbw >= 1u ? ctx.frame.fbw : 10u;
    const u32 fpsm = ctx.frame.psm;
    const u32 zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const u32 zpsm = ctx.zbuf.psm;

    const PixelWriteMask writeMask = classifyAlphaTest(ctx.test, a, static_cast<uint8_t>(fpsm));
    // A rejected fragment cannot affect VRAM. Avoid the depth lookup as well,
    // including ZB_ONLY alpha failures when depth writes are masked.
    if (!writeMask.writesFramebuffer() && (!writeMask.writeDepth || ctx.zbuf.zmask))
        return;

    const bool ztestEnabled = ((ctx.test >> 16) & 1u) != 0u;
    bool zpass = !ztestEnabled;
    uint32_t storedZ = 0u;
    uint32_t ztestMethod = 0u;
    if (ztestEnabled)
    {
        ztestMethod = static_cast<uint32_t>((ctx.test >> 17) & 3u);
        switch (ztestMethod)
        {
        case 0:
            zpass = false;
            break;
        case 1:
            zpass = true;
            break;
        case 2:
            storedZ = ReadVramUnlocked(zpsm, zbp, fbw, x, y);
            zpass = static_cast<uint32_t>(z) >= storedZ;
            break;
        case 3:
            storedZ = ReadVramUnlocked(zpsm, zbp, fbw, x, y);
            zpass = static_cast<uint32_t>(z) > storedZ;
            break;
        }
    }

    if (ztestEnabled && (fbp == 0 || fbp == 140) && x == 320 && y == 240)
    {
        static uint32_t s_zDbg = 0u;
        if (s_zDbg < 32u)
        {
            std::fprintf(stderr,
                "[ztest:center] z=%u storedZ=%u method=%u pass=%d fbp=%u zbp=%u tme=%u\n",
                static_cast<uint32_t>(z), storedZ, ztestMethod, zpass ? 1 : 0, fbp, zbp,
                static_cast<uint32_t>(state.prim.tme));
            ++s_zDbg;
        }
    }

    if (!zpass)
    {
        return;
    }

    // PABE-bypassed pixels do not need destination color for blending.
    // Masks and destination-alpha testing can still require the read below.
    const bool alphaBlendEnabled = state.prim.abe && !(state.pabe && (a & 0x80u) == 0u);
    const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && fpsm == GS_PSM_CT32;
    const bool destinationAlphaTestNeedsRead = ((ctx.test >> 14) & 0x1u) != 0u && (fpsm == GS_PSM_CT32 || fpsm == GS_PSM_CT16 || fpsm == GS_PSM_CT16S);

    // small optimization, avoid reading the framebuffer for simple draws
    // TODO: only one address lookup for rmw
    const bool frmw = destinationAlphaTestNeedsRead || (writeMask.writesFramebuffer() && ((ctx.frame.fbmsk != 0) || alphaBlendEnabled || preserveDestinationAlpha));

    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        rawFramebufferPixel = ReadVramUnlocked(fpsm, fbp, fbw, x, y);
        fbrgba = rawFramebufferPixel;

        if (bitsPerPixel(fpsm) == 16)
        {
            fbrgba = Rgba5551ToRgba8888(fbrgba);
        }
        else if (fpsm == GS_PSM_CT24)
        {
            // The GS supplies 0x80 as destination alpha for RGB24 blending.
            fbrgba |= 0x80000000u;
        }
    }

    if (!passesDestinationAlphaTest(ctx.test, static_cast<uint8_t>(fpsm), rawFramebufferPixel))
    {
        return;
    }

    if (writeMask.writesFramebuffer())
    {
        const u8 srcR = r;
        const u8 srcG = g;
        const u8 srcB = b;

        if (alphaBlendEnabled)
        {
            uint8_t dr = fbrgba & 0xFF;
            uint8_t dg = (fbrgba >> 8) & 0xFF;
            uint8_t db = (fbrgba >> 16) & 0xFF;
            uint8_t da = (fbrgba >> 24) & 0xFF;

            // PABE disables alpha blending when the source alpha MSB is clear.
            if (!(state.pabe && (a & 0x80u) == 0u))
            {
                uint64_t alphaReg = ctx.alpha;
                uint8_t asel = alphaReg & 3;
                uint8_t bsel = (alphaReg >> 2) & 3;
                uint8_t csel = (alphaReg >> 4) & 3;
                uint8_t dsel = (alphaReg >> 6) & 3;
                uint8_t fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);

                auto pickRGB = [&](uint8_t sel, int cs, int cd) -> int
                {
                    if (sel == 0)
                        return cs;
                    if (sel == 1)
                        return cd;
                    return 0;
                };
                int cAlpha = (csel == 0) ? a : (csel == 1) ? da
                                                           : fix;

                r = clampU8(((pickRGB(asel, r, dr) - pickRGB(bsel, r, dr)) * cAlpha >> 7) + pickRGB(dsel, r, dr));
                g = clampU8(((pickRGB(asel, g, dg) - pickRGB(bsel, g, dg)) * cAlpha >> 7) + pickRGB(dsel, g, dg));
                b = clampU8(((pickRGB(asel, b, db) - pickRGB(bsel, b, db)) * cAlpha >> 7) + pickRGB(dsel, b, db));
            }
            else
            {
                r = srcR;
                g = srcG;
                b = srcB;
            }
        }

        if (writeMask.writeAlpha && (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(r, g, b, a);

        if (ctx.frame.fbmsk != 0)
        {
            pixel = (pixel & ~ctx.frame.fbmsk) | (fbrgba & ctx.frame.fbmsk);
        }

        if (preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        // format conversion
        if (bitsPerPixel(fpsm) == 16)
        {
            pixel = Rgba8888ToRgba5551(pixel);
        }

        WriteVramUnlocked(fpsm, fbp, fbw, x, y, pixel);
    }

    if (writeMask.writeDepth && !ctx.zbuf.zmask)
    {
        if (g_ps2XgkickCount >= 1000u && (x == 250 && y == 400))
        {
            static uint32_t s_zWriteLog = 0u;
            if (s_zWriteLog < 32u)
            {
                std::fprintf(stderr,
                    "[zwrite] x=%d y=%d z=%u fbp=%u tbp0=0x%x type=%u tme=%u fge=%u rgb=(%u,%u,%u) a=%u\n",
                    x, y, static_cast<uint32_t>(z), ctx.frame.fbp, ctx.tex0.tbp0,
                    static_cast<uint32_t>(state.prim.type),
                    static_cast<uint32_t>(state.prim.tme),
                    static_cast<uint32_t>(state.prim.fge),
                    static_cast<uint32_t>(r), static_cast<uint32_t>(g), static_cast<uint32_t>(b),
                    static_cast<uint32_t>(a));
                ++s_zWriteLog;
            }
        }
        WriteVramUnlocked(zpsm, zbp, fbw, x, y, z);
    }
}

uint32_t GSCpuBackend::LookupCLUT(const GSDrawState &state,
                                  uint8_t index,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    const bool sixteenBit = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;
    const uint32_t sourceIndex = isFourBitIndexedPsm(sourcePsm)
                                     ? (static_cast<uint32_t>(index) & 0x0Fu)
                                     : static_cast<uint32_t>(index);

    const uint16_t *clut = t_drawClut ? t_drawClut : m_clut.data();
    uint32_t clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
    if (!sixteenBit && csm == 0u && isEightBitIndexedPsm(sourcePsm))
    {
        const uint32_t block = std::min((sourceIndex & 0xF0u) + clutBase, 240u);
        clutIndex = block + (sourceIndex & 0x0Fu);
    }

    switch (cpsm)
    {
    case GS_PSM_CT32:
    {
        const uint32_t raw = static_cast<uint32_t>(clut[clutIndex]) | (static_cast<uint32_t>(clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw);
    }
    case GS_PSM_CT24:
    {
        const uint32_t raw = static_cast<uint32_t>(clut[clutIndex]) | (static_cast<uint32_t>(clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw & 0x00FFFFFFu);
    }
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(clut[clutIndex]));
    default:
        static uint32_t s_badCpsmLog = 0u;
        if (s_badCpsmLog < 16u)
        {
            std::fprintf(stderr, "[gs:bad_cpsm] cpsm=0x%x index=%u srcPsm=0x%x\n",
                         static_cast<uint32_t>(cpsm), static_cast<uint32_t>(index), static_cast<uint32_t>(sourcePsm));
            ++s_badCpsmLog;
        }
        break;
    }

    return 0x00000000u;
}

uint32_t GSCpuBackend::SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v)
{
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;

    const int texW = state.textureWidth;
    const int texH = state.textureHeight;
    const uint64_t clamp = ctx.clamp;
    const uint8_t wrapU = static_cast<uint8_t>(clamp & 0x3u);
    const uint8_t wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    const uint16_t minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
    const uint16_t maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
    const uint16_t minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
    const uint16_t maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);

    float texUf, texVf;
    if (state.prim.fst)
    {
        texUf = static_cast<float>(u) / 16.0f;
        texVf = static_cast<float>(v) / 16.0f;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        texUf = s * invQ * static_cast<float>(texW);
        texVf = t * invQ * static_cast<float>(texH);
    }

    auto decode = [&](u32 out) -> uint32_t
    {
        switch (tex.psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        case GS_PSM_CT24:
        case GS_PSM_Z24:
            return applyTexa(state.texa, tex.psm, out);
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            return applyTexa(state.texa, tex.psm, Rgba5551ToRgba8888(out));
        case GS_PSM_T8:
        case GS_PSM_T8H:
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            return LookupCLUT(state, static_cast<u8>(out), tex.cpsm, tex.csm, tex.csa, tex.psm);
        default:
            static uint32_t s_badPsmLog = 0u;
            if (s_badPsmLog < 16u)
            {
                std::fprintf(stderr, "[gs:bad_psm] psm=0x%x tbp0=0x%x tbw=%u w=%d h=%d\n",
                             tex.psm, tex.tbp0, tex.tbw, texW, texH);
                ++s_badPsmLog;
            }
            break;
        }

        return 0x00000000u;
    };

    if (!state.linearFilter)
    {
        return decode(ReadTextureVramUnlocked(tex.psm, tex.tbp0, tex.tbw,
                                              wrapTextureCoordinate(static_cast<int>(texUf), texW, wrapU, minU, maxU),
                                              wrapTextureCoordinate(static_cast<int>(texVf), texH, wrapV, minV, maxV)));
    }

    const float sampleU = texUf - 0.5f;
    const float sampleV = texVf - 0.5f;
    const int u0 = static_cast<int>(std::floor(sampleU));
    const int v0 = static_cast<int>(std::floor(sampleV));
    const float fx = sampleU - static_cast<float>(u0);
    const float fy = sampleV - static_cast<float>(v0);
    const u32 wu0 = static_cast<u32>(wrapTextureCoordinate(u0, texW, wrapU, minU, maxU));
    const u32 wu1 = static_cast<u32>(wrapTextureCoordinate(u0 + 1, texW, wrapU, minU, maxU));
    const u32 wv0 = static_cast<u32>(wrapTextureCoordinate(v0, texH, wrapV, minV, maxV));
    const u32 wv1 = static_cast<u32>(wrapTextureCoordinate(v0 + 1, texH, wrapV, minV, maxV));

    u32 raw[4] = {};
    if (m_vram)
        GSMem::ReadTexture4(t_bandCache ? *t_bandCache : m_texturePageCache, m_vram, tex.psm, tex.tbp0, tex.tbw,
                            {wu0, wu1, wu0, wu1}, {wv0, wv0, wv1, wv1}, raw);
    return lerpTexels(decode(raw[0]), decode(raw[1]), decode(raw[2]), decode(raw[3]), fx, fy);
}

void GSCpuBackend::DrawSprite(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;
    u32 z1 = static_cast<u32>(v1.z);

    static uint32_t s_menuSpriteLogs = 0u;
    if (s_menuSpriteLogs < 40u && (ctx.frame.fbp == 0u || ctx.frame.fbp == 140u) && state.prim.tme)
    {
        std::fprintf(stderr,
            "[menu:sprite] n=%u fbp=%u tme=%u fst=%u tbp0=%u tbw=%u psm=0x%x "
            "v0=(%.1f,%.1f) v1=(%.1f,%.1f) ofx=%d ofy=%d "
            "xy0=(%d,%d) xy1=(%d,%d) "
            "uv0=(%u,%u) uv1=(%u,%u) st0=(%.2f,%.2f) st1=(%.2f,%.2f) tw=%u th=%u\n",
            s_menuSpriteLogs, ctx.frame.fbp,
            static_cast<uint32_t>(state.prim.tme),
            static_cast<uint32_t>(state.prim.fst),
            ctx.tex0.tbp0, static_cast<uint32_t>(ctx.tex0.tbw),
            static_cast<uint32_t>(ctx.tex0.psm),
            v0.x, v0.y, v1.x, v1.y, ofx, ofy,
            x0, y0, x1, y1,
            v0.u, v0.v, v1.u, v1.v,
            v0.s, v0.t, v1.s, v1.t,
            static_cast<uint32_t>(state.textureWidth),
            static_cast<uint32_t>(state.textureHeight));
        ++s_menuSpriteLogs;
    }

    bool swappedX = false;
    bool swappedY = false;
    if (x0 > x1)
    {
        std::swap(x0, x1);
        swappedX = true;
    }
    if (y0 > y1)
    {
        std::swap(y0, y1);
        swappedY = true;
    }

    const int unclippedX0 = x0;
    const int unclippedY0 = y0;
    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int unclippedX1 = unclippedX0 + spanX - 1;
    const int unclippedY1 = unclippedY0 + spanY - 1;

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return;

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);
    const int firstY = firstBandRow(drawY0);
    if (firstY > drawY1)
        return;

    static uint32_t s_onscreenSpriteLogs = 0u;
    if (s_onscreenSpriteLogs < 40u && (ctx.frame.fbp == 0u || ctx.frame.fbp == 140u) && state.prim.tme)
    {
        std::fprintf(stderr,
            "[onscreen:sprite] n=%u fbp=%u tbp0=%u tbw=%u psm=0x%x "
            "draw=(%d,%d)-(%d,%d) span=(%d,%d) fst=%u "
            "uv0=(%u,%u) uv1=(%u,%u) st0=(%.2f,%.2f) st1=(%.2f,%.2f) tw=%u th=%u\n",
            s_onscreenSpriteLogs, ctx.frame.fbp,
            ctx.tex0.tbp0, static_cast<uint32_t>(ctx.tex0.tbw),
            static_cast<uint32_t>(ctx.tex0.psm),
            drawX0, drawY0, drawX1, drawY1, spanX, spanY,
            static_cast<uint32_t>(state.prim.fst),
            v0.u, v0.v, v1.u, v1.v,
            v0.s, v0.t, v1.s, v1.t,
            static_cast<uint32_t>(state.textureWidth),
            static_cast<uint32_t>(state.textureHeight));
        ++s_onscreenSpriteLogs;
    }

    const bool coversDisplay = drawX0 <= 0 && drawY0 <= 0 && drawX1 >= 639 && drawY1 >= 447;
    if (g_ps2XgkickCount >= 1000u && (drawX0 <= 320 && drawX1 >= 320 && drawY0 <= 280 && drawY1 >= 280))
    {
        static uint32_t s_raceSpriteLogs = 0u;
        if (s_raceSpriteLogs < 16u)
        {
            std::fprintf(stderr,
                "[sprite:race] n=%u fbp=%u tbp0=%u tme=%u fge=%u abe=%u xy=(%d,%d)-(%d,%d) z=%u rgba=(%u,%u,%u,%u)\n",
                s_raceSpriteLogs, ctx.frame.fbp, ctx.tex0.tbp0,
                static_cast<uint32_t>(state.prim.tme),
                static_cast<uint32_t>(state.prim.fge),
                static_cast<uint32_t>(state.prim.abe),
                drawX0, drawY0, drawX1, drawY1, z1,
                static_cast<uint32_t>(v1.r), static_cast<uint32_t>(v1.g), static_cast<uint32_t>(v1.b), static_cast<uint32_t>(v1.a));
            ++s_raceSpriteLogs;
        }
    }


    const uint64_t alphaReg = ctx.alpha;
    const uint8_t alphaMode = static_cast<uint8_t>(alphaReg & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((alphaReg >> 32) & 0xFFu);

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;

    // ATV_GS_NO_PIXEL_PIPE=1: per-pixel state decoding as before session 33.
    static const bool s_noPixelPipe = std::getenv("ATV_GS_NO_PIXEL_PIPE") != nullptr;
    PixelPipe pixelPipe;
    TexturePipe texturePipe;
    if (!s_noPixelPipe)
    {
        setupPixelPipe(state, m_vram, m_readVramFuncs.data(), m_writeVramFuncs.data(), pixelPipe);
        if (state.prim.tme)
            setupTexturePipe(state, m_vram, t_bandCache ? *t_bandCache : m_texturePageCache, t_drawClut ? t_drawClut : m_clut.data(), texturePipe);
    }

    if (state.prim.tme)
    {
        const auto &tex = ctx.tex0;
        const int texW = state.textureWidth;
        const int texH = state.textureHeight;

        float u0f, v0f, u1f, v1f;
        if (state.prim.fst)
        {
            u0f = static_cast<float>(v0.u) / 16.0f;
            v0f = static_cast<float>(v0.v) / 16.0f;
            u1f = static_cast<float>(v1.u) / 16.0f;
            v1f = static_cast<float>(v1.v) / 16.0f;
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }

        float spriteW = static_cast<float>(spanX);
        float spriteH = static_cast<float>(spanY);
        if (spriteW < 1.0f)
            spriteW = 1.0f;
        if (spriteH < 1.0f)
            spriteH = 1.0f;

        for (int y = firstY; y <= drawY1; y = nextBandRow(y))
        {
            t_ownedBandRow = y;
            float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / spriteH;
            float texVf = v0f + (v1f - v0f) * ty;

            for (int x = drawX0; x <= drawX1; ++x)
            {
                float tx = (static_cast<float>(x - unclippedX0) + 0.5f) / spriteW;
                float texUf = u0f + (u1f - u0f) * tx;
                uint32_t texel = 0x00000000u;
                if (state.prim.fst)
                {
                    const int fixedU = static_cast<int>((texUf * 16.0f) + 0.5f);
                    const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                    const uint16_t sampleU = static_cast<uint16_t>(clampInt(fixedU, 0, 0xFFFF));
                    const uint16_t sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
                    texel = texturePipe.ready ? sampleTextureFast(texturePipe, 0.0f, 0.0f, 1.0f, sampleU, sampleV)
                                              : SampleTexture(state, 0.0f, 0.0f, 1.0f, sampleU, sampleV);
                }
                else
                {
                    texel = texturePipe.ready
                                ? sampleTextureFast(texturePipe, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH), 1.0f, 0u, 0u)
                                : SampleTexture(state, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH), 1.0f, 0u, 0u);
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);

                if (pixelPipe.vram)
                    writePixelFast(pixelPipe, x, y, z1, color.r, color.g, color.b, color.a);
                else
                    WritePixel(state, x, y, z1, color.r, color.g, color.b, color.a, v1.fog);
            }
        }
    }
    else
    {
        uint32_t clearZ = z1;
        if (coversDisplay && !state.prim.tme && z1 > 0x800000u)
        {
            clearZ = 0u;
        }
        for (int y = firstY; y <= drawY1; y = nextBandRow(y))
        {
            t_ownedBandRow = y;
            for (int x = drawX0; x <= drawX1; ++x)
            {
                if (pixelPipe.vram)
                    writePixelFast(pixelPipe, x, y, clearZ, r, g, b, a);
                else
                    WritePixel(state, x, y, clearZ, r, g, b, a, v1.fog);
            }
        }
    }
}

// ATV_GS_IDBUF=<seconds>: per-pixel "last draw state that changed this pixel"
// for the race framebuffers, dumped with the displayed frame every N seconds
// to idbuf_dump/ (frame_N.ppm, frame_N.ids = uint32 per pixel, ids.txt).
namespace
{
    constexpr uint32_t kIdBufW = 640u, kIdBufH = 512u;
    struct GsIdBuf
    {
        double intervalSec = []
        {
            const char *e = std::getenv("ATV_GS_IDBUF");
            return e != nullptr ? std::atof(e) : 0.0;
        }();
        std::mutex mutex;
        std::unordered_map<std::string, uint32_t> keyToId;
        std::vector<std::string> descs{"none"};
        std::unordered_map<uint32_t, std::vector<uint32_t>> perFbp;
        std::unordered_map<uint32_t, std::vector<uint32_t>> perFbpUv; // u | v << 16
        std::unordered_map<uint32_t, std::string> lastUploads;        // dbp -> recent uploads
        std::chrono::steady_clock::time_point lastDump{};
        uint32_t dumps = 0u;
    };
    GsIdBuf &gsIdBuf()
    {
        static GsIdBuf s;
        return s;
    }
    std::atomic<uint32_t> g_gsRacePresents{0u};
    uint32_t gsRacePresentCount() { return g_gsRacePresents.load(std::memory_order_relaxed); }

    uint32_t *gsIdRows(uint32_t fbp, bool uv = false)
    {
        GsIdBuf &b = gsIdBuf();
        std::lock_guard<std::mutex> lock(b.mutex);
        auto &v = uv ? b.perFbpUv[fbp] : b.perFbp[fbp];
        if (v.empty())
            v.assign(kIdBufW * kIdBufH, 0u);
        return v.data();
    }
    std::string gsIdLastUploads(uint32_t dbp)
    {
        GsIdBuf &b = gsIdBuf();
        std::lock_guard<std::mutex> lock(b.mutex);
        auto it = b.lastUploads.find(dbp);
        return it != b.lastUploads.end() ? it->second : std::string("-");
    }
    void gsIdNoteUpload(uint32_t dbp, uint32_t dbw, uint32_t dpsm, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
    {
        GsIdBuf &b = gsIdBuf();
        if (b.intervalSec <= 0.0)
            return;
        char s[96];
        std::snprintf(s, sizeof(s), "[%ux%u@%u,%u bw=%u psm=0x%x]", w, h, x, y, dbw, dpsm);
        std::lock_guard<std::mutex> lock(b.mutex);
        std::string &v = b.lastUploads[dbp];
        // A transfer at (0,0) starts a new texture in this slot.
        v = ((x == 0u && y == 0u) || v.size() > 120u ? std::string() : v) + s;
    }
    uint32_t gsIdFor(const std::string &desc)
    {
        GsIdBuf &b = gsIdBuf();
        std::lock_guard<std::mutex> lock(b.mutex);
        auto [it, inserted] = b.keyToId.try_emplace(desc, static_cast<uint32_t>(b.descs.size()));
        if (inserted)
            b.descs.push_back(desc);
        return it->second;
    }
}

// Mirrors DrawTriangle's rejects and its coverage loop (all rows, no band
// split): the GIF thread drops triangles that draw nothing before they are
// copied, queued and banded. Keep in step with DrawTriangle.
bool gsTriangleDrawsNothing(const GSPrimitiveBatch &batch)
{
    if (batch.vertexCount < 3u)
        return false;
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;
    const int ofx = ctx.xyoffset.ofx >> 4;
    const int ofy = ctx.xyoffset.ofy >> 4;
    const float fx0 = v0.x - static_cast<float>(ofx);
    const float fy0 = v0.y - static_cast<float>(ofy);
    const float fx1 = v1.x - static_cast<float>(ofx);
    const float fy1 = v1.y - static_cast<float>(ofy);
    const float fx2 = v2.x - static_cast<float>(ofx);
    const float fy2 = v2.y - static_cast<float>(ofy);

    if (!state.prim.fst && state.prim.fge &&
        (v0.q <= 1.0e-5f || v1.q <= 1.0e-5f || v2.q <= 1.0e-5f))
        return true;
    const int unclippedMinX = floorToInt(min3(fx0, fx1, fx2));
    const int unclippedMaxX = ceilToInt(max3(fx0, fx1, fx2));
    const int unclippedMinY = floorToInt(min3(fy0, fy1, fy2));
    const int unclippedMaxY = ceilToInt(max3(fy0, fy1, fy2));
    const int viewportWidth = static_cast<int>(ctx.scissor.x1) - static_cast<int>(ctx.scissor.x0) + 1;
    const int viewportHeight = static_cast<int>(ctx.scissor.y1) - static_cast<int>(ctx.scissor.y0) + 1;
    if (!state.prim.fst && state.prim.fge &&
        (unclippedMaxX - unclippedMinX > viewportWidth ||
         unclippedMaxY - unclippedMinY > viewportHeight))
        return true;
    if (unclippedMaxX < ctx.scissor.x0 || unclippedMinX > ctx.scissor.x1 ||
        unclippedMaxY < ctx.scissor.y0 || unclippedMinY > ctx.scissor.y1)
        return true;
    if (std::fabs(fx0) > 30000.0f || std::fabs(fx1) > 30000.0f || std::fabs(fx2) > 30000.0f ||
        std::fabs(fy0) > 30000.0f || std::fabs(fy1) > 30000.0f || std::fabs(fy2) > 30000.0f)
        return true;
    const int minX = clampInt(unclippedMinX, ctx.scissor.x0, ctx.scissor.x1);
    const int maxX = clampInt(unclippedMaxX, ctx.scissor.x0, ctx.scissor.x1);
    const int minY = clampInt(unclippedMinY, ctx.scissor.y0, ctx.scissor.y1);
    const int maxY = clampInt(unclippedMaxY, ctx.scissor.y0, ctx.scissor.y1);
    if (minY > maxY)
        return true;
    const float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return true;
    // Larger boxes almost always cover a sample; testing them costs more than drawing.
    if (maxX - minX > 3 || maxY - minY > 3)
        return false;

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;
    const float a01 = (fy1 - fy2) * winding * invAbsDenom;
    const float b01 = (fx2 - fx1) * winding * invAbsDenom;
    const float a12 = (fy2 - fy0) * winding * invAbsDenom;
    const float b12 = (fx0 - fx2) * winding * invAbsDenom;
    for (int y = minY; y <= maxY; ++y)
    {
        const float py = static_cast<float>(y) + 0.5f;
        const float px0 = static_cast<float>(minX) + 0.5f;
        float w0 = a01 * (px0 - fx2) + b01 * (py - fy2);
        float w1 = a12 * (px0 - fx2) + b12 * (py - fy2);
        for (int x = minX; x <= maxX; ++x)
        {
            const float w2 = 1.0f - w0 - w1;
            if (!(w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon))
                return false;
            w0 += a01;
            w1 += a12;
        }
    }
    return true;
}

void GSCpuBackend::DrawTriangle(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;

    // getenv scans the whole environment under a CRT lock; per-triangle calls dominated race frame time.
    static const bool s_traceTbpCensus = std::getenv("ATV_TRACE_TBP_CENSUS") != nullptr;
    static const bool s_traceLargeTri = std::getenv("ATV_TRACE_LARGE_TRI") != nullptr;

    // Bounded, opt-in diagnostic: census EVERY DrawTriangle entry by texture,
    // before any reject/scissor/Q logic runs, so "never submitted at all" can
    // be told apart from "submitted then rejected here" for missing-geometry
    // investigations (vehicle/vegetation/track-decal not appearing on screen).
    if (g_raceLoadStarted && s_traceTbpCensus)
    {
        static std::unordered_map<uint64_t, uint64_t> s_entryCounts;
        const uint64_t key = (static_cast<uint64_t>(ctx.frame.fbp) << 32) |
                              (static_cast<uint64_t>(static_cast<uint32_t>(ctx.tex0.psm)) << 24) |
                              (static_cast<uint64_t>(static_cast<uint32_t>(state.prim.tme)) << 20) |
                              static_cast<uint64_t>(ctx.tex0.tbp0 & 0x3FFFu);
        uint64_t &cnt = s_entryCounts[key];
        if (cnt == 0u)
        {
            std::fprintf(stderr, "[tbp:entry] NEW fbp=%u tbp=0x%x psm=0x%x tme=%u v0=(%g,%g,q=%g) v1=(%g,%g,q=%g) v2=(%g,%g,q=%g)\n",
                         ctx.frame.fbp, ctx.tex0.tbp0, static_cast<uint32_t>(ctx.tex0.psm),
                         static_cast<uint32_t>(state.prim.tme),
                         v0.x, v0.y, v0.q, v1.x, v1.y, v1.q, v2.x, v2.y, v2.q);
        }
        ++cnt;
    }

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    float fx2 = v2.x - static_cast<float>(ofx);
    float fy2 = v2.y - static_cast<float>(ofy);

    // ATV_TRACE_MIP=1: census of TEX1 (MXL/LCM/MMIN) on textured race triangles.
    {
        static const bool s_traceMip = std::getenv("ATV_TRACE_MIP") != nullptr;
        if (s_traceMip && g_raceLoadStarted && state.prim.tme)
        {
            static std::mutex s_mipMutex;
            static std::unordered_map<uint32_t, uint64_t> s_mipCounts;
            static uint64_t s_mipTotal = 0u;
            const uint64_t t1 = ctx.tex1;
            const uint32_t mxl = static_cast<uint32_t>((t1 >> 2) & 7u);
            const uint32_t mmin = static_cast<uint32_t>((t1 >> 6) & 7u);
            const uint32_t key = (mxl << 8) | (mmin << 4) | static_cast<uint32_t>(t1 & 1u);
            std::lock_guard<std::mutex> lock(s_mipMutex);
            if (s_mipCounts[key]++ == 0u)
                std::fprintf(stderr, "[gs-mip] NEW mxl=%u mmin=%u lcm=%u l=%u k=%d tbp=0x%x tw=%u miptbp1=0x%llx miptbp2=0x%llx\n",
                             mxl, mmin, static_cast<uint32_t>(t1 & 1u), static_cast<uint32_t>((t1 >> 19) & 3u),
                             static_cast<int>(static_cast<int16_t>(static_cast<uint16_t>((t1 >> 32) & 0xFFFu) << 4) >> 4),
                             ctx.tex0.tbp0, static_cast<uint32_t>(state.textureWidth),
                             static_cast<unsigned long long>(ctx.miptbp1), static_cast<unsigned long long>(ctx.miptbp2));
            if ((++s_mipTotal % 200000u) == 0u)
                for (const auto &kv : s_mipCounts)
                    std::fprintf(stderr, "[gs-mip] mxl=%u mmin=%u lcm=%u count=%llu\n", kv.first >> 8, (kv.first >> 4) & 0xFu,
                                 kv.first & 1u, static_cast<unsigned long long>(kv.second));
        }
    }

    // ATV_TRACE_SKYTRI=<n>: log sky-cube-like triangles (128x128 CT16, no fog,
    // no blend, TEST=0x5000c) drawn between race presents n and n+1.
    {
        static const int s_skyTriAt = []
        {
            const char *e = std::getenv("ATV_TRACE_SKYTRI");
            return e ? std::atoi(e) : -1;
        }();
        static std::atomic<uint32_t> s_skyTriLogged{0u};
        if (s_skyTriAt >= 0 && g_raceLoadStarted && gsRacePresentCount() >= static_cast<uint32_t>(s_skyTriAt) &&
            state.prim.tme && !state.prim.abe && !state.prim.fge && ctx.test == 0x5000cull &&
            state.textureWidth == 128u && state.textureHeight == 128u &&
            s_skyTriLogged.fetch_add(1u, std::memory_order_relaxed) < 60000u)
        {
            auto stq = [](const GSVertex &v) { return std::pair<float, float>(v.s / v.q * 128.0f, v.t / v.q * 128.0f); };
            const auto a = stq(v0), b = stq(v1), c = stq(v2);
            std::fprintf(stderr, "[skytri] present=%u fbp=%u tbp=0x%x p0=(%.1f,%.1f,q=%.6f uv=%.1f,%.1f) p1=(%.1f,%.1f,q=%.6f uv=%.1f,%.1f) p2=(%.1f,%.1f,q=%.6f uv=%.1f,%.1f)\n",
                         gsRacePresentCount(), ctx.frame.fbp, ctx.tex0.tbp0, fx0, fy0, v0.q, a.first, a.second, fx1, fy1, v1.q, b.first, b.second,
                         fx2, fy2, v2.q, c.first, c.second);
        }
    }

    // ATV_GS_PROBE=x,y: in races, log the triangles that cover framebuffer
    // pixel (x,y) (first 3 per fbp/tbp/psm/state key) to find which draw
    // produces a stray object.
    {
        static const std::array<int, 2> s_probe = []
        {
            std::array<int, 2> p{-1, -1};
            if (const char *env = std::getenv("ATV_GS_PROBE"))
                std::sscanf(env, "%d,%d", &p[0], &p[1]);
            return p;
        }();
        if (s_probe[0] >= 0 && g_raceLoadStarted)
        {
            const float px = static_cast<float>(s_probe[0]) + 0.5f;
            const float py = static_cast<float>(s_probe[1]) + 0.5f;
            const float e0 = (fx1 - fx0) * (py - fy0) - (fy1 - fy0) * (px - fx0);
            const float e1 = (fx2 - fx1) * (py - fy1) - (fy2 - fy1) * (px - fx1);
            const float e2 = (fx0 - fx2) * (py - fy2) - (fy0 - fy2) * (px - fx2);
            if ((e0 >= 0.0f && e1 >= 0.0f && e2 >= 0.0f) || (e0 <= 0.0f && e1 <= 0.0f && e2 <= 0.0f))
            {
                static std::mutex s_probeMutex;
                static std::unordered_map<uint64_t, uint32_t> s_probeSeen;
                const uint64_t key = (static_cast<uint64_t>(ctx.frame.fbp) << 40) |
                                     (static_cast<uint64_t>(ctx.tex0.tbp0 & 0x3FFFu) << 24) |
                                     (static_cast<uint64_t>(static_cast<uint32_t>(ctx.tex0.psm)) << 16) |
                                     (static_cast<uint64_t>(state.prim.tme) << 3) |
                                     (static_cast<uint64_t>(state.prim.abe) << 2) |
                                     (static_cast<uint64_t>(state.prim.fge) << 1) |
                                     static_cast<uint64_t>(state.prim.fst);
                std::lock_guard<std::mutex> lock(s_probeMutex);
                if (s_probeSeen.size() < 256u && s_probeSeen[key]++ < 3u)
                {
                    std::fprintf(stderr,
                        "[gs-probe] fbp=%u fbw=%u tme=%u fst=%u fge=%u abe=%u iip=%u tbp=0x%x tbw=%u psm=0x%x cbp=0x%x tw=%u th=%u "
                        "zbp=%u ztst=%u ate=%u "
                        "v0=(%.1f,%.1f z=%.0f q=%g rgba=%02x%02x%02x%02x st=%g,%g uv=%u,%u f=%u) "
                        "v1=(%.1f,%.1f z=%.0f q=%g rgba=%02x%02x%02x%02x) v2=(%.1f,%.1f z=%.0f q=%g rgba=%02x%02x%02x%02x)\n",
                        ctx.frame.fbp, ctx.frame.fbw, static_cast<uint32_t>(state.prim.tme),
                        static_cast<uint32_t>(state.prim.fst), static_cast<uint32_t>(state.prim.fge),
                        static_cast<uint32_t>(state.prim.abe), static_cast<uint32_t>(state.prim.iip),
                        ctx.tex0.tbp0, static_cast<uint32_t>(ctx.tex0.tbw), static_cast<uint32_t>(ctx.tex0.psm),
                        static_cast<uint32_t>(ctx.tex0.cbp),
                        static_cast<uint32_t>(state.textureWidth), static_cast<uint32_t>(state.textureHeight),
                        static_cast<uint32_t>(ctx.zbuf.zbp), static_cast<uint32_t>((ctx.test >> 17) & 3u),
                        static_cast<uint32_t>(ctx.test & 1u),
                        fx0, fy0, v0.z, v0.q, v0.r, v0.g, v0.b, v0.a, v0.s, v0.t, static_cast<uint32_t>(v0.u), static_cast<uint32_t>(v0.v), static_cast<uint32_t>(v0.fog),
                        fx1, fy1, v1.z, v1.q, v1.r, v1.g, v1.b, v1.a,
                        fx2, fy2, v2.z, v2.q, v2.r, v2.g, v2.b, v2.a);
                }
            }
        }
    }

    // Perspective-correct 3D vertices carry Q = 1/W.  A non-positive Q is
    // behind the camera (or on its projection plane).  The GS receives
    // already projected coordinates and cannot clip such a triangle for us;
    // rasterizing it after substituting Q=1 creates the screen-sized shards
    // seen in gameplay.  The game's VU path normally suppresses these kicks,
    // so reject any surviving triangle that crosses that invalid boundary.
    if (!state.prim.fst && state.prim.fge &&
        (v0.q <= 1.0e-5f || v1.q <= 1.0e-5f || v2.q <= 1.0e-5f))
    {
        return;
    }

    static uint32_t s_menuTriLogs = 0u;
    if (s_menuTriLogs < 40u && (ctx.frame.fbp == 0u || ctx.frame.fbp == 140u) && state.prim.tme)
    {
        std::fprintf(stderr,
            "[menu:tri] n=%u type=%u fbp=%u tme=%u fst=%u tbp0=%u tbw=%u psm=0x%x "
            "v0=(%.1f,%.1f) v1=(%.1f,%.1f) v2=(%.1f,%.1f) ofx=%d ofy=%d "
            "fx=(%.1f,%.1f) (%.1f,%.1f) (%.1f,%.1f) "
            "uv0=(%u,%u) uv1=(%u,%u) uv2=(%u,%u) tw=%u th=%u\n",
            s_menuTriLogs, static_cast<uint32_t>(state.prim.type), ctx.frame.fbp,
            static_cast<uint32_t>(state.prim.tme),
            static_cast<uint32_t>(state.prim.fst),
            ctx.tex0.tbp0, static_cast<uint32_t>(ctx.tex0.tbw),
            static_cast<uint32_t>(ctx.tex0.psm),
            v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, ofx, ofy,
            fx0, fy0, fx1, fy1, fx2, fy2,
            v0.u, v0.v, v1.u, v1.v, v2.u, v2.v,
            static_cast<uint32_t>(state.textureWidth),
            static_cast<uint32_t>(state.textureHeight));
        ++s_menuTriLogs;
    }

    {
        static uint32_t s_windowLogs = 0u;
        if (s_windowLogs < 8u && ctx.frame.fbp == 140u && ctx.frame.fbw >= 8u &&
            state.prim.fge)
        {
            const float wminX = std::min({fx0, fx1, fx2});
            const float wmaxX = std::max({fx0, fx1, fx2});
            const float wminY = std::min({fy0, fy1, fy2});
            const float wmaxY = std::max({fy0, fy1, fy2});
            if (wmaxX >= static_cast<float>(ctx.scissor.x0) &&
                wminX <= static_cast<float>(ctx.scissor.x1) &&
                wmaxY >= static_cast<float>(ctx.scissor.y0) &&
                wminY <= static_cast<float>(ctx.scissor.y1))
            {
                std::fprintf(stderr,
                             "[gs:prim] window fbp=%u tme=%u fge=%u fog=%u fogcol=(%u,%u,%u) "
                             "rgba=(%u,%u,%u,%u) win=(%.1f,%.1f)-(%.1f,%.1f) "
                             "gs=(%.1f,%.1f)\n",
                             ctx.frame.fbp,
                             static_cast<uint32_t>(state.prim.tme),
                             static_cast<uint32_t>(state.prim.fge),
                             static_cast<uint32_t>(v0.fog),
                             static_cast<uint32_t>(state.fogR),
                             static_cast<uint32_t>(state.fogG),
                             static_cast<uint32_t>(state.fogB),
                             static_cast<uint32_t>(v0.r),
                             static_cast<uint32_t>(v0.g),
                             static_cast<uint32_t>(v0.b),
                             static_cast<uint32_t>(v0.a),
                             wminX, wminY, wmaxX, wmaxY,
                             v0.x, v0.y);
                ++s_windowLogs;
            }
        }
    }

    static uint32_t s_raceTriDbg = 0u;
    if (s_raceTriDbg < 30u)
    {
        std::fprintf(stderr,
            "[tri:dbg] n=%u fbp=%u tme=%u fst=%u fx=(%.1f,%.1f)(%.1f,%.1f)(%.1f,%.1f) "
            "z=(%.0f,%.0f,%.0f) q=(%.3f,%.3f,%.3f) rgb0=(%u,%u,%u,%u)\n",
            s_raceTriDbg, ctx.frame.fbp,
            static_cast<uint32_t>(state.prim.tme),
            static_cast<uint32_t>(state.prim.fst),
            fx0, fy0, fx1, fy1, fx2, fy2,
            v0.z, v1.z, v2.z,
            v0.q, v1.q, v2.q,
            static_cast<uint32_t>(v0.r), static_cast<uint32_t>(v0.g), static_cast<uint32_t>(v0.b), static_cast<uint32_t>(v0.a));
        ++s_raceTriDbg;
    }

    const int unclippedMinX = floorToInt(min3(fx0, fx1, fx2));
    const int unclippedMaxX = ceilToInt(max3(fx0, fx1, fx2));
    const int unclippedMinY = floorToInt(min3(fy0, fy1, fy2));
    const int unclippedMaxY = ceilToInt(max3(fy0, fy1, fy2));

    if (s_traceLargeTri && state.prim.fge &&
        (unclippedMaxX - unclippedMinX > 500 || unclippedMaxY - unclippedMinY > 350) &&
        unclippedMaxX >= ctx.scissor.x0 && unclippedMinX <= ctx.scissor.x1 &&
        unclippedMaxY >= ctx.scissor.y0 && unclippedMinY <= ctx.scissor.y1)
    {
        static uint32_t s_largeTriangleLogs = 0u;
        if (s_largeTriangleLogs < 48u)
        {
            std::fprintf(stderr,
                "[large:tri] n=%u fbp=%u prim=%u tme=%u fst=%u fge=%u abe=%u tbp=0x%x test=%llx "
                "xy=(%.1f,%.1f)(%.1f,%.1f)(%.1f,%.1f) q=(%g,%g,%g) z=(%.0f,%.0f,%.0f)\n",
                s_largeTriangleLogs++, ctx.frame.fbp, unsigned(state.prim.type),
                unsigned(state.prim.tme), unsigned(state.prim.fst), unsigned(state.prim.fge),
                unsigned(state.prim.abe), ctx.tex0.tbp0,
                static_cast<unsigned long long>(ctx.test),
                fx0, fy0, fx1, fy1, fx2, fy2, v0.q, v1.q, v2.q, v0.z, v1.z, v2.z);
        }
    }

    // The hardware pipeline receives terrain already clipped by VU1.  A few
    // near-plane crossings currently survive that emulation and arrive as
    // triangles several screens wide, covering the whole race view.  Until
    // homogeneous polygon clipping is implemented, discard only those 3D
    // outliers; normal off-screen terrain still reaches the scissor clip.
    const int viewportWidth = static_cast<int>(ctx.scissor.x1) - static_cast<int>(ctx.scissor.x0) + 1;
    const int viewportHeight = static_cast<int>(ctx.scissor.y1) - static_cast<int>(ctx.scissor.y0) + 1;
    if (!state.prim.fst && state.prim.fge &&
        (unclippedMaxX - unclippedMinX > viewportWidth ||
         unclippedMaxY - unclippedMinY > viewportHeight))
    {
        return;
    }

    if (unclippedMaxX < ctx.scissor.x0 || unclippedMinX > ctx.scissor.x1 ||
        unclippedMaxY < ctx.scissor.y0 || unclippedMinY > ctx.scissor.y1)
        return;

    if (std::fabs(fx0) > 30000.0f || std::fabs(fx1) > 30000.0f || std::fabs(fx2) > 30000.0f ||
        std::fabs(fy0) > 30000.0f || std::fabs(fy1) > 30000.0f || std::fabs(fy2) > 30000.0f)
        return;

    // Reject triangles that span across the viewport due to extreme 16-bit coordinate wrap
    int minX = clampInt(unclippedMinX, ctx.scissor.x0, ctx.scissor.x1);
    int maxX = clampInt(unclippedMaxX, ctx.scissor.x0, ctx.scissor.x1);
    int minY = clampInt(unclippedMinY, ctx.scissor.y0, ctx.scissor.y1);
    int maxY = clampInt(unclippedMaxY, ctx.scissor.y0, ctx.scissor.y1);
    minY = firstBandRow(minY);
    if (minY > maxY)
        return;

    float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return;

    // Bounded, opt-in diagnostic: census which textures/fbp combos actually
    // reach rasterization during a race, to tell "never submitted" apart
    // from "submitted but rejected/invisible" for missing geometry (track
    // surface, trees, vehicle mesh) investigations.
    if (g_raceLoadStarted && s_traceTbpCensus)
    {
        static std::unordered_map<uint64_t, uint64_t> s_tbpCounts;
        static uint64_t s_totalTris = 0u;
        const uint64_t key = (static_cast<uint64_t>(ctx.frame.fbp) << 32) |
                              (static_cast<uint64_t>(static_cast<uint32_t>(ctx.tex0.psm)) << 24) |
                              (static_cast<uint64_t>(static_cast<uint32_t>(state.prim.tme)) << 20) |
                              static_cast<uint64_t>(ctx.tex0.tbp0 & 0x3FFFu);
        uint64_t &cnt = s_tbpCounts[key];
        if (cnt == 0u)
        {
            std::fprintf(stderr, "[tbp:census] NEW fbp=%u tbp=0x%x psm=0x%x tme=%u\n",
                         ctx.frame.fbp, ctx.tex0.tbp0,
                         static_cast<uint32_t>(ctx.tex0.psm), static_cast<uint32_t>(state.prim.tme));
        }
        ++cnt;
        if ((++s_totalTris % 3000u) == 0u)
        {
            std::fprintf(stderr, "[tbp:census] --- summary at %llu triangles, %zu distinct textures ---\n",
                         static_cast<unsigned long long>(s_totalTris), s_tbpCounts.size());
            for (const auto &kv : s_tbpCounts)
            {
                std::fprintf(stderr, "[tbp:census]   fbp=%u tbp=0x%x psm=0x%x tme=%u count=%llu\n",
                             static_cast<uint32_t>(kv.first >> 32),
                             static_cast<uint32_t>(kv.first & 0x3FFFu),
                             static_cast<uint32_t>((kv.first >> 24) & 0xFFu),
                             static_cast<uint32_t>((kv.first >> 20) & 0xFu),
                             static_cast<unsigned long long>(kv.second));
            }
        }
    }

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;

    // Barycentric edge equations stepped across the bbox. Row starts are
    // recomputed exactly; within a row the weights advance by constants.
    const float a01 = (fy1 - fy2) * winding * invAbsDenom;
    const float b01 = (fx2 - fx1) * winding * invAbsDenom;
    const float a12 = (fy2 - fy0) * winding * invAbsDenom;
    const float b12 = (fx0 - fx2) * winding * invAbsDenom;
    const float q0 = (v0.q > 1e-5f) ? v0.q : 1.0f;
    const float q1 = (v1.q > 1e-5f) ? v1.q : 1.0f;
    const float q2 = (v2.q > 1e-5f) ? v2.q : 1.0f;
    const bool textured = state.prim.tme;
    const bool flatUv = state.prim.fst;
    // ATV_GS_NO_PIXEL_PIPE=1: per-pixel state decoding as before session 33.
    static const bool s_noPixelPipe = std::getenv("ATV_GS_NO_PIXEL_PIPE") != nullptr;
    PixelPipe pixelPipe;
    TexturePipe texturePipe;
    if (!s_noPixelPipe)
    {
        setupPixelPipe(state, m_vram, m_readVramFuncs.data(), m_writeVramFuncs.data(), pixelPipe);
        if (textured)
            setupTexturePipe(state, m_vram, t_bandCache ? *t_bandCache : m_texturePageCache, t_drawClut ? t_drawClut : m_clut.data(), texturePipe);
    }
    const bool gouraud = state.prim.iip;
    // ATV_GS_NO_FOG=1: skip the fog blend (pre-session-54 look).
    static const bool s_noFog = std::getenv("ATV_GS_NO_FOG") != nullptr;
    const bool fogOn = state.prim.fge && !s_noFog;
    uint32_t *idRows = nullptr;
    uint32_t *uvRows = nullptr;
    uint32_t drawId = 0u;
    const uint32_t idFbpBlock = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const uint32_t idFbw = ctx.frame.fbw >= 1u ? ctx.frame.fbw : 10u;
    if (gsIdBuf().intervalSec > 0.0 && g_raceLoadStarted)
    {
        char desc[256];
        std::snprintf(desc, sizeof(desc),
                      "tme=%u tbp=0x%x tbw=%u psm=0x%x cbp=0x%x cpsm=0x%x csa=%u tw=%u th=%u tcc=%u tfx=%u "
                      "abe=%u alpha=0x%llx test=0x%llx fge=%u fst=%u iip=%u up=%s",
                      static_cast<uint32_t>(state.prim.tme), ctx.tex0.tbp0, static_cast<uint32_t>(ctx.tex0.tbw),
                      static_cast<uint32_t>(ctx.tex0.psm), static_cast<uint32_t>(ctx.tex0.cbp),
                      static_cast<uint32_t>(ctx.tex0.cpsm), static_cast<uint32_t>(ctx.tex0.csa),
                      static_cast<uint32_t>(state.textureWidth), static_cast<uint32_t>(state.textureHeight),
                      static_cast<uint32_t>(ctx.tex0.tcc), static_cast<uint32_t>(ctx.tex0.tfx),
                      static_cast<uint32_t>(state.prim.abe), static_cast<unsigned long long>(ctx.alpha),
                      static_cast<unsigned long long>(ctx.test), static_cast<uint32_t>(state.prim.fge),
                      static_cast<uint32_t>(state.prim.fst), static_cast<uint32_t>(state.prim.iip),
                      state.prim.tme ? gsIdLastUploads(ctx.tex0.tbp0).c_str() : "-");
        drawId = gsIdFor(desc);
        idRows = gsIdRows(ctx.frame.fbp);
        uvRows = gsIdRows(ctx.frame.fbp, true);
    }
    const auto &tex = ctx.tex0;
    static const bool s_traceGroundSample = std::getenv("ATV_TRACE_GROUND_SAMPLE") != nullptr;

    for (int y = minY; y <= maxY; y = nextBandRow(y))
    {
        t_ownedBandRow = y;
        const float py = static_cast<float>(y) + 0.5f;
        const float px0 = static_cast<float>(minX) + 0.5f;
        float w0 = a01 * (px0 - fx2) + b01 * (py - fy2);
        float w1 = a12 * (px0 - fx2) + b12 * (py - fy2);
        bool insideSpan = false;

        for (int x = minX; x <= maxX; ++x)
        {
            const float w2 = 1.0f - w0 - w1;
            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
            {
                w0 += a01;
                w1 += a12;
                if (insideSpan)
                    break;
                continue;
            }
            insideSpan = true;
            t_triangleCovered = true;

            const double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

            uint8_t r, g, b, a;
            uint32_t texelUv = 0u;
            if (gouraud)
            {
                r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (textured)
            {
                float is, it, iq;
                uint16_t iu, iv;
                if (flatUv)
                {
                    iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                    iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    // The GS DDA interpolates the homogeneous S, T and Q
                    // values. Texel coordinates are calculated from S/Q and
                    // T/Q only after interpolation.
                    is = v0.s * w0 + v1.s * w1 + v2.s * w2;
                    it = v0.t * w0 + v1.t * w1 + v2.t * w2;
                    iq = q0 * w0 + q1 * w1 + q2 * w2;
                    iu = 0;
                    iv = 0;
                }

                if (uvRows != nullptr)
                {
                    const float fu = flatUv ? static_cast<float>(iu) / 16.0f : is / iq * static_cast<float>(state.textureWidth);
                    const float fv = flatUv ? static_cast<float>(iv) / 16.0f : it / iq * static_cast<float>(state.textureHeight);
                    texelUv = (static_cast<uint32_t>(std::clamp(fu, 0.0f, 65535.0f)) & 0xFFFFu) |
                              (static_cast<uint32_t>(std::clamp(fv, 0.0f, 65535.0f)) << 16);
                }
                const uint32_t texel = texturePipe.ready ? sampleTextureFast(texturePipe, is, it, iq, iu, iv)
                                                         : SampleTexture(state, is, it, iq, iu, iv);
                // ATV_GS_IDBUF_TEXEL=1: the .uv dump holds the sampled RGBA texel instead.
                static const bool s_idTexel = std::getenv("ATV_GS_IDBUF_TEXEL") != nullptr;
                if (uvRows != nullptr && s_idTexel)
                    texelUv = texel;

                const uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                const uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                const uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                const uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const uint8_t shadeR = r;
                const uint8_t shadeG = g;
                const uint8_t shadeB = b;
                const uint8_t shadeA = a;
                const TextureCombineResult color = combineTexture(tex, shadeR, shadeG, shadeB, shadeA, tr, tg, tb, ta);

                // Opt-in probe at a ground pixel: distinguish missing texture
                // data from zero vertex colour before changing rendering.
                static uint32_t s_groundSampleLogs = 0u;
                const bool terrainTexture =
                    tex.tbp0 == 0x3880u || tex.tbp0 == 0x3A80u || tex.tbp0 == 0x3C80u;
                if (s_traceGroundSample && g_raceLoadStarted && terrainTexture && x == 250 && y == 400 &&
                    s_groundSampleLogs < 64u)
                {
                    ++s_groundSampleLogs;
                    std::fprintf(stderr,
                        "[ground:sample] tbp=%u tbw=%u psm=%u tcc=%u tfx=%u abe=%u texa=(%u,%u,%u) alpha=%llx "
                        "stq=(%g,%g,%g) texel=%08x shade=(%u,%u,%u,%u) out=(%u,%u,%u,%u) z=%g\n",
                        tex.tbp0, unsigned(tex.tbw), unsigned(tex.psm),
                        unsigned(tex.tcc), unsigned(tex.tfx), unsigned(state.prim.abe),
                        unsigned(state.texa.ta0), state.texa.aem ? 1u : 0u, unsigned(state.texa.ta1),
                        static_cast<unsigned long long>(ctx.alpha),
                        is, it, iq, texel,
                        unsigned(shadeR), unsigned(shadeG), unsigned(shadeB), unsigned(shadeA),
                        unsigned(color.r), unsigned(color.g), unsigned(color.b), unsigned(color.a), z);
                }

                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            // GS fog: C = (F*C + (255-F)*FOGCOL) >> 8, F interpolated per
            // pixel. Applied before alpha blending; alpha is untouched.
            if (fogOn)
            {
                const int f = clampU8(static_cast<int>(v0.fog * w0 + v1.fog * w1 + v2.fog * w2 + 0.5f));
                const int nf = 255 - f;
                r = static_cast<uint8_t>((f * r + nf * state.fogR) >> 8);
                g = static_cast<uint8_t>((f * g + nf * state.fogG) >> 8);
                b = static_cast<uint8_t>((f * b + nf * state.fogB) >> 8);
            }

            const bool idTrack = idRows != nullptr && x >= 0 && y >= 0 &&
                                 static_cast<uint32_t>(x) < kIdBufW && static_cast<uint32_t>(y) < kIdBufH;
            const uint32_t idBefore = idTrack ? ReadVramUnlocked(ctx.frame.psm, idFbpBlock, idFbw, x, y) : 0u;
            if (pixelPipe.vram)
                writePixelFast(pixelPipe, x, y, static_cast<u32>(z + 0.5), r, g, b, a);
            else
                WritePixel(state, x, y, static_cast<u32>(z + 0.5), r, g, b, a, 0u);
            if (idTrack && ReadVramUnlocked(ctx.frame.psm, idFbpBlock, idFbw, x, y) != idBefore)
            {
                idRows[static_cast<uint32_t>(y) * kIdBufW + static_cast<uint32_t>(x)] = drawId;
                uvRows[static_cast<uint32_t>(y) * kIdBufW + static_cast<uint32_t>(x)] = texelUv;
            }

            w0 += a01;
            w1 += a12;
        }
    }
}

void GSCpuBackend::DrawLine(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    if (std::abs(x1 - x0) > 2048 || std::abs(y1 - y0) > 2048)
        return;

    const int lineMinX = std::min(x0, x1);
    const int lineMaxX = std::max(x0, x1);
    const int lineMinY = std::min(y0, y1);
    const int lineMaxY = std::max(y0, y1);
    if (g_ps2XgkickCount >= 1000u && lineMinX <= 550 && lineMaxX >= 550 && lineMinY <= 40 && lineMaxY >= 40 &&
        (ctx.frame.fbp == 0u || ctx.frame.fbp == 140u))
    {
        static uint32_t s_lineLogs = 0u;
        if (s_lineLogs < 16u)
        {
            std::fprintf(stderr,
                "[line:target] fbp=%u pt0=(%d,%d) pt1=(%d,%d) v0.z=%.0f v1.z=%.0f rgba0=(%u,%u,%u,%u)\n",
                ctx.frame.fbp, x0, y0, x1, y1, v0.z, v1.z,
                static_cast<uint32_t>(v0.r), static_cast<uint32_t>(v0.g),
                static_cast<uint32_t>(v0.b), static_cast<uint32_t>(v0.a));
            ++s_lineLogs;
        }
    }

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r, g, b, a;
        if (state.prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        double z = (v0.z + (v1.z - v0.z) * t);
        const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
        WritePixel(state, x0, y0, static_cast<u32>(z), r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
        ++step;
    }
}

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (command.direction == 0u)
        gsIdNoteUpload(command.bitbltbuf.dbp, command.bitbltbuf.dbw, command.bitbltbuf.dpsm,
                       command.trxpos.dsax, command.trxpos.dsay, command.trxreg.rrw, command.trxreg.rrh);
    if (gsCapture().enabled)
    {
        MaybeStartCapture();
        if (gsCaptureActive(this))
            gsCaptureWrite(GSCaptureBeginTransfer, &command, sizeof(command));
    }
    m_uploadRangeValid = false;
    if (command.direction == 0u && m_uploadNoFlush && m_bandCount > 1u)
    {
        // Host-to-local writes only its destination. Queued draws that
        // neither sample nor draw those pages give the same result whether
        // they run before or after it, so they can stay queued.
        const uint32_t dbw = std::max<uint32_t>(command.bitbltbuf.dbw, 1u);
        m_uploadRange = surfaceRange<BandWrite>(command.bitbltbuf.dbp * kVramBlockBytes,
                                                std::max<uint32_t>(dbw * 64u, static_cast<uint32_t>(command.trxpos.dsax) + command.trxreg.rrw),
                                                command.bitbltbuf.dpsm,
                                                static_cast<uint32_t>(command.trxpos.dsay) + command.trxreg.rrh, 0u, kVramPageBytes);
        m_uploadRange.begin &= ~(kVramPageBytes - 1u);
        if (m_uploadRange.end != UINT32_MAX)
            m_uploadRange.end = (m_uploadRange.end + kVramPageBytes - 1u) & ~(kVramPageBytes - 1u);
        m_uploadRangeValid = true;
        // Bands still drawing the previous batch never touch these pages.
        if (InFlightTouches(m_uploadRange))
            WaitBandsIdle();
        if (PendingTouches(m_uploadRange))
            FlushBands(BandFlushReason::Upload);
    }
    else
    {
        FlushBands(BandFlushReason::Upload);
    }
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this) && data)
    {
        gsCaptureWrite(GSCaptureUploadImage, &sizeBytes, sizeof(sizeBytes), data, sizeBytes);
    }
    if (m_uploadRangeValid && m_transferState.direction == 0u)
    {
        if (InFlightTouches(m_uploadRange))
            WaitBandsIdle();
        if (PendingTouches(m_uploadRange))
            FlushBands(BandFlushReason::Upload);
    }
    else
    {
        FlushBands(BandFlushReason::Upload);
    }
    // An upload makes later draws see new texels even without a TEXFLUSH. The
    // game recycles 4 texture slots all frame; with 64 cached pages a slot's
    // old texels survived re-uploads (garbage on the first object drawn with
    // the new texture). Mark a texture flush for queued band draws and drop
    // the immediate cache. ATV_GS_UPLOAD_NO_TEXFLUSH=1 restores the old path.
    static const bool s_uploadNoTexFlush = std::getenv("ATV_GS_UPLOAD_NO_TEXFLUSH") != nullptr;
    if (!s_uploadNoTexFlush && data && sizeBytes != 0u && m_transferState.direction == 0u)
    {
        if (!m_bandPending.empty())
        {
            if (m_bandTextureFlushes.empty() || m_bandTextureFlushes.back() != m_bandPending.size())
                m_bandTextureFlushes.push_back(m_bandPending.size());
        }
        else if (m_bandInFlightActive)
        {
            // Bands 1.. may still be drawing: drop their caches before the
            // next batch's first primitive instead (same effect for draws).
            if (m_bandTextureFlushes.empty() || m_bandTextureFlushes.back() != 0u)
                m_bandTextureFlushes.push_back(0u);
        }
        else
        {
            for (auto &cache : m_bandCaches)
                cache->Invalidate();
        }
        m_texturePageCache.Invalidate();
    }
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    if (m_transfer.trxreg.rrw == 0u || m_transfer.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
        return;

    const uint32_t dbp = m_transfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = m_transfer.bitbltbuf.dpsm;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t dsax = m_transfer.trxpos.dsax;
    uint32_t offset = 0u;

    static const bool s_traceGroundSample = std::getenv("ATV_TRACE_GROUND_SAMPLE") != nullptr;
    if (s_traceGroundSample &&
        (dbp == 0x3880u || dbp == 0x3A80u))
    {
        static uint32_t s_terrainUploadLogs = 0u;
        if (s_terrainUploadLogs < 24u)
        {
            uint32_t nonzeroBytes = 0u;
            for (uint32_t i = 0u; i < sizeBytes; ++i)
                nonzeroBytes += data[i] != 0u ? 1u : 0u;
            std::fprintf(stderr,
                "[ground:upload] n=%u dbp=0x%x bytes=%u nonzero=%u first=%02x%02x%02x%02x copied=%u/%u\n",
                s_terrainUploadLogs++, dbp, sizeBytes, nonzeroBytes,
                sizeBytes > 0u ? data[0] : 0u, sizeBytes > 1u ? data[1] : 0u,
                sizeBytes > 2u ? data[2] : 0u, sizeBytes > 3u ? data[3] : 0u,
                m_transferState.copiedPixels, m_transferState.totalPixels);
        }
    }

    // Column and row of copiedPixels in the transfer rectangle.
    uint32_t column = m_transferState.copiedPixels % rrw;
    uint32_t row = m_transferState.copiedPixels / rrw;
    auto advancePixel = [&](uint32_t count)
    {
        const uint32_t totalPixels = m_transferState.totalPixels;
        m_transferState.copiedPixels =
            std::min<uint32_t>(totalPixels, m_transferState.copiedPixels + count);

        if (m_transferState.copiedPixels >= totalPixels)
        {
            m_transferState.direction = 3u;
            m_transferState.totalPixels = 0u;
            return;
        }

        column += count;
        while (column >= rrw)
        {
            column -= rrw;
            ++row;
        }
        m_transferState.x = dsax + column;
        m_transferState.y = m_transfer.trxpos.dsay + row;
    };

    while (offset < sizeBytes && m_transferState.direction == 0u)
    {
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            if (sizeBytes - offset < 4u)
                return;
            uint32_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 4u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            if (sizeBytes - offset < 3u)
                return;
            const uint32_t value = static_cast<uint32_t>(data[offset]) |
                                   (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
                                   (static_cast<uint32_t>(data[offset + 2u]) << 16u);
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 3u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            if (sizeBytes - offset < 2u)
                return;
            uint16_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 2u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_T8:
        case GS_PSM_T8H:
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, data[offset++]);
            advancePixel(1u);
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint8_t packed = data[offset++];
            const uint32_t firstPixel = m_transferState.copiedPixels;
            const uint32_t dsay = m_transfer.trxpos.dsay;
            WriteVramUnlocked(dpsm, dbp, dbw, dsax + column, dsay + row, packed & 0x0Fu);
            if (firstPixel + 1u < m_transferState.totalPixels)
            {
                const bool wraps = column + 1u >= rrw;
                WriteVramUnlocked(dpsm, dbp, dbw,
                                  wraps ? dsax : dsax + column + 1u,
                                  wraps ? dsay + row + 1u : dsay + row,
                                  (packed >> 4u) & 0x0Fu);
            }
            advancePixel(std::min<uint32_t>(2u, m_transferState.totalPixels - firstPixel));
            break;
        }
        default:
            return;
        }
    }
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }

    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this))
        gsCaptureWrite(GSCaptureConsume, &maxBytes, sizeof(maxBytes));
    FlushBands();
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (gsCaptureActive(this))
    {
        struct { GSContext context; uint32_t rgba; } record{context, rgba};
        gsCaptureWrite(GSCaptureClear, &record, sizeof(record));
    }
    FlushBands(BandFlushReason::Clear);
    if (!m_vram)
        return false;

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = context.frame.fbw >= 1u ? context.frame.fbw : 10u;
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }
    return false;
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    // Snapshot local memory under the backend lock, then perform the expensive
    // display conversion without holding the producer-side raster lock.
    thread_local std::vector<uint8_t> snapshot;
    SnapshotVram(snapshot);
    return PresentSnapshot(snapshot, request);
}

PresentationFrame GSCpuBackend::PresentSnapshot(std::vector<uint8_t> &snapshot, const GSPresentationRequest &request)
{
    if (snapshot.empty())
        return {};
    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    const bool fieldMode = smode2.interlaced && !smode2.frameMode;
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);

    if (!valid1 && !valid2)
    {
        // This is the exact path that produces the MAGENTA fallback in
        // UploadFrame (no frame available to present). The log below used to
        // sit AFTER this early-out, so it never fired for the failure case --
        // grepping [gs:present] for a pmode transition to 0x0 found nothing
        // because the failing frames were never logged at all. Log the first
        // occurrences and a steady trickle afterward so a repro can show the
        // exact transition from a healthy pmode to whatever it reads here.
        static std::atomic<uint32_t> s_presentFailLogs{0u};
        const uint32_t n = s_presentFailLogs.fetch_add(1u, std::memory_order_relaxed);
        if (n < 32u || (n % 120u) == 0u)
        {
            std::cerr << "[gs:present-fail] n=" << n
                      << " v1=" << valid1 << " v2=" << valid2
                      << " pmode=0x" << std::hex << request.pmode
                      << " enableCrt1=" << std::dec << pmode.enableCrt1 << " enableCrt2=" << pmode.enableCrt2
                      << " dispfb1=0x" << std::hex << request.dispfb1 << " disp1=0x" << request.display1
                      << " dispfb2=0x" << request.dispfb2 << " disp2=0x" << request.display2
                      << std::dec
                      << std::endl;
        }
        return result;
    }

    {
        static std::atomic<uint32_t> s_presentLogs{0u};
        const uint32_t n = s_presentLogs.fetch_add(1u, std::memory_order_relaxed);
        if (n < 16u || (n % 60u) == 0u)
        {
            std::cerr << "[gs:present] n=" << n
                      << " v1=" << valid1 << " v2=" << valid2
                      << " pmode=0x" << std::hex << request.pmode
                      << " dispfb1=0x" << request.dispfb1 << " disp1=0x" << request.display1
                      << " dispfb2=0x" << request.dispfb2 << " disp2=0x" << request.display2
                      << " ctx0=(fbp=" << std::dec << request.contextFrames[0].fbp << ",fbw=" << request.contextFrames[0].fbw << ",psm=" << (unsigned)request.contextFrames[0].psm << ")"
                      << " ctx1=(fbp=" << request.contextFrames[1].fbp << ",fbw=" << request.contextFrames[1].fbw << ",psm=" << (unsigned)request.contextFrames[1].psm << ")"
                      << std::endl;
        }
    }

    // Both race display circuits scan the same completed framebuffer. Drawing
    // contexts refer to the next frame and must never be substituted here or
    // composited as a supposed HUD: that exposes unfinished backbuffer pixels.
    // With no background-color blend, merging identical sources is an identity.
    // Read once, including the extra line used by the larger display circuit.
    if (valid1 && valid2 && !pmode.slbg &&
        displayFrame1.fbp == displayFrame2.fbp && displayFrame1.fbw == displayFrame2.fbw &&
        displayFrame1.psm == displayFrame2.psm && origin1.x == origin2.x && origin1.y == origin2.y)
    {
        result.width = std::max(width1, width2);
        result.height = std::max(height1, height2);
        GSFrameReg selected = displayFrame2;
        repairDisplayFrame(selected, result.width, request.contextFrames, 2u);
        if (!CopyFrameToHostRgba(selected, result.width, result.height, result.pixels,
                                 false, true, true, origin2.x, origin2.y))
            return {};
        result.displayFbp = result.sourceFbp = displayFrame2.fbp;
        if (g_raceLoadStarted)
        {
            const uint32_t n = g_gsRacePresents.fetch_add(1u, std::memory_order_relaxed);
            static const int s_skyTriAt = []
            {
                const char *e = std::getenv("ATV_TRACE_SKYTRI");
                return e ? std::atoi(e) : -1;
            }();
            // Save the frame shown right after the traced draws.
            if (s_skyTriAt >= 0 && n == static_cast<uint32_t>(s_skyTriAt) + 1u)
                if (FILE *fp = std::fopen("skytri_frame.ppm", "wb"))
                {
                    std::fprintf(fp, "P6\n%u %u\n255\n", result.width, result.height);
                    for (uint32_t y = 0; y < result.height; ++y)
                        for (uint32_t x = 0; x < result.width; ++x)
                            std::fwrite(&result.pixels[(y * kHostFrameWidth + x) * 4u], 1, 3, fp);
                    std::fclose(fp);
                }
        }
        {
            GsIdBuf &ib = gsIdBuf();
            const auto now = std::chrono::steady_clock::now();
            if (ib.intervalSec > 0.0 && g_raceLoadStarted &&
                std::chrono::duration<double>(now - ib.lastDump).count() >= ib.intervalSec)
            {
                ib.lastDump = now;
                std::lock_guard<std::mutex> lock(ib.mutex);
                auto it = ib.perFbp.find(displayFrame2.fbp);
                if (it != ib.perFbp.end())
                {
                    std::filesystem::create_directories("idbuf_dump");
                    const uint32_t n = ib.dumps++;
                    std::fprintf(stderr, "[idbuf] frame_%03u present=%u\n", n, gsRacePresentCount());
                    char name[64];
                    std::snprintf(name, sizeof(name), "idbuf_dump/frame_%03u.ppm", n);
                    if (FILE *fp = std::fopen(name, "wb"))
                    {
                        std::fprintf(fp, "P6\n%u %u\n255\n", result.width, result.height);
                        for (uint32_t y = 0; y < result.height; ++y)
                            for (uint32_t x = 0; x < result.width; ++x)
                                std::fwrite(&result.pixels[(y * kHostFrameWidth + x) * 4u], 1, 3, fp);
                        std::fclose(fp);
                    }
                    std::snprintf(name, sizeof(name), "idbuf_dump/frame_%03u.ids", n);
                    if (FILE *fp = std::fopen(name, "wb"))
                    {
                        const uint32_t hdr[2] = {result.width, result.height};
                        std::fwrite(hdr, 4, 2, fp);
                        for (uint32_t y = 0; y < result.height; ++y)
                            for (uint32_t x = 0; x < result.width; ++x)
                            {
                                const uint32_t sx = origin2.x + x, sy = origin2.y + y;
                                const uint32_t id = (sx < kIdBufW && sy < kIdBufH) ? it->second[sy * kIdBufW + sx] : 0u;
                                std::fwrite(&id, 4, 1, fp);
                            }
                        std::fclose(fp);
                    }
                    auto uvIt = ib.perFbpUv.find(displayFrame2.fbp);
                    std::snprintf(name, sizeof(name), "idbuf_dump/frame_%03u.uv", n);
                    if (uvIt != ib.perFbpUv.end())
                        if (FILE *fp = std::fopen(name, "wb"))
                        {
                            const uint32_t hdr[2] = {result.width, result.height};
                            std::fwrite(hdr, 4, 2, fp);
                            for (uint32_t y = 0; y < result.height; ++y)
                                for (uint32_t x = 0; x < result.width; ++x)
                                {
                                    const uint32_t sx = origin2.x + x, sy = origin2.y + y;
                                    const uint32_t uv = (sx < kIdBufW && sy < kIdBufH) ? uvIt->second[sy * kIdBufW + sx] : 0u;
                                    std::fwrite(&uv, 4, 1, fp);
                                }
                            std::fclose(fp);
                        }
                    if (FILE *fp = std::fopen("idbuf_dump/ids.txt", "w"))
                    {
                        for (size_t i = 0; i < ib.descs.size(); ++i)
                            std::fprintf(fp, "%zu %s\n", i, ib.descs[i].c_str());
                        std::fclose(fp);
                    }
                }
            }
        }
        if (fieldMode && result.height < 400u)
            applyFieldPresentation(result.pixels, result.width, result.height, oddField);
        return result;
    }

    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t height,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        selected = displayFrame;
        repairDisplayFrame(selected, width, request.contextFrames, 2u);
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(selected, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;

        if (allowPreferred && !usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbw != (width + 63u) / 64u || candidate.psm != 0u)
                    continue;
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        // DISPFB stuck on page 0 while draw context is another FBP: prefer the
        // scene buffer. Do NOT require candidate nonblack > display nonblack —
        // FBP0 often holds a denser HUD while the 3D scene lives at FBP140.
        if (allowPreferred && !usedPreferred)
        {
            const bool displayFlat = looksLikeFlatFill(pixels, width, height);
            if (!displayFlat)
            {
                return true;
            }
            GSFrameReg best{};
            std::vector<uint8_t> bestPixels;
            uint32_t bestScore = 0u;
            bool haveBest = false;
            std::vector<GSFrameReg> candidateList;
            for (const GSFrameReg &c : request.contextFrames)
                candidateList.push_back(c);
            if (displayFrame.fbp != 140u)
                candidateList.push_back(GSFrameReg{140u, 10u, 0u});
            if (displayFrame.fbp != 0u)
                candidateList.push_back(GSFrameReg{0u, 10u, 0u});
            for (const GSFrameReg &candidate : candidateList)
            {
                if (candidate.fbw != (width + 63u) / 64u || candidate.psm != 0u)
                    continue;
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                const uint32_t score = countNonBlackPixels(candidatePixels, width, height);
                if (score == 0u)
                    continue;
                if (looksLikeFlatFill(candidatePixels, width, height))
                    continue;
                if (!haveBest || score > bestScore)
                {
                    best = candidate;
                    bestPixels.swap(candidatePixels);
                    bestScore = score;
                    haveBest = true;
                }
            }
            if (haveBest)
            {
                selected = best;
                pixels.swap(bestPixels);
                usedPreferred = true;
            }
        }
        return true;
    };

    const bool isFull1 = (width1 >= 320u && height1 >= 224u);
    const bool isFull2 = (width2 >= 320u && height2 >= 224u);

    // Page 0 is never composited over the display as a "HUD": in this game
    // FBP 0 and FBP 140 are the two draw buffers, so page 0 is the frame
    // being drawn (menu: fresh movie picture under a half-drawn menu).
    // If CRT2 is a full display buffer while CRT1 is only a sub-tile (e.g. 64x64 IPU scratch buffer),
    // present CRT2 directly as the active game display buffer without blending CRT1 over the screen.
    if (valid2 && isFull2 && (!valid1 || !isFull1))
    {
        result.width = width2;
        result.height = height2;
        GSFrameReg selected2{};
        bool usedPreferred2 = false;
        if (copySource(displayFrame2, origin2, width2, height2, true, false, selected2, result.pixels, usedPreferred2))
        {
            result.usedPreferred = usedPreferred2;
            result.displayFbp = displayFrame2.fbp;
            result.sourceFbp = selected2.fbp;
            if (fieldMode && result.height < 400u)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            return result;
        }
    }

    // Conversely, if CRT1 is a full display buffer while CRT2 is not, present CRT1 directly.
    if (valid1 && isFull1 && (!valid2 || !isFull2))
    {
        result.width = width1;
        result.height = height1;
        GSFrameReg selected1{};
        bool usedPreferred1 = false;
        if (copySource(displayFrame1, origin1, width1, height1, true, false, selected1, result.pixels, usedPreferred1))
        {
            result.usedPreferred = usedPreferred1;
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            if (fieldMode && result.height < 400u)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            return result;
        }
    }

    if (valid1 && valid2)
    {
        GSFrameReg selected1 = displayFrame1;
        GSFrameReg selected2 = displayFrame2;
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copySource(selected1, origin1, width1, height1, false, true, selected1, crt1, preferred1) &&
            copySource(selected2, origin2, width2, height2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (pmode.enableCrt2 || !crt2.empty())
            {
                for (uint32_t y = 0; y < height2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            }
            else if (pmode.enableCrt1 || !crt1.empty())
            {
                for (uint32_t y = 0; y < height1; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt1.data() + y * kHostFrameWidth * 4u, width1 * 4u);
            }

            const bool crt1Flat = looksLikeFlatFill(crt1, width1, height1);
            const bool crt2Flat = looksLikeFlatFill(crt2, width2, height2);

            if (crt2Flat && !crt1Flat)
            {
                // CRT2 is empty/flat while CRT1 has the scene/menu: display CRT1 directly
                for (uint32_t y = 0; y < height1; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt1.data() + y * kHostFrameWidth * 4u, width1 * 4u);
            }
            else if (!crt1.empty())
            {
                const uint8_t r0 = crt1[0], g0 = crt1[1], b0 = crt1[2];
                for (uint32_t y = 0; y < height1; ++y)
                {
                    for (uint32_t x = 0; x < width1; ++x)
                    {
                        const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                        uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                        if ((src[0] | src[1] | src[2]) == 0u)
                            continue;
                        // If this pixel in CRT1 is the flat clear background color with zero alpha, skip so CRT2 shows through
                        if (src[3] == 0u && std::abs((int)src[0] - (int)r0) <= 4 &&
                                            std::abs((int)src[1] - (int)g0) <= 4 &&
                                            std::abs((int)src[2] - (int)b0) <= 4)
                        {
                            continue;
                        }
                        const uint32_t factor = (src[3] != 0u) ? std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u) : 255u;
                        dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                        dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                        dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    }
                }
            }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            // Host is progressive. CRT field weave on a full 448-line UI framebuffer
            // duplicates even/odd rows and makes menu text comb-toothed.
            if (fieldMode && result.height < 400u)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            result.displayFbp = pmode.enableCrt1 ? displayFrame1.fbp : displayFrame2.fbp;
            result.sourceFbp = pmode.enableCrt1 ? selected1.fbp : selected2.fbp;
            return result;
        }
    }

    const GSFrameReg &displayFrame = (valid2 && isFull2) ? displayFrame2 : (valid1 ? displayFrame1 : displayFrame2);
    const GSDisplayReadOrigin &origin = (valid2 && isFull2) ? origin2 : (valid1 ? origin1 : origin2);
    result.width = (valid2 && isFull2) ? width2 : (valid1 ? width1 : width2);
    result.height = (valid2 && isFull2) ? height2 : (valid1 ? height1 : height2);
    GSFrameReg selected = displayFrame;
    if (!copySource(displayFrame, origin, result.width, result.height, true, false, selected, result.pixels, result.usedPreferred))
        return {};
    if (fieldMode && result.height < 400u)
        applyFieldPresentation(result.pixels, result.width, result.height, oddField);
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return result;
}
