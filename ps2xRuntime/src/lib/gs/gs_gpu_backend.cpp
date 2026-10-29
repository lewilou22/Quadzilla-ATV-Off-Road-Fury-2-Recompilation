#include "gs_gpu_backend.h"

#if defined(_WIN32)

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "gs_texture_replace.h"
#define XXH_INLINE_ALL
#include "xxhash.h" // xxHash 0.8.3 (BSD 2-Clause), as gs_texture_replace.cpp

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <xmmintrin.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

extern bool g_raceLoadStarted;

namespace
{
    constexpr uint32_t kPages = 512u;
    constexpr uint32_t kPageBytes = 8192u;
    constexpr uint32_t kHostFrameWidth = 640u; // row stride of 1x presentation frames (gs_frontend.cpp)
    using PageSet = std::bitset<kPages>;

    bool isFourBit(uint8_t psm) { return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH; }
    bool isEightBit(uint8_t psm) { return psm == GS_PSM_T8 || psm == GS_PSM_T8H; }
    bool isSixteenBit(uint8_t psm) { return psm == GS_PSM_CT16 || psm == GS_PSM_CT16S || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S; }

    void pageDims(uint8_t psm, uint32_t &pw, uint32_t &ph)
    {
        if (isSixteenBit(psm))
            pw = 64u, ph = 64u;
        else if (psm == GS_PSM_T8)
            pw = 128u, ph = 64u;
        else if (psm == GS_PSM_T4)
            pw = 128u, ph = 128u;
        else
            pw = 64u, ph = 32u;
    }

    // Block numbers inside a page (GS manual): 8x4 blocks for CT32/T8 pages, 4x8 for CT16/T4.
    constexpr uint8_t kBlocks32[4][8] = {{0, 1, 4, 5, 16, 17, 20, 21}, {2, 3, 6, 7, 18, 19, 22, 23},
                                         {8, 9, 12, 13, 24, 25, 28, 29}, {10, 11, 14, 15, 26, 27, 30, 31}};
    constexpr uint8_t kBlocks16[8][4] = {{0, 2, 8, 10}, {1, 3, 9, 11}, {4, 6, 12, 14}, {5, 7, 13, 15},
                                         {16, 18, 24, 26}, {17, 19, 25, 27}, {20, 22, 28, 30}, {21, 23, 29, 31}};

    // Highest block number a [0, x1] x [y0 % page, y1 % page] rectangle inside one page uses (31 = unknown).
    uint32_t maxBlockInPage(uint8_t psm, uint32_t x1, uint32_t y1)
    {
        uint32_t best = 0;
        if (psm == GS_PSM_CT32 || psm == GS_PSM_CT24 || psm == GS_PSM_T8)
        {
            const uint32_t bw = psm == GS_PSM_T8 ? 16u : 8u, bh = psm == GS_PSM_T8 ? 16u : 8u;
            for (uint32_t by = 0; by <= std::min(3u, y1 / bh); ++by)
                for (uint32_t bx = 0; bx <= std::min(7u, x1 / bw); ++bx)
                    best = std::max<uint32_t>(best, kBlocks32[by][bx]);
            return best;
        }
        if (psm == GS_PSM_CT16 || psm == GS_PSM_T4)
        {
            const uint32_t bw = psm == GS_PSM_T4 ? 32u : 16u, bh = psm == GS_PSM_T4 ? 16u : 8u;
            for (uint32_t by = 0; by <= std::min(7u, y1 / bh); ++by)
                for (uint32_t bx = 0; bx <= std::min(3u, x1 / bw); ++bx)
                    best = std::max<uint32_t>(best, kBlocks16[by][bx]);
            return best;
        }
        return 31u;
    }

    // Pages a bw-wide surface at block bp covers on rows [y0, y1], columns [0, x1] (conservative:
    // whole page rows). Columns past the buffer width (a 1024-wide texture of a 640-wide frame)
    // address the pages after each row's, so the last row reaches that much further.
    void addPages(PageSet &set, uint32_t bp, uint32_t bw, uint8_t psm, uint32_t y0, uint32_t y1, uint32_t x1 = UINT32_MAX)
    {
        uint32_t pw, ph;
        pageDims(psm, pw, ph);
        y1 = std::max(y0, y1);
        const uint32_t perRow = std::max<uint32_t>(1u, std::max<uint32_t>(bw, 1u) * 64u / pw);
        const uint32_t first = (bp >> 5) + (y0 / ph) * perRow;
        const uint32_t rows = y1 / ph - y0 / ph + 1u;
        const uint32_t beyond = x1 != UINT32_MAX && x1 / pw + 1u > perRow ? x1 / pw + 1u - perRow : 0u;
        // A base inside a page spills into the next page when the blocks used run past it.
        bool spill = (bp & 31u) != 0u;
        if (spill && rows == 1u && x1 < pw)
            spill = (bp & 31u) + maxBlockInPage(psm, x1, y1 % ph) >= 32u;
        const uint32_t count = std::min<uint32_t>(kPages, rows * perRow + beyond + (spill ? 1u : 0u));
        for (uint32_t i = 0; i < count; ++i)
            set.set((first + i) % kPages);
    }

    uint32_t rgba5551To8888(uint32_t c)
    {
        return (((c >> 0) & 0x1Fu) << 3) | ((((c >> 5) & 0x1Fu) << 3) << 8) | ((((c >> 10) & 0x1Fu) << 3) << 16) |
               ((((c >> 15) & 1u) << 7) << 24);
    }

    uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;
        const bool rgbZero = (texel & 0x00FFFFFFu) == 0u;
        uint32_t a = texel >> 24;
        if (psm == GS_PSM_CT24)
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
        else if (psm == GS_PSM_CT16 || psm == GS_PSM_CT16S)
            a = (a & 0x80u) ? texa.ta1 : ((texa.aem && rgbZero) ? 0u : texa.ta0);
        return (texel & 0x00FFFFFFu) | (a << 24);
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    float zScale(uint8_t zpsm)
    {
        switch (zpsm)
        {
        case GS_PSM_Z32: return 1.0f / 4294967296.0f;
        case GS_PSM_Z16: case GS_PSM_Z16S: return 1.0f / 65536.0f;
        default: return 1.0f / 16777216.0f;
        }
    }

    const char *kShaderSource = R"HLSL(
cbuffer Draw : register(b0)
{
    float4 g_ndc;   // x: 2/W, y: 2/H of the 1x target
    uint4  g_tex;   // x: kind (0 none, 1 CT32, 2 CT24, 3 CT16, 4 palette, 5 HD image, 6 / 7 a GPU render
                    // target as CT32 / CT24), y: width, z: height, w: bilinear
    uint4  g_wrap;  // x: wrap U, y: wrap V, z: MINU | MAXU << 16, w: MINV | MAXV << 16
    uint4  g_texa;  // x: TA0, y: TA1, z: AEM, w: TFX | TCC << 4
    uint4  g_test;  // x: ATST (8 = off), y: AREF, z: 0 every fragment / 1 passing / 2 failing, w: FBA
    uint4  g_blend; // x: 1 = blend computed here (no Cd), y: A | B << 2 | C << 4 | D << 6, z: FIX
    float4 g_hd;    // base texel -> HD texture uv: xy scale, zw offset; kinds 6 / 7: x scale, y half of it
    uint4  g_fog;   // xyz: FOGCOL, w: fog on (triangles only, like the CPU GS)
};
cbuffer Palette : register(b1)
{
    uint4 g_pal[64];
};
Texture2D<uint> g_raw : register(t0);
Texture2D<float4> g_hdTex : register(t1); // the HD image, or the render target of kinds 6 / 7
SamplerState g_hdSampler : register(s0);

struct VSIn
{
    float3 pos : POSITION;
    float3 stq : TEXCOORD0;
    uint4 col : COLOR0;
    float fog : FOG;
};
struct PSIn
{
    float4 pos : SV_Position;
    noperspective float3 stq : TEXCOORD0;
    noperspective float4 col : COLOR0;
    noperspective float fog : FOG;
};
struct PSOut
{
    float4 c0 : SV_Target0;
    float4 c1 : SV_Target1;
};

PSIn VSMain(VSIn i)
{
    PSIn o;
    o.pos = float4(i.pos.x * g_ndc.x - 1.0, 1.0 - i.pos.y * g_ndc.y, i.pos.z, 1.0);
    o.stq = i.stq;
    o.col = float4(i.col);
    o.fog = i.fog;
    return o;
}

int wrapCoord(int c, int size, uint mode, uint mn, uint mx)
{
    if (mode == 0) return c & (size - 1);
    if (mode == 1) return clamp(c, 0, size - 1);
    if (mode == 2) return clamp(c, (int)mn, (int)mx);
    return (int)(((uint)c & mn) | mx);
}

uint4 unpackRgba(uint c)
{
    return uint4(c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, c >> 24);
}

uint4 texel(int2 c)
{
    int x = wrapCoord(c.x, (int)g_tex.y, g_wrap.x, g_wrap.z & 0xFFFF, g_wrap.z >> 16);
    int y = wrapCoord(c.y, (int)g_tex.z, g_wrap.y, g_wrap.w & 0xFFFF, g_wrap.w >> 16);
    uint raw;
    if (g_tex.x >= 6)
    {
        // The GS pixel's centre sample, as a download writes it to local memory.
        int s = (int)g_hd.x, h = (int)g_hd.y;
        uint4 b = (uint4)(g_hdTex.Load(int3(x * s + h, y * s + h, 0)) * 255.0 + 0.5);
        raw = b.r | (b.g << 8) | (b.b << 16) | (b.a << 24);
    }
    else
        raw = g_raw.Load(int3(x, y, 0));
    if (g_tex.x == 4)
    {
        uint i = raw & 255;
        return unpackRgba(g_pal[i >> 2][i & 3]);
    }
    uint4 t;
    if (g_tex.x == 3)
    {
        t = uint4((raw & 0x1F) << 3, ((raw >> 5) & 0x1F) << 3, ((raw >> 10) & 0x1F) << 3, 0);
        bool rgbZero = t.r == 0 && t.g == 0 && t.b == 0;
        t.a = (raw & 0x8000) ? g_texa.y : ((g_texa.z != 0 && rgbZero) ? 0 : g_texa.x);
        return t;
    }
    t = unpackRgba(raw);
    if (g_tex.x == 2 || g_tex.x == 7)
        t.a = (g_texa.z != 0 && t.r == 0 && t.g == 0 && t.b == 0) ? 0 : g_texa.x;
    return t;
}

uint4 sampleGs(float2 uv)
{
    if (g_tex.w == 0)
        return texel(int2((int)uv.x, (int)uv.y));
    float2 s = uv - 0.5;
    int2 c0 = (int2)floor(s);
    float2 f = s - (float2)c0;
    float4 c00 = (float4)texel(c0), c10 = (float4)texel(c0 + int2(1, 0));
    float4 c01 = (float4)texel(c0 + int2(0, 1)), c11 = (float4)texel(c0 + int2(1, 1));
    float4 top = c00 + (c10 - c00) * f.x;
    float4 bottom = c01 + (c11 - c01) * f.x;
    return (uint4)(top + (bottom - top) * f.y + 0.5);
}

uint4 combine(uint4 t, uint4 v)
{
    uint tfx = g_texa.w & 15;
    bool tcc = ((g_texa.w >> 4) & 1) != 0;
    uint4 o;
    if (tfx == 0)
    {
        o.rgb = min((t.rgb * v.rgb) >> 7, 255);
        o.a = tcc ? min((t.a * v.a) >> 7, 255) : v.a;
    }
    else if (tfx == 1)
    {
        o.rgb = t.rgb;
        o.a = tcc ? t.a : v.a;
    }
    else if (tfx == 2)
    {
        o.rgb = min(((t.rgb * v.rgb) >> 7) + v.a, 255);
        o.a = tcc ? min(t.a + v.a, 255) : v.a;
    }
    else
    {
        o.rgb = min(((t.rgb * v.rgb) >> 7) + v.a, 255);
        o.a = tcc ? t.a : v.a;
    }
    return o;
}

bool alphaPasses(uint a)
{
    switch (g_test.x)
    {
    case 0: return false;
    case 2: return a < g_test.y;
    case 3: return a <= g_test.y;
    case 4: return a == g_test.y;
    case 5: return a >= g_test.y;
    case 6: return a > g_test.y;
    case 7: return a != g_test.y;
    default: return true;
    }
}

PSOut PSMain(PSIn i)
{
    uint4 v = (uint4)clamp(i.col, 0.0, 255.0);
    uint4 c = v;
    float invQ = 1.0 / (abs(i.stq.z) > 1.0e-5 ? abs(i.stq.z) : 1.0);
    float2 uv = i.stq.xy * invQ * float2(g_tex.yz);
    // Gradients outside the branch: the HD path samples with mips.
    float2 hdUv = uv * g_hd.xy + g_hd.zw;
    float2 hdDx = ddx(hdUv), hdDy = ddy(hdUv);
    if (g_tex.x == 5)
        c = combine((uint4)(g_hdTex.SampleGrad(g_hdSampler, hdUv, hdDx, hdDy) * 255.0 + 0.5), v);
    else if (g_tex.x != 0)
        c = combine(sampleGs(uv), v);

    if (g_fog.w != 0)
    {
        uint f = (uint)clamp(i.fog + 0.5, 0.0, 255.0);
        c.rgb = (f * c.rgb + (255 - f) * g_fog.xyz) >> 8;
    }

    bool passed = alphaPasses(c.a);
    if ((g_test.z == 1 && !passed) || (g_test.z == 2 && passed))
        discard;

    uint aOut = c.a | (g_test.w != 0 ? 0x80 : 0);
    PSOut o;
    if (g_blend.x != 0)
    {
        // No destination term: the whole GS blend equation, exactly.
        uint sel = g_blend.y;
        int3 cs = (int3)c.rgb;
        int3 a = ((sel & 3) == 0) ? cs : int3(0, 0, 0);
        int3 b = (((sel >> 2) & 3) == 0) ? cs : int3(0, 0, 0);
        int f = (((sel >> 4) & 3) == 2) ? (int)g_blend.z : (int)c.a;
        int3 d = (((sel >> 6) & 3) == 0) ? cs : int3(0, 0, 0);
        c.rgb = (uint3)clamp(((a - b) * f >> 7) + d, 0, 255);
    }
    o.c0 = float4(c.rgb, aOut) / 255.0;
    o.c1 = float4(0.0, 0.0, 0.0, min(c.a, 128u) / 128.0);
    return o;
}

// Seeding a render target from local memory: full-screen triangle, point upscale.
Texture2D<float4> g_seedColor : register(t2);
Texture2D<float> g_seedDepth : register(t3);
cbuffer Seed : register(b2)
{
    uint4 g_seed; // x: scale
};
float4 VSFull(uint id : SV_VertexID) : SV_Position
{
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
float4 PSSeedColor(float4 pos : SV_Position) : SV_Target0
{
    return g_seedColor.Load(int3(int2(pos.xy) / (int)g_seed.x, 0));
}
float PSSeedDepth(float4 pos : SV_Position) : SV_Depth
{
    return g_seedDepth.Load(int3(int2(pos.xy) / (int)g_seed.x, 0));
}
)HLSL";

    struct Vertex
    {
        float x, y, z;
        float s, t, q;
        uint32_t rgba;
        float fog;
    };

    struct DrawConstants
    {
        float ndc[4];
        uint32_t tex[4];
        uint32_t wrap[4];
        uint32_t texa[4];
        uint32_t test[4];
        uint32_t blend[4];
        float hd[4];
        uint32_t fog[4];
    };

    struct Target
    {
        bool depth = false;
        uint32_t base = 0;  // FBP / ZBP (pages)
        uint32_t fbw = 0;
        uint8_t psm = 0;
        uint32_t w = 0, h = 0; // 1x pixels
        ComPtr<ID3D11Texture2D> tex;
        ComPtr<ID3D11RenderTargetView> rtv;
        ComPtr<ID3D11ShaderResourceView> srv; // colour targets: sampled as textures (renderTargetTextureLocked)
        ComPtr<ID3D11DepthStencilView> dsv;
        PageSet pages;
        bool gpuNewer = false;  // GPU draws local memory lacks
        bool vramNewer = true;  // local memory writes the GPU copy lacks
        char staleWhy[112] = "new target"; // the CPU-side write that last set vramNewer (stats logs)
    };

    // One decoded image of local memory, kept by its content (see textureFor).
    struct TexEntry
    {
        ComPtr<ID3D11Texture2D> tex;
        ComPtr<ID3D11ShaderResourceView> srv;
        uint32_t w = 0, h = 0;
        uint64_t lastUse = 0;
        std::vector<uint32_t> texels; // ATV_GS_GPU_TEXCHECK only
    };

    // A TEX0 (TBP0/TBW/PSM/TW/TH): its pages and the image they held at its last use.
    struct TexSlot
    {
        PageSet pages;
        uint64_t versionSum = 0;
        uint64_t content = 0; // m_textures key of that image; 0 = none yet
    };

    struct HdEntry
    {
        const void *texels = nullptr;
        uint32_t w = 0, h = 0, levels = 0;
        ComPtr<ID3D11ShaderResourceView> srv;
        uint64_t bytes = 0;
        uint64_t lastUse = 0;
    };

    struct Pass
    {
        uint32_t mode = 0;   // g_test.z
        uint8_t colorMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        bool depthWrite = false;
    };

    // D3D11 and the driver may change the calling thread's SSE control word (MXCSR). Backend calls
    // also come from the EE and VIF1/VU1 threads (GSThreadedBackend::syncCall runs readbacks and
    // snapshots on the caller), whose FPU/VU emulation relies on their FTZ|DAZ: every entry point
    // gives the caller its MXCSR back.
    struct FpGuard
    {
        unsigned int csr = _mm_getcsr();
        ~FpGuard()
        {
            const unsigned int now = _mm_getcsr();
            if (((now ^ csr) & ~0x3Fu) == 0u) // the exception flags (low 6 bits) may differ
                return;
            static std::atomic<uint32_t> s_logs{0};
            if (s_logs.fetch_add(1u) < 8u)
                std::fprintf(stderr, "[gs-gpu] MXCSR changed during a GS call (0x%04x -> 0x%04x), restored\n", csr, now);
            _mm_setcsr(csr);
        }
    };

    class GSGpuBackend final : public GSRasterBackend
    {
    public:
        explicit GSGpuBackend(uint32_t scale) : m_scale(scale)
        {
            // The keeper draws only what the GPU does not own (PSXWater's VIF stream, the race's
            // 128x128 shadow passes: a third of this thread's time when drawn inline), on the CPU
            // GS's band threads; this backend finishes its queued draws (cpuIdleLocked) before it
            // reads or writes local memory itself. ATV_GS_GPU_KEEPER_BANDS=n sets their count
            // (1 = inline, the old behaviour), else ATV_GS_BANDS / the CPU GS default.
            if (const char *keeperBands = std::getenv("ATV_GS_GPU_KEEPER_BANDS"))
            {
                char *old = nullptr;
                size_t oldLen = 0;
                _dupenv_s(&old, &oldLen, "ATV_GS_BANDS");
                _putenv_s("ATV_GS_BANDS", keeperBands);
                m_cpu = std::make_unique<GSCpuBackend>();
                _putenv_s("ATV_GS_BANDS", old ? old : "");
                std::free(old);
            }
            else
                m_cpu = std::make_unique<GSCpuBackend>();
            // ATV_GS_GPU_STATS=1: every 5 s; a value with a decimal point is the interval in seconds.
            if (const char *stats = std::getenv("ATV_GS_GPU_STATS"))
            {
                m_stats = true;
                if (std::strchr(stats, '.'))
                    m_statInterval = std::max(0.01, std::atof(stats));
            }
            m_noCache = std::getenv("ATV_GS_GPU_NOCACHE") != nullptr;
            m_allCpu = std::getenv("ATV_GS_GPU_ALLCPU") != nullptr;
            if (const char *trace = std::getenv("ATV_GS_GPU_TRACE"))
                m_traceFbp = std::atoi(trace);
            if (const char *probe = std::getenv("ATV_GS_GPU_PROBE"))
                std::sscanf(probe, "%llu,%llu", &m_probeFrom, &m_probeTo);
            if (const char *path = std::getenv("ATV_GS_GPU_CAPTURE"))
                m_capEnabled = *path != '\0';
            if (const char *skip = std::getenv("ATV_GS_GPU_CAPTURE_SKIP"))
                m_capSkip = std::strtoull(skip, nullptr, 10);
            if (const char *after = std::getenv("ATV_GS_GPU_CAPTURE_AFTER_FALLBACKS"))
                m_capAfterFallbacks = std::strtoull(after, nullptr, 10);
            if (const char *count = std::getenv("ATV_GS_GPU_CAPTURE_COMMANDS"))
                m_capRemaining = std::max<uint64_t>(1u, std::strtoull(count, nullptr, 10));
            m_ok = createDevice();
        }

        bool ok() const { return m_ok; }

        void Initialize(uint8_t *vram, uint32_t vramSize) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            dropAllLocked();
            m_vram = vram;
            m_cpu->Initialize(vram, vramSize);
            resetClutLocked();
        }

        void Reset() override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            m_cpu->Reset();
            dropAllLocked();
            resetClutLocked();
        }

        void Submit(const GSPrimitiveBatch &batch) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_capEnabled)
            {
                if (g_raceLoadStarted && !m_capFile && !m_capDone)
                    ++m_capRacePrims;
                captureWriteLocked(GSCaptureSubmit, &batch, sizeof(batch));
            }
            submitLocked(batch);
        }

        // Consecutive primitives from the GS worker under one lock (gsGpuSubmitMany).
        void SubmitMany(const GSPrimitiveBatch *const *batches, size_t count)
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            for (size_t i = 0; i < count; ++i)
            {
                if (m_capEnabled)
                {
                    if (g_raceLoadStarted && !m_capFile && !m_capDone)
                        ++m_capRacePrims;
                    captureWriteLocked(GSCaptureSubmit, batches[i], sizeof(GSPrimitiveBatch));
                }
                submitLocked(*batches[i]);
            }
        }

        void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_capFile)
            {
                struct { GSTex0Reg tex0; GSTexClutReg texclut; } record{tex0, texclut};
                captureWriteLocked(GSCaptureLoadClut, &record, sizeof(record));
            }
            if (!m_vram || (!isFourBit(tex0.psm) && !isEightBit(tex0.psm)))
                return;
            switch (tex0.cld)
            {
            case 1u: break;
            case 2u: m_clutCbp[0] = tex0.cbp; break;
            case 3u: m_clutCbp[1] = tex0.cbp; break;
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
            PageSet src;
            addPages(src, tex0.cbp, std::max<uint32_t>(1u, texclut.cbw), tex0.cpsm, 0u, 15u, tex0.csm ? UINT32_MAX : 15u);
            if (m_stats)
                std::snprintf(m_why, sizeof(m_why), "clut cbp=0x%x cpsm=0x%x", tex0.cbp, tex0.cpsm);
            syncForCpuLocked(src, false);
            m_cpu->LoadClut(tex0, texclut);
            cpuIdleLocked(&src);
            const std::array<uint16_t, 512> before = m_clut;
            loadClutLocked(tex0, texclut);
            ++m_statClutLoads;
            if (m_clut != before || m_noCache)
            {
                ++m_statClutChanges;
                flushBatchLocked();
                ++m_clutVersion;
            }
        }

        void BeginTransfer(const GSTransferCommand &command) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            if (m_capEnabled)
            {
                captureStartLocked();
                captureWriteLocked(GSCaptureBeginTransfer, &command, sizeof(command));
            }
            m_transfer = command;
            const GSBitBltBuf &b = command.bitbltbuf;
            if (m_stats)
                std::snprintf(m_why, sizeof(m_why), "transfer dir=%u sbp=0x%x dbp=0x%x dbw=%u dpsm=0x%x ds=(%u,%u) %ux%u", command.direction, b.sbp, b.dbp,
                              b.dbw, b.dpsm, command.trxpos.dsax, command.trxpos.dsay, command.trxreg.rrw, command.trxreg.rrh);
            m_transferDst.reset();
            if (command.direction == 1u || command.direction == 2u)
            {
                PageSet src;
                addPages(src, b.sbp, b.sbw, b.spsm, command.trxpos.ssay,
                         command.trxpos.ssay + std::max<uint32_t>(1u, command.trxreg.rrh) - 1u,
                         command.trxpos.ssax + std::max<uint32_t>(1u, command.trxreg.rrw) - 1u);
                if (command.direction == 1u)
                    noteReadbackLocked(src);
                syncForCpuLocked(src, false);
            }
            if (command.direction == 0u || command.direction == 2u)
            {
                addPages(m_transferDst, b.dbp, b.dbw, b.dpsm, command.trxpos.dsay,
                         command.trxpos.dsay + std::max<uint32_t>(1u, command.trxreg.rrh) - 1u,
                         command.trxpos.dsax + std::max<uint32_t>(1u, command.trxreg.rrw) - 1u);
                // Partly covered GPU targets must be complete in local memory before it changes.
                syncForCpuLocked(m_transferDst, true);
            }
            m_cpu->BeginTransfer(command);
            if (command.direction == 2u)
                noteCpuWriteLocked(m_transferDst);
            if (m_capFile && command.direction == 1u && m_capVramDumps < 4u)
            {
                // The live local memory at the first readbacks: tag 12, the pages a GPU target holds
                // newer data for (64 bytes, bit per page), then all 4 MB.
                ++m_capVramDumps;
                uint8_t stale[kPages / 8u] = {};
                for (const auto &t : m_targets)
                    if (t->gpuNewer)
                        for (uint32_t p = 0; p < kPages; ++p)
                            if (t->pages.test(p))
                                stale[p / 8u] |= static_cast<uint8_t>(1u << (p % 8u));
                cpuIdleLocked();
                captureLiveLocked(GSCaptureSnapshot, stale, sizeof(stale), m_vram, GSMem::MEMORY_SIZE);
            }
        }

        void UploadImage(const uint8_t *data, uint32_t sizeBytes) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            if (m_capFile && data)
                captureWriteLocked(GSCaptureUploadImage, &sizeBytes, sizeof(sizeBytes), data, sizeBytes);
            if (m_transfer.direction == 0u)
                syncForCpuLocked(m_transferDst, true);
            m_cpu->UploadImage(data, sizeBytes);
            ++m_statUploads;
            m_statUploadBytes += sizeBytes;
            if (m_transfer.direction == 0u)
                noteCpuWriteLocked(m_transferDst);
        }

        void Flush() override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            captureWriteLocked(GSCaptureFlush, nullptr, 0u);
            m_cpu->Flush();
        }

        void TextureFlush() override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            captureWriteLocked(GSCaptureTextureFlush, nullptr, 0u);
            m_cpu->TextureFlush();
        }

        void Sync(GSSyncReason reason) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            captureWriteLocked(GSCaptureSync, &reason, sizeof(reason));
            m_cpu->Sync(reason);
        }

        PresentationFrame Present(const GSPresentationRequest &request) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            captureWriteLocked(GSCaptureSnapshot, nullptr, 0u);
            ++m_presents;
            PresentationFrame frame;
            if (presentFromGpuLocked(request, frame))
                return frame;
            syncAllLocked();
            return m_cpu->Present(request);
        }

        bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            if (m_capFile)
            {
                struct { GSContext context; uint32_t rgba; } record{context, rgba};
                captureWriteLocked(GSCaptureClear, &record, sizeof(record));
            }
            if (m_stats)
                std::snprintf(m_why, sizeof(m_why), "clear fbp=%u fbw=%u", context.frame.fbp, context.frame.fbw);
            PageSet pages;
            addPages(pages, context.frame.fbp << 5, context.frame.fbw, context.frame.psm, context.scissor.y0, context.scissor.y1);
            syncForCpuLocked(pages, true);
            const bool cleared = m_cpu->ClearFramebuffer(context, rgba);
            noteCpuWriteLocked(pages);
            return cleared;
        }

        uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            const uint32_t copied = m_cpu->ConsumeLocalToHostBytes(dst, maxBytes);
            if (m_capFile)
            {
                captureLiveLocked(GSCaptureConsume, &copied, sizeof(copied), dst, copied);
                captureWriteLocked(GSCaptureConsume, &maxBytes, sizeof(maxBytes));
            }
            return copied;
        }

        uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            auto *self = const_cast<GSGpuBackend *>(this);
            self->flushBatchLocked();
            PageSet pages;
            addPages(pages, base, bw, static_cast<uint8_t>(psm), y, y);
            self->syncForCpuLocked(pages, false);
            const uint32_t value = m_cpu->ReadVram(psm, base, bw, x, y);
            if (m_capFile)
            {
                const uint32_t record[5] = {psm, base, bw, x, y};
                self->captureLiveLocked(GSCaptureReadVram, &value, sizeof(value));
                self->captureWriteLocked(GSCaptureReadVram, record, sizeof(record));
            }
            return value;
        }

        void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            if (m_capFile)
            {
                const uint32_t record[6] = {psm, base, bw, x, y, value};
                captureWriteLocked(GSCaptureWriteVram, record, sizeof(record));
            }
            if (m_stats)
                std::snprintf(m_why, sizeof(m_why), "writevram base=0x%x psm=0x%x (%u,%u)", base, psm, x, y);
            PageSet pages;
            addPages(pages, base, bw, static_cast<uint8_t>(psm), y, y);
            syncForCpuLocked(pages, true);
            m_cpu->WriteVram(psm, base, bw, x, y, value);
            noteCpuWriteLocked(pages);
        }

        void SnapshotVram(std::vector<uint8_t> &out) const override
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            auto *self = const_cast<GSGpuBackend *>(this);
            self->flushBatchLocked();
            self->captureWriteLocked(GSCaptureSnapshot, nullptr, 0u);
            self->syncAllLocked();
            m_cpu->SnapshotVram(out);
        }

        GSTransferSnapshot GetTransferSnapshot() const override { return m_cpu->GetTransferSnapshot(); }

        void RestoreClutState(const std::array<uint16_t, 512> &clut, const std::array<uint32_t, 2> &cbp)
        {
            FpGuard fpGuard;
            std::lock_guard<std::mutex> lock(m_mutex);
            flushBatchLocked();
            m_cpu->RestoreClutState(clut, cbp);
            m_clut = clut;
            m_clutCbp = cbp;
            ++m_clutVersion;
        }

    private:
        // --------------------------------------------------------------- capture
        // ATV_GS_GPU_CAPTURE=<file>: record the commands this backend receives in the
        // ATV_GS_CAPTURE format (gsreplay, ps2_gs_replay_bench) from the first BeginTransfer after
        // ATV_GS_GPU_CAPTURE_SKIP race primitives (default 0), for ATV_GS_GPU_CAPTURE_COMMANDS
        // commands (default 2M). Local memory at the start is taken after every GPU target is
        // downloaded. <file>.live holds what this backend returned: per Consume tag 11 + u32 bytes +
        // the bytes, per ReadVram tag 10 + u32 value.
        void captureStartLocked()
        {
            if (m_capFile || m_capDone || !m_vram || !g_raceLoadStarted || m_capRacePrims < m_capSkip || m_capFallbacks < m_capAfterFallbacks)
                return;
            m_capDone = true;
            const char *path = std::getenv("ATV_GS_GPU_CAPTURE");
            m_capFile = std::fopen(path, "wb");
            m_capLive = std::fopen((std::string(path) + ".live").c_str(), "wb");
            if (!m_capFile || !m_capLive)
            {
                std::fprintf(stderr, "[gs-gpu] cannot open capture %s\n", path);
                if (m_capFile)
                    std::fclose(m_capFile);
                if (m_capLive)
                    std::fclose(m_capLive);
                m_capFile = m_capLive = nullptr;
                return;
            }
            syncAllLocked();
            cpuIdleLocked();
            const char magic[8] = {'G', 'S', 'C', 'A', 'P', '1', 0, 0};
            const uint32_t sizes[4] = {static_cast<uint32_t>(GSMem::MEMORY_SIZE), static_cast<uint32_t>(sizeof(GSPrimitiveBatch)),
                                       static_cast<uint32_t>(sizeof(GSContext)), 1u};
            std::fwrite(magic, 1u, sizeof(magic), m_capFile);
            std::fwrite(sizes, 1u, sizeof(sizes), m_capFile);
            std::fwrite(m_vram, 1u, GSMem::MEMORY_SIZE, m_capFile);
            std::fwrite(m_clut.data(), sizeof(m_clut[0]), m_clut.size(), m_capFile);
            std::fwrite(m_clutCbp.data(), sizeof(m_clutCbp[0]), m_clutCbp.size(), m_capFile);
            std::fprintf(stderr, "[gs-gpu] capture started after %llu race primitives, %llu commands\n",
                         static_cast<unsigned long long>(m_capRacePrims), static_cast<unsigned long long>(m_capRemaining));
        }

        void captureWriteLocked(uint8_t tag, const void *payload, size_t bytes, const void *tail = nullptr, size_t tailBytes = 0u)
        {
            if (!m_capFile)
                return;
            std::fwrite(&tag, 1u, 1u, m_capFile);
            if (bytes)
                std::fwrite(payload, 1u, bytes, m_capFile);
            if (tailBytes)
                std::fwrite(tail, 1u, tailBytes, m_capFile);
            if (--m_capRemaining == 0u)
            {
                const uint8_t end = GSCaptureEnd;
                std::fwrite(&end, 1u, 1u, m_capFile);
                std::fclose(m_capFile);
                std::fclose(m_capLive);
                m_capFile = m_capLive = nullptr;
                std::fprintf(stderr, "[gs-gpu] capture finished\n");
            }
        }

        void captureLiveLocked(uint8_t tag, const void *payload, size_t bytes, const void *tail = nullptr, size_t tailBytes = 0u)
        {
            std::fwrite(&tag, 1u, 1u, m_capLive);
            std::fwrite(payload, 1u, bytes, m_capLive);
            if (tailBytes)
                std::fwrite(tail, 1u, tailBytes, m_capLive);
        }

        // ---------------------------------------------------------------- device
        bool createDevice()
        {
            UINT flags = 0;
            if (std::getenv("ATV_GS_GPU_DEBUG"))
                flags |= D3D11_CREATE_DEVICE_DEBUG;
            const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
            D3D_FEATURE_LEVEL got{};
            HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels, 2, D3D11_SDK_VERSION,
                                           &m_dev, &got, &m_ctx);
            if (FAILED(hr))
                hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels + 1, 1, D3D11_SDK_VERSION,
                                       &m_dev, &got, &m_ctx);
            if (FAILED(hr))
            {
                std::fprintf(stderr, "[gs-gpu] D3D11CreateDevice failed 0x%08lx\n", static_cast<unsigned long>(hr));
                return false;
            }
            ComPtr<ID3DBlob> vs, ps, vsFull, psSeedColor, psSeedDepth;
            if (!compile("VSMain", "vs_5_0", vs) || !compile("PSMain", "ps_5_0", ps) || !compile("VSFull", "vs_5_0", vsFull) ||
                !compile("PSSeedColor", "ps_5_0", psSeedColor) || !compile("PSSeedDepth", "ps_5_0", psSeedDepth))
                return false;
            m_dev->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), nullptr, &m_vs);
            m_dev->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(), nullptr, &m_ps);
            m_dev->CreateVertexShader(vsFull->GetBufferPointer(), vsFull->GetBufferSize(), nullptr, &m_vsFull);
            m_dev->CreatePixelShader(psSeedColor->GetBufferPointer(), psSeedColor->GetBufferSize(), nullptr, &m_psSeedColor);
            m_dev->CreatePixelShader(psSeedDepth->GetBufferPointer(), psSeedDepth->GetBufferSize(), nullptr, &m_psSeedDepth);
            const D3D11_INPUT_ELEMENT_DESC layout[] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
                {"TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
                {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UINT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
                {"FOG", 0, DXGI_FORMAT_R32_FLOAT, 0, 28, D3D11_INPUT_PER_VERTEX_DATA, 0},
            };
            if (FAILED(m_dev->CreateInputLayout(layout, 4, vs->GetBufferPointer(), vs->GetBufferSize(), &m_layout)))
                return false;

            D3D11_BUFFER_DESC bd{};
            bd.ByteWidth = kVertexBufferBytes;
            bd.Usage = D3D11_USAGE_DYNAMIC;
            bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            m_dev->CreateBuffer(&bd, nullptr, &m_vb);
            bd.ByteWidth = sizeof(DrawConstants);
            bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            m_dev->CreateBuffer(&bd, nullptr, &m_cbDraw);
            bd.ByteWidth = 1024;
            m_dev->CreateBuffer(&bd, nullptr, &m_cbPalette);
            // Written with UpdateSubresource, which a DYNAMIC buffer ignores (the seed scale stayed 0,
            // so every seed loaded out of range and wrote black / Z 0).
            bd.ByteWidth = 16;
            bd.Usage = D3D11_USAGE_DEFAULT;
            bd.CPUAccessFlags = 0;
            m_dev->CreateBuffer(&bd, nullptr, &m_cbSeed);

            D3D11_RASTERIZER_DESC rd{};
            rd.FillMode = D3D11_FILL_SOLID;
            rd.CullMode = D3D11_CULL_NONE;
            rd.DepthClipEnable = FALSE;
            rd.ScissorEnable = TRUE;
            m_dev->CreateRasterizerState(&rd, &m_raster);
            rd.ScissorEnable = FALSE;
            m_dev->CreateRasterizerState(&rd, &m_rasterNoScissor);

            for (int i = 0; i < 4; ++i)
            {
                D3D11_SAMPLER_DESC sd{};
                sd.Filter = D3D11_FILTER_ANISOTROPIC;
                sd.MaxAnisotropy = 8;
                sd.AddressU = (i & 1) ? D3D11_TEXTURE_ADDRESS_CLAMP : D3D11_TEXTURE_ADDRESS_WRAP;
                sd.AddressV = (i & 2) ? D3D11_TEXTURE_ADDRESS_CLAMP : D3D11_TEXTURE_ADDRESS_WRAP;
                sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
                sd.MaxLOD = D3D11_FLOAT32_MAX;
                m_dev->CreateSamplerState(&sd, &m_hdSamplers[i]);
            }
            std::fprintf(stderr, "[gs-gpu] D3D11 hardware GS ready, render targets at %ux\n", m_scale);
            return m_vs && m_ps && m_vb && m_cbDraw && m_raster;
        }

        bool compile(const char *entry, const char *profile, ComPtr<ID3DBlob> &out)
        {
            ComPtr<ID3DBlob> errors;
            const HRESULT hr = D3DCompile(kShaderSource, std::strlen(kShaderSource), "gs_gpu_backend", nullptr, nullptr, entry, profile,
                                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &out, &errors);
            if (FAILED(hr))
            {
                std::fprintf(stderr, "[gs-gpu] shader %s failed: %s\n", entry,
                             errors ? static_cast<const char *>(errors->GetBufferPointer()) : "?");
                return false;
            }
            return true;
        }

        ID3D11BlendState *blendState(bool enable, D3D11_BLEND src, D3D11_BLEND dst, D3D11_BLEND_OP op, uint8_t mask)
        {
            const uint32_t key = (enable ? 1u : 0u) | (static_cast<uint32_t>(src) << 1) | (static_cast<uint32_t>(dst) << 6) |
                                 (static_cast<uint32_t>(op) << 11) | (static_cast<uint32_t>(mask) << 14);
            auto &slot = m_blendStates[key];
            if (!slot)
            {
                D3D11_BLEND_DESC d{};
                auto &rt = d.RenderTarget[0];
                rt.BlendEnable = enable;
                rt.SrcBlend = src;
                rt.DestBlend = dst;
                rt.BlendOp = op;
                rt.SrcBlendAlpha = D3D11_BLEND_ONE;
                rt.DestBlendAlpha = D3D11_BLEND_ZERO;
                rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
                rt.RenderTargetWriteMask = mask;
                m_dev->CreateBlendState(&d, &slot);
            }
            return slot.Get();
        }

        ID3D11DepthStencilState *depthState(bool enable, D3D11_COMPARISON_FUNC func, bool write)
        {
            const uint32_t key = (enable ? 1u : 0u) | (static_cast<uint32_t>(func) << 1) | (write ? 64u : 0u);
            auto &slot = m_depthStates[key];
            if (!slot)
            {
                D3D11_DEPTH_STENCIL_DESC d{};
                d.DepthEnable = enable;
                d.DepthFunc = func;
                d.DepthWriteMask = write ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
                m_dev->CreateDepthStencilState(&d, &slot);
            }
            return slot.Get();
        }

        // --------------------------------------------------------------- targets
        // Like the CPU GS: Z is read for ZTST GEQUAL/GREATER (and NEVER needs the depth test), and
        // written whenever ZMSK is off, ZTE off included (then the test always passes).
        static bool readsDepth(const GSContext &ctx)
        {
            const uint32_t ztst = static_cast<uint32_t>((ctx.test >> 17) & 3u);
            return ((ctx.test >> 16) & 1u) != 0u && ztst != 1u;
        }

        static bool writesDepth(const GSContext &ctx)
        {
            const bool never = ((ctx.test >> 16) & 1u) != 0u && ((ctx.test >> 17) & 3u) == 0u;
            return !ctx.zbuf.zmask && !never;
        }

        static bool usesDepth(const GSContext &ctx) { return readsDepth(ctx) || writesDepth(ctx); }

        // Full-size colour targets of at most 512 lines whose pages (and their Z buffer's) end inside
        // local memory. Bigger ones (the boot clear of all 4 MB as one 1024x1024 target) are drawn by
        // the CPU GS: a GPU target that wraps around would claim, and on download overwrite, the
        // texture and CLUT pages.
        bool gpuOwnsFrameLocked(const GSContext &ctx)
        {
            if ((ctx.frame.psm != GS_PSM_CT32 && ctx.frame.psm != GS_PSM_CT24) || ctx.frame.fbw == 0u || ctx.scissor.y1 >= 512u)
                return false;
            if (ctx.frame.fbw < 8u && !smallTargetOnGpuLocked(ctx))
                return false;
            const uint32_t rows = static_cast<uint32_t>(ctx.scissor.y1) + 1u;
            if (ctx.frame.fbp + (rows + 31u) / 32u * ctx.frame.fbw > kPages)
                return false;
            if (usesDepth(ctx))
            {
                const uint32_t zRows = isSixteenBit(ctx.zbuf.psm) ? 64u : 32u;
                if (ctx.zbuf.zbp + (rows + zRows - 1u) / zRows * ctx.frame.fbw > kPages)
                    return false;
            }
            return true;
        }

        // Buffers narrower than 512 pixels (FBW < 8) are drawn on the GPU only once they have been
        // drawn for kSmallProbation presents without the host reading back their pages or those
        // of a Z buffer drawn with them: PSXWater draws the VIF stream it reads back as the Z of
        // FBP 280 FBW 6 (ZBP 352), and that has to stay exact; the race's 128x128 shadow pair
        // (FBP 420/428) is only sampled as a texture. ATV_GS_GPU_SMALL=0: never.
        bool smallTargetOnGpuLocked(const GSContext &ctx)
        {
            if (!m_smallTargets)
                return false;
            const uint32_t key = ctx.frame.fbp | (ctx.frame.fbw << 9) | (static_cast<uint32_t>(ctx.frame.psm) << 15);
            const auto [it, fresh] = m_small.try_emplace(key);
            SmallTarget &s = it->second;
            bool grew = false;
            if (fresh)
            {
                s.since = m_presents;
                addPages(s.pages, ctx.frame.fbp << 5, ctx.frame.fbw, ctx.frame.psm, 0u, ctx.scissor.y1);
                grew = true;
            }
            if (usesDepth(ctx) && s.zbpSeen != ctx.zbuf.zbp + 1u)
            {
                s.zbpSeen = ctx.zbuf.zbp + 1u;
                addPages(s.pages, ctx.zbuf.zbp << 5, ctx.frame.fbw, ctx.zbuf.psm, 0u, ctx.scissor.y1);
                grew = true;
            }
            if (grew)
                s.readBack = s.readBack || (s.pages & m_readbackPages).any();
            return !s.readBack && m_presents - s.since >= kSmallProbation;
        }

        // A local -> host transfer reads these pages: small buffers there stay on the CPU.
        void noteReadbackLocked(const PageSet &pages)
        {
            m_readbackPages |= pages;
            for (auto &[key, s] : m_small)
                if ((s.pages & pages).any())
                    s.readBack = true;
        }

        // A CT32 / CT24 texture that is the top-left part of a colour target the GPU holds current
        // is sampled from that target, without a download and decode (the shader reads each GS
        // pixel's centre, as a download does). ATV_GS_GPU_RT_TEX=0: always through local memory.
        Target *renderTargetTextureLocked(const GSTex0Reg &tex, uint32_t w, uint32_t h)
        {
            if (!m_rtTextures || (tex.psm != GS_PSM_CT32 && tex.psm != GS_PSM_CT24) || (tex.tbp0 & 31u) != 0u)
                return nullptr;
            Target *t = findTarget(false, tex.tbp0 >> 5);
            if (!t || t == m_color || t->vramNewer || !t->srv || t->fbw != tex.tbw || w > t->w || h > t->h ||
                (t->psm != GS_PSM_CT32 && tex.psm == GS_PSM_CT32))
                return nullptr;
            return t;
        }

        Target *findTarget(bool depth, uint32_t base)
        {
            for (auto &t : m_targets)
                if (t->depth == depth && t->base == base)
                    return t.get();
            return nullptr;
        }

        void dropTargetLocked(Target *target)
        {
            traceLocked(*target, "drop");
            if (target->gpuNewer)
                downloadLocked(*target);
            m_targets.erase(std::remove_if(m_targets.begin(), m_targets.end(), [&](const auto &t) { return t.get() == target; }),
                            m_targets.end());
        }

        Target *acquireTargetLocked(bool depth, uint32_t base, uint32_t fbw, uint8_t psm, uint32_t rows)
        {
            uint32_t pw, ph;
            pageDims(psm, pw, ph);
            uint32_t h = std::min<uint32_t>(2048u, (rows + ph - 1u) / ph * ph);
            // A scissor a few rows past a page row (the race's 128x128 shadow pass at FBP 420 has
            // 129 rows) would make this target own the first pages of the buffer after it (FBP 428),
            // and the two would drop each other every frame: stop at the page row before it.
            if (rows % ph != 0u && h > ph)
            {
                PageSet tail;
                addPages(tail, base << 5, fbw, psm, h - ph, h - 1u);
                for (const auto &other : m_targets)
                    if (other->base != base && (other->pages & tail).any())
                    {
                        h -= ph;
                        break;
                    }
            }
            Target *t = findTarget(depth, base);
            if (t && t->fbw == fbw && t->h >= h && (depth || t->psm == psm))
            {
                t->psm = psm;
                return t;
            }
            if (t)
                dropTargetLocked(t);
            auto fresh = std::make_unique<Target>();
            fresh->depth = depth;
            fresh->base = base;
            fresh->fbw = fbw;
            fresh->psm = psm;
            fresh->w = fbw * 64u;
            fresh->h = h;
            addPages(fresh->pages, base << 5, fbw, psm, 0u, h - 1u);
            // A new target takes its pages from any other.
            for (size_t i = 0; i < m_targets.size();)
            {
                if ((m_targets[i]->pages & fresh->pages).any())
                    dropTargetLocked(m_targets[i].get());
                else
                    ++i;
            }
            D3D11_TEXTURE2D_DESC td{};
            td.Width = fresh->w * m_scale;
            td.Height = fresh->h * m_scale;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            if (depth)
            {
                td.Format = DXGI_FORMAT_R32_TYPELESS;
                td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
                m_dev->CreateTexture2D(&td, nullptr, &fresh->tex);
                D3D11_DEPTH_STENCIL_VIEW_DESC dd{};
                dd.Format = DXGI_FORMAT_D32_FLOAT;
                dd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
                m_dev->CreateDepthStencilView(fresh->tex.Get(), &dd, &fresh->dsv);
            }
            else
            {
                td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                m_dev->CreateTexture2D(&td, nullptr, &fresh->tex);
                m_dev->CreateRenderTargetView(fresh->tex.Get(), nullptr, &fresh->rtv);
                m_dev->CreateShaderResourceView(fresh->tex.Get(), nullptr, &fresh->srv);
            }
            fresh->vramNewer = true;
            traceLocked(*fresh, "create");
            ++m_statTargets;
            m_targets.push_back(std::move(fresh));
            return m_targets.back().get();
        }

        void dropAllLocked()
        {
            m_targets.clear();
            m_texSlots.clear();
            m_textures.clear();
            m_texBytes = 0;
            m_small.clear();
            m_readbackPages.reset();
            m_hd.clear();
            m_pageVersion.fill(0u);
        }

        // Before the CPU side reads (or partly overwrites) these pages: GPU draws go to local memory.
        // The keeper's band threads may still draw queued primitives: finish them before this
        // backend reads or writes local memory those primitives touch (all of it without `pages`).
        // The keeper's own entry points wait for its bands where they need to.
        void cpuIdleLocked(const PageSet *pages = nullptr)
        {
            if (!m_cpuQueued || (pages && !(*pages & m_cpuQueuedPages).any()))
                return;
            const auto start = std::chrono::steady_clock::now();
            m_cpu->Flush();
            m_cpuQueued = false;
            m_cpuQueuedPages.reset();
            m_statCpuWaitNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        }

        void syncForCpuLocked(const PageSet &pages, bool writing)
        {
            for (auto &t : m_targets)
            {
                if (!(t->pages & pages).any())
                    continue;
                if (t->gpuNewer)
                {
                    m_statDownloadPagesNeeded += (t->pages & pages).count();
                    m_statDownloadPagesTotal += t->pages.count();
                    if (m_stats)
                    {
                        // Counted by target and the operation that asked (its first two words).
                        const char *end = std::strchr(m_why, ' ');
                        end = end ? std::strchr(end + 1, ' ') : nullptr;
                        char key[96];
                        std::snprintf(key, sizeof(key), "%s %u <- %.*s", t->depth ? "Z" : "C", t->base,
                                      static_cast<int>(end ? end - m_why : std::strlen(m_why)), m_why);
                        ++m_downloadWhy[key];
                    }
                    downloadLocked(*t);
                    m_statDepthDownloads += t->depth ? 1u : 0u;
                    t->gpuNewer = false;
                }
                if (writing)
                    markStaleLocked(*t);
            }
        }

        // ATV_GS_GPU_TRACE=<fbp>: log what happens to the colour target at that FBP (debug).
        void traceLocked(const Target &t, const char *what, const char *detail = "")
        {
            if (m_traceFbp < 0 || t.depth || t.base != static_cast<uint32_t>(m_traceFbp))
                return;
            std::fprintf(stderr, "[gs-gpu-trace] prim %llu %s fbp=%u gpuNewer=%d vramNewer=%d %s\n", static_cast<unsigned long long>(m_statPrimsTotal),
                         what, t.base, t.gpuNewer ? 1 : 0, t.vramNewer ? 1 : 0, detail);
        }

        // ATV_GS_GPU_PROBE=a,b (with ATV_GS_GPU_TRACE): mean colour of the traced target after every
        // batch flushed between primitives a and b, with the batch's state (debug).
        void probeLocked(Target &t)
        {
            ID3D11Texture2D *staging = stagingLocked(m_stagingDownload, t.w * m_scale, t.h * m_scale, DXGI_FORMAT_R8G8B8A8_UNORM);
            m_ctx->CopyResource(staging, t.tex.Get());
            D3D11_MAPPED_SUBRESOURCE map{};
            if (FAILED(m_ctx->Map(staging, 0, D3D11_MAP_READ, 0, &map)))
                return;
            uint64_t sum = 0;
            for (uint32_t y = 0; y < t.h * m_scale; ++y)
            {
                const uint8_t *row = static_cast<const uint8_t *>(map.pData) + static_cast<size_t>(y) * map.RowPitch;
                for (uint32_t x = 0; x < t.w * m_scale; ++x)
                    sum += row[x * 4u] + row[x * 4u + 1u] + row[x * 4u + 2u];
            }
            m_ctx->Unmap(staging, 0);
            const GSContext &c = m_batchState.context;
            char detail[200];
            std::snprintf(detail, sizeof(detail), "mean rgb %.1f after %zu verts type=%u tex=%x/%u/%x tme=%u abe=%u alpha=%llx test=%llx fbmsk=%x zbp=%u",
                          static_cast<double>(sum) / (3.0 * t.w * t.h * m_scale * m_scale), m_vertices.size(), static_cast<uint32_t>(m_batchState.prim.type),
                          c.tex0.tbp0, c.tex0.tbw, c.tex0.psm, static_cast<uint32_t>(m_batchState.prim.tme), static_cast<uint32_t>(m_batchState.prim.abe),
                          static_cast<unsigned long long>(c.alpha), static_cast<unsigned long long>(c.test), c.frame.fbmsk, c.zbuf.zbp);
            traceLocked(t, "probe", detail);
        }

        void markStaleLocked(Target &t)
        {
            if (!t.vramNewer)
                traceLocked(t, "stale", m_why);
            t.vramNewer = true;
            if (m_stats)
                std::snprintf(t.staleWhy, sizeof(t.staleWhy), "%s", m_why);
        }

        void noteCpuWriteLocked(const PageSet &pages)
        {
            for (uint32_t p = 0; p < kPages; ++p)
                if (pages.test(p))
                    ++m_pageVersion[p];
            for (auto &t : m_targets)
                if ((t->pages & pages).any())
                    markStaleLocked(*t);
        }

        void syncAllLocked()
        {
            for (auto &t : m_targets)
                if (t->gpuNewer)
                {
                    downloadLocked(*t);
                    t->gpuNewer = false;
                }
        }

        ID3D11Texture2D *stagingLocked(ComPtr<ID3D11Texture2D> &slot, uint32_t w, uint32_t h, DXGI_FORMAT format)
        {
            D3D11_TEXTURE2D_DESC have{};
            if (slot)
                slot->GetDesc(&have);
            if (!slot || have.Width != w || have.Height != h || have.Format != format)
            {
                slot.Reset();
                D3D11_TEXTURE2D_DESC td{};
                td.Width = w;
                td.Height = h;
                td.MipLevels = 1;
                td.ArraySize = 1;
                td.Format = format;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_STAGING;
                td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                m_dev->CreateTexture2D(&td, nullptr, &slot);
            }
            return slot.Get();
        }

        // GPU target -> local memory (point sample at each GS pixel's centre). The race needs depth
        // too: its per-frame side pass copies (and the game reads back) part of the main Z buffer.
        void downloadLocked(Target &t)
        {
            cpuIdleLocked(&t.pages);
            const auto start = std::chrono::steady_clock::now();
            ID3D11Texture2D *staging = t.depth ? stagingLocked(m_stagingDepth, t.w * m_scale, t.h * m_scale, DXGI_FORMAT_R32_TYPELESS)
                                               : stagingLocked(m_stagingDownload, t.w * m_scale, t.h * m_scale, DXGI_FORMAT_R8G8B8A8_UNORM);
            m_ctx->CopyResource(staging, t.tex.Get());
            D3D11_MAPPED_SUBRESOURCE map{};
            if (FAILED(m_ctx->Map(staging, 0, D3D11_MAP_READ, 0, &map)))
                return;
            m_statDownloadWaitNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            const uint32_t half = m_scale / 2u;
            const uint32_t base = t.base << 5;
            const double toZ = 1.0 / zScale(t.psm);
            for (uint32_t y = 0; y < t.h; ++y)
            {
                const uint8_t *row = static_cast<const uint8_t *>(map.pData) + static_cast<size_t>(y * m_scale + half) * map.RowPitch;
                for (uint32_t x = 0; x < t.w; ++x)
                {
                    uint32_t c;
                    std::memcpy(&c, row + static_cast<size_t>(x * m_scale + half) * 4u, 4u);
                    if (t.depth)
                    {
                        float f;
                        std::memcpy(&f, &c, 4u);
                        const uint32_t z = static_cast<uint32_t>(std::min(static_cast<double>(f) * toZ + 0.5, 4294967295.0));
                        if (t.psm == GS_PSM_Z32)
                            GSMem::WriteZ32(m_vram, base, t.fbw, x, y, z);
                        else if (t.psm == GS_PSM_Z16 || t.psm == GS_PSM_Z16S)
                            GSMem::WriteZ16(m_vram, base, t.fbw, x, y, std::min(z, 0xFFFFu));
                        else
                            GSMem::WriteZ24(m_vram, base, t.fbw, x, y, std::min(z, 0xFFFFFFu));
                    }
                    else if (t.psm == GS_PSM_CT24)
                        GSMem::WriteCT24(m_vram, base, t.fbw, x, y, c);
                    else
                        GSMem::WriteCT32(m_vram, base, t.fbw, x, y, c);
                }
            }
            m_ctx->Unmap(staging, 0);
            if (m_traceFbp >= 0 && !t.depth && t.base == static_cast<uint32_t>(m_traceFbp))
            {
                uint64_t sum = 0;
                for (uint32_t y = 0; y < t.h; ++y)
                    for (uint32_t x = 0; x < t.w; ++x)
                    {
                        const uint32_t c = GSMem::ReadCT32(m_vram, base, t.fbw, x, y);
                        sum += (c & 0xFFu) + ((c >> 8) & 0xFFu) + ((c >> 16) & 0xFFu);
                    }
                char detail[64];
                std::snprintf(detail, sizeof(detail), "mean rgb %.1f", static_cast<double>(sum) / (3.0 * t.w * t.h));
                traceLocked(t, "download", detail);
            }
            for (uint32_t p = 0; p < kPages; ++p)
                if (t.pages.test(p))
                    ++m_pageVersion[p];
            m_cpu->TextureFlush();
            ++m_statDownloads;
            m_statDownloadNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        }

        // Local memory -> GPU target (point upscale).
        void seedLocked(Target &t)
        {
            cpuIdleLocked(&t.pages);
            const auto start = std::chrono::steady_clock::now();
            if (m_stats && !t.depth && m_colorSeedLogs < 40u)
            {
                ++m_colorSeedLogs;
                std::fprintf(stderr, "[gs-gpu] seed colour target fbp=%u fbw=%u %ux%u\n", t.base, t.fbw, t.w, t.h);
            }
            m_seedPixels.resize(static_cast<size_t>(t.w) * t.h);
            const uint32_t base = t.base << 5;
            D3D11_TEXTURE2D_DESC td{};
            td.Width = t.w;
            td.Height = t.h;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            if (t.depth)
            {
                const float scale = zScale(t.psm);
                float *out = reinterpret_cast<float *>(m_seedPixels.data());
                for (uint32_t y = 0; y < t.h; ++y)
                    for (uint32_t x = 0; x < t.w; ++x)
                    {
                        uint32_t z;
                        if (t.psm == GS_PSM_Z32)
                            z = GSMem::ReadZ32(m_vram, base, t.fbw, x, y);
                        else if (t.psm == GS_PSM_Z16 || t.psm == GS_PSM_Z16S)
                            z = GSMem::ReadZ16(m_vram, base, t.fbw, x, y);
                        else
                            z = GSMem::ReadZ24(m_vram, base, t.fbw, x, y) & 0xFFFFFFu;
                        out[y * t.w + x] = std::min(static_cast<float>(z) * scale, 1.0f);
                    }
                td.Format = DXGI_FORMAT_R32_FLOAT;
            }
            else
            {
                for (uint32_t y = 0; y < t.h; ++y)
                    for (uint32_t x = 0; x < t.w; ++x)
                        m_seedPixels[y * t.w + x] = GSMem::ReadCT32(m_vram, base, t.fbw, x, y);
                td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                if (m_traceFbp >= 0)
                {
                    uint64_t sum = 0;
                    for (const uint32_t c : m_seedPixels)
                        sum += (c & 0xFFu) + ((c >> 8) & 0xFFu) + ((c >> 16) & 0xFFu);
                    char detail[64];
                    std::snprintf(detail, sizeof(detail), "mean rgb %.1f", static_cast<double>(sum) / (3.0 * t.w * t.h));
                    traceLocked(t, "seed", detail);
                }
            }
            D3D11_SUBRESOURCE_DATA init{m_seedPixels.data(), t.w * 4u, 0};
            ComPtr<ID3D11Texture2D> seedTex;
            ComPtr<ID3D11ShaderResourceView> seedSrv;
            const HRESULT texHr = m_dev->CreateTexture2D(&td, &init, &seedTex);
            const HRESULT srvHr = m_dev->CreateShaderResourceView(seedTex.Get(), nullptr, &seedSrv);
            if (m_traceFbp >= 0 && (FAILED(texHr) || FAILED(srvHr)))
                std::fprintf(stderr, "[gs-gpu-trace] seed texture failed: %08lx %08lx\n", static_cast<unsigned long>(texHr), static_cast<unsigned long>(srvHr));

            const uint32_t seedCb[4] = {m_scale, 0, 0, 0};
            m_ctx->UpdateSubresource(m_cbSeed.Get(), 0, nullptr, seedCb, 0, 0);
            D3D11_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(t.w * m_scale), static_cast<float>(t.h * m_scale), 0.0f, 1.0f};
            m_ctx->RSSetViewports(1, &vp);
            m_ctx->RSSetState(m_rasterNoScissor.Get());
            m_ctx->IASetInputLayout(nullptr);
            m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            m_ctx->VSSetShader(m_vsFull.Get(), nullptr, 0);
            m_ctx->PSSetConstantBuffers(2, 1, m_cbSeed.GetAddressOf());
            ID3D11ShaderResourceView *srv = seedSrv.Get();
            if (t.depth)
            {
                m_ctx->OMSetRenderTargets(0, nullptr, t.dsv.Get());
                m_ctx->OMSetBlendState(blendState(false, D3D11_BLEND_ONE, D3D11_BLEND_ZERO, D3D11_BLEND_OP_ADD, 0), nullptr, 0xFFFFFFFFu);
                m_ctx->OMSetDepthStencilState(depthState(true, D3D11_COMPARISON_ALWAYS, true), 0);
                m_ctx->PSSetShader(m_psSeedDepth.Get(), nullptr, 0);
                m_ctx->PSSetShaderResources(3, 1, &srv);
            }
            else
            {
                ID3D11RenderTargetView *rtv = t.rtv.Get();
                m_ctx->OMSetRenderTargets(1, &rtv, nullptr);
                m_ctx->OMSetBlendState(blendState(false, D3D11_BLEND_ONE, D3D11_BLEND_ZERO, D3D11_BLEND_OP_ADD,
                                                  D3D11_COLOR_WRITE_ENABLE_ALL),
                                       nullptr, 0xFFFFFFFFu);
                m_ctx->OMSetDepthStencilState(depthState(false, D3D11_COMPARISON_ALWAYS, false), 0);
                m_ctx->PSSetShader(m_psSeedColor.Get(), nullptr, 0);
                m_ctx->PSSetShaderResources(2, 1, &srv);
            }
            m_ctx->Draw(3, 0);
            ID3D11ShaderResourceView *none = nullptr;
            m_ctx->PSSetShaderResources(t.depth ? 3 : 2, 1, &none);
            m_boundStateValid = false;
            t.vramNewer = false;
            if (m_probeFrom <= m_statPrimsTotal && m_statPrimsTotal <= m_probeTo && !t.depth)
                probeLocked(t);
            ++m_statSeeds;
            m_statSeedNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        }

        // -------------------------------------------------------------- textures
        // The race re-uploads its textures into a small pool of local memory every frame, so a TEX0
        // whose pages were written since its last use is hashed (pages, seeded with the TEX0) and
        // decoded only when those bytes were not decoded before.
        void textureFor(const GSDrawState &state, TexEntry *&out)
        {
            const GSTex0Reg &tex = state.context.tex0;
            const uint32_t w = state.textureWidth, h = state.textureHeight;
            const uint64_t key = static_cast<uint64_t>(tex.tbp0 & 0x3FFFu) | (static_cast<uint64_t>(tex.tbw & 0x3Fu) << 14) |
                                 (static_cast<uint64_t>(tex.psm & 0x3Fu) << 20) | (static_cast<uint64_t>(tex.tw & 0xFu) << 26) |
                                 (static_cast<uint64_t>(tex.th & 0xFu) << 30);
            TexSlot &slot = m_texSlots[key];
            if (slot.content == 0u)
                addPages(slot.pages, tex.tbp0, tex.tbw, tex.psm, 0u, h - 1u, w - 1u);
            if (m_stats)
                std::snprintf(m_why, sizeof(m_why), "texture tbp0=0x%x tbw=%u psm=0x%x %ux%u", tex.tbp0, tex.tbw, tex.psm, w, h);
            // Render-to-texture: GPU draws into these pages go to local memory first.
            syncForCpuLocked(slot.pages, false);
            uint64_t sum = 0;
            for (uint32_t p = 0; p < kPages; ++p)
                if (slot.pages.test(p))
                    sum += m_pageVersion[p];
            if (slot.content != 0u && slot.versionSum == sum && !m_noCache)
            {
                const auto it = m_textures.find(slot.content);
                if (it != m_textures.end())
                {
                    it->second.lastUse = ++m_useClock;
                    out = &it->second;
                    if (m_texCheck)
                        checkTextureLocked(it->second, tex, w, h, slot.pages);
                    return;
                }
            }
            cpuIdleLocked(&slot.pages);
            const auto hashStart = std::chrono::steady_clock::now();
            uint64_t content = key;
            for (uint32_t p = 0; p < kPages; ++p)
                if (slot.pages.test(p))
                    content = XXH3_64bits_withSeed(m_vram + static_cast<size_t>(p) * kPageBytes, kPageBytes, content);
            content += content == 0u ? 1u : 0u;
            slot.versionSum = sum;
            slot.content = content;
            TexEntry &e = m_textures[content];
            e.lastUse = ++m_useClock;
            out = &e;
            const auto start = std::chrono::steady_clock::now();
            m_statHashNs += std::chrono::duration_cast<std::chrono::nanoseconds>(start - hashStart).count();
            if (e.tex && !m_noCache)
            {
                ++m_statTexHits;
                if (m_texCheck)
                    checkTextureLocked(e, tex, w, h, slot.pages);
                return;
            }
            m_decode.resize(static_cast<size_t>(w) * h);
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x)
                    m_decode[y * w + x] = GSMem::ReadTexture(m_decodeCache, m_vram, tex.psm, tex.tbp0, tex.tbw, x, y);
            m_decodeCache.Invalidate();
            if (m_texCheck)
                e.texels = m_decode;
            if (!e.tex)
            {
                D3D11_TEXTURE2D_DESC td{};
                td.Width = w;
                td.Height = h;
                td.MipLevels = 1;
                td.ArraySize = 1;
                td.Format = DXGI_FORMAT_R32_UINT;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_DEFAULT;
                td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                D3D11_SUBRESOURCE_DATA init{m_decode.data(), w * 4u, 0};
                m_dev->CreateTexture2D(&td, &init, &e.tex);
                m_dev->CreateShaderResourceView(e.tex.Get(), nullptr, &e.srv);
                e.w = w;
                e.h = h;
                m_texBytes += static_cast<uint64_t>(w) * h * 4u;
            }
            else
                m_ctx->UpdateSubresource(e.tex.Get(), 0, nullptr, m_decode.data(), w * 4u, 0);
            ++m_statDecodes;
            m_statDecodeNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            if (m_texBytes > kTextureCacheBytes)
                evictTexturesLocked();
        }

        // ATV_GS_GPU_TEXCHECK=1 (debug): a texture found by its content is decoded again and compared.
        void checkTextureLocked(const TexEntry &e, const GSTex0Reg &tex, uint32_t w, uint32_t h, const PageSet &pages)
        {
            std::vector<uint32_t> now(static_cast<size_t>(w) * h);
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x)
                    now[y * w + x] = GSMem::ReadTexture(m_decodeCache, m_vram, tex.psm, tex.tbp0, tex.tbw, x, y);
            m_decodeCache.Invalidate();
            ++m_statTexChecks;
            if (e.texels.size() != now.size() || e.texels == now)
                return;
            uint32_t count = 0, x0 = w, y0 = h, x1 = 0, y1 = 0, fx = 0, fy = 0;
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x)
                    if (e.texels[y * w + x] != now[y * w + x])
                    {
                        if (count++ == 0u)
                            fx = x, fy = y;
                        x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
                    }
            ++m_statTexMismatches;
            static uint32_t s_logs = 0;
            if (s_logs++ >= 60u)
                return;
            const GSMem::PixelAddressFunc address = GSMem::GetPixelAddressFunc(tex.psm);
            const int page = address ? static_cast<int>(address(tex.tbp0, tex.tbw, fx, fy) / kPageBytes) : -1;
            std::fprintf(stderr, "[gs-gpu] texcheck MISMATCH tbp0=0x%x tbw=%u psm=0x%x %ux%u: %u texels in (%u,%u)-(%u,%u), first (%u,%u) "
                                 "page %d %s the %zu hashed pages\n",
                         tex.tbp0, tex.tbw, tex.psm, w, h, count, x0, y0, x1, y1, fx, fy, page,
                         page >= 0 && pages.test(static_cast<size_t>(page) % kPages) ? "in" : "NOT in", pages.count());
        }

        // Drops the least recently used decoded textures down to half the budget (the one just
        // used is the newest, so it stays).
        void evictTexturesLocked()
        {
            std::vector<std::pair<uint64_t, uint64_t>> uses;
            for (auto &[k, e] : m_textures)
                uses.push_back({e.lastUse, k});
            std::sort(uses.begin(), uses.end());
            for (size_t i = 0; i < uses.size() && m_texBytes > kTextureCacheBytes / 2u; ++i)
            {
                const auto it = m_textures.find(uses[i].second);
                m_texBytes -= static_cast<uint64_t>(it->second.w) * it->second.h * 4u;
                m_textures.erase(it);
            }
        }

        ID3D11ShaderResourceView *hdTextureLocked(const GSTexReplace::Image &img)
        {
            auto &e = m_hd[&img];
            if (!e.srv || e.texels != img.levels[0].texels || e.w != img.levels[0].w || e.h != img.levels[0].h ||
                e.levels != img.levelCount)
            {
                e = HdEntry{};
                D3D11_TEXTURE2D_DESC td{};
                td.Width = img.levels[0].w;
                td.Height = img.levels[0].h;
                td.MipLevels = img.levelCount;
                td.ArraySize = 1;
                td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_IMMUTABLE;
                td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                D3D11_SUBRESOURCE_DATA init[16]{};
                for (uint32_t i = 0; i < img.levelCount; ++i)
                {
                    init[i].pSysMem = img.levels[i].texels;
                    init[i].SysMemPitch = img.levels[i].w * 4u;
                    e.bytes += static_cast<uint64_t>(img.levels[i].w) * img.levels[i].h * 4u;
                }
                ComPtr<ID3D11Texture2D> tex;
                m_dev->CreateTexture2D(&td, init, &tex);
                if (tex)
                    m_dev->CreateShaderResourceView(tex.Get(), nullptr, &e.srv);
                e.texels = img.levels[0].texels;
                e.w = img.levels[0].w;
                e.h = img.levels[0].h;
                e.levels = img.levelCount;
                m_hdBytes += e.bytes;
                ++m_statHdUploads;
                if (m_hdBytes > (1536ull << 20))
                    evictHdLocked(&img);
            }
            e.lastUse = ++m_useClock;
            return e.srv.Get();
        }

        void evictHdLocked(const void *keep)
        {
            std::vector<std::pair<uint64_t, const void *>> uses;
            for (auto &[k, e] : m_hd)
                if (k != keep)
                    uses.push_back({e.lastUse, k});
            std::sort(uses.begin(), uses.end());
            for (size_t i = 0; i < uses.size() && m_hdBytes > (1024ull << 20); ++i)
            {
                m_hdBytes -= m_hd[uses[i].second].bytes;
                m_hd.erase(uses[i].second);
            }
        }

        // ------------------------------------------------------------------ CLUT
        void resetClutLocked()
        {
            m_clut.fill(0u);
            m_clutCbp.fill(0u);
            ++m_clutVersion;
        }

        void loadClutLocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
        {
            const bool fourBit = isFourBit(tex0.psm);
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
                uint32_t sourceX, sourceY, sourceWidth = 1u;
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
                const uint32_t raw = sixteenBit ? (tex0.cpsm == GS_PSM_CT16S ? GSMem::ReadCT16S(m_vram, tex0.cbp, sourceWidth, sourceX, sourceY)
                                                                              : GSMem::ReadCT16(m_vram, tex0.cbp, sourceWidth, sourceX, sourceY))
                                                : GSMem::ReadCT32(m_vram, tex0.cbp, sourceWidth, sourceX, sourceY);
                const uint32_t destination = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteenBit ? 0x1FFu : 0x0FFu);
                if (sixteenBit)
                    m_clut[destination] = static_cast<uint16_t>(raw);
                else
                {
                    m_clut[destination] = static_cast<uint16_t>(raw & 0xFFFFu);
                    m_clut[destination + 256u] = static_cast<uint16_t>(raw >> 16u);
                }
            }
        }

        uint32_t lookupClut(const GSDrawState &state, uint32_t index) const
        {
            const GSTex0Reg &tex = state.context.tex0;
            const bool sixteenBit = tex.cpsm == GS_PSM_CT16 || tex.cpsm == GS_PSM_CT16S;
            const uint32_t clutBase = (static_cast<uint32_t>(tex.csa) & (sixteenBit ? 0x1Fu : 0x0Fu)) << 4u;
            const uint32_t sourceIndex = isFourBit(tex.psm) ? (index & 0x0Fu) : index;
            uint32_t clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
            if (!sixteenBit && tex.csm == 0u && isEightBit(tex.psm))
                clutIndex = std::min((sourceIndex & 0xF0u) + clutBase, 240u) + (sourceIndex & 0x0Fu);
            switch (tex.cpsm)
            {
            case GS_PSM_CT32:
                return static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u);
            case GS_PSM_CT24:
                return applyTexa(state.texa, GS_PSM_CT24,
                                 (static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u)) &
                                     0x00FFFFFFu);
            case GS_PSM_CT16:
            case GS_PSM_CT16S:
                return applyTexa(state.texa, tex.cpsm, rgba5551To8888(m_clut[clutIndex]));
            default:
                return 0u;
            }
        }

        // -------------------------------------------------------------- drawing
        void submitLocked(const GSPrimitiveBatch &batch)
        {
            if (!m_vram || batch.vertexCount == 0u)
                return;
            const GSDrawState &state = batch.state;
            ++m_statPrims;
            ++m_statPrimsTotal;
            if (m_allCpu || !gpuOwnsFrameLocked(state.context) || state.prim.type == GS_PRIM_LINE || state.prim.type == GS_PRIM_LINESTRIP ||
                state.prim.type == GS_PRIM_POINT)
            {
                cpuDrawLocked(batch);
                return;
            }
            const GSTexReplace::Image *hd = nullptr;
            if (GSTexReplace::Active())
            {
                cpuIdleLocked();
                GSTexReplace::OnSubmit(m_cpu.get(), batch, m_vram, m_clut.data());
                hd = state.prim.tme ? GSTexReplace::Current() : nullptr;
            }
            if (!m_batchOpen || hd != m_batchHd || std::memcmp(&state, &m_batchState, sizeof(GSDrawState)) != 0)
            {
                flushBatchLocked();
                openBatchLocked(state, hd);
            }
            if (state.prim.type == GS_PRIM_SPRITE)
                appendSprite(batch);
            else
                appendTriangle(batch);
        }

        // Narrows [y0, y1] (the scissor) to the rows the primitive's vertices span.
        static void primitiveRows(const GSPrimitiveBatch &batch, uint32_t &y0, uint32_t &y1)
        {
            const float ofy = static_cast<float>(batch.state.context.xyoffset.ofy >> 4);
            float lo = 1.0e9f, hi = -1.0e9f;
            for (uint32_t i = 0; i < batch.vertexCount && i < batch.vertices.size(); ++i)
            {
                lo = std::min(lo, batch.vertices[i].y - ofy);
                hi = std::max(hi, batch.vertices[i].y - ofy);
            }
            if (lo > hi || !(std::fabs(lo) < 1.0e6f) || !(std::fabs(hi) < 1.0e6f))
                return;
            const int64_t first = static_cast<int64_t>(std::floor(lo)) - 1, last = static_cast<int64_t>(std::ceil(hi)) + 1;
            if (last < static_cast<int64_t>(y0) || first > static_cast<int64_t>(y1))
            {
                y1 = y0; // draws nothing: one row keeps the page sets non-empty
                return;
            }
            y0 = static_cast<uint32_t>(std::max<int64_t>(first, y0));
            y1 = static_cast<uint32_t>(std::min<int64_t>(last, y1));
        }

        void cpuDrawLocked(const GSPrimitiveBatch &batch)
        {
            flushBatchLocked();
            const GSContext &ctx = batch.state.context;
            if (m_stats)
            {
                const uint64_t key = (static_cast<uint64_t>(batch.state.prim.type) << 40) | (static_cast<uint64_t>(ctx.frame.fbp) << 20) |
                                     (static_cast<uint64_t>(ctx.frame.fbw) << 12) | ctx.frame.psm;
                if (m_cpuKeys.size() < 64u && m_cpuKeys.insert(key).second)
                    std::fprintf(stderr, "[gs-gpu] CPU draw: type=%u fbp=%u fbw=%u fpsm=0x%x zbp=%u test=0x%llx sc=(%u,%u)-(%u,%u) tme=%u\n",
                                 static_cast<uint32_t>(batch.state.prim.type), ctx.frame.fbp, ctx.frame.fbw, ctx.frame.psm, ctx.zbuf.zbp,
                                 static_cast<unsigned long long>(ctx.test), ctx.scissor.x0, ctx.scissor.y0, ctx.scissor.x1, ctx.scissor.y1,
                                 static_cast<uint32_t>(batch.state.prim.tme));
            }
            if (m_stats)
                std::snprintf(m_why, sizeof(m_why), "cpu draw type=%u fbp=%u fbw=%u psm=0x%x zbp=%u test=0x%llx zmsk=%u sc=(%u,%u)-(%u,%u)",
                              static_cast<uint32_t>(batch.state.prim.type), ctx.frame.fbp, ctx.frame.fbw, ctx.frame.psm, ctx.zbuf.zbp,
                              static_cast<unsigned long long>(ctx.test), static_cast<uint32_t>(ctx.zbuf.zmask), ctx.scissor.x0, ctx.scissor.y0,
                              ctx.scissor.x1, ctx.scissor.y1);
            PageSet writes, reads;
            // The rows the primitive can reach (its vertices inside the scissor): with the scissor
            // alone PSXWater's Z pass (ZBP 352, 384 rows, data in the first 188) covered the race's
            // shadow buffer at FBP 420, and downloaded and re-seeded it a few times a frame.
            uint32_t rowFirst = ctx.scissor.y0, rowLast = ctx.scissor.y1;
            primitiveRows(batch, rowFirst, rowLast);
            addPages(writes, ctx.frame.fbp << 5, ctx.frame.fbw, ctx.frame.psm, rowFirst, rowLast);
            // Z pages only as the CPU GS touches them: the race's shadow pass has ZBP 0 (= the display
            // buffer) with ZMSK on, and counting it as a write re-seeded that target at 1x every frame.
            if (writesDepth(ctx))
                addPages(writes, ctx.zbuf.zbp << 5, ctx.frame.fbw, ctx.zbuf.psm, rowFirst, rowLast);
            else if (readsDepth(ctx))
                addPages(reads, ctx.zbuf.zbp << 5, ctx.frame.fbw, ctx.zbuf.psm, rowFirst, rowLast);
            if (batch.state.prim.tme)
                addPages(reads, ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0u, batch.state.textureHeight - 1u,
                         batch.state.textureWidth - 1u);
            syncForCpuLocked(reads, false);
            syncForCpuLocked(writes, true);
            const auto drawStart = m_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            m_cpu->Submit(batch);
            m_cpuQueued = true;
            m_cpuQueuedPages |= writes;
            m_cpuQueuedPages |= reads;
            if (m_stats)
            {
                // Per kind of CPU draw (prim type, FBP, FBW): primitives and time in the CPU GS.
                const uint64_t kind = (static_cast<uint64_t>(batch.state.prim.type) << 32) | (static_cast<uint64_t>(ctx.frame.fbp) << 8) | ctx.frame.fbw;
                auto &[prims, ns] = m_cpuKindStats[kind];
                ++prims;
                ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - drawStart).count();
            }
            noteCpuWriteLocked(writes);
            ++m_statCpuPrims;
        }

        void openBatchLocked(const GSDrawState &state, const GSTexReplace::Image *hd)
        {
            const auto openStart = std::chrono::steady_clock::now();
            const GSContext &ctx = state.context;
            m_batchOpen = true;
            m_batchState = state;
            m_batchHd = hd;
            m_vertices.clear();

            const uint32_t rows = static_cast<uint32_t>(ctx.scissor.y1) + 1u;
            m_color = acquireTargetLocked(false, ctx.frame.fbp, ctx.frame.fbw, ctx.frame.psm, rows);
            m_depth = nullptr;
            if (usesDepth(ctx))
            {
                PageSet zPages;
                addPages(zPages, ctx.zbuf.zbp << 5, ctx.frame.fbw, ctx.zbuf.psm, 0u, rows - 1u);
                // FRAME and ZBUF on the same pages: no GPU depth (would evict the colour target).
                if (!(zPages & m_color->pages).any())
                    m_depth = acquireTargetLocked(true, ctx.zbuf.zbp, ctx.frame.fbw, ctx.zbuf.psm, rows);
                else
                    ++m_statApproxMask;
            }
            if (m_color->vramNewer)
                seedLocked(*m_color);
            if (m_depth && m_depth->vramNewer)
                seedLocked(*m_depth);
            m_zScale = zScale(ctx.zbuf.psm);

            DrawConstants &k = m_constants;
            k = {};
            k.ndc[0] = 2.0f / static_cast<float>(m_color->w);
            k.ndc[1] = 2.0f / static_cast<float>(m_color->h);
            ID3D11ShaderResourceView *srvs[2] = {nullptr, nullptr};
            uint32_t hdSampler = 0;
            if (state.prim.tme)
            {
                const GSTex0Reg &tex = ctx.tex0;
                k.tex[1] = state.textureWidth;
                k.tex[2] = state.textureHeight;
                k.tex[3] = state.linearFilter ? 1u : 0u;
                k.wrap[0] = static_cast<uint32_t>(ctx.clamp & 3u);
                k.wrap[1] = static_cast<uint32_t>((ctx.clamp >> 2) & 3u);
                k.wrap[2] = static_cast<uint32_t>((ctx.clamp >> 4) & 0x3FFu) | (static_cast<uint32_t>((ctx.clamp >> 14) & 0x3FFu) << 16);
                k.wrap[3] = static_cast<uint32_t>((ctx.clamp >> 24) & 0x3FFu) | (static_cast<uint32_t>((ctx.clamp >> 34) & 0x3FFu) << 16);
                k.texa[0] = state.texa.ta0;
                k.texa[1] = state.texa.ta1;
                k.texa[2] = state.texa.aem ? 1u : 0u;
                k.texa[3] = (tex.tfx & 15u) | ((tex.tcc & 1u) << 4);
                if (hd)
                {
                    GSTexReplace::Sampler s;
                    GSTexReplace::Prepare(*hd, 0u, state, s);
                    k.tex[0] = 5u;
                    k.hd[0] = s.u.scale / static_cast<float>(hd->levels[0].w);
                    k.hd[1] = s.v.scale / static_cast<float>(hd->levels[0].h);
                    k.hd[2] = s.u.offset / static_cast<float>(hd->levels[0].w);
                    k.hd[3] = s.v.offset / static_cast<float>(hd->levels[0].h);
                    hdSampler = (s.u.repeat ? 0u : 1u) | (s.v.repeat ? 0u : 2u);
                    srvs[1] = hdTextureLocked(*hd);
                }
                else if (Target *rt = renderTargetTextureLocked(tex, state.textureWidth, state.textureHeight))
                {
                    srvs[1] = rt->srv.Get();
                    k.tex[0] = tex.psm == GS_PSM_CT24 ? 7u : 6u;
                    k.hd[0] = static_cast<float>(m_scale);
                    k.hd[1] = static_cast<float>(m_scale / 2u);
                    ++m_statRtTextures;
                }
                else
                {
                    TexEntry *entry = nullptr;
                    textureFor(state, entry);
                    srvs[0] = entry->srv.Get();
                    if (isFourBit(tex.psm) || isEightBit(tex.psm))
                    {
                        k.tex[0] = 4u;
                        uploadPaletteLocked(state);
                    }
                    else if (tex.psm == GS_PSM_CT16 || tex.psm == GS_PSM_CT16S)
                        k.tex[0] = 3u;
                    else if (tex.psm == GS_PSM_CT24)
                        k.tex[0] = 2u;
                    else
                        k.tex[0] = 1u;
                }
            }
            if (state.prim.fge && state.prim.type != GS_PRIM_SPRITE)
                k.fog[0] = state.fogR, k.fog[1] = state.fogG, k.fog[2] = state.fogB, k.fog[3] = 1u;
            const uint64_t test = ctx.test;
            k.test[0] = (test & 1u) ? static_cast<uint32_t>((test >> 1) & 7u) : 8u;
            k.test[1] = static_cast<uint32_t>((test >> 4) & 0xFFu);
            k.test[3] = ((ctx.fba & 1u) != 0u && ctx.frame.psm != GS_PSM_CT24) ? 1u : 0u;

            // Blend: (A - B) * C >> 7 + D.
            const uint32_t sel = static_cast<uint32_t>(ctx.alpha & 0xFFu);
            const uint32_t A = sel & 3u, B = (sel >> 2) & 3u, C = (sel >> 4) & 3u, D = (sel >> 6) & 3u;
            const uint32_t fix = static_cast<uint32_t>((ctx.alpha >> 32) & 0xFFu);
            k.blend[1] = sel;
            k.blend[2] = fix;
            bool enable = false;
            D3D11_BLEND src = D3D11_BLEND_ONE, dst = D3D11_BLEND_ZERO;
            D3D11_BLEND_OP op = D3D11_BLEND_OP_ADD;
            float factor = 1.0f;
            if (state.prim.abe && !(state.pabe))
            {
                const bool usesCd = A == 1u || B == 1u || D == 1u;
                if (!usesCd || A == B)
                {
                    if (A == B && D == 1u)
                    {
                        enable = true;
                        src = D3D11_BLEND_ZERO;
                        dst = D3D11_BLEND_ONE;
                    }
                    else
                        k.blend[0] = 1u;
                }
                else
                {
                    D3D11_BLEND f = D3D11_BLEND_SRC1_ALPHA, invF = D3D11_BLEND_INV_SRC1_ALPHA;
                    if (C == 2u)
                    {
                        f = D3D11_BLEND_BLEND_FACTOR;
                        invF = D3D11_BLEND_INV_BLEND_FACTOR;
                        factor = std::min(1.0f, static_cast<float>(fix) / 128.0f);
                    }
                    else if (C == 1u)
                    {
                        f = D3D11_BLEND_DEST_ALPHA; // Ad/255, not Ad/128: approximate
                        invF = D3D11_BLEND_INV_DEST_ALPHA;
                        ++m_statApproxBlend;
                    }
                    enable = true;
                    const uint32_t abd = A | (B << 2) | (D << 4);
                    switch (abd)
                    {
                    case 0u | (1u << 2) | (1u << 4): src = f; dst = invF; break;                                  // Cs*F + Cd*(1-F)
                    case 0u | (1u << 2) | (2u << 4): src = f; dst = f; op = D3D11_BLEND_OP_SUBTRACT; break;      // (Cs-Cd)*F
                    case 0u | (2u << 2) | (1u << 4): src = f; dst = D3D11_BLEND_ONE; break;                     // Cs*F + Cd
                    case 1u | (0u << 2) | (0u << 4): src = invF; dst = f; break;                                // Cd*F + Cs*(1-F)
                    case 1u | (0u << 2) | (2u << 4): src = f; dst = f; op = D3D11_BLEND_OP_REV_SUBTRACT; break; // (Cd-Cs)*F
                    case 1u | (2u << 2) | (0u << 4): src = D3D11_BLEND_ONE; dst = f; break;                     // Cs + Cd*F
                    case 1u | (2u << 2) | (2u << 4): src = D3D11_BLEND_ZERO; dst = f; break;                    // Cd*F
                    case 2u | (0u << 2) | (1u << 4): src = f; dst = D3D11_BLEND_ONE; op = D3D11_BLEND_OP_REV_SUBTRACT; break; // Cd - Cs*F
                    case 2u | (1u << 2) | (0u << 4): src = D3D11_BLEND_ONE; dst = f; op = D3D11_BLEND_OP_SUBTRACT; break;     // Cs - Cd*F
                    case 2u | (1u << 2) | (1u << 4): src = D3D11_BLEND_ZERO; dst = invF; break;                 // Cd*(1-F)
                    case 2u | (1u << 2) | (2u << 4): src = D3D11_BLEND_ZERO; dst = D3D11_BLEND_ZERO; break;     // -Cd*F -> 0
                    default: // factors above 1 (e.g. Cs*(1+F) - Cd*F): approximate as Cs*F + Cd*(1-F)
                        src = f;
                        dst = invF;
                        ++m_statApproxBlend;
                        break;
                    }
                }
            }
            m_blendFactor = factor;

            // Alpha test with AFAIL: KEEP discards; the others draw the failing fragments in a
            // second pass with the writes AFAIL allows.
            const uint8_t fullMask = D3D11_COLOR_WRITE_ENABLE_ALL;
            uint8_t colorMask = fullMask;
            const uint32_t fbmsk = ctx.frame.fbmsk;
            if (fbmsk != 0u)
            {
                colorMask = 0;
                for (int c = 0; c < 4; ++c)
                    if (((fbmsk >> (c * 8)) & 0xFFu) != 0xFFu)
                        colorMask |= static_cast<uint8_t>(1u << c);
                if (fbmsk != 0xFF000000u && fbmsk != 0x00FFFFFFu && fbmsk != 0xFFFFFFFFu)
                    ++m_statApproxMask;
            }
            const bool zWrite = m_depth && !ctx.zbuf.zmask;
            m_passCount = 0;
            const bool ate = (test & 1u) != 0u;
            const uint32_t atst = static_cast<uint32_t>((test >> 1) & 7u);
            const uint32_t afail = static_cast<uint32_t>((test >> 12) & 3u);
            if (!ate || atst == 1u || afail == 0u)
                m_passes[m_passCount++] = {ate && atst != 1u ? 1u : 0u, colorMask, zWrite};
            else
            {
                if (atst != 0u)
                    m_passes[m_passCount++] = {1u, colorMask, zWrite};
                Pass fail{2u, 0, false};
                if (afail == 1u)
                    fail.colorMask = colorMask;
                else if (afail == 2u)
                    fail.depthWrite = zWrite;
                else
                    fail.colorMask = static_cast<uint8_t>(colorMask & (ctx.frame.psm == GS_PSM_CT32 ? 7u : 15u));
                m_passes[m_passCount++] = fail;
            }

            // Pipeline state for the whole batch.
            D3D11_COMPARISON_FUNC func = D3D11_COMPARISON_ALWAYS;
            if (m_depth && ((test >> 16) & 1u) != 0u)
            {
                switch ((test >> 17) & 3u)
                {
                case 0: func = D3D11_COMPARISON_NEVER; break;
                case 1: func = D3D11_COMPARISON_ALWAYS; break;
                case 2: func = D3D11_COMPARISON_GREATER_EQUAL; break;
                default: func = D3D11_COMPARISON_GREATER; break;
                }
            }
            m_depthFunc = func;
            m_blendEnable = enable;
            m_blendSrc = src;
            m_blendDst = dst;
            m_blendOp = op;

            ID3D11RenderTargetView *rtv = m_color->rtv.Get();
            m_ctx->OMSetRenderTargets(1, &rtv, m_depth ? m_depth->dsv.Get() : nullptr);
            D3D11_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(m_color->w * m_scale), static_cast<float>(m_color->h * m_scale), 0.0f, 1.0f};
            m_ctx->RSSetViewports(1, &vp);
            const D3D11_RECT sc{static_cast<LONG>(ctx.scissor.x0 * m_scale), static_cast<LONG>(ctx.scissor.y0 * m_scale),
                                static_cast<LONG>((ctx.scissor.x1 + 1u) * m_scale), static_cast<LONG>((ctx.scissor.y1 + 1u) * m_scale)};
            m_ctx->RSSetScissorRects(1, &sc);
            if (!m_boundStateValid)
            {
                m_ctx->RSSetState(m_raster.Get());
                m_ctx->IASetInputLayout(m_layout.Get());
                m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                const UINT stride = sizeof(Vertex), offset = 0;
                m_ctx->IASetVertexBuffers(0, 1, m_vb.GetAddressOf(), &stride, &offset);
                m_ctx->VSSetShader(m_vs.Get(), nullptr, 0);
                m_ctx->PSSetShader(m_ps.Get(), nullptr, 0);
                ID3D11Buffer *cbs[2] = {m_cbDraw.Get(), m_cbPalette.Get()};
                m_ctx->VSSetConstantBuffers(0, 1, cbs);
                m_ctx->PSSetConstantBuffers(0, 2, cbs);
                m_boundStateValid = true;
            }
            m_ctx->PSSetShaderResources(0, 2, srvs);
            ID3D11SamplerState *sampler = m_hdSamplers[hdSampler].Get();
            m_ctx->PSSetSamplers(0, 1, &sampler);
            ++m_statBatches;
            m_statOpenNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - openStart).count();
        }

        void uploadPaletteLocked(const GSDrawState &state)
        {
            const GSTex0Reg &tex = state.context.tex0;
            const uint64_t key = m_clutVersion ^ (static_cast<uint64_t>(tex.cpsm) << 40) ^ (static_cast<uint64_t>(tex.csa) << 46) ^
                                 (static_cast<uint64_t>(tex.csm) << 51) ^ (static_cast<uint64_t>(isFourBit(tex.psm)) << 52) ^
                                 (static_cast<uint64_t>(state.texa.ta0) << 53) ^ (static_cast<uint64_t>(state.texa.ta1) << 24) ^
                                 (static_cast<uint64_t>(state.texa.aem) << 63);
            if (key == m_paletteKey && !m_noCache)
                return;
            m_paletteKey = key;
            uint32_t pal[256];
            const uint32_t n = isFourBit(tex.psm) ? 16u : 256u;
            for (uint32_t i = 0; i < 256u; ++i)
                pal[i] = i < n ? lookupClut(state, i) : 0u;
            D3D11_MAPPED_SUBRESOURCE map{};
            if (SUCCEEDED(m_ctx->Map(m_cbPalette.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &map)))
            {
                std::memcpy(map.pData, pal, sizeof(pal));
                m_ctx->Unmap(m_cbPalette.Get(), 0);
            }
            ++m_statPalettes;
        }

        Vertex makeVertex(const GSVertex &v, const GSContext &ctx, float s, float t, float q, const GSVertex &colour, double z) const
        {
            Vertex out;
            out.x = v.x - static_cast<float>(ctx.xyoffset.ofx >> 4);
            out.y = v.y - static_cast<float>(ctx.xyoffset.ofy >> 4);
            out.z = std::min(static_cast<float>(z) * m_zScale, 1.0f);
            out.s = s;
            out.t = t;
            out.q = q;
            out.rgba = static_cast<uint32_t>(colour.r) | (static_cast<uint32_t>(colour.g) << 8) | (static_cast<uint32_t>(colour.b) << 16) |
                       (static_cast<uint32_t>(colour.a) << 24);
            out.fog = static_cast<float>(v.fog);
            return out;
        }

        // Same rejections as GSCpuBackend::DrawTriangle.
        void appendTriangle(const GSPrimitiveBatch &batch)
        {
            const GSDrawState &state = batch.state;
            const GSContext &ctx = state.context;
            const GSVertex *v = batch.vertices.data();
            const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4), ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
            float fx[3], fy[3];
            for (int i = 0; i < 3; ++i)
            {
                fx[i] = v[i].x - ofx;
                fy[i] = v[i].y - ofy;
            }
            const bool fogged3d = !state.prim.fst && state.prim.fge;
            if (fogged3d && (v[0].q <= 1.0e-5f || v[1].q <= 1.0e-5f || v[2].q <= 1.0e-5f))
                return;
            const int minX = static_cast<int>(std::floor(std::min({fx[0], fx[1], fx[2]})));
            const int maxX = static_cast<int>(std::ceil(std::max({fx[0], fx[1], fx[2]})));
            const int minY = static_cast<int>(std::floor(std::min({fy[0], fy[1], fy[2]})));
            const int maxY = static_cast<int>(std::ceil(std::max({fy[0], fy[1], fy[2]})));
            const int viewW = static_cast<int>(ctx.scissor.x1) - static_cast<int>(ctx.scissor.x0) + 1;
            const int viewH = static_cast<int>(ctx.scissor.y1) - static_cast<int>(ctx.scissor.y0) + 1;
            if (fogged3d && (maxX - minX > viewW || maxY - minY > viewH))
                return;
            if (maxX < ctx.scissor.x0 || minX > ctx.scissor.x1 || maxY < ctx.scissor.y0 || minY > ctx.scissor.y1)
                return;
            for (int i = 0; i < 3; ++i)
                if (std::fabs(fx[i]) > 30000.0f || std::fabs(fy[i]) > 30000.0f)
                    return;
            const int64_t gx0 = std::llround(fx[0] * 16.0f), gx1 = std::llround(fx[1] * 16.0f), gx2 = std::llround(fx[2] * 16.0f);
            const int64_t gy0 = std::llround(fy[0] * 16.0f), gy1 = std::llround(fy[1] * 16.0f), gy2 = std::llround(fy[2] * 16.0f);
            if ((gx1 - gx0) * (gy2 - gy0) - (gy1 - gy0) * (gx2 - gx0) == 0)
                return;
            const bool gouraud = state.prim.iip;
            for (int i = 0; i < 3; ++i)
            {
                float s, t, q;
                if (state.prim.fst)
                {
                    s = static_cast<float>(v[i].u) / 16.0f / static_cast<float>(state.textureWidth);
                    t = static_cast<float>(v[i].v) / 16.0f / static_cast<float>(state.textureHeight);
                    q = 1.0f;
                }
                else
                {
                    s = v[i].s;
                    t = v[i].t;
                    q = v[i].q > 1.0e-5f ? v[i].q : 1.0f;
                }
                m_vertices.push_back(makeVertex(v[i], ctx, s, t, q, gouraud ? v[i] : v[2], v[i].z));
            }
        }

        void appendSprite(const GSPrimitiveBatch &batch)
        {
            const GSDrawState &state = batch.state;
            const GSContext &ctx = state.context;
            const GSVertex &v0 = batch.vertices[0], &v1 = batch.vertices[1];
            const int ofx = ctx.xyoffset.ofx >> 4, ofy = ctx.xyoffset.ofy >> 4;
            int x0 = static_cast<int>(v0.x) - ofx, y0 = static_cast<int>(v0.y) - ofy;
            int x1 = static_cast<int>(v1.x) - ofx, y1 = static_cast<int>(v1.y) - ofy;
            float u0 = 0.0f, t0 = 0.0f, u1 = 0.0f, t1 = 0.0f;
            if (state.prim.tme)
            {
                const float w = static_cast<float>(state.textureWidth), h = static_cast<float>(state.textureHeight);
                if (state.prim.fst)
                {
                    u0 = static_cast<float>(v0.u) / 16.0f / w;
                    t0 = static_cast<float>(v0.v) / 16.0f / h;
                    u1 = static_cast<float>(v1.u) / 16.0f / w;
                    t1 = static_cast<float>(v1.v) / 16.0f / h;
                }
                else
                {
                    const float q0 = std::fabs(v0.q) > 1.0e-5f ? std::fabs(v0.q) : 1.0f;
                    const float q1 = std::fabs(v1.q) > 1.0e-5f ? std::fabs(v1.q) : 1.0f;
                    u0 = v0.s / q0;
                    t0 = v0.t / q0;
                    u1 = v1.s / q1;
                    t1 = v1.t / q1;
                }
            }
            if (x0 > x1)
            {
                std::swap(x0, x1);
                std::swap(u0, u1);
            }
            if (y0 > y1)
            {
                std::swap(y0, y1);
                std::swap(t0, t1);
            }
            x1 = x0 + std::max(1, x1 - x0);
            y1 = y0 + std::max(1, y1 - y0);
            uint32_t z = static_cast<uint32_t>(v1.z);
            const bool coversDisplay = std::max<int>(x0, ctx.scissor.x0) <= 0 && std::max<int>(y0, ctx.scissor.y0) <= 0 &&
                                       std::min<int>(x1 - 1, ctx.scissor.x1) >= 639 && std::min<int>(y1 - 1, ctx.scissor.y1) >= 447;
            if (coversDisplay && !state.prim.tme && z > 0x800000u)
                z = 0u;
            GSVertex corner = v1;
            auto push = [&](int x, int y, float s, float t)
            {
                corner.x = static_cast<float>(x + ofx);
                corner.y = static_cast<float>(y + ofy);
                m_vertices.push_back(makeVertex(corner, ctx, s, t, 1.0f, v1, z));
            };
            push(x0, y0, u0, t0);
            push(x1, y0, u1, t0);
            push(x0, y1, u0, t1);
            push(x1, y0, u1, t0);
            push(x1, y1, u1, t1);
            push(x0, y1, u0, t1);
        }

        void flushBatchLocked()
        {
            if (!m_batchOpen)
                return;
            m_batchOpen = false;
            if (m_vertices.empty())
                return;
            const auto flushStart = std::chrono::steady_clock::now();
            const UINT bytes = static_cast<UINT>(m_vertices.size() * sizeof(Vertex));
            size_t done = 0;
            while (done < m_vertices.size())
            {
                // Whole triangles per chunk.
                const size_t maxVerts = (kVertexBufferBytes / sizeof(Vertex)) / 3u * 3u;
                const size_t count = std::min(maxVerts, m_vertices.size() - done);
                const UINT chunkBytes = static_cast<UINT>(count * sizeof(Vertex));
                D3D11_MAP mode = D3D11_MAP_WRITE_NO_OVERWRITE;
                if (m_vbOffset + chunkBytes > kVertexBufferBytes)
                {
                    mode = D3D11_MAP_WRITE_DISCARD;
                    m_vbOffset = 0;
                }
                D3D11_MAPPED_SUBRESOURCE map{};
                if (FAILED(m_ctx->Map(m_vb.Get(), 0, mode, 0, &map)))
                    return;
                std::memcpy(static_cast<uint8_t *>(map.pData) + m_vbOffset, m_vertices.data() + done, chunkBytes);
                m_ctx->Unmap(m_vb.Get(), 0);
                const UINT first = m_vbOffset / sizeof(Vertex);
                for (uint32_t p = 0; p < m_passCount; ++p)
                {
                    const Pass &pass = m_passes[p];
                    m_constants.test[2] = pass.mode;
                    D3D11_MAPPED_SUBRESOURCE cb{};
                    if (SUCCEEDED(m_ctx->Map(m_cbDraw.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &cb)))
                    {
                        std::memcpy(cb.pData, &m_constants, sizeof(m_constants));
                        m_ctx->Unmap(m_cbDraw.Get(), 0);
                    }
                    const float f = m_blendFactor;
                    const float factors[4] = {f, f, f, f};
                    m_ctx->OMSetBlendState(blendState(m_blendEnable, m_blendSrc, m_blendDst, m_blendOp, pass.colorMask), factors, 0xFFFFFFFFu);
                    m_ctx->OMSetDepthStencilState(depthState(m_depth != nullptr, m_depthFunc, pass.depthWrite), 0);
                    m_ctx->Draw(static_cast<UINT>(count), first);
                    ++m_statDraws;
                    if (pass.colorMask)
                        m_color->gpuNewer = true;
                    if (pass.depthWrite && m_depth)
                        m_depth->gpuNewer = true;
                }
                m_vbOffset += chunkBytes; // stays a multiple of sizeof(Vertex): Draw addresses it in vertices
                done += count;
            }
            m_statVertexBytes += bytes;
            if (m_probeFrom <= m_statPrimsTotal && m_statPrimsTotal <= m_probeTo && m_traceFbp >= 0 && m_color &&
                m_color->base == static_cast<uint32_t>(m_traceFbp))
                probeLocked(*m_color);
            m_vertices.clear();
            m_statFlushNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - flushStart).count();
            logStatsLocked();
        }

        // ---------------------------------------------------------- presentation
        bool presentFromGpuLocked(const GSPresentationRequest &request, PresentationFrame &frame)
        {
            const bool en1 = (request.pmode & 1u) != 0u, en2 = (request.pmode & 2u) != 0u;
            auto displayOk = [](uint64_t display) {
                const uint32_t dw = static_cast<uint32_t>((display >> 32) & 0xFFFu), dh = static_cast<uint32_t>((display >> 44) & 0x7FFu);
                return dw != 0xFFFu && dh != 0x7FFu && (dw != 0u || dh != 0u);
            };
            const uint64_t dispfb = (en2 && displayOk(request.display2) && ((request.dispfb2 >> 9) & 0x3Fu)) ? request.dispfb2
                                    : (en1 && displayOk(request.display1) && ((request.dispfb1 >> 9) & 0x3Fu)) ? request.dispfb1
                                                                                                                : 0u;
            if (!dispfb)
                return false;
            const uint64_t display = dispfb == request.dispfb2 ? request.display2 : request.display1;
            const uint32_t fbp = static_cast<uint32_t>(dispfb & 0x1FFu);
            const uint32_t fbw = static_cast<uint32_t>((dispfb >> 9) & 0x3Fu);
            const uint32_t originX = static_cast<uint32_t>((dispfb >> 32) & 0x7FFu), originY = static_cast<uint32_t>((dispfb >> 43) & 0x7FFu);
            const uint32_t magh = static_cast<uint32_t>((display >> 23) & 0xFu);
            uint32_t w = (static_cast<uint32_t>((display >> 32) & 0xFFFu) + 1u) / (magh + 1u);
            if (w == 639u)
                w = 640u;
            uint32_t h = static_cast<uint32_t>((display >> 44) & 0x7FFu) + 1u;
            w = std::min(w, 640u);
            h = std::min(h, 512u);
            Target *t = findTarget(false, fbp);
            if (!t || t->fbw != fbw || t->vramNewer || originX + w > t->w || originY + h > t->h)
            {
                if (g_raceLoadStarted)
                    ++m_capFallbacks;
                if (m_stats && m_presentFallbackLogs < 200u)
                {
                    ++m_presentFallbackLogs;
                    std::fprintf(stderr, "[gs-gpu] present from local memory: fbp=%u fbw=%u target=%d vramNewer=%d gpuNewer=%d origin=(%u,%u) %ux%u why: %s\n",
                                 fbp, fbw, t ? 1 : 0, t ? (t->vramNewer ? 1 : 0) : -1, t ? (t->gpuNewer ? 1 : 0) : -1, originX, originY, w, h,
                                 t ? t->staleWhy : "-");
                }
                return false;
            }
            const auto presentStart = std::chrono::steady_clock::now();
            const uint32_t sw = w * m_scale, sh = h * m_scale;
            ID3D11Texture2D *staging = stagingLocked(m_stagingPresent, sw, sh, DXGI_FORMAT_R8G8B8A8_UNORM);
            const D3D11_BOX box{originX * m_scale, originY * m_scale, 0, (originX + w) * m_scale, (originY + h) * m_scale, 1};
            m_ctx->CopySubresourceRegion(staging, 0, 0, 0, 0, t->tex.Get(), 0, &box);
            D3D11_MAPPED_SUBRESOURCE map{};
            if (FAILED(m_ctx->Map(staging, 0, D3D11_MAP_READ, 0, &map)))
                return false;
            m_statPresentWaitNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - presentStart).count();
            const uint32_t stride = std::max(kHostFrameWidth, sw);
            frame.pixels.resize(static_cast<size_t>(stride) * sh * 4u);
            for (uint32_t y = 0; y < sh; ++y)
            {
                const uint32_t *src = reinterpret_cast<const uint32_t *>(static_cast<const uint8_t *>(map.pData) + static_cast<size_t>(y) * map.RowPitch);
                uint32_t *dst = reinterpret_cast<uint32_t *>(frame.pixels.data() + static_cast<size_t>(y) * stride * 4u);
                for (uint32_t x = 0; x < sw; ++x)
                    dst[x] = src[x] | 0xFF000000u;
            }
            m_ctx->Unmap(staging, 0);
            frame.width = sw;
            frame.height = sh;
            frame.displayFbp = frame.sourceFbp = fbp;
            ++m_statPresents;
            m_statPresentNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - presentStart).count();
            return true;
        }

        void logStatsLocked()
        {
            if (!m_stats)
                return;
            const auto now = std::chrono::steady_clock::now();
            if (m_statStart.time_since_epoch().count() == 0)
                m_statStart = now;
            const double secs = std::chrono::duration<double>(now - m_statStart).count();
            if (secs < m_statInterval)
                return;
            std::fprintf(stderr,
                         "[gs-gpu] %.1fs presents=%llu prims=%llu cpuPrims=%llu batches=%llu draws=%llu vertexMB=%.1f palettes=%llu "
                         "decodes=%llu (%.1f ms) seeds=%llu (%.1f ms) downloads=%llu (%.1f ms) depthDownloads=%llu targets=%llu "
                         "textures=%zu texMB=%.0f texHits=%llu hash=%.1f ms hdUploads=%llu approxBlend=%llu approxMask=%llu open=%.1f ms flush=%.1f ms present=%.1f ms clutLoads=%llu clutChanges=%llu uploads=%llu uploadKB=%llu\n",
                         secs, (unsigned long long)m_statPresents, (unsigned long long)m_statPrims, (unsigned long long)m_statCpuPrims,
                         (unsigned long long)m_statBatches, (unsigned long long)m_statDraws, m_statVertexBytes / 1048576.0,
                         (unsigned long long)m_statPalettes, (unsigned long long)m_statDecodes, m_statDecodeNs / 1e6,
                         (unsigned long long)m_statSeeds, m_statSeedNs / 1e6, (unsigned long long)m_statDownloads, m_statDownloadNs / 1e6,
                         (unsigned long long)m_statDepthDownloads, (unsigned long long)m_statTargets, m_textures.size(),
                         m_texBytes / 1048576.0, (unsigned long long)m_statTexHits, m_statHashNs / 1e6, (unsigned long long)m_statHdUploads, (unsigned long long)m_statApproxBlend, (unsigned long long)m_statApproxMask,
                         m_statOpenNs / 1e6, m_statFlushNs / 1e6, m_statPresentNs / 1e6, (unsigned long long)m_statClutLoads,
                         (unsigned long long)m_statClutChanges, (unsigned long long)m_statUploads,
                         (unsigned long long)(m_statUploadBytes / 1024u));
            uint32_t smallOnGpu = 0;
            for (const auto &[key, s] : m_small)
                smallOnGpu += !s.readBack && m_presents - s.since >= kSmallProbation ? 1u : 0u;
            std::fprintf(stderr, "[gs-gpu] waits: download %.1f ms (pages needed %.0f%%) present %.1f ms keeper %.1f ms; rtTextures=%llu "
                                 "small=%u/%zu on GPU; cpu draws (type fbp/fbw: prims ms):",
                         m_statDownloadWaitNs / 1e6,
                         m_statDownloadPagesTotal ? 100.0 * m_statDownloadPagesNeeded / m_statDownloadPagesTotal : 0.0,
                         m_statPresentWaitNs / 1e6, m_statCpuWaitNs / 1e6, (unsigned long long)m_statRtTextures, smallOnGpu, m_small.size());
            std::vector<std::pair<uint64_t, uint64_t>> kinds; // ns, kind
            for (const auto &[kind, v] : m_cpuKindStats)
                kinds.push_back({v.second, kind});
            std::sort(kinds.rbegin(), kinds.rend());
            for (size_t i = 0; i < kinds.size() && i < 8u; ++i)
            {
                const auto &v = m_cpuKindStats[kinds[i].second];
                std::fprintf(stderr, " %u %u/%u: %llu %.1f;", static_cast<uint32_t>(kinds[i].second >> 32),
                             static_cast<uint32_t>((kinds[i].second >> 8) & 0xFFFFFFu), static_cast<uint32_t>(kinds[i].second & 0xFFu),
                             (unsigned long long)v.first, v.second / 1e6);
            }
            if (m_texCheck)
                std::fprintf(stderr, " texcheck %llu/%llu mismatched", (unsigned long long)m_statTexMismatches, (unsigned long long)m_statTexChecks);
            std::vector<std::pair<uint32_t, std::string>> why;
            for (const auto &[key, count] : m_downloadWhy)
                why.push_back({count, key});
            std::sort(why.rbegin(), why.rend());
            std::fprintf(stderr, " downloads by cause:");
            for (size_t i = 0; i < why.size() && i < 6u; ++i)
                std::fprintf(stderr, " %u x %s;", why[i].first, why[i].second.c_str());
            m_downloadWhy.clear();
            std::fprintf(stderr, "\n");
            m_cpuKindStats.clear();
            m_statDownloadWaitNs = m_statDownloadPagesNeeded = m_statDownloadPagesTotal = m_statPresentWaitNs = m_statCpuWaitNs = 0;
            m_statRtTextures = 0;
            m_statStart = now;
            m_statPresents = m_statPrims = m_statCpuPrims = m_statBatches = m_statDraws = m_statVertexBytes = m_statPalettes = 0;
            m_statDecodes = m_statDecodeNs = m_statSeeds = m_statSeedNs = m_statDownloads = m_statDownloadNs = m_statDepthDownloads = 0;
            m_statTargets = m_statHdUploads = m_statApproxBlend = m_statApproxMask = 0;
            m_statOpenNs = m_statFlushNs = m_statPresentNs = 0;
            m_statClutLoads = m_statClutChanges = m_statUploads = m_statUploadBytes = 0;
            m_statTexHits = m_statHashNs = 0;
        }

        static constexpr UINT kVertexBufferBytes = 16u << 20;

        mutable std::mutex m_mutex;
        std::unique_ptr<GSCpuBackend> m_cpu;
        uint8_t *m_vram = nullptr;
        uint32_t m_scale = 2;
        bool m_ok = false;
        bool m_stats = false;
        bool m_noCache = false; // ATV_GS_GPU_NOCACHE=1: decode every texture and palette at every batch (debug)
        bool m_allCpu = false;  // ATV_GS_GPU_ALLCPU=1: every draw goes to the CPU GS (debug)
        int m_traceFbp = -1;    // ATV_GS_GPU_TRACE
        uint64_t m_statPrimsTotal = 0;
        unsigned long long m_probeFrom = ~0ull, m_probeTo = 0; // ATV_GS_GPU_PROBE
        std::unordered_set<uint64_t> m_cpuKeys;
        uint32_t m_presentFallbackLogs = 0, m_colorSeedLogs = 0;
        double m_statInterval = 5.0;

        // ATV_GS_GPU_CAPTURE (see captureStartLocked).
        std::FILE *m_capFile = nullptr, *m_capLive = nullptr;
        bool m_capEnabled = false, m_capDone = false;
        uint64_t m_capSkip = 0, m_capRemaining = 2000000, m_capRacePrims = 0;
        uint32_t m_capVramDumps = 0;
        uint64_t m_capFallbacks = 0, m_capAfterFallbacks = 0; // ATV_GS_GPU_CAPTURE_AFTER_FALLBACKS: start after that many race presents from local memory
        char m_why[112] = "";                                 // the CPU-side write being done (stats logs)

        ComPtr<ID3D11Device> m_dev;
        ComPtr<ID3D11DeviceContext> m_ctx;
        ComPtr<ID3D11VertexShader> m_vs, m_vsFull;
        ComPtr<ID3D11PixelShader> m_ps, m_psSeedColor, m_psSeedDepth;
        ComPtr<ID3D11InputLayout> m_layout;
        ComPtr<ID3D11Buffer> m_vb, m_cbDraw, m_cbPalette, m_cbSeed;
        ComPtr<ID3D11RasterizerState> m_raster, m_rasterNoScissor;
        ComPtr<ID3D11SamplerState> m_hdSamplers[4];
        ComPtr<ID3D11Texture2D> m_stagingDownload, m_stagingDepth, m_stagingPresent;
        std::unordered_map<uint32_t, ComPtr<ID3D11BlendState>> m_blendStates;
        std::unordered_map<uint32_t, ComPtr<ID3D11DepthStencilState>> m_depthStates;
        bool m_boundStateValid = false;
        UINT m_vbOffset = 0;

        std::vector<std::unique_ptr<Target>> m_targets;
        std::array<uint32_t, kPages> m_pageVersion{};
        static constexpr uint64_t kTextureCacheBytes = 256ull << 20; // decoded textures (4 bytes a texel)
        std::unordered_map<uint64_t, TexSlot> m_texSlots;  // by TEX0
        std::unordered_map<uint64_t, TexEntry> m_textures; // by content (TexSlot::content)
        uint64_t m_texBytes = 0;
        std::unordered_map<const void *, HdEntry> m_hd;
        uint64_t m_hdBytes = 0;
        uint64_t m_useClock = 0;
        GSMem::TexturePageCache m_decodeCache;
        std::vector<uint32_t> m_decode;
        std::vector<uint32_t> m_seedPixels;

        std::array<uint16_t, 512> m_clut{};
        std::array<uint32_t, 2> m_clutCbp{};
        uint64_t m_clutVersion = 1;
        uint64_t m_paletteKey = 0;

        GSTransferCommand m_transfer{};
        PageSet m_transferDst;

        // The open batch.
        bool m_batchOpen = false;
        GSDrawState m_batchState{};
        const GSTexReplace::Image *m_batchHd = nullptr;
        std::vector<Vertex> m_vertices;
        Target *m_color = nullptr;
        Target *m_depth = nullptr;
        float m_zScale = 1.0f;
        DrawConstants m_constants{};
        Pass m_passes[2];
        uint32_t m_passCount = 0;
        bool m_blendEnable = false;
        D3D11_BLEND m_blendSrc = D3D11_BLEND_ONE, m_blendDst = D3D11_BLEND_ZERO;
        D3D11_BLEND_OP m_blendOp = D3D11_BLEND_OP_ADD;
        float m_blendFactor = 1.0f;
        D3D11_COMPARISON_FUNC m_depthFunc = D3D11_COMPARISON_ALWAYS;

        std::chrono::steady_clock::time_point m_statStart{};
        uint64_t m_statPresents = 0, m_statPrims = 0, m_statCpuPrims = 0, m_statBatches = 0, m_statDraws = 0, m_statVertexBytes = 0;
        uint64_t m_statPalettes = 0, m_statDecodes = 0, m_statDecodeNs = 0, m_statSeeds = 0, m_statSeedNs = 0;
        uint64_t m_statDownloads = 0, m_statDownloadNs = 0, m_statDepthDownloads = 0, m_statTargets = 0, m_statHdUploads = 0;
        uint64_t m_statApproxBlend = 0, m_statApproxMask = 0, m_statOpenNs = 0, m_statFlushNs = 0, m_statPresentNs = 0;
        uint64_t m_statClutLoads = 0, m_statClutChanges = 0, m_statUploads = 0, m_statUploadBytes = 0;
        uint64_t m_statTexHits = 0, m_statHashNs = 0;
        uint64_t m_statDownloadWaitNs = 0, m_statDownloadPagesNeeded = 0, m_statDownloadPagesTotal = 0, m_statPresentWaitNs = 0;
        uint64_t m_statCpuWaitNs = 0, m_statTexChecks = 0, m_statTexMismatches = 0, m_statRtTextures = 0;
        // Small buffers (smallTargetOnGpuLocked).
        struct SmallTarget
        {
            PageSet pages;      // its frame and the Z buffers drawn with it
            uint32_t zbpSeen = 0; // last of those ZBPs + 1
            uint64_t since = 0;   // m_presents when first drawn
            bool readBack = false;
        };
        static constexpr uint64_t kSmallProbation = 60u;
        std::unordered_map<uint32_t, SmallTarget> m_small; // by FBP | FBW << 9 | PSM << 15
        PageSet m_readbackPages;                           // ever read by a local -> host transfer
        uint64_t m_presents = 0;
        bool m_smallTargets = [] { const char *e = std::getenv("ATV_GS_GPU_SMALL"); return !e || e[0] != '0'; }();
        bool m_rtTextures = [] { const char *e = std::getenv("ATV_GS_GPU_RT_TEX"); return !e || e[0] != '0'; }();
        bool m_texCheck = std::getenv("ATV_GS_GPU_TEXCHECK") != nullptr;
        std::unordered_map<std::string, uint32_t> m_downloadWhy; // ATV_GS_GPU_STATS: downloads by target and cause
        bool m_cpuQueued = false; // the keeper got primitives since its last Flush (cpuIdleLocked)
        PageSet m_cpuQueuedPages; // the pages those primitives read or write
        std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> m_cpuKindStats; // see cpuDrawLocked
    };
}

std::unique_ptr<GSRasterBackend> gsCreateGpuBackend(uint32_t scale)
{
    if (scale == 0u)
    {
        const char *env = std::getenv("ATV_GS_SCALE");
        scale = env ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 2u;
    }
    scale = std::clamp<uint32_t>(scale, 1u, 8u);
    FpGuard fpGuard;
    auto backend = std::make_unique<GSGpuBackend>(scale);
    if (!backend->ok())
        return nullptr;
    return backend;
}

bool gsIsGpuBackend(const GSRasterBackend *backend)
{
    return dynamic_cast<const GSGpuBackend *>(backend) != nullptr;
}

void gsGpuRestoreClutState(GSRasterBackend *backend, const std::array<uint16_t, 512> &clut, const std::array<uint32_t, 2> &cbp)
{
    if (auto *gpu = dynamic_cast<GSGpuBackend *>(backend))
        gpu->RestoreClutState(clut, cbp);
}

// For GSThreadedBackend's worker (declared in gs_frontend.cpp): `backend` is a GPU backend.
void gsGpuSubmitMany(GSRasterBackend *backend, const GSPrimitiveBatch *const *batches, size_t count)
{
    static_cast<GSGpuBackend *>(backend)->SubmitMany(batches, count);
}

#else

std::unique_ptr<GSRasterBackend> gsCreateGpuBackend(uint32_t) { return nullptr; }
bool gsIsGpuBackend(const GSRasterBackend *) { return false; }
void gsGpuRestoreClutState(GSRasterBackend *, const std::array<uint16_t, 512> &, const std::array<uint32_t, 2> &) {}
void gsGpuSubmitMany(GSRasterBackend *, const GSPrimitiveBatch *const *, size_t) {}

#endif
