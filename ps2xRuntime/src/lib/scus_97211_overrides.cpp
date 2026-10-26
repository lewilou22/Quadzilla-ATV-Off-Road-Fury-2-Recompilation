#include "game_overrides.h"
#include "ps2_runtime.h"
#include "runtime/ee_scheduler.h"
#include "ps2_recompiled_functions.h"
#include "Kernel/Stubs/Pad.h"

#include <cerrno>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>
#include "runtime/gs/ps2_gif_arbiter.h"

extern bool g_ps2Vif1AbortStream;
extern uint32_t (*g_ps2Vif1ReadbackHook)(uint8_t *dst, uint32_t maxBytes);

extern uint32_t g_ps2GuestVsyncCounterAddr;
extern uint32_t g_ps2GuestVsyncCounterAddr2;
extern uint32_t g_ps2SkipSceGsSyncVWait;
extern uint32_t g_ps2Pc0ResumeRangeBegin;
extern uint32_t g_ps2Pc0ResumeRangeEnd;
extern uint32_t g_ps2Pc0ResumeTarget;
extern uint32_t g_ps2IdleResumePc;
extern uint32_t g_ps2IdleResumeA0;
extern uint32_t g_ps2HoldEeTimeslices;
extern void (*g_ps2Vif1IbitHook)(uint32_t vifcode); // ps2_vif1_interpreter.cpp
// ps2_memory.cpp, GS worker (ATV_VU1_THREAD): I-bit texture sends and GS downloads captured on
// the EE, and the worker's download handler.
extern void (*g_ps2Vif1PrecaptureHook)(PS2Memory::GifCapture &capture, std::vector<std::vector<uint8_t>> &tail);
extern bool (*g_ps2Vif1DownloadCaptureHook)(std::vector<uint8_t> &records);
extern bool (*g_ps2Vif1WorkerDownloadHook)(const uint8_t *record, uint32_t size, std::vector<uint8_t> &continuation);
extern uint64_t (*g_ps2GsCsrReadHook)(uint32_t pc, uint64_t value); // ps2_runtime.cpp
bool ps2IpuDecodeTextureRgb16(const uint8_t *src, size_t srcSize, uint32_t width, uint32_t height,
                              uint8_t *dst); // Kernel/Stubs/IPU.cpp
extern bool g_raceLoadStarted; // gates the race-only diagnostics (ATV_VU_FAST_VERIFY, GS traces)
extern uint32_t g_ps2TraceDispatchRemaining;
extern uint32_t g_ps2TraceRange0Lo, g_ps2TraceRange0Hi, g_ps2TraceRange1Lo, g_ps2TraceRange1Hi;

void atv2ArmWriteWatch(uint8_t *rdram, uint32_t guestAddr, uint32_t len);
void registerAtv2NetworkHle(PS2Runtime &runtime); // atv2_network_hle.cpp
void atvStartHostProfiler();                       // Kernel/Atv2HostProfiler.cpp
void atvInstallCrashTrace();                       // Kernel/Atv2HostProfiler.cpp
bool atv2BuildSuzukiDz400Files(const std::function<bool(const std::string &, std::vector<uint8_t> &)> &readDiscFile,
                               std::vector<std::pair<std::string, std::vector<uint8_t>>> &out); // Kernel/Atv2SuzukiDz400.cpp
namespace ps2_stubs
{
    bool registerCdOverlayFile(const std::string &ps2Path, const std::filesystem::path &hostPath,
                               uint32_t baseLbn); // Kernel/Stubs/CD.cpp
}

namespace
{
    // A scratch-context call cannot be resumed after a scheduler checkpoint: the callee
    // would be abandoned mid-way (e.g. with the pool allocator's free tree half-updated).
    struct HoldEeTimeslices
    {
        // Counted, not save/restore: holds from different guest threads can interleave.
        HoldEeTimeslices() { ++g_ps2HoldEeTimeslices; }
        ~HoldEeTimeslices() { --g_ps2HoldEeTimeslices; }
    };

    // Guest call from inside an override. If the scheduler can resume at returnPc, behave like
    // dispatchGuestBranch (false = thread suspended). Overrides of merged functions have no
    // resume entry at their return addresses, so a suspension there would strand the thread:
    // run the callee to completion instead.
    bool callGuestFromOverride(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t targetPc,
                               uint32_t sourcePc, uint32_t returnPc, PS2Runtime::GuestBranchKind kind,
                               const char *debugName)
    {
        if (runtime->hasFunction(returnPc) || !runtime->hasFunction(targetPc))
            return runtime->dispatchGuestBranch(rdram, ctx, targetPc, sourcePc, returnPc, kind, debugName);

        HoldEeTimeslices hold;
        const uint32_t entrySp = getRegU32(ctx, 29);
        ctx->pc = targetPc;
        runtime->lookupFunction(targetPc)(rdram, ctx, runtime);
        if (ctx->pc == targetPc && getRegU32(ctx, 29) == entrySp)
            ctx->pc = returnPc;
        static int s_continued = 0;
        if (ctx->pc != returnPc && s_continued < 40)
        {
            ++s_continued;
            std::cerr << std::hex << "[ATV2:OVRCALL] continue target=0x" << targetPc << " ret=0x" << returnPc
                      << " pc=0x" << ctx->pc << " sp=0x" << getRegU32(ctx, 29) << std::dec << std::endl;
        }
        for (uint32_t guard = 0; ctx->pc != returnPc; ++guard)
        {
            if (ctx->pc == 0u || guard >= 1000000u || runtime->isStopRequested() ||
                !runtime->hasFunction(ctx->pc))
            {
                std::cerr << "[SCUS_972.11] override call 0x" << std::hex << targetPc << " from 0x" << sourcePc
                          << " stopped at pc=0x" << ctx->pc << std::dec << std::endl;
                return false;
            }
            runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
        }
        return true;
    }

    // Screen object configuration parser (0x00204390..0x002043B4 prologue).
    // Rejoin the generated body at 0x002043B8.
    void screenConfig204390(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x150u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x20u), &ra, sizeof(ra));
        const auto s1 = GPR_VEC(ctx, 17);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &s1, 16);
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp), &s0, 16);
        SET_GPR_U64(ctx, 17, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 5));
        SET_GPR_U64(ctx, 4, GPR_U64(ctx, 5));
        SET_GPR_U32(ctx, 5, 0x00429E80u);
        SET_GPR_U32(ctx, 31, 0x002043B8u);
        ctx->pc = 0x00141810u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00141810u,
                0x002043B0u, 0x002043B8u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x002043B8u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Highlight_Icon configuration initializer (0x2AD6C0..0x2AD6E0 prologue).
    // Rejoin the generated body at 0x2AD6E4.
    void highlightInit2ad6c0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x1D0u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x40u), &ra, sizeof(ra));
        const auto s3 = GPR_VEC(ctx, 19);
        std::memcpy(getMemPtr(rdram, sp + 0x30u), &s3, 16);
        const auto s2 = GPR_VEC(ctx, 18);
        std::memcpy(getMemPtr(rdram, sp + 0x20u), &s2, 16);
        const auto s1 = GPR_VEC(ctx, 17);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &s1, 16);
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp), &s0, 16);
        SET_GPR_U64(ctx, 19, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 18, GPR_U64(ctx, 6));
        SET_GPR_U32(ctx, 31, 0x002AD6E4u);
        ctx->pc = 0x00234FB0u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00234FB0u,
                0x002AD6DCu, 0x002AD6E4u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x002AD6E4u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Animated_Icon configuration initializer (0x2AE600..0x2AE610 prologue).
    // Rejoin the generated body at 0x2AE614.
    void iconInit2ae600(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x20u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &ra, sizeof(ra));
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp), &s0, 16);
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 4));
        SET_GPR_U32(ctx, 31, 0x002AE614u);
        ctx->pc = 0x00239420u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00239420u,
                0x002AE60Cu, 0x002AE614u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x002AE614u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // UI element initializer (0x2AE370..0x2AE380 prologue + call 0x235170).
    // Rejoin the generated body at 0x2AE384.
    void uiInit2ae370(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x20u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &ra, sizeof(ra));
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp), &s0, 16);
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 4));
        SET_GPR_U32(ctx, 31, 0x002AE384u);
        ctx->pc = 0x00235170u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00235170u,
                0x002AE37Cu, 0x002AE384u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x002AE384u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // UI virtual method merged into sub_002AD5E0 like 0x2AE370: prologue + `jal 0x2350D0`,
    // then rejoins the generated body at return address 0x2AE2A4.
    void uiUpdate2ae290(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x20u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &ra, sizeof(ra));
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp), &s0, 16);
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 4));
        SET_GPR_U32(ctx, 31, 0x002AE2A4u);
        ctx->pc = 0x002350D0u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x002350D0u,
                0x002AE29Cu, 0x002AE2A4u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x002AE2A4u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Remaining methods of the UI class with vtable 0x45D7C0 that the analyzer merged into
    // sub_002AD5E0 (0x2AE220, 0x2AE280, 0x2AE3B0, 0x2AE400).

    // 0x2AE220: prologue, `jal 0x2B4BD0` on child +0x68 (active) or +0x6C (idle), then rejoin.
    void uiRelease2ae220(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t self = getRegU32(ctx, 4);
        const uint32_t sp = getRegU32(ctx, 29) - 0x20u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &ra, sizeof(ra));
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp), &s0, 16);
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 4));

        uint32_t v0 = READ32(self + 0x7Cu);
        bool active = false;
        if (v0 != 0u)
        {
            v0 = READ32(self + 0xC8u);
            active = (v0 != 0u);
        }
        SET_GPR_S32(ctx, 2, static_cast<int32_t>(v0));
        const uint32_t child = READ32(self + (active ? 0x68u : 0x6Cu));
        SET_GPR_S32(ctx, 4, static_cast<int32_t>(child));
        const uint32_t site = active ? 0x002AE244u : 0x002AE254u;
        const uint32_t ret = site + 8u;
        SET_GPR_U32(ctx, 31, ret);
        ctx->pc = 0x002B4BD0u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x002B4BD0u, site, ret,
                PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = ret;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // 0x2AE280: j 0x235050 (base-class method).
    void uiTail2ae280(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        ctx->pc = 0x00235050u;
    }

    // 0x2AE3B0: leaf timer step (f12 = dt), then tail-jumps to 0x1064B0.
    void uiTimer2ae3b0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t self = getRegU32(ctx, 4);
        const uint32_t active = READ32(self + 0x7Cu);
        SET_GPR_S32(ctx, 2, static_cast<int32_t>(active));
        if (active != 0u)
        {
            const float elapsed = std::bit_cast<float>(READ32(self + 0xC4u));
            const float limit = std::bit_cast<float>(READ32(self + 0xC0u));
            ctx->f[1] = elapsed;
            const bool running = elapsed < limit;
            if (running)
                ctx->fcr31 |= 0x00800000u;
            else
                ctx->fcr31 &= ~0x00800000u;
            ctx->f[0] = elapsed + ctx->f[12];
            if (running)
            {
                WRITE32(self + 0xC4u, std::bit_cast<uint32_t>(ctx->f[0]));
            }
            else
            {
                SET_GPR_U64(ctx, 2, 0u);
                WRITE32(self + 0x7Cu, 0u);
                WRITE32(self + 0xC4u, 0u);
            }
        }
        ctx->pc = 0x001064B0u;
    }

    // 0x2AE400: leaf. If entry `t0` (5th arg) is one of the 16 ids at +0x80, marks the widget
    // active and clears bit 3 / the +0xC link of every child node; otherwise resets +0x7C/+0xC4.
    void uiSelect2ae400(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t self = getRegU32(ctx, 4);
        const uint32_t key = getRegU32(ctx, 8);
        bool found = false;
        if (READ32(self + 0xC8u) != 0u)
        {
            for (uint32_t i = 0; i < 16u && !found; ++i)
            {
                if (READ32(self + 0x80u + i * 4u) != key)
                    continue;
                WRITE32(self + 0x7Cu, 1u);
                WRITE32(self + 0xC4u, 0u);
                const uint32_t table = READ32(READ32(0x00408398u) + 0x1Cu);
                const uint32_t group = READ32(table + (READ32(self + 0x70u) << 4) + 4u);
                found = true;
                if (group == 0u)
                    break;
                const int32_t count = static_cast<int32_t>(READ32(group + 0x58u));
                for (int32_t n = 0; n < count; ++n)
                {
                    const uint32_t node = READ32(group + static_cast<uint32_t>(n) * 4u + 0x5Cu);
                    if (READ32(node + 0x1Cu) == 0u)
                        continue;
                    for (uint32_t k = 0; k < 2u; ++k)
                    {
                        const uint32_t slot = READ32(node + 0x1Cu) + k * 4u;
                        const uint32_t item = READ32(slot + 0xCu);
                        if (item == 0u)
                            continue;
                        WRITE8(item + 4u, static_cast<uint8_t>(READ8(item + 4u) & 0xF7u));
                        WRITE32(item + 0xCu, 0u);
                        const uint32_t item2 = READ32(READ32(node + 0x1Cu) + k * 4u + 0xCu);
                        WRITE32(item2 + 0x10u, READ32(READ32(0x00404818u) + 0x234u));
                    }
                }
            }
        }
        else
        {
            WRITE32(self + 0x7Cu, 0u);
            WRITE32(self + 0xC4u, 0u);
        }
        ctx->pc = getRegU32(ctx, 31);
    }

    // Renderer virtual method merged into sub_00155850: prologue, one-time init of the
    // gp-0x7DD4 flag / 0x486F20, `jal 0x1A8CC0` on +0x189C, then rejoins at 0x155AEC.
    void renderBegin155ab0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0xE0u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x40u), &ra, sizeof(ra));
        for (uint32_t r = 0; r < 4u; ++r)
        {
            const auto s = GPR_VEC(ctx, 16 + r);
            std::memcpy(getMemPtr(rdram, sp + r * 0x10u), &s, 16);
        }
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 4));

        const uint32_t flagAddr = getRegU32(ctx, 28) - 0x7DD4u;
        int32_t v0 = static_cast<int8_t>(READ8(flagAddr));
        if (v0 == 0)
        {
            WRITE32(0x00486F20u, 0u);
            v0 = 1;
            WRITE8(flagAddr, 1u);
        }
        SET_GPR_S32(ctx, 2, v0);
        SET_GPR_S32(ctx, 4, static_cast<int32_t>(READ32(getRegU32(ctx, 16) + 0x189Cu)));
        SET_GPR_U32(ctx, 31, 0x00155AECu);
        ctx->pc = 0x001A8CC0u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x001A8CC0u,
                0x00155AE4u, 0x00155AECu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x00155AECu;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Child-factory virtual method (0x2D5880..0x2D58C4 prologue + allocator call).
    // Returns 0 unless a1 == 1; otherwise rejoins the generated body at 0x2D58C8.
    void childFactory2d5880(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (getRegU32(ctx, 5) != 1u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }
        const uint32_t sp = getRegU32(ctx, 29) - 0x40u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x30u), &ra, sizeof(ra));
        const auto s2 = GPR_VEC(ctx, 18);
        std::memcpy(getMemPtr(rdram, sp + 0x20u), &s2, 16);
        const auto s1 = GPR_VEC(ctx, 17);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &s1, 16);
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp), &s0, 16);
        SET_GPR_U64(ctx, 18, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 17, GPR_U64(ctx, 5));
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 6));
        SET_GPR_U64(ctx, 3, 1u);
        SET_GPR_U32(ctx, 4, 0x568u);
        SET_GPR_U32(ctx, 5, 0x00439660u);
        SET_GPR_U32(ctx, 6, 0x816u);
        SET_GPR_U32(ctx, 31, 0x002D58C8u);
        ctx->pc = 0x00103E30u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E30u,
                0x002D58C0u, 0x002D58C8u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x002D58C8u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Attach-child virtual method at 0x2343B0: if (child = *(a0+0x238)) { 0x105FA0(a0, child, -1); return 1; } return 0.
    // Rejoin the generated epilogue at 0x2343CC after the call.
    void attachChild2343b0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t child = READ32(getRegU32(ctx, 4) + 0x238u);
        if (child == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }
        const uint32_t sp = getRegU32(ctx, 29) - 0x10u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp), &ra, sizeof(ra));
        SET_GPR_U32(ctx, 5, child);
        SET_GPR_S32(ctx, 6, -1);
        SET_GPR_U64(ctx, 2, 0u);
        SET_GPR_U32(ctx, 31, 0x002343CCu);
        ctx->pc = 0x00105FA0u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00105FA0u,
                0x002343C4u, 0x002343CCu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x002343CCu;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Leaf virtual methods following 0x2343B0.
    void leafReturn2343e0(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        ctx->pc = getRegU32(ctx, 31);
    }

    void leafReturnOne234400(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        SET_GPR_U64(ctx, 2, 1u);
        ctx->pc = getRegU32(ctx, 31);
    }

    void tailNodeUpdate234410(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        ctx->pc = 0x001064B0u;
    }

    // Leaf getter at 0x2B5600: jr ra; lw v0, 0x44(a0).
    void getField44_2b5600(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SET_GPR_S32(ctx, 2, static_cast<int32_t>(READ32(getRegU32(ctx, 4) + 0x44u)));
        ctx->pc = getRegU32(ctx, 31);
    }

    void replaceIfMissing(PS2Runtime &runtime, uint32_t addr, PS2Runtime::RecompiledFunction fn)
    {
        if (!runtime.lookupFunction(addr))
            runtime.replaceFunction(addr, fn);
    }

    // Leaf accessor at 0x126FC0: swc1 f12, 0x24(a0); jr ra.
    void leafStoreF12_126fc0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const float f = ctx->f[12];
        std::memcpy(getMemPtr(rdram, getRegU32(ctx, 4) + 0x24u), &f, sizeof(f));
        ctx->pc = getRegU32(ctx, 31);
    }

    // Overlay_TextInput configuration initializer (0x2364C0..0x2364E0 prologue).
    // Rejoin the generated body at 0x2364E4.
    void textInputInit2364c0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x2B0u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x50u), &ra, sizeof(ra));
        const auto s3 = GPR_VEC(ctx, 19);
        std::memcpy(getMemPtr(rdram, sp + 0x40u), &s3, 16);
        const auto s2 = GPR_VEC(ctx, 18);
        std::memcpy(getMemPtr(rdram, sp + 0x30u), &s2, 16);
        const auto s1 = GPR_VEC(ctx, 17);
        std::memcpy(getMemPtr(rdram, sp + 0x20u), &s1, 16);
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &s0, 16);
        SET_GPR_U64(ctx, 19, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 18, GPR_U64(ctx, 6));
        SET_GPR_U32(ctx, 31, 0x002364E4u);
        ctx->pc = 0x00234FB0u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00234FB0u,
                0x002364DCu, 0x002364E4u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x002364E4u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Constructors 0x1C7AC0 / 0x1C7C00 (JALR targets from the factory at 0x34D288),
    // merged into generated sub_001C7A70 without re-entry cases. Both share the prologue
    //   addiu sp,-0x20; sd ra,0x10(sp); sq s0,0(sp); jal 0x1C7A40; daddu s0,a0
    // and rejoin the generated body right after the jal.
    template <uint32_t Entry>
    void ctorPrologue1c7a40(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        constexpr uint32_t kJal = Entry + 0xCu;
        constexpr uint32_t kRejoin = Entry + 0x14u;
        const uint32_t sp = getRegU32(ctx, 29) - 0x20u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &ra, sizeof(ra));
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp + 0x00u), &s0, 16);
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 4));
        SET_GPR_U32(ctx, 31, kRejoin);
        ctx->pc = 0x001C7A40u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x001C7A40u,
                kJal, kRejoin, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = kRejoin;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Slider configuration initializer (0x2AE950..0x2AE968 prologue).
    // Rejoin the generated body at 0x2AE96C.
    void sliderInit2ae950(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x80u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x20u), &ra, sizeof(ra));
        const auto s1 = GPR_VEC(ctx, 17);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &s1, 16);
        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp), &s0, 16);
        SET_GPR_U64(ctx, 17, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 6));
        SET_GPR_U32(ctx, 31, 0x002AE96Cu);
        ctx->pc = 0x00239420u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00239420u,
                0x002AE964u, 0x002AE96Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x002AE96Cu;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Resource read wrapper verified by PCSX2: pass its allocated word buffer
    // and byte count to the archive reader, preserving the file handle in a1.
    void resourceRead1ce020(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t object = getRegU32(ctx, 4);
        uint32_t count = 0, buffer = 0, archive = 0;
        std::memcpy(&count, getMemPtr(rdram, object + 0x10u), 4);
        std::memcpy(&buffer, getMemPtr(rdram, object + 0x14u), 4);
        std::memcpy(&archive, getMemPtr(rdram, 0x004975B0u), 4);
        SET_GPR_S32(ctx, 2, static_cast<int32_t>(count));
        SET_GPR_S32(ctx, 1, 0x00490000);
        SET_GPR_S32(ctx, 6, static_cast<int32_t>(buffer));
        SET_GPR_S32(ctx, 8, 1);
        SET_GPR_S32(ctx, 4, static_cast<int32_t>(archive));
        SET_GPR_S32(ctx, 7, static_cast<int32_t>(count << 2));
        ctx->pc = 0x001806C0u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Doubly-linked list insert tail @ 0x1749B0:
    // a0 = list (header at a0: +0 head, +4 tail, +8 ?, +c count)
    // a1 = item (+0xa8 next, +0xac prev)
    void listInsertTail1749b0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t list = GPR_U32(ctx, 4);
        const uint32_t item = GPR_U32(ctx, 5);
        if (list != 0u && item != 0u)
        {
            const uint32_t head = READ32(list);
            if (head == 0u)
            {
                WRITE32(list, item);       // list->head = item
                WRITE32(list + 4u, item);  // list->tail = item
                WRITE32(item + 0xACu, 0u); // item->prev = NULL
                WRITE32(item + 0xA8u, 0u); // item->next = NULL
            }
            else
            {
                const uint32_t tail = READ32(list + 4u);
                if (tail != 0u)
                    WRITE32(tail + 0xA8u, item); // tail->next = item
                WRITE32(item + 0xACu, tail);     // item->prev = tail
                WRITE32(list + 4u, item);        // list->tail = item
                WRITE32(item + 0xA8u, 0u);       // item->next = NULL
            }
            const uint32_t count = READ32(list + 0xCu);
            WRITE32(list + 0xCu, count + 1u);
        }
        ctx->pc = GPR_U32(ctx, 31);
    }

    // Resource list insert @ 0x177AE0:
    // a0 = resource manager (object at *(a0 + 0x30))
    // a1 = item
    // Forwards to list at (object + 0x10), calls listInsertTail1749b0.
    void resourceForward177ae0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t mgr = GPR_U32(ctx, 4);
        const uint32_t object = (mgr != 0u) ? READ32(mgr + 0x30u) : 0u;
        SET_GPR_U32(ctx, 4, object + 0x10u);
        listInsertTail1749b0(rdram, ctx, runtime);
    }

    // Resource list remove @ 0x177AF0:
    // a0 = resource manager (object at *(a0 + 0x30))
    // a1 = item
    // Forwards to list at (object + 0x10), calls sub_00174A00_0x174a00.
    void resourceRemove177af0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t mgr = GPR_U32(ctx, 4);
        const uint32_t object = (mgr != 0u) ? READ32(mgr + 0x30u) : 0u;
        SET_GPR_U32(ctx, 4, object + 0x10u);
        ctx->pc = 0x00174A00u;
        sub_00174A00_0x174a00(rdram, ctx, runtime);
    }

    // Forwarder @ 0x1D0890: a0 += 0x10; j 0x1749B0
    void forward1d0890(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SET_GPR_U32(ctx, 4, GPR_U32(ctx, 4) + 0x10u);
        listInsertTail1749b0(rdram, ctx, runtime);
    }

    // Forwarder @ 0x1D08A0: a0 += 0x10; j 0x174A00
    void forward1d08a0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SET_GPR_U32(ctx, 4, GPR_U32(ctx, 4) + 0x10u);
        ctx->pc = 0x00174A00u;
        sub_00174A00_0x174a00(rdram, ctx, runtime);
    }

    // Registry flags wrapper: preserve existing flags before the tail call.
    void registryFlags1d2fa0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t records = 0, flags = 0;
        std::memcpy(&records, getMemPtr(rdram, getRegU32(ctx, 4) + 0x1Cu), 4);
        SET_GPR_S32(ctx, 3, static_cast<int32_t>(records));
        std::memcpy(&flags, getMemPtr(rdram, records + (getRegU32(ctx, 5) << 4) + 8u), 4);
        SET_GPR_S32(ctx, 2, static_cast<int32_t>(flags));
        SET_GPR_U64(ctx, 6, GPR_U64(ctx, 6) | GPR_U64(ctx, 2));
        ctx->pc = 0x001AE5F0u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Name registry entry, verified with both MCPs. Rejoin the generated
    // search loop or its first allocation return, retaining the guest frame.
    void nameRegistry1ae800(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x60u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x50u), &ra, sizeof(ra));
        for (unsigned reg = 16; reg <= 20; ++reg)
        {
            const auto saved = GPR_VEC(ctx, reg);
            std::memcpy(getMemPtr(rdram, sp + (reg - 16u) * 0x10u), &saved, 16);
        }
        SET_GPR_U64(ctx, 20, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 19, GPR_U64(ctx, 5));
        SET_GPR_U64(ctx, 18, GPR_U64(ctx, 6));
        SET_GPR_U64(ctx, 17, 0);
        SET_GPR_U64(ctx, 16, 0);
        int32_t count = 0;
        std::memcpy(&count, getMemPtr(rdram, getRegU32(ctx, 4) + 0x10u), sizeof(count));
        SET_GPR_U64(ctx, 2, count > 0 ? 1 : 0);
        if (count > 0)
        {
            ctx->pc = 0x001AE834u;
        }
        else
        {
            SET_GPR_U32(ctx, 31, 0x001AE89Cu);
            ctx->pc = 0x001AE4C0u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x001AE4C0u,
                    0x001AE894u, 0x001AE89Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;
            ctx->pc = 0x001AE89Cu;
        }
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // PCSX2 native decode: ld v0,0(a0); daddiu v0,1; jr ra; sd v0,0(a0).
    void addRef107460(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        uint64_t count = 0;
        uint8_t *object = getMemPtr(rdram, getRegU32(ctx, 4));
        std::memcpy(&count, object, sizeof(count));
        ++count;
        SET_GPR_U64(ctx, 2, count);
        std::memcpy(object, &count, sizeof(count));
        ctx->pc = getRegU32(ctx, 31);
    }

    // Vertex attribute callback: restore the missed prologue and reuse the
    // generated position/color/UV updates beginning at 0x3425B0.
    void vertexUpdate342590(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x20u), &ra, sizeof(ra));
        for (unsigned reg = 16; reg <= 17; ++reg)
        {
            const auto saved = GPR_VEC(ctx, reg);
            std::memcpy(getMemPtr(rdram, sp + (reg - 16u) * 0x10u), &saved, 16);
        }
        SET_GPR_U64(ctx, 17, GPR_U64(ctx, 4));
        uint32_t object = 0;
        std::memcpy(&object, getMemPtr(rdram, getRegU32(ctx, 4) + 0x30u), sizeof(object));
        SET_GPR_S32(ctx, 4, static_cast<int32_t>(object));
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 6));
        SET_GPR_U32(ctx, 31, 0x003425B0u);
        ctx->pc = 0x003423F0u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x003423F0u,
                0x003425A8u, 0x003425B0u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x003425B0u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // MCP-verified wrapper: call 0x1C0930, then return 1. Reuse its
    // generated epilogue so allocator/scheduler yields retain the guest frame.
    void resourceHook1bd4f0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x10u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp), &ra, sizeof(ra));
        SET_GPR_U32(ctx, 31, 0x001BD500u);
        ctx->pc = 0x001C0930u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x001C0930u,
                0x001BD4F8u, 0x001BD500u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x001BD500u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Ghydra and PCSX2 agree on 0x14F610..0x14F658. Only this prologue
    // lacks an entry; the generated body already resumes at 0x14F65C.
    void resourceInit14f610(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0x70u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x60u), &ra, sizeof(ra));
        for (unsigned reg = 16; reg <= 21; ++reg)
        {
            const auto saved = GPR_VEC(ctx, reg);
            std::memcpy(getMemPtr(rdram, sp + (reg - 16u) * 0x10u), &saved, 16);
        }
        SET_GPR_U64(ctx, 21, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 20, GPR_U64(ctx, 5));
        const uint32_t count = getRegU32(ctx, 6);
        std::memcpy(getMemPtr(rdram, getRegU32(ctx, 4) + 0x34u), &count, sizeof(count));
        SET_GPR_S32(ctx, 3, static_cast<int32_t>(count));
        SET_GPR_S32(ctx, 2, static_cast<int32_t>(count * 33u));
        SET_GPR_S32(ctx, 4, static_cast<int32_t>(count * 0x108u));
        SET_GPR_U32(ctx, 5, 0x0041F068u);
        SET_GPR_U32(ctx, 6, 0x524u);
        SET_GPR_U32(ctx, 31, 0x0014F65Cu);
        ctx->pc = 0x00103C20u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103C20u,
                0x0014F654u, 0x0014F65Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = 0x0014F65Cu;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    void ctor44c100(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)ctx;
        (void)runtime;

        uint8_t *dst = getMemPtr(rdram, 0x00461750u);
        if (dst == nullptr)
        {
            std::cerr << "[SCUS_972.11] ctor44c100: cannot resolve guest dest 0x461750" << std::endl;
            return;
        }

        // FUN_0044c100 writes 4x128-bit const quads {w,x,y,z} at 0x461750..0x46177F:
        // {0,0,0,1} {1,0,0,1} {0,1,0,1} {0,0,1,1} (float 1.0 = 0x3F800000).
        static const uint32_t kMatrixQuads[16] = {
            0x00000000u, 0x00000000u, 0x00000000u, 0x3F800000u,
            0x3F800000u, 0x00000000u, 0x00000000u, 0x3F800000u,
            0x00000000u, 0x3F800000u, 0x00000000u, 0x3F800000u,
            0x00000000u, 0x00000000u, 0x3F800000u, 0x3F800000u,
        };
        std::memcpy(dst, kMatrixQuads, sizeof(kMatrixQuads));
    }

    // default_new_handler__3stdFv @ 0x34d4b0 — called from __nw__FUi (operator new)
    // when allocation fails; address stored in data table at 0x4107A0.
    // Generated code sets up frame, loads v0=0x46ED0, calls 34F210 + 34E090, loops, returns via jr $ra.
    // Runtime policy = SkipCallDebug; we return 1 (non-null handler -> retry alloc) and advance PC via ra.
    void default_new_handler_34d4b0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;

        // Return non-zero (valid handler) in v0 ($2) to indicate "retry allocation"
        SET_GPR_U32(ctx, 2, 1u);

        // CRITICAL: advance PC via return address (ra = $31), otherwise infinite loop
        const uint32_t entryPc = ctx->pc;
        if (ctx->pc == entryPc)
        {
            ctx->pc = getRegU32(ctx, 31);
        }
    }

    // Session-lock enter @ 0x341e60 (vtable slot [8], JALR'd from the stream/IO ctor
    // path: 0x341da0 -> ... -> 0x341f90). Analyzer's jal heuristic missed the boundary.
    // Object layout (u32 offsets): +0x4 mutex sema, +0x8 holder id (-1 = free),
    // +0xC refcount, +0x10 wait sema (WaitsSema target), +0x14 waiter count.
    // Helper twins: func_0x00341e40 = WaitSema(+0x4), func_0x00341e50 = SignalSema(+0x4).
    // Retry path: release mutex, WaitSema(+0x10), re-acquire, retry.
    void sessionLockEnter341e60(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t objAddr = getRegU32(ctx, 4);
        const uint32_t newId = getRegU32(ctx, 5);

        uint32_t *obj = reinterpret_cast<uint32_t *>(getMemPtr(rdram, objAddr));
        if (obj == nullptr)
        {
            std::cerr << "[SCUS_972.11] sessionLockEnter341e60: cannot resolve guest obj 0x"
                      << std::hex << objAddr << std::dec << std::endl;
            return;
        }

        static unsigned enterReports = 0;
        if (enterReports++ < 10)
        {
            std::cerr << "[sessionLockEnter341e60] obj=0x" << std::hex << objAddr
                      << " newId=0x" << newId << " mutex=" << obj[1] << " holder=0x" << obj[2]
                      << " refcount=" << obj[3] << " waitSema=" << obj[4] << " waiters=" << obj[5]
                      << " ra=0x" << getRegU32(ctx, 31) << " sp=0x" << getRegU32(ctx, 29)
                      << " pc=0x" << ctx->pc << std::dec << std::endl;
        }

        EeScheduler &ee = runtime->eeScheduler();
        ee.bindMainContextForSyscall(*ctx, rdram);

        const int mutexSema = static_cast<int>(obj[1]); // +0x4
        const int waitSema = static_cast<int>(obj[4]);  // +0x10

        ee.waitSemaphore(mutexSema);
        for (;;)
        {
            const uint32_t holder = obj[2]; // +0x8
            if (holder == 0xFFFFFFFFu)
            {
                obj[2] = newId;
                ++obj[3]; // +0xC refcount
                break;
            }
            if (holder == newId)
            {
                ++obj[3];
                break;
            }

            // Held by another session: release the mutex, block on the per-object
            // wait semaphore until the holder signals, re-acquire, retry.
            static unsigned contentionReports = 0;
            if (contentionReports++ < 4)
            {
                std::cerr << "[SCUS_972.11] lock-contention obj=0x" << std::hex << objAddr
                          << " holder=0x" << holder << " requested=0x" << newId
                          << " refcount=0x" << obj[3] << " ra=0x" << getRegU32(ctx, 31)
                          << " sp=0x" << getRegU32(ctx, 29) << std::dec << std::endl;
            }
            ++obj[5]; // +0x14 waiter count
            ee.signalSemaphore(mutexSema, false);
            ee.waitSemaphore(waitSema);
            ee.waitSemaphore(mutexSema);
            --obj[5];
        }
        ee.signalSemaphore(mutexSema, false);
        ctx->pc = getRegU32(ctx, 31);
    }

    // Session-lock leave @ 0x341f30 (vtable slot [c], twin of 0x341e60; JALR'd from
    // 0x341ff8 after the enter wrapper FUN_00341f90). Also missed by the analyzer.
    // Decoded: WaitSema(+0x4); --refcount(+0xC); if 0: if waiters(+0x14)>0
    // SignalSema(+0x10); holder(+0x8)=-1; SignalSema(+0x4).
    void sessionLockLeave341f30(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t objAddr = getRegU32(ctx, 4);

        uint32_t *obj = reinterpret_cast<uint32_t *>(getMemPtr(rdram, objAddr));
        if (obj == nullptr)
        {
            std::cerr << "[SCUS_972.11] sessionLockLeave341f30: cannot resolve guest obj 0x"
                      << std::hex << objAddr << std::dec << std::endl;
            return;
        }

        static unsigned leaveReports = 0;
        if (leaveReports++ < 10)
        {
            std::cerr << "[sessionLockLeave341f30] obj=0x" << std::hex << objAddr
                      << " mutex=" << obj[1] << " holder=0x" << obj[2]
                      << " refcount=" << obj[3] << " waitSema=" << obj[4] << " waiters=" << obj[5]
                      << " ra=0x" << getRegU32(ctx, 31) << " sp=0x" << getRegU32(ctx, 29)
                      << " pc=0x" << ctx->pc << std::dec << std::endl;
        }

        EeScheduler &ee = runtime->eeScheduler();
        ee.bindMainContextForSyscall(*ctx, rdram);

        const int mutexSema = static_cast<int>(obj[1]); // +0x4
        const int waitSema = static_cast<int>(obj[4]);  // +0x10

        ee.waitSemaphore(mutexSema);
        const uint32_t remaining = obj[3] - 1u; // +0xC refcount
        obj[3] = remaining;
        if (remaining == 0u)
        {
            if (obj[5] > 0u) // +0x14 waiter count
            {
                ee.signalSemaphore(waitSema, false);
            }
            obj[2] = 0xFFFFFFFFu; // +0x8 holder free
        }
        ee.signalSemaphore(mutexSema, false);
        ctx->pc = getRegU32(ctx, 31);
    }

    // Base vector ctor @ 0x11a3d0: `sqc2 vf00,(a0); jr ra; dmove v0,a0` — writes the
    // VF00 {0,0,0,1} 128-bit quad to the object and returns it. Called in a loop by
    // array ctor FUN_0034d0d0 (16 elements, stride 0x10, base 0x461d30). Analyzer
    // merged the one-instruction body into the following function.
    void ctor11a3d0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t objAddr = getRegU32(ctx, 4);

        uint8_t *dst = getMemPtr(rdram, objAddr);
        if (dst == nullptr)
        {
            std::cerr << "[SCUS_972.11] ctor11a3d0: cannot resolve guest dest 0x"
                      << std::hex << objAddr << std::dec << std::endl;
            return;
        }

        // VF00 = {0, 0, 0, 1.0f} (0x3F800000), 16 bytes.
        static const uint32_t kVf0Quad[4] = {0u, 0u, 0u, 0x3F800000u};
        std::memcpy(dst, kVf0Quad, sizeof(kVf0Quad));

        SET_GPR_U32(ctx, 2, objAddr); // dmove v0, a0
        ctx->pc = getRegU32(ctx, 31);
    }

    // 4×VF00-quad ctor @ 0x16b5e0: writes VF00 {0,0,0,1} to obj+0x20..+0x5F (4 quads).
    // Called by array ctor FUN_0034d0d0 (base 0x4891b0, stride 0x80).
    void ctor16b5e0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t objAddr = getRegU32(ctx, 4);

        uint8_t *dst = getMemPtr(rdram, objAddr);
        if (dst == nullptr)
        {
            std::cerr << "[SCUS_972.11] ctor16b5e0: cannot resolve guest dest 0x"
                      << std::hex << objAddr << std::dec << std::endl;
            return;
        }

        static const uint32_t kVf0Quad[4] = {0u, 0u, 0u, 0x3F800000u};
        for (int i = 0; i < 4; ++i)
        {
            std::memcpy(dst + 0x20 + i * 0x10, kVf0Quad, sizeof(kVf0Quad));
        }

        SET_GPR_U32(ctx, 2, objAddr); // dmove v0, a0
        ctx->pc = getRegU32(ctx, 31);
    }

    // 2×VF00-quad ctor @ 0x2d1f10: `sqc2 vf00,(a0); addiu v0,a0,0x10; sqc2 vf00,(v0); jr ra; dmove v0,a0`
    // Writes VF00 {0,0,0,1} to obj+0x00 and obj+0x10 (2 quads, 32 bytes), returns obj in v0.
    void ctor2d1f10(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t objAddr = getRegU32(ctx, 4);

        uint8_t *dst = getMemPtr(rdram, objAddr);
        if (dst == nullptr)
        {
            std::cerr << "[SCUS_972.11] ctor2d1f10: cannot resolve guest dest 0x"
                      << std::hex << objAddr << std::dec << std::endl;
            return;
        }

        static const uint32_t kVf0Quad[4] = {0u, 0u, 0u, 0x3F800000u};
        std::memcpy(dst, kVf0Quad, sizeof(kVf0Quad));
        std::memcpy(dst + 0x10u, kVf0Quad, sizeof(kVf0Quad));

        SET_GPR_U32(ctx, 2, objAddr); // dmove v0, a0
        ctx->pc = getRegU32(ctx, 31);
    }

    // Element ctor @ 0x137a10: zeroes +0x0/+0x8/+0xC/+0x10, clears bit0 of byte+0x4
    // (flag arg is discarded — reads $zero — per original MIPS), returns a0.
    void ctor137a10(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t objAddr = getRegU32(ctx, 4);

        uint8_t *p = getMemPtr(rdram, objAddr);
        if (p == nullptr)
        {
            std::cerr << "[SCUS_972.11] ctor137a10: cannot resolve guest dest 0x"
                      << std::hex << objAddr << std::dec << std::endl;
            return;
        }

        std::memset(p, 0, 4);      // +0x0
        p[4] = static_cast<uint8_t>(p[4] & 0xFEu); // +0x4 bit0 clear
        std::memset(p + 8, 0, 12); // +0x8..+0x13

        SET_GPR_U32(ctx, 2, objAddr); // dmove v0, a0
        ctx->pc = getRegU32(ctx, 31);
    }

    // No-op accessor @ 0x153970: `jr ra; dmove v0, a2` — returns its third argument.
    // Virtual no-op stub missed by the analyzer (merged into the next function).
    void noopRetA2_153970(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        SET_GPR_U32(ctx, 2, getRegU32(ctx, 6)); // v0 = a2
        ctx->pc = getRegU32(ctx, 31);
    }

    // Element ctor @ 0x331690: calls 0x325580(obj + 4) which zeroes +4..+7, clears bit 0 of +8,
    // zeroes +12..+23 (24-byte element struct). Merged into sub_00331400 without a table slot.
    void ctor331690(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t objAddr = getRegU32(ctx, 4);

        uint8_t *p = getMemPtr(rdram, objAddr + 4u);
        if (p == nullptr)
        {
            std::cerr << "[SCUS_972.11] ctor331690: cannot resolve guest dest 0x"
                      << std::hex << objAddr << std::dec << std::endl;
            return;
        }

        std::memset(p, 0, 4);                      // obj+4..obj+7
        p[4] = static_cast<uint8_t>(p[4] & 0xFEu); // obj+8 bit0 clear
        std::memset(p + 8, 0, 12);                 // obj+12..obj+23

        SET_GPR_U32(ctx, 2, objAddr);              // v0 = a0 (returns object pointer)
        ctx->pc = getRegU32(ctx, 31);
    }

    // Element ctor @ 0x1C5430 (0x34-byte mesh-part entry, JALR target of the array ctor at
    // 0x34D200): zeroes +0x00..+0x33 and sets +0x10 = -1. Merged into sub_001C5060 without a
    // table slot, so the array ctor skipped it and left garbage vertex-list counts.
    void ctor1c5430(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        const uint32_t objAddr = getRegU32(ctx, 4);
        for (uint32_t off = 0; off < 0x34u; off += 4u)
            WRITE32(objAddr + off, 0u);
        WRITE32(objAddr + 0x10u, 0xFFFFFFFFu);
        SET_GPR_S32(ctx, 3, -1);
        SET_GPR_U32(ctx, 2, objAddr);
        ctx->pc = getRegU32(ctx, 31);
    }

    // Element ctor @ 0x1C61E0 (20-byte vertex-list entry built by the array ctor in 0x1C5B50):
    // zeroes +0x00..+0x13. Merged into sub_001C5B50 without a table slot; the count at +0 is
    // incremented afterwards, so skipping it left runaway counts that stomped low RAM.
    void ctor1c61e0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        const uint32_t objAddr = getRegU32(ctx, 4);
        for (uint32_t off = 0; off < 0x14u; off += 4u)
            WRITE32(objAddr + off, 0u);
        SET_GPR_U32(ctx, 2, objAddr);
        ctx->pc = getRegU32(ctx, 31);
    }

    // Array-factory (0x34D288) element ctors merged into neighbouring functions without table slots.
    void ctor1af900(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        const uint32_t objAddr = getRegU32(ctx, 4);
        for (uint32_t off = 0; off < 0x0Cu; off += 4u)
            WRITE32(objAddr + off, 0u);
        SET_GPR_U32(ctx, 2, objAddr);
        ctx->pc = getRegU32(ctx, 31);
    }

    void ctor11a3b0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        const uint32_t objAddr = getRegU32(ctx, 4);
        for (uint32_t off = 0x0Cu; off <= 0x20u; off += 4u)
            WRITE32(objAddr + off, 0u);
        SET_GPR_U32(ctx, 2, objAddr);
        ctx->pc = getRegU32(ctx, 31);
    }

    void ctorReturnThis(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        SET_GPR_U32(ctx, 2, getRegU32(ctx, 4));
        ctx->pc = getRegU32(ctx, 31);
    }

    // Element ctor @ 0x321BD0 (JALR target of the array ctor at 0x34D138): calls 0x325580(this),
    // returns this. Merged into sub_00320DB0; frame matches the original.
    void ctor321bd0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x20u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x10u, GPR_U64(ctx, 31));
        WRITE128(sp, GPR_VEC(ctx, 16));
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 4));

        SET_GPR_U32(ctx, 31, 0x00321BE4u);
        ctx->pc = 0x00325580u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00325580u, 0x00321BDCu, 0x00321BE4u,
                                   PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;

        SET_GPR_U32(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x10u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x20u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual method @ 0x206690 (JALR from 0x206B7C): calls 0x2064F0, returns 1.
    // Merged into sub_00206610; frame matches the original.
    void method206690(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x10u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp, GPR_U64(ctx, 31));

        SET_GPR_U32(ctx, 31, 0x002066A0u);
        ctx->pc = 0x002064F0u;
        if (!callGuestFromOverride(runtime, rdram, ctx, 0x002064F0u, 0x00206698u, 0x002066A0u,
                                   PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;

        const uint64_t retRa = READ64(sp);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_S32(ctx, 2, 1);
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x10u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual method @ 0x2B4710 (JALR from 0x1BEBC4), merged into sub_002B4680. Leaf.
    void method2b4710(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        const uint32_t self = getRegU32(ctx, 4);
        const uint32_t arg = getRegU32(ctx, 5);
        const uint32_t mode = READ32(self + 0x44u);
        SET_GPR_U32(ctx, 6, mode);
        uint32_t result = 1u;
        if (mode == 2u)
        {
            WRITE32(READ32(arg + 0x0Cu) + 0x0Cu, 0u);
            const uint32_t rec = READ32(arg + 0x0Cu);
            WRITE8(rec + 4u, static_cast<uint8_t>(READ8(rec + 4u) & 0xF7u));
            WRITE32(READ32(arg + 0x0Cu) + 0x10u, 0xFFFFFFFFu);
            WRITE32(self + 0x44u, 0u);
        }
        else if (mode == 1u)
        {
            const uint32_t value = READ32(READ32(0x00404818u) + 0x234u);
            WRITE32(READ32(arg + 0x0Cu) + 0x10u, value);
            result = 0u;
        }
        SET_GPR_U32(ctx, 2, result);
        ctx->pc = getRegU32(ctx, 31);
    }

    // Merged functions whose first JAL return address has a re-entry case in the generated host
    // function: run the prologue here, make the first call, then resume the generated body there.
    void resumeMergedAfterCall(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx, uint32_t callee,
                               uint32_t callSite)
    {
        const uint32_t returnPc = callSite + 8u;
        SET_GPR_U32(ctx, 31, returnPc);
        ctx->pc = callee;
        if (!callGuestFromOverride(runtime, rdram, ctx, callee, callSite, returnPc,
                                   PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;
        ctx->pc = returnPc;
        runtime->lookupFunction(returnPc)(rdram, ctx, runtime);
    }

    // Property-load virtual @ 0x23BFC0 (JALR from 0x23CD94), merged into sub_0023BF80.
    void loadProps23bfc0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x150u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x30u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x20u, GPR_VEC(ctx, 18));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        SET_GPR_U64(ctx, 18, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 17, GPR_U64(ctx, 5));
        WRITE128(sp, GPR_VEC(ctx, 16));
        resumeMergedAfterCall(runtime, rdram, ctx, 0x0023BCC0u, 0x0023BFD8u);
    }

    // Property-load virtual @ 0x240EB0 (JALR from 0x2410FC), merged into sub_00240DD0.
    void loadProps240eb0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        SET_GPR_U64(ctx, 6, 0u);
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));
        SET_GPR_U64(ctx, 17, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 5));
        SET_GPR_U64(ctx, 7, GPR_U64(ctx, 17));
        SET_GPR_U64(ctx, 4, GPR_U64(ctx, 16));
        SET_GPR_U32(ctx, 5, 0x0042EB60u);
        resumeMergedAfterCall(runtime, rdram, ctx, 0x001421B0u, 0x00240ED8u);
    }

    // Property-load virtual @ 0x1B8150 (JALR from 0x1BF680), merged into sub_001B7FC0.
    void loadProps1b8150(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x2A0u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x90u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x80u, GPR_VEC(ctx, 19));
        WRITE128(sp + 0x70u, GPR_VEC(ctx, 18));
        SET_GPR_U64(ctx, 19, GPR_U64(ctx, 4));
        WRITE128(sp + 0x60u, GPR_VEC(ctx, 17));
        SET_GPR_U64(ctx, 18, GPR_U64(ctx, 5));
        WRITE128(sp + 0x50u, GPR_VEC(ctx, 16));
        resumeMergedAfterCall(runtime, rdram, ctx, 0x001BF160u, 0x001B816Cu);
    }

    // Property-load virtual @ 0x1B8790 (JALR from 0x1BF854), merged into sub_001B86D0.
    void loadProps1b8790(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0xE0u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x30u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x20u, GPR_VEC(ctx, 18));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        SET_GPR_U64(ctx, 18, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 17, GPR_U64(ctx, 5));
        WRITE128(sp, GPR_VEC(ctx, 16));
        resumeMergedAfterCall(runtime, rdram, ctx, 0x001BF1E0u, 0x001B87A8u);
    }

    // Array-factory element ctor @ 0x2CA1F0, merged into sub_002C9FF0.
    void ctor2ca1f0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x20u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp, GPR_U64(ctx, 31));
        WRITE32(sp + 0x1Cu, GPR_U32(ctx, 4));
        resumeMergedAfterCall(runtime, rdram, ctx, 0x002184B0u, 0x002CA1FCu);
    }

    // Element dtor @ 0x1C5170 (pair of ctor1c5430): frees +0x14..+0x2C, frees the +0x30 block
    // (header at -0x10), deletes self when (int16)a1 > 0. Same frame as the original so a
    // suspension inside a free resumes correctly in sub_001C5060.
    void dtor1c5170(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        SET_GPR_U64(ctx, 17, GPR_U64(ctx, 4));
        SET_GPR_U64(ctx, 16, GPR_U64(ctx, 5));
        const int16_t delFlag = static_cast<int16_t>(GPR_U32(ctx, 5));

        if (thisPtr != 0u)
        {
            static constexpr uint32_t kFreeSites[7] = {0x1C5198u, 0x1C51ACu, 0x1C51C0u, 0x1C51D4u,
                                                       0x1C51E8u, 0x1C51FCu, 0x1C5210u};
            for (uint32_t i = 0; i < 7u; ++i)
            {
                const uint32_t p = READ32(thisPtr + 0x14u + i * 4u);
                if (p == 0u)
                    continue;
                SET_GPR_U64(ctx, 4, p);
                SET_GPR_U32(ctx, 31, kFreeSites[i] + 8u);
                ctx->pc = 0x00103F20u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103F20u, kFreeSites[i], kFreeSites[i] + 8u,
                                           PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            const uint32_t p30 = READ32(thisPtr + 0x30u);
            if (p30 != 0u)
            {
                SET_GPR_U32(ctx, 4, p30 - 0x10u);
                SET_GPR_U32(ctx, 31, 0x001C5234u);
                ctx->pc = 0x00103F20u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103F20u, 0x001C522Cu, 0x001C5234u,
                                           PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x001C524Cu);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u, 0x001C5244u, 0x001C524Cu,
                                           PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    using RecompiledFn = void (*)(uint8_t *, R5900Context *, PS2Runtime *);
    using RecompiledFunction = PS2Runtime::RecompiledFunction;

    // Generated code can still return early with pc at an inner return address (self-recursive
    // JALs are lowered to gotos, so the innermost JR $ra leaves the whole host function).
    // Keep re-entering at pc until the callee returns to the sentinel.
    void runScratchToCompletion(PS2Runtime *runtime, uint8_t *rdram, R5900Context *scratch,
                                RecompiledFn fn, uint32_t fnAddr)
    {
        constexpr uint32_t kScratchReturn = 0x0000FFF0u;
        HoldEeTimeslices hold;
        SET_GPR_U32(scratch, 31, kScratchReturn);
        const uint32_t entrySp = getRegU32(scratch, 29);
        scratch->pc = fnAddr;
        fn(rdram, scratch, runtime);
        if (scratch->pc == fnAddr && getRegU32(scratch, 29) == entrySp)
            return;
        for (uint32_t guard = 0; scratch->pc != kScratchReturn; ++guard)
        {
            if (scratch->pc == 0u || guard >= 1000000u || runtime->isStopRequested() ||
                !runtime->hasFunction(scratch->pc))
            {
                std::cerr << "[SCUS_972.11] scratch call 0x" << std::hex << fnAddr
                          << " stopped at pc=0x" << scratch->pc << std::dec << std::endl;
                return;
            }
            runtime->lookupFunction(scratch->pc)(rdram, scratch, runtime);
        }
    }

    uint32_t callRecompiled4(PS2Runtime *runtime, uint8_t *rdram, R5900Context *callerCtx,
                             uint32_t fnAddr, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4)
    {
        RecompiledFn fn = runtime->lookupFunction(fnAddr);
        if (fn == nullptr)
        {
            std::cerr << "[SCUS_972.11] callRecompiled4: helper 0x" << std::hex << fnAddr
                      << " not found in table" << std::dec << std::endl;
            return 0u;
        }

        R5900Context scratch = *callerCtx;
        SET_GPR_U32(&scratch, 4, a1);
        SET_GPR_U32(&scratch, 5, a2);
        SET_GPR_U32(&scratch, 6, a3);
        SET_GPR_U32(&scratch, 7, a4);
        runScratchToCompletion(runtime, rdram, &scratch, fn, fnAddr);
        return getRegU32(&scratch, 2);
    }

    uint32_t callRecompiled5(PS2Runtime *runtime, uint8_t *rdram, R5900Context *callerCtx,
                             uint32_t fnAddr, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5)
    {
        RecompiledFn fn = runtime->lookupFunction(fnAddr);
        if (fn == nullptr)
        {
            std::cerr << "[SCUS_972.11] callRecompiled5: helper 0x" << std::hex << fnAddr
                      << " not found in table" << std::dec << std::endl;
            return 0u;
        }

        R5900Context scratch = *callerCtx;
        SET_GPR_U32(&scratch, 4, a1);
        SET_GPR_U32(&scratch, 5, a2);
        SET_GPR_U32(&scratch, 6, a3);
        SET_GPR_U32(&scratch, 7, a4);
        SET_GPR_U32(&scratch, 8, a5);
        runScratchToCompletion(runtime, rdram, &scratch, fn, fnAddr);
        return getRegU32(&scratch, 2);
    }

    uint32_t callRecompiledFloat(PS2Runtime *runtime, uint8_t *rdram, R5900Context *callerCtx,
                                 uint32_t fnAddr, uint32_t a1, uint32_t a2, float f12Val, uint32_t a3)
    {
        RecompiledFn fn = runtime->lookupFunction(fnAddr);
        if (fn == nullptr)
        {
            std::cerr << "[SCUS_972.11] callRecompiledFloat: helper 0x" << std::hex << fnAddr
                      << " not found in table" << std::dec << std::endl;
            return 0u;
        }

        R5900Context scratch = *callerCtx;
        SET_GPR_U32(&scratch, 4, a1);
        SET_GPR_U32(&scratch, 5, a2);
        scratch.f[12] = f12Val;
        SET_GPR_U32(&scratch, 6, a3);
        runScratchToCompletion(runtime, rdram, &scratch, fn, fnAddr);
        return getRegU32(&scratch, 2);
    }

    // Lock wrapper constructor @ 0x341f90:
    // a0 = wrapper struct, a1 = holder/session ID, a2 = lock object
    // Stores a1 at 0(a0), a2 at 4(a0). If a2 != 0, calls lock's enterLock method (slot [2]).
    // If a2 == 0, safely skips calling method on NULL object.
    // Returns wrapper in v0.
    void lockWrapperEnter341f90(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t wrapperAddr = getRegU32(ctx, 4); // a0
        const uint32_t holderId = getRegU32(ctx, 5);    // a1
        const uint32_t lockAddr = getRegU32(ctx, 6);    // a2

        uint32_t *wrapper = reinterpret_cast<uint32_t *>(getMemPtr(rdram, wrapperAddr));
        if (wrapper != nullptr)
        {
            wrapper[0] = holderId;
            wrapper[1] = lockAddr;
        }

        if (lockAddr != 0u)
        {
            const uint32_t *lockObj = reinterpret_cast<const uint32_t *>(getMemPtr(rdram, lockAddr));
            if (lockObj != nullptr && lockObj[0] != 0u)
            {
                const uint32_t *vtable = reinterpret_cast<const uint32_t *>(getMemPtr(rdram, lockObj[0]));
                if (vtable != nullptr && vtable[2] != 0u)
                {
                    RecompiledFunction enterFn = runtime->lookupFunction(vtable[2]);
                    if (enterFn != nullptr)
                    {
                        SET_GPR_U32(ctx, 4, lockAddr);
                        SET_GPR_U32(ctx, 5, holderId);
                        enterFn(rdram, ctx, runtime);
                    }
                }
            }
        }

        SET_GPR_U32(ctx, 2, wrapperAddr); // v0 = wrapper
        ctx->pc = getRegU32(ctx, 31);
    }

    // Lock wrapper destructor @ 0x341fd0:
    // a0 = wrapper struct, a1 = flags
    // If wrapper != 0 and wrapper->lock != 0, calls lock's leaveLock method (slot [3]).
    // If (flags & 1) calls node/object free helper 0x103E70(wrapper).
    // Returns wrapper in v0.
    void lockWrapperLeave341fd0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t wrapperAddr = getRegU32(ctx, 4); // a0
        const uint32_t flags = getRegU32(ctx, 5);       // a1

        if (wrapperAddr != 0u)
        {
            const uint32_t *wrapper = reinterpret_cast<const uint32_t *>(getMemPtr(rdram, wrapperAddr));
            if (wrapper != nullptr)
            {
                const uint32_t holderId = wrapper[0];
                const uint32_t lockAddr = wrapper[1];
                if (lockAddr != 0u)
                {
                    const uint32_t *lockObj = reinterpret_cast<const uint32_t *>(getMemPtr(rdram, lockAddr));
                    if (lockObj != nullptr && lockObj[0] != 0u)
                    {
                        const uint32_t *vtable = reinterpret_cast<const uint32_t *>(getMemPtr(rdram, lockObj[0]));
                        if (vtable != nullptr && vtable[3] != 0u)
                        {
                            RecompiledFunction leaveFn = runtime->lookupFunction(vtable[3]);
                            if (leaveFn != nullptr)
                            {
                                SET_GPR_U32(ctx, 4, lockAddr);
                                SET_GPR_U32(ctx, 5, holderId);
                                leaveFn(rdram, ctx, runtime);
                            }
                        }
                    }
                }
            }
            if ((flags & 1u) != 0u)
            {
                RecompiledFunction freeFn = runtime->lookupFunction(0x00103E70u);
                if (freeFn != nullptr)
                {
                    SET_GPR_U32(ctx, 4, wrapperAddr);
                    freeFn(rdram, ctx, runtime);
                }
            }
        }

        SET_GPR_U32(ctx, 2, wrapperAddr);
        ctx->pc = getRegU32(ctx, 31);
    }

    // Mode/display vtable method @ 0x155950 — allocated by the display-modes vtable
    // (slot [4] of the mode object at s0+0x34+0x18). Analyzer missed the IndirectCall JALR.
    // Creates 2 display-mode nodes via 0x103C80, inits via 0x1A8C20/CC0/DE0, computes a
    // capability flag at this+0x1C, returns 1.
    void modeDisplayInit155950(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisObj = getRegU32(ctx, 4);
        uint8_t *thisBase = getMemPtr(rdram, thisObj);
        if (thisBase == nullptr)
        {
            std::cerr << "[SCUS_972.11] modeDisplayInit155950: cannot resolve guest this 0x"
                      << std::hex << thisObj << std::dec << std::endl;
            return;
        }

        const uint32_t nameAddr = 0x0042f590u; // "vu0_stats" / display-mode tag

        const uint32_t node0 = callRecompiled4(runtime, rdram, ctx, 0x103C80u,
                                               0x40u, 0x80u, nameAddr, 0x3FBu);
        auto *ws = reinterpret_cast<uint32_t *>(thisBase);
        ws[0x189C / 4] = node0;

        const uint32_t node1 = callRecompiled4(runtime, rdram, ctx, 0x103C80u,
                                               0x40u, 0x80u, nameAddr, 0x3FCu);
        ws[0x18A4 / 4] = node1;

        callRecompiled4(runtime, rdram, ctx, 0x1A8C20u, 0u, 0u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x1A8CC0u, node0, 0u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x1A8DE0u, 0u, 0u, 0u, 0u);

        uint32_t flag = 0u;
        if (node0 != 0u)
        {
            const uint8_t *nb = getMemPtr(rdram, node0);
            if (nb != nullptr && (nb[2] != 0 || nb[3] != 0))
            {
                flag = 1u;
            }
        }
        ws[0x1C / 4] = flag;

        SET_GPR_U32(ctx, 2, 1u); // return 1
    }

    // Node-lookup wrapper @ 0x187880: `dmove a0,a1; li a3,0xE; dmove a1,a2; j 0x103D80`
    // with a2=0x4209a0 — tail-calls the node-registry get (mem_alloc).
    // Original MIPS:
    //   daddu $a0, $a1, $zero (items)
    //   addiu $a3, $zero, 0xE (line = 14)
    //   daddu $a1, $a2, $zero (size)
    //   lui   $a2, 0x42
    //   j     func_103D80
    //   addiu $a2, $a2, 0x9A0 (filename = 0x004209A0)
    void nodeLookup187880(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t items = getRegU32(ctx, 5); // orig a1
        const uint32_t size = getRegU32(ctx, 6);  // orig a2
        SET_GPR_U32(ctx, 4, items);
        SET_GPR_U32(ctx, 5, size);
        SET_GPR_U32(ctx, 6, 0x004209A0u);
        SET_GPR_U32(ctx, 7, 0xEu);
        ctx->pc = 0x00103D80u;
        RecompiledFunction fn = runtime->lookupFunction(0x00103D80u);
        if (fn != nullptr)
        {
            fn(rdram, ctx, runtime);
        }
        else
        {
            std::cerr << "[nodeLookup187880] helper 0x103D80 not found!" << std::endl;
        }
    }

    // Node-release wrapper @ 0x1878a0:
    // Original MIPS:
    //   daddu $a0, $a1, $zero (ptr)
    //   addiu $a2, $zero, 0x13 (line = 19)
    //   lui   $a1, 0x42
    //   j     func_103DF0
    //   addiu $a1, $a1, 0x9A0 (filename = 0x004209A0)
    void nodeRelease1878a0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t ptr = getRegU32(ctx, 5); // orig a1
        SET_GPR_U32(ctx, 4, ptr);
        SET_GPR_U32(ctx, 5, 0x004209A0u);
        SET_GPR_U32(ctx, 6, 0x13u);
        ctx->pc = 0x00103DF0u;
        RecompiledFunction fn = runtime->lookupFunction(0x00103DF0u);
        if (fn != nullptr)
        {
            fn(rdram, ctx, runtime);
        }
        else
        {
            std::cerr << "[nodeRelease1878a0] helper 0x103DF0 not found!" << std::endl;
        }
    }

    // VBlank-chain vtable method @ 0x156e80 (slot [3] of obj+0x50 table), reached via
    // JALR from 0x1538ac inside FUN_00153680. Merged into generated sub_00156C80 but
    // its re-entry switch lacks case 0x156e80u. PCSX2-native MIPS decode (verified 2026-09-15):
    //
    //   addiu sp,-0x30; dmove s0,a0; sw s0,0x489038 (lui at,0x0049 + sw -0x6FC8)
    //   this+0x20=0x22B; this+0xC=2
    //   node0 = 0x103C80(0x40, 0x00020000, 0x0041F950, 0x22E);  this+0x5C = node0
    //   node1 = 0x103C80(0x40, 0x00020000, 0x0041F950, 0x230);  this+0x60=node0 (delay), this+0x68=node1
    //   ctx = this+0x90;  this+0x6C = node0 (delay of first 0x127560 call)
    //   0x127560(ctx, 1, 0x80)   — param-list start (cmd=1, init data=0x80)
    //   0x127B30(ctx, 0x1100); 0x127B30(ctx,0x0300); 0x127B30(ctx,0x0200)
    //   0x127960(ctx, 0x320, 0xC) — begin DMA header (returns index)
    //   stack tag (16B, zeroed via 0x36CB20(buf,0,0x10)) then byte-fiddled to:
    //       {0x01,0x80,0,0, 0,0x40,0,0x10, 0x0F,0,...}  (GIF-tag-like: NLOOP=1 EOP=1 PRIM FLG...)
    //   0x127870(ctx, buf) — append 16-byte tag to the command list
    //   0x1279E0(ctx)        — write DMA command word
    //   0x127B30(ctx,0x1400); 0x127B30(ctx,0x1000)
    //   0x127620(ctx)        — build GIF-tag header at ctx+0x10
    //   0x1588B0(this+0x78, 0x10, 0x10) — alloc vector at this+0x78
    //   this+0x4=0x280; this+0x8=0x1C0
    //   res = 0x103E30(0x20, 0x0041F950, 0x251)  — allocator dispatch
    //   if (res != 0) 0x187270(res, 0xC, 1)      — descriptor node on the buffer
    //   this+0xD4 = res
    //   this+0x24 = 0x3F800000 (delay of 0x1570D0 call)
    //   0x1570D0(this)                            — alloc 3 timer buffers (+0xC8/CC/D0)
    //   0x157510(this, 2, 0x003BF540, 0)          — VIF1 DMA kick #1
    //   0x156760(1, 1)                            — VIF1 wait/tick
    //   0x157510(this, 2, 0x003C0600, 0)          — VIF1 DMA kick #2
    //   0x156760(1, 1)                            — VIF1 wait/tick
    //   return this in v0
    void vtableMethod156e80(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisObj = getRegU32(ctx, 4);
        uint8_t *thisBase = getMemPtr(rdram, thisObj);
        if (thisBase == nullptr)
        {
            std::cerr << "[SCUS_972.11] vtableMethod156e80: cannot resolve guest this 0x"
                      << std::hex << thisObj << std::dec << std::endl;
            return;
        }

        // sw s0, 0x489038 (lui at,0x0049; sw s0,-0x6FC8(at))
        uint8_t *guestGlobal = getMemPtr(rdram, 0x489038u);
        if (guestGlobal == nullptr)
        {
            std::cerr << "[SCUS_972.11] vtableMethod156e80: cannot resolve guest global 0x489038"
                      << std::endl;
            return;
        }
        std::memcpy(guestGlobal, &thisObj, sizeof(thisObj));

        auto *ws = reinterpret_cast<uint32_t *>(thisBase);
        ws[0x20 / 4] = 0x22Bu; // li v0,0x22B; sw v0,0x20(a0)
        ws[0xC / 4] = 0x2u;    // li v0,0x2; sw v0,0xC(a0)

        // 2 display nodes via 0x103C80. a1 = lui 0x0002 = 0x00020000;
        // a2 = lui 0x0042 + addiu -0x6B0 = 0x0041F950; a3 = 0x22E / 0x230.
        const uint32_t kTag = 0x0041F950u;
        const uint32_t kA1 = 0x00020000u;

        const uint32_t node0 = callRecompiled4(runtime, rdram, ctx, 0x103C80u,
                                               0x40u, kA1, kTag, 0x22Eu);
        ws[0x5C / 4] = node0; // this+0x5C (after first call)

        const uint32_t node1 = callRecompiled4(runtime, rdram, ctx, 0x103C80u,
                                               0x40u, kA1, kTag, 0x230u);
        ws[0x60 / 4] = node0; // delay slot of 2nd jal: stores OLD v0 (node0)
        ws[0x68 / 4] = node1; // this+0x68 (after 2nd call returns)

        // Context list at this+0x90
        const uint32_t ctxAddr = thisObj + 0x90u;
        const uint32_t ctxListAddr = ctxAddr;

        // this+0x6C = node0 — delay slot of the 0x127560 jal (lw v0,0x5C(s0))
        ws[0x6C / 4] = node0;

        // 0x127560(ctx, 1, 0x80): param-list start (cmd=a1, init=a2).
        callRecompiled4(runtime, rdram, ctx, 0x127560u, ctxListAddr, 0x1u, 0x80u, 0u);

        // 0x127B30 context values
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, ctxListAddr, 0x1100u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, ctxListAddr, 0x0300u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, ctxListAddr, 0x0200u, 0u, 0u);

        // 0x127960(ctx, 0x320, 0xC): begin DMA header.
        callRecompiled4(runtime, rdram, ctx, 0x127960u, ctxListAddr, 0x320u, 0xCu, 0u);

        // Stack GIF-tag: zero a 16-byte guest scratch below the caller's sp (the original
        // fn allocated its own 0x30 frame; writing below sp is safe in this runtime model),
        // apply the exact byte fiddle from the MIPS, then append it via 0x127870(ctx, buf).
        const uint32_t sp = getRegU32(ctx, 29);
        const uint32_t tagBufAddr = (sp - 0x20u) & ~0xFu;
        uint8_t *tagBuf = getMemPtr(rdram, tagBufAddr);
        if (tagBuf == nullptr)
        {
            std::cerr << "[SCUS_972.11] vtableMethod156e80: cannot resolve guest tag buf 0x"
                      << std::hex << tagBufAddr << std::dec << std::endl;
            return;
        }
        std::memset(tagBuf, 0, 0x10); // 0x36CB20(buf, 0, 0x10)

        uint16_t t0 = static_cast<uint16_t>(tagBuf[0] | (tagBuf[1] << 8));
        uint16_t t5 = static_cast<uint16_t>((t0 & 0x8000u) | 0x1u); // lhu + andi/ori
        tagBuf[0] = static_cast<uint8_t>(t5 & 0xFFu);
        tagBuf[1] = static_cast<uint8_t>(t5 >> 8);

        tagBuf[5] = static_cast<uint8_t>((tagBuf[5] & ~0x40u) | 0x40u);   // lbu 0x25(sp)
        tagBuf[1] = static_cast<uint8_t>((tagBuf[1] & ~0x80u) | 0x80u);   // lbu 0x21(sp)
        tagBuf[7] = static_cast<uint8_t>((tagBuf[7] & ~0xCu) | 0x0u);     // lbu 0x27(sp), clear bits 2-3
        tagBuf[7] = static_cast<uint8_t>((tagBuf[7] & ~0xF0u) | 0x10u);   // set bits 4
        tagBuf[8] = static_cast<uint8_t>((tagBuf[8] & ~0xFu) | 0xFu);     // lbu 0x28(sp)

        callRecompiled4(runtime, rdram, ctx, 0x127870u, ctxListAddr, tagBufAddr, 0u, 0u);

        // 0x1279E0(ctx) — write the DMA command word.
        callRecompiled4(runtime, rdram, ctx, 0x1279E0u, ctxListAddr, 0u, 0u, 0u);

        // 0x127B30 again
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, ctxListAddr, 0x1400u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, ctxListAddr, 0x1000u, 0u, 0u);

        // 0x127620(ctx) — GIF-tag header at ctx+0x10.
        callRecompiled4(runtime, rdram, ctx, 0x127620u, ctxListAddr, 0u, 0u, 0u);

        // 0x1588B0(this+0x78, 0x10, 0x10) — alloc 64-byte vector descriptor.
        callRecompiled4(runtime, rdram, ctx, 0x1588B0u, thisObj + 0x78u, 0x10u, 0x10u, 0u);

        // Display size fields.
        ws[0x4 / 4] = 0x280u; // li v1,0x280; sw v1,0x4(s0)
        ws[0x8 / 4] = 0x1C0u; // li v0,0x1C0; sw v0,0x8(s0)

        // res = 0x103E30(0x20, 0x0041F950, 0x251) — allocator dispatch (a0=0x20).
        const uint32_t res = callRecompiled4(runtime, rdram, ctx, 0x103E30u,
                                             0x20u, 0x0041F950u, 0x251u, 0u);

        // beqz res -> skip 0x187270(res, 0xC, 1); descriptor node on the buffer.
        if (res != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x187270u, res, 0xCu, 0x1u, 0u);
        }
        ws[0xD4 / 4] = res; // this+0xD4 = res

        // Tail: this+0x24 = 1.0f (delay slot of 0x1570D0 jal), then 0x1570D0(this).
        ws[0x24 / 4] = 0x3F800000u; // lui v0,0x3F80 (delay of jal 0x1570D0)
        callRecompiled4(runtime, rdram, ctx, 0x1570D0u, thisObj, 0u, 0u, 0u);

        // VIF1 DMA kick #1: 0x157510(this, 2, 0x003BF540, 0).
        callRecompiled4(runtime, rdram, ctx, 0x157510u, thisObj, 0x2u, 0x003BF540u, 0u);
        // 0x156760(1, 1) — VIF1 wait/tick.
        callRecompiled4(runtime, rdram, ctx, 0x156760u, 0x1u, 0x1u, 0u, 0u);
        // VIF1 DMA kick #2: 0x157510(this, 2, 0x003C0600, 0) then 0x156760(1, 1).
        callRecompiled4(runtime, rdram, ctx, 0x157510u, thisObj, 0x2u, 0x003C0600u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x156760u, 0x1u, 0x1u, 0u, 0u);

        SET_GPR_U32(ctx, 2, thisObj); // dmove v0, s0
    }

    // Boot-path vtable method @ 0x176d60, reached via JALR from 0x10455c:
    //   t9 = *(*(a0+8) + 0x30) — virtual dispatch on the display-system object that
    //   FUN_0010451c created (allocated via 0x103E30, stored at 0x4891C0). Called with
    //   (this=0x531c10, 0x3, 0x0, 0x3). Analyzer missed the IndirectCall boundary.
    // PCSX2-native MIPS decode (verified 2026-09-15):
    //
    //   0x16B900(this)                     — alloc + init sub-struct, this+0x30
    //   v = 0x103E30(4, 0x4207A0, 0x61C); if (v) 0x185000(v)   (zero-init)
    //       this+0x80 = 0x185010(v, 0x609) (alloc + descriptor node)
    //   v = 0x103E30(4, 0x4207A0, 0x61D); if (v) v = 0x187120(v); this+0x84 = v
    //   v = 0x103E30(0x20, 0x4207A0, 0x61F); if (v) v = 0x187270(v, 4, 1); this+0x88 = v
    //   if (this+0x88): 0x1873A0 / 0x1873C0 / 0x187430(x,1) / 0x1874C0(x,0x3F,0) / 0x1875E0
    //   0x359E80(5, 0x174AE0, 0); 0x35ACE8(5); 0x359EB0(1, 0x174D40, 0)
    //   VIF1_TOPS (0x10003C30) = 0
    //   param lists: +0x90 {0x127560(1,0x80), 0x9300, 0x127620}
    //                +0xC0 {0x127560(1,0x80), 0x07000001, 0x9300, 0x127620}
    //                +0x50 {0x127560(1,0x80), 0x1100, 0x0300, 0x0200, 0x127960(0x320,0xC),
    //                       tag(+0x127870,+0x1279E0), 0x1400, 0x1000, 0x1400, 0x127620}
    //                +0xF0 {0x127560(1,0x80), 0x1300, 0x127620}
    //   cnt = (int16)this+0x14C; this+0x144 = 0x103C20(cnt<<5, 0x4207A0, 0x658)
    //   0x36CB20(this+0x144, 0, cnt<<5)   — zero fill
    //   return this in v0
    void vtableMethod176d60(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisObj = getRegU32(ctx, 4);
        uint8_t *thisBase = getMemPtr(rdram, thisObj);
        if (thisBase == nullptr)
        {
            std::cerr << "[SCUS_972.11] vtableMethod176d60: cannot resolve guest this 0x"
                      << std::hex << thisObj << std::dec << std::endl;
            return;
        }
        auto *ws = reinterpret_cast<uint32_t *>(thisBase);

        // jal 0x16B900(this): allocate 0x28 blob (0x103E30(0x28, 0x425720, 0x25)), init via
        // 0x1D0780, store this+0x30, then 0x106230(this).
        callRecompiled4(runtime, rdram, ctx, 0x16B900u, thisObj, 0u, 0u, 0u);

        // First descriptor: 0x103E30(4, 0x4207A0, 0x61C); if (v) 0x185000(v);
        // then this+0x80 = 0x185010(v, 0x609).
        uint32_t v = callRecompiled4(runtime, rdram, ctx, 0x103E30u, 0x4u, 0x004207A0u, 0x61Cu, 0u);
        if (v != 0u)
        {
            v = callRecompiled4(runtime, rdram, ctx, 0x185000u, v, 0u, 0u, 0u);
        }
        ws[0x80 / 4] = callRecompiled4(runtime, rdram, ctx, 0x185010u, v, 0x609u, 0u, 0u);

        // Second descriptor: 0x103E30(4, 0x4207A0, 0x61D); if (v) v = 0x187120(v);
        // this+0x84 = v.
        v = callRecompiled4(runtime, rdram, ctx, 0x103E30u, 0x4u, 0x004207A0u, 0x61Du, 0u);
        if (v != 0u)
        {
            v = callRecompiled4(runtime, rdram, ctx, 0x187120u, v, 0u, 0u, 0u);
        }
        ws[0x84 / 4] = v;

        // Frame buffer descriptor: 0x103E30(0x20, 0x4207A0, 0x61F); if (v) v = 0x187270(v, 4, 1);
        // this+0x88 = v.
        v = callRecompiled4(runtime, rdram, ctx, 0x103E30u, 0x20u, 0x004207A0u, 0x61Fu, 0u);
        if (v != 0u)
        {
            v = callRecompiled4(runtime, rdram, ctx, 0x187270u, v, 0x4u, 0x1u, 0u);
        }
        ws[0x88 / 4] = v;

        // Descriptor ops on this+0x88 (all unconditional in MIPS).
        const uint32_t r88 = ws[0x88 / 4];
        if (r88 != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x1873A0u, r88, 0u, 0u, 0u);
            callRecompiled4(runtime, rdram, ctx, 0x1873C0u, r88, 0u, 0u, 0u);
            callRecompiled4(runtime, rdram, ctx, 0x187430u, r88, 0x1u, 0u, 0u);
            callRecompiled4(runtime, rdram, ctx, 0x1874C0u, r88, 0x3Fu, 0u, 0u);
            callRecompiled4(runtime, rdram, ctx, 0x1875E0u, r88, 0u, 0u, 0u);
        }

        // Thread/intc setup.
        callRecompiled4(runtime, rdram, ctx, 0x359E80u, 0x5u, 0x00174AE0u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x35ACE8u, 0x5u, 0u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x359EB0u, 0x1u, 0x00174D40u, 0u, 0u);

        // VIF1_TOPS = 0 (lui at,0x1000; sw zero,0x3C30(at)) — via the memory model so
        // MMIO routing applies; ignore if the address is unmapped.
        try
        {
            runtime->memory().write32(0x10003C30u, 0u);
        }
        catch (...)
        {
        }

        // this+0x90 list: {cmd=1, init=0x80}, 0x9300, gif-tag header.
        const uint32_t list90 = thisObj + 0x90u;
        callRecompiled4(runtime, rdram, ctx, 0x127560u, list90, 0x1u, 0x80u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, list90, 0x9300u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127620u, list90, 0u, 0u, 0u);

        // this+0xC0 list: {cmd=1, init=0x80}, 0x07000001, 0x9300, gif-tag header.
        const uint32_t listC0 = thisObj + 0xC0u;
        callRecompiled4(runtime, rdram, ctx, 0x127560u, listC0, 0x1u, 0x80u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, listC0, 0x07000001u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, listC0, 0x9300u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127620u, listC0, 0u, 0u, 0u);

        // this+0x50 list: cmd start, values, 0x127960(0x320, 0xC), GIF-tag, three paddings.
        const uint32_t list50 = thisObj + 0x50u;
        callRecompiled4(runtime, rdram, ctx, 0x127560u, list50, 0x1u, 0x80u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, list50, 0x1100u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, list50, 0x0300u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, list50, 0x0200u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127960u, list50, 0x320u, 0xCu, 0u);

        // Stack GIF-tag (same fiddle as 0x156e80) appended via 0x127870(list50, buf).
        const uint32_t sp = getRegU32(ctx, 29);
        const uint32_t tagBufAddr = (sp - 0x20u) & ~0xFu;
        uint8_t *tagBuf = getMemPtr(rdram, tagBufAddr);
        if (tagBuf == nullptr)
        {
            std::cerr << "[SCUS_972.11] vtableMethod176d60: cannot resolve guest tag buf 0x"
                      << std::hex << tagBufAddr << std::dec << std::endl;
            return;
        }
        std::memset(tagBuf, 0, 0x10); // 0x36CB20(buf, 0, 0x10)

        uint16_t t0 = static_cast<uint16_t>(tagBuf[0] | (tagBuf[1] << 8));
        uint16_t t5 = static_cast<uint16_t>((t0 & 0x8000u) | 0x1u);
        tagBuf[0] = static_cast<uint8_t>(t5 & 0xFFu);
        tagBuf[1] = static_cast<uint8_t>(t5 >> 8);
        tagBuf[5] = static_cast<uint8_t>((tagBuf[5] & ~0x40u) | 0x40u);
        tagBuf[1] = static_cast<uint8_t>((tagBuf[1] & ~0x80u) | 0x80u);
        tagBuf[7] = static_cast<uint8_t>(tagBuf[7] & ~0xCu);
        tagBuf[7] = static_cast<uint8_t>((tagBuf[7] & ~0xF0u) | 0x10u);
        tagBuf[8] = static_cast<uint8_t>((tagBuf[8] & ~0xFu) | 0xFu);

        callRecompiled4(runtime, rdram, ctx, 0x127870u, list50, tagBufAddr, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x1279E0u, list50, 0u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, list50, 0x1400u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, list50, 0x1000u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, list50, 0x1400u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127620u, list50, 0u, 0u, 0u);

        // this+0xF0 list: {cmd=1, init=0x80}, 0x1300, gif-tag header.
        const uint32_t listF0 = thisObj + 0xF0u;
        callRecompiled4(runtime, rdram, ctx, 0x127560u, listF0, 0x1u, 0x80u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127B30u, listF0, 0x1300u, 0u, 0u);
        callRecompiled4(runtime, rdram, ctx, 0x127620u, listF0, 0u, 0u, 0u);

        // cnt = (int16)this+0x14C; this+0x144 = 0x103C20(cnt<<5, 0x4207A0, 0x658);
        // 0x36CB20(this+0x144, 0, cnt<<5).
        const int32_t cnt = static_cast<int16_t>(ws[0x14C / 4] & 0xFFFFu);
        const uint32_t size144 = static_cast<uint32_t>(cnt) << 5;
        ws[0x144 / 4] = callRecompiled4(runtime, rdram, ctx, 0x103C20u,
                                        size144, 0x004207A0u, 0x658u, 0u);
        uint8_t *buf144 = getMemPtr(rdram, ws[0x144 / 4]);
        if (buf144 != nullptr && size144 != 0u)
        {
            std::memset(buf144, 0, size144);
        }

        SET_GPR_U32(ctx, 2, thisObj); // dmove v0, s0
    }

    // Sound driver RPC dispatcher @ 0x002f0150 (sub_002F0150, PS2SOUND.IRX).
    // Called with $a0=context, $a1=cmd, $a2=flags, $a3=sendbuf, $t0=sendsize, $t1=recvbuf, $t2=recvsize.
    // Cmd 1 = query driver version -> caller at 0x2f1288 checks for 0x0221 (version 2.21).
    void soundRpcCall2f0150(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t ctxAddr = getRegU32(ctx, 4);  // $a0
        const uint32_t cmd = getRegU32(ctx, 5);      // $a1
        const uint32_t respAddr = ctxAddr + 0x300u;  // receive buffer offset
        uint32_t *resp = reinterpret_cast<uint32_t *>(getMemPtr(rdram, respAddr));

        uint32_t retVal = 0u;
        if (cmd == 1u)
        {
            retVal = 0x0221u; // Driver v2.21 expected by 0x2f1288
        }

        if (resp != nullptr)
        {
            *resp = retVal;
        }

        SET_GPR_U32(ctx, 2, retVal); // v0 = return value
        ctx->pc = getRegU32(ctx, 31); // return via ra
    }

    // Display/timer IOP RPC @ 0x001a8cc0 (sub_001A8CC0, sid=0x80000211).
    // Returns 0 on success.
    void displayRpcCall1a8cc0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        SET_GPR_U32(ctx, 2, 0u); // v0 = 0 (success)
        ctx->pc = getRegU32(ctx, 31); // return via ra
    }

    // sceMtapGetConnection @ 0x00378928 (sub_00378928).
    // $a0 = port (0 or 1). Returns 1 for direct pad attached without multitap.
    void mtapGetConnection378928(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        SET_GPR_U32(ctx, 2, 1u); // v0 = 1 (1 controller slot)
        ctx->pc = getRegU32(ctx, 31); // return via ra
    }

    // sceMtapChangeThreadPriority @ 0x00378848 (sub_00378848).
    // Returns 0 on success.
    void mtapChangeThreadPriority378848(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        SET_GPR_U32(ctx, 2, 0u); // v0 = 0 (success)
        ctx->pc = getRegU32(ctx, 31); // return via ra
    }

    // sceMtapInit @ 0x003785c8 (sub_003785C8).
    // Returns 0 on success.
    void mtapInit3785c8(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        SET_GPR_U32(ctx, 2, 0u); // v0 = 0 (success)
        ctx->pc = getRegU32(ctx, 31); // return via ra
    }

    // CDVD streaming status / check @ 0x00376408 (sub_00376408, calls SIF RPC sid=0x8000059A).
    // In libcdvd, this checks streaming/disk ready status.
    // Returns 0 on success. Returning 6 is an error code that aborts streaming CD reads in sub_0033F5C0.
    void cdvdStreamStatus376408(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        SET_GPR_U32(ctx, 2, 0u); // v0 = 0 (success / ready)
        ctx->pc = getRegU32(ctx, 31); // return via ra
    }

    // newlib _strtod_r @ 0x00370430.
    // Signature: double _strtod_r(struct _reent *reent, const char *text, char **endptr).
    // The analyzer identified this 3860-byte libc routine, but it was excluded as a
    // heavy-loop function and therefore has no generated function-table entry.
    void strtodR370430(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;

        const uint32_t reentAddr = getRegU32(ctx, 4);
        const uint32_t textAddr = getRegU32(ctx, 5);
        const uint32_t endPtrAddr = getRegU32(ctx, 6);
        constexpr size_t kMaxInputBytes = 4096u;

        std::string text;
        text.reserve(64u);
        bool terminated = false;
        if (textAddr != 0u)
        {
            for (size_t i = 0; i < kMaxInputBytes; ++i)
            {
                const uint64_t guestAddr = static_cast<uint64_t>(textAddr) + i;
                if (guestAddr > 0xFFFFFFFFull)
                {
                    break;
                }

                const uint8_t *guestChar = getMemPtr(rdram, static_cast<uint32_t>(guestAddr));
                if (guestChar == nullptr)
                {
                    break;
                }
                if (*guestChar == 0u)
                {
                    terminated = true;
                    break;
                }
                text.push_back(static_cast<char>(*guestChar));
            }
        }

        uint32_t guestEndAddr = textAddr;
        double value = 0.0;
        int parseErrno = 0;
        if (terminated)
        {
            errno = 0;
            char *hostEnd = nullptr;
            value = std::strtod(text.c_str(), &hostEnd);
            parseErrno = errno;
            const size_t consumed = (hostEnd != nullptr)
                                        ? static_cast<size_t>(hostEnd - text.c_str())
                                        : 0u;
            guestEndAddr += static_cast<uint32_t>(consumed);
        }
        else
        {
            std::cerr << "[SCUS_972.11] _strtod_r: invalid or unterminated guest string at 0x"
                      << std::hex << textAddr << std::dec << std::endl;
        }

        if (endPtrAddr != 0u)
        {
            uint8_t *guestEndPtr = getMemPtr(rdram, endPtrAddr);
            if (guestEndPtr != nullptr)
            {
                std::memcpy(guestEndPtr, &guestEndAddr, sizeof(guestEndAddr));
            }
            else
            {
                std::cerr << "[SCUS_972.11] _strtod_r: cannot resolve endptr at 0x"
                          << std::hex << endPtrAddr << std::dec << std::endl;
            }
        }

        // newlib's struct _reent begins with its errno field. Mirror host range errors.
        if (parseErrno != 0 && reentAddr != 0u)
        {
            uint8_t *guestErrno = getMemPtr(rdram, reentAddr);
            if (guestErrno != nullptr)
            {
                const int32_t guestErrnoValue = static_cast<int32_t>(parseErrno);
                std::memcpy(guestErrno, &guestErrnoValue, sizeof(guestErrnoValue));
            }
        }

        uint64_t resultBits = 0u;
        static_assert(sizeof(resultBits) == sizeof(value));
        std::memcpy(&resultBits, &value, sizeof(resultBits));
        setReturnU64(ctx, resultBits);

        static uint32_t logCount = 0u;
        if (logCount < 32u)
        {
            std::cerr << "[SCUS_972.11] _strtod_r: text='" << text
                      << "' consumed=" << (guestEndAddr - textAddr)
                      << " result=" << value << std::endl;
            ++logCount;
        }

        // 0x371364 tail-jumps here, so there is no DirectCall fallthrough to advance PC.
        ctx->pc = getRegU32(ctx, 31);
    }

    // Expanding chunk-list element allocator @ 0x001ABC50:
    // Merged into generated sub_001ABAE0 without re-entry switch support.
    // If $a1 != 0, returns $a1 in $v0.
    // If $a1 == 0, returns obj->0x40 if non-zero; else if obj->0x38 == obj->0x30,
    // expands chunk table via 0x103CF0 (realloc) and allocates new chunk via 0x103C20 (alloc);
    // returns chunk[offset] in $v0 and advances ctx->pc = $ra.
    void chunkListAlloc1abc50(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t a0 = getRegU32(ctx, 4);
        const uint32_t a1 = getRegU32(ctx, 5);

        if (a1 != 0u)
        {
            SET_GPR_U32(ctx, 2, a1);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        uint8_t *obj = getMemPtr(rdram, a0);
        if (obj == nullptr)
        {
            SET_GPR_U32(ctx, 2, 0u);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        const uint32_t f40 = *reinterpret_cast<const uint32_t *>(obj + 0x40);
        if (f40 != 0u)
        {
            SET_GPR_U32(ctx, 2, f40);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        const uint32_t f38 = *reinterpret_cast<const uint32_t *>(obj + 0x38);
        const uint32_t f30 = *reinterpret_cast<const uint32_t *>(obj + 0x30);

        if (f38 == f30)
        {
            const uint32_t f34 = *reinterpret_cast<const uint32_t *>(obj + 0x34);
            const uint32_t f3c = *reinterpret_cast<const uint32_t *>(obj + 0x3C);

            // 0x103CF0: realloc chunk pointer table
            const uint32_t newChunkTable = callRecompiled4(
                runtime, rdram, ctx, 0x00103CF0u,
                f3c, (f34 + 1u) << 2, 0x004223B8u, 0x56u);
            *reinterpret_cast<uint32_t *>(obj + 0x3C) = newChunkTable;

            // 0x103C20: alloc new chunk
            const uint32_t newChunk = callRecompiled4(
                runtime, rdram, ctx, 0x00103C20u,
                f30 << 2, 0x004223B8u, 0x59u, 0u);

            uint8_t *tblPtr = getMemPtr(rdram, newChunkTable + (f34 << 2));
            if (tblPtr != nullptr)
            {
                *reinterpret_cast<uint32_t *>(tblPtr) = newChunk;
            }

            *reinterpret_cast<uint32_t *>(obj + 0x34) = f34 + 1u;
            *reinterpret_cast<uint32_t *>(obj + 0x38) = 0u;
        }

        const uint32_t f34_now = *reinterpret_cast<const uint32_t *>(obj + 0x34);
        const uint32_t f3c_now = *reinterpret_cast<const uint32_t *>(obj + 0x3C);
        const uint32_t f38_now = *reinterpret_cast<const uint32_t *>(obj + 0x38);

        uint8_t *tblPtr = getMemPtr(rdram, f3c_now + (f34_now << 2) - 4u);
        const uint32_t chunkBase = (tblPtr != nullptr) ? *reinterpret_cast<const uint32_t *>(tblPtr) : 0u;
        const uint32_t retVal = chunkBase + (f38_now << 2);

        SET_GPR_U32(ctx, 2, retVal);
        ctx->pc = getRegU32(ctx, 31);
    }

    // 0x001ABC40: tail calls 0x1abc50 with a1 = a3
    void forward1abc40(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SET_GPR_U64(ctx, 5, GPR_U64(ctx, 7));
        chunkListAlloc1abc50(rdram, ctx, runtime);
    }

    // 0x001ABC20: tail calls 0x1abc50 with a1 = a2
    void forward1abc20(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SET_GPR_U64(ctx, 5, GPR_U64(ctx, 6));
        chunkListAlloc1abc50(rdram, ctx, runtime);
    }

    // 0x001ABC30: tail calls 0x1abc50 with current a1
    void forward1abc30(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        chunkListAlloc1abc50(rdram, ctx, runtime);
    }

    // 0x001ABD20: inserts element into chunk list and updates ref/count
    void chunkListInsert1abd20(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t a0 = getRegU32(ctx, 4);
        const uint32_t a1 = getRegU32(ctx, 5);
        const uint32_t a2 = getRegU32(ctx, 6);

        uint8_t *obj = getMemPtr(rdram, a0);
        if (obj != nullptr)
        {
            uint32_t v1 = *reinterpret_cast<const uint32_t *>(obj + 0x40);
            if (v1 != 0u && a2 == v1)
            {
                uint8_t *v1Ptr = getMemPtr(rdram, v1);
                if (v1Ptr != nullptr)
                {
                    const uint32_t a3 = *reinterpret_cast<const uint32_t *>(v1Ptr);
                    *reinterpret_cast<uint32_t *>(obj + 0x40) = a3 & ~3u;
                }
            }
            else
            {
                const uint32_t cnt = *reinterpret_cast<const uint32_t *>(obj + 0x38);
                *reinterpret_cast<uint32_t *>(obj + 0x38) = cnt + 1u;
            }
        }

        uint8_t *dest = getMemPtr(rdram, a2);
        if (dest != nullptr)
        {
            *reinterpret_cast<uint32_t *>(dest) = a1;
        }

        ctx->pc = getRegU32(ctx, 31);
    }

    // 0x001ABD60: leaf vtable method (obj+0x40 tracking)
    void chunkListMethod1abd60(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t a0 = getRegU32(ctx, 4);
        const uint32_t a2 = getRegU32(ctx, 6);
        uint8_t *obj = getMemPtr(rdram, a0);
        uint8_t *dest = getMemPtr(rdram, a2);
        if (obj != nullptr && dest != nullptr)
        {
            const uint32_t v1 = *reinterpret_cast<const uint32_t *>(obj + 0x40);
            *reinterpret_cast<uint32_t *>(dest) = v1 | 2u;
            *reinterpret_cast<uint32_t *>(obj + 0x40) = a2;
        }
        ctx->pc = getRegU32(ctx, 31);
    }

    // 0x001ABD80: leaf vtable method (flags reset)
    void chunkListMethod1abd80(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t a0 = getRegU32(ctx, 4);
        uint8_t *obj = getMemPtr(rdram, a0);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 0x44) = 0u;
            *reinterpret_cast<uint32_t *>(obj + 0x48) = 0u;
            const uint8_t b16 = obj[0x16];
            obj[0x16] = static_cast<uint8_t>((b16 & 0xFEu) | 1u);
        }
        SET_GPR_U32(ctx, 2, 1u);
        ctx->pc = getRegU32(ctx, 31);
    }

    // 0x001ABF30: leaf vtable method (flags reset)
    void chunkListMethod1abf30(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t a0 = getRegU32(ctx, 4);
        uint8_t *obj = getMemPtr(rdram, a0);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 0x44) = 0u;
            *reinterpret_cast<uint32_t *>(obj + 0x48) = 0u;
            const uint8_t b16 = obj[0x16];
            obj[0x16] = static_cast<uint8_t>((b16 & 0xFEu) | 1u);
        }
        ctx->pc = getRegU32(ctx, 31);
    }

    // 0x001ABF60: leaf vtable method (flags reset)
    void chunkListMethod1abf60(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t a0 = getRegU32(ctx, 4);
        uint8_t *obj = getMemPtr(rdram, a0);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 0x44) = 0u;
            *reinterpret_cast<uint32_t *>(obj + 0x48) = 0u;
            const uint8_t b16 = obj[0x16];
            obj[0x16] = static_cast<uint8_t>(b16 & 0xFEu);
        }
        ctx->pc = getRegU32(ctx, 31);
    }

    // 0x00101E80: leaf refcount getter (returns 64-bit count from obj+0x00)
    void getRef101e80(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        uint64_t val = 0;
        uint8_t *p = getMemPtr(rdram, getRegU32(ctx, 4));
        if (p != nullptr)
        {
            std::memcpy(&val, p, sizeof(val));
        }
        SET_GPR_U64(ctx, 2, val);
        ctx->pc = getRegU32(ctx, 31);
    }

    // 0x001A23F0: no-op leaf method
    void noop1a23f0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        ctx->pc = getRegU32(ctx, 31);
    }

    // Resource node copier @ 0x001BAA90:
    // Merged into generated sub_001BA970 without a table slot.
    // Deep-copies payload from src+0x38 into a newly allocated node at obj+0x2C via 0x103C80.
    void resourceNodeCopy1baa90(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t objAddr = getRegU32(ctx, 4); // a0
        const uint32_t srcAddr = getRegU32(ctx, 5); // a1

        uint8_t *obj = getMemPtr(rdram, objAddr);
        uint8_t *src = getMemPtr(rdram, srcAddr);
        if (obj == nullptr || src == nullptr)
        {
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        const uint32_t existingNode = *reinterpret_cast<const uint32_t *>(obj + 0x2C);
        if (existingNode != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00103DF0u, existingNode, 0x00424970u, 0x292u, 0u);
        }

        const uint32_t srcSubAddr = *reinterpret_cast<const uint32_t *>(src + 0x38);
        uint8_t *srcSub = getMemPtr(rdram, srcSubAddr);
        if (srcSub != nullptr)
        {
            const uint16_t count = *reinterpret_cast<const uint16_t *>(srcSub);
            const uint32_t payloadSize = static_cast<uint32_t>(count) << 4;

            const uint32_t newNode = callRecompiled4(
                runtime, rdram, ctx, 0x00103C80u,
                0x40u, payloadSize + 0x10u, 0x00424970u, 0x295u);

            *reinterpret_cast<uint32_t *>(obj + 0x2C) = newNode;

            uint8_t *newNodePtr = getMemPtr(rdram, newNode);
            if (newNodePtr != nullptr)
            {
                std::memcpy(newNodePtr + 0x10u, srcSub + 0x10u, payloadSize);

                *reinterpret_cast<uint16_t *>(newNodePtr + 0) = *reinterpret_cast<const uint16_t *>(srcSub + 0);
                newNodePtr[2] = srcSub[2];
                newNodePtr[3] = srcSub[3];
                *reinterpret_cast<uint32_t *>(newNodePtr + 4) = *reinterpret_cast<const uint32_t *>(srcSub + 4);
                *reinterpret_cast<uint32_t *>(newNodePtr + 8) = *reinterpret_cast<const uint32_t *>(srcSub + 8);
                *reinterpret_cast<uint32_t *>(newNodePtr + 12) = *reinterpret_cast<const uint32_t *>(srcSub + 12);

                *reinterpret_cast<uint32_t *>(newNodePtr + 4) = newNode + 0x10u;
            }
        }

        ctx->pc = getRegU32(ctx, 31);
    }

    // Equality comparison @ 0x001B7E10:
    // Merged into generated sub_001B7C20 without a function table slot.
    // Compares two resource/sound descriptors.
    // Returns 1 if equal, 0 if not equal.
    void resourceDescEquals1b7e10(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t a0Addr = getRegU32(ctx, 4);
        const uint32_t a1Addr = getRegU32(ctx, 5);

        if (a0Addr == a1Addr)
        {
            SET_GPR_U32(ctx, 2, 1u);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        uint8_t *a0 = getMemPtr(rdram, a0Addr);
        uint8_t *a1 = getMemPtr(rdram, a1Addr);
        if (a0 == nullptr || a1 == nullptr)
        {
            SET_GPR_U32(ctx, 2, 0u);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        // Compare scalar fields
        if (*reinterpret_cast<const int16_t *>(a0 + 0x28) != *reinterpret_cast<const int16_t *>(a1 + 0x28) ||
            a0[0x12] != a1[0x12] ||
            *reinterpret_cast<const int16_t *>(a0 + 0x14) != *reinterpret_cast<const int16_t *>(a1 + 0x14) ||
            *reinterpret_cast<const int16_t *>(a0 + 0x16) != *reinterpret_cast<const int16_t *>(a1 + 0x16) ||
            a0[0x18] != a1[0x18] ||
            a0[0x19] != a1[0x19])
        {
            SET_GPR_U32(ctx, 2, 0u);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        // Bitfields at 0x2A (bits 1 and 2)
        const uint8_t b0 = a0[0x2A];
        const uint8_t b1 = a1[0x2A];
        if (((b0 >> 1) & 3u) != ((b1 >> 1) & 3u))
        {
            SET_GPR_U32(ctx, 2, 0u);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        // Sub-payload at +0x38
        const uint32_t sub0Addr = *reinterpret_cast<const uint32_t *>(a0 + 0x38);
        const uint32_t sub1Addr = *reinterpret_cast<const uint32_t *>(a1 + 0x38);
        if (sub0Addr != sub1Addr)
        {
            uint8_t *sub0 = getMemPtr(rdram, sub0Addr);
            uint8_t *sub1 = getMemPtr(rdram, sub1Addr);
            if (sub0 == nullptr || sub1 == nullptr)
            {
                SET_GPR_U32(ctx, 2, 0u);
                ctx->pc = getRegU32(ctx, 31);
                return;
            }
            const uint16_t cnt0 = *reinterpret_cast<const uint16_t *>(sub0);
            const uint16_t cnt1 = *reinterpret_cast<const uint16_t *>(sub1);
            if (cnt0 != cnt1)
            {
                SET_GPR_U32(ctx, 2, 0u);
                ctx->pc = getRegU32(ctx, 31);
                return;
            }
            const size_t sz = static_cast<size_t>(cnt0) << 4;
            if (std::memcmp(sub0 + 0x10, sub1 + 0x10, sz) != 0)
            {
                SET_GPR_U32(ctx, 2, 0u);
                ctx->pc = getRegU32(ctx, 31);
                return;
            }
        }

        // Compare field +0x10
        const int16_t h0 = *reinterpret_cast<const int16_t *>(a0 + 0x10);
        const int16_t h1 = *reinterpret_cast<const int16_t *>(a1 + 0x10);
        if (h0 == -2 || h0 != h1)
        {
            SET_GPR_U32(ctx, 2, 0u);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        SET_GPR_U32(ctx, 2, 1u);
        ctx->pc = getRegU32(ctx, 31);
    }

    // 0x00179320: leaf vtable method `jr ra; li v0, 1`
    void leafRet1_179320(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        SET_GPR_U32(ctx, 2, 1u);
        ctx->pc = getRegU32(ctx, 31);
    }

    // Resource configuration parser (0x0014E210..0x0014E240 prologue).
    // Rejoin the generated body sub_0014DFD0 at 0x0014E244.
    void resourceConfig14e210(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = getRegU32(ctx, 29) - 0xC0u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));

        const uint64_t ra = GPR_U64(ctx, 31);
        std::memcpy(getMemPtr(rdram, sp + 0x30u), &ra, sizeof(ra));

        const auto s2 = GPR_VEC(ctx, 18);
        std::memcpy(getMemPtr(rdram, sp + 0x20u), &s2, 16);

        const auto s1 = GPR_VEC(ctx, 17);
        std::memcpy(getMemPtr(rdram, sp + 0x10u), &s1, 16);

        const auto s0 = GPR_VEC(ctx, 16);
        std::memcpy(getMemPtr(rdram, sp), &s0, 16);

        const uint64_t a0Val = GPR_U64(ctx, 4);
        const uint64_t a1Val = GPR_U64(ctx, 5);
        const uint64_t a2Val = GPR_U64(ctx, 6);

        SET_GPR_U64(ctx, 18, a0Val); // s2 = a0
        SET_GPR_U64(ctx, 17, a1Val); // s1 = a1
        SET_GPR_U64(ctx, 16, a2Val); // s0 = a2

        uint8_t *a0Ptr = getMemPtr(rdram, static_cast<uint32_t>(a0Val));
        if (a0Ptr != nullptr)
        {
            *reinterpret_cast<uint32_t *>(a0Ptr + 0x114u) = static_cast<uint32_t>(a2Val);
        }

        SET_GPR_U64(ctx, 4, a1Val);          // a0 = s1 (input filename)
        SET_GPR_U32(ctx, 5, 0x0041F078u);     // a1 = 0x41F078
        SET_GPR_U32(ctx, 31, 0x0014E244u);    // ra = 0x14E244
        ctx->pc = 0x0036F8B8u;

        if (!callGuestFromOverride(runtime, rdram, ctx, 0x0036F8B8u,
                0x0014E23Cu, 0x0014E244u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
            return;

        ctx->pc = 0x0014E244u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // VIF1 sync routine @ 0x00157BC0:
    //   li a0, 1
    //   j  0x00156760
    //   li a1, 3
    void vif1Sync157bc0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SET_GPR_U32(ctx, 4, 1u);
        SET_GPR_U32(ctx, 5, 3u);
        ctx->pc = 0x00156760u;
        runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
    }

    // Bypass movie playback in func_25B7C0 (0x25B7C0).
    // The frontend state machine uses this to play "Rainbow.pss" (state 10 -> next 9)
    // and "Intro.pss" (state 11 -> next 14).
    void movieStart25b7c0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        const uint32_t movieObj = getRegU32(ctx, 4);
        const uint32_t parentObj = getRegU32(ctx, 5);
        const uint32_t moviePathAddr = getRegU32(ctx, 6);
        const uint32_t nextState = getRegU32(ctx, 7);
        const char *moviePath = moviePathAddr ? reinterpret_cast<const char*>(getMemPtr(rdram, moviePathAddr)) : "null";

        std::cerr << "[MOVIE:BYPASS] movieObj=0x" << std::hex << movieObj
                  << " parentObj=0x" << parentObj
                  << " path=" << (moviePath ? moviePath : "null")
                  << " nextState=" << std::dec << nextState << std::endl;

        if (parentObj != 0u)
        {
            *reinterpret_cast<uint32_t*>(getMemPtr(rdram, parentObj + 0xC8u)) = nextState;
        }
        if (movieObj != 0u)
        {
            *reinterpret_cast<uint32_t*>(getMemPtr(rdram, movieObj + 0x40u)) = parentObj;
            *reinterpret_cast<uint32_t*>(getMemPtr(rdram, movieObj + 0x150u)) = nextState;
        }

        SET_GPR_S32(ctx, 2, 0);
        ctx->pc = GPR_U32(ctx, 31);
    }

    // PSXMediaControl::IsFinished (0x18AC80)
    void mediaIsFinished18ac80(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        SET_GPR_S32(ctx, 2, 0);
        ctx->pc = GPR_U32(ctx, 31);
    }

    // PSXMediaControl::Stop (0x18AC20)
    void mediaStop18ac20(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        SET_GPR_S32(ctx, 2, 0);
        ctx->pc = GPR_U32(ctx, 31);
    }

    // PSXMediaControl::StartPlayback (0x18A510)
    void mediaStartPlayback18a510(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        SET_GPR_S32(ctx, 2, 1);
        ctx->pc = GPR_U32(ctx, 31);
    }

    // Node virtual destructor @ 0x00152D50
    // Merged into sub_00152C90; called via vtable[0] from 0x1074A0
    void dtor152d50(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        if (thisPtr != 0u)
        {
            // Set vtable to base Node vtable 0x00452A40
            WRITE32(thisPtr + 8u, 0x00452A40u);
            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x00152D80u);
            ctx->pc = 0x0011B2D0u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x0011B2D0u,
                    0x00152D78u, 0x00152D80u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x00152D98u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x00152D90u, 0x00152D98u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for Node subclass @ 0x00108070
    // Merged into sub_00107FF0; called via vtable[0] from 0x1074A0
    void dtor108070(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00451C80u);
            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U32(ctx, 31, 0x0010809Cu);
            ctx->pc = 0x001082C0u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x001082C0u,
                    0x00108094u, 0x0010809Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x001080A8u);
            ctx->pc = 0x00107410u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x00107410u,
                    0x001080A0u, 0x001080A8u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x001080C0u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x001080B8u, 0x001080C0u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // ========================================================================
    // Base Destructor @ 0x001C0880 (PSXSurfaceGeometry.cpp)
    // ========================================================================
    void dtor1c0880(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t thisPtr, uint32_t deleteFlag)
    {
        if (thisPtr == 0u)
            return;

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x00454B10u;

            const int32_t a1 = *reinterpret_cast<const int32_t *>(obj + 0);
            if (a1 >= 0)
            {
                uint8_t *globPtr = getMemPtr(rdram, 0x00408398u);
                const uint32_t globalA0 = (globPtr != nullptr) ? *reinterpret_cast<const uint32_t *>(globPtr) : 0u;
                callRecompiled4(runtime, rdram, ctx, 0x001D2C00u, globalA0, static_cast<uint32_t>(a1), 0u, 0u);
            }

            const uint32_t p4 = *reinterpret_cast<const uint32_t *>(obj + 4);
            if (p4 != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, p4, 0u, 0u, 0u);
            }
        }

        if (static_cast<int32_t>(static_cast<int16_t>(deleteFlag)) > 0)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }
    }

    void dtorWrapper1c0880(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t deleteFlag = GPR_U32(ctx, 5);
        std::cerr << "[ATV2:DTOR1C0880] this=0x" << std::hex << thisPtr
                  << " delFlag=" << std::dec << deleteFlag << std::endl;
        dtor1c0880(rdram, ctx, runtime, thisPtr, deleteFlag);
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Virtual Destructor @ 0x001BD180 (PSXSurfaceGeometry.cpp)
    // ========================================================================
    void dtor1bd180(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t delFlag = GPR_U32(ctx, 5);

        std::cerr << "[ATV2:DTOR1BD180] this=0x" << std::hex << thisPtr
                  << " delFlag=" << std::dec << delFlag << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x00454A30u;
            const uint8_t flag0f = obj[0x0F];
            if ((flag0f & 1u) != 0u)
            {
                const uint32_t p10 = *reinterpret_cast<const uint32_t *>(obj + 0x10);
                if (p10 != 0u)
                {
                    callRecompiled4(runtime, rdram, ctx, 0x00103DF0u, p10, 0x00424E90u, 37u, 0u);
                }
            }

            const uint32_t p14 = *reinterpret_cast<const uint32_t *>(obj + 0x14);
            if (p14 != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103F20u, p14, 0u, 0u, 0u);
            }
        }

        dtor1c0880(rdram, ctx, runtime, thisPtr, 0u);

        if (static_cast<int32_t>(static_cast<int16_t>(delFlag)) > 0)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // Virtual destructor for Node subclass @ 0x00137EF0
    // vtable = 0x00452480 (called via vtable[0] from 0x1074A0)
    void dtor137ef0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR137EF0] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00452480u);
            const uint32_t globVal = READ32(0x0047D700u);
            if (globVal != 0u)
            {
                SET_GPR_U64(ctx, 4, globVal);
                SET_GPR_U64(ctx, 5, thisPtr);
                SET_GPR_U32(ctx, 31, 0x00137F30u);
                ctx->pc = 0x00139CE0u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00139CE0u,
                        0x00137F28u, 0x00137F30u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            const uint32_t pD0 = READ32(thisPtr + 0xD0u);
            if (pD0 != 0u)
            {
                SET_GPR_U64(ctx, 4, pD0);
                SET_GPR_U64(ctx, 5, 1u);
                SET_GPR_U32(ctx, 31, 0x00137F44u);
                ctx->pc = 0x001A4F50u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x001A4F50u,
                        0x00137F3Cu, 0x00137F44u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            const uint32_t pD8 = READ32(thisPtr + 0xD8u);
            if (pD8 != 0u)
            {
                SET_GPR_U64(ctx, 4, pD8);
                SET_GPR_U32(ctx, 31, 0x00137F58u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x00137F50u, 0x00137F58u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x00137F64u);
            ctx->pc = 0x00105C10u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x00105C10u,
                    0x00137F5Cu, 0x00137F64u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x00137F7Cu);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x00137F74u, 0x00137F7Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for Resource node subclass @ 0x001CE0E0
    // vtable = 0x00455220 (slot +0x8)
    void dtor1ce0e0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR1CE0E0] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00455220u);

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x001CE110u);
            ctx->pc = 0x001CDF20u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x001CDF20u,
                    0x001CE108u, 0x001CE110u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x001CE128u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x001CE120u, 0x001CE128u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for Resource node subclass @ 0x001CE140
    // vtable = 0x00455240 (slot +0x8)
    void dtor1ce140(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR1CE140] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00455240u);

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x001CE170u);
            ctx->pc = 0x001CDF20u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x001CDF20u,
                    0x001CE168u, 0x001CE170u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x001CE188u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x001CE180u, 0x001CE188u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for Resource node subclass @ 0x001CE330
    // vtable = 0x00455280 (slot +0x8)
    void dtor1ce330(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR1CE330] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00455280u);

            static const uint32_t kOffsets[] = {
                0x5Cu, 0x84u, 0xD8u, 0x88u, 0x8Cu, 0x90u, 0x94u, 0x98u,
                0x9Cu, 0xA0u, 0xA4u, 0xACu, 0xA8u, 0xB8u, 0xBCu, 0xC0u,
                0xB0u, 0xB4u, 0xC4u, 0xC8u
            };
            for (uint32_t off : kOffsets)
            {
                const uint32_t ptrVal = READ32(thisPtr + off);
                if (ptrVal != 0u)
                {
                    SET_GPR_U64(ctx, 4, ptrVal);
                    SET_GPR_U32(ctx, 31, 0x001CE36Cu);
                    ctx->pc = 0x00103F20u;
                    if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103F20u,
                            0x001CE364u, 0x001CE36Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                        return;
                }
            }

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x001CE4F4u);
            ctx->pc = 0x00105C10u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x00105C10u,
                    0x001CE4ECu, 0x001CE4F4u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x001CE50Cu);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x001CE504u, 0x001CE50Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for VertexBuffer node subclass @ 0x001C2A70
    // vtable = 0x00454CB0 (slot +0x8)
    void dtor1c2a70(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x50u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x40u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x30u, GPR_VEC(ctx, 19));
        WRITE128(sp + 0x20u, GPR_VEC(ctx, 18));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR1C2A70] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00454CB0u);

            const uint32_t count = READ32(thisPtr + 0x1E4u);
            for (uint32_t i = 0; i < count; ++i)
            {
                const uint32_t dataPtr = READ32(thisPtr + 0x1E4u + 8u);
                const uint32_t item = (dataPtr != 0u) ? READ32(dataPtr + i * 4u) : 0u;
                if (item != 0u)
                {
                    const uint32_t v210 = READ32(thisPtr + 0x210u);
                    if (v210 != 0u)
                    {
                        const uint32_t item4 = READ32(item + 4u);
                        const uint32_t arg0 = READ32(thisPtr + 0x214u);
                        SET_GPR_U64(ctx, 4, arg0);
                        SET_GPR_U64(ctx, 5, item4);
                        SET_GPR_U32(ctx, 31, 0x001C2AD4u);
                        ctx->pc = 0x00151300u;
                        if (!callGuestFromOverride(runtime, rdram, ctx, 0x00151300u,
                                0x001C2ACCu, 0x001C2AD4u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                            return;
                    }
                    SET_GPR_U64(ctx, 4, item);
                    SET_GPR_U32(ctx, 31, 0x001C2AE0u);
                    ctx->pc = 0x00103E70u;
                    if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                            0x001C2AD8u, 0x001C2AE0u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                        return;
                }
            }

            SET_GPR_U64(ctx, 4, thisPtr + 0x1E4u);
            SET_GPR_U32(ctx, 31, 0x001C2B00u);
            ctx->pc = 0x001C4C50u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x001C4C50u,
                    0x001C2AF8u, 0x001C2B00u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            SET_GPR_U64(ctx, 4, thisPtr + 0x1F8u);
            SET_GPR_U32(ctx, 31, 0x001C2B08u);
            ctx->pc = 0x001C4C50u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x001C4C50u,
                    0x001C2B00u, 0x001C2B08u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            const uint32_t v20c = READ32(thisPtr + 0x20Cu);
            if (v20c != 0u)
            {
                const uint32_t glob404818 = READ32(0x00404818u);
                const uint32_t a0 = (glob404818 != 0u) ? READ32(glob404818) : 0u;
                SET_GPR_U64(ctx, 4, a0);
                SET_GPR_U64(ctx, 5, thisPtr);
                SET_GPR_U32(ctx, 31, 0x001C2B2Cu);
                ctx->pc = 0x00173FC0u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00173FC0u,
                        0x001C2B20u, 0x001C2B2Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            const uint32_t p200 = READ32(thisPtr + 0x200u);
            if (p200 != 0u)
            {
                SET_GPR_U64(ctx, 4, p200);
                SET_GPR_U32(ctx, 31, 0x001C2B40u);
                ctx->pc = 0x00103F20u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103F20u,
                        0x001C2B34u, 0x001C2B40u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            const uint32_t p1ec = READ32(thisPtr + 0x1ECu);
            if (p1ec != 0u)
            {
                SET_GPR_U64(ctx, 4, p1ec);
                SET_GPR_U32(ctx, 31, 0x001C2B54u);
                ctx->pc = 0x00103F20u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103F20u,
                        0x001C2B48u, 0x001C2B54u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x001C2B5Cu);
            ctx->pc = 0x001C1110u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x001C1110u,
                    0x001C2B54u, 0x001C2B5Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x001C2B74u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x001C2B6Cu, 0x001C2B74u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x40u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 19, READ128(sp + 0x30u));
        SET_GPR_VEC(ctx, 18, READ128(sp + 0x20u));
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x50u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for Highlight_Icon node subclass @ 0x002AD640
    // vtable = 0x0045D790 (slot +0x8)
    void dtor2ad640(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR2AD640] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x0045D790u);

            const uint32_t p68 = READ32(thisPtr + 0x68u);
            if (p68 != 0u)
            {
                SET_GPR_U64(ctx, 4, p68);
                SET_GPR_U64(ctx, 5, 1u);
                SET_GPR_U32(ctx, 31, 0x002AD674u);
                ctx->pc = 0x002B48B0u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x002B48B0u,
                        0x002AD66Cu, 0x002AD674u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            const uint32_t p6c = READ32(thisPtr + 0x6Cu);
            if (p6c != 0u)
            {
                SET_GPR_U64(ctx, 4, p6c);
                SET_GPR_U64(ctx, 5, 1u);
                SET_GPR_U32(ctx, 31, 0x002AD680u);
                ctx->pc = 0x002B48B0u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x002B48B0u,
                        0x002AD678u, 0x002AD680u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            WRITE32(thisPtr + 0x78u, 0u);
            SET_GPR_U32(ctx, 31, 0x002AD690u);
            ctx->pc = 0x00234F50u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x00234F50u,
                    0x002AD688u, 0x002AD690u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x002AD6A8u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x002AD6A0u, 0x002AD6A8u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for Animated_Icon node subclass @ 0x002AE5A0
    // vtable = 0x0045D6D0 (slot +0x8)
    void dtor2ae5a0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR2AE5A0] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x0045D6D0u);

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x002AE5D0u);
            ctx->pc = 0x002393A0u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x002393A0u,
                    0x002AE5C8u, 0x002AE5D0u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x002AE5E8u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x002AE5E0u, 0x002AE5E8u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for Slider node subclass @ 0x002AE8F0
    // vtable = 0x0045D610 (slot +0x8)
    void dtor2ae8f0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR2AE8F0] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x0045D610u);

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x002AE920u);
            ctx->pc = 0x002393A0u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x002393A0u,
                    0x002AE918u, 0x002AE920u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x002AE938u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x002AE930u, 0x002AE938u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for TextInput node subclass @ 0x00236440
    // vtable = 0x00458A70 (slot +0x8)
    void dtor236440(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR236440] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00458A70u);

            const uint32_t p68 = READ32(thisPtr + 0x68u);
            if (p68 != 0u)
            {
                SET_GPR_U64(ctx, 4, p68);
                SET_GPR_U64(ctx, 5, 0x0043E0D0u);
                SET_GPR_U64(ctx, 6, 0x318u); // line 792
                SET_GPR_U32(ctx, 31, 0x00236480u);
                ctx->pc = 0x00103DF0u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103DF0u,
                        0x00236478u, 0x00236480u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x0023648Cu);
            ctx->pc = 0x00234F50u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x00234F50u,
                    0x00236484u, 0x0023648Cu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x002364A4u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x0023649Cu, 0x002364A4u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for Resource node subclass @ 0x0014E170
    // vtable = 0x00452980 (slot +0x8)
    void dtor14e170(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR14E170] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00452980u);

            const uint32_t v114 = READ32(thisPtr + 0x114u);
            if (v114 == 2u)
            {
                const int32_t a1 = static_cast<int32_t>(READ32(thisPtr + 0x134u));
                if (a1 >= 0)
                {
                    const uint32_t v138 = READ32(thisPtr + 0x138u);
                    const uint32_t a2 = READ32(thisPtr + 0x1B8u);
                    SET_GPR_U64(ctx, 4, v138 + 0xC0u);
                    SET_GPR_S32(ctx, 5, a1);
                    SET_GPR_U64(ctx, 6, a2);
                    SET_GPR_U64(ctx, 7, 0u);
                    SET_GPR_U32(ctx, 31, 0x0014E1C8u);
                    ctx->pc = 0x002F3280u;
                    if (!callGuestFromOverride(runtime, rdram, ctx, 0x002F3280u,
                            0x0014E1C0u, 0x0014E1C8u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                        return;
                }
            }

            SET_GPR_U64(ctx, 4, thisPtr);
            SET_GPR_U64(ctx, 5, 0u);
            SET_GPR_U32(ctx, 31, 0x0014E1D4u);
            ctx->pc = 0x001072C0u;
            if (!callGuestFromOverride(runtime, rdram, ctx, 0x001072C0u,
                    0x0014E1CCu, 0x0014E1D4u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                return;

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x0014E1ECu);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x0014E1E4u, 0x0014E1ECu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Virtual destructor for SearchList base class @ 0x001A2160
    // vtable = 0x00454070 (slot +0x8)
    void dtor1a2160(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        std::cerr << "[ATV2:DTOR1A2160] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x00454070u;

            const uint32_t memObj = *reinterpret_cast<const uint32_t *>(obj + 0x20);
            if (memObj != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103F20u, memObj, 0u, 0u, 0u);
            }

            // Base class destructor @ 0x00107410
            callRecompiled4(runtime, rdram, ctx, 0x00107410u, thisPtr, 0u, 0u, 0u);

            if (delFlag > 0)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // Virtual destructor for SearchList derived class @ 0x001ABB40
    // vtable = 0x004543E0 (slot +0x8)
    void dtor1abb40(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        std::cerr << "[ATV2:DTOR1ABB40] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x004543E0u;

            const uint32_t chunkTable = *reinterpret_cast<const uint32_t *>(obj + 60);
            if (chunkTable != 0u)
            {
                const uint32_t count = *reinterpret_cast<const uint32_t *>(obj + 52);
                for (uint32_t i = 0u; i < count; ++i)
                {
                    uint8_t *slotPtr = getMemPtr(rdram, chunkTable + i * 4);
                    if (slotPtr != nullptr)
                    {
                        const uint32_t chunkPtr = *reinterpret_cast<const uint32_t *>(slotPtr);
                        if (chunkPtr != 0u)
                        {
                            callRecompiled4(runtime, rdram, ctx, 0x00103DF0u, chunkPtr, 0x004223B8u, 30u, 0u);
                        }
                    }
                }

                callRecompiled4(runtime, rdram, ctx, 0x00103DF0u, chunkTable, 0x004223B8u, 32u, 0u);
            }

            // Base class destructor 0x001A2160(thisPtr, 0)
            callRecompiled4(runtime, rdram, ctx, 0x001A2160u, thisPtr, 0u, 0u, 0u);

            if (delFlag > 0)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // PSXIcon helper destructor @ 0x00342060
    void dtor342060(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR342060] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            const uint32_t p0c = READ32(thisPtr + 0x0Cu);
            if (p0c != 0u)
            {
                SET_GPR_U64(ctx, 4, p0c);
                SET_GPR_U64(ctx, 5, 0x00447610u); // "PSXIcon.cpp"
                SET_GPR_U64(ctx, 6, 45u);
                SET_GPR_U32(ctx, 31, 0x00342094u);
                ctx->pc = 0x00103DF0u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103DF0u,
                        0x0034208Cu, 0x00342094u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            const uint32_t p18 = READ32(thisPtr + 0x18u);
            if (p18 != 0u)
            {
                SET_GPR_U64(ctx, 4, p18);
                SET_GPR_U32(ctx, 31, 0x003420ACu);
                ctx->pc = 0x00103F20u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103F20u,
                        0x003420A0u, 0x003420ACu, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }

            if (delFlag > 0)
            {
                SET_GPR_U64(ctx, 4, thisPtr);
                SET_GPR_U32(ctx, 31, 0x003420C4u);
                ctx->pc = 0x00103E70u;
                if (!callGuestFromOverride(runtime, rdram, ctx, 0x00103E70u,
                        0x003420B8u, 0x003420C4u, PS2Runtime::GuestBranchKind::DirectCall, "JAL"))
                    return;
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // PSXIcon helper destructor @ 0x003424B0
    // vtable = 0x00460E10
    void dtor3424b0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29) - 0x30u;
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp));
        WRITE64(sp + 0x20u, GPR_U64(ctx, 31));
        WRITE128(sp + 0x10u, GPR_VEC(ctx, 17));
        WRITE128(sp, GPR_VEC(ctx, 16));

        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        SET_GPR_U64(ctx, 17, thisPtr);
        SET_GPR_S32(ctx, 16, delFlag);

        std::cerr << "[ATV2:DTOR3424B0] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 0x20u, 0x00460E10u);

            const uint32_t p30 = READ32(thisPtr + 0x30u);
            if (p30 != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00342060u, p30, 1u, 0u, 0u);
            }

            if (delFlag > 0)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // Tail / epilogue of PSXIcon helper destructor @ 0x003424F0
    void epilogue3424f0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sp = GPR_U32(ctx, 29);
        const int32_t s0 = static_cast<int32_t>(GPR_U32(ctx, 16));
        const uint32_t s1 = GPR_U32(ctx, 17);

        if (s0 > 0 && s1 != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, s1, 0u, 0u, 0u);
        }

        SET_GPR_U64(ctx, 2, s1);
        const uint64_t retRa = READ64(sp + 0x20u);
        SET_GPR_U64(ctx, 31, retRa);
        SET_GPR_VEC(ctx, 17, READ128(sp + 0x10u));
        SET_GPR_VEC(ctx, 16, READ128(sp));
        SET_GPR_S32(ctx, 29, static_cast<int32_t>(sp + 0x30u));
        ctx->pc = static_cast<uint32_t>(retRa);
    }

    // PSXIcon base class destructor @ 0x00343D60
    // vtable = 0x00460E60
    void dtor343d60(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        std::cerr << "[ATV2:DTOR343D60] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x00460E60u;

            const uint32_t p00 = *reinterpret_cast<const uint32_t *>(obj + 0);
            if (p00 != 0u)
            {
                uint8_t *p00Mem = getMemPtr(rdram, p00);
                if (p00Mem != nullptr)
                {
                    const uint32_t vtbl = *reinterpret_cast<const uint32_t *>(p00Mem + 8);
                    uint8_t *vtblMem = getMemPtr(rdram, vtbl);
                    if (vtblMem != nullptr)
                    {
                        const uint32_t dtorFn = *reinterpret_cast<const uint32_t *>(vtblMem + 8);
                        if (dtorFn != 0u)
                        {
                            callRecompiled4(runtime, rdram, ctx, dtorFn, p00, 1u, 0u, 0u);
                        }
                    }
                }
            }

            if (delFlag > 0)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // PSXIcon base destructor @ 0x003441E0
    // vtable = 0x00460E50
    void dtor3441e0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        std::cerr << "[ATV2:DTOR3441E0] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x00460E50u;

            // Base class destructor 0x00343D60(thisPtr, 0)
            callRecompiled4(runtime, rdram, ctx, 0x00343D60u, thisPtr, 0u, 0u, 0u);

            if (delFlag > 0)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // PSXIcon middle destructor @ 0x003426D0
    // vtable = 0x00460E00
    void dtor3426d0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        std::cerr << "[ATV2:DTOR3426D0] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x00460E00u;

            const uint32_t p0c = *reinterpret_cast<const uint32_t *>(obj + 0x0C);
            if (p0c != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x003424B0u, p0c, 1u, 0u, 0u);
            }

            // Base class destructor 0x003441E0(thisPtr, 0)
            callRecompiled4(runtime, rdram, ctx, 0x003441E0u, thisPtr, 0u, 0u, 0u);

            if (delFlag > 0)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // PSXIcon derived destructor @ 0x003429C0
    // vtable = 0x00460DF0
    void dtor3429c0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));
        std::cerr << "[ATV2:DTOR3429C0] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x00460DF0u;

            const uint32_t p10 = *reinterpret_cast<const uint32_t *>(obj + 0x10);
            if (p10 != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, p10, 0u, 0u, 0u);
            }

            // Base class destructor 0x003426D0(thisPtr, 0)
            callRecompiled4(runtime, rdram, ctx, 0x003426D0u, thisPtr, 0u, 0u, 0u);

            if (delFlag > 0)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
            }
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Structure / buffer initializer @ 0x002F1820
    // Called via thunk 0x151970 (a0 += 0xC0).
    // Clears internal buffers, initialises 24×2 flag-byte table, frees stale
    // container entries, and resets the container.
    // ========================================================================
    void init2F1820(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4); // a0 = this (already offset by caller)

        // std::cerr << "[ATV2:INIT2F1820] this=0x" << std::hex << thisPtr << std::dec << std::endl;

        // --- this->DF0 = 0 ---
        WRITE32(thisPtr + 0x0DF0u, 0u);

        // --- memset(this+0x380, 0, 0x640) ---
        std::memset(getMemPtr(rdram, thisPtr + 0x0380u), 0, 0x640u);

        // --- memset(this+0x340, 0, 0x30) ---
        std::memset(getMemPtr(rdram, thisPtr + 0x0340u), 0, 0x030u);

        // --- memset(this+0x9D0, 0, 0x60) ---
        std::memset(getMemPtr(rdram, thisPtr + 0x09D0u), 0, 0x060u);

        // --- Nested loop: outer a3 = 0..1, inner t0 = 0..23 ---
        // Each inner iteration operates on this + outerOff + t1, where
        //   outerOff starts at 0 and increments by 0x1E0 each outer iter,
        //   t1 starts at 0 and increments by 0x14 each inner iter.
        // Operations per entry (byte at this + outerOff + t1 + 0x0A30):
        //   bit0 cleared then set to 0 (AND 0xFFFE, OR 0x0000)  → bit0 = 0
        //   bit1 cleared then set to 0 (AND 0xFFFD, OR 0x0000)  → bit1 = 0
        //   bit2 cleared then set to 0 (AND 0xFFFB, OR 0x0000)  → bit2 = 0
        //   bit3 cleared then set to 0 (AND 0xFFF7, OR 0x0000)  → bit3 = 0
        //   → net effect: low 4 bits of byte cleared.
        // Words at +4, +8, +12 set too.
        {
            uint32_t outerOff = 0u;
            for (int outer = 0; outer < 2; ++outer)
            {
                uint32_t innerOff = 0u;
                for (int inner = 0; inner < 24; ++inner)
                {
                    const uint32_t entryBase = thisPtr + outerOff + innerOff + 0x0A30u;

                    // Clear low 4 bits of the flag byte
                    uint8_t flagByte = READ8(entryBase);
                    flagByte &= 0xF0u;
                    WRITE8(entryBase, flagByte);

                    // Remaining words in the 20-byte entry
                    WRITE32(entryBase + 4u, 0xFFFFFFFFu);  // t6 = -1
                    WRITE32(entryBase + 8u, 0xFFFFFFFFu);  // t6 = -1
                    WRITE32(entryBase + 12u, 0x00000000u); // zero
                    WRITE32(entryBase + 16u, 0x43340000u); // t5 = 0x4334 << 16 = float 180.0f

                    innerOff += 0x14u;
                }
                outerOff += 0x1E0u;
            }
        }

        // --- Loop: clear array entries via pointers at this+0x9C0 / this+0x9C8 ---
        // for (a0=0, a1=0; a0 < this[0x37C]; a0++, a1+=4) {
        //   *(this[0x9C0] + a1) = 0;
        //   *(this[0x9C8] + a1) = 0x43340000; // float 180.0
        // }
        {
            const uint32_t count = READ32(thisPtr + 0x037Cu);
            const uint32_t ptrA  = READ32(thisPtr + 0x09C0u);
            const uint32_t ptrB  = READ32(thisPtr + 0x09C8u);
            for (uint32_t i = 0; i < count; ++i)
            {
                if (ptrA != 0u) WRITE32(ptrA + i * 4u, 0u);
                if (ptrB != 0u) WRITE32(ptrB + i * 4u, 0x43340000u);
            }
        }

        // --- Loop: for each entry in container at this+0x988C, call 0x2F5370 ---
        // then delete the returned pointer.
        // for (s1 = 0; s1 < *(this+0x988C); s1++) {
        //   v0 = call 0x2F5370(this+0x988C, s1);
        //   call delete(v0);
        // }
        {
            const uint32_t containerAddr = thisPtr + 0x988Cu;
            // Read count from *(this+0x988C) — but the loop loads it indirectly:
            // ori at, zero, 0x988C; addu v0, s0, at; lw v0, 0(v0) → count = *(this+0x988C)
            // Actually the lw is at (s0 + 0x988C + 0) so count = READ32(thisPtr + 0x988C)
            // Wait, let me re-read. The MIPS is:
            //   0x2F1980: ori v0, zero, 0x988C    v0 = 0x988C
            //   0x2F1984: addu v0, s0, v0          v0 = s0 + 0x988C = this+0x988C
            //   0x2F1988: lw v0, 0(v0)             v0 = *(this+0x988C)
            //   0x2F198C: slt v0, s1, v0            v0 = (s1 < count)
            // So count = READ32(thisPtr + 0x988C). But this is likely a struct with
            // count at offset 0 from containerAddr.
            const uint32_t containerCount = READ32(containerAddr);
            for (uint32_t idx = 0; idx < containerCount; ++idx)
            {
                // call 0x2F5370(containerAddr, idx)
                uint32_t ptr = callRecompiled4(runtime, rdram, ctx, 0x002F5370u,
                                               containerAddr, idx, 0u, 0u);
                // call delete(ptr) = 0x103E70
                if (ptr != 0u)
                {
                    callRecompiled4(runtime, rdram, ctx, 0x00103E70u,
                                    ptr, 0u, 0u, 0u);
                }
            }

            // call 0x2F5390(containerAddr) — reset/clear container
            callRecompiled4(runtime, rdram, ctx, 0x002F5390u,
                            containerAddr, 0u, 0u, 0u);
        }

        // --- call 0x2F0150(this, 35, 0, 0) ---
        callRecompiled4(runtime, rdram, ctx, 0x002F0150u,
                        thisPtr, 35u, 0u, 0u);

        // Return: v0 is not used by caller in the thunk path
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Forwarding thunk @ 0x00151970
    //   j 0x002F1820        ; jump to init2F1820
    //   addiu a0, a0, 0xC0  ; (delay slot) offset this by +0xC0
    // Called from vtable at 0x00452A20 slot +0x14.
    // ========================================================================
    void thunk151970(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // Apply the delay-slot offset: a0 += 0xC0
        const uint32_t a0 = GPR_U32(ctx, 4) + 0xC0u;
        SET_GPR_U32(ctx, 4, a0);

        // Tail-call to init2F1820
        init2F1820(rdram, ctx, runtime);
    }

    // ========================================================================
    // Virtual destructor for Node/Resource subclass @ 0x001B8DB0
    // vtable 0x00454930 (+0x08).
    // Cleans up linked list at thisPtr + 488, calls base dtor 0x001BF310,
    // and if deleteFlag > 0 calls operator delete 0x00103E70.
    // ========================================================================
    void dtor1b8db0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));

        std::cerr << "[ATV2:DTOR1B8DB0] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00454930u);
            uint32_t p = READ32(thisPtr + 488u);
            while (p != 0u)
            {
                uint32_t next = READ32(p + 8u);
                callRecompiled4(runtime, rdram, ctx, 0x001B8CB0u, p, 1u, 0u, 0u);
                p = next;
            }

            // Base destructor @ 0x001BF310
            callRecompiled4(runtime, rdram, ctx, 0x001BF310u, thisPtr, 0u, 0u, 0u);

            if (delFlag > 0)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
            }
        }

        SET_GPR_U64(ctx, 2, static_cast<uint64_t>(thisPtr));
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Virtual destructor for Node subclass @ 0x0012B160
    // vtable 0x00452120 (+0x08).
    // Cleans up member at thisPtr + 92 (calls 0x001274F0), calls base dtor 0x00105C10,
    // and if deleteFlag > 0 calls operator delete 0x00103E70.
    // ========================================================================
    void dtor12b160(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const int32_t delFlag = static_cast<int32_t>(GPR_U32(ctx, 5));

        std::cerr << "[ATV2:DTOR12B160] this=0x" << std::hex << thisPtr << " delFlag=" << delFlag << std::dec << std::endl;

        if (thisPtr != 0u)
        {
            WRITE32(thisPtr + 8u, 0x00452120u);
            const uint32_t p92 = READ32(thisPtr + 92u);
            if (p92 != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x001274F0u, p92, 1u, 0u, 0u);
            }

            // Base destructor @ 0x00105C10
            callRecompiled4(runtime, rdram, ctx, 0x00105C10u, thisPtr, 0u, 0u, 0u);

            if (delFlag > 0)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
            }
        }

        SET_GPR_U64(ctx, 2, static_cast<uint64_t>(thisPtr));
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Destructor @ 0x00206DB0 (Scene member)
    // ========================================================================
    void dtor206db0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t thisPtr, uint32_t deleteFlag)
    {
        if (thisPtr == 0u) return;
        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj == nullptr) return;
        *reinterpret_cast<uint32_t *>(obj + 12) = 0x00456930u;
        const uint32_t ptr8 = *reinterpret_cast<const uint32_t *>(obj + 8);
        if (ptr8 != 0u) {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, ptr8, 0u, 0u, 0u);
            *reinterpret_cast<uint32_t *>(obj + 8) = 0u;
        }
        if (static_cast<int32_t>(deleteFlag) > 0) {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }
    }

    void dtorWrapper206db0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t deleteFlag = GPR_U32(ctx, 5);
        dtor206db0(rdram, ctx, runtime, thisPtr, deleteFlag);
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Base Destructor @ 0x00207930 (Scene member array)
    // ========================================================================
    void dtor207930(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t thisPtr, uint32_t deleteFlag)
    {
        if (thisPtr == 0u) return;
        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj == nullptr) return;
        *reinterpret_cast<uint32_t *>(obj + 8) = 0x00456800u;
        const uint32_t arrPtr = *reinterpret_cast<const uint32_t *>(obj + 52);
        if (arrPtr != 0u) {
            const uint32_t count = *reinterpret_cast<const uint32_t *>(obj + 48);
            for (uint32_t i = 0u; i < count; ++i) {
                uint8_t *slotPtr = getMemPtr(rdram, arrPtr + i * 4);
                if (slotPtr != nullptr) {
                    const uint32_t elem = *reinterpret_cast<const uint32_t *>(slotPtr);
                    if (elem != 0u) {
                        dtor206db0(rdram, ctx, runtime, elem, 1u);
                        *reinterpret_cast<uint32_t *>(slotPtr) = 0u;
                    }
                }
            }
            callRecompiled4(runtime, rdram, ctx, 0x00103F20u, arrPtr, 0u, 0u, 0u);
            *reinterpret_cast<uint32_t *>(obj + 52) = 0u;
        }
        callRecompiled4(runtime, rdram, ctx, 0x00105C10u, thisPtr, 0u, 0u, 0u);
        if (static_cast<int32_t>(deleteFlag) > 0) {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }
    }

    void dtorWrapper207930(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t deleteFlag = GPR_U32(ctx, 5);
        dtor207930(rdram, ctx, runtime, thisPtr, deleteFlag);
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Virtual Destructor @ 0x0020C6C0 (Scene subclass)
    // ========================================================================
    void dtor20c6c0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t deleteFlag = GPR_U32(ctx, 5);
        if (thisPtr == 0u) {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }
        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr) {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x00456FC0u;
        }
        dtor207930(rdram, ctx, runtime, thisPtr, 0u);
        if (static_cast<int32_t>(deleteFlag) > 0) {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Base Destructor @ 0x001585A0 (PSXRenderTarget.cpp)
    // ========================================================================
    void dtor1585a0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t thisPtr, uint32_t deleteFlag)
    {
        if (thisPtr == 0u)
            return;

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 0) = 0x00452C60u;

            const uint32_t p4 = *reinterpret_cast<const uint32_t *>(obj + 4);
            if (p4 != 0u)
            {
                uint8_t *p4Ptr = getMemPtr(rdram, p4);
                if (p4Ptr != nullptr)
                {
                    const uint32_t vptr = *reinterpret_cast<const uint32_t *>(p4Ptr + 8);
                    uint8_t *vptrMem = getMemPtr(rdram, vptr);
                    if (vptrMem != nullptr)
                    {
                        const uint32_t vfunc = *reinterpret_cast<const uint32_t *>(vptrMem + 8);
                        if (vfunc != 0u)
                        {
                            callRecompiled4(runtime, rdram, ctx, vfunc, p4, 1u, 0u, 0u);
                        }
                    }
                }
            }

            const uint32_t p8 = *reinterpret_cast<const uint32_t *>(obj + 8);
            if (p8 != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x001274F0u, p8, 1u, 0u, 0u);
            }
        }

        if (static_cast<int32_t>(static_cast<int16_t>(deleteFlag)) > 0)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }
    }

    void dtorWrapper1585a0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t deleteFlag = GPR_U32(ctx, 5);
        std::cerr << "[ATV2:DTOR1585A0] this=0x" << std::hex << thisPtr
                  << " delFlag=" << std::dec << deleteFlag << std::endl;
        dtor1585a0(rdram, ctx, runtime, thisPtr, deleteFlag);
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Virtual Destructor @ 0x001587A0 (PSXRenderTarget.cpp)
    // ========================================================================
    void dtor1587a0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t deleteFlag = GPR_U32(ctx, 5);
        std::cerr << "[ATV2:DTOR1587A0] this=0x" << std::hex << thisPtr
                  << " delFlag=" << std::dec << deleteFlag << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 0) = 0x00452C50u;
        }

        dtor1585a0(rdram, ctx, runtime, thisPtr, 0u);

        if (static_cast<int32_t>(static_cast<int16_t>(deleteFlag)) > 0)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Screen/Scene element constructor @ 0x002046D0 (Scene.cpp)
    // ========================================================================
    void ctor2046d0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        if (thisPtr != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00204460u, thisPtr, 0u, 0u, 0u);

            uint8_t *obj = getMemPtr(rdram, thisPtr);
            if (obj != nullptr)
            {
                *reinterpret_cast<uint32_t *>(obj + 376) = 0x00456CE0u;
                *reinterpret_cast<uint32_t *>(obj + 384) = 0u;
                *reinterpret_cast<uint32_t *>(obj + 388) = 0u;
                *reinterpret_cast<uint32_t *>(obj + 392) = 0u;
                *reinterpret_cast<uint32_t *>(obj + 396) = 0u;
                *reinterpret_cast<uint32_t *>(obj + 400) = 0u;
                *reinterpret_cast<uint32_t *>(obj + 404) = 0u;
            }
        }
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // TerrainZoneInfo constructor @ 0x00202770 (Scene.cpp)
    // ========================================================================
    void ctor202770(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        if (thisPtr != 0u)
        {
            uint8_t *obj = getMemPtr(rdram, thisPtr);
            if (obj != nullptr)
            {
                const uint32_t oneF = 0x3F800000u; // 1.0f
                *reinterpret_cast<uint32_t *>(obj + 28) = 0x00456DD0u;
                *reinterpret_cast<uint32_t *>(obj + 4) = 0u;        // drag = 0.0f
                *reinterpret_cast<uint32_t *>(obj + 0) = oneF;      // friction = 1.0f
                *reinterpret_cast<uint32_t *>(obj + 8) = oneF;      // traction = 1.0f
                *reinterpret_cast<uint32_t *>(obj + 12) = oneF;     // powerslide = 1.0f
                *reinterpret_cast<uint32_t *>(obj + 16) = oneF;     // braking = 1.0f
                *reinterpret_cast<uint32_t *>(obj + 20) = 1u;       // generateDust = 1
                *reinterpret_cast<uint32_t *>(obj + 24) = 1u;       // generateDirt = 1
            }
        }
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // WeatherInfo virtual destructor @ 0x00209040 (QuadzillaScene.cpp)
    // ========================================================================
    void dtor209040(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t delFlag = GPR_U32(ctx, 5);
        std::cerr << "[ATV2:DTOR209040] this=0x" << std::hex << thisPtr
                  << " delFlag=" << std::dec << delFlag << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            *reinterpret_cast<uint32_t *>(obj + 8) = 0x004571E0u;

            const uint32_t count = *reinterpret_cast<const uint32_t *>(obj + 56);
            for (uint32_t i = 0; i < count; ++i)
            {
                const uint32_t elem = callRecompiled4(runtime, rdram, ctx, 0x0020C850u, thisPtr + 56u, i, 0u, 0u);
                if (elem != 0u)
                {
                    callRecompiled4(runtime, rdram, ctx, 0x00103DF0u, elem, 0x0042AA40u, 436u, 0u);
                }
            }

            const uint32_t p76 = *reinterpret_cast<const uint32_t *>(obj + 76);
            if (p76 != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103F20u, p76, 0u, 0u, 0u);
                *reinterpret_cast<uint32_t *>(obj + 76) = 0u;
            }

            const uint32_t p64 = *reinterpret_cast<const uint32_t *>(obj + 64);
            if (p64 != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103F20u, p64, 0u, 0u, 0u);
            }
        }

        callRecompiled4(runtime, rdram, ctx, 0x00105C10u, thisPtr, 0u, 0u, 0u);

        if (static_cast<int32_t>(static_cast<int16_t>(delFlag)) > 0)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Utility @ 0x0034A340
    // Decrements *a0 by a1, returns 0x004FBE30 + (*a0 * 8)
    // ========================================================================
    void util34a340(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t a0 = GPR_U32(ctx, 4);
        const uint32_t a1 = GPR_U32(ctx, 5);

        uint32_t val = READ32(a0);
        val -= a1;
        WRITE32(a0, val);

        const uint32_t ret = 0x004FBE30u + (val * 8u);
        SET_GPR_U64(ctx, 2, static_cast<uint64_t>(ret));
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Screen/Scene element constructor @ 0x002047F0 (Scene.cpp)
    // ========================================================================
    void ctor2047f0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        if (thisPtr != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00204460u, thisPtr, 0u, 0u, 0u);

            uint8_t *obj = getMemPtr(rdram, thisPtr);
            if (obj != nullptr)
            {
                *reinterpret_cast<uint32_t *>(obj + 376) = 0x00456CD0u;
                std::memcpy(obj + 384, "body\0", 5);
                *reinterpret_cast<uint32_t *>(obj + 904) = 0u;
            }
        }
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // VisibilityQuadTreeNode constructor @ 0x00162800 (VisibilityQuadTree.cpp)
    // ========================================================================
    void ctor162800(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        if (thisPtr != 0u)
        {
            uint8_t *obj = getMemPtr(rdram, thisPtr);
            if (obj != nullptr)
            {
                *reinterpret_cast<uint32_t *>(obj + 16) = 0x00452D50u;
                *reinterpret_cast<uint32_t *>(obj + 0) = 0u;
                *reinterpret_cast<uint32_t *>(obj + 4) = 0u;
                *reinterpret_cast<uint32_t *>(obj + 8) = 0u;
                *reinterpret_cast<uint32_t *>(obj + 12) = 0xFF7FFFFFu;

                uint8_t *globPtr = getMemPtr(rdram, 0x00489020u);
                if (globPtr != nullptr)
                {
                    const uint32_t a2 = *reinterpret_cast<const uint32_t *>(globPtr);
                    uint8_t *a2Ptr = getMemPtr(rdram, a2 + 2184u);
                    if (a2Ptr != nullptr)
                    {
                        *reinterpret_cast<uint32_t *>(a2Ptr) += 1u;
                    }
                }
            }
        }
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // VisibilityQuadTreeLeaf constructor @ 0x0016DE70 (VisibilityQuadTree.cpp)
    // ========================================================================
    void ctor16de70(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        if (thisPtr != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00162800u, thisPtr, 0u, 0u, 0u);
            uint8_t *obj = getMemPtr(rdram, thisPtr);
            if (obj != nullptr)
            {
                *reinterpret_cast<uint32_t *>(obj + 16) = 0x004531A0u;
            }
        }
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // VisibilityQuadTreeLeaf factory @ 0x0016DE30 (VisibilityQuadTree.cpp)
    // ========================================================================
    void factory16de30(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t obj = callRecompiled4(runtime, rdram, ctx, 0x00103E30u, 20u, 0x00420420u, 91u, 0u);
        if (obj != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x0016DE70u, obj, 0u, 0u, 0u);
        }
        SET_GPR_U64(ctx, 2, static_cast<uint64_t>(obj));
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Screen/Scene element constructor @ 0x002049F0 (Scene.cpp)
    // ========================================================================
    void ctor2049f0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        if (thisPtr != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00204460u, thisPtr, 0u, 0u, 0u);

            uint8_t *obj = getMemPtr(rdram, thisPtr);
            if (obj != nullptr)
            {
                *reinterpret_cast<uint32_t *>(obj + 376) = 0x00456CB0u;
                *reinterpret_cast<uint32_t *>(obj + 384) = 0u;
            }
        }
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Screen/Scene element constructor @ 0x00204910 (Scene.cpp)
    // ========================================================================
    void ctor204910(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        if (thisPtr != 0u)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00204460u, thisPtr, 0u, 0u, 0u);

            uint8_t *obj = getMemPtr(rdram, thisPtr);
            if (obj != nullptr)
            {
                *reinterpret_cast<uint32_t *>(obj + 376) = 0x00456CC0u;
                std::memcpy(obj + 384, "body\0", 5);
            }
        }
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Base Destructor @ 0x001C0570 (PSXSurfaceShader.cpp base / Resource list)
    // ========================================================================
    void dtor1c0570(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t thisPtr, uint32_t deleteFlag)
    {
        if (thisPtr == 0u)
            return;

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj == nullptr)
            return;

        // 0x001C0598: vtable = 0x00454AD8
        *reinterpret_cast<uint32_t *>(obj + 16396) = 0x00454AD8u;

        // 0x001C05A0: member at +8
        const uint32_t member8 = *reinterpret_cast<const uint32_t *>(obj + 8);
        if (member8 != 0u)
        {
            uint8_t *m8Ptr = getMemPtr(rdram, member8);
            if (m8Ptr != nullptr)
            {
                const uint32_t vt = *reinterpret_cast<const uint32_t *>(m8Ptr + 8);
                uint8_t *vtPtr = getMemPtr(rdram, vt);
                if (vtPtr != nullptr)
                {
                    const uint32_t fn = *reinterpret_cast<const uint32_t *>(vtPtr + 16);
                    if (fn != 0u)
                    {
                        callRecompiled4(runtime, rdram, ctx, fn, member8, 0u, 0u, 0u);
                    }
                }
            }
        }

        // 0x001C05BC: check global singleton pointer at 0x00499038
        uint8_t *globMem = getMemPtr(rdram, 0x00499038u);
        if (globMem != nullptr)
        {
            if (*reinterpret_cast<const uint32_t *>(globMem) == thisPtr)
            {
                *reinterpret_cast<uint32_t *>(globMem) = 0u;
            }
        }

        // 0x001C0608: loop over array of elements at this + 12
        const uint32_t count = *reinterpret_cast<const uint32_t *>(obj + 0);
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t elem = *reinterpret_cast<const uint32_t *>(obj + 12 + i * 4);
            if (elem != 0u)
            {
                uint8_t *elemPtr = getMemPtr(rdram, elem);
                if (elemPtr != nullptr)
                {
                    const uint32_t vt = *reinterpret_cast<const uint32_t *>(elemPtr + 40);
                    uint8_t *vtPtr = getMemPtr(rdram, vt);
                    if (vtPtr != nullptr)
                    {
                        const uint32_t dtorFn = *reinterpret_cast<const uint32_t *>(vtPtr + 8);
                        if (dtorFn != 0u)
                        {
                            callRecompiled4(runtime, rdram, ctx, dtorFn, elem, 1u, 0u, 0u);
                        }
                    }
                }
            }
        }

        // 0x001C0620: delete this if deleteFlag > 0
        if (static_cast<int32_t>(deleteFlag) > 0)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }
    }

    void dtorWrapper1c0570(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t deleteFlag = GPR_U32(ctx, 5);
        std::cerr << "[ATV2:DTOR1C0570] this=0x" << std::hex << thisPtr
                  << " delFlag=" << std::dec << deleteFlag << std::endl;
        dtor1c0570(rdram, ctx, runtime, thisPtr, deleteFlag);
        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // ========================================================================
    // Virtual Destructor @ 0x001BABA0 (PSXSurfaceShader.cpp)
    // ========================================================================
    void dtor1baba0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t thisPtr = GPR_U32(ctx, 4);
        const uint32_t deleteFlag = GPR_U32(ctx, 5);
        std::cerr << "[ATV2:DTOR1BABA0] this=0x" << std::hex << thisPtr
                  << " delFlag=" << std::dec << deleteFlag << std::endl;

        if (thisPtr == 0u)
        {
            SET_GPR_U64(ctx, 2, 0u);
            ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
            return;
        }

        uint8_t *obj = getMemPtr(rdram, thisPtr);
        if (obj != nullptr)
        {
            // Set derived vtable
            *reinterpret_cast<uint32_t *>(obj + 16396) = 0x004549F8u;

            const uint32_t ptr16400 = *reinterpret_cast<const uint32_t *>(obj + 16400);
            if (ptr16400 != 0u)
            {
                callRecompiled4(runtime, rdram, ctx, 0x00103DF0u, ptr16400, 0x00424970u, 677u, 0u);
            }
        }

        // Call base destructor with deleteFlag = 0
        dtor1c0570(rdram, ctx, runtime, thisPtr, 0u);

        // Free this object if deleteFlag > 0
        if (static_cast<int32_t>(deleteFlag) > 0)
        {
            callRecompiled4(runtime, rdram, ctx, 0x00103E70u, thisPtr, 0u, 0u, 0u);
        }

        SET_GPR_U64(ctx, 2, thisPtr);
        ctx->pc = static_cast<uint32_t>(GPR_U64(ctx, 31));
    }

    // MSL soft-float dptoul (Double-Precision to Unsigned Long) @ 0x003B5590
    void dptoul3b5590(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        const uint64_t a0Bits = GPR_U64(ctx, 4);
        double val = 0.0;
        std::memcpy(&val, &a0Bits, sizeof(double));

        uint32_t result = 0;
        if (!std::isnan(val) && val > 0.0)
        {
            if (val >= 4294967295.0)
                result = 0xFFFFFFFFu;
            else
                result = static_cast<uint32_t>(val);
        }

        SET_GPR_U64(ctx, 2, static_cast<uint64_t>(result));
        ctx->pc = GPR_U32(ctx, 31);
    }

    // Diagnostics wrapper for func_256A70 (child teardown)
    void func256a70_wrapper(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        std::cerr << "[ATV2:256A70] ENTER: a0=0x" << std::hex << GPR_U32(ctx, 4) << std::dec << std::endl;
        sub_00256A70_0x256a70(rdram, ctx, runtime);
        std::cerr << "[ATV2:256A70] EXIT: pc=0x" << std::hex << ctx->pc << std::dec << std::endl;
    }

    // Diagnostics wrapper for func_2FF730
    void func2ff730_wrapper(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        std::cerr << "[ATV2:2FF730] ENTER: a0=0x" << std::hex << GPR_U32(ctx, 4) << " a1=0x" << GPR_U32(ctx, 5) << std::dec << std::endl;
        sub_002FF730_0x2ff730(rdram, ctx, runtime);
        std::cerr << "[ATV2:2FF730] EXIT: pc=0x" << std::hex << ctx->pc << std::dec << std::endl;
    }

    // Diagnostics wrapper for func_1DFB00 (UI resource archive & viewport init)
    void func1dfb00_wrapper(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t gameObj = GPR_U32(ctx, 4);
        uint32_t renderer = 0;
        uint32_t mode = 0;
        uint32_t snd = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, 0x004975B0u));
        uint32_t vpBefore = 0;
        if (gameObj != 0) {
            renderer = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x298u));
            if (renderer != 0) {
                mode = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, renderer + 0x4u));
            }
            vpBefore = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x29Cu));
        }
        std::cerr << "[ATV2:1DFB00] ENTER: gameObj=0x" << std::hex << gameObj 
                  << " renderer=0x" << renderer 
                  << " mode=" << std::dec << mode 
                  << " vpBefore=0x" << std::hex << vpBefore
                  << " snd=0x" << snd << std::dec << std::endl;

        sub_001DFB00_0x1dfb00(rdram, ctx, runtime);

        uint32_t vpAfter = 0;
        if (gameObj != 0) {
            vpAfter = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x29Cu));
        }
        std::cerr << "[ATV2:1DFB00] YIELD/EXIT: pc=0x" << std::hex << ctx->pc 
                  << " v0=" << GPR_U32(ctx, 2) 
                  << " vpAfter=0x" << vpAfter << std::dec << std::endl;
    }

    // ATV2_TIMER0_PER_FRAME=1: old behaviour, the host pump bumps the timer-0 counter 0x488F38
    // once per host frame instead of at 10 Hz real time.
    bool timer0CounterPerHostFrame()
    {
        static const bool s_perFrame = []() {
            const char *v = std::getenv("ATV2_TIMER0_PER_FRAME");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        return s_perFrame;
    }

    // Frame limiter hook @ 0x0015DA90
    // Runs on the main EE thread every frame.
    // Phase 1: Create viewport by calling func_1DFB00 (skipped because we bypass video).
    // Phase 2: Tick FrontendManager (Update + Draw) every frame.
    void frameLimiter15da90(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static int s_frameCount = 0;
        static bool s_state5Triggered = false;
        // The game's own main loop drives the front end by default. ATV2_NATIVE_LOOP=0 restores the
        // hand-driven FrontendManager harness below (bring-up aid): it forces every UI node visible
        // each frame, which leaves closed dialogs on screen and the profile-save / network-config
        // pages unusable.
        static const bool s_nativeLoop = []() {
            const char *v = std::getenv("ATV2_NATIVE_LOOP");
            return v == nullptr || v[0] == '\0' || v[0] != '0';
        }();
        ++s_frameCount;

        // 0x488F38 counts timer-0 compare interrupts (handler FUN_00156590: RCNT0 %= 0xE100,
        // counter++), i.e. one per 100 ms at RCNT0's 576 kHz; the game's millisecond clock is
        // (RCNT0 + counter * 0xE100) / 576. The runtime doesn't dispatch that interrupt, and bumping
        // the counter once per host frame made the clock run several times too fast (network
        // timeouts expired before replies were processed). Advance it at the real 10 Hz instead.
        if (!timer0CounterPerHostFrame())
        {
            static const auto s_start = std::chrono::steady_clock::now();
            static uint64_t s_ticks = 0;
            const uint64_t ticks = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                             std::chrono::steady_clock::now() - s_start)
                                                             .count()) /
                                   100u;
            if (ticks > s_ticks)
            {
                uint32_t *counter = reinterpret_cast<uint32_t *>(getMemPtr(rdram, 0x00488F38u));
                *counter += static_cast<uint32_t>(ticks - s_ticks);
                s_ticks = ticks;
            }
        }

        // ATV2_STACKSCAN=1: at frames 30 and 90, list guest return addresses on the stack
        // (words preceded by jal/jalr) to recover the call chain that reached the limiter.
        static const bool s_stackScan = []() {
            const char *v = std::getenv("ATV2_STACKSCAN");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        if (s_stackScan && (s_frameCount == 30 || s_frameCount == 90))
        {
            const uint32_t sp = GPR_U32(ctx, 29);
            std::cerr << "[ATV2:STACKSCAN] frame=" << s_frameCount << " sp=0x" << std::hex << sp
                      << " ra=0x" << GPR_U32(ctx, 31) << std::dec << std::endl;
            for (uint32_t a = sp & ~3u; a < 0x01FFFFF0u; a += 4u)
            {
                const uint32_t w = *reinterpret_cast<uint32_t *>(getMemPtr(rdram, a));
                if (w < 0x00100008u || w >= 0x003C0000u || (w & 3u) != 0u)
                    continue;
                const uint32_t call = *reinterpret_cast<uint32_t *>(getMemPtr(rdram, w - 8u));
                const bool isJal = (call >> 26) == 3u;
                const bool isJalr = (call & 0xFC00003Fu) == 0x00000009u;
                if (!isJal && !isJalr)
                    continue;
                std::cerr << "[ATV2:STACKSCAN]   @0x" << std::hex << a << " ret=0x" << w;
                if (isJal)
                    std::cerr << " jal 0x" << ((call & 0x03FFFFFFu) << 2);
                else
                    std::cerr << " jalr";
                std::cerr << std::dec << std::endl;
            }
        }

        if (!s_nativeLoop && s_frameCount > 60)
        {
            uint32_t gameObj = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, 0x00404818u));
            if (gameObj != 0u)
            {
                // --- Phase 2: Tick FrontendManager every frame ---
                uint32_t feMgr = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x2B0u));
                if (feMgr != 0u)
                {
                    uint32_t *pCur = reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0xCCu));
                    uint32_t *pPend = reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0xC8u));
                    uint32_t *pFlags20 = reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0x20u));
                    uint32_t *pFlags24 = reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0x24u));

                    auto enableNodeTree = [&](auto& self, uint32_t node) -> void {
                        if (node == 0u || node >= 0x02000000u) return;
                        *reinterpret_cast<uint32_t*>(getMemPtr(rdram, node + 0x20u)) |= 0xAu;
                        *reinterpret_cast<uint32_t*>(getMemPtr(rdram, node + 0x24u)) |= 0xAu;
                        uint8_t *p28 = reinterpret_cast<uint8_t*>(getMemPtr(rdram, node + 0x28u));
                        uint8_t *p29 = reinterpret_cast<uint8_t*>(getMemPtr(rdram, node + 0x29u));
                        *p28 = 0xFFu;
                        *p29 = (*p29 | 0x01u) & ~0x0Cu;
                        for (uint32_t ch = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, node + 0x18u));
                             ch != 0u && ch < 0x02000000u;
                             ch = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, ch + 0x14u))) {
                            self(self, ch);
                        }
                    };

                    uint32_t root = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x10u));
                    uint32_t feScreen = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0xB8u));
                    uint32_t feMenu = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0xB0u));
                    uint32_t feTitle = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0xB4u));

                    enableNodeTree(enableNodeTree, feMgr);
                    if (root != 0u) enableNodeTree(enableNodeTree, root);
                    if (feScreen != 0u) enableNodeTree(enableNodeTree, feScreen);
                    if (feMenu != 0u) enableNodeTree(enableNodeTree, feMenu);
                    if (feTitle != 0u) enableNodeTree(enableNodeTree, feTitle);

                    // Status logging every 30 frames:
                    if (s_frameCount % 30 == 0)
                    {
                        uint32_t renderer = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x298u));
                        uint32_t mode = (renderer != 0u) ? *reinterpret_cast<uint32_t*>(getMemPtr(rdram, renderer + 0x4u)) : 0u;
                        uint32_t vp = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x29Cu));
                        uint32_t snd = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, 0x004975B0u));
                        std::cerr << "[ATV2:STATUS] frame=" << s_frameCount
                                  << " gameObj=0x" << std::hex << gameObj
                                  << " renderer=0x" << renderer
                                  << " mode=" << std::dec << mode
                                  << " vp=0x" << std::hex << vp
                                  << " snd=0x" << snd
                                  << " cur=" << std::dec << *pCur
                                  << " pend=" << *pPend
                                  << " feMenu=0x" << std::hex << feMenu
                                  << " feTitle=0x" << feTitle
                                  << " root=0x" << root
                                  << std::endl;
                    }

                    // Drive state transitions:
                    // State 5 (0x05) triggers Case 5 (0x25663C), which runs func_257910 and func_255F30.
                    // func_255F30 creates TitleScreen (0x004555E0) into feMgr + 0xB4, and attaches it to root (gameObj + 0x10).
                    if (s_frameCount >= 62 && !s_state5Triggered && *pCur == 0u && *pPend == 0u)
                    {
                        std::cerr << "[ATV2:STATE] Triggering transition State 0 -> State 5 (0x5) via pendState=5" << std::endl;
                        *pPend = 5u;
                        s_state5Triggered = true;
                    }

                    auto runGuestToCompletion = [&](uint32_t entryPc, uint32_t a0, float f12 = 0.0f) -> uint32_t
                    {
                        R5900Context savedCtx = *ctx;
                        SET_GPR_U32(ctx, 4, a0);
                        ctx->f[12] = f12;
                        const uint32_t targetRa = 0x0015DA90u;
                        SET_GPR_U32(ctx, 31, targetRa);
                        ctx->pc = entryPc;

                        int steps = 0;
                        uint32_t prevVp = (gameObj != 0u) ? *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x29Cu)) : 0u;
                        while (ctx->pc != targetRa && ctx->pc != 0u && !runtime->isStopRequested() && steps < 20000000)
                        {
                            ++steps;
                            PS2Runtime::RecompiledFunction fn = runtime->lookupFunction(ctx->pc);
                            if (!fn)
                            {
                                std::cerr << "[ATV2:RUN] Missing function for PC 0x" << std::hex << ctx->pc << std::dec << std::endl;
                                break;
                            }
                            fn(rdram, ctx, runtime);

                            if (gameObj != 0u)
                            {
                                uint32_t curVp = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x29Cu));
                                if (curVp != 0u && prevVp == 0u)
                                {
                                    std::cerr << "[ATV2:VIEWPORT_ALLOCATED!] gameObj->vp=0x" << std::hex << curVp 
                                              << " at step " << std::dec << steps 
                                              << " pc=0x" << std::hex << ctx->pc << std::dec << std::endl;
                                    prevVp = curVp;
                                }
                            }
                        }
                        uint32_t finalPc = ctx->pc;
                        if (steps >= 50000)
                        {
                            std::cerr << "[ATV2:RUN] entryPc=0x" << std::hex << entryPc 
                                      << " steps=" << std::dec << steps 
                                      << " finalPc=0x" << std::hex << finalPc << std::dec << std::endl;
                        }
                        *ctx = savedCtx;
                        return finalPc;
                    };

                    // --- Phase 1: Initialize 3D Viewport via func_1DFB00 once ---
                    static bool s_vpInitialized = false;
                    if (!s_vpInitialized && s_frameCount >= 61)
                    {
                        uint32_t vp = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x29Cu));
                        if (vp == 0u)
                        {
                            std::cerr << "[ATV2:VIEWPORT] Initializing 3D Viewport via func_1DFB00..." << std::endl;
                            runGuestToCompletion(0x001DFB00u, gameObj);
                            vp = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x29Cu));
                            std::cerr << "[ATV2:VIEWPORT] Result vp=0x" << std::hex << vp << std::dec << std::endl;
                        }
                        s_vpInitialized = true;
                    }

                    // 1. Tick FrontendManager::Update(feMgr, dt=0.03333333f)
                    uint32_t updRet = runGuestToCompletion(0x00256110u, feMgr, 0.03333333f);
                    feMenu = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0xB0u));

                    if (s_frameCount >= 60 && s_frameCount <= 75)
                    {
                        uint32_t feScr = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0xB8u));
                        uint32_t vp = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x29Cu));
                        std::cerr << "[ATV2:UPDATE_COMPLETION] frame=" << s_frameCount
                                  << " updRet=0x" << std::hex << updRet
                                  << " vp=0x" << vp
                                  << " cur=" << std::dec << *pCur
                                  << " pend=" << *pPend
                                  << " feMenu=0x" << std::hex << feMenu << std::endl;
                    }

                    // 2. Advance Node::Update(dt=0.03333333f) on scene trees so transformations/animations step
                    if (root != 0u)
                    {
                        runGuestToCompletion(0x001064B0u, root, 0.03333333f);
                    }
                    runGuestToCompletion(0x001064B0u, feMgr, 0.03333333f);

                    // 3. Draw root via Node::Draw (0x00106370) - traverses full scene hierarchy
                    if (root != 0u)
                    {
                        runGuestToCompletion(0x00106370u, root);
                    }

                    // 4. Draw feMgr via Node::Draw (0x00106370)
                    runGuestToCompletion(0x00106370u, feMgr);

                    // 5. Draw feTitle directly via TitleScreen::Render (0x001D3F50)
                    feTitle = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, feMgr + 0xB4u));
                    if (feTitle != 0u)
                    {
                        enableNodeTree(enableNodeTree, feTitle);
                        runGuestToCompletion(0x001D3F50u, feTitle);
                    }

                    // 6. Also explicitly ensure feMenu is drawn if instantiated:
                    if (feMenu != 0u)
                    {
                        runGuestToCompletion(0x00106370u, feMenu);
                    }

                    // 7. Draw feScreen directly (via 0x0025C680)
                    if (feScreen != 0u)
                    {
                        runGuestToCompletion(0x0025C680u, feScreen);
                    }

                    static int s_feTicks = 0;
                    if (++s_feTicks <= 10 || s_feTicks % 60 == 0)
                    {
                        uint32_t viewport = *reinterpret_cast<uint32_t*>(getMemPtr(rdram, gameObj + 0x29Cu));
                        std::cerr << "[ATV2:FE_TICK] tick=" << s_feTicks
                                  << " feMgr=0x" << std::hex << feMgr
                                  << " cur=" << std::dec << *pCur
                                  << " pend=" << *pPend
                                  << " feMenu=0x" << std::hex << feMenu
                                  << " feScreen=0x" << std::hex << feScreen
                                  << " viewport=0x" << viewport
                                  << std::dec << std::endl;
                    }
                }
            }
        }
        if (s_nativeLoop && s_frameCount == 120)
        {
            auto rd32 = [&](uint32_t a) -> uint32_t {
                if (a == 0u || a >= 0x02000000u) return 0u;
                return *reinterpret_cast<uint32_t *>(getMemPtr(rdram, a));
            };
            const uint32_t gameObj = rd32(0x00404818u);
            const uint32_t core = rd32(gameObj + 0x284u);
            const uint32_t coreVt = rd32(core);
            std::cerr << std::hex << "[ATV2:TREE] gameObj=0x" << gameObj << " core(+0x284)=0x" << core
                      << " coreVt=0x" << coreVt;
            for (uint32_t off : {0x10u, 0x34u, 0x48u, 0x50u})
                std::cerr << " vt+0x" << off << "=0x" << rd32(coreVt + off);
            std::cerr << " root(+0x10)=0x" << rd32(gameObj + 0x10u)
                      << " feMgr(+0x2b0)=0x" << rd32(gameObj + 0x2B0u)
                      << " +0x2b4=0x" << rd32(gameObj + 0x2B4u) << std::dec << std::endl;
            const uint32_t root = rd32(gameObj + 0x10u);
            int shown = 0;
            for (uint32_t n = rd32(root + 0x18u); n != 0u && shown < 40; n = rd32(n + 0x14u), ++shown)
            {
                const uint32_t vt = rd32(n + 8u);
                std::cerr << std::hex << "[ATV2:TREE]   child=0x" << n << " vt=0x" << vt
                          << " flags20=0x" << rd32(n + 0x20u) << " flags24=0x" << rd32(n + 0x24u)
                          << " b29=0x" << (rd32(n + 0x28u) >> 8 & 0xFFu)
                          << " upd(+0x44)=0x" << rd32(vt + 0x44u)
                          << " draw(+0x48)=0x" << rd32(vt + 0x48u) << std::dec << std::endl;
            }
            std::cerr << "[ATV2:TREE] root flags24=0x" << std::hex << rd32(root + 0x24u)
                      << " flags20=0x" << rd32(root + 0x20u) << std::dec << std::endl;
        }

        ctx->pc = 0x0015DA90u;
        sub_0015DA90_0x15da90(rdram, ctx, runtime);
    }

    // PSXVideoCard::Flip @ 0x0015E370: DMA sync, then CSR FINISH handshake (clears FINISH, kicks a
    // GIF FINISH packet, spins on CSR), frame timing, the limiter 0x15DA90 (only when the frame took
    // under 2 vsyncs), display registers (0x15E8E0), DMA sync. With the GS worker (ATV_VU1_THREAD)
    // the FINISH packet sits behind the whole frame's VIF1/VU1 work, so the spin made the EE wait
    // for the worker every frame. As ATV1 (a4248f2), Flip now waits only for the work queued before
    // the previous Flip, and its FINISH poll (ld CSR at 0x15E4A0) reads FINISH at once; its display
    // register writes are queued behind the frame's drawing, so the buffers still swap only after
    // the frame is drawn. ATV_FLIP_FENCE_NOW=1 waits for this frame's work instead.
    // ATV_TRACE_FPS=1: game frames (Flips) per wall second, every 5 s.
    PS2Runtime::RecompiledFunction g_videoCardFlipOrig = nullptr;
    PS2Runtime *g_flipRuntime = nullptr;
    bool g_flipFenceNow = false;

    uint64_t csrReadForFlip(uint32_t pc, uint64_t value)
    {
        if (pc == 0x0015E4A0u && !g_flipFenceNow && g_flipRuntime && g_flipRuntime->memory().gsWorkerEnabled())
            return value | 2u; // FINISH
        return value;
    }

    void videoCardFlip15e370(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!g_raceLoadStarted)
        {
            const uint32_t feState = READ32(0x005425E0u + 0xCCu); // 5 loading an event, 2 racing
            g_raceLoadStarted = feState == 5u || feState == 2u;
        }
        static const bool s_traceFps = std::getenv("ATV_TRACE_FPS") != nullptr;
        if (s_traceFps)
        {
            static auto s_fpsStart = std::chrono::steady_clock::now();
            static uint32_t s_fpsFrames = 0u;
            ++s_fpsFrames;
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - s_fpsStart).count();
            if (secs >= 5.0)
            {
                std::fprintf(stderr, "[fps] game frames/s=%.1f\n", s_fpsFrames / secs);
                s_fpsStart = std::chrono::steady_clock::now();
                s_fpsFrames = 0u;
            }
        }
        PS2Memory &mem = runtime->memory();
        if (!mem.gsWorkerEnabled())
        {
            g_videoCardFlipOrig(rdram, ctx, runtime);
            return;
        }
        static uint64_t s_prevFlipMark = 0;
        if (g_flipFenceNow)
        {
            mem.fenceGsWork(4);
            g_videoCardFlipOrig(rdram, ctx, runtime);
            return;
        }
        // Flip may yield at a checkpoint and resume without this wrapper; the mark then lands a
        // little earlier, which only shortens the next wait.
        mem.fenceGsWorkUpTo(s_prevFlipMark, 4);
        g_videoCardFlipOrig(rdram, ctx, runtime);
        s_prevFlipMark = mem.gsWorkMark();
    }

    // ---- Opt-in front-end trace (ATV2_TRACE_FE=1) -------------------------------------
    // Wraps a few functions of the front-end start-up path. Each wrapper logs its arguments
    // and return address for a bounded number of calls, then runs the original generated
    // function unchanged. Used to find out why the game's own flow never leaves state 0.
    struct FeTraceEntry
    {
        uint32_t addr;
        const char *name;
        PS2Runtime::RecompiledFunction orig;
        int calls;
        int limit;
    };

    FeTraceEntry g_feTrace[] = {
        {0x001DF2C0u, "StartFrontEnd(0x1DF2C0)", nullptr, 0, 20},
        {0x00256A20u, "FeSetPendingState(0x256A20)", nullptr, 0, 60},
        {0x00256110u, "FeUpdate(0x256110)", nullptr, 0, 4},
        {0x001DF080u, "GameInitA(0x1DF080)", nullptr, 0, 20},
        {0x001DF5D0u, "StartFE_AfterInit(0x1DF5D0)", nullptr, 0, 20},
        {0x00176B60u, "VifRendererCtor(0x176B60)", nullptr, 0, 8},
        {0x00176D60u, "VifRendererInit(0x176D60)", nullptr, 0, 8},
        {0x0016B900u, "RenderNodeInit(0x16B900)", nullptr, 0, 8},
        {0x00177190u, "VifRendererDraw(0x177190)", nullptr, 0, 4},
        {0x001D0780u, "DrawListCtor(0x1D0780)", nullptr, 0, 8},
        {0x00157900u, "GsRendererVt24(0x157900)", nullptr, 0, 8},
        {0x00157A60u, "GsRendererKick(0x157A60)", nullptr, 0, 8},
        {0x00157510u, "GsDmaSend(0x157510)", nullptr, 0, 8},
        {0x0035A420u, "FlushCache(0x35A420)", nullptr, 0, 8},
        {0x00187B20u, "Water_187B20", nullptr, 0, 6},
        {0x00187BC4u, "WaterInit(0x187BC4)", nullptr, 0, 6},
        {0x00188274u, "Water_188274", nullptr, 0, 6},
        {0x00188390u, "Water_188390", nullptr, 0, 6},
        {0x00188680u, "WaterBuild(0x188680)", nullptr, 0, 6},
        {0x00186CA0u, "GsStoreImage(0x186CA0)", nullptr, 0, 12},
        {0x00186EA0u, "GsStoreImageB(0x186EA0)", nullptr, 0, 12},
        {0x0025B7C0u, "MovieStart(0x25B7C0)", nullptr, 0, 8},
        {0x0018A510u, "MediaStartPlayback(0x18A510)", nullptr, 0, 8},
        {0x0018AC80u, "MediaIsFinished(0x18AC80)", nullptr, 0, 6},
        {0x0018AC20u, "MediaStop(0x18AC20)", nullptr, 0, 8},
        {0x0018C7C0u, "MpegPictureUpdate(0x18C7C0)", nullptr, 0, 4},
        {0x00256DF0u, "SplashUpdate(0x256DF0)", nullptr, 0, 12},
    };

    template <int I>
    void feTraceWrap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        FeTraceEntry &e = g_feTrace[I];
        if (I == 0 && e.calls == 0)
        {
            g_ps2TraceDispatchRemaining = 50000u;
        }
        if (++e.calls <= e.limit)
        {
            std::cerr << "[ATV2:FETRACE] " << e.name << " call#" << e.calls
                      << " a0=0x" << std::hex << GPR_U32(ctx, 4)
                      << " a1=0x" << GPR_U32(ctx, 5)
                      << " f12=" << ctx->f[12]
                      << " ra=0x" << GPR_U32(ctx, 31)
                      << " sp=0x" << GPR_U32(ctx, 29);
            const uint32_t a0 = GPR_U32(ctx, 4);
            if (a0 >= 0x00100000u && a0 < 0x02000000u)
                std::cerr << " a0[+30]=0x" << READ32(a0 + 0x30u)
                          << " a0[+5c/60/64]=0x" << READ32(a0 + 0x5Cu) << "/0x" << READ32(a0 + 0x60u)
                          << "/0x" << READ32(a0 + 0x64u);
            std::cerr << " [sp+10]=0x" << READ32(GPR_U32(ctx, 29) + 0x10u);
            std::cerr << std::dec << std::endl;
        }
        const uint32_t ra = GPR_U32(ctx, 31);
        const uint32_t slot = GPR_U32(ctx, 29) + 0x10u;
        if (e.orig)
            e.orig(rdram, ctx, runtime);
        if (e.calls <= e.limit)
        {
            std::cerr << "[ATV2:FETRACE] " << e.name << (ctx->pc == ra ? " return#" : " exit#") << e.calls
                      << " pc=0x" << std::hex << ctx->pc << " ra=0x" << GPR_U32(ctx, 31)
                      << " v0=0x" << GPR_U32(ctx, 2) << " sp=0x" << GPR_U32(ctx, 29)
                      << " [entrySp+10]=0x" << READ32(slot) << std::dec << std::endl;
        }
    }

    template <int... Is>
    void registerFeTraceImpl(PS2Runtime &runtime, std::integer_sequence<int, Is...>)
    {
        PS2Runtime::RecompiledFunction wrappers[] = {&feTraceWrap<Is>...};
        for (int i = 0; i < static_cast<int>(sizeof...(Is)); ++i)
        {
            FeTraceEntry &e = g_feTrace[i];
            e.orig = runtime.lookupFunction(e.addr);
            if (!e.orig)
            {
                std::cerr << "[ATV2:FETRACE] no generated function at 0x" << std::hex << e.addr
                          << std::dec << " (" << e.name << "); not wrapped" << std::endl;
                continue;
            }
            runtime.replaceFunction(e.addr, wrappers[i]);
        }
    }

    // Screen fade 0x15FE20 computes alpha = cvt.w.s(fade * 128 + 0.5) with fade += elapsed / duration.
    // Callers pass duration 0: the R5900 returns +FLT_MAX for x/0 and cvt.w.s saturates to 0x7FFFFFFF,
    // ending the loop after one step. Host IEEE gives inf -> 0x80000000, so the loop never exits.
    // The converted alpha is stored at caller sp+0xD8 right before `jal 0x160D10` (return 0x16019C).
    PS2Runtime::RecompiledFunction g_fadeStep160d10Orig = nullptr;

    void fadeStep160d10(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (GPR_U32(ctx, 31) == 0x0016019Cu && std::isinf(ctx->f[21]))
        {
            uint32_t *alpha = reinterpret_cast<uint32_t *>(getMemPtr(rdram, GPR_U32(ctx, 29) + 0xD8u));
            if (*alpha == 0x80000000u)
                *alpha = (ctx->f[21] > 0.0f) ? 0x7FFFFFFFu : 0x80000000u;
            ctx->f[21] = std::copysign(std::numeric_limits<float>::max(), ctx->f[21]);
        }
        g_fadeStep160d10Orig(rdram, ctx, runtime);
    }

    // ---- VIF1 I-bit texture streaming (same engine as ATV1 0x1EEEC0) ---------------
    // PSXTextureMapManager::Create (0x176D60) installs 0x174AE0 as the VIF1 INTC
    // handler and builds a FLUSHA+I packet (VIFcode 0x93000000) that the render
    // lists insert ahead of each part needing a new texture. BeginTransfer sends
    // entry 0; each I-bit makes the handler call SendCurrentTransfer (0x178DF0,
    // mgr=*0x48A1C0, a1=0) for the next entry and restart VIF1 (FBRST.STC).
    // The runtime cannot deliver that interrupt mid-stream, so do the send inline.
    constexpr uint32_t kTextureMapManagerPtr = 0x0048A1C0u;
    constexpr uint32_t kSendCurrentTransfer = 0x00178DF0u;
    constexpr uint32_t kTransferListIndex = 0x0048A390u;
    constexpr uint32_t kTextureSendDonePc = 0xFFFFFFE0u;
    PS2Runtime *g_ibitRuntime = nullptr;
    PS2Runtime::RecompiledFunction g_sendCurrentTransfer = nullptr;

    // MARK=1 at the FLUSHA+I stall: the INTC handler (0x174AE0) runs a queued GS
    // download instead of a texture send. Entry = *(mgr+0x144) + *(s16*)(mgr+0x148)*0x20:
    // +0 width, +2/+4 SSAX/SSAY, +6/+8 RRW/RRH, +A PSM (u8), +C SBP, +10 RAM dst,
    // +14 callback(+18, +10), +1C continuation chain. 0x186EA0 sends BITBLTBUF/TRXPOS/
    // TRXREG/TRXDIR=1 and reads the rectangle back over VIF1 (DIR=0); the DMAC1
    // handler (0x174D40) then clears MARK, runs the callback, restarts VIF1 at the
    // continuation and advances the index. The rest of the current VIF1 stream is
    // abandoned. PSXWater builds its VU1 vertex/VIF buffer this way (it draws the
    // VIFcodes as pixel colours and downloads the rectangle).
    void textureDownloadAtIbit(uint8_t *rdram, R5900Context *live, uint32_t mgr)
    {
        static uint32_t s_logs = 0u;
        PS2Runtime *runtime = g_ibitRuntime;
        PS2Memory &mem = runtime->memory();
        auto rd = [&](uint32_t a) { return *reinterpret_cast<uint32_t *>(getMemPtr(rdram, a)); };
        auto rd16 = [&](uint32_t a) { return *reinterpret_cast<uint16_t *>(getMemPtr(rdram, a)); };
        const int16_t idx = static_cast<int16_t>(rd16(mgr + 0x148u));
        const uint32_t entry = rd(mgr + 0x144u) + static_cast<uint32_t>(static_cast<int32_t>(idx)) * 0x20u;
        const uint32_t width = rd16(entry + 0x0u);
        const uint32_t ssax = rd16(entry + 0x2u);
        const uint32_t ssay = rd16(entry + 0x4u);
        const uint32_t rrw = rd16(entry + 0x6u);
        const uint32_t rrh = rd16(entry + 0x8u);
        const uint32_t psm = *getMemPtr(rdram, entry + 0xAu);
        const uint32_t sbp = rd16(entry + 0xCu);
        const uint32_t dst = rd(entry + 0x10u);
        const uint32_t bpp = (psm == 0x3Au || psm == 0x32u || psm == 0x0Au || psm == 0x02u) ? 2u : 4u;
        const uint32_t bytes = bpp * rrw * rrh;

        uint64_t pkt[10] = {};
        pkt[0] = 4ull | (1ull << 15) | (1ull << 60); // NLOOP=4, EOP, PACKED, NREG=1
        pkt[1] = 0xEull;                             // A+D
        pkt[2] = (static_cast<uint64_t>(psm) << 24) | sbp | (static_cast<uint64_t>(width >> 6) << 16);
        pkt[3] = 0x50u; // BITBLTBUF
        pkt[4] = ssax | (static_cast<uint64_t>(ssay) << 16);
        pkt[5] = 0x51u; // TRXPOS
        pkt[6] = rrw | (static_cast<uint64_t>(rrh) << 32);
        pkt[7] = 0x52u; // TRXREG
        pkt[8] = 1u;    // local -> host
        pkt[9] = 0x53u; // TRXDIR
        mem.submitGifPacket(GifPathId::Path2, reinterpret_cast<const uint8_t *>(pkt), sizeof(pkt), true, false);
        uint32_t copied = 0u;
        if (dst != 0u && (dst & 0x01FFFFFFu) + bytes <= PS2_RAM_SIZE)
            copied = runtime->gs().consumeLocalToHostBytes(getMemPtr(rdram, dst), bytes);

        // 0x174D40 completion.
        mem.vif1_regs.mark = 0u;
        const uint32_t callback = rd(entry + 0x14u);
        if (callback != 0u)
        {
            R5900Context tmp = *live;
            SET_GPR_U32(&tmp, 29, (GPR_U32(live, 29) - 0x400u) & ~0xFu);
            HoldEeTimeslices hold;
            callRecompiled4(runtime, rdram, &tmp, callback, rd(entry + 0x18u), dst, 0u, 0u);
        }
        const uint32_t cont = rd(entry + 0x1Cu);
        const bool restart = cont != 0u && *getMemPtr(rdram, cont + 3u) != static_cast<uint8_t>('p');
        *reinterpret_cast<int16_t *>(getMemPtr(rdram, mgr + 0x148u)) = static_cast<int16_t>(idx + 1);
        if (s_logs < 32u)
        {
            ++s_logs;
            std::cerr << "[ATV2:DOWNLOAD] idx=" << idx << std::hex << " sbp=0x" << sbp << std::dec
                      << " w=" << width << std::hex << " psm=0x" << psm << std::dec
                      << " ssa=" << ssax << "," << ssay << " rr=" << rrw << "x" << rrh
                      << std::hex << " dst=0x" << dst << std::dec << " bytes=" << bytes
                      << " copied=" << copied << std::hex << " cb=0x" << callback
                      << " cont=0x" << cont << std::dec << (restart ? "" : " (END)") << std::endl;
        }
        if (restart)
        {
            mem.writeIORegister(0x10009030u, cont & 0x0FFFFFFFu); // D1_TADR
            mem.writeIORegister(0x10009020u, 0u);                 // D1_QWC
            mem.writeIORegister(0x10009000u, 0x1C5u);             // D1_CHCR: chain, TTE, TIE, STR
        }
        g_ps2Vif1AbortStream = true;
    }

    void onVif1TextureIbit(uint32_t vifcode)
    {
        static bool s_inHook = false;
        static uint32_t s_logs = 0u;
        if (s_inHook || g_ibitRuntime == nullptr || g_sendCurrentTransfer == nullptr)
            return;
        uint8_t *rdram = g_ibitRuntime->memory().getRDRAM();
        R5900Context *live = g_ibitRuntime->eeScheduler().currentContext();
        if (rdram == nullptr || live == nullptr)
            return;
        auto rd = [&](uint32_t a) { return *reinterpret_cast<uint32_t *>(getMemPtr(rdram, a)); };
        const uint32_t mgr = rd(kTextureMapManagerPtr);
        if (mgr == 0u)
            return;
        if (g_ibitRuntime->memory().vif1_regs.mark == 1u)
        {
            s_inHook = true;
            textureDownloadAtIbit(rdram, live, mgr);
            s_inHook = false;
            return;
        }

        const uint32_t idxBefore = rd(kTransferListIndex);
        R5900Context tmp = *live;
        SET_GPR_U32(&tmp, 29, (GPR_U32(live, 29) - 0x400u) & ~0xFu);
        SET_GPR_U32(&tmp, 4, mgr);
        SET_GPR_U32(&tmp, 5, 0u);
        SET_GPR_U32(&tmp, 31, kTextureSendDonePc);
        tmp.pc = kSendCurrentTransfer;
        {
            HoldEeTimeslices hold;
            struct InHookScope
            {
                bool &flag;
                explicit InHookScope(bool &f) : flag(f) { flag = true; }
                ~InHookScope() { flag = false; }
            } inHook(s_inHook);
            g_sendCurrentTransfer(rdram, &tmp, g_ibitRuntime);
        }

        if (s_logs < 24u)
        {
            ++s_logs;
            std::cerr << "[ATV2:IBIT] vifcode=0x" << std::hex << vifcode << std::dec
                      << " texture send " << idxBefore << "->" << rd(kTransferListIndex)
                      << " exitPc=0x" << std::hex << tmp.pc << std::dec << std::endl;
        }
    }

    // ---- GS worker (ATV_VU1_THREAD): I-bit texture sends -----------------------------------------
    // The worker cannot run guest code. Right before a VIF1 chain is queued, run every remaining
    // SendCurrentTransfer of the send list on the EE (same calls, same order as the I-bits would
    // make them) and capture each one's GIF upload; the worker replays one capture per I-bit and
    // sends what is left after the chain (as ATV1). Send list: index 0x48A390, entries at 0x48A394
    // (count 0x48F394) when *(*0x489038+0x54) == 1, else 0x48F398 (count 0x494398).
    // `tail` is what an I-bit past the end of the list sends: SendCurrentTransfer then runs the
    // manager's fixed GIF chain (0x187650 with *(mgr+0x88)). It is captured once per chain address.
    constexpr uint32_t kRenderEnginePtr = 0x00489038u;

    void precaptureIbitTextures(PS2Memory::GifCapture &capture, std::vector<std::vector<uint8_t>> &tail)
    {
        if (g_ibitRuntime == nullptr || g_sendCurrentTransfer == nullptr)
            return;
        PS2Memory &mem = g_ibitRuntime->memory();
        uint8_t *rdram = mem.getRDRAM();
        R5900Context *live = g_ibitRuntime->eeScheduler().currentContext();
        if (rdram == nullptr || live == nullptr)
            return;
        auto rd = [&](uint32_t a) { return *reinterpret_cast<uint32_t *>(getMemPtr(rdram, a)); };
        const uint32_t mgr = rd(kTextureMapManagerPtr);
        const uint32_t engine = rd(kRenderEnginePtr);
        if (mgr == 0u || engine == 0u)
            return;
        auto sendOne = [&]
        {
            R5900Context tmp = *live;
            SET_GPR_U32(&tmp, 29, (GPR_U32(live, 29) - 0x400u) & ~0xFu);
            SET_GPR_U32(&tmp, 4, mgr);
            SET_GPR_U32(&tmp, 5, 0u);
            SET_GPR_U32(&tmp, 31, kTextureSendDonePc);
            tmp.pc = kSendCurrentTransfer;
            HoldEeTimeslices hold;
            g_sendCurrentTransfer(rdram, &tmp, g_ibitRuntime);
        };
        const uint32_t countAddr = rd(engine + 0x54u) == 1u ? 0x0048F394u : 0x00494398u;
        int32_t idx = static_cast<int32_t>(rd(kTransferListIndex));
        const int32_t total = static_cast<int32_t>(rd(countAddr));
        mem.beginGifCapture(&capture);
        while (idx < total && capture.size() < 0x400u)
        {
            capture.emplace_back();
            sendOne();
            const int32_t next = static_cast<int32_t>(rd(kTransferListIndex));
            if (next <= idx)
                break;
            idx = next;
        }
        mem.endGifCapture();

        static std::vector<std::vector<uint8_t>> s_tail;
        static uint32_t s_tailKey = 0u, s_tailTadr = 0u;
        const uint32_t key = rd(mgr + 0x88u);
        const uint32_t tadr = key != 0u ? rd(key) : 0u;
        if (key != 0u && (key != s_tailKey || tadr != s_tailTadr) && static_cast<int32_t>(rd(kTransferListIndex)) >= total)
        {
            PS2Memory::GifCapture tailCapture;
            mem.beginGifCapture(&tailCapture);
            sendOne(); // index == count: SendCurrentTransfer's fixed-chain branch
            mem.endGifCapture();
            s_tail.clear();
            for (auto &send : tailCapture)
                for (auto &packet : send)
                    s_tail.push_back(std::move(packet));
            s_tailKey = key;
            s_tailTadr = tadr;
        }
        tail = s_tail;
    }

    // GS worker: the PSXWater downloads run on the worker. When a chain with a MARK 1 VIFcode is
    // queued, record each pending download (index *(s16*)(mgr+0x148) up to the count
    // *(s16*)(mgr+0x14A); entries at *(mgr+0x144), 0x20 bytes) and mark them consumed. The game
    // reuses its DMA buffers once the transfer looks done, so each record also carries the entry's
    // continuation chain (as CHCR 0x1C5 would send it: tag VIFcodes, then payload) captured now,
    // except REF payloads inside a download destination: those are read on the worker after the
    // download wrote them. Record: entry (0x20) + ops: [0, u32 len, bytes] literal, [1, u32 addr,
    // u32 len] read later, [0xFF] end.
    bool captureDownloads(std::vector<uint8_t> &records)
    {
        if (g_ibitRuntime == nullptr)
            return false;
        uint8_t *rdram = g_ibitRuntime->memory().getRDRAM();
        auto rd = [&](uint32_t a) { return *reinterpret_cast<uint32_t *>(getMemPtr(rdram, a)); };
        const uint32_t mgr = rd(kTextureMapManagerPtr);
        if (mgr == 0u)
            return false;
        int16_t &idx = *reinterpret_cast<int16_t *>(getMemPtr(rdram, mgr + 0x148u));
        const int16_t count = *reinterpret_cast<int16_t *>(getMemPtr(rdram, mgr + 0x14Au));
        const uint32_t list = rd(mgr + 0x144u);
        if (idx < 0 || idx >= count || list == 0u)
            return false;
        auto entryDst = [&](uint32_t e, uint32_t &dst, uint32_t &bytes)
        {
            const uint32_t psm = *getMemPtr(rdram, e + 0xAu);
            const uint32_t bpp = (psm == 0x3Au || psm == 0x32u || psm == 0x0Au || psm == 0x02u) ? 2u : 4u;
            dst = rd(e + 0x10u) & 0x01FFFFFFu;
            bytes = bpp * (rd(e + 0x6u) & 0xFFFFu) * (rd(e + 0x8u) & 0xFFFFu);
        };
        std::vector<std::pair<uint32_t, uint32_t>> dstRanges;
        for (int16_t i = idx; i < count; ++i)
        {
            uint32_t dst = 0u, bytes = 0u;
            entryDst(list + static_cast<uint32_t>(i) * 0x20u, dst, bytes);
            dstRanges.emplace_back(dst, dst + bytes);
        }
        auto put32 = [](std::vector<uint8_t> &v, uint32_t x) { v.insert(v.end(), reinterpret_cast<uint8_t *>(&x), reinterpret_cast<uint8_t *>(&x) + 4); };
        for (int16_t i = idx; i < count; ++i)
        {
            const uint32_t e = list + static_cast<uint32_t>(i) * 0x20u;
            std::vector<uint8_t> rec(getMemPtr(rdram, e), getMemPtr(rdram, e) + 0x20u);
            const uint32_t cont = rd(e + 0x1Cu);
            if (cont != 0u && *getMemPtr(rdram, cont + 3u) != static_cast<uint8_t>('p'))
            {
                uint32_t tadr = cont & 0x0FFFFFF0u;
                uint32_t asr[2] = {0u, 0u};
                uint32_t asp = 0u;
                for (int tags = 0; tags < 1024; ++tags)
                {
                    if ((tadr & 0x80000000u) != 0u || (tadr & 0x01FFFFFFu) + 16u > PS2_RAM_SIZE)
                        break; // scratchpad chains are not expected here
                    const uint8_t *tp = getMemPtr(rdram, tadr);
                    uint64_t tag;
                    std::memcpy(&tag, tp, sizeof(tag));
                    const uint32_t qwc = static_cast<uint32_t>(tag & 0xFFFFu);
                    const uint32_t id = static_cast<uint32_t>((tag >> 28) & 7u);
                    const bool irq = ((tag >> 31) & 1u) != 0u;
                    const uint32_t addr = static_cast<uint32_t>((tag >> 32) & 0x7FFFFFF0u);
                    uint32_t data = 0u;
                    bool ref = false, end = false;
                    switch (id)
                    {
                    case 0: data = addr; ref = true; end = true; break;                     // refe
                    case 1: data = tadr + 16u; tadr = data + qwc * 16u; break;             // cnt
                    case 2: data = tadr + 16u; tadr = addr; break;                         // next
                    case 3: case 4: data = addr; ref = true; tadr += 16u; break;           // ref, refs
                    case 5:                                                                 // call
                        data = tadr + 16u;
                        if (asp < 2u)
                            asr[asp++] = data + qwc * 16u;
                        tadr = addr;
                        break;
                    case 6:                                                                 // ret
                        data = tadr + 16u;
                        if (asp > 0u)
                            tadr = asr[--asp];
                        else
                            end = true;
                        break;
                    default: data = tadr + 16u; end = true; break;                         // end
                    }
                    rec.push_back(0u);
                    put32(rec, 8u);
                    rec.insert(rec.end(), tp + 8, tp + 16);
                    const uint32_t len = qwc * 16u;
                    const uint32_t d = data & 0x01FFFFFFu;
                    if (len != 0u && d + len <= PS2_RAM_SIZE)
                    {
                        bool deferred = false;
                        if (ref)
                            for (const auto &range : dstRanges)
                                deferred |= d < range.second && range.first < d + len;
                        if (deferred)
                        {
                            rec.push_back(1u);
                            put32(rec, d);
                            put32(rec, len);
                        }
                        else
                        {
                            rec.push_back(0u);
                            put32(rec, len);
                            rec.insert(rec.end(), getMemPtr(rdram, d), getMemPtr(rdram, d) + len);
                        }
                    }
                    if (end || irq)
                        break;
                }
            }
            rec.push_back(0xFFu);
            put32(records, static_cast<uint32_t>(rec.size()));
            records.insert(records.end(), rec.begin(), rec.end());
        }
        idx = count;
        return true;
    }

    bool workerDownload(const uint8_t *record, uint32_t size, std::vector<uint8_t> &continuation)
    {
        static std::atomic<uint32_t> s_logs{0};
        if (size < 0x21u)
            return false;
        PS2Runtime *runtime = g_ibitRuntime;
        uint8_t *rdram = runtime->memory().getRDRAM();
        const uint8_t *entry = record;
        auto e16 = [&](uint32_t o) { uint16_t v; std::memcpy(&v, entry + o, 2); return static_cast<uint32_t>(v); };
        auto e32 = [&](uint32_t o) { uint32_t v; std::memcpy(&v, entry + o, 4); return v; };
        const uint32_t width = e16(0x0u), ssax = e16(0x2u), ssay = e16(0x4u), rrw = e16(0x6u), rrh = e16(0x8u);
        const uint32_t psm = entry[0xAu];
        const uint32_t sbp = e16(0xCu);
        const uint32_t dst = e32(0x10u);
        const uint32_t callback = e32(0x14u);
        const uint32_t bpp = (psm == 0x3Au || psm == 0x32u || psm == 0x0Au || psm == 0x02u) ? 2u : 4u;
        const uint32_t bytes = bpp * rrw * rrh;
        uint64_t pkt[10] = {};
        pkt[0] = 4ull | (1ull << 15) | (1ull << 60); // NLOOP=4, EOP, PACKED, NREG=1
        pkt[1] = 0xEull;                             // A+D
        pkt[2] = (static_cast<uint64_t>(psm) << 24) | sbp | (static_cast<uint64_t>(width >> 6) << 16);
        pkt[3] = 0x50u; // BITBLTBUF
        pkt[4] = ssax | (static_cast<uint64_t>(ssay) << 16);
        pkt[5] = 0x51u; // TRXPOS
        pkt[6] = rrw | (static_cast<uint64_t>(rrh) << 32);
        pkt[7] = 0x52u; // TRXREG
        pkt[8] = 1u;    // local -> host
        pkt[9] = 0x53u; // TRXDIR
        runtime->memory().submitGifPacket(GifPathId::Path2, reinterpret_cast<const uint8_t *>(pkt), sizeof(pkt), true, false);
        uint32_t copied = 0u;
        if (dst != 0u && (dst & 0x01FFFFFFu) + bytes <= PS2_RAM_SIZE)
            copied = runtime->gs().consumeLocalToHostBytes(getMemPtr(rdram, dst), bytes);
        if (callback != 0u)
        {
            static std::atomic<uint32_t> s_cbLogs{0};
            if (s_cbLogs.fetch_add(1u) < 4u)
                std::fprintf(stderr, "[ATV2:DOWNLOAD] callback 0x%x not run on the GS worker\n", callback);
        }
        // Continuation stream from the captured plan.
        size_t pos = 0x20u;
        auto get32 = [&](uint32_t &v) { if (pos + 4u > size) return false; std::memcpy(&v, record + pos, 4); pos += 4u; return true; };
        while (pos < size)
        {
            const uint8_t op = record[pos++];
            if (op == 0xFFu)
                break;
            uint32_t a = 0u, len = 0u;
            if (op == 0u)
            {
                if (!get32(len) || pos + len > size)
                    break;
                continuation.insert(continuation.end(), record + pos, record + pos + len);
                pos += len;
            }
            else if (op == 1u)
            {
                if (!get32(a) || !get32(len) || a + len > PS2_RAM_SIZE)
                    break;
                continuation.insert(continuation.end(), getMemPtr(rdram, a), getMemPtr(rdram, a) + len);
            }
            else
                break;
        }
        if (s_logs.fetch_add(1u) < 8u)
            std::fprintf(stderr, "[ATV2:DOWNLOAD] worker sbp=0x%x rr=%ux%u dst=0x%x bytes=%u copied=%u cb=0x%x continuation=%zu bytes\n",
                         sbp, rrw, rrh, dst, bytes, copied, callback, continuation.size());
        return true;
    }

    void registerVif1TextureIbit(PS2Runtime &runtime)
    {
        const char *v = std::getenv("ATV2_IBIT_TEX_SYNC");
        if (v != nullptr && v[0] == '0')
            return;
        g_sendCurrentTransfer = runtime.lookupFunction(kSendCurrentTransfer);
        if (!g_sendCurrentTransfer)
        {
            std::cerr << "[ATV2:IBIT] no generated function at 0x178DF0; texture I-bit sync not installed" << std::endl;
            return;
        }
        g_ibitRuntime = &runtime;
        g_ps2Vif1IbitHook = &onVif1TextureIbit;
        g_ps2Vif1PrecaptureHook = &precaptureIbitTextures;
        g_ps2Vif1DownloadCaptureHook = &captureDownloads;
        g_ps2Vif1WorkerDownloadHook = &workerDownload;
        g_ps2Vif1ReadbackHook = [](uint8_t *dst, uint32_t maxBytes) -> uint32_t
        { return g_ibitRuntime ? g_ibitRuntime->gs().consumeLocalToHostBytes(dst, maxBytes) : 0u; };
    }

    // PSXTextureMapManager IPU decode @ 0x179190: (mgr, dst, src, size, width, height, ?, wait).
    // It feeds src to IDEC (RGB16 output) over IPU_TO and DMAs the result to dst over IPU_FROM;
    // the caller then copies dst's macroblocks into the texture cache (0x176260). The runtime's
    // IPU registers are inert, so dst was never written and race textures sampled stale memory.
    void ipuDecompress179190(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        static uint32_t s_logs = 0u;
        static uint32_t s_decoded = 0u;
        static uint32_t s_failed = 0u;
        const uint32_t dst = getRegU32(ctx, 5) & 0x1FFFFFFFu;
        const uint32_t src = getRegU32(ctx, 6) & 0x1FFFFFFFu;
        const uint32_t size = getRegU32(ctx, 7);
        const uint32_t width = getRegU32(ctx, 8);
        const uint32_t height = getRegU32(ctx, 9);
        const uint64_t outBytes = static_cast<uint64_t>(width) * height * 2u;

        bool ok = false;
        if (width != 0u && height != 0u && width <= 1024u && height <= 1024u &&
            static_cast<uint64_t>(src) + size <= PS2_RAM_SIZE && static_cast<uint64_t>(dst) + outBytes <= PS2_RAM_SIZE)
        {
            ok = ps2IpuDecodeTextureRgb16(getMemPtr(rdram, src), size, width, height, getMemPtr(rdram, dst));
        }
        if (ok)
            ++s_decoded;
        else
            ++s_failed;

        if (s_logs < 12u || (!ok && s_failed <= 32u))
        {
            ++s_logs;
            std::cerr << "[ATV2:IPU] decode dst=0x" << std::hex << dst << " src=0x" << src << " size=0x" << size
                      << std::dec << " " << width << "x" << height << " ok=" << (ok ? 1 : 0)
                      << " decoded=" << s_decoded << " failed=" << s_failed << std::endl;
        }
        ctx->pc = getRegU32(ctx, 31);
    }

    void registerIpuTextureDecode(PS2Runtime &runtime)
    {
        const char *v = std::getenv("ATV2_IPU_HLE");
        if (v != nullptr && v[0] == '0')
            return;
        runtime.replaceFunction(0x00179190u, &ipuDecompress179190);
    }

    // ATV2 links libpad "PsIIlibpad 2500" without symbols, and nothing on the IOP side services
    // padman's SIF RPC (0x80000100/0x80000101) or fills its DMA buffer. Route the library's entry
    // points to the runtime's HLE pad implementation, which reads the host keyboard/gamepad.
    struct PadHleEntry
    {
        uint32_t addr;
        const char *name;
        PS2Runtime::RecompiledFunction hle;
        int32_t forcedSuccessReturn;
    };

    const PadHleEntry kPadHleEntries[] = {
        {0x00377320u, "scePadInit", &ps2_stubs::scePadInit, 0},
        {0x00377580u, "scePadPortOpen", &ps2_stubs::scePadPortOpen, 0},
        {0x003778D0u, "scePadRead", &ps2_stubs::scePadRead, 32},
        {0x00377950u, "scePadGetState", &ps2_stubs::scePadGetState, 0},
        {0x00377A00u, "scePadSetReqState", &ps2_stubs::scePadSetReqState, 0},
        {0x00377A68u, "scePadGetReqState", &ps2_stubs::scePadGetReqState, 0},
        {0x00377AF0u, "scePadInfoAct", &ps2_stubs::scePadInfoAct, 0},
        {0x00377D30u, "scePadInfoMode", &ps2_stubs::scePadInfoMode, 0},
        {0x00377E68u, "scePadSetMainMode", &ps2_stubs::scePadSetMainMode, 0},
        {0x00377F20u, "scePadSetActDirect", &ps2_stubs::scePadSetActDirect, 0},
        {0x00377FE0u, "scePadSetActAlign", &ps2_stubs::scePadSetActAlign, 0},
        {0x003780B8u, "scePadGetButtonMask", &ps2_stubs::scePadGetButtonMask, 0},
        {0x00378170u, "scePadSetButtonInfo", &ps2_stubs::scePadSetButtonInfo, 0},
        {0x00378220u, "scePadInfoPressMode", &ps2_stubs::scePadInfoPressMode, 0},
        {0x00378280u, "scePadEnterPressMode", &ps2_stubs::scePadEnterPressMode, 0},
        {0x003784D0u, "scePadGetModVersion", &ps2_stubs::scePadGetModVersion, 0},
    };

    template <size_t I>
    void padHleWrap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static int s_logged = 0;
        const PadHleEntry &e = kPadHleEntries[I];
        const uint32_t ra = getRegU32(ctx, 31);
        const uint32_t a0 = getRegU32(ctx, 4), a1 = getRegU32(ctx, 5), a2 = getRegU32(ctx, 6);
        if constexpr (I == 2)
        {
            // ATV2_AUTO_CROSS=<n>: test aid, holds Cross for 8 port-0 reads starting at read n.
            static const int s_autoCross = []() { const char *s = std::getenv("ATV2_AUTO_CROSS"); return s ? std::atoi(s) : 0; }();
            // ATV2_AUTO_CROSS_ONCE=1: repeat presses only while the front end is in state 12
            // (memory card alert), so movies and menus are left alone.
            static const bool s_autoCrossOnce = std::getenv("ATV2_AUTO_CROSS_ONCE") != nullptr;
            // ATV2_AUTO_CROSS_UNTIL_RACE=1: stop repeating once the race is running (front end state 2
            // with no parent screen), so the presses don't pause the race.
            static const bool s_untilRace = std::getenv("ATV2_AUTO_CROSS_UNTIL_RACE") != nullptr;
            static int s_reads = 0;
            // ATV2_PAD_SCRIPT=<read>:<button>,...: test aid, presses each button for 6 port-0 reads
            // starting at that read. Buttons: X O T S U D L R ST SEL L1 R1 L2 R2.
            static const std::vector<std::pair<int, uint16_t>> s_script = []() {
                std::vector<std::pair<int, uint16_t>> out;
                const char *s = std::getenv("ATV2_PAD_SCRIPT");
                if (s == nullptr)
                    return out;
                static const std::pair<const char *, uint16_t> kButtons[] = {
                    {"X", 0x4000u}, {"O", 0x2000u}, {"T", 0x1000u}, {"S", 0x8000u}, {"U", 0x0010u},
                    {"D", 0x0040u}, {"L", 0x0080u}, {"R", 0x0020u}, {"ST", 0x0008u}, {"SEL", 0x0001u},
                    {"L1", 0x0400u}, {"R1", 0x0800u}, {"L2", 0x0100u}, {"R2", 0x0200u}};
                std::string all(s);
                size_t pos = 0;
                while (pos < all.size())
                {
                    size_t end = all.find(',', pos);
                    if (end == std::string::npos)
                        end = all.size();
                    const std::string tok = all.substr(pos, end - pos);
                    const size_t colon = tok.find(':');
                    if (colon != std::string::npos)
                        for (const auto &b : kButtons)
                            if (tok.compare(colon + 1, std::string::npos, b.first) == 0)
                                out.emplace_back(std::atoi(tok.c_str()), b.second);
                    pos = end + 1;
                }
                return out;
            }();
            static int s_scriptReads = 0;
            if (!s_script.empty() && a0 == 0u)
            {
                ++s_scriptReads;
                uint16_t held = 0;
                for (const auto &step : s_script)
                    if (s_scriptReads >= step.first && s_scriptReads < step.first + 6)
                        held |= step.second;
                for (const auto &step : s_script)
                    if (s_scriptReads == step.first)
                        std::cerr << "[ATV2:PAD] script press 0x" << std::hex << step.second << std::dec << " at read "
                                  << s_scriptReads << std::endl;
                if (held != 0)
                    ps2_stubs::setPadOverrideState(0xFFFFu & ~held, 0x80u, 0x80u, 0x80u, 0x80u);
                else
                    ps2_stubs::clearPadOverrideState();
            }
            else if (s_untilRace && READ32(0x005425E0u + 0xCCu) == 2u && READ32(0x005425E0u + 0x1Cu) == 0u)
            {
                // Race running: release any held press once, then leave the pad alone.
                // ATV2_RACE_THROTTLE=1 instead holds Cross (throttle) for the whole race.
                static const bool s_throttle = std::getenv("ATV2_RACE_THROTTLE") != nullptr;
                static bool s_released = false;
                if (s_throttle)
                    ps2_stubs::setPadOverrideState(0xFFFFu & ~0x4000u, 0x80u, 0x80u, 0x80u, 0x80u);
                else if (!s_released)
                {
                    s_released = true;
                    ps2_stubs::clearPadOverrideState();
                }
            }
            else if (s_autoCross > 0 && a0 == 0u)
            {
                ++s_reads;
                if ((s_reads % 25) == 0)
                    std::cerr << "[ATV2:PAD] port0 reads=" << s_reads << " ra=0x" << std::hex << ra << std::dec << std::endl;
                if (s_reads == s_autoCross)
                {
                    std::cerr << "[ATV2:PAD] auto Cross press at read " << s_reads << std::endl;
                    for (int reg = 16; reg <= 23; ++reg)
                    {
                        const uint32_t obj = getRegU32(ctx, reg);
                        if (obj < 0x00100000u || obj >= 0x01FF0000u || READ32(obj + 4u) != 16u)
                            continue;
                        const uint32_t table = READ32(obj + 0x14u);
                        std::cerr << "[ATV2:PAD] map obj=0x" << std::hex << obj << " table=0x" << table << std::dec;
                        for (uint32_t i = 0; i < 16u; ++i)
                        {
                            const uint32_t e = table + i * 12u;
                            std::cerr << " [" << i << ":f" << static_cast<int>(READ8(e)) << ",p" << READ32(e + 4u)
                                      << ",m0x" << std::hex << READ16(e + 8u) << std::dec << "]";
                        }
                        std::cerr << std::endl;
                        break;
                    }
                    ps2_stubs::setPadOverrideState(0xFFFFu & ~0x4000u, 0x80u, 0x80u, 0x80u, 0x80u);
                }
                else if (s_reads > s_autoCross && ((s_reads - s_autoCross) % 30) == 0 && (!s_autoCrossOnce || READ32(0x005425E0u + 0xCCu) == 12u))
                {
                    const bool start = (((s_reads - s_autoCross) / 30) % 2) == 1;
                    std::cerr << "[ATV2:PAD] auto " << (start ? "Start" : "Cross") << " repeat at read " << s_reads << std::endl;
                    ps2_stubs::setPadOverrideState(0xFFFFu & ~(start ? 0x0008u : 0x4000u), 0x80u, 0x80u, 0x80u, 0x80u);
                }
                else if (s_reads > s_autoCross && ((s_reads - s_autoCross) % 30) == 4)
                {
                    ps2_stubs::clearPadOverrideState();
                }
            }
        }
        e.hle(rdram, ctx, runtime);
        if (e.forcedSuccessReturn != 0 && getRegU32(ctx, 2) != 0u)
            SET_GPR_S32(ctx, 2, e.forcedSuccessReturn);
        if (s_logged < 4)
        {
            ++s_logged;
            std::cerr << "[ATV2:PAD] " << e.name << "(0x" << std::hex << a0 << ",0x" << a1 << ",0x" << a2
                      << ") -> 0x" << getRegU32(ctx, 2) << " ra=0x" << ra << std::dec << std::endl;
        }
        ctx->pc = ra;
    }

    template <size_t... I>
    void registerPadHleImpl(PS2Runtime &runtime, std::index_sequence<I...>)
    {
        (runtime.replaceFunction(kPadHleEntries[I].addr, &padHleWrap<I>), ...);
    }

    PS2Runtime::RecompiledFunction g_padEvent122930Orig = nullptr;

    void padEvent122930(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static int s_logged = 0;
        if (s_logged < 40)
        {
            ++s_logged;
            std::cerr << "[ATV2:PAD] event(0x122930) a1(button)=" << getRegU32(ctx, 5)
                      << " a2=" << getRegU32(ctx, 6) << " a3(down)=" << getRegU32(ctx, 7)
                      << " ra=0x" << std::hex << getRegU32(ctx, 31) << std::dec << std::endl;
        }
        g_padEvent122930Orig(rdram, ctx, runtime);
    }

    PS2Runtime::RecompiledFunction g_countdownOrig = nullptr;

    void countdownWatch248020(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        atvStartHostProfiler(); // ATV_HOST_PROFILE=1: sample the game thread from the race countdown on
        static int s_calls = 0;
        const uint32_t self = getRegU32(ctx, 4);
        if (s_calls < 12 || (s_calls % 120) == 0)
            ++s_calls;
        else
        {
            ++s_calls;
            g_countdownOrig(rdram, ctx, runtime);
            return;
        }
            std::cerr << "[ATV2:COUNTDOWN] call=" << (s_calls - 1) << " dt=" << ctx->f[12]
                      << std::hex << " self=0x" << self << " flags=0x" << READ32(self + 0x5Cu)
                      << " mode=0x" << READ32(self + 0x64u) << std::dec
                      << " len=" << std::bit_cast<float>(READ32(self + 0xB4u))
                      << " step=" << std::bit_cast<float>(READ32(self + 0xB8u))
                      << " elapsed=" << std::bit_cast<float>(READ32(self + 0xBCu))
                      << std::hex << " ra=0x" << getRegU32(ctx, 31) << std::dec << std::endl;
        g_countdownOrig(rdram, ctx, runtime);
    }

    template <uint32_t Addr>
    struct CallCounter
    {
        static inline PS2Runtime::RecompiledFunction orig = nullptr;
        static void hook(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
        {
            static uint32_t s_calls = 0;
            if (++s_calls <= 4u || (s_calls & (s_calls - 1u)) == 0u)
                std::cerr << "[ATV2:CALLCOUNT] fn=0x" << std::hex << Addr << " n=" << std::dec << s_calls
                          << std::hex << " a0=0x" << getRegU32(ctx, 4) << " ra=0x" << getRegU32(ctx, 31)
                          << std::dec << " f12=" << ctx->f[12] << std::endl;
            orig(rdram, ctx, runtime);
        }
        static void install(PS2Runtime &runtime)
        {
            orig = runtime.lookupFunction(Addr);
            if (orig)
                runtime.replaceFunction(Addr, &hook);
        }
    };

    PS2Runtime::RecompiledFunction g_pageSwitchOrig = nullptr;

    void pageSwitchWatch2d39d0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static int s_calls = 0;
        const uint32_t slot = getRegU32(ctx, 29) + 0x50u;
        uint32_t saved = 0;
        std::memcpy(&saved, getMemPtr(rdram, slot), 4);
        std::cerr << std::hex << "[ATV2:PAGESW] #" << std::dec << ++s_calls << std::hex << " a1=0x" << getRegU32(ctx, 5)
                  << " slot=0x" << slot << " savedRa=0x" << saved << std::dec << std::endl;
        if (getRegU32(ctx, 31) == 0x002D38C4u)
            atv2ArmWriteWatch(rdram, slot, 8u);
        g_pageSwitchOrig(rdram, ctx, runtime);
    }

    void registerPadHle(PS2Runtime &runtime)
    {
        const char *v = std::getenv("ATV2_PAD_HLE");
        if (v != nullptr && v[0] == '0')
            return;
        registerPadHleImpl(runtime, std::make_index_sequence<std::size(kPadHleEntries)>{});
        if (std::getenv("ATV2_AUTO_CROSS") != nullptr)
        {
            g_padEvent122930Orig = runtime.lookupFunction(0x00122930u);
            if (g_padEvent122930Orig)
                runtime.replaceFunction(0x00122930u, &padEvent122930);
            g_countdownOrig = runtime.lookupFunction(0x00248020u);
            if (g_countdownOrig)
                runtime.replaceFunction(0x00248020u, &countdownWatch248020);
            CallCounter<0x00157260u>::install(runtime);
            CallCounter<0x00211490u>::install(runtime);
        }
    }

    // The game's libm cos/sin/floor/fabs (0x3684D0 / 0x368D18 / 0x368650 / 0x3685F8) are the
    // double-precision versions. The EE has no double FPU, so GCC passes the double as 64-bit bits
    // in a0 and returns it in v0. The recompiler bound them to the generic float stubs (f12 -> f0),
    // which left v0 untouched: cos(x) == sin(x) == x. That broke the sin/cos table built at boot
    // (0x1AACC0), so any camera roll (0x325B90 -> 0x1AB4D0) scaled the view to nearly nothing.
    double softDoubleArg(R5900Context *ctx)
    {
        return std::bit_cast<double>(static_cast<uint64_t>(GPR_U64(ctx, 4)));
    }

    void softDoubleReturn(R5900Context *ctx, double value)
    {
        SET_GPR_U64(ctx, 2, std::bit_cast<uint64_t>(value));
        ctx->pc = getRegU32(ctx, 31);
    }

    void libmCos3684d0(uint8_t *, R5900Context *ctx, PS2Runtime *) { softDoubleReturn(ctx, std::cos(softDoubleArg(ctx))); }
    void libmSin368d18(uint8_t *, R5900Context *ctx, PS2Runtime *) { softDoubleReturn(ctx, std::sin(softDoubleArg(ctx))); }
    void libmFloor368650(uint8_t *, R5900Context *ctx, PS2Runtime *) { softDoubleReturn(ctx, std::floor(softDoubleArg(ctx))); }
    void libmFabs3685f8(uint8_t *, R5900Context *ctx, PS2Runtime *) { softDoubleReturn(ctx, std::fabs(softDoubleArg(ctx))); }

    void registerFadeStepFix(PS2Runtime &runtime)
    {
        runtime.replaceFunction(0x003684D0u, &libmCos3684d0);
        runtime.replaceFunction(0x00368D18u, &libmSin368d18);
        runtime.replaceFunction(0x00368650u, &libmFloor368650);
        runtime.replaceFunction(0x003685F8u, &libmFabs3685f8);
        g_fadeStep160d10Orig = runtime.lookupFunction(0x00160D10u);
        if (!g_fadeStep160d10Orig)
        {
            std::cerr << "[ATV2:FADE] no generated function at 0x160D10; fade fix not installed" << std::endl;
            return;
        }
        runtime.replaceFunction(0x00160D10u, &fadeStep160d10);
    }

    // ---- Opt-in heap trace (ATV2_HEAPTRACE=1) ---------------------------------------
    // Logs malloc (0x36BD98) calls that are large or touch the top of the heap,
    // where the 26MB pool (0x103B20) and its 0x18-byte lock must not overlap.
    PS2Runtime::RecompiledFunction g_malloc36bd98Orig = nullptr;
    int g_heapTraceLines = 0;

    bool heapTraceInteresting(uint32_t size, uint32_t ptr)
    {
        return size >= 0x10000u || ptr >= 0x01F70000u || size == 0x18u;
    }

    void heapTraceDumpAllocator(uint8_t *rdram)
    {
        auto rd = [&](uint32_t a) { return *reinterpret_cast<uint32_t *>(getMemPtr(rdram, a)); };
        std::cerr << "[ATV2:HEAP] top=0x" << std::hex << rd(0x004124D0u)
                  << " break=0x" << rd(0x00410AD4u)
                  << " base=0x" << rd(0x004128E0u)
                  << " ext=0x" << rd(0x004128E8u) << "/0x" << rd(0x004128F0u) << "/0x" << rd(0x004128F8u)
                  << std::dec << std::endl;
    }

    void malloc36bd98Trace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t size = GPR_U32(ctx, 4);
        const uint32_t ra = GPR_U32(ctx, 31);
        g_malloc36bd98Orig(rdram, ctx, runtime);
        if (ctx->pc != ra || g_heapTraceLines >= 400)
            return;
        const uint32_t ptr = GPR_U32(ctx, 2);
        if (!heapTraceInteresting(size, ptr))
            return;
        ++g_heapTraceLines;
        std::cerr << "[ATV2:HEAP] malloc(0x" << std::hex << size << ") = 0x" << ptr
                  << " ra=0x" << ra << std::dec << std::endl;
        if (size >= 0x100000u || ptr >= 0x01F70000u)
            heapTraceDumpAllocator(rdram);
    }

    void registerHeapTrace(PS2Runtime &runtime)
    {
        const char *v = std::getenv("ATV2_HEAPTRACE");
        if (v == nullptr || v[0] == '\0' || v[0] == '0')
            return;
        g_malloc36bd98Orig = runtime.lookupFunction(0x0036BD98u);
        if (!g_malloc36bd98Orig)
        {
            std::cerr << "[ATV2:HEAP] no generated function at 0x36BD98; heap trace not installed" << std::endl;
            return;
        }
        runtime.replaceFunction(0x0036BD98u, &malloc36bd98Trace);
    }

    void registerFeTrace(PS2Runtime &runtime)
    {
        const char *v = std::getenv("ATV2_TRACE_FE");
        if (v == nullptr || v[0] == '\0' || v[0] == '0')
            return;
        registerFeTraceImpl(runtime, std::make_integer_sequence<int, 27>{});
    }

    // ---- Suzuki DZ400 in the menus and races -------------------------------------------------
    // Kernel/Atv2SuzukiDz400.cpp rebuilds the roster archives (QUADINFO.RES, PAGE07.RES, PAGE20.RES)
    // with a 26th quad, IngameID 6, and makes its menu model archive UI\RESOURCE\QUAD\ATV06.RES.
    // The game finds disc files by name in its table at 0x408B80 (742 records: u64 path, lbn, end lbn,
    // sectors, bytes) through 0x33F510, which returns a new 0x34-byte record filled by 0x33F3D0:
    // +0 lbn, +4 sectors, +8 leaf name (to the ';', 16 bytes at most), +0x24 lbn, +0x28 sectors,
    // +0x2C 0, +0x30 bytes. The lookup answers for the rebuilt files with sectors past the end of the
    // image, which the CD layer serves from .cdcache/atv2_dz400. ATV2_DZ400=0 keeps the retail menus;
    // a profile saved with the DZ400 selected stores roster index 25, which the retail roster lacks
    // (empty ATV & Rider Select, race load hangs until another quad is picked).
    struct DiscOverlay
    {
        std::string discPath; // as in the table, e.g. "\\UI\\RESOURCE\\QUADINFO.RES;1"
        uint32_t lbn = 0u;
        uint32_t bytes = 0u;
    };
    std::vector<DiscOverlay> g_discOverlays;
    PS2Runtime::RecompiledFunction g_findDiscFile33f510Orig = nullptr;
    constexpr uint32_t kDiscFileTable = 0x00408B80u;
    constexpr uint32_t kDiscFileCount = 742u;
    constexpr uint32_t kDiscOverlayLbn = 0x00300000u; // the image ends at sector 0x216F20

    bool sameDiscPath(const std::string &a, const std::string &b)
    {
        return a.size() == b.size() &&
               std::equal(a.begin(), a.end(), b.begin(), [](char x, char y)
                          { return std::toupper(static_cast<unsigned char>(x)) == std::toupper(static_cast<unsigned char>(y)); });
    }

    bool readRetailDiscFile(uint8_t *rdram, const std::string &discPath, std::vector<uint8_t> &data)
    {
        for (uint32_t i = 0; i < kDiscFileCount; ++i)
        {
            const uint8_t *record = getConstMemPtr(rdram, kDiscFileTable + i * 0x28u);
            uint32_t path = 0, lbn = 0, bytes = 0;
            std::memcpy(&path, record, 4u);
            std::memcpy(&lbn, record + 0x08u, 4u);
            std::memcpy(&bytes, record + 0x20u, 4u);
            if (!sameDiscPath(reinterpret_cast<const char *>(getConstMemPtr(rdram, path)), discPath))
                continue;
            std::ifstream image(PS2Runtime::getIoPaths().cdImage, std::ios::binary);
            data.resize(bytes);
            image.seekg(static_cast<std::streamoff>(lbn) * 2048);
            image.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
            return image.gcount() == static_cast<std::streamsize>(bytes);
        }
        return false;
    }

    void installDiscOverlays(uint8_t *rdram)
    {
        std::vector<std::pair<std::string, std::vector<uint8_t>>> files;
        if (!atv2BuildSuzukiDz400Files([rdram](const std::string &path, std::vector<uint8_t> &data)
                                       { return readRetailDiscFile(rdram, path, data); },
                                       files))
            return;
        const PS2Runtime::IoPaths &paths = PS2Runtime::getIoPaths();
        const std::filesystem::path dir =
            (paths.elfDirectory.empty() ? std::filesystem::path(".cdcache") : paths.elfDirectory / ".cdcache") /
            "atv2_dz400";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        std::vector<DiscOverlay> overlays;
        uint32_t lbn = kDiscOverlayLbn;
        for (const auto &[discPath, data] : files)
        {
            const size_t leaf = discPath.find_last_of('\\') + 1u;
            const std::filesystem::path host = dir / discPath.substr(leaf, discPath.find(';') - leaf);
            std::ofstream out(host, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
            out.close();
            if (!out || !ps2_stubs::registerCdOverlayFile(discPath, host, lbn))
            {
                std::cerr << "[ATV2:DZ400] could not serve " << host.string()
                          << "; the Suzuki DZ400 stays out of the menus" << std::endl;
                return;
            }
            overlays.push_back({discPath, lbn, static_cast<uint32_t>(data.size())});
            lbn += (static_cast<uint32_t>(data.size()) + 2047u) / 2048u + 16u;
        }
        g_discOverlays = std::move(overlays);
        std::cerr << "[ATV2:DZ400] Suzuki DZ400 added to the quad roster (" << g_discOverlays.size()
                  << " disc files in " << dir.string() << ")" << std::endl;
    }

    void findDiscFile33f510(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static bool s_installed = false;
        if (!s_installed)
        {
            s_installed = true;
            installDiscOverlays(rdram);
        }
        const char *name = reinterpret_cast<const char *>(getConstMemPtr(rdram, getRegU32(ctx, 4)));
        const std::string discPath = "\\" + std::string(name != nullptr ? name : "") + ";1"; // "\\%s;1"
        static const bool s_trace = std::getenv("ATV2_TRACE_FILES") != nullptr;
        if (s_trace)
            std::cerr << "[ATV2:FILE] " << discPath << " ra=0x" << std::hex << getRegU32(ctx, 31) << std::dec << std::endl;
        for (const DiscOverlay &overlay : g_discOverlays)
        {
            if (!sameDiscPath(overlay.discPath, discPath))
                continue;
            const uint32_t record = callRecompiled4(runtime, rdram, ctx, 0x00103E30u, 0x34u, 0x00447328u, 0x5Fu, 0u);
            if (record != 0u)
            {
                const uint32_t sectors = (overlay.bytes + 2047u) / 2048u;
                WRITE32(record + 0x00u, overlay.lbn);
                WRITE32(record + 0x04u, sectors);
                for (size_t i = overlay.discPath.find_last_of('\\') + 1u, n = 0; n < 16u; ++i, ++n)
                {
                    const char c = overlay.discPath[i];
                    WRITE8(record + 0x08u + static_cast<uint32_t>(n), c == ';' ? 0u : static_cast<uint8_t>(c));
                    if (c == ';')
                        break;
                }
                WRITE32(record + 0x24u, overlay.lbn);
                WRITE32(record + 0x28u, sectors);
                WRITE32(record + 0x2Cu, 0u);
                WRITE32(record + 0x30u, overlay.bytes);
            }
            SET_GPR_U32(ctx, 2, record);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }
        g_findDiscFile33f510Orig(rdram, ctx, runtime);
    }

    // Opponents come from the human's class in quad\ATVGroups.pb: 0x210D20 looks each human's quad ID
    // (settings +0x514, settings = *(*0x404818 + 0x298); a0 +0x84 = humans) up in the groups' ATVn
    // lists. No group lists the DZ400 (ID 6), so the candidate list stays empty and the AI gets a
    // garbage quad ID. The DZ400 is the KFX400 (ID 7, Sport Series) in Suzuki colours: pick opponents
    // as for the KFX400. The lookup runs to completion with ID 7 in place; were it ever cut short, the
    // human keeps ID 7 and races the KFX400.
    PS2Runtime::RecompiledFunction g_aiOpponents210d20Orig = nullptr;

    void aiOpponents210d20(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t settings = READ32(READ32(0x00404818u) + 0x298u);
        const uint32_t humans = std::min<uint32_t>(READ32(getRegU32(ctx, 4) + 0x84u), 5u);
        uint32_t asKfx400 = 0u;
        for (uint32_t i = 0; i < humans; ++i)
        {
            if (READ32(settings + 0x514u + i * 4u) == 6u)
            {
                WRITE32(settings + 0x514u + i * 4u, 7u);
                asKfx400 |= 1u << i;
            }
        }
        if (asKfx400 == 0u)
        {
            g_aiOpponents210d20Orig(rdram, ctx, runtime);
            return;
        }
        const uint32_t returnPc = getRegU32(ctx, 31);
        {
            HoldEeTimeslices hold;
            g_aiOpponents210d20Orig(rdram, ctx, runtime);
            for (uint32_t guard = 0; ctx->pc != returnPc; ++guard)
            {
                if (ctx->pc == 0u || guard >= 1000000u || runtime->isStopRequested() || !runtime->hasFunction(ctx->pc))
                {
                    std::cerr << "[ATV2:DZ400] AI opponent pick stopped at pc=0x" << std::hex << ctx->pc << std::dec
                              << "; racing the KFX400" << std::endl;
                    return;
                }
                runtime->lookupFunction(ctx->pc)(rdram, ctx, runtime);
            }
        }
        for (uint32_t i = 0; i < humans; ++i)
        {
            if (asKfx400 & (1u << i))
                WRITE32(settings + 0x514u + i * 4u, 6u);
        }
    }

    void registerSuzukiDz400(PS2Runtime &runtime)
    {
        const char *env = std::getenv("ATV2_DZ400");
        if (env != nullptr && env[0] == '0')
            return;
        g_findDiscFile33f510Orig = runtime.lookupFunction(0x0033F510u);
        g_aiOpponents210d20Orig = runtime.lookupFunction(0x00210D20u);
        if (!g_findDiscFile33f510Orig || !g_aiOpponents210d20Orig)
        {
            std::cerr << "[ATV2:DZ400] no generated function at 0x33F510 or 0x210D20; Suzuki DZ400 not added"
                      << std::endl;
            return;
        }
        runtime.replaceFunction(0x0033F510u, &findDiscFile33f510);
        runtime.replaceFunction(0x00210D20u, &aiOpponents210d20);
    }

    void applyScus97211Overrides(PS2Runtime &runtime)
    {
        runtime.replaceFunction(0x0014F610u, &resourceInit14f610);
        runtime.replaceFunction(0x001BD4F0u, &resourceHook1bd4f0);
        runtime.replaceFunction(0x00342590u, &vertexUpdate342590);
        runtime.replaceFunction(0x00107460u, &addRef107460);
        runtime.replaceFunction(0x001AE800u, &nameRegistry1ae800);
        runtime.replaceFunction(0x001D2FA0u, &registryFlags1d2fa0);
        runtime.replaceFunction(0x00177AE0u, &resourceForward177ae0);
        runtime.replaceFunction(0x00177AF0u, &resourceRemove177af0);
        runtime.replaceFunction(0x001D0890u, &forward1d0890);
        runtime.replaceFunction(0x001D08A0u, &forward1d08a0);
        runtime.replaceFunction(0x001749B0u, &listInsertTail1749b0);
        runtime.replaceFunction(0x001CE020u, &resourceRead1ce020);
        // PCSX2 confirms the adjacent resource type uses identical instructions.
        runtime.replaceFunction(0x001CE0C0u, &resourceRead1ce020);
        runtime.replaceFunction(0x002AE950u, &sliderInit2ae950);
        runtime.replaceFunction(0x002364C0u, &textInputInit2364c0);
        runtime.replaceFunction(0x001C7AC0u, &ctorPrologue1c7a40<0x001C7AC0u>);
        runtime.replaceFunction(0x001C7C00u, &ctorPrologue1c7a40<0x001C7C00u>);
        runtime.replaceFunction(0x00126FC0u, &leafStoreF12_126fc0);
        runtime.replaceFunction(0x002AE600u, &iconInit2ae600);
        // Method-table entry discovery in the recompiler now registers these; the overrides only
        // cover runners generated without it.
        replaceIfMissing(runtime, 0x002AE370u, &uiInit2ae370);
        replaceIfMissing(runtime, 0x002AE290u, &uiUpdate2ae290);
        replaceIfMissing(runtime, 0x002AE220u, &uiRelease2ae220);
        replaceIfMissing(runtime, 0x002AE280u, &uiTail2ae280);
        replaceIfMissing(runtime, 0x002AE3B0u, &uiTimer2ae3b0);
        replaceIfMissing(runtime, 0x002AE400u, &uiSelect2ae400);
        replaceIfMissing(runtime, 0x00155AB0u, &renderBegin155ab0);
        replaceIfMissing(runtime, 0x002D5880u, &childFactory2d5880);
        replaceIfMissing(runtime, 0x002343B0u, &attachChild2343b0);
        replaceIfMissing(runtime, 0x002343E0u, &leafReturn2343e0);
        replaceIfMissing(runtime, 0x002343F0u, &leafReturn2343e0);
        replaceIfMissing(runtime, 0x00234400u, &leafReturnOne234400);
        replaceIfMissing(runtime, 0x00234410u, &tailNodeUpdate234410);
        replaceIfMissing(runtime, 0x002B5600u, &getField44_2b5600);
        runtime.replaceFunction(0x002AD6C0u, &highlightInit2ad6c0);
        runtime.replaceFunction(0x00204390u, &screenConfig204390);
        runtime.replaceFunction(0x002D1F10u, &ctor2d1f10);
        runtime.replaceFunction(0x001ABC50u, &chunkListAlloc1abc50);
        runtime.replaceFunction(0x001ABC40u, &forward1abc40);
        runtime.replaceFunction(0x001ABC20u, &forward1abc20);
        runtime.replaceFunction(0x001ABC30u, &forward1abc30);
        runtime.replaceFunction(0x001ABD20u, &chunkListInsert1abd20);
        runtime.replaceFunction(0x001ABD60u, &chunkListMethod1abd60);
        runtime.replaceFunction(0x001ABD80u, &chunkListMethod1abd80);
        runtime.replaceFunction(0x001ABDB0u, &chunkListMethod1abd80);
        runtime.replaceFunction(0x001ABF30u, &chunkListMethod1abf30);
        runtime.replaceFunction(0x001ABF60u, &chunkListMethod1abf60);
        runtime.replaceFunction(0x00101E80u, &getRef101e80);
        runtime.replaceFunction(0x001A23F0u, &noop1a23f0);
        runtime.replaceFunction(0x001BAA90u, &resourceNodeCopy1baa90);
        runtime.replaceFunction(0x001B7E10u, &resourceDescEquals1b7e10);
        runtime.replaceFunction(0x00179320u, &leafRet1_179320);
        runtime.replaceFunction(0x0014E210u, &resourceConfig14e210);
        runtime.replaceFunction(0x00157BC0u, &vif1Sync157bc0);
        std::cerr << "[SCUS_972.11] applying ATV Offroad Fury 2 override" << std::endl;
        runtime.setMissingFunctionPolicy(PS2Runtime::MissingFunctionPolicy::SkipCallDebug);
        g_ps2SkipSceGsSyncVWait = 1u;
        // ATV2 guest frame counter = 0x489008 (read by the frame-lockstep limiter
        // FUN_0015da90: `while (iRam00489008 < param_1)`). Its only in-game writer is the
        // GS-flip tail of sub_0015D4A0 (called from sub_001CD090), which never runs because
        // the game's VBlank INTC handler (FUN_00156590, registered on channel 9) is not
        // dispatched by the runtime as a frame tick. Host pump bumps this once per host frame
        // (ps2_runtime.cpp bumpGuestCounter). The timer-0 counter 0x488f38 is advanced at 10 Hz
        // real time by frameLimiter15da90; ATV2_TIMER0_PER_FRAME=1 restores the old per-frame bump.
        g_ps2GuestVsyncCounterAddr = 0x00489008u;
        g_ps2GuestVsyncCounterAddr2 = timer0CounterPerHostFrame() ? 0x00488f38u : 0u;

        // pc=0 resume machinery (ATV1 pattern): when main thread goes Dormant at pc=0,
        // EeScheduler's usableResume() will resume it to the GS flip handler (0x164160)
        // if the last non-zero PC was in the frame-limiter range.
        g_ps2Pc0ResumeRangeBegin = 0x15DA90u;
        g_ps2Pc0ResumeRangeEnd = 0x15E500u;
        g_ps2Pc0ResumeTarget = 0x164160u;  // GS flip entry — writes 0x489008
        g_ps2IdleResumePc = 0x164160u;
        g_ps2IdleResumeA0 = 0u;

        // Static-init ctor table walker FUN_0034d9f0 JALR target missed by the analyzer's
        // jal heuristic. Slot is empty in the generated dense table; fill it natively.
        runtime.replaceFunction(0x0044c100u, &ctor44c100);

        // default_new_handler__3stdFv @ 0x34d4b0 — called via data-table pointer from __nw__FUi
        // (sub_0034D5A0 at 0x34d5d0 JALRs to it). Merged fn sub_0034D3A0 has no re-entry case for 0x34d4b0.
        // Dense table now spans to 0x4515f4 so replaceFunction succeeds.
        runtime.replaceFunction(0x0034d4b0u, &default_new_handler_34d4b0);

        // Session-lock enter @ 0x341e60 — vtable slot [8] target missed by the analyzer.
        runtime.replaceFunction(0x00341e60u, &sessionLockEnter341e60);

        // Session-lock leave @ 0x341f30 — vtable slot [c] twin, also missed.
        runtime.replaceFunction(0x00341f30u, &sessionLockLeave341f30);

        // Keep the generated lock wrapper constructor/destructor (@ 0x341f90 / 0x341fd0)
        // so guest call/return state and virtual dispatch are preserved.
        // runtime.replaceFunction(0x00341f90u, &lockWrapperEnter341f90);
        // runtime.replaceFunction(0x00341fd0u, &lockWrapperLeave341fd0);

        // 1-instruction vector ctor @ 0x11a3d0 (array-init via FUN_0034d0d0).
        runtime.replaceFunction(0x0011a3d0u, &ctor11a3d0);

        // 4×VF00-quad ctor @ 0x16b5e0 (array-init: writes {0,0,0,1} at +0x20..+0x5F).
        runtime.replaceFunction(0x0016b5e0u, &ctor16b5e0);

        // Element ctor @ 0x137a10 (queue/stream node init, stride 0x14).
        runtime.replaceFunction(0x00137a10u, &ctor137a10);

        // Element ctor @ 0x331690 (stride 0x18, called by __arrctor 0x34d0d0).
        runtime.replaceFunction(0x00331690u, &ctor331690);
        runtime.replaceFunction(0x001C5430u, &ctor1c5430);
        runtime.replaceFunction(0x001C5170u, &dtor1c5170);
        runtime.replaceFunction(0x001C61E0u, &ctor1c61e0);
        runtime.replaceFunction(0x001AF900u, &ctor1af900);
        runtime.replaceFunction(0x0011A3B0u, &ctor11a3b0);
        runtime.replaceFunction(0x0011A3A0u, &ctorReturnThis);
        runtime.replaceFunction(0x00321BD0u, &ctor321bd0);
        runtime.replaceFunction(0x00206690u, &method206690);
        runtime.replaceFunction(0x002B4710u, &method2b4710);
        runtime.replaceFunction(0x0023BFC0u, &loadProps23bfc0);
        runtime.replaceFunction(0x00240EB0u, &loadProps240eb0);
        runtime.replaceFunction(0x001B8150u, &loadProps1b8150);
        runtime.replaceFunction(0x001B8790u, &loadProps1b8790);
        runtime.replaceFunction(0x0032D320u, &ctorReturnThis);
        runtime.replaceFunction(0x002CA1F0u, &ctor2ca1f0);

        // No-op accessor @ 0x153970 (returns 3rd arg).
        runtime.replaceFunction(0x00153970u, &noopRetA2_153970);

        // Display-mode vtable method @ 0x155950 (mode init, missed IndirectCall JALR).
        runtime.replaceFunction(0x00155950u, &modeDisplayInit155950);

        // Node-lookup tail-call wrapper @ 0x187880.
        runtime.replaceFunction(0x00187880u, &nodeLookup187880);

        // Node-release tail-call wrapper @ 0x1878a0.
        runtime.replaceFunction(0x001878a0u, &nodeRelease1878a0);

        // VBlank-chain vtable method @ 0x156e80 (JALR from 0x1538ac in the VBlank
        // chain). Merged boundary; re-entry switch lacks case 0x156e80u.
        replaceIfMissing(runtime, 0x00156e80u, &vtableMethod156e80);

        // Boot-path vtable method @ 0x176d60 (JALR from 0x10455c after the display
        // system allocates its object at 0x4891C0). Merged boundary; missed target.
        // callRecompiled4 loses the rest of the init when the pool allocator yields, and the
        // packet codes it writes are truncated; generated code is used whenever it has the entry.
        replaceIfMissing(runtime, 0x00176d60u, &vtableMethod176d60);

        // Sound driver RPC dispatcher (PS2SOUND.IRX, sid=0x34567).
        runtime.replaceFunction(0x002f0150u, &soundRpcCall2f0150);

        // Display/engine IOP RPC (sid=0x80000211).
        runtime.replaceFunction(0x001a8cc0u, &displayRpcCall1a8cc0);

        // MTAPMAN (Multitap Manager) stubs: report single controller connected.
        runtime.replaceFunction(0x00378928u, &mtapGetConnection378928);
        runtime.replaceFunction(0x00378848u, &mtapChangeThreadPriority378848);
        runtime.replaceFunction(0x003785c8u, &mtapInit3785c8);

        // CDVD streaming status / check (sid=0x8000059A).
        runtime.replaceFunction(0x00376408u, &cdvdStreamStatus376408);

        // newlib _strtod_r. The TOML names this routine but its heavy-loop skip leaves
        // 0x370430 without a generated entry; the strtod wrapper tail-jumps here.
        runtime.replaceFunction(0x00370430u, &strtodR370430);

        // Movie & video playback bypass
        // Movies are bypassed by default. ATV2_PLAY_MOVIES=1 plays them through the sceMpeg HLE
        // (Rainbow plays to the end), but the front end then stalls on the title screen: its node
        // is detached right after being attached (FUN_00256a70 from a resumed FE update) - open.
        if (const char *play = std::getenv("ATV2_PLAY_MOVIES"); play == nullptr || play[0] == '0')
        {
            runtime.replaceFunction(0x0025B7C0u, &movieStart25b7c0);
            runtime.replaceFunction(0x0018AC80u, &mediaIsFinished18ac80);
            runtime.replaceFunction(0x0018AC20u, &mediaStop18ac20);
            runtime.replaceFunction(0x0018A510u, &mediaStartPlayback18a510);
        }

        // Frame limiter: ticks FrontendManager update and draw every frame
        runtime.replaceFunction(0x0015DA90u, &frameLimiter15da90);
        g_videoCardFlipOrig = runtime.lookupFunction(0x0015E370u);
        if (g_videoCardFlipOrig)
        {
            g_flipRuntime = &runtime;
            g_flipFenceNow = std::getenv("ATV_FLIP_FENCE_NOW") != nullptr;
            runtime.replaceFunction(0x0015E370u, &videoCardFlip15e370);
            g_ps2GsCsrReadHook = &csrReadForFlip;
        }

        // Diagnostics wrapper for func_1DFB00
        runtime.replaceFunction(0x001DFB00u, &func1dfb00_wrapper);

        // Diagnostics wrappers for State 2 sub-calls
        runtime.replaceFunction(0x00256A70u, &func256a70_wrapper);
        runtime.replaceFunction(0x002FF730u, &func2ff730_wrapper);

        // Virtual destructor for Node subclass @ 0x00152D50
        runtime.replaceFunction(0x00152D50u, &dtor152d50);

        // Virtual destructor for Node subclass @ 0x00108070
        runtime.replaceFunction(0x00108070u, &dtor108070);

        // MSL soft-float dptoul (Double-Precision to Unsigned Long) @ 0x003B5590
        runtime.replaceFunction(0x003B5590u, &dptoul3b5590);

        // Virtual destructor for Resource node subclass @ 0x001BD180
        runtime.replaceFunction(0x001BD180u, &dtor1bd180);

        // Virtual destructor for Node subclass @ 0x00137EF0
        runtime.replaceFunction(0x00137EF0u, &dtor137ef0);

        // Virtual destructors for Resource node subclasses @ 0x001CE0E0, 0x001CE140, 0x001CE330
        runtime.replaceFunction(0x001CE0E0u, &dtor1ce0e0);
        runtime.replaceFunction(0x001CE140u, &dtor1ce140);
        runtime.replaceFunction(0x001CE330u, &dtor1ce330);

        // Virtual destructor for VertexBuffer node subclass @ 0x001C2A70
        runtime.replaceFunction(0x001C2A70u, &dtor1c2a70);

        // Virtual destructors for UI node subclasses: Highlight_Icon, Animated_Icon, Slider, TextInput
        runtime.replaceFunction(0x002AD640u, &dtor2ad640);
        runtime.replaceFunction(0x002AE5A0u, &dtor2ae5a0);
        runtime.replaceFunction(0x002AE8F0u, &dtor2ae8f0);
        runtime.replaceFunction(0x00236440u, &dtor236440);

        // Virtual destructor for Resource node subclass @ 0x0014E170
        runtime.replaceFunction(0x0014E170u, &dtor14e170);

        // Virtual destructors for SearchList classes @ 0x001A2160 and 0x001ABB40
        runtime.replaceFunction(0x001A2160u, &dtor1a2160);
        runtime.replaceFunction(0x001ABB40u, &dtor1abb40);

        // Destructors for PSXIcon family classes
        runtime.replaceFunction(0x00342060u, &dtor342060);
        runtime.replaceFunction(0x003424B0u, &dtor3424b0);
        runtime.replaceFunction(0x00343D60u, &dtor343d60);
        runtime.replaceFunction(0x003441E0u, &dtor3441e0);
        runtime.replaceFunction(0x003426D0u, &dtor3426d0);
        runtime.replaceFunction(0x003429C0u, &dtor3429c0);

        // Structure initializer and forwarding thunk for viewport/scene init
        runtime.replaceFunction(0x002F1820u, &init2F1820);
        runtime.replaceFunction(0x00151970u, &thunk151970);

        // PSXIcon destructor epilogue @ 0x003424F0
        runtime.replaceFunction(0x003424F0u, &epilogue3424f0);

        // Virtual destructor for Node/Resource subclass @ 0x001B8DB0
        runtime.replaceFunction(0x001B8DB0u, &dtor1b8db0);

        // Virtual destructor for Node subclass @ 0x0012B160
        runtime.replaceFunction(0x0012B160u, &dtor12b160);

        // Virtual destructor for PSXSurfaceShader subclass @ 0x001BABA0
        runtime.replaceFunction(0x001BABA0u, &dtor1baba0);

        // Base destructor for PSXSurfaceShader base @ 0x001C0570
        runtime.replaceFunction(0x001C0570u, &dtorWrapper1c0570);

        // Screen element constructor @ 0x002049F0
        runtime.replaceFunction(0x002049F0u, &ctor2049f0);

        // Screen element constructor @ 0x00204910
        runtime.replaceFunction(0x00204910u, &ctor204910);

        // Screen element constructor @ 0x002046D0
        runtime.replaceFunction(0x002046D0u, &ctor2046d0);

        // Screen element constructor @ 0x002047F0
        runtime.replaceFunction(0x002047F0u, &ctor2047f0);

        // Scene member destructor @ 0x00206DB0
        runtime.replaceFunction(0x00206DB0u, &dtorWrapper206db0);

        // Scene member array destructor @ 0x00207930
        runtime.replaceFunction(0x00207930u, &dtorWrapper207930);

        // Scene subclass destructor @ 0x0020C6C0
        runtime.replaceFunction(0x0020C6C0u, &dtor20c6c0);

        // PSXSurfaceGeometry base destructor @ 0x001C0880
        runtime.replaceFunction(0x001C0880u, &dtorWrapper1c0880);

        // PSXRenderTarget base destructor @ 0x001585A0
        runtime.replaceFunction(0x001585A0u, &dtorWrapper1585a0);

        // PSXRenderTarget virtual destructor @ 0x001587A0
        runtime.replaceFunction(0x001587A0u, &dtor1587a0);

        // WeatherInfo virtual destructor @ 0x00209040
        runtime.replaceFunction(0x00209040u, &dtor209040);

        // TerrainZoneInfo constructor @ 0x00202770
        runtime.replaceFunction(0x00202770u, &ctor202770);

        // Utility @ 0x0034A340
        runtime.replaceFunction(0x0034A340u, &util34a340);

        // VisibilityQuadTreeNode constructor @ 0x00162800
        runtime.replaceFunction(0x00162800u, &ctor162800);

        // VisibilityQuadTreeLeaf constructor @ 0x0016DE70
        runtime.replaceFunction(0x0016DE70u, &ctor16de70);

        // VisibilityQuadTreeLeaf factory @ 0x0016DE30
        runtime.replaceFunction(0x0016DE30u, &factory16de30);

        // Secondary function entry point @ 0x001AC030
        runtime.replaceFunction(0x001AC030u, &sub_001ABAE0_0x1abae0);

        // Secondary function entry point @ 0x001D47E0
        runtime.replaceFunction(0x001D47E0u, &sub_001D4790_0x1d4790);

        // Secondary function entry point @ 0x001B9EC0
        runtime.replaceFunction(0x001B9EC0u, &sub_001B9E80_0x1b9e80);

        // Secondary function entry point @ 0x00177A60 and fused sub-functions
        runtime.replaceFunction(0x00177180u, &sub_00177160_0x177160);
        runtime.replaceFunction(0x00177190u, &sub_00177160_0x177160);
        runtime.replaceFunction(0x001771E0u, &sub_00177160_0x177160);
        runtime.replaceFunction(0x00177A60u, &sub_00177160_0x177160);
        runtime.replaceFunction(0x00177AC0u, &sub_00177160_0x177160);
        runtime.replaceFunction(0x00177AD0u, &sub_00177160_0x177160);
        runtime.replaceFunction(0x00177AE0u, &sub_00177160_0x177160);
        runtime.replaceFunction(0x00177AF0u, &sub_00177160_0x177160);

        // Secondary function entry point @ 0x002D4F20
        runtime.replaceFunction(0x002D4F20u, &sub_002D4EE0_0x2d4ee0);

        // UI & TitleScreen secondary entry points for virtual dispatch
        runtime.replaceFunction(0x001D3AE0u, &sub_001D3A40_0x1d3a40);
        runtime.replaceFunction(0x001D3F50u, &sub_001D3F30_0x1d3f30);
        runtime.replaceFunction(0x001D4140u, &sub_001D4130_0x1d4130);
        runtime.replaceFunction(0x001D4200u, &sub_001D41C0_0x1d41c0);
        runtime.replaceFunction(0x001D4260u, &sub_001D41C0_0x1d41c0);
        runtime.replaceFunction(0x001D44B0u, &sub_001D4400_0x1d4400);
        runtime.replaceFunction(0x001D49D0u, &sub_001D4890_0x1d4890);
        runtime.replaceFunction(0x001D4B20u, &sub_001D4890_0x1d4890);
        runtime.replaceFunction(0x001D4C30u, &sub_001D4890_0x1d4890);
        runtime.replaceFunction(0x001D4CC0u, &sub_001D4890_0x1d4890);

        // Subclass virtual methods in sub_0014EBB0
        runtime.replaceFunction(0x0014EBD0u, &sub_0014EBB0_0x14ebb0);
        runtime.replaceFunction(0x0014ED40u, &sub_0014EBB0_0x14ebb0);
        runtime.replaceFunction(0x0014EF60u, &sub_0014EBB0_0x14ebb0);
        runtime.replaceFunction(0x0014EF70u, &sub_0014EBB0_0x14ebb0);
        runtime.replaceFunction(0x0014EF80u, &sub_0014EBB0_0x14ebb0);
        runtime.replaceFunction(0x0014EFB0u, &sub_0014EBB0_0x14ebb0);
        runtime.replaceFunction(0x0014EFE0u, &sub_0014EBB0_0x14ebb0);

        // Subclass virtual methods in sub_0022FB70 and sub_0022FDA0
        runtime.replaceFunction(0x0022FBB0u, &sub_0022FB70_0x22fb70);
        runtime.replaceFunction(0x0022FF00u, &sub_0022FDA0_0x22fda0);

        // PSXIcon methods in sub_00342750 and sub_00342AD0
        runtime.replaceFunction(0x003427E0u, &sub_00342750_0x342750);
        runtime.replaceFunction(0x00342B10u, &sub_00342AD0_0x342ad0);

        // Animated_Icon methods in sub_002AE560
        runtime.replaceFunction(0x002AE5A0u, &sub_002AE560_0x2ae560);
        runtime.replaceFunction(0x002AE630u, &sub_002AE560_0x2ae560);
        runtime.replaceFunction(0x002AE660u, &sub_002AE560_0x2ae560);
        runtime.replaceFunction(0x002AE670u, &sub_002AE560_0x2ae560);
        runtime.replaceFunction(0x002AE750u, &sub_002AE560_0x2ae560);
        runtime.replaceFunction(0x002AE830u, &sub_002AE560_0x2ae560);

        // UI Slider methods in sub_002AE840
        runtime.replaceFunction(0x002AE8F0u, &sub_002AE840_0x2ae840);
        runtime.replaceFunction(0x002AEBF0u, &sub_002AE840_0x2ae840);
        runtime.replaceFunction(0x002AEC10u, &sub_002AE840_0x2ae840);
        runtime.replaceFunction(0x002AEC20u, &sub_002AE840_0x2ae840);

        // UI Node Draw / Accessors in sub_00239940
        runtime.replaceFunction(0x00239950u, &sub_00239940_0x239940);
        runtime.replaceFunction(0x00239980u, &sub_00239940_0x239940);
        runtime.replaceFunction(0x002399B0u, &sub_00239940_0x239940);

        // UI Node Update in sub_002399F0
        runtime.replaceFunction(0x00239A00u, &sub_002399F0_0x2399f0);
        runtime.replaceFunction(0x00239A10u, &sub_002399F0_0x2399f0);

        // Game/Renderer virtual dispatch methods in sub_00104290 and sub_00153680
        runtime.replaceFunction(0x00104680u, &sub_00104290_0x104290);
        runtime.replaceFunction(0x00153840u, &sub_00153680_0x153680);
        runtime.replaceFunction(0x00153850u, &sub_00153680_0x153680);
        runtime.replaceFunction(0x00153860u, &sub_00153680_0x153680);
        runtime.replaceFunction(0x00153870u, &sub_00153680_0x153680);
        runtime.replaceFunction(0x001538C0u, &sub_00153680_0x153680);

        registerFadeStepFix(runtime);
        registerFeTrace(runtime);
        registerHeapTrace(runtime);
        registerVif1TextureIbit(runtime);
        registerIpuTextureDecode(runtime);
        registerPadHle(runtime);
        registerAtv2NetworkHle(runtime);
        registerSuzukiDz400(runtime);
        atvInstallCrashTrace();
    }
}

PS2_REGISTER_GAME_OVERRIDE(
    "atv-offroad-fury-2-us",
    "SCUS_972.11",
    0x00100008u,
    0u,
    applyScus97211Overrides);
