// Temporary debug aid: write-protects a host page backing guest RAM and reports which
// recompiled function writes a watched guest range.
#include "ps2_runtime.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
namespace
{
    std::vector<std::pair<uintptr_t, uint32_t>> g_hostToGuest;
    uint8_t *g_watchHost = nullptr;
    uint32_t g_watchGuest = 0u;
    uint32_t g_watchLen = 0u;
    void *g_pageBase = nullptr;
    std::atomic<int> g_reports{0};
    thread_local bool t_stepping = false;
    PVOID g_handler = nullptr;
    std::string g_ring[64];
    uint32_t g_ringCount = 0u;

    uint32_t guestForHost(uintptr_t rip, uintptr_t &delta)
    {
        auto it = std::upper_bound(g_hostToGuest.begin(), g_hostToGuest.end(), std::make_pair(rip, UINT32_MAX));
        if (it == g_hostToGuest.begin())
            return 0u;
        --it;
        delta = rip - it->first;
        return it->second;
    }

    void describe(uintptr_t rip, char *buf, size_t n)
    {
        uintptr_t delta = 0;
        const uint32_t g = guestForHost(rip, delta);
        if (g != 0u && delta < 0x40000u)
            std::snprintf(buf, n, "0x%x+%zx", g, static_cast<size_t>(delta));
        else
            std::snprintf(buf, n, "host:%p", reinterpret_cast<void *>(rip));
    }

    LONG CALLBACK handler(PEXCEPTION_POINTERS ep)
    {
        const DWORD code = ep->ExceptionRecord->ExceptionCode;
        if (code == EXCEPTION_SINGLE_STEP && t_stepping)
        {
            t_stepping = false;
            DWORD old = 0;
            if (g_pageBase)
                VirtualProtect(g_pageBase, 1, PAGE_READONLY, &old);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (code != EXCEPTION_ACCESS_VIOLATION || g_pageBase == nullptr)
            return EXCEPTION_CONTINUE_SEARCH;
        const uintptr_t addr = static_cast<uintptr_t>(ep->ExceptionRecord->ExceptionInformation[1]);
        const uintptr_t page = reinterpret_cast<uintptr_t>(g_pageBase);
        if (ep->ExceptionRecord->ExceptionInformation[0] != 1 || addr < page || addr >= page + 0x1000u)
            return EXCEPTION_CONTINUE_SEARCH;

        const uintptr_t lo = reinterpret_cast<uintptr_t>(g_watchHost);
        if (addr + 16u > lo && addr < lo + g_watchLen)
        {
            CONTEXT c = *ep->ContextRecord;
            char frames[14][48];
            int nf = 0;
            for (; nf < 14; ++nf)
            {
                describe(static_cast<uintptr_t>(c.Rip), frames[nf], sizeof(frames[nf]));
                DWORD64 imageBase = 0;
                PRUNTIME_FUNCTION fe = RtlLookupFunctionEntry(c.Rip, &imageBase, nullptr);
                if (!fe)
                {
                    ++nf;
                    break;
                }
                PVOID hd = nullptr;
                DWORD64 est = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, c.Rip, fe, &c, &hd, &est, nullptr);
                if (c.Rip == 0)
                {
                    ++nf;
                    break;
                }
            }
            const int r = ++g_reports;
            char line[1024];
            int len = std::snprintf(line, sizeof(line), "[ATV2:WWATCH] #%d write guest=0x%x by", r,
                                    g_watchGuest + static_cast<uint32_t>(addr - lo));
            for (int i = 0; i < nf && len > 0 && len < static_cast<int>(sizeof(line)); ++i)
                len += std::snprintf(line + len, sizeof(line) - len, " %s%s", i ? "<" : "", frames[i]);
            g_ring[g_ringCount++ & 63u] = line;
        }
        DWORD old = 0;
        VirtualProtect(g_pageBase, 1, PAGE_READWRITE, &old);
        ep->ContextRecord->EFlags |= 0x100u;
        t_stepping = true;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
}

void atv2ArmWriteWatch(uint8_t *rdram, uint32_t guestAddr, uint32_t len)
{
    g_ringCount = 0u;
    if (g_hostToGuest.empty())
    {
        g_hostToGuest.reserve(g_ps2RecompiledFunctionTableSlotCount);
        for (uint32_t i = 0; i < g_ps2RecompiledFunctionTableSlotCount; ++i)
            if (auto fn = g_ps2RecompiledFunctionTable[i])
                g_hostToGuest.emplace_back(reinterpret_cast<uintptr_t>(fn), g_ps2RecompiledFunctionTableBase + i * 4u);
        std::sort(g_hostToGuest.begin(), g_hostToGuest.end());
        g_hostToGuest.erase(std::unique(g_hostToGuest.begin(), g_hostToGuest.end(),
                                        [](auto &a, auto &b) { return a.first == b.first; }),
                            g_hostToGuest.end());
        g_handler = AddVectoredExceptionHandler(1, handler);
    }
    DWORD old = 0;
    if (g_pageBase != nullptr)
        VirtualProtect(g_pageBase, 1, PAGE_READWRITE, &old);
    g_watchHost = getMemPtr(rdram, guestAddr);
    g_watchGuest = guestAddr;
    g_watchLen = len;
    g_pageBase = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(g_watchHost) & ~uintptr_t(0xFFF));
    VirtualProtect(g_pageBase, 1, PAGE_READONLY, &old);
}
void atv2DumpWriteWatchRing()
{
    for (uint32_t k = g_ringCount > 64u ? g_ringCount - 64u : 0u; k < g_ringCount; ++k)
        std::fprintf(stderr, "%s\n", g_ring[k & 63u].c_str());
    std::fflush(stderr);
}
#else
void atv2ArmWriteWatch(uint8_t *, uint32_t, uint32_t) {}
void atv2DumpWriteWatchRing() {}
#endif
