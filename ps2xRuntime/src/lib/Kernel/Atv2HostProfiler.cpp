// Host sampling profiler (ported from ATV1's scus_97104_overrides.cpp). A sampler thread suspends the
// profiled thread about every millisecond, stack-walks it and periodically logs the host symbols with
// the most exclusive/inclusive samples. Symbol names need a PDB: link with _LINK_=/DEBUG.
//   ATV_HOST_PROFILE=1              sample the game thread from the race countdown on (scus overrides)
//   ATV_HOST_PROFILE_GS=1|2         sample the GS worker in the race / from boot (gs_frontend.cpp)
//   ATV_HOST_PROFILE_INTERVAL=<s>   seconds per report (default 20)
//   ATV_HOST_PROFILE_LINES=1        also bucket leaf samples by source line (needs debug info)
//   ATV_HOST_PROFILE_FILE=<name>    with LINES: every sampled line of that file
// Do not profile two threads in one run: DbgHelp is not thread-safe.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>

#include <algorithm>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#pragma comment(lib, "dbghelp.lib")

namespace
{
    struct HostProfSym
    {
        std::string name;
        uint64_t self = 0u;
        uint64_t incl = 0u;
    };

    // Unwinds with RtlVirtualUnwind only while the target is suspended: it may hold the heap
    // or dbghelp locks, so nothing here may allocate until ResumeThread.
    int captureHostStack(HANDLE target, DWORD64 *pcs, int maxPcs)
    {
        CONTEXT c{};
        c.ContextFlags = CONTEXT_FULL;
        if (!GetThreadContext(target, &c))
            return 0;
        int n = 0;
        while (n < maxPcs && c.Rip != 0u)
        {
            pcs[n++] = c.Rip;
            DWORD64 imageBase = 0u;
            PRUNTIME_FUNCTION fe = RtlLookupFunctionEntry(c.Rip, &imageBase, nullptr);
            if (fe == nullptr)
            {
                if (n > 1)
                    break;
                c.Rip = *reinterpret_cast<const DWORD64 *>(c.Rsp);
                c.Rsp += 8u;
                continue;
            }
            PVOID handlerData = nullptr;
            DWORD64 establisher = 0u;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, c.Rip, fe, &c, &handlerData, &establisher, nullptr);
        }
        return n;
    }

    void hostProfilerLoop(HANDLE target)
    {
        HANDLE proc = GetCurrentProcess();
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        SymInitialize(proc, nullptr, TRUE);
        std::unordered_map<DWORD64, uint32_t> pcToSym;
        std::unordered_map<std::string, uint32_t> nameToSym;
        std::vector<HostProfSym> syms;
        auto symFor = [&](DWORD64 pc) -> uint32_t {
            const auto it = pcToSym.find(pc);
            if (it != pcToSym.end())
                return it->second;
            alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
            auto *si = reinterpret_cast<SYMBOL_INFO *>(buf);
            si->SizeOfStruct = sizeof(SYMBOL_INFO);
            si->MaxNameLen = 255;
            DWORD64 disp = 0u;
            std::string name;
            if (SymFromAddr(proc, pc, &disp, si))
            {
                name.assign(si->Name, si->NameLen);
            }
            else
            {
                const DWORD64 base = SymGetModuleBase64(proc, pc);
                char tmp[48];
                std::snprintf(tmp, sizeof(tmp), "?mod+0x%llx", static_cast<unsigned long long>((pc - base) & ~0xFFFull));
                name = tmp;
            }
            uint32_t idx;
            const auto nit = nameToSym.find(name);
            if (nit != nameToSym.end())
            {
                idx = nit->second;
            }
            else
            {
                idx = static_cast<uint32_t>(syms.size());
                syms.push_back({name, 0u, 0u});
                nameToSym.emplace(name, idx);
            }
            pcToSym.emplace(pc, idx);
            return idx;
        };

        static const bool s_lines = std::getenv("ATV_HOST_PROFILE_LINES") != nullptr;
        std::unordered_map<DWORD64, uint32_t> pcToLine;
        std::unordered_map<std::string, uint32_t> nameToLine;
        std::vector<HostProfSym> lines;
        auto lineFor = [&](DWORD64 pc) -> uint32_t {
            const auto it = pcToLine.find(pc);
            if (it != pcToLine.end())
                return it->second;
            IMAGEHLP_LINE64 info{};
            info.SizeOfStruct = sizeof(info);
            DWORD disp = 0u;
            std::string name = "?";
            if (SymGetLineFromAddr64(proc, pc, &disp, &info) && info.FileName)
            {
                const char *file = std::strrchr(info.FileName, '\\');
                name = std::string(file ? file + 1 : info.FileName) + ":" + std::to_string(info.LineNumber);
            }
            uint32_t idx;
            const auto nit = nameToLine.find(name);
            if (nit != nameToLine.end())
            {
                idx = nit->second;
            }
            else
            {
                idx = static_cast<uint32_t>(lines.size());
                lines.push_back({name, 0u, 0u});
                nameToLine.emplace(name, idx);
            }
            pcToLine.emplace(pc, idx);
            return idx;
        };

        static const ULONGLONG s_intervalMs = []
        {
            const char *env = std::getenv("ATV_HOST_PROFILE_INTERVAL");
            const unsigned long secs = env ? std::strtoul(env, nullptr, 10) : 0ul;
            return static_cast<ULONGLONG>(secs ? secs : 20ul) * 1000ull;
        }();
        constexpr int kMaxDepth = 64;
        DWORD64 pcs[kMaxDepth];
        std::vector<uint32_t> seen;
        uint64_t samples = 0u;
        ULONGLONG lastReport = GetTickCount64();
        for (int report = 0; report < 40;)
        {
            Sleep(1);
            if (SuspendThread(target) == static_cast<DWORD>(-1))
                break;
            const int n = captureHostStack(target, pcs, kMaxDepth);
            ResumeThread(target);
            if (n <= 0)
                continue;
            ++samples;
            if (s_lines)
                ++lines[lineFor(pcs[0])].self;
            seen.clear();
            for (int d = 0; d < n; ++d)
            {
                const uint32_t s = symFor(d == 0 ? pcs[d] : pcs[d] - 1u);
                if (d == 0)
                    ++syms[s].self;
                if (std::find(seen.begin(), seen.end(), s) == seen.end())
                {
                    seen.push_back(s);
                    ++syms[s].incl;
                }
            }
            const ULONGLONG now = GetTickCount64();
            if (now - lastReport < s_intervalMs)
                continue;
            lastReport = now;
            ++report;
            std::vector<uint32_t> order(syms.size());
            for (uint32_t i = 0; i < order.size(); ++i)
                order[i] = i;
            std::fprintf(stderr, "[ATV2] host-prof report=%d thread=%lu samples=%llu\n", report,
                         static_cast<unsigned long>(GetThreadId(target)), static_cast<unsigned long long>(samples));
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return syms[a].self > syms[b].self; });
            for (size_t i = 0; i < order.size() && i < 30u && syms[order[i]].self != 0u; ++i)
            {
                const HostProfSym &s = syms[order[i]];
                std::fprintf(stderr, "[ATV2] host-prof self %5.1f%% %s\n", 100.0 * s.self / samples, s.name.c_str());
            }
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return syms[a].incl > syms[b].incl; });
            for (size_t i = 0; i < order.size() && i < 60u; ++i)
            {
                const HostProfSym &s = syms[order[i]];
                std::fprintf(stderr, "[ATV2] host-prof incl %5.1f%% %s\n", 100.0 * s.incl / samples, s.name.c_str());
            }
            if (s_lines)
            {
                std::vector<uint32_t> lineOrder(lines.size());
                for (uint32_t i = 0; i < lineOrder.size(); ++i)
                    lineOrder[i] = i;
                std::sort(lineOrder.begin(), lineOrder.end(),
                          [&](uint32_t a, uint32_t b) { return lines[a].self > lines[b].self; });
                for (size_t i = 0; i < lineOrder.size() && i < 60u && lines[lineOrder[i]].self != 0u; ++i)
                {
                    const HostProfSym &s = lines[lineOrder[i]];
                    std::fprintf(stderr, "[ATV2] host-prof line %5.2f%% %s\n", 100.0 * s.self / samples, s.name.c_str());
                }
                // Per-file totals: generated code spreads over many lines.
                std::vector<std::pair<std::string, uint64_t>> files;
                for (const HostProfSym &s : lines)
                {
                    const std::string file = s.name.substr(0, s.name.rfind(':'));
                    auto it = std::find_if(files.begin(), files.end(), [&](const auto &f) { return f.first == file; });
                    if (it == files.end())
                        files.emplace_back(file, s.self);
                    else
                        it->second += s.self;
                }
                std::sort(files.begin(), files.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
                for (size_t i = 0; i < files.size() && i < 15u && files[i].second != 0u; ++i)
                    std::fprintf(stderr, "[ATV2] host-prof file %5.2f%% %s\n", 100.0 * files[i].second / samples,
                                 files[i].first.c_str());
                static const char *s_lineFile = std::getenv("ATV_HOST_PROFILE_FILE");
                if (s_lineFile)
                {
                    for (const HostProfSym &s : lines)
                    {
                        const size_t colon = s.name.rfind(':');
                        if (s.self != 0u && colon != std::string::npos && s.name.compare(0, colon, s_lineFile) == 0)
                            std::fprintf(stderr, "[ATV2] host-prof fileline %s %u\n", s.name.c_str() + colon + 1,
                                         static_cast<unsigned>(s.self));
                    }
                }
                for (HostProfSym &s : lines)
                    s.self = 0u;
            }
            std::fflush(stderr);
            for (HostProfSym &s : syms)
            {
                s.self = 0u;
                s.incl = 0u;
            }
            samples = 0u;
        }
        CloseHandle(target);
    }

    void startSamplingThisThread()
    {
        HANDLE target = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &target,
                             THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0u))
            return;
        std::fprintf(stderr, "[ATV2] host-prof started thread=%lu\n", static_cast<unsigned long>(GetCurrentThreadId()));
        std::thread(hostProfilerLoop, target).detach();
    }
}

// Samples the calling thread once ATV_HOST_PROFILE is set (first call wins).
void atvStartHostProfiler()
{
    static const bool s_enabled = std::getenv("ATV_HOST_PROFILE") != nullptr;
    static std::once_flag s_once;
    if (!s_enabled)
        return;
    std::call_once(s_once, [] { startSamplingThisThread(); });
}

// Samples the calling thread (the GS workers call this when ATV_HOST_PROFILE_GS is set).
void atvStartHostProfilerForThisThread()
{
    startSamplingThisThread();
}

namespace
{
    // First few faults: code, thread, address and host call stack. Generated functions are named
    // after their guest entry (sub_XXXXXXXX), so the stack is the guest chain. C++ throws are not
    // faults (the EE scheduler switches threads with them).
    LONG CALLBACK crashTraceHandler(EXCEPTION_POINTERS *info)
    {
        static LONG s_reported = 0;
        const DWORD code = info->ExceptionRecord->ExceptionCode;
        if (code < 0xC0000000u || code == 0xE06D7363u /* C++ */ || code == EXCEPTION_GUARD_PAGE ||
            InterlockedIncrement(&s_reported) > 4)
            return EXCEPTION_CONTINUE_SEARCH;
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lock(s_mutex);
        HANDLE proc = GetCurrentProcess();
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        SymInitialize(proc, nullptr, TRUE);
        std::fprintf(stderr, "[ATV2] crash: exception 0x%08lx (info %llx 0x%llx) on thread %lu\n",
                     static_cast<unsigned long>(code),
                     static_cast<unsigned long long>(info->ExceptionRecord->ExceptionInformation[0]),
                     static_cast<unsigned long long>(info->ExceptionRecord->ExceptionInformation[1]),
                     static_cast<unsigned long>(GetCurrentThreadId()));
        CONTEXT c = *info->ContextRecord;
        for (int depth = 0; depth < 48 && c.Rip != 0u; ++depth)
        {
            alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
            auto *si = reinterpret_cast<SYMBOL_INFO *>(buf);
            si->SizeOfStruct = sizeof(SYMBOL_INFO);
            si->MaxNameLen = 255;
            DWORD64 disp = 0u;
            IMAGEHLP_LINE64 line{};
            line.SizeOfStruct = sizeof(line);
            DWORD lineDisp = 0u;
            const bool named = SymFromAddr(proc, c.Rip, &disp, si) != FALSE;
            const bool lined = SymGetLineFromAddr64(proc, c.Rip, &lineDisp, &line) != FALSE;
            std::fprintf(stderr, "[ATV2] crash:   #%d %s+0x%llx %s:%lu\n", depth, named ? si->Name : "?",
                         static_cast<unsigned long long>(disp), lined ? line.FileName : "",
                         lined ? static_cast<unsigned long>(line.LineNumber) : 0ul);
            DWORD64 imageBase = 0u;
            PRUNTIME_FUNCTION fe = RtlLookupFunctionEntry(c.Rip, &imageBase, nullptr);
            if (fe == nullptr)
                break;
            PVOID handlerData = nullptr;
            DWORD64 establisher = 0u;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, c.Rip, fe, &c, &handlerData, &establisher, nullptr);
        }
        std::fflush(stderr);
        return EXCEPTION_CONTINUE_SEARCH;
    }
}

// ATV_CRASH_TRACE=1: report host faults (see crashTraceHandler).
void atvInstallCrashTrace()
{
    static std::once_flag s_once;
    if (std::getenv("ATV_CRASH_TRACE") == nullptr)
        return;
    std::call_once(s_once, []
                   {
                       AddVectoredExceptionHandler(1, &crashTraceHandler);
                       std::set_terminate([]
                                          {
                                              try
                                              {
                                                  if (std::current_exception())
                                                      std::rethrow_exception(std::current_exception());
                                                  std::fprintf(stderr, "[ATV2] crash: std::terminate\n");
                                              }
                                              catch (const std::exception &e)
                                              {
                                                  std::fprintf(stderr, "[ATV2] crash: uncaught exception: %s\n", e.what());
                                              }
                                              catch (...)
                                              {
                                                  std::fprintf(stderr, "[ATV2] crash: uncaught non-std exception\n");
                                              }
                                              std::fflush(stderr);
                                              std::abort();
                                          }); });
}

#else
void atvStartHostProfiler()
{
}
void atvStartHostProfilerForThisThread()
{
}
void atvInstallCrashTrace()
{
}
#endif
