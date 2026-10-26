#include "runtime/ps2_vu1.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <ps2_log.h>

#ifdef _WIN32
extern "C" __declspec(dllimport) int __stdcall QueryThreadCycleTime(void *thread, unsigned long long *cycles);
extern "C" __declspec(dllimport) void *__stdcall GetCurrentThread();
#endif

namespace
{
    // CPU cycles charged to this thread (excludes time preempted); falls back to TSC.
    uint64_t threadCycles()
    {
#ifdef _WIN32
        unsigned long long cycles = 0;
        QueryThreadCycleTime(GetCurrentThread(), &cycles);
        return cycles;
#else
        return __rdtsc();
#endif
    }
}

namespace
{
    // ATV_VU_VERIFY_FAST=1 cross-checks the scheduling and FMAC flag shortcuts
    // against the full computation and logs any mismatch.
    const bool g_vuVerifyFast = std::getenv("ATV_VU_VERIFY_FAST") != nullptr;
    uint32_t g_vuVerifyFastReports = 0;
    // Set while the fast loop runs a code image with no FS*/FM* reads: skips the
    // MAC/status flag pipeline; CLIP/Q/P keep their own queues. The fast loop
    // still waits for g_vuSkippedFlagsReady so run() ends on the same cycle.
    bool g_vuSkipMacFlags = false;
    // g_vuSkipMacFlags or g_vuMacOnlyFlags without approx, verify or FMAC stats: native upper ops
    // take VU1Native::fastUpper's lean FMAC path (one check per op).
    bool g_vuFmacLean = false;
    uint64_t g_vuSkippedFlagsReady = 0;
    // ATV_VU_FMAC_STATS=1: why fast FMAC ops leave the all-lanes-normal path
    // (printed by the VU workload replay). Indices: see kFmacStat*.
    const bool g_vuFmacStats = std::getenv("ATV_VU_FMAC_STATS") != nullptr;
    uint64_t g_vuFmacStat[10] = {};
    enum : uint32_t
    {
        kFmacStatOps,        // fast FMAC ops
        kFmacStatSlowOps,    // ops with at least one lane not provably normal
        kFmacStatSlowNoFlag, // ... of which MAC/status flags were skipped
        kFmacStatZero,       // slow lane: result +-0
        kFmacStatZeroFactor, // ... zero result of a plain MUL with a zero operand
        kFmacStatDenormal,   // slow lane: denormal result
        kFmacStatInfNan,     // slow lane: exponent 0xFF
        kFmacStatEdge,       // slow lane: exponent 1/2/0xFC..0xFE (range edge)
        kFmacStatProduct,    // slow lane: product-sum bound not met (result in range)
    };
    // Set while the fast loop runs a code image whose only flag reads are
    // FMEQ/FMAND/FMOR: status is never read, so FMAC ops queue just their MAC
    // value here (in issue order, one per cycle at most) instead of a flag
    // pipeline entry. applyPendingMac makes the ones visible by a cycle current.
    bool g_vuMacOnlyFlags = false;
    struct PendingMac
    {
        uint64_t readyCycle;
        uint32_t mac;
    };
    constexpr uint32_t kPendingMacSlots = 8u; // > FMAC latency, so a full ring's oldest entry is visible
    PendingMac g_vuPendingMac[kPendingMacSlots];
    uint32_t g_vuPendingMacHead = 0;
    uint32_t g_vuPendingMacCount = 0;

    void applyPendingMac(uint32_t &mac, uint64_t cycle)
    {
        while (g_vuPendingMacCount != 0u && g_vuPendingMac[g_vuPendingMacHead].readyCycle <= cycle)
        {
            mac = g_vuPendingMac[g_vuPendingMacHead].mac;
            g_vuPendingMacHead = (g_vuPendingMacHead + 1u) % kPendingMacSlots;
            --g_vuPendingMacCount;
        }
    }

    void pushPendingMac(uint32_t &mac, uint64_t readyCycle, uint32_t value)
    {
        if (g_vuPendingMacCount == kPendingMacSlots)
            applyPendingMac(mac, g_vuPendingMac[g_vuPendingMacHead].readyCycle);
        g_vuPendingMac[(g_vuPendingMacHead + g_vuPendingMacCount) % kPendingMacSlots] = {readyCycle, value};
        ++g_vuPendingMacCount;
        g_vuSkippedFlagsReady = std::max(g_vuSkippedFlagsReady, readyCycle);
    }
    // Set while the fast loop runs: CLIP and FCSET queue their clip flag here
    // (issue order, at most one per cycle) instead of a flag
    // pipeline entry, and clip readers apply the ones visible by their cycle
    // (applyPendingClip). Saves the commit pass a pipeline entry costs.
    bool g_vuClipRing = false;
    // Set while the fast loop runs a code image without status-flag reads: a
    // DIV/SQRT/RSQRT result schedules no commit pass; Q readers (execUpper,
    // the fast MULq) commit it once its cycle has come. A new FDIV op commits
    // the previous result first (it stalled until that result was ready).
    bool g_vuLazyQ = false;
    struct PendingClip
    {
        uint64_t readyCycle;
        uint32_t clip;
    };
    constexpr uint32_t kPendingClipSlots = 8u; // > FMAC latency, like the MAC ring
    PendingClip g_vuPendingClip[kPendingClipSlots];
    uint32_t g_vuPendingClipHead = 0;
    uint32_t g_vuPendingClipCount = 0;

    void applyPendingClip(uint32_t &clip, uint64_t cycle)
    {
        while (g_vuPendingClipCount != 0u && g_vuPendingClip[g_vuPendingClipHead].readyCycle <= cycle)
        {
            clip = g_vuPendingClip[g_vuPendingClipHead].clip;
            g_vuPendingClipHead = (g_vuPendingClipHead + 1u) % kPendingClipSlots;
            --g_vuPendingClipCount;
        }
    }
    // Set while the fast loop runs: stores write VU memory at issue. A queued
    // store would land at the next cycle boundary, before the next pair and
    // before PATH1 reads that cycle's qword, and a pair holds at most one
    // memory access, so nothing can observe the difference.
    bool g_vuDirectStores = false;

    // ATV_VU_FAST_VERIFY runs each fast-loop invocation twice (fast, then
    // interpreted from the same snapshot), compares the operands each pair
    // reads, final state, data memory and XGKICK packets, and keeps the
    // interpreted result.
    bool g_vuForceInterp = false;
    // ATV_VU_APPROX=1: approximate VU1 timing (not bit-exact against the
    // interpreter; validated by gameplay). Never applied to interpreter runs.
    const bool g_vuApprox = []
    {
        const char *env = std::getenv("ATV_VU_APPROX");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    uint64_t g_vuRecompRuns = 0;
    struct VuVerifyRecord
    {
        uint32_t pc;
        uint64_t cycle;
        float vf[32][4];
        int32_t vi[16];
        float acc[4];
        float q;
        float p;
        float i;
        uint32_t clip;
    };
    std::vector<VuVerifyRecord> *g_vuVerifyTrace = nullptr;
    std::vector<std::vector<uint8_t>> *g_vuKickCapture = nullptr;

    // Records only the operands the pair reads (pending writes may legitimately
    // land at different times in the two paths), plus scalar state.
    template <typename Usage>
    void recordVerifyPair(const VU1State &state, uint64_t cycle, const Usage &upper, const Usage &lower)
    {
        if (!g_vuVerifyTrace || g_vuVerifyTrace->size() >= 16384u)
            return;
        VuVerifyRecord &rec = g_vuVerifyTrace->emplace_back();
        std::memset(&rec, 0, sizeof(rec));
        rec.pc = state.pc;
        rec.cycle = cycle;
        for (const Usage *usage : {&upper, &lower})
        {
            for (uint32_t n = 0; n < usage->vfReadCount; ++n)
            {
                const uint8_t reg = usage->vfRead[n].reg;
                for (uint32_t c = 0; c < 4u; ++c)
                    if ((usage->vfRead[n].lanes & (1u << (3u - c))) != 0u)
                        rec.vf[reg][c] = state.vf[reg][c];
            }
            for (uint32_t reg = 0; reg < 16u; ++reg)
                if ((usage->viRead & (1u << reg)) != 0u)
                    rec.vi[reg] = state.vi[reg];
            for (uint32_t c = 0; c < 4u; ++c)
                if ((usage->accRead & (1u << (3u - c))) != 0u)
                    rec.acc[c] = state.acc[c];
        }
        rec.q = state.q;
        rec.p = state.p;
        rec.i = state.i;
        rec.clip = state.clip;
    }

    uint32_t floatBits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    // Returns a short description of the first architectural difference, or empty.
    std::string diffVerifyState(const float (&vfA)[32][4], const float (&vfB)[32][4],
                                const int32_t (&viA)[16], const int32_t (&viB)[16],
                                const float (&accA)[4], const float (&accB)[4],
                                float qA, float qB, float pA, float pB, float iA, float iB,
                                uint32_t clipA, uint32_t clipB)
    {
        std::string out;
        char buf[96];
        uint32_t shown = 0u;
        for (uint32_t r = 0; r < 32u; ++r)
            for (uint32_t c = 0; c < 4u; ++c)
                if (floatBits(vfA[r][c]) != floatBits(vfB[r][c]) && shown++ < 6u)
                {
                    std::snprintf(buf, sizeof(buf), " vf%02u.%c rc=%08x(%g) in=%08x(%g)", r, "xyzw"[c],
                                  floatBits(vfA[r][c]), vfA[r][c], floatBits(vfB[r][c]), vfB[r][c]);
                    out += buf;
                }
        for (uint32_t r = 0; r < 16u; ++r)
            if (viA[r] != viB[r] && shown++ < 10u)
            {
                std::snprintf(buf, sizeof(buf), " vi%02u rc=%d in=%d", r, viA[r], viB[r]);
                out += buf;
            }
        for (uint32_t c = 0; c < 4u; ++c)
            if (floatBits(accA[c]) != floatBits(accB[c]) && shown++ < 12u)
            {
                std::snprintf(buf, sizeof(buf), " acc.%c rc=%08x in=%08x", "xyzw"[c],
                              floatBits(accA[c]), floatBits(accB[c]));
                out += buf;
            }
        const auto scalar = [&](const char *name, float a, float b) {
            if (floatBits(a) != floatBits(b))
            {
                std::snprintf(buf, sizeof(buf), " %s rc=%08x in=%08x", name, floatBits(a), floatBits(b));
                out += buf;
            }
        };
        scalar("q", qA, qB);
        scalar("p", pA, pB);
        scalar("i", iA, iB);
        if (clipA != clipB)
        {
            std::snprintf(buf, sizeof(buf), " clip rc=%06x in=%06x", clipA, clipB);
            out += buf;
        }
        return out;
    }

    void reportRecompVerify(uint32_t run,
                            const std::vector<VuVerifyRecord> &recomp,
                            const std::vector<VuVerifyRecord> &interp,
                            const VU1State &stateA, const VU1State &stateB,
                            uint64_t cycleA, uint64_t cycleB,
                            const std::vector<uint8_t> &dataA, const uint8_t *dataB, uint32_t dataSize,
                            const std::vector<std::vector<uint8_t>> &kicksA,
                            const std::vector<std::vector<uint8_t>> &kicksB)
    {
        static uint32_t s_reports = 0u;
        static uint32_t s_ok = 0u;
        std::string problems;
        char buf[256];
        const size_t pairs = std::min(recomp.size(), interp.size());
        for (size_t n = 0; n < pairs; ++n)
        {
            const VuVerifyRecord &a = recomp[n];
            const VuVerifyRecord &b = interp[n];
            std::string diff = diffVerifyState(a.vf, b.vf, a.vi, b.vi, a.acc, b.acc,
                                               a.q, b.q, a.p, b.p, a.i, b.i, a.clip, b.clip);
            if (a.pc != b.pc || a.cycle != b.cycle || !diff.empty())
            {
                std::snprintf(buf, sizeof(buf), "\n  pair#%zu pc rc=%04x in=%04x cyc rc=%llu in=%llu prev=%04x,%04x,%04x:",
                              n, a.pc, b.pc,
                              static_cast<unsigned long long>(a.cycle), static_cast<unsigned long long>(b.cycle),
                              n >= 3 ? interp[n - 3].pc : 0u, n >= 2 ? interp[n - 2].pc : 0u,
                              n >= 1 ? interp[n - 1].pc : 0u);
                problems += buf;
                problems += diff;
                break;
            }
        }
        if (recomp.size() != interp.size())
        {
            std::snprintf(buf, sizeof(buf), "\n  pairs rc=%zu in=%zu", recomp.size(), interp.size());
            problems += buf;
        }
        std::string finalDiff = diffVerifyState(stateA.vf, stateB.vf, stateA.vi, stateB.vi, stateA.acc, stateB.acc,
                                                stateA.q, stateB.q, stateA.p, stateB.p, stateA.i, stateB.i,
                                                stateA.clip, stateB.clip);
        if (!finalDiff.empty() || stateA.pc != stateB.pc || cycleA != cycleB || stateA.ebit != stateB.ebit)
        {
            std::snprintf(buf, sizeof(buf), "\n  final pc rc=%04x in=%04x cyc rc=%llu in=%llu ebit rc=%d in=%d:",
                          stateA.pc, stateB.pc, static_cast<unsigned long long>(cycleA),
                          static_cast<unsigned long long>(cycleB), stateA.ebit ? 1 : 0, stateB.ebit ? 1 : 0);
            problems += buf;
            problems += finalDiff;
        }
        uint32_t dataDiffs = 0u;
        uint32_t firstData = 0u;
        for (uint32_t off = 0; off + 4u <= dataSize && off + 4u <= dataA.size(); off += 4u)
        {
            if (std::memcmp(dataA.data() + off, dataB + off, 4u) != 0)
            {
                if (dataDiffs++ == 0u)
                    firstData = off;
            }
        }
        if (dataDiffs != 0u)
        {
            uint32_t wa = 0, wb = 0;
            std::memcpy(&wa, dataA.data() + firstData, 4u);
            std::memcpy(&wb, dataB + firstData, 4u);
            std::snprintf(buf, sizeof(buf), "\n  data words=%u first=0x%04x rc=%08x in=%08x",
                          dataDiffs, firstData, wa, wb);
            problems += buf;
        }
        if (kicksA.size() != kicksB.size())
        {
            std::snprintf(buf, sizeof(buf), "\n  kicks rc=%zu in=%zu", kicksA.size(), kicksB.size());
            problems += buf;
        }
        for (size_t k = 0; k < std::min(kicksA.size(), kicksB.size()); ++k)
        {
            if (kicksA[k] != kicksB[k])
            {
                size_t off = 0;
                while (off < kicksA[k].size() && off < kicksB[k].size() && kicksA[k][off] == kicksB[k][off])
                    ++off;
                std::snprintf(buf, sizeof(buf), "\n  kick#%zu bytes rc=%zu in=%zu first diff at +0x%zx",
                              k, kicksA[k].size(), kicksB[k].size(), off);
                problems += buf;
                break;
            }
        }
        if (problems.empty())
        {
            ++s_ok;
            if (s_ok <= 3u || (s_ok % 1000u) == 0u)
                std::fprintf(stderr, "[vu-verify] run#%u ok pairs=%zu kicks=%zu (ok=%u recompRuns=%llu)\n",
                             run, interp.size(), kicksB.size(), s_ok,
                             static_cast<unsigned long long>(g_vuRecompRuns));
            return;
        }
        if (s_reports++ < 12u)
            std::fprintf(stderr, "[vu-verify] run#%u MISMATCH start=%04x%s\n", run,
                         interp.empty() ? 0u : interp.front().pc, problems.c_str());
    }

    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    enum CommitPipe : uint32_t
    {
        kPipeFlag,
        kPipeEfu,
        kPipeStore,
        kPipeVf,
        kPipeVi,
        kPipeAcc,
        kPipeCount
    };

    // Per-interpreter scheduling summary so commitReadyPipelines can skip
    // cycles where nothing becomes visible and only visit occupied slots.
    // nextReady is a lower bound on any queued readyCycle (0 forces a scan);
    // occupied is a superset of the valid slots in each pipeline.
    // Lives here because ps2_vu1.h is in the runner PCH.
    struct CommitHint
    {
        const void *owner = nullptr;
        uint64_t nextReady = 0;
        uint32_t occupied[kPipeCount]{};
    };
    CommitHint g_commitHints[4];
    CommitHint g_commitHintOverflow;

    void resetCommitHint(CommitHint &hint, uint32_t occupied)
    {
        hint.nextReady = 0;
        for (uint32_t &mask : hint.occupied)
            mask = occupied;
    }

    CommitHint &commitHintFor(const void *owner)
    {
        for (CommitHint &hint : g_commitHints)
        {
            if (hint.owner == owner)
                return hint;
        }
        for (CommitHint &hint : g_commitHints)
        {
            if (hint.owner == nullptr)
            {
                hint.owner = owner;
                resetCommitHint(hint, 0xFFFFFFFFu);
                return hint;
            }
        }
        resetCommitHint(g_commitHintOverflow, 0xFFFFFFFFu);
        return g_commitHintOverflow;
    }

    template <typename Entry, size_t N>
    void noteQueued(const void *owner, CommitPipe pipe,
                    const std::array<Entry, N> &pipeline, const Entry &entry)
    {
        static_assert(N <= 32u, "occupancy mask is 32 bits");
        CommitHint &hint = commitHintFor(owner);
        hint.occupied[pipe] |= 1u << static_cast<uint32_t>(&entry - pipeline.data());
        if (entry.readyCycle < hint.nextReady)
            hint.nextReady = entry.readyCycle;
    }

    void noteQueued(const void *owner, uint64_t readyCycle)
    {
        CommitHint &hint = commitHintFor(owner);
        if (readyCycle < hint.nextReady)
            hint.nextReady = readyCycle;
    }

    template <size_t N>
    constexpr uint32_t slotMask()
    {
        return N >= 32u ? 0xFFFFFFFFu : ((1u << N) - 1u);
    }

    // All-ones lanes selected by a DEST field (bit 3 = x).
    struct DestLaneMasks
    {
        alignas(16) uint32_t lanes[16][4];
        constexpr DestLaneMasks() : lanes{}
        {
            for (uint32_t dest = 0; dest < 16u; ++dest)
                for (uint32_t component = 0; component < 4u; ++component)
                    lanes[dest][component] = (dest & (8u >> component)) != 0u ? 0xFFFFFFFFu : 0u;
        }
    };
    constexpr DestLaneMasks kDestLaneMasks;

    inline __m128i vuDestMask(uint8_t dest)
    {
        return _mm_load_si128(reinterpret_cast<const __m128i *>(kDestLaneMasks.lanes[dest & 0xFu]));
    }

    // The ready tables index lanes by bit position (countr_zero of the lane
    // mask), unlike the DEST-ordered register values.
    struct BitLaneMasks64
    {
        alignas(32) uint64_t lanes[16][4];
        constexpr BitLaneMasks64() : lanes{}
        {
            for (uint32_t mask = 0; mask < 16u; ++mask)
                for (uint32_t bit = 0; bit < 4u; ++bit)
                    lanes[mask][bit] = (mask & (1u << bit)) != 0u ? ~0ull : 0ull;
        }
    };
    constexpr BitLaneMasks64 kBitLaneMasks64;

    // ready[b] = cycle for each set bit b of lanes.
    inline void vuSetReadyLanes(uint64_t *ready, uint8_t lanes, uint64_t cycle)
    {
        const __m256i mask = _mm256_load_si256(reinterpret_cast<const __m256i *>(kBitLaneMasks64.lanes[lanes & 0xFu]));
        _mm256_maskstore_epi64(reinterpret_cast<long long *>(ready), mask,
                               _mm256_set1_epi64x(static_cast<long long>(cycle)));
    }

    // Four-lane vuNormalizeOperand.
    inline __m128 vuNormalize4(__m128 value)
    {
        const __m128i bits = _mm_castps_si128(value);
        const __m128i sign = _mm_and_si128(bits, _mm_set1_epi32(static_cast<int>(0x80000000u)));
        const __m128i exponent = _mm_and_si128(bits, _mm_set1_epi32(0x7F800000));
        const __m128i zeroExp = _mm_cmpeq_epi32(exponent, _mm_setzero_si128());
        const __m128i maxExp = _mm_cmpeq_epi32(exponent, _mm_set1_epi32(0x7F800000));
        __m128i out = _mm_andnot_si128(_mm_or_si128(zeroExp, maxExp), bits);
        out = _mm_or_si128(out, _mm_and_si128(zeroExp, sign));
        out = _mm_or_si128(out, _mm_and_si128(maxExp, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF))));
        return _mm_castsi128_ps(out);
    }

    inline __m128i vuExponent4(__m128 value)
    {
        return _mm_and_si128(_mm_srli_epi32(_mm_castps_si128(value), 23), _mm_set1_epi32(0xFF));
    }

    inline __m128i vuInRange4(__m128i value, int low, int high)
    {
        return _mm_and_si128(_mm_cmpgt_epi32(value, _mm_set1_epi32(low - 1)),
                             _mm_cmpgt_epi32(_mm_set1_epi32(high + 1), value));
    }

    bool isFmacProductSum(uint32_t upper)
    {
        const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
        const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
        return (op >= 0x08u && op <= 0x0Fu) ||
               op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
               op == 0x29u || op == 0x2Du || op == 0x2Eu ||
               (special >= 0x08u && special <= 0x0Fu) ||
               special == 0x21u || special == 0x23u || special == 0x25u ||
               special == 0x27u || special == 0x29u || special == 0x2Du;
    }

    // Normalized factors of the product term of MADD/MSUB-family instructions
    // (same operand selection as calculateFmacProductSticky). Returns false
    // for any other instruction, including OPMSUB.
    bool fmacProductOperands(uint32_t upper, uint32_t component, const VU1State &state,
                             float &left, float &right)
    {
        const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
        const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
        const uint8_t sel = op >= 0x3Cu ? special : op;
        const uint8_t fs = FS(upper);
        const uint8_t ft = FT(upper);
        if (sel >= 0x08u && sel <= 0x0Fu)
            right = state.vf[ft][sel & 3u];
        else if (sel == 0x21u || sel == 0x25u)
            right = state.q;
        else if (sel == 0x23u || sel == 0x27u)
            right = state.i;
        else if (sel == 0x29u || sel == 0x2Du)
            right = state.vf[ft][component];
        else
            return false;
        left = vuNormalizeOperand(state.vf[fs][component]);
        right = vuNormalizeOperand(right);
        return true;
    }

    // Resets everything except the 64 KB packet buffer; only bytes copied in
    // by the current kick are ever read back from it.
    template <typename Xgkick>
    void resetXgkickState(Xgkick &xgkick)
    {
        xgkick.sourceAddress = 0;
        xgkick.totalBytes = 0;
        xgkick.copiedBytes = 0;
        xgkick.currentTagEnd = 0;
        xgkick.cycleCredit = 0;
        xgkick.issueCycle = 0;
        xgkick.active = false;
        xgkick.currentTagEop = false;
    }
}

#include "ps2_vu1_native.inc"

void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    reset();
}

void VU1Interpreter::resetScheduler()
{
    m_flagPipeline = {};
    m_fdiv = {};
    m_efu = {};
    m_storePipeline = {};
    m_vfWritePipeline = {};
    m_viWritePipeline = {};
    m_accWritePipeline = {};
    resetXgkickState(m_xgkick);
    m_vfReady = {};
    m_viReady = {};
    m_accReady = {};
    m_vfLatestWrite = {};
    m_viLatestWrite = {};
    m_accLatestWrite = {};
    m_nextWriteSequence = 0;
    m_efuResourceReady = 0;
    m_workingClip = m_state.clip;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
    resetCommitHint(commitHintFor(this), 0u);
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    resetScheduler();
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return vuBroadcast(vf, bc);
}

float VU1Interpreter::normalizeOperand(float value) const
{
    return vuNormalizeOperand(value);
}

float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4])
{
    const bool productSum = isFmacProductSum(m_currentUpperInstruction);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        // When the exact result is provably normal, nonzero and of the same
        // sign as the float result, both slow paths below report only the
        // sign flag and leave the value unchanged.
        //  - Single rounded add/sub/mul: a result exponent in [2, 0xFD]
        //    suffices.
        //  - acc +/- a*b: |f - exact| < ulp(a*b) + ulp(f). With the rounded
        //    product normal and exp(f) >= exp(a*b) - 20 that error is below
        //    |f|/8 + ulp(f), and a result exponent in [3, 0xFC] keeps the
        //    exact value inside the normal range. A zero factor makes the
        //    product exact, which reduces to the single-rounding case.
        uint32_t bits = 0;
        std::memcpy(&bits, &result[component], sizeof(bits));
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        bool provablyNormal = false;
        if (!productSum)
        {
            provablyNormal = exponent >= 2u && exponent <= 0xFDu;
        }
        else if (exponent >= 3u && exponent <= 0xFCu)
        {
            float left = 0.0f;
            float right = 0.0f;
            if (fmacProductOperands(m_currentUpperInstruction, component, m_state, left, right))
            {
                if (left == 0.0f || right == 0.0f)
                {
                    provablyNormal = true;
                }
                else
                {
                    const float product = left * right;
                    uint32_t productBits = 0;
                    std::memcpy(&productBits, &product, sizeof(productBits));
                    const uint32_t productExponent = (productBits >> 23) & 0xFFu;
                    provablyNormal = productExponent >= 1u && productExponent <= 0xFDu &&
                                     exponent + 20u >= productExponent;
                }
            }
        }
        if (provablyNormal)
        {
            laneFlags[component] = (bits & 0x80000000u) != 0u ? 0x2u : 0u;
            if (g_vuVerifyFast)
            {
                float slowValue = result[component];
                uint8_t slowFlags = 0u;
                long double exact = 0.0L;
                if (calculateFmacExactResult(component, exact))
                    slowFlags = normalizeFmacExactResult(slowValue, exact);
                else
                {
                    uint32_t flags = 0u;
                    slowValue = normalizeResult(slowValue, flags);
                    slowFlags = static_cast<uint8_t>(flags);
                }
                if ((slowFlags != laneFlags[component] ||
                     std::memcmp(&slowValue, &result[component], sizeof(float)) != 0) &&
                    g_vuVerifyFastReports < 16u)
                {
                    ++g_vuVerifyFastReports;
                    std::fprintf(stderr, "[vu-verify] fmac flag mismatch upper=%08x lane=%u value=%08x fast=%x slow=%x\n",
                                 m_currentUpperInstruction, component, bits, laneFlags[component], slowFlags);
                }
            }
            continue;
        }

        long double exactResult = 0.0L;
        if (calculateFmacExactResult(component, exactResult))
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
            continue;
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               long double &result) const
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<long double>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const long double q = operand(m_state.q);
    const long double i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0L
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0L
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  long double exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const long double magnitude = std::fabs(exactResult);
    const long double maximum = static_cast<long double>(std::numeric_limits<float>::max());
    const long double minimum = static_cast<long double>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0L)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest) const
{
    uint32_t extraSticky = 0u;
    const uint32_t upper = m_currentUpperInstruction;
    if (!isFmacProductSum(upper))
        return 0u;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = normalizeOperand(m_state.q);
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = normalizeOperand(m_state.i);
        }
        else if (op == 0x2Eu)
        {
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        }
        else
        {
            right = normalizeOperand(m_state.vf[ft][component]);
        }

        float product = left * right;
        uint32_t productBits = 0;
        std::memcpy(&productBits, &product, sizeof(productBits));
        const uint32_t productExponent = (productBits >> 23) & 0xFFu;
        // Same single-rounding argument as normalizeFmacResult: the exact
        // product is normal, nonzero and of the same sign.
        if (productExponent >= 2u && productExponent <= 0xFDu && !g_vuVerifyFast)
        {
            extraSticky |= (productBits & 0x80000000u) != 0u ? 0x2u : 0u;
            continue;
        }
        const long double exactProduct = static_cast<long double>(left) * static_cast<long double>(right);
        const uint8_t productFlags = normalizeFmacExactResult(product, exactProduct);
        if (g_vuVerifyFast && productExponent >= 2u && productExponent <= 0xFDu &&
            (productFlags & 0xFu) != ((productBits & 0x80000000u) != 0u ? 0x2u : 0u) &&
            g_vuVerifyFastReports < 16u)
        {
            ++g_vuVerifyFastReports;
            std::fprintf(stderr, "[vu-verify] fmac sticky mismatch upper=%08x lane=%u product=%08x slow=%x\n",
                         upper, component, productBits, productFlags);
        }
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;
    if (g_vuMacOnlyFlags)
    {
        uint32_t mac = 0u;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            const uint32_t lane = laneForComponent(component);
            if ((dest & lane) == 0u)
                continue;
            for (uint32_t bit = 0; bit < 4u; ++bit)
                if ((laneFlags[component] & (1u << bit)) != 0u)
                    mac |= lane << (4u * bit);
        }
        pushPendingMac(m_state.mac, m_cycle + kFmacLatency, mac);
        return;
    }
    if (g_vuSkipMacFlags)
    {
        g_vuSkippedFlagsReady = std::max(g_vuSkippedFlagsReady, m_cycle + kFmacLatency);
        return;
    }

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    FlagPipelineEntry *entry = nullptr;
    for (FlagPipelineEntry &candidate : m_flagPipeline)
    {
        if (!candidate.valid)
        {
            entry = &candidate;
            break;
        }
    }
    if (!entry)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return;
    }

    *entry = {};
    entry->valid = true;
    entry->issueCycle = m_cycle;
    entry->readyCycle = m_cycle + kFmacLatency;
    noteQueued(this, kPipeFlag, m_flagPipeline, *entry);
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDest(dst, result, dest);
}

void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDestAcc(result, dest);
}

void VU1Interpreter::queueFsset(uint16_t immediate)
{
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (entry.valid && entry.issueCycle == m_cycle)
            entry.writesStatus = false;
    }

    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (!entry.valid)
        {
            entry = {};
            entry.valid = true;
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            noteQueued(this, kPipeFlag, m_flagPipeline, entry);
            entry.status = static_cast<uint32_t>(immediate) & 0xFC0u;
            entry.writesSticky = true;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFEu);
}

// commitReadyPipelines calls that found a due entry (ATV_VU_NATIVE_STATS).
static uint64_t g_vuCommitPasses = 0;

void vu1ApplyPendingClip(uint32_t &clip, uint64_t cycle)
{
    if (g_vuPendingClipCount != 0u)
        applyPendingClip(clip, cycle);
}

void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    if (g_vuClipRing)
    {
        if (g_vuPendingClipCount == kPendingClipSlots)
            applyPendingClip(m_state.clip, g_vuPendingClip[g_vuPendingClipHead].readyCycle);
        const uint64_t ready = m_cycle + kFmacLatency;
        g_vuPendingClip[(g_vuPendingClipHead + g_vuPendingClipCount) % kPendingClipSlots] = {ready, m_workingClip};
        ++g_vuPendingClipCount;
        g_vuSkippedFlagsReady = std::max(g_vuSkippedFlagsReady, ready);
        return;
    }
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (!entry.valid)
        {
            entry = {};
            entry.valid = true;
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            noteQueued(this, kPipeFlag, m_flagPipeline, entry);
            entry.clip = m_workingClip;
            entry.writesClip = true;
            return;
        }
    }
    reportReservedInstruction(true, 0xFFFFFFFDu);
}

void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    if (g_vuClipRing)
    {
        // Replaces a CLIP issued this cycle (the only clip entries are in the
        // ring), else queues its own entry with the same latency.
        const uint64_t ready = m_cycle + kFmacLatency;
        const uint32_t last = (g_vuPendingClipHead + g_vuPendingClipCount + kPendingClipSlots - 1u) % kPendingClipSlots;
        if (g_vuPendingClipCount != 0u && g_vuPendingClip[last].readyCycle == ready)
        {
            g_vuPendingClip[last].clip = m_workingClip;
            return;
        }
        if (g_vuPendingClipCount == kPendingClipSlots)
            applyPendingClip(m_state.clip, g_vuPendingClip[g_vuPendingClipHead].readyCycle);
        g_vuPendingClip[(g_vuPendingClipHead + g_vuPendingClipCount) % kPendingClipSlots] = {ready, m_workingClip};
        ++g_vuPendingClipCount;
        g_vuSkippedFlagsReady = std::max(g_vuSkippedFlagsReady, ready);
        return;
    }
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (entry.valid && entry.issueCycle == m_cycle)
            entry.writesClip = false;
    }
    for (FlagPipelineEntry &entry : m_flagPipeline)
    {
        if (!entry.valid)
        {
            entry = {};
            entry.valid = true;
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            noteQueued(this, kPipeFlag, m_flagPipeline, entry);
            entry.clip = m_workingClip;
            entry.writesClip = true;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFAu);
}

void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    if (g_vuLazyQ)
    {
        if (m_fdiv.valid && m_fdiv.readyCycle <= m_cycle)
        {
            m_state.q = m_fdiv.value;
            const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
            m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            m_fdiv = {};
        }
    }
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    if (!g_vuLazyQ)
        noteQueued(this, m_fdiv.readyCycle);
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            noteQueued(this, kPipeEfu, m_efu, entry);
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    if (g_vuDirectStores)
    {
        if (m_activeVuData && address + 16u <= m_activeVuDataSize)
        {
            uint8_t *dst = m_activeVuData + address;
            const __m128 keep = _mm_castsi128_ps(vuDestMask(laneMask));
            const __m128 value = _mm_loadu_ps(reinterpret_cast<const float *>(words));
            _mm_storeu_ps(reinterpret_cast<float *>(dst),
                          _mm_or_ps(_mm_and_ps(keep, value),
                                    _mm_andnot_ps(keep, _mm_loadu_ps(reinterpret_cast<const float *>(dst)))));
        }
        return;
    }
    for (PendingStore &store : m_storePipeline)
    {
        if (!store.valid)
        {
            store.valid = true;
            store.readyCycle = m_cycle + 1u;
            noteQueued(this, kPipeStore, m_storePipeline, store);
            store.address = address;
            store.laneMask = laneMask;
            std::copy(words, words + 4, store.words.begin());
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFCu);
}

void VU1Interpreter::queueVfWrite(uint8_t reg, uint8_t laneMask,
                                  const float value[4], uint32_t latency)
{
    if (reg == 0u || laneMask == 0u)
        return;
    for (PendingVfWrite &write : m_vfWritePipeline)
    {
        if (!write.valid)
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            noteQueued(this, kPipeVf, m_vfWritePipeline, write);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_vfLatestWrite[reg][component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF7u);
}

void VU1Interpreter::queueViWrite(uint8_t reg, int32_t value, uint32_t latency)
{
    if (reg == 0u)
        return;
    for (PendingViWrite &write : m_viWritePipeline)
    {
        if (!write.valid)
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            noteQueued(this, kPipeVi, m_viWritePipeline, write);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.value = value;
            m_viLatestWrite[reg] = write.sequence;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF6u);
}

void VU1Interpreter::queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency)
{
    if (laneMask == 0u)
        return;
    for (PendingAccWrite &write : m_accWritePipeline)
    {
        if (!write.valid)
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            noteQueued(this, kPipeAcc, m_accWritePipeline, write);
            write.sequence = ++m_nextWriteSequence;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_accLatestWrite[component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(true, 0xFFFFFFF5u);
}

void VU1Interpreter::commitReadyPipelines()
{
    CommitHint &hint = commitHintFor(this);
    if (g_vuVerifyFast)
    {
        static bool s_announced = false;
        if (!s_announced)
        {
            s_announced = true;
            std::fprintf(stderr, "[vu-verify] enabled\n");
        }
        const auto check = [&](const char *pipe, uint32_t occupied, uint32_t slot, bool valid, uint64_t readyCycle)
        {
            if (!valid)
                return;
            const bool hiddenSlot = slot < 32u && (occupied & (1u << slot)) == 0u;
            const bool hiddenReady = readyCycle <= m_cycle && m_cycle < hint.nextReady;
            if ((hiddenSlot || hiddenReady) && g_vuVerifyFastReports < 16u)
            {
                ++g_vuVerifyFastReports;
                std::fprintf(stderr, "[vu-verify] commit hint mismatch pipe=%s slot=%u cycle=%llu ready=%llu hint=%llu occupied=%08x\n",
                             pipe, slot, static_cast<unsigned long long>(m_cycle),
                             static_cast<unsigned long long>(readyCycle),
                             static_cast<unsigned long long>(hint.nextReady), occupied);
            }
        };
        for (uint32_t i = 0; i < m_flagPipeline.size(); ++i)
            check("flag", hint.occupied[kPipeFlag], i, m_flagPipeline[i].valid, m_flagPipeline[i].readyCycle);
        check("fdiv", 0xFFFFFFFFu, 0u, m_fdiv.valid, m_fdiv.readyCycle);
        for (uint32_t i = 0; i < m_efu.size(); ++i)
            check("efu", hint.occupied[kPipeEfu], i, m_efu[i].valid, m_efu[i].readyCycle);
        for (uint32_t i = 0; i < m_storePipeline.size(); ++i)
            check("store", hint.occupied[kPipeStore], i, m_storePipeline[i].valid, m_storePipeline[i].readyCycle);
        for (uint32_t i = 0; i < m_vfWritePipeline.size(); ++i)
            check("vf", hint.occupied[kPipeVf], i, m_vfWritePipeline[i].valid, m_vfWritePipeline[i].readyCycle);
        for (uint32_t i = 0; i < m_viWritePipeline.size(); ++i)
            check("vi", hint.occupied[kPipeVi], i, m_viWritePipeline[i].valid, m_viWritePipeline[i].readyCycle);
        for (uint32_t i = 0; i < m_accWritePipeline.size(); ++i)
            check("acc", hint.occupied[kPipeAcc], i, m_accWritePipeline[i].valid, m_accWritePipeline[i].readyCycle);
    }
    if (m_cycle < hint.nextReady)
        return;
    ++g_vuCommitPasses;
    uint64_t nextReady = std::numeric_limits<uint64_t>::max();
    const auto notReady = [&](uint64_t readyCycle) {
        if (readyCycle <= m_cycle)
            return false;
        nextReady = std::min(nextReady, readyCycle);
        return true;
    };
    // Visits occupied slots in ascending index order (commit order matters
    // when several entries of one pipeline become visible on the same cycle)
    // and drops slots that are no longer valid from the mask.
    const auto forEachOccupied = [&](CommitPipe pipe, auto &pipeline, auto &&commit)
    {
        uint32_t &occupied = hint.occupied[pipe];
        occupied &= slotMask<std::tuple_size_v<std::remove_reference_t<decltype(pipeline)>>>();
        for (uint32_t bits = occupied; bits != 0u; bits &= bits - 1u)
        {
            const uint32_t slot = static_cast<uint32_t>(std::countr_zero(bits));
            auto &entry = pipeline[slot];
            if (entry.valid && notReady(entry.readyCycle))
                continue;
            if (entry.valid)
                commit(entry);
            occupied &= ~(1u << slot);
        }
    };

    forEachOccupied(kPipeFlag, m_flagPipeline, [&](FlagPipelineEntry &entry)
    {

        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
        {
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        }
        if (entry.writesClip)
            m_state.clip = entry.clip;
        entry = {};
    });

    if (m_fdiv.valid && !notReady(m_fdiv.readyCycle))
    {
        m_state.q = m_fdiv.value;
        const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
        m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
        m_fdiv = {};
    }

    forEachOccupied(kPipeEfu, m_efu, [&](ScalarPipelineEntry &entry)
    {
        m_state.p = entry.value;
        entry = {};
    });

    forEachOccupied(kPipeStore, m_storePipeline, [&](PendingStore &store)
    {
        if (m_activeVuData && store.address + 16u <= m_activeVuDataSize)
        {
            uint32_t oldWords[4]{};
            std::memcpy(oldWords, m_activeVuData + store.address, sizeof(oldWords));
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((store.laneMask & laneForComponent(component)) != 0u)
                    oldWords[component] = store.words[component];
            }
            std::memcpy(m_activeVuData + store.address, oldWords, sizeof(oldWords));
        }
        store = {};
    });

    forEachOccupied(kPipeVf, m_vfWritePipeline, [&](PendingVfWrite &write)
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_vfLatestWrite[write.reg][component] == write.sequence)
            {
                m_state.vf[write.reg][component] = write.value[component];
            }
        }
        write = {};
    });

    forEachOccupied(kPipeVi, m_viWritePipeline, [&](PendingViWrite &write)
    {
        if (m_viLatestWrite[write.reg] == write.sequence)
            m_state.vi[write.reg] = static_cast<int16_t>(write.value);
        write = {};
    });

    forEachOccupied(kPipeAcc, m_accWritePipeline, [&](PendingAccWrite &write)
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_accLatestWrite[component] == write.sequence)
            {
                m_state.acc[component] = write.value[component];
            }
        }
        write = {};
    });
    hint.nextReady = nextReady;
}

static uint32_t s_xgkickFaultStartPc = 0u;

static void logXgkickFault(const char *reason, uint32_t startPc, uint32_t pc, uint32_t top, uint32_t itop,
                           uint32_t source, uint32_t offset, const uint8_t *packet)
{
    static uint32_t s_logged = 0u;
    if (s_logged >= 24u)
        return;
    ++s_logged;
    uint32_t q[8]{};
    std::memcpy(q, packet + offset, 16u);
    if (offset >= 16u)
        std::memcpy(q + 4, packet + offset - 16u, 16u);
    std::fprintf(stderr,
                 "[VU1:XGKICK-FAULT] %s start=0x%x pc=0x%x top=0x%x itop=0x%x src=0x%x off=0x%x"
                 " tag=%08x.%08x.%08x.%08x prev=%08x.%08x.%08x.%08x\n",
                 reason, startPc, pc, top, itop, source, offset, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
}

void VU1Interpreter::progressXgkick()
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    ++m_xgkick.cycleCredit;
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            logXgkickFault("overflow", s_xgkickFaultStartPc, m_state.pc, m_state.top, m_state.itop,
                           m_xgkick.sourceAddress, m_xgkick.copiedBytes - 16u, m_xgkick.packet.data());
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        // VU data memory is a power of two (16 KB); avoid the division per qword.
        const uint32_t sourceOffset = m_xgkick.sourceAddress + m_xgkick.copiedBytes;
        const uint32_t firstSource = (m_activeVuDataSize & (m_activeVuDataSize - 1u)) == 0u
                                         ? sourceOffset & (m_activeVuDataSize - 1u)
                                         : sourceOffset % m_activeVuDataSize;
        if (firstSource + 16u <= m_activeVuDataSize)
        {
            std::memcpy(m_xgkick.packet.data() + qwordOffset, m_activeVuData + firstSource, 16u);
        }
        else
        {
            for (uint32_t i = 0; i < 16u; ++i)
            {
                const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes + i) % m_activeVuDataSize;
                m_xgkick.packet[m_xgkick.copiedBytes + i] = m_activeVuData[source];
            }
        }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else // IMAGE, or FLG 3 which the GIF treats as IMAGE
                tagBytes += static_cast<uint64_t>(nloop) * 16u;

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                logXgkickFault("tag-too-big", s_xgkickFaultStartPc, m_state.pc, m_state.top, m_state.itop,
                               m_xgkick.sourceAddress, qwordOffset, m_xgkick.packet.data());
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

uint32_t g_ps2XgkickCount = 0;
// VU1 cycles and instruction pairs executed by run(), for ATV_BENCHMARK.
uint64_t g_vu1CycleTotal = 0;
uint64_t g_vu1PairTotal = 0;
// Set by the game override once the player's racer is moving (gates VU workload capture).
bool g_atvRaceStarted = false;
uint64_t g_vu1DecodeRebuilds = 0;
// Fast-loop block replay: pairs replayed, schedules recorded, replays left
// early, replays started, replays entered through a link (no signature check).
uint64_t g_vu1ReplayPairs = 0;
uint64_t g_vu1ReplayRecords = 0;
uint64_t g_vu1ReplayBails = 0;
uint64_t g_vu1ReplayBlocks = 0;
uint64_t g_vu1ReplayLinked = 0;

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;
    ++g_ps2XgkickCount;

    if (g_vuKickCapture)
        g_vuKickCapture->emplace_back(m_xgkick.packet.data(), m_xgkick.packet.data() + m_xgkick.totalBytes);
    else if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
    {
        static uint32_t s_kickLogs = 0u;
        if (s_kickLogs < 8u)
        {
            uint32_t tag0 = 0u;
            uint32_t tag1 = 0u;
            uint32_t tag2 = 0u;
            uint32_t tag3 = 0u;
            if (m_xgkick.totalBytes >= 4u)
                std::memcpy(&tag0, m_xgkick.packet.data(), 4);
            if (m_xgkick.totalBytes >= 16u)
            {
                std::memcpy(&tag1, m_xgkick.packet.data() + 4, 4);
                std::memcpy(&tag2, m_xgkick.packet.data() + 8, 4);
                std::memcpy(&tag3, m_xgkick.packet.data() + 12, 4);
            }
            uint32_t d0 = 0u;
            uint32_t d1 = 0u;
            uint32_t r0 = 0u;
            if (m_xgkick.totalBytes >= 32u)
            {
                std::memcpy(&d0, m_xgkick.packet.data() + 16, 4);
                std::memcpy(&d1, m_xgkick.packet.data() + 20, 4);
                std::memcpy(&r0, m_xgkick.packet.data() + 24, 4);
            }
            uint32_t d2 = 0u;
            uint32_t d3 = 0u;
            uint32_t r1 = 0u;
            if (m_xgkick.totalBytes >= 48u)
            {
                std::memcpy(&d2, m_xgkick.packet.data() + 32, 4);
                std::memcpy(&d3, m_xgkick.packet.data() + 36, 4);
                std::memcpy(&r1, m_xgkick.packet.data() + 40, 4);
            }
            std::fprintf(stderr,
                         "[vu1] XGKICK bytes=%u tag=%08x.%08x.%08x.%08x src=0x%x "
                         "ad0=%08x.%08x r0=0x%x ad1=%08x.%08x r1=0x%x\n",
                         m_xgkick.totalBytes, tag0, tag1, tag2, tag3, m_xgkick.sourceAddress,
                         d0, d1, r0, d2, d3, r1);
            ++s_kickLogs;
        }
        if (m_xgkick.totalBytes >= 160u)
        {
            static uint32_t s_xyzLogs = 0u;
            uint32_t sample[3][2] = {};
            bool allCenter = true;
            for (uint32_t vi = 0u; vi < 3u; ++vi)
            {
                const uint32_t off = 48u + vi * 48u;
                uint32_t lo0 = 0u;
                uint32_t lo1 = 0u;
                if (off + 8u <= m_xgkick.totalBytes)
                {
                    std::memcpy(&lo0, m_xgkick.packet.data() + off, 4);
                    std::memcpy(&lo1, m_xgkick.packet.data() + off + 4, 4);
                }
                sample[vi][0] = lo0 & 0xFFFFu;
                sample[vi][1] = lo1 & 0xFFFFu;
                if (sample[vi][0] != 0x7FFFu || sample[vi][1] != 0x7FFFu)
                {
                    allCenter = false;
                }
            }
            if (!allCenter && s_xyzLogs < 8u)
            {
                std::fprintf(stderr,
                             "[vu1] XGKICK xyz src=0x%x bytes=%u",
                             m_xgkick.sourceAddress,
                             m_xgkick.totalBytes);
                for (uint32_t vi = 0u; vi < 3u; ++vi)
                {
                    std::fprintf(stderr,
                                 " v%u=(%u,%.2f,%u,%.2f)",
                                 vi,
                                 sample[vi][0],
                                 sample[vi][0] / 16.0f,
                                 sample[vi][1],
                                 sample[vi][1] / 16.0f);
                }
                std::fprintf(stderr, "\n");
                ++s_xyzLogs;
            }
        }
    }
    m_xgkick.active = false;
}

void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    resetXgkickState(m_xgkick);
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
    // ATV_VU_APPROX: transfer the whole packet now instead of one qword every
    // second VU cycle (PCSX2's default). Differs only if the program rewrites
    // the packet while it is still being sent; the VU no longer waits on PATH1.
    if (g_vuApprox && !g_vuForceInterp)
    {
        while (m_xgkick.active)
        {
            m_xgkick.cycleCredit = 1u;
            progressXgkick();
        }
    }
}

void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    m_state.cycles = m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    commitReadyPipelines();
    progressXgkick();
}

void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
        advanceOneCycle();
}

bool VU1Interpreter::pipelinesPending() const
{
    if (m_fdiv.valid || m_xgkick.active)
        return true;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    for (const FlagPipelineEntry &entry : m_flagPipeline)
        if (entry.valid)
            return true;
    for (const PendingStore &store : m_storePipeline)
        if (store.valid)
            return true;
    for (const PendingVfWrite &write : m_vfWritePipeline)
        if (write.valid)
            return true;
    for (const PendingViWrite &write : m_viWritePipeline)
        if (write.valid)
            return true;
    for (const PendingAccWrite &write : m_accWritePipeline)
        if (write.valid)
            return true;
    return false;
}

void VU1Interpreter::flushPipelines()
{
    while (pipelinesPending())
        advanceOneCycle();
}

uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    uint64_t ready = m_cycle;
    const InstructionUsage *usages[2] = {
        &decoded.upperUsage,
        &decoded.lowerUsage};
    for (const InstructionUsage *usage : usages)
    {
        for (uint32_t index = 0; index < usage->vfReadCount; ++index)
        {
            const VfAccess &access = usage->vfRead[index];
            uint8_t lanes = access.lanes;
            while (lanes != 0u)
            {
                const uint32_t component = static_cast<uint32_t>(std::countr_zero(lanes));
                lanes = static_cast<uint8_t>(lanes & (lanes - 1u));
                ready = std::max(ready, m_vfReady[access.reg][component]);
            }
        }
        uint32_t viRead = usage->viRead;
        while (viRead != 0u)
        {
            const uint32_t reg = static_cast<uint32_t>(std::countr_zero(viRead));
            viRead &= viRead - 1u;
            if (reg != 0u && reg < m_viReady.size())
                ready = std::max(ready, m_viReady[reg]);
        }
        uint8_t accRead = usage->accRead;
        while (accRead != 0u)
        {
            const uint32_t component = static_cast<uint32_t>(std::countr_zero(accRead));
            accRead = static_cast<uint8_t>(accRead & (accRead - 1u));
            ready = std::max(ready, m_accReady[component]);
        }
    }

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u &&
        decoded.suppressedLowerVf != lowerWrite.reg)
    {
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        uint8_t lanes = lowerWrite.lanes;
        while (lanes != 0u)
        {
            const uint32_t component = static_cast<uint32_t>(std::countr_zero(lanes));
            lanes = static_cast<uint8_t>(lanes & (lanes - 1u));
            m_vfReady[lowerWrite.reg][component] = m_cycle + latency;
        }
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                     ? decoded.upperUsage.vfLatency
                                     : decoded.upperUsage.latency;
        uint8_t lanes = upperWrite.lanes;
        while (lanes != 0u)
        {
            const uint32_t component = static_cast<uint32_t>(std::countr_zero(lanes));
            lanes = static_cast<uint8_t>(lanes & (lanes - 1u));
            m_vfReady[upperWrite.reg][component] = m_cycle + latency;
        }
    }

    uint32_t viWrite = decoded.lowerUsage.viWrite;
    const uint32_t viLatency = decoded.lowerUsage.viLatency != 0u
                                   ? decoded.lowerUsage.viLatency
                                   : decoded.lowerUsage.latency;
    while (viWrite != 0u)
    {
        const uint32_t reg = static_cast<uint32_t>(std::countr_zero(viWrite));
        viWrite &= viWrite - 1u;
        if (reg != 0u && reg < m_viReady.size())
            m_viReady[reg] = m_cycle + viLatency;
    }
    uint8_t accWrite = decoded.upperUsage.accWrite;
    while (accWrite != 0u)
    {
        const uint32_t component = static_cast<uint32_t>(std::countr_zero(accWrite));
        accWrite = static_cast<uint8_t>(accWrite & (accWrite - 1u));
        m_accReady[component] = m_cycle + kAccForwardLatency;
    }
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper) const
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        if (op == 0x08u || op == 0x09u || op == 0x0Au || op == 0x0Bu ||
            op == 0x0Cu || op == 0x0Du || op == 0x0Eu || op == 0x0Fu ||
            op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
            op == 0x29u || op == 0x2Du || op == 0x2Eu)
        {
            usage.accRead = dest;
        }
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
            usage.accWrite = dest;
            if ((special >= 0x08u && special <= 0x0Fu) ||
                special == 0x21u || special == 0x23u || special == 0x25u ||
                special == 0x27u || special == 0x29u || special == 0x2Du)
            {
                usage.accRead = dest;
            }
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special != 0x2Fu && special != 0x30u)
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(uint32_t lower) const
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x77:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7C:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x79:
            usage.latency = 29u;
            break;
        case 0x7D:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(decoded.lower);

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }
    return decoded;
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    // A pair's decode depends only on its own 8 bytes. Remember the bytes the
    // cache was built from and re-decode only the pairs an upload changed.
    struct DecodeShadow
    {
        const VU1Interpreter *owner = nullptr;
        const uint8_t *code = nullptr;
        uint32_t size = 0;
        uint64_t generation = 0;
        std::array<uint8_t, kMaxDecodedPairs * 8u> bytes{};
    };
    static DecodeShadow s_shadows[2];
    DecodeShadow &shadow = s_shadows[m_unit == Unit::VU1 ? 1 : 0];
    const bool incremental = m_decodedCodeCacheValid && shadow.owner == this && shadow.code == vuCode &&
                             shadow.size == codeSize && shadow.generation == m_cachedCodeGeneration &&
                             m_cachedVuCode == vuCode && m_cachedCodeSize == codeSize;
    const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
    for (uint32_t i = 0; i < pairCount; ++i)
    {
        uint8_t *bytes = shadow.bytes.data() + i * 8u;
        if (incremental && std::memcmp(bytes, vuCode + i * 8u, 8u) == 0)
            continue;
        std::memcpy(bytes, vuCode + i * 8u, 8u);
        m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u);
    }
    shadow.owner = this;
    shadow.code = vuCode;
    shadow.size = codeSize;
    shadow.generation = generation;
    if (m_unit == Unit::VU1)
        ++g_vu1DecodeRebuilds;

    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::getDecodedInstructionPairForPc(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
{
    if ((pc & 7u) != 0u)
        return decodeInstructionPair(vuCode, pc);

    const bool trackedVu1Code = memory != nullptr &&
                                ((m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                                 (m_unit == Unit::VU0 && vuCode == memory->getVU0Code()));
    if (!trackedVu1Code)
        return decodeInstructionPair(vuCode, pc);

    const uint64_t generation = m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration();
    if (!m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize ||
        m_cachedCodeGeneration != generation)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }
    const uint32_t pairIndex = pc / 8u;
    if (pairIndex >= kMaxDecodedPairs)
        return decodeInstructionPair(vuCode, pc);
    return m_decodedCodeCache[pairIndex];
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    m_stopRequested = true;
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    resetScheduler();
    m_cycle = 0;
    m_state.pc = startPC & microAddressMask();
    s_xgkickFaultStartPc = m_state.pc;
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;

    // Opt-in dump of unique (entry PC, code-image) pairs for VU1 static recompilation.
    static const char *s_dumpDir = std::getenv("ATV_DUMP_VU1_MPG");
    if (s_dumpDir && s_dumpDir[0] && m_unit == Unit::VU1 && vuCode && codeSize >= 8u)
    {
        uint64_t hash = 1469598103934665603ull;
        const uint32_t bytes = std::min(codeSize, 16u * 1024u);
        for (uint32_t i = 0; i < bytes; ++i)
            hash = (hash ^ vuCode[i]) * 1099511628211ull;
        hash ^= static_cast<uint64_t>(m_state.pc) << 48;
        static uint64_t s_seen[64];
        static uint32_t s_seenCount = 0u;
        static uint64_t s_runs = 0u;
        ++s_runs;
        bool seen = false;
        for (uint32_t i = 0; i < s_seenCount && !seen; ++i)
            seen = s_seen[i] == hash;
        if (!seen && s_seenCount < 64u)
        {
            s_seen[s_seenCount++] = hash;
            std::error_code ec;
            std::filesystem::create_directories(s_dumpDir, ec);
            char path[512];
            std::snprintf(path, sizeof(path), "%s/vu1_pc%04x_%016llx.bin",
                          s_dumpDir, m_state.pc, static_cast<unsigned long long>(hash));
            if (FILE *f = std::fopen(path, "wb"))
            {
                std::fwrite(vuCode, 1, bytes, f);
                std::fclose(f);
            }
            std::fprintf(stderr, "[vu1-mpg] image#%u pc=0x%04x bytes=%u hash=%016llx runs=%llu\n",
                         s_seenCount, m_state.pc, bytes,
                         static_cast<unsigned long long>(hash),
                         static_cast<unsigned long long>(s_runs));
        }
    }

    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    // Fast loop (exact against the per-pair interpreter) on by default in ATV2; ATV_VU_FAST=0 = interpreter.
    static const bool s_vuFast = []
    {
        const char *env = std::getenv("ATV_VU_FAST");
        return env == nullptr || env[0] != '0';
    }();
    static const char *s_vuFastVerifyEnv = std::getenv("ATV_VU_FAST_VERIFY");
    static const bool s_vuRecompVerify = s_vuFastVerifyEnv != nullptr;
    // ATV_VU_FAST_VERIFY=N verifies the first N in-race runs; 1 means every run.
    static const uint32_t s_verifyLimit =
        s_vuFastVerifyEnv ? static_cast<uint32_t>(std::strtoul(s_vuFastVerifyEnv, nullptr, 10)) : 0u;
    static bool s_inVerify = false;
    static uint32_t s_verifyRuns = 0u;
    const bool recompEligible = s_vuFast && !g_vuForceInterp && m_unit == Unit::VU1 && vuCode != nullptr &&
                                memory != nullptr && vuCode == memory->getVU1Code();
    extern bool g_raceLoadStarted;
    if (recompEligible && s_vuRecompVerify && !s_inVerify && vuData != nullptr && g_raceLoadStarted &&
        (s_verifyLimit <= 1u || s_verifyRuns < s_verifyLimit))
    {
        ++s_verifyRuns;
        s_inVerify = true;
        const auto before = std::make_unique<VU1Interpreter>(*this);
        const std::vector<uint8_t> dataBefore(vuData, vuData + dataSize);
        std::vector<VuVerifyRecord> traceRecomp;
        std::vector<VuVerifyRecord> traceInterp;
        std::vector<std::vector<uint8_t>> kicksRecomp;
        std::vector<std::vector<uint8_t>> kicksInterp;

        // ATV_VU_FAST_VERIFY_END=1: no per-pair traces, so the fast loop keeps
        // its production shortcuts (batched PC, bare cycles); only final state,
        // cycle, data memory and XGKICK packets are compared.
        static const bool s_endOnly = std::getenv("ATV_VU_FAST_VERIFY_END") != nullptr;
        g_vuVerifyTrace = s_endOnly ? nullptr : &traceRecomp;
        g_vuKickCapture = &kicksRecomp;
        run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
        const VU1State recompState = m_state;
        const uint64_t recompCycle = m_cycle;
        const std::vector<uint8_t> dataRecomp(vuData, vuData + dataSize);

        *this = *before;
        std::memcpy(vuData, dataBefore.data(), dataSize);
        resetCommitHint(commitHintFor(this), 0xFFFFFFFFu);
        g_vuForceInterp = true;
        g_vuVerifyTrace = s_endOnly ? nullptr : &traceInterp;
        g_vuKickCapture = &kicksInterp;
        run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
        g_vuForceInterp = false;
        g_vuVerifyTrace = nullptr;
        g_vuKickCapture = nullptr;

        reportRecompVerify(s_verifyRuns, traceRecomp, traceInterp, recompState, m_state, recompCycle, m_cycle,
                           dataRecomp, vuData, dataSize, kicksRecomp, kicksInterp);
        for (const std::vector<uint8_t> &packet : kicksInterp)
        {
            if (m_activeMemory)
                m_activeMemory->submitGifPacket(GifPathId::Path1, packet.data(), static_cast<uint32_t>(packet.size()));
            else if (m_activeGs)
                m_activeGs->processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
        }
        s_inVerify = false;
        return;
    }
#include "ps2_vu1_bench.inc"

    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    const int previousRoundingMode = std::fegetround();
    const bool useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    // ATV_VU_APPROX: flush denormal operands and results to zero in hardware
    // (FTZ|DAZ), as the approximate FMAC path relies on.
    const bool approxCsr = g_vuApprox && !g_vuForceInterp;
    const unsigned int savedCsr = _mm_getcsr();
    if (approxCsr)
        _mm_setcsr(savedCsr | 0x8040u);
    const uint64_t budgetEnd = m_cycle + maxCycles;
    const uint32_t censusStartPc = m_state.pc;
    bool programEnded = false;
    // pc=0xc8 is the shared entry point of a generic transform/light VU1
    // microprogram reused by many object types (menu 3D logo, vehicle, terrain,
    // ...) -- the original one-shot trace armed on the very FIRST ever hit,
    // which (confirmed via cross-referencing the resulting log against the race
    // start marker) fires long before terrain even loads, during early menu
    // rendering. Gate on g_raceLoadStarted so this instead captures a genuine
    // in-race (and very likely terrain, given how much of the frame terrain
    // dominates) invocation.
    extern bool g_raceLoadStarted;
    static bool s_tracedGeometry = false;
    static const bool s_traceVuGeometry = std::getenv("ATV_TRACE_VU_GEOMETRY") != nullptr;
    const bool traceGeometry = !s_tracedGeometry && m_state.pc == 0xc8u && g_raceLoadStarted &&
                               s_traceVuGeometry;
    if (traceGeometry)
        s_tracedGeometry = true;
    uint32_t tracePairs = 0u;
    bool recompRan = false;
    uint64_t fastDrainCycle = 0;
    if (recompEligible)
#include "ps2_vu1_fast_loop.inc"
    while (!recompRan && m_cycle < budgetEnd && !m_stopRequested)
    {
        commitReadyPipelines();
        if (m_state.pc + 8u > codeSize)
            break;

        const DecodedInstructionPair decoded = getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
        if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
        {
            reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
            break;
        }

        // Dependency ready times only increase with the cycle counter; a
        // single advance is enough (recomputing after advance was a no-op).
        const uint64_t readyCycle = calculatePairReadyCycle(decoded);
        if (readyCycle > m_cycle)
        {
            if (readyCycle >= budgetEnd)
            {
                advanceTo(budgetEnd);
                break;
            }
            advanceTo(readyCycle);
        }
        // XGKICK waits for the previous PATH1 transfer, whose end is not a fixed
        // ready cycle; starting early would discard the packet still in flight.
        if (decoded.lowerUsage.pipeline == PipelineXgkick)
        {
            while (m_xgkick.active && m_cycle < budgetEnd)
                advanceOneCycle();
            if (m_xgkick.active)
                break;
        }
        if (g_vuVerifyTrace)
            recordVerifyPair(m_state, m_cycle, decoded.upperUsage, decoded.lowerUsage);

        if (traceGeometry && tracePairs++ < 2048u)
        {
            const float *fs = m_state.vf[FS(decoded.upper)];
            const float *ft = m_state.vf[FT(decoded.upper)];
            std::fprintf(stderr,
                         "[vu-geom] pc=%04x u=%08x l=%08x fs=%g,%g,%g,%g ft=%g,%g,%g,%g acc=%g,%g,%g,%g q=%g "
                         "vi0=%d vi1=%d vi2=%d vi3=%d\n",
                         m_state.pc, decoded.upper, decoded.lower,
                         fs[0], fs[1], fs[2], fs[3], ft[0], ft[1], ft[2], ft[3],
                         m_state.acc[0], m_state.acc[1], m_state.acc[2], m_state.acc[3], m_state.q,
                         m_state.vi[0], m_state.vi[1], m_state.vi[2], m_state.vi[3]);
        }

        uint8_t writtenVi = 0u;
        int32_t oldVi = 0;
        for (uint32_t reg = 1; decoded.lowerUsage.viWrite != 0u && reg < 16u; ++reg)
        {
            if ((decoded.lowerUsage.viWrite & (1u << reg)) != 0u)
            {
                writtenVi = static_cast<uint8_t>(reg);
                oldVi = m_state.vi[reg];
                break;
            }
        }

        const VfAccess upperWrite = decoded.upperUsage.vfWrite;
        const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
        const bool hasUpperWrite = upperWrite.reg != 0u;
        const bool hasLowerWrite = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
        const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
        float oldUpperVf[4]{};
        float newUpperVf[4]{};
        float oldLowerVf[4]{};
        float newLowerVf[4]{};
        float oldAcc[4]{};
        float newAcc[4]{};
        if (hasUpperWrite)
            std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
        if (hasDistinctLowerWrite)
            std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
        if (decoded.upperUsage.accWrite != 0u)
            std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));

        if (decoded.iBit)
        {
            execUpper(decoded.upper);
            float immediate = 0.0f;
            std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
            m_state.i = normalizeOperand(immediate);
        }
        else if (decoded.upperVfShadowReg != 0u)
        {
            float oldVf[4]{};
            float upperVf[4]{};
            std::memcpy(oldVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(oldVf));
            execUpper(decoded.upper);
            std::memcpy(upperVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(upperVf));
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        oldVf,
                        sizeof(oldVf));
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        upperVf,
                        sizeof(upperVf));
        }
        else
        {
            execUpper(decoded.upper);
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        }

        m_viBranchBackupValid = false;

        if (hasUpperWrite)
        {
            std::memcpy(newUpperVf, m_state.vf[upperWrite.reg], sizeof(newUpperVf));
            std::memcpy(m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
            const uint32_t latency =
                decoded.upperUsage.vfLatency != 0u
                    ? decoded.upperUsage.vfLatency
                    : decoded.upperUsage.latency;
            queueVfWrite(upperWrite.reg, upperWrite.lanes, newUpperVf, latency);
        }
        if (hasDistinctLowerWrite)
        {
            std::memcpy(newLowerVf, m_state.vf[lowerWrite.reg], sizeof(newLowerVf));
            std::memcpy(m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
            const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                         ? decoded.lowerUsage.vfLatency
                                         : decoded.lowerUsage.latency;
            queueVfWrite(lowerWrite.reg, lowerWrite.lanes, newLowerVf, latency);
        }
        if (decoded.upperUsage.accWrite != 0u)
        {
            std::memcpy(newAcc, m_state.acc, sizeof(newAcc));
            std::memcpy(m_state.acc, oldAcc, sizeof(oldAcc));
            // ACC is forwarded to the next upper instruction. Its arithmetic
            // flags still use the normal four-cycle FMAC timeline.
            queueAccWrite(decoded.upperUsage.accWrite, newAcc,
                          kAccForwardLatency);
        }
        if (writtenVi != 0u)
        {
            const int32_t newVi = m_state.vi[writtenVi];
            m_state.vi[writtenVi] = oldVi;
            const uint32_t latency =
                decoded.lowerUsage.viLatency != 0u
                    ? decoded.lowerUsage.viLatency
                    : decoded.lowerUsage.latency;
            queueViWrite(writtenVi, newVi, latency);
        }

        markPairWrites(decoded);
        if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
            recordViWriteForBranch(writtenVi, oldVi);

        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        m_state.vi[0] = 0;

        uint32_t nextPc = m_state.pc + 8u;
        if (nextPc >= codeSize)
            nextPc = 0u;
        m_state.pc = nextPc;

        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0u)
            {
                m_state.pc = m_state.branchTarget & microAddressMask();
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        const bool dHalt = decoded.dBit && m_state.dBitEnabled;
        const bool tHalt = decoded.tBit && m_state.tBitEnabled;
        const bool haltBit = dHalt || tHalt;
        const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

        if (m_state.haltAfterDelaySlot)
        {
            m_state.stoppedByD = m_pendingHaltD;
            m_state.stoppedByT = m_pendingHaltT;
            programEnded = true;
        }
        else if (m_state.ebit)
            programEnded = true;
        else if (haltBit && !haltBranch)
        {
            m_state.stoppedByD = dHalt;
            m_state.stoppedByT = tHalt;
            programEnded = true;
        }
        else if (decoded.eBit)
            m_state.ebit = true;
        else if (haltBranch)
        {
            m_state.haltAfterDelaySlot = true;
            m_pendingHaltD = dHalt;
            m_pendingHaltT = tHalt;
        }

        advanceOneCycle();
        if (programEnded)
            break;
    }

    if (!programEnded && m_cycle >= budgetEnd)
    {
        static uint32_t s_budgetLogs = 0u;
        if (s_budgetLogs < 8u)
        {
            std::fprintf(stderr,
                         "[vu1] budget pc=0x%x cyc=%llu max=%u\n",
                         m_state.pc,
                         static_cast<unsigned long long>(m_cycle),
                         maxCycles);
            ++s_budgetLogs;
        }
    }

    // The fast loop commits VF/VI/ACC at issue (and may skip flag entries);
    // the interpreter would still be draining those here.
    advanceTo(fastDrainCycle);
    // The fast loop requires empty pipelines on entry, so with ATV_VU_FAST every
    // run drains. Otherwise writes stay queued across a budget-limited run.
    if (programEnded || s_vuFast)
        flushPipelines();
    if (m_unit == Unit::VU1)
        g_vu1CycleTotal += m_cycle - (budgetEnd - maxCycles);

    // ATV_VU_CENSUS=1: VU1 cycles per (code image, entry PC) during the race.
    static const bool s_vuCensus = std::getenv("ATV_VU_CENSUS") != nullptr;
    if (s_vuCensus && m_unit == Unit::VU1 && vuCode && g_raceLoadStarted && !s_inVerify)
    {
        struct CensusEntry
        {
            uint64_t image;
            uint32_t pc;
            uint64_t runs;
            uint64_t cycles;
        };
        static std::vector<CensusEntry> s_census;
        static uint64_t s_censusGen = ~0ull;
        static uint64_t s_censusImage = 0;
        static uint64_t s_censusRuns = 0;
        const uint64_t gen = memory ? memory->getVU1CodeGeneration() : 0ull;
        if (gen != s_censusGen || !memory)
        {
            s_censusGen = gen;
            uint64_t h = 1469598103934665603ull;
            for (uint32_t i = 0; i < std::min(codeSize, 16u * 1024u); ++i)
                h = (h ^ vuCode[i]) * 1099511628211ull;
            s_censusImage = h;
        }
        const uint64_t spent = m_cycle - (budgetEnd - maxCycles);
        CensusEntry *entry = nullptr;
        for (CensusEntry &e : s_census)
            if (e.image == s_censusImage && e.pc == censusStartPc)
            {
                entry = &e;
                break;
            }
        if (!entry && s_census.size() < 256u)
            entry = &s_census.emplace_back(CensusEntry{s_censusImage, censusStartPc, 0u, 0u});
        if (entry)
        {
            ++entry->runs;
            entry->cycles += spent;
        }
        if ((++s_censusRuns % 200000u) == 0u)
        {
            uint64_t total = 0;
            for (const CensusEntry &e : s_census)
                total += e.cycles;
            std::vector<CensusEntry> sorted = s_census;
            std::sort(sorted.begin(), sorted.end(),
                      [](const CensusEntry &a, const CensusEntry &b) { return a.cycles > b.cycles; });
            std::fprintf(stderr, "[vu-census] runs=%llu cycles=%llu images=%zu\n",
                         static_cast<unsigned long long>(s_censusRuns), static_cast<unsigned long long>(total),
                         sorted.size());
            for (size_t i = 0; i < sorted.size() && i < 16u; ++i)
                std::fprintf(stderr, "[vu-census]   %5.1f%% image=%016llx pc=%04x runs=%llu cyc/run=%llu\n",
                             total ? 100.0 * static_cast<double>(sorted[i].cycles) / static_cast<double>(total) : 0.0,
                             static_cast<unsigned long long>(sorted[i].image), sorted[i].pc,
                             static_cast<unsigned long long>(sorted[i].runs),
                             static_cast<unsigned long long>(sorted[i].runs ? sorted[i].cycles / sorted[i].runs : 0u));
        }
    }

    if (programEnded)
    {
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    m_state.cycles = m_cycle;
    if (approxCsr)
        _mm_setcsr(savedCsr);
    if (useVuRounding && previousRoundingMode != -1)
        std::fesetround(previousRoundingMode);
}

#include "ps2_vu1_native_blocks.inc"
