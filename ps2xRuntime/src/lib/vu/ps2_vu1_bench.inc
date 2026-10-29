// Repeatable VU1 workload. Included in VU1Interpreter::run after the verify block.
//
// ATV_VU_BENCH_CAPTURE=file  Once the race has started, skip ATV_VU_BENCH_SKIP eligible
//                            runs (default 2000), then save the input of every
//                            ATV_VU_BENCH_STRIDE-th run (default 16) until
//                            ATV_VU_BENCH_COUNT runs are saved (default 2000).
// ATV_VU_BENCH_REPLAY=file   At the first eligible run, replay the saved runs once
//                            through the interpreter and ATV_VU_BENCH_REPS times
//                            (default 7) through the fast loop, report timings and
//                            result checksums, then exit the process.
//
// XGKICK packets are captured, not sent to the GS, so GIF parsing is not timed.
// The file holds raw interpreter bytes: it only loads into builds with the same
// VU1Interpreter layout (checked against the header).
{
    static const char *const s_benchCapture = std::getenv("ATV_VU_BENCH_CAPTURE");
    static const char *const s_benchReplay = std::getenv("ATV_VU_BENCH_REPLAY");
    static bool s_inBench = false;
    if ((s_benchCapture || s_benchReplay) && recompEligible && !s_inBench && !s_inVerify && vuData != nullptr)
    {
        struct BenchHeader
        {
            char magic[8];
            uint32_t objectSize, skipBegin, skipEnd, packetBegin, codeSize, dataSize, count;
        };
        const auto envU32 = [](const char *name, uint32_t fallback) {
            const char *value = std::getenv(name);
            return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : fallback;
        };
        const auto offsetOf = [this](const void *member) {
            return static_cast<uint32_t>(static_cast<const uint8_t *>(member) - reinterpret_cast<const uint8_t *>(this));
        };
        // Skipped on save and load: the decode cache and its keys stay live in the replaying process.
        const uint32_t skipBegin = offsetOf(&m_decodedCodeCache);
        const uint32_t skipEnd = offsetOf(&m_flagPipeline);
        const uint32_t packetBegin = offsetOf(m_xgkick.packet.data());
        const uint32_t packetEnd = packetBegin + static_cast<uint32_t>(m_xgkick.packet.size());
        const uint32_t objectSize = static_cast<uint32_t>(sizeof(*this));
        const BenchHeader expected{{'A', 'T', 'V', 'V', 'U', 'B', '1', 0},
                                   objectSize, skipBegin, skipEnd, packetBegin, codeSize, dataSize, 0u};

        if (s_benchCapture)
        {
            extern bool g_atvRaceStarted;
            static FILE *s_file = nullptr;
            static bool s_done = false;
            static uint32_t s_seen = 0u;
            static uint32_t s_saved = 0u;
            static std::vector<uint8_t> s_lastCode;
            static const uint32_t s_skip = envU32("ATV_VU_BENCH_SKIP", 2000u);
            static const uint32_t s_stride = std::max(1u, envU32("ATV_VU_BENCH_STRIDE", 16u));
            static const uint32_t s_count = std::max(1u, envU32("ATV_VU_BENCH_COUNT", 2000u));
            if (!s_done && g_atvRaceStarted && s_seen++ >= s_skip && (s_seen - 1u - s_skip) % s_stride == 0u)
            {
                if (!s_file)
                {
                    s_file = std::fopen(s_benchCapture, "wb");
                    if (!s_file)
                    {
                        std::fprintf(stderr, "[vu-bench] cannot open %s\n", s_benchCapture);
                        s_done = true;
                    }
                    else
                    {
                        std::fwrite(&expected, sizeof(expected), 1, s_file);
                        std::fprintf(stderr, "[vu-bench] capture start recompRuns=%llu\n",
                                     static_cast<unsigned long long>(g_vuRecompRuns));
                    }
                }
                if (s_file)
                {
                    const uint8_t newCode = s_lastCode.size() != codeSize ||
                                            std::memcmp(s_lastCode.data(), vuCode, codeSize) != 0;
                    if (newCode)
                        s_lastCode.assign(vuCode, vuCode + codeSize);
                    const uint8_t *self = reinterpret_cast<const uint8_t *>(this);
                    const uint32_t packetBytes = m_xgkick.active ? std::min<uint32_t>(m_xgkick.totalBytes, packetEnd - packetBegin) : 0u;
                    std::fwrite(&maxCycles, 4, 1, s_file);
                    std::fwrite(&newCode, 1, 1, s_file);
                    if (newCode)
                        std::fwrite(vuCode, 1, codeSize, s_file);
                    std::fwrite(vuData, 1, dataSize, s_file);
                    std::fwrite(self, 1, skipBegin, s_file);
                    std::fwrite(self + skipEnd, 1, packetBegin - skipEnd, s_file);
                    std::fwrite(&packetBytes, 4, 1, s_file);
                    std::fwrite(self + packetBegin, 1, packetBytes, s_file);
                    std::fwrite(self + packetEnd, 1, objectSize - packetEnd, s_file);
                    if (++s_saved == s_count)
                    {
                        std::fseek(s_file, static_cast<long>(offsetof(BenchHeader, count)), SEEK_SET);
                        std::fwrite(&s_saved, 4, 1, s_file);
                        std::fclose(s_file);
                        s_file = nullptr;
                        s_done = true;
                        std::fprintf(stderr, "[vu-bench] captured %u runs to %s recompRuns=%llu\n", s_saved,
                                     s_benchCapture, static_cast<unsigned long long>(g_vuRecompRuns));
                    }
                }
            }
        }
        else
        {
            s_inBench = true;
            const auto fail = [](const char *why) {
                std::fprintf(stderr, "[vu-bench] replay failed: %s\n", why);
                std::fflush(stderr);
                std::_Exit(3);
            };
            std::vector<uint8_t> file;
            if (FILE *f = std::fopen(s_benchReplay, "rb"))
            {
                std::fseek(f, 0, SEEK_END);
                file.resize(static_cast<size_t>(std::ftell(f)));
                std::fseek(f, 0, SEEK_SET);
                if (std::fread(file.data(), 1, file.size(), f) != file.size())
                    file.clear();
                std::fclose(f);
            }
            BenchHeader header{};
            if (file.size() < sizeof(header))
                fail("cannot read file");
            std::memcpy(&header, file.data(), sizeof(header));
            if (std::memcmp(header.magic, expected.magic, sizeof(header.magic)) != 0 ||
                header.objectSize != objectSize || header.skipBegin != skipBegin || header.skipEnd != skipEnd ||
                header.packetBegin != packetBegin || header.codeSize != codeSize || header.dataSize != dataSize)
                fail("header does not match this build's VU1Interpreter layout");
            if (header.count == 0u)
                fail("capture incomplete (count=0)");

            struct Sample
            {
                uint32_t maxCycles = 0;
                uint32_t code = 0;
                size_t data = 0;
                size_t object = 0;
            };
            std::vector<Sample> samples;
            std::vector<size_t> codes;
            size_t pos = sizeof(header);
            const auto need = [&](size_t bytes) {
                if (pos + bytes > file.size())
                    fail("truncated file");
            };
            for (uint32_t n = 0; n < header.count; ++n)
            {
                Sample s;
                need(5);
                std::memcpy(&s.maxCycles, file.data() + pos, 4);
                const uint8_t newCode = file[pos + 4];
                pos += 5;
                if (newCode)
                {
                    need(codeSize);
                    codes.push_back(pos);
                    pos += codeSize;
                }
                if (codes.empty())
                    fail("first record has no code image");
                s.code = static_cast<uint32_t>(codes.size() - 1u);
                need(dataSize);
                s.data = pos;
                pos += dataSize;
                s.object = pos;
                need(skipBegin + (packetBegin - skipEnd) + 4u);
                pos += skipBegin + (packetBegin - skipEnd);
                uint32_t packetBytes = 0;
                std::memcpy(&packetBytes, file.data() + pos, 4);
                pos += 4;
                need(static_cast<size_t>(packetBytes) + (objectSize - packetEnd));
                pos += packetBytes + (objectSize - packetEnd);
                samples.push_back(s);
            }

            const auto apply = [&](const Sample &s) {
                const uint8_t *code = file.data() + codes[s.code];
                if (std::memcmp(vuCode, code, codeSize) != 0)
                {
                    std::memcpy(vuCode, code, codeSize);
                    memory->markVU1CodeModified();
                }
                std::memcpy(vuData, file.data() + s.data, dataSize);
                uint8_t *self = reinterpret_cast<uint8_t *>(this);
                const uint8_t *p = file.data() + s.object;
                std::memcpy(self, p, skipBegin);
                p += skipBegin;
                std::memcpy(self + skipEnd, p, packetBegin - skipEnd);
                p += packetBegin - skipEnd;
                uint32_t packetBytes = 0;
                std::memcpy(&packetBytes, p, 4);
                p += 4;
                std::memcpy(self + packetBegin, p, packetBytes);
                p += packetBytes;
                std::memcpy(self + packetEnd, p, objectSize - packetEnd);
                resetCommitHint(commitHintFor(this), 0xFFFFFFFFu);
            };
            std::vector<std::vector<uint8_t>> kicks;
            uint64_t lastCycles = 0;
            const auto runSample = [&](const Sample &s, bool interp) {
                apply(s);
                kicks.clear();
                g_vuKickCapture = &kicks;
                g_vuForceInterp = interp;
                const auto t0 = std::chrono::steady_clock::now();
                const uint64_t c0 = threadCycles();
                run(vuCode, codeSize, vuData, dataSize, gs, memory, s.maxCycles);
                lastCycles = threadCycles() - c0;
                const auto t1 = std::chrono::steady_clock::now();
                g_vuForceInterp = false;
                g_vuKickCapture = nullptr;
                return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            };
            const auto resultHash = [&]() {
                uint64_t h = 1469598103934665603ull;
                const auto mix = [&h](const void *bytes, size_t size) {
                    const uint8_t *b = static_cast<const uint8_t *>(bytes);
                    for (size_t k = 0; k < size; ++k)
                        h = (h ^ b[k]) * 1099511628211ull;
                };
                mix(m_state.vf, sizeof(m_state.vf));
                mix(m_state.vi, sizeof(m_state.vi));
                mix(m_state.acc, sizeof(m_state.acc));
                mix(&m_state.q, sizeof(float) * 3u);
                mix(&m_state.clip, sizeof(m_state.clip));
                mix(&m_state.pc, sizeof(m_state.pc));
                mix(&m_state.ebit, sizeof(m_state.ebit));
                // ATV_VU_BENCH_NO_CYCLE=1: compare state only (approximate timing).
                static const bool s_noCycle = std::getenv("ATV_VU_BENCH_NO_CYCLE") != nullptr;
                if (!s_noCycle)
                    mix(&m_cycle, sizeof(m_cycle));
                mix(vuData, dataSize);
                for (const std::vector<uint8_t> &k : kicks)
                {
                    const size_t size = k.size();
                    mix(&size, sizeof(size));
                    mix(k.data(), k.size());
                }
                return h;
            };

            const size_t count = samples.size();
            std::vector<uint64_t> fastHash(count);
            std::vector<uint64_t> fastBest(count, ~0ull);
            std::vector<uint64_t> fastBestCycles(count, ~0ull);
            uint64_t interpNs = 0;
            uint64_t interpCycles = 0;
            uint32_t mismatches = 0;
            uint64_t passPairs = 0;
            const uint64_t nativeBefore = g_vu1NativePairs;
            for (size_t n = 0; n < count; ++n)
            {
                const uint64_t pairsBefore = g_vu1PairTotal;
                runSample(samples[n], false);
                passPairs += g_vu1PairTotal - pairsBefore;
                fastHash[n] = resultHash();
                const VU1State fastState = m_state;
                const uint64_t fastCycle = m_cycle;
                const std::vector<uint8_t> fastData(vuData, vuData + dataSize);
                const std::vector<std::vector<uint8_t>> fastKicks = kicks;
                interpNs += runSample(samples[n], true);
                interpCycles += lastCycles;
                if (resultHash() != fastHash[n])
                {
                    ++mismatches;
                    reportRecompVerify(static_cast<uint32_t>(n), {}, {}, fastState, m_state, fastCycle, m_cycle,
                                       fastData, vuData, dataSize, fastKicks, kicks);
                }
            }
            std::fprintf(stderr, "[vu-bench] native blocks ran %llu pairs/pass\n",
                         static_cast<unsigned long long>(g_vu1NativePairs - nativeBefore));
            std::fprintf(stderr, "[vu-bench] samples=%zu codeImages=%zu pairs/pass=%llu interpMismatch=%u\n", count,
                         codes.size(), static_cast<unsigned long long>(passPairs), mismatches);
            if (g_vuFmacStats)
                std::fprintf(stderr,
                             "[vu-bench] fmac (one fast pass) ops=%llu slowOps=%llu slowNoFlags=%llu lanes: zero=%llu "
                             "(zeroVs=%llu) denormal=%llu infNan=%llu edge=%llu productBound=%llu\n",
                             static_cast<unsigned long long>(g_vuFmacStat[kFmacStatOps]),
                             static_cast<unsigned long long>(g_vuFmacStat[kFmacStatSlowOps]),
                             static_cast<unsigned long long>(g_vuFmacStat[kFmacStatSlowNoFlag]),
                             static_cast<unsigned long long>(g_vuFmacStat[kFmacStatZero]),
                             static_cast<unsigned long long>(g_vuFmacStat[kFmacStatZeroFactor]),
                             static_cast<unsigned long long>(g_vuFmacStat[kFmacStatDenormal]),
                             static_cast<unsigned long long>(g_vuFmacStat[kFmacStatInfNan]),
                             static_cast<unsigned long long>(g_vuFmacStat[kFmacStatEdge]),
                             static_cast<unsigned long long>(g_vuFmacStat[kFmacStatProduct]));

            const uint32_t reps = std::max(1u, envU32("ATV_VU_BENCH_REPS", 7u));
            // ATV_HOST_PROFILE samples only the timed fast reps (reports every 20 s).
            void atvStartHostProfiler();
            atvStartHostProfiler();
            std::vector<uint64_t> repNs;
            std::vector<uint64_t> repCycles;
            uint32_t unstable = 0;
            const uint64_t steppedBefore = g_vu1RunsStepped, genericRunsBefore = g_vu1RunsGeneric,
                           genericBefore = g_vu1GenericPairs, rebuildsBefore = g_vu1DecodeRebuilds;
            const double pairs = passPairs ? static_cast<double>(passPairs) : 1.0;
            for (uint32_t r = 0; r < reps; ++r)
            {
                uint64_t total = 0;
                uint64_t cycles = 0;
                for (size_t n = 0; n < count; ++n)
                {
                    const uint64_t ns = runSample(samples[n], false);
                    total += ns;
                    cycles += lastCycles;
                    fastBest[n] = std::min(fastBest[n], ns);
                    fastBestCycles[n] = std::min(fastBestCycles[n], lastCycles);
                    if (resultHash() != fastHash[n])
                        ++unstable;
                }
                repNs.push_back(total);
                repCycles.push_back(cycles);
                std::fprintf(stderr, "[vu-bench] rep %u fast ms=%.3f ns/pair=%.2f cycles/pair=%.2f\n", r + 1u,
                             total / 1e6, total / pairs, cycles / pairs);
            }
            std::fprintf(stderr,
                         "[vu-bench] per fast pass: runs=%zu steppedRuns=%llu genericRuns=%llu genericPairs=%llu "
                         "codeRebuilds=%llu\n",
                         count, static_cast<unsigned long long>((g_vu1RunsStepped - steppedBefore) / reps),
                         static_cast<unsigned long long>((g_vu1RunsGeneric - genericRunsBefore) / reps),
                         static_cast<unsigned long long>((g_vu1GenericPairs - genericBefore) / reps),
                         static_cast<unsigned long long>((g_vu1DecodeRebuilds - rebuildsBefore) / reps));
            uint64_t bestSum = 0;
            uint64_t bestCycleSum = 0;
            uint64_t combined = 1469598103934665603ull;
            for (size_t n = 0; n < count; ++n)
            {
                bestSum += fastBest[n];
                bestCycleSum += fastBestCycles[n];
                combined = (combined ^ fastHash[n]) * 1099511628211ull;
            }
            std::vector<uint64_t> sorted = repNs;
            std::sort(sorted.begin(), sorted.end());
            std::sort(repCycles.begin(), repCycles.end());
            std::fprintf(stderr,
                         "[vu-bench] result fastMedianMs=%.3f fastMinRepMs=%.3f fastSumOfMinMs=%.3f "
                         "fastMedianNsPerPair=%.2f fastSumOfMinNsPerPair=%.2f "
                         "fastMedianCyclesPerPair=%.2f fastMinRepCyclesPerPair=%.2f fastSumOfMinCyclesPerPair=%.2f "
                         "interpMs=%.3f interpNsPerPair=%.2f interpCyclesPerPair=%.2f "
                         "resultHash=%016llx interpMismatch=%u unstable=%u\n",
                         sorted[sorted.size() / 2] / 1e6, sorted.front() / 1e6, bestSum / 1e6,
                         sorted[sorted.size() / 2] / pairs, bestSum / pairs,
                         repCycles[repCycles.size() / 2] / pairs, repCycles.front() / pairs, bestCycleSum / pairs,
                         interpNs / 1e6, interpNs / pairs, interpCycles / pairs,
                         static_cast<unsigned long long>(combined), mismatches, unstable);
            std::fprintf(stderr, "[vu-bench] done\n");
            std::fflush(stderr);
            std::_Exit(0);
        }
    }
}
