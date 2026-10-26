// Fast VU1 execution loop, included inside VU1Interpreter::run.
//
// Mirrors the interpreter loop (same pipelines for Q/P/clip/stores/XGKICK,
// same stall model and branch/E-bit handling) with two shortcuts:
//  - common upper FMAC ops are pre-classified per code image and executed
//    without the execUpper dispatch (MAC/status flags skipped unless
//    ATV_VU_FAST_FLAGS is set);
//  - VF/VI/ACC results commit at issue. Every later read of those registers
//    is interlocked through m_vfReady/m_viReady/m_accReady, so committing at
//    issue is indistinguishable from committing at readyCycle.
// Block replay (ATV_VU_NO_REPLAY=1 disables): a straight-line block's stall
// schedule depends only on how far in the future its live-in registers (and
// FDIV/EFU, when it waits on them) become ready at entry. The first run from
// a given entry state records each pair's issue cycle; later runs from the
// same state issue on those cycles without the per-pair ready checks and
// ready-table writes, then write the block's final ready times once. Cycles
// still advance one at a time, so Q/P/flag commits and PATH1 progress are
// unchanged. XGKICK waits depend on packet sizes and are re-checked; any
// mismatch rebuilds the ready tables and continues in the per-pair loop.
// A completed replay fixes most of the next block's signature, so schedules
// link to their successors and only re-check the entries left open.
// ATV_VU_FAST_VERIFY cross-checks this loop against the interpreter.
{
    // Op classification and the fast halves live in VU1Native
    // (ps2_vu1_native.inc) so natively compiled blocks can share them.
    using FastOp = VU1Native::FastOp;
    using FastReads = VU1Native::FastReads;
    using FastWrites = VU1Native::FastWrites;
    using enum VU1Native::LowerKind;
    using enum VU1Native::UpperKind;
    static constexpr auto classifyLowerWord = [](uint32_t lower) { return VU1Native::classifyLowerWord(lower); };
    auto classifyLower = [](const DecodedInstructionPair &pair) { return VU1Native::classifyLower(pair); };
    // Ready-table slots used by block replay: VF lane reg*4+bit (the
    // m_vfReady[reg][bit] indexing), then VI, then ACC lanes.
    enum : uint32_t
    {
        kSlotVi = 128,
        kSlotAcc = 144,
        kSlotCount = 148,
        kMaxBlockPairs = 128,
        kMaxBlockSchedules = 8,
        kMaxSignature = kSlotCount + 3,
        kMaxRecordedCycle = 60000,
        kMaxBlockLinks = 4,
        kMaxLinkChecks = 8,
        // Link checks past the ready-table slots.
        kCheckFdiv = kSlotCount,
        kCheckEfu,
        kCheckP,
        // Longest queueQ / queueP latencies.
        kMaxFdivLatency = 13,
        kMaxEfuLatency = 54,
    };
    // How replay executes a pair: through execPair, or with the fast halves
    // alone (both write exactly their declared lanes), optionally followed by
    // the branch VI-delay record. kPairIBitFast loads I from the lower word
    // then runs the fast upper only (lower is an immediate, not an opcode).
    enum : uint8_t
    {
        kPairGeneric = 0,
        kPairFast,
        kPairFastViRecord,
        kPairIBitFast,
    };
    struct BlockInfo;
    // A block replayed on a schedule whose exit state fixes the next block's
    // signature, except for the few entries in `check`: the next block starts
    // without computing the signature when those still hold.
    struct BlockLink
    {
        uint32_t pc = ~0u;
        uint32_t index = 0;
        BlockInfo *block = nullptr;
        uint64_t serial = 0;
        uint64_t epoch = 0;
        uint32_t checkCount = 0;
        uint8_t check[kMaxLinkChecks]{}; // ready-table slot or kCheck*
        uint8_t expect[kMaxLinkChecks]{}; // its signature value
    };
    struct BlockSchedule
    {
        std::vector<uint8_t> signature;
        std::vector<uint16_t> issue;     // issue cycle of each pair, relative to block entry
        std::vector<uint16_t> dataReady; // XGKICK pairs: operand-ready cycle, relative to entry
        // Final ready times, (index << 16) | cycle relative to entry; index is
        // the VF slot, VI register or ACC lane.
        std::vector<uint32_t> exitVf, exitVi, exitAcc;
        uint32_t total = 0;
        uint32_t writesReadyOffset = 0;
        uint64_t serial = 0;
        BlockLink links[kMaxBlockLinks];
        uint32_t nextLink = 0;
    };
    struct BlockInfo
    {
        uint32_t first = 0;
        uint32_t length = 0;
        bool replayable = false;
        bool needFdiv = false;
        bool issuesFdiv = false;
        bool needEfu = false; // also: the block issues EFU ops
        bool needP = false;
        // Signature order: VF slots, VI registers, ACC lanes, then FDIV/EFU/P.
        std::vector<uint8_t> liveVf, liveVi, liveAcc;
        std::vector<uint8_t> simple; // kPair* per pair
        std::vector<uint8_t> xgkick;
        std::vector<BlockSchedule> schedules;
        uint32_t nextVictim = 0;
        uint32_t records = 0;
        uint64_t hits = 0;
        uint32_t censusIndex = 0;
        // No generic pairs and no XGKICK: replay can use the translated path.
        bool translated = false;
        // Translated and never observes pipelined Q/P/clip mid-block, so a dense
        // run can bump cycles without commitReadyPipelines when nothing is due.
        bool cycleQuiet = false;
        // Code words match a natively compiled block (kNative*); `native`
        // selects it for translated replay unless ATV_VU_NO_NATIVE is set.
        uint8_t nativeCandidate = 0;
        uint8_t native = 0;
    };
    struct FastOpTable
    {
        const uint8_t *code = nullptr;
        uint64_t generation = ~0ull;
        uint32_t codeSize = 0;
        uint8_t readsFlags = 0; // kReads* of any pair
        std::array<FastOp, kMaxDecodedPairs> ops{};
        std::array<FastReads, kMaxDecodedPairs> reads{};
        std::array<FastWrites, kMaxDecodedPairs> writes{};
        // Code bytes the tables were built from; uploads only reclassify the
        // pairs (and drop the blocks) whose bytes changed.
        std::array<uint8_t, kMaxDecodedPairs * 8u> shadow{};
        std::array<uint8_t, kMaxDecodedPairs> pairReadsFlags{};
        std::array<std::unique_ptr<BlockInfo>, kMaxDecodedPairs> blocks{};
        // Bumped whenever a block is dropped; older links are stale.
        uint64_t blockEpoch = 1;
        // Upper bound on (ready - cycle) of any VF/VI/ACC ready-table entry;
        // never lowered, since earlier code images may have set them.
        uint32_t maxWriteLatency = 0;
    };
    auto classifyWrites = [](const DecodedInstructionPair &pair) { return VU1Native::classifyWrites(pair); };
    enum : uint8_t
    {
        kReadsStatus = VU1Native::kReadsStatus,
        kReadsMac = VU1Native::kReadsMac,
    };
    auto lowerReadsFlags = [](const DecodedInstructionPair &pair) { return VU1Native::lowerReadsFlags(pair); };
    static FastOpTable s_fastOps;
    static const bool s_cachedReads = std::getenv("ATV_VU_NO_CACHED_READS") == nullptr;
    // Diagnostic only: aggregate by actual instructions across MPG replacements.
    // Addresses alone alias many different uploaded programs.
    static const bool s_blockCensus = std::getenv("ATV_VU_BLOCK_CENSUS") != nullptr;
    struct BlockCensus
    {
        uint32_t pc;
        std::vector<uint32_t> words;
        uint64_t calls = 0;
        bool dumped = false;
    };
    static std::vector<BlockCensus> s_blockCensusEntries;
    static uint64_t s_blockCensusCalls = 0;

    auto classifyReads = [](const DecodedInstructionPair &pair) { return VU1Native::classifyReads(pair); };
    static constexpr auto classifyUpper = [](uint32_t up) { return VU1Native::classifyUpper(up); };
    static constexpr auto packFastOp = [](const FastOp &op) { return VU1Native::packFastOp(op); };
    static constexpr auto unpackFastOp = [](uint64_t packed) { return VU1Native::unpackFastOp(packed); };

    // Natively compiled blocks: X(index, lower, upper) per pair, the exact
    // code words of a hot block (ATV_VU_BLOCK_CENSUS). Each pair becomes its
    // own instantiation of the fast halves with decode, register indices,
    // dest masks, broadcasts and I immediates folded at compile time. Cycle,
    // pipeline and PC handling stay those of translated replay.
    // pc 0x658: per-vertex transform, clip, perspective divide, FTOI, store.
#define ATV_VU_NATIVE_0658(X)                                                                                  \
    X(0, 0x8000033Cu, 0x01F409BCu) X(1, 0x8000033Cu, 0x01F410BDu) X(2, 0x81F6BB3Cu, 0x01F418BEu)                \
    X(3, 0x81F7C33Cu, 0x01F4264Bu) X(4, 0x10031801u, 0x01F469BCu) X(5, 0x8000033Cu, 0x01F470BDu)                \
    X(6, 0x8000033Cu, 0x01F478BEu) X(7, 0x81F903BCu, 0x000002FFu) X(8, 0x42FE0000u, 0x81F4860Bu)                \
    X(9, 0x01F54800u, 0x01D6B1FFu) X(10, 0x80050130u, 0x01E006A2u) X(11, 0x80060170u, 0x01D7B9FFu)              \
    X(12, 0x42800000u, 0x81D8C1FFu) X(13, 0x01F41800u, 0x0020069Eu) X(14, 0x8000033Cu, 0x01FAD17Cu)             \
    X(15, 0x8000033Cu, 0x01E0CF1Cu) X(16, 0x8000033Cu, 0x01C0AD5Cu) X(17, 0x2400003Fu, 0x01C5A258u)             \
    X(18, 0x800101B0u, 0x000002FFu) X(19, 0x03E7D000u, 0x000002FFu) X(20, 0x8000033Cu, 0x000002FFu)             \
    X(21, 0x03E7AFFFu, 0x01EA4A6Cu) X(22, 0x8059067Cu, 0x000002FFu) X(23, 0x81C04F3Eu, 0x000002FFu)             \
    X(24, 0x8000033Cu, 0x000002FFu) X(25, 0x8000033Cu, 0x000002FFu) X(26, 0x8000033Cu, 0x0051CCACu)             \
    X(27, 0x437F0000u, 0x80328F1Au) X(28, 0x8000033Cu, 0x0020E71Fu) X(29, 0x8000033Cu, 0x000002FFu)             \
    X(30, 0x8000033Cu, 0x000002FFu) X(31, 0x8000033Cu, 0x000002FFu) X(32, 0x8000033Cu, 0x0020E710u)             \
    X(33, 0x8000033Cu, 0x000002FFu) X(34, 0x8000033Cu, 0x000002FFu) X(35, 0x8000033Cu, 0x000002FFu)             \
    X(36, 0x8000033Cu, 0x01FBE17Du) X(37, 0x800522F0u, 0x000002FFu) X(38, 0x80065AF0u, 0x000002FFu)             \
    X(39, 0x5800783Au, 0x000002FFu) X(40, 0x03E7D801u, 0x000002FFu)
#define ATV_VU_NATIVE_WORDS(index, lower, upper) lower, upper,
    static constexpr uint32_t kNative0658Words[] = {ATV_VU_NATIVE_0658(ATV_VU_NATIVE_WORDS)};
#undef ATV_VU_NATIVE_WORDS
    static constexpr uint32_t kNative0658Pc = 0x658u;
    static constexpr uint32_t kNative0658Pairs = sizeof(kNative0658Words) / 8u;
    static const bool s_native = std::getenv("ATV_VU_NO_NATIVE") == nullptr;
    static const bool s_nativeStats = std::getenv("ATV_VU_NATIVE_STATS") != nullptr;
    // Diagnostic (ATV_VU_NATIVE_SPLIT=1, with STATS): TSC ticks of the native
    // block split into stall waits, each pair's execution, and end-of-pair work.
    static const bool s_nativeSplit = s_nativeStats && std::getenv("ATV_VU_NATIVE_SPLIT") != nullptr;
    static uint64_t s_splitWait = 0, s_splitEnd = 0, s_splitT2 = 0;
    static uint64_t s_splitExec[kNative0658Pairs]{};

    // Warm the decoded-pair cache; the fast loop then reads it by reference.
    (void)getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc & ~7u);
    const bool cacheUsable = m_decodedCodeCacheValid && m_cachedVuCode == vuCode && m_cachedMemory == memory &&
                             m_cachedCodeSize == codeSize && (m_state.pc & 7u) == 0u;
    if (cacheUsable)
    {
        if (s_fastOps.code != vuCode || s_fastOps.generation != m_cachedCodeGeneration ||
            s_fastOps.codeSize != codeSize)
        {
            const uint32_t pairs = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
            const bool incremental = s_fastOps.code == vuCode && s_fastOps.codeSize == codeSize;
            // dirtyBefore[i] = changed pairs before pair i.
            static std::array<uint32_t, kMaxDecodedPairs + 1u> dirtyBefore;
            dirtyBefore[0] = 0u;
            s_fastOps.readsFlags = 0u;
            for (uint32_t i = 0; i < pairs; ++i)
            {
                uint8_t *shadow = s_fastOps.shadow.data() + i * 8u;
                const bool changed = !incremental || std::memcmp(shadow, vuCode + i * 8u, 8u) != 0;
                if (changed)
                {
                    std::memcpy(shadow, vuCode + i * 8u, 8u);
                    s_fastOps.ops[i] = classifyUpper(m_decodedCodeCache[i].upper);
                    s_fastOps.ops[i].lowerKind = classifyLower(m_decodedCodeCache[i]);
                    if (s_cachedReads)
                        s_fastOps.reads[i] = classifyReads(m_decodedCodeCache[i]);
                    const FastWrites &w = s_fastOps.writes[i] = classifyWrites(m_decodedCodeCache[i]);
                    s_fastOps.pairReadsFlags[i] = lowerReadsFlags(m_decodedCodeCache[i]);
                    uint32_t &maxLatency = s_fastOps.maxWriteLatency;
                    maxLatency = std::max<uint32_t>(maxLatency, kAccForwardLatency);
                    if (w.upperLanes != 0u)
                        maxLatency = std::max(maxLatency, w.upperLat);
                    if (w.lowerLanes != 0u)
                        maxLatency = std::max(maxLatency, w.lowerLat);
                    if (w.viReg != 0u)
                        maxLatency = std::max(maxLatency, w.viLat);
                }
                s_fastOps.readsFlags |= s_fastOps.pairReadsFlags[i];
                dirtyBefore[i + 1u] = dirtyBefore[i] + (changed ? 1u : 0u);
            }
            for (uint32_t i = 0; i < pairs; ++i)
            {
                std::unique_ptr<BlockInfo> &block = s_fastOps.blocks[i];
                if (!block)
                    continue;
                // The pair after the block decided where (and whether) it ends.
                const uint32_t end = std::min(pairs, block->first + block->length + 1u);
                if (dirtyBefore[end] != dirtyBefore[block->first])
                {
                    block.reset();
                    ++s_fastOps.blockEpoch;
                }
            }
            for (uint32_t i = pairs; i < kMaxDecodedPairs; ++i)
            {
                if (s_fastOps.blocks[i])
                {
                    s_fastOps.blocks[i].reset();
                    ++s_fastOps.blockEpoch;
                }
            }
            s_fastOps.code = vuCode;
            s_fastOps.generation = m_cachedCodeGeneration;
            s_fastOps.codeSize = codeSize;
        }

        static const bool s_keepFlags = std::getenv("ATV_VU_FAST_FLAGS") != nullptr;
        auto fastReadyCycle = [&](const DecodedInstructionPair &decoded, const FastReads &reads) {
            const uint64_t *vfReady = m_vfReady[0].data();
            uint64_t ready = m_cycle;
            for (uint32_t i = 0; i < 16u; ++i)
                ready = std::max(ready, vfReady[reads.vf[i]]);
            for (uint32_t i = 0; i < 4u; ++i)
                ready = std::max(ready, m_viReady[reads.vi[i]]);
            for (uint32_t i = 4; i < reads.viCount; ++i)
                ready = std::max(ready, m_viReady[reads.vi[i]]);
            uint64_t accReady = 0;
            for (uint32_t i = 0; i < 4u; ++i)
                accReady = std::max(accReady, m_accReady[reads.acc[i]]);
            ready = std::max(ready, reads.accCount != 0u ? accReady : 0u);
            if (!reads.scalarWait)
                return ready;
            if ((decoded.lowerUsage.pipeline == PipelineFdiv || decoded.lowerUsage.waitQ) && m_fdiv.valid)
                ready = std::max(ready, m_fdiv.readyCycle);
            if (decoded.lowerUsage.pipeline == PipelineEfu)
                ready = std::max(ready, m_efuResourceReady);
            if (decoded.lowerUsage.waitP)
                for (const ScalarPipelineEntry &entry : m_efu)
                    if (entry.valid)
                        ready = std::max(ready, entry.readyCycle);
            if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
                ready = std::max(ready, m_cycle + 1u);
            return ready;
        };
        auto fastUpper = [&](const auto &opArg, uint32_t upper) { VU1Native::fastUpper(*this, opArg, upper); };
        auto fastLower = [&](auto kindArg, auto instrArg, uint32_t upper) {
            VU1Native::fastLower(*this, kindArg, instrArg, upper, vuData, dataSize, gs, memory);
        };

        g_vuSkipMacFlags = !s_keepFlags && s_fastOps.readsFlags == 0u;
        g_vuMacOnlyFlags = !s_keepFlags && s_fastOps.readsFlags == kReadsMac;
        g_vuFmacLean = (g_vuSkipMacFlags || g_vuMacOnlyFlags) && !g_vuApprox && !g_vuVerifyFast && !g_vuFmacStats && VU1Native::s_fmacZero;
        // ATV_VU_TRACE_FLAGMODE=1: fast-loop runs per FMAC flag mode, every 50000 runs.
        static const bool s_traceFlagMode = std::getenv("ATV_VU_TRACE_FLAGMODE") != nullptr;
        if (s_traceFlagMode)
        {
            static uint64_t s_modeRuns[4]{};
            ++s_modeRuns[g_vuFmacLean ? 0 : g_vuSkipMacFlags ? 1 : g_vuMacOnlyFlags ? 2 : 3];
            if ((s_modeRuns[0] + s_modeRuns[1] + s_modeRuns[2] + s_modeRuns[3]) % 50000u == 0u)
                std::fprintf(stderr, "[vu-flagmode] lean=%llu skip=%llu macOnly=%llu full=%llu readsFlags=%u\n",
                             static_cast<unsigned long long>(s_modeRuns[0]), static_cast<unsigned long long>(s_modeRuns[1]),
                             static_cast<unsigned long long>(s_modeRuns[2]), static_cast<unsigned long long>(s_modeRuns[3]),
                             static_cast<unsigned>(s_fastOps.readsFlags));
        }
        g_vuSkippedFlagsReady = 0;
        g_vuDirectStores = true;
        uint64_t writesReadyAt = 0;
        bool fastLeft = false;
        const CommitHint &hint = commitHintFor(this);

        // Everything a pair does between issue and the PC update, except the
        // ready-table writes.
        // NOP halves skip the calls; fastUpper only records the upper word for
        // FMAC flag helpers.
        const auto fastPair = [&](const FastOp &op, const DecodedInstructionPair &decoded) {
            if (op.kind != kNop)
                fastUpper(op, decoded.upper);
            if (op.lowerKind != kLowerNop)
                fastLower(op.lowerKind, decoded.lower, decoded.upper);
        };
        // One pair of a natively compiled block, with the same effects as the
        // translated replay's kPairFast / kPairFastViRecord / kPairIBitFast
        // handling of that pair.
        const auto nativePair = [&](auto lowerC, auto upperC, uint8_t mode, uint8_t viReg) {
            constexpr uint32_t lower = decltype(lowerC)::value;
            constexpr uint32_t upper = decltype(upperC)::value;
            constexpr bool iBit = (upper >> 31) != 0u;
            constexpr FastOp op = unpackFastOp(packFastOp(classifyUpper(upper)) |
                                               uint64_t{iBit ? VU1Native::kLowerExec : classifyLowerWord(lower)} << 48);
            static_assert(op.kind != VU1Native::kExec, "native pair needs a fast upper op");
            static_assert(iBit || op.lowerKind != VU1Native::kLowerExec, "native pair needs a fast lower op");
            constexpr std::integral_constant<uint64_t, packFastOp(op)> opC{};
            if constexpr (iBit)
            {
                if constexpr (op.kind != VU1Native::kNop)
                    fastUpper(opC, upper);
                constexpr float immediate = std::bit_cast<float>(lower);
                m_state.i = normalizeOperand(immediate);
                m_viBranchBackupValid = false;
            }
            else
            {
                const bool viRecord = mode == kPairFastViRecord;
                const int32_t oldVi = viRecord ? m_state.vi[viReg] : 0;
                if constexpr (op.kind != VU1Native::kNop)
                    fastUpper(opC, upper);
                if constexpr (op.lowerKind != VU1Native::kLowerNop)
                    fastLower(std::integral_constant<uint8_t, op.lowerKind>{}, lowerC, upper);
                m_viBranchBackupValid = false;
                if (viRecord)
                    recordViWriteForBranch(viReg, oldVi);
            }
        };
        const auto execPair = [&](const DecodedInstructionPair &decoded, const FastOp &op) {
            uint8_t writtenVi = 0u;
            int32_t oldVi = 0;
            if (decoded.lowerUsage.viWrite != 0u)
            {
                const uint32_t bits = decoded.lowerUsage.viWrite & 0xFFFEu;
                if (bits != 0u)
                {
                    writtenVi = static_cast<uint8_t>(std::countr_zero(bits));
                    oldVi = m_state.vi[writtenVi];
                }
            }

            const VfAccess upperWrite = decoded.upperUsage.vfWrite;
            const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
            const bool hasUpperWrite = upperWrite.reg != 0u;
            const bool hasLowerWrite = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
            const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
            float oldUpperVf[4];
            float oldLowerVf[4];
            float oldAcc[4];
            if (hasUpperWrite)
                std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
            if (hasDistinctLowerWrite)
                std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
            if (decoded.upperUsage.accWrite != 0u)
                std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));

            if (decoded.iBit)
            {
                fastUpper(op, decoded.upper);
                float immediate = 0.0f;
                std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
                m_state.i = normalizeOperand(immediate);
            }
            else if (decoded.upperVfShadowReg != 0u)
            {
                float oldVf[4];
                float upperVf[4];
                std::memcpy(oldVf, m_state.vf[decoded.upperVfShadowReg], sizeof(oldVf));
                fastUpper(op, decoded.upper);
                std::memcpy(upperVf, m_state.vf[decoded.upperVfShadowReg], sizeof(upperVf));
                std::memcpy(m_state.vf[decoded.upperVfShadowReg], oldVf, sizeof(oldVf));
                execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
                std::memcpy(m_state.vf[decoded.upperVfShadowReg], upperVf, sizeof(upperVf));
            }
            else
            {
                fastPair(op, decoded);
            }

            m_viBranchBackupValid = false;

            // Only the interpreter loop queues VF/VI/ACC writes and every run
            // ends with flushPipelines, so no queued write can land on top of
            // these and the write sequence numbers need no update.
            const auto keepDeclaredLanes = [](float *reg, const float *old, uint8_t lanes) {
                const __m128 keep = _mm_castsi128_ps(vuDestMask(lanes));
                _mm_storeu_ps(reg, _mm_or_ps(_mm_and_ps(keep, _mm_loadu_ps(reg)),
                                             _mm_andnot_ps(keep, _mm_loadu_ps(old))));
            };
            if (hasUpperWrite)
                keepDeclaredLanes(m_state.vf[upperWrite.reg], oldUpperVf, upperWrite.lanes);
            if (hasDistinctLowerWrite)
                keepDeclaredLanes(m_state.vf[lowerWrite.reg], oldLowerVf, lowerWrite.lanes);
            if (decoded.upperUsage.accWrite != 0u)
                keepDeclaredLanes(m_state.acc, oldAcc, decoded.upperUsage.accWrite);
            if (writtenVi != 0u)
            {
                m_state.vi[writtenVi] = static_cast<int16_t>(m_state.vi[writtenVi]);
                if (decoded.lowerUsage.delaysNextBranchRead)
                    recordViWriteForBranch(writtenVi, oldVi);
            }
        };
        // Ready-table writes of the pair just issued on m_cycle.
        const auto markWrites = [&](uint32_t pairIndex, const DecodedInstructionPair &decoded) {
            const FastWrites &w = s_fastOps.writes[pairIndex];
            if (w.multiVi)
                markPairWrites(decoded);
            else
            {
                // Same order as markPairWrites: lower VF, upper VF, VI, ACC.
                vuSetReadyLanes(m_vfReady[w.lowerReg].data(), w.lowerLanes, m_cycle + w.lowerLat);
                vuSetReadyLanes(m_vfReady[w.upperReg].data(), w.upperLanes, m_cycle + w.upperLat);
                m_viReady[w.viReg] = w.viReg != 0u ? m_cycle + w.viLat : 0u;
                vuSetReadyLanes(m_accReady.data(), w.accLanes, m_cycle + kAccForwardLatency);
            }
            writesReadyAt = std::max(writesReadyAt, m_cycle + w.readyOffset);
        };
        // One 16-byte store: the next pair's vector load of VF0 cannot be
        // forwarded from four scalar stores.
        const auto resetZeroRegisters = [&]() {
            _mm_storeu_ps(m_state.vf[0], _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f));
            m_state.vi[0] = 0;
        };
        // PC, branch and halt handling after a pair; true when the program ends.
        const auto finishPc = [&](const DecodedInstructionPair &decoded) {
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

            bool ended = false;
            if (m_state.haltAfterDelaySlot)
            {
                m_state.stoppedByD = m_pendingHaltD;
                m_state.stoppedByT = m_pendingHaltT;
                ended = true;
            }
            else if (m_state.ebit)
                ended = true;
            else if (haltBit && !haltBranch)
            {
                m_state.stoppedByD = dHalt;
                m_state.stoppedByT = tHalt;
                ended = true;
            }
            else if (decoded.eBit)
                m_state.ebit = true;
            else if (haltBranch)
            {
                m_state.haltAfterDelaySlot = true;
                m_pendingHaltD = dHalt;
                m_pendingHaltT = tHalt;
            }
            return ended;
        };
        // advanceOneCycle with the commit-hint check inlined.
        const auto endPairCycle = [&]() {
            ++m_cycle;
            m_state.cycles = m_cycle;
            if (m_cycle >= hint.nextReady)
                commitReadyPipelines();
            if (m_xgkick.active)
                progressXgkick();
        };
        // advanceTo for the fast loop. Without a live PATH1 transfer, the
        // cycles before hint.nextReady change nothing but the counter, so they
        // are crossed in one step (ATV_VU_VERIFY_FAST keeps every commit call;
        // ATV_VU_NO_IDLE_SKIP=1 disables).
        static const bool s_idleSkip = std::getenv("ATV_VU_NO_IDLE_SKIP") == nullptr;
        const auto fastAdvanceTo = [&](uint64_t target) {
            while (m_cycle < target)
            {
                if (s_idleSkip && !m_xgkick.active && !g_vuVerifyFast && hint.nextReady > m_cycle + 1u)
                {
                    m_cycle = std::min(target, hint.nextReady - 1u);
                    continue;
                }
                advanceOneCycle();
            }
        };

        static const bool s_replay = s_cachedReads && std::getenv("ATV_VU_NO_REPLAY") == nullptr;
        const uint32_t codePairs = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
        const auto buildBlock = [&](uint32_t first) {
            auto block = std::make_unique<BlockInfo>();
            block->first = first;
            bool written[kSlotCount]{};
            bool live[kSlotCount]{};
            bool tail = false; // the next pair is a delay slot or follows an E-bit
            bool ok = true;
            for (uint32_t i = first;; ++i)
            {
                if (i >= codePairs || (!tail && i - first >= kMaxBlockPairs))
                {
                    ok = !tail;
                    break;
                }
                const DecodedInstructionPair &d = m_decodedCodeCache[i];
                const FastWrites &w = s_fastOps.writes[i];
                const FastReads &r = s_fastOps.reads[i];
                const InstructionUsage &lu = d.lowerUsage;
                const bool branch = !d.iBit && lu.pipeline == PipelineBranch;
                if (d.upperUsage.reserved || lu.reserved || d.dBit || d.tBit || w.multiVi ||
                    (tail && (branch || d.eBit)))
                {
                    // Without a pending terminator the block just ends before this pair.
                    ok = !tail;
                    break;
                }
                const auto use = [&](std::vector<uint8_t> &list, uint32_t base, uint32_t index) {
                    const uint32_t slot = base + index;
                    if (!written[slot] && !live[slot])
                    {
                        live[slot] = true;
                        list.push_back(static_cast<uint8_t>(index));
                    }
                };
                for (uint32_t n = 0; n < r.vfCount; ++n)
                    use(block->liveVf, 0u, r.vf[n]);
                for (uint32_t n = 0; n < r.viCount; ++n)
                    use(block->liveVi, kSlotVi, r.vi[n]);
                for (uint32_t n = 0; n < r.accCount; ++n)
                    use(block->liveAcc, kSlotAcc, r.acc[n]);
                for (uint32_t b = 0; b < 4u; ++b)
                {
                    if ((w.lowerLanes & (1u << b)) != 0u)
                        written[w.lowerReg * 4u + b] = true;
                    if ((w.upperLanes & (1u << b)) != 0u)
                        written[w.upperReg * 4u + b] = true;
                    if ((w.accLanes & (1u << b)) != 0u)
                        written[kSlotAcc + b] = true;
                }
                if (w.viReg != 0u)
                    written[kSlotVi + w.viReg] = true;
                block->needFdiv |= lu.pipeline == PipelineFdiv || lu.waitQ;
                block->issuesFdiv |= lu.pipeline == PipelineFdiv;
                block->needEfu |= lu.pipeline == PipelineEfu;
                block->needP |= lu.waitP;

                const FastOp &op = s_fastOps.ops[i];
                const bool viBranchRecord = lu.delaysNextBranchRead && w.viReg != 0u;
                const bool ibitFast = d.iBit && d.upperVfShadowReg == 0u && op.kind != kExec;
                const bool fast =
                    !d.iBit && d.upperVfShadowReg == 0u && op.kind != kExec && op.lowerKind != kLowerExec;
                uint8_t mode = kPairGeneric;
                if (ibitFast)
                    mode = kPairIBitFast;
                else if (fast)
                    mode = viBranchRecord ? kPairFastViRecord : kPairFast;
                block->simple.push_back(mode);
                block->xgkick.push_back(lu.pipeline == PipelineXgkick);
                ++block->length;
                if (tail)
                    break;
                tail = branch || d.eBit;
            }
            block->replayable = ok && block->length != 0u;
            if (block->replayable)
            {
                bool translated = true;
                bool readsQ = false;
                bool readsP = false;
                bool readsClip = false;
                bool issuesClip = false;
                for (uint32_t j = 0; j < block->length; ++j)
                {
                    const uint8_t mode = block->simple[j];
                    if (block->xgkick[j] ||
                        (mode != kPairFast && mode != kPairFastViRecord && mode != kPairIBitFast))
                    {
                        translated = false;
                        break;
                    }
                    const FastOp &fop = s_fastOps.ops[first + j];
                    issuesClip |= fop.kind == kClip;
                    readsQ |= fop.kind == kMulQ || fop.lowerKind == kWaitq || fop.lowerKind == kDiv;
                    readsP |= fop.lowerKind == kMfp;
                    readsClip |=
                        fop.lowerKind == kFcand || fop.lowerKind == kFcor || fop.lowerKind == kFcget;
                }
                block->translated = translated;
                block->cycleQuiet = translated && !block->issuesFdiv && !block->needEfu && !issuesClip &&
                                    !readsQ && !readsP && !readsClip;
                // Same words imply the same classification, so a translated
                // match has exactly the pair modes the native code handles.
                if (first * 8u == kNative0658Pc && block->length == kNative0658Pairs &&
                    std::memcmp(s_fastOps.shadow.data() + first * 8u, kNative0658Words, sizeof(kNative0658Words)) == 0)
                {
                    block->nativeCandidate = 1u;
                    block->native = s_native && translated ? 1u : 0u;
                }
            }
            if (s_blockCensus && block->replayable)
            {
                std::vector<uint32_t> words(block->length * 2u);
                std::memcpy(words.data(), s_fastOps.shadow.data() + first * 8u, words.size() * sizeof(uint32_t));
                uint32_t index = 0;
                for (; index < s_blockCensusEntries.size(); ++index)
                    if (s_blockCensusEntries[index].pc == first * 8u && s_blockCensusEntries[index].words == words)
                        break;
                if (index == s_blockCensusEntries.size())
                    s_blockCensusEntries.push_back({first * 8u, std::move(words)});
                block->censusIndex = index;
            }
            return block;
        };
        const auto computeSignature = [&](const BlockInfo &block, uint8_t *signature, uint32_t &length) {
            const uint64_t now = m_cycle;
            uint32_t n = 0;
            bool fits = true;
            const auto put = [&](uint64_t ready) {
                const uint64_t rel = ready > now ? ready - now : 0u;
                fits &= rel <= 255u;
                signature[n++] = static_cast<uint8_t>(rel);
            };
            const uint64_t *vfReady = m_vfReady[0].data();
            for (const uint8_t slot : block.liveVf)
                put(vfReady[slot]);
            for (const uint8_t reg : block.liveVi)
                put(m_viReady[reg]);
            for (const uint8_t lane : block.liveAcc)
                put(m_accReady[lane]);
            if (block.needFdiv)
                put(m_fdiv.valid ? m_fdiv.readyCycle : 0u);
            if (block.needEfu)
                put(m_efuResourceReady);
            if (block.needP)
            {
                uint64_t p = 0;
                for (const ScalarPipelineEntry &entry : m_efu)
                    if (entry.valid)
                        p = std::max(p, entry.readyCycle);
                put(p);
            }
            length = n;
            return fits;
        };
        // Ready-table writes of the first `count` pairs of a replayed block.
        const auto applyScheduleWrites = [&](const BlockInfo &block, const BlockSchedule &schedule, uint64_t entry,
                                             uint32_t count) {
            for (uint32_t j = 0; j < count; ++j)
            {
                const FastWrites &w = s_fastOps.writes[block.first + j];
                const uint64_t cycle = entry + schedule.issue[j];
                vuSetReadyLanes(m_vfReady[w.lowerReg].data(), w.lowerLanes, cycle + w.lowerLat);
                vuSetReadyLanes(m_vfReady[w.upperReg].data(), w.upperLanes, cycle + w.upperLat);
                m_viReady[w.viReg] = w.viReg != 0u ? cycle + w.viLat : 0u;
                vuSetReadyLanes(m_accReady.data(), w.accLanes, cycle + kAccForwardLatency);
                writesReadyAt = std::max(writesReadyAt, cycle + w.readyOffset);
            }
            g_vu1PairTotal += count;
            g_vu1ReplayPairs += count;
        };
        enum class ReplayEnd
        {
            Done,
            Left, // continue per pair at m_state.pc
            ProgramEnded,
        };
        static const bool s_translate = std::getenv("ATV_VU_NO_TRANSLATE") == nullptr;
        static const bool s_eventReplay = std::getenv("ATV_VU_NO_EVENT_REPLAY") == nullptr;
        // Verifier negative control: event replay skips due commits (wrong on purpose).
        static const bool s_eventSelfTest = std::getenv("ATV_VU_EVENT_SELFTEST") != nullptr;
        const auto replay = [&](BlockInfo &block, const BlockSchedule &schedule) {
            const uint64_t entry = m_cycle;
            const uint32_t length = block.length;
            bool ended = false;
            ++g_vu1ReplayBlocks;
            if (s_blockCensus)
            {
                ++s_blockCensusEntries[block.censusIndex].calls;
                if ((++s_blockCensusCalls & 0xFFFFFu) == 0u)
                {
                    std::vector<uint32_t> order(s_blockCensusEntries.size());
                    for (uint32_t j = 0; j < order.size(); ++j) order[j] = j;
                    std::sort(order.begin(), order.end(), [](uint32_t a, uint32_t b) {
                        const auto &aa = s_blockCensusEntries[a];
                        const auto &bb = s_blockCensusEntries[b];
                        return aa.calls * aa.words.size() > bb.calls * bb.words.size();
                    });
                    for (uint32_t j = 0; j < std::min<size_t>(12u, order.size()); ++j)
                    {
                        auto &item = s_blockCensusEntries[order[j]];
                        std::fprintf(stderr, "[vu:block] id=%u pc=0x%x length=%zu calls=%llu pairs=%llu totalCalls=%llu\n",
                                     order[j], item.pc, item.words.size() / 2u, item.calls,
                                     item.calls * item.words.size() / 2u, s_blockCensusCalls);
                        if (!item.dumped)
                        {
                            item.dumped = true;
                            for (uint32_t n = 0; n < item.words.size(); n += 2u)
                                std::fprintf(stderr, "[vu:block-code] id=%u pc=0x%x lower=%08x upper=%08x\n",
                                             order[j], item.pc + n * 4u, item.words[n], item.words[n + 1u]);
                        }
                    }
                }
            }

            // Translated blocks: only fast / I-bit-fast / VI-record pairs, no
            // XGKICK. Dense schedules (issue[i] == base+i) skip per-pair waits.
            // cycleQuiet + no due commits: bare cycle bumps (no commit/xgkick).
            if (s_translate && block.translated)
            {
                const uint32_t base = schedule.issue.empty() ? 0u : schedule.issue[0];
                bool dense = schedule.issue.size() == length;
                for (uint32_t i = 0; dense && i < length; ++i)
                    dense = schedule.issue[i] == base + i;
                if (dense && base != 0u)
                {
                    const uint64_t firstIssue = entry + base;
                    fastAdvanceTo(firstIssue);
                }
                // Skip PATH1 progress when no transfer is live (translated
                // blocks never start one). Bare cycle bumps are only safe when
                // nothing pipelined can become visible mid-block: no Q/P/clip
                // traffic in the block, MAC-only/skipped flags (FMAC does not
                // queue the flag pipeline), and no commit due before blockEnd.
                // Disabled under verify so clip/Q/P observability stays exact.
                // Always re-read hint.nextReady after the pair: CLIP/DIV/ELENG
                // lower it during the pair, so a cached commitAt would miss.
                const bool skipKick = !m_xgkick.active;
                const uint64_t blockEnd = dense ? entry + base + length : entry + schedule.total;
                const bool bareCycle = dense && skipKick && block.cycleQuiet && !g_vuVerifyTrace &&
                                       (g_vuSkipMacFlags || g_vuMacOnlyFlags) &&
                                       hint.nextReady >= blockEnd;
                const auto endTranslatedCycle = [&]() {
                    ++m_cycle;
                    if (bareCycle)
                        return;
                    if (m_cycle >= hint.nextReady)
                        commitReadyPipelines();
                    if (!skipKick)
                        progressXgkick();
                };
                const FastOp *const ops = s_fastOps.ops.data() + block.first;
                const uint8_t *const modes = block.simple.data();
                const uint8_t *const flagReads = s_fastOps.pairReadsFlags.data() + block.first;
                const DecodedInstructionPair *const pairs = m_decodedCodeCache.data() + block.first;
                const bool batchPc = !g_vuVerifyTrace && length >= 2u;
                const uint32_t body = batchPc ? length - 2u : length;
                ReplayEnd bail = ReplayEnd::Done;
                // One pair: wait, execute, PC, cycle. False when replay bails.
                const bool split = s_nativeSplit && block.native != 0u;
                // Event mode: with no live PATH1 transfer, each pair moves the
                // cycle straight to the next pair's issue cycle, stopping only on
                // cycles where hint.nextReady says something becomes visible
                // (fastAdvanceTo commits those one cycle at a time).
                const uint16_t *const issueAt = schedule.issue.data();
                const bool eventMode = s_eventReplay && skipKick && !g_vuVerifyTrace && !split;
                if (eventMode)
                    fastAdvanceTo(entry + issueAt[0]);
                const auto step = [&](uint32_t i, const auto &execute) {
                    const DecodedInstructionPair &decoded = pairs[i];
                    const uint64_t t0 = split ? __rdtsc() : 0u;
                    if (!dense && !eventMode)
                    {
                        fastAdvanceTo(entry + schedule.issue[i]);
                    }
                    if (g_vuVerifyTrace)
                        recordVerifyPair(m_state, m_cycle, decoded.upperUsage, decoded.lowerUsage);
                    if (g_vuMacOnlyFlags && flagReads[i] != 0u)
                        applyPendingMac(m_state.mac, m_cycle);
                    const uint64_t t1 = split ? __rdtsc() : 0u;
                    execute(decoded);
                    if (split)
                    {
                        const uint64_t t2 = __rdtsc();
                        s_splitWait += t1 - t0;
                        s_splitExec[i] += t2 - t1;
                        s_splitT2 = t2;
                    }
                    resetZeroRegisters();
                    if (batchPc)
                    {
                        if (i + 1u == body)
                            m_state.pc += body * 8u;
                        else if (i >= body)
                            ended = finishPc(decoded);
                    }
                    else if (i + 2u < length)
                        m_state.pc += 8u;
                    else
                        ended = finishPc(decoded);
                    const bool leaving = (ended || m_stopRequested) && i + 1u < length;
                    if (eventMode && !leaving)
                    {
                        const uint64_t next = i + 1u < length ? entry + issueAt[i + 1u] : m_cycle + 1u;
                        if (hint.nextReady > next || s_eventSelfTest)
                            m_cycle = next;
                        else
                            fastAdvanceTo(next);
                    }
                    else
                        endTranslatedCycle();
                    if (split)
                        s_splitEnd += __rdtsc() - s_splitT2;
                    if (leaving)
                    {
                        applyScheduleWrites(block, schedule, entry, i + 1u);
                        ++g_vu1ReplayBails;
                        bail = ended ? ReplayEnd::ProgramEnded : ReplayEnd::Left;
                        return false;
                    }
                    return true;
                };
                if (block.native == 1u)
                {
                    const FastWrites *const writes = s_fastOps.writes.data() + block.first;
#define ATV_VU_NATIVE_STEP(index, lower, upper)                                                                \
    &&step(index, [&](const DecodedInstructionPair &) {                                                    \
        nativePair(std::integral_constant<uint32_t, lower>{}, std::integral_constant<uint32_t, upper>{},   \
                   modes[index], writes[index].viReg);                                                     \
    })
                    if (!(true ATV_VU_NATIVE_0658(ATV_VU_NATIVE_STEP)))
                        return bail;
#undef ATV_VU_NATIVE_STEP
                }
                else
                {
                    for (uint32_t i = 0; i < length; ++i)
                    {
                        const bool more = step(i, [&](const DecodedInstructionPair &decoded) {
                            const FastOp &op = ops[i];
                            const uint8_t mode = modes[i];
                            if (mode == kPairIBitFast)
                            {
                                // Same-pair upper uses the previous I; the immediate
                                // lands after the upper executes (matches execPair).
                                if (op.kind != kNop)
                                    fastUpper(op, decoded.upper);
                                float immediate;
                                std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
                                m_state.i = normalizeOperand(immediate);
                                m_viBranchBackupValid = false;
                            }
                            else if (mode == kPairFastViRecord)
                            {
                                const uint8_t reg = s_fastOps.writes[block.first + i].viReg;
                                const int32_t oldVi = m_state.vi[reg];
                                fastPair(op, decoded);
                                m_viBranchBackupValid = false;
                                recordViWriteForBranch(reg, oldVi);
                            }
                            else
                            {
                                fastPair(op, decoded);
                                m_viBranchBackupValid = false;
                            }
                        });
                        if (!more)
                            return bail;
                    }
                }
            }
            else
            {
                for (uint32_t i = 0; i < length; ++i)
                {
                    const uint32_t pairIndex = block.first + i;
                    const DecodedInstructionPair &decoded = m_decodedCodeCache[pairIndex];
                    const uint64_t issue = entry + schedule.issue[i];
                    if (block.xgkick[i])
                    {
                        // Same wait as the per-pair loop; the previous PATH1
                        // transfer's length depends on packet data.
                        uint64_t ready = std::max(entry + schedule.dataReady[i], m_cycle);
                        if (m_xgkick.active)
                            ready = std::max(ready, m_cycle + 1u);
                        if (ready > m_cycle)
                        {
                            if (ready >= budgetEnd)
                                fastAdvanceTo(budgetEnd);
                            else
                                fastAdvanceTo(ready);
                        }
                        while (m_xgkick.active && m_cycle < budgetEnd)
                            advanceOneCycle();
                        if (m_cycle != issue || m_xgkick.active)
                        {
                            applyScheduleWrites(block, schedule, entry, i);
                            ++g_vu1ReplayBails;
                            return ReplayEnd::Left;
                        }
                    }
                    else
                    {
                        fastAdvanceTo(issue);
                    }
                    if (g_vuVerifyTrace)
                        recordVerifyPair(m_state, m_cycle, decoded.upperUsage, decoded.lowerUsage);

                    const FastOp &op = s_fastOps.ops[pairIndex];
                    if (g_vuMacOnlyFlags && s_fastOps.pairReadsFlags[pairIndex] != 0u)
                        applyPendingMac(m_state.mac, m_cycle);
                    const uint8_t mode = block.simple[i];
                    if (mode == kPairFast)
                    {
                        fastPair(op, decoded);
                        m_viBranchBackupValid = false;
                    }
                    else if (mode == kPairFastViRecord)
                    {
                        const uint8_t reg = s_fastOps.writes[pairIndex].viReg;
                        const int32_t oldVi = m_state.vi[reg];
                        fastPair(op, decoded);
                        m_viBranchBackupValid = false;
                        recordViWriteForBranch(reg, oldVi);
                    }
                    else if (mode == kPairIBitFast)
                    {
                        if (op.kind != kNop)
                            fastUpper(op, decoded.upper);
                        float immediate;
                        std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
                        m_state.i = normalizeOperand(immediate);
                        m_viBranchBackupValid = false;
                    }
                    else
                    {
                        execPair(decoded, op);
                    }
                    resetZeroRegisters();
                    if (i + 2u < length)
                        m_state.pc += 8u;
                    else
                        ended = finishPc(decoded);
                    endPairCycle();
                    if ((ended || m_stopRequested) && i + 1u < length)
                    {
                        applyScheduleWrites(block, schedule, entry, i + 1u);
                        ++g_vu1ReplayBails;
                        return ended ? ReplayEnd::ProgramEnded : ReplayEnd::Left;
                    }
                }
            }
            uint64_t *vfReady = m_vfReady[0].data();
            for (const uint32_t packed : schedule.exitVf)
                vfReady[packed >> 16] = entry + (packed & 0xFFFFu);
            for (const uint32_t packed : schedule.exitVi)
                m_viReady[packed >> 16] = entry + (packed & 0xFFFFu);
            for (const uint32_t packed : schedule.exitAcc)
                m_accReady[packed >> 16] = entry + (packed & 0xFFFFu);
            writesReadyAt = std::max(writesReadyAt, entry + schedule.writesReadyOffset);
            g_vu1PairTotal += length;
            g_vu1ReplayPairs += length;
            ++block.hits;
            return ended ? ReplayEnd::ProgramEnded : ReplayEnd::Done;
        };

        struct Recording
        {
            BlockInfo *block = nullptr;
            uint64_t entry = 0;
            uint32_t count = 0;
            uint32_t signatureLength = 0;
            uint8_t signature[kMaxSignature];
            uint16_t issue[kMaxBlockPairs + 1u];
            uint16_t dataReady[kMaxBlockPairs + 1u];
        };
        static Recording s_rec;
        s_rec.block = nullptr;
        const auto finishRecording = [&]() {
            BlockInfo &block = *s_rec.block;
            s_rec.block = nullptr;
            BlockSchedule schedule;
            schedule.signature.assign(s_rec.signature, s_rec.signature + s_rec.signatureLength);
            schedule.issue.assign(s_rec.issue, s_rec.issue + block.length);
            schedule.dataReady.assign(s_rec.dataReady, s_rec.dataReady + block.length);
            schedule.total = s_rec.issue[block.length - 1u] + 1u;
            uint32_t finalReady[kSlotCount];
            bool hasFinal[kSlotCount]{};
            const auto set = [&](uint32_t slot, uint32_t cycle) {
                finalReady[slot] = cycle;
                hasFinal[slot] = true;
            };
            for (uint32_t j = 0; j < block.length; ++j)
            {
                // Same order as markWrites: lower VF, upper VF, VI, ACC.
                const FastWrites &w = s_fastOps.writes[block.first + j];
                const uint32_t cycle = s_rec.issue[j];
                for (uint32_t b = 0; b < 4u; ++b)
                    if ((w.lowerLanes & (1u << b)) != 0u)
                        set(w.lowerReg * 4u + b, cycle + w.lowerLat);
                for (uint32_t b = 0; b < 4u; ++b)
                    if ((w.upperLanes & (1u << b)) != 0u)
                        set(w.upperReg * 4u + b, cycle + w.upperLat);
                if (w.viReg != 0u)
                    set(kSlotVi + w.viReg, cycle + w.viLat);
                for (uint32_t b = 0; b < 4u; ++b)
                    if ((w.accLanes & (1u << b)) != 0u)
                        set(kSlotAcc + b, cycle + kAccForwardLatency);
                schedule.writesReadyOffset = std::max(schedule.writesReadyOffset, cycle + w.readyOffset);
            }
            for (uint32_t slot = 0; slot < kSlotCount; ++slot)
            {
                if (!hasFinal[slot])
                    continue;
                if (slot < kSlotVi)
                    schedule.exitVf.push_back((slot << 16) | finalReady[slot]);
                else if (slot < kSlotAcc)
                    schedule.exitVi.push_back(((slot - kSlotVi) << 16) | finalReady[slot]);
                else
                    schedule.exitAcc.push_back(((slot - kSlotAcc) << 16) | finalReady[slot]);
            }
            static uint64_t s_nextSerial = 0;
            schedule.serial = ++s_nextSerial;
            if (block.schedules.empty())
                block.schedules.reserve(kMaxBlockSchedules);
            if (block.schedules.size() < kMaxBlockSchedules)
                block.schedules.push_back(std::move(schedule));
            else
            {
                block.schedules[block.nextVictim] = std::move(schedule);
                block.nextVictim = (block.nextVictim + 1u) % kMaxBlockSchedules;
            }
            ++block.records;
            ++g_vu1ReplayRecords;
        };
        // Links `from`'s schedule to `to`'s schedule for the current PC when
        // the exit state of `from` (replayed on that schedule) fixes every
        // entry of `to`'s signature; `signature` is the one just computed.
        const auto tryLink = [&](BlockInfo &from, uint32_t fromIndex, BlockInfo &to, uint32_t toIndex,
                                 const uint8_t *signature) {
            BlockSchedule &schedule = from.schedules[fromIndex];
            const int32_t total = static_cast<int32_t>(schedule.total);
            const auto after = [&](int32_t ready) { return ready > total ? ready - total : 0; };
            // Ready time relative to the exit of `from`, or -1 when unknown.
            int32_t known[kSlotCount];
            std::fill(std::begin(known), std::end(known),
                      schedule.total >= s_fastOps.maxWriteLatency ? 0 : -1);
            const uint8_t *fromSignature = schedule.signature.data();
            uint32_t n = 0;
            for (const uint8_t slot : from.liveVf)
                known[slot] = after(fromSignature[n++]);
            for (const uint8_t reg : from.liveVi)
                known[kSlotVi + reg] = after(fromSignature[n++]);
            for (const uint8_t lane : from.liveAcc)
                known[kSlotAcc + lane] = after(fromSignature[n++]);
            const int32_t fromFdiv = from.needFdiv ? fromSignature[n++] : -1;
            if (from.needEfu)
                ++n;
            const int32_t fromP = from.needP ? fromSignature[n++] : -1;
            for (const uint32_t packed : schedule.exitVf)
                known[packed >> 16] = after(packed & 0xFFFFu);
            for (const uint32_t packed : schedule.exitVi)
                known[kSlotVi + (packed >> 16)] = after(packed & 0xFFFFu);
            for (const uint32_t packed : schedule.exitAcc)
                known[kSlotAcc + (packed >> 16)] = after(packed & 0xFFFFu);

            BlockLink link;
            uint32_t m = 0;
            // Entries `from` does not fix become checks on the link.
            const auto take = [&](int32_t value, uint32_t code) {
                const uint8_t actual = signature[m++];
                if (value >= 0)
                    return actual == static_cast<uint8_t>(value);
                if (link.checkCount == kMaxLinkChecks)
                    return false;
                link.check[link.checkCount] = static_cast<uint8_t>(code);
                link.expect[link.checkCount] = actual;
                ++link.checkCount;
                return true;
            };
            for (const uint8_t slot : to.liveVf)
                if (!take(known[slot], slot))
                    return;
            for (const uint8_t reg : to.liveVi)
                if (!take(known[kSlotVi + reg], kSlotVi + reg))
                    return;
            for (const uint8_t lane : to.liveAcc)
                if (!take(known[kSlotAcc + lane], kSlotAcc + lane))
                    return;
            // A block that issues DIV/EFU ops fixes those pipelines' exit state.
            if (to.needFdiv &&
                !take(from.issuesFdiv ? signature[m]
                      : fromFdiv >= 0 ? after(fromFdiv)
                      : schedule.total >= kMaxFdivLatency ? 0 : -1,
                      kCheckFdiv))
                return;
            if (to.needEfu &&
                !take(from.needEfu ? signature[m] : schedule.total >= kMaxEfuLatency ? 0 : -1, kCheckEfu))
                return;
            if (to.needP &&
                !take(from.needEfu ? signature[m]
                      : fromP >= 0 ? after(fromP)
                      : schedule.total >= kMaxEfuLatency ? 0 : -1,
                      kCheckP))
                return;
            link.pc = m_state.pc;
            link.index = toIndex;
            link.block = &to;
            link.serial = to.schedules[toIndex].serial;
            link.epoch = s_fastOps.blockEpoch;
            schedule.links[schedule.nextLink] = link;
            schedule.nextLink = (schedule.nextLink + 1u) % kMaxBlockLinks;
        };
        // Signature value of a link check in the current state.
        const auto checkValue = [&](uint32_t code) -> uint64_t {
            uint64_t ready;
            if (code < kSlotVi)
                ready = m_vfReady[0].data()[code];
            else if (code < kSlotAcc)
                ready = m_viReady[code - kSlotVi];
            else if (code < kSlotCount)
                ready = m_accReady[code - kSlotAcc];
            else if (code == kCheckFdiv)
                ready = m_fdiv.valid ? m_fdiv.readyCycle : 0u;
            else if (code == kCheckEfu)
                ready = m_efuResourceReady;
            else
            {
                ready = 0;
                for (const ScalarPipelineEntry &entry : m_efu)
                    if (entry.valid)
                        ready = std::max(ready, entry.readyCycle);
            }
            return ready > m_cycle ? ready - m_cycle : 0u;
        };

        uint8_t signature[kMaxSignature];
        // Block (and schedule) replayed to completion just before this pair.
        BlockInfo *chainBlock = nullptr;
        uint32_t chainSchedule = 0;
        // Natively compiled blocks (ps2_vu1_native.inc). They need the exact
        // per-pair semantics, so not while per-pair traces are recorded.
        const bool nativeOn = VU1Native::s_enabled && codeSize == 0x4000u && !g_vuVerifyTrace;
        VU1Native::Ctx nativeCtx;
        VU1Native::initCtx(nativeCtx, *this, vuData, dataSize, gs, memory, budgetEnd, &writesReadyAt, hint);
        // ATV_VU_NATIVE_GEN: blocks are generated where the previous pair
        // ended one (a branch delay slot or E-bit pair), and at run entry.
        bool nativeBoundary = true;
        bool nativeTail = false;
        while (m_cycle < budgetEnd && !m_stopRequested)
        {
            if (m_cycle >= hint.nextReady)
                commitReadyPipelines();
            const uint32_t pairIndex = m_state.pc >> 3;
            if ((m_state.pc & 7u) != 0u || m_state.pc + 8u > codeSize || pairIndex >= kMaxDecodedPairs)
            {
                fastLeft = true;
                break;
            }
            if (nativeOn && !m_state.branchPending && !m_state.ebit && !m_state.haltAfterDelaySlot)
            {
                const VU1Native::BlockFn fn = VU1Native::lookup(vuCode, m_cachedCodeGeneration, pairIndex);
                const uint64_t nativeCycleBefore = m_cycle;
                const uint32_t end = fn ? (s_rec.block = nullptr, chainBlock = nullptr, fn(nativeCtx)) : VU1Native::kNativeBail;
                if (fn && end == VU1Native::kNativeEnded)
                {
                    programEnded = true;
                    break;
                }
                // A block that bails before its first pair without advancing (that pair would stall
                // past the budget) would be dispatched again forever: let the per-pair path below
                // take this pair; it advances to the budget and stops.
                if (fn && !(end == VU1Native::kNativeBail && m_state.pc == (pairIndex << 3) && m_cycle == nativeCycleBefore))
                {
                    nativeBoundary = end == VU1Native::kNativeContinue;
                    nativeTail = false;
                    continue;
                }
                if (!fn && VU1Native::s_genPath && nativeBoundary)
                    VU1Native::generate(*this, pairIndex, codePairs);
            }
            BlockInfo *const fromBlock = chainBlock;
            const uint32_t fromSchedule = chainSchedule;
            chainBlock = nullptr;
            if (s_replay && s_rec.block == nullptr && !m_state.branchPending && !m_state.ebit &&
                !m_state.haltAfterDelaySlot)
            {
                BlockInfo *block = nullptr;
                const BlockSchedule *found = nullptr;
                uint32_t foundIndex = 0;
                if (fromBlock)
                {
                    for (const BlockLink &link : fromBlock->schedules[fromSchedule].links)
                    {
                        if (link.pc != m_state.pc || link.epoch != s_fastOps.blockEpoch ||
                            link.block->schedules[link.index].serial != link.serial)
                            continue;
                        bool holds = true;
                        for (uint32_t c = 0; c < link.checkCount && holds; ++c)
                            holds = checkValue(link.check[c]) == link.expect[c];
                        if (holds)
                        {
                            block = link.block;
                            foundIndex = link.index;
                            found = &block->schedules[foundIndex];
                            ++g_vu1ReplayLinked;
                            break;
                        }
                    }
                }
                if (!found)
                {
                    std::unique_ptr<BlockInfo> &blockSlot = s_fastOps.blocks[pairIndex];
                    if (!blockSlot)
                        blockSlot = buildBlock(pairIndex);
                    block = blockSlot.get();
                    uint32_t signatureLength = 0;
                    if (block->replayable && computeSignature(*block, signature, signatureLength))
                    {
                        for (uint32_t s = 0; s < block->schedules.size(); ++s)
                        {
                            const BlockSchedule &schedule = block->schedules[s];
                            if (schedule.signature.size() == signatureLength &&
                                std::memcmp(schedule.signature.data(), signature, signatureLength) == 0)
                            {
                                found = &schedule;
                                foundIndex = s;
                                break;
                            }
                        }
                        if (found && fromBlock)
                            tryLink(*fromBlock, fromSchedule, *block, foundIndex, signature);
                        if (!found)
                        {
                            // Entry states that rarely repeat only cost recording time.
                            if (block->records >= 64u && block->hits < 8u * block->records)
                                block->replayable = false;
                            else
                            {
                                s_rec.block = block;
                                s_rec.entry = m_cycle;
                                s_rec.count = 0;
                                s_rec.signatureLength = signatureLength;
                                std::memcpy(s_rec.signature, signature, signatureLength);
                            }
                        }
                    }
                }
                if (found && m_cycle + found->total <= budgetEnd)
                {
                    const bool timed = s_nativeStats && block->nativeCandidate != 0u;
                    const uint64_t startTicks = timed ? __rdtsc() : 0u;
                    const uint64_t startCycle = m_cycle;
                    const uint64_t startCommits = g_vuCommitPasses;
                    const ReplayEnd end = replay(*block, *found);
                    if (timed)
                    {
                        // Host TSC ticks per replay of the native-candidate block,
                        // comparable between ATV_VU_NO_NATIVE=1 and default runs.
                        static uint64_t s_ticks = 0, s_calls = 0, s_nativeCalls = 0, s_cycles = 0, s_commits = 0,
                                        s_dense = 0;
                        s_ticks += __rdtsc() - startTicks;
                        s_cycles += m_cycle - startCycle;
                        s_commits += g_vuCommitPasses - startCommits;
                        s_dense += found->total == block->length ? 1u : 0u;
                        s_nativeCalls += block->native;
                        if ((++s_calls & 0x3FFFFu) == 0u)
                            std::fprintf(stderr,
                                         "[vu:native] pc=0x%x calls=%llu native=%llu translated=%d ticks/call=%.1f "
                                         "ticks/pair=%.2f cycles/call=%.1f commits/call=%.2f dense=%.3f schedules=%zu\n",
                                         block->first * 8u, static_cast<unsigned long long>(s_calls),
                                         static_cast<unsigned long long>(s_nativeCalls), block->translated ? 1 : 0,
                                         static_cast<double>(s_ticks) / static_cast<double>(s_calls),
                                         static_cast<double>(s_ticks) / static_cast<double>(s_calls * block->length),
                                         static_cast<double>(s_cycles) / static_cast<double>(s_calls),
                                         static_cast<double>(s_commits) / static_cast<double>(s_calls),
                                         static_cast<double>(s_dense) / static_cast<double>(s_calls),
                                         block->schedules.size());
                        if (s_nativeSplit && (s_calls & 0x3FFFFu) == 0u && s_nativeCalls != 0u)
                        {
                            const double n = static_cast<double>(s_nativeCalls);
                            uint64_t exec = 0;
                            for (const uint64_t t : s_splitExec)
                                exec += t;
                            std::fprintf(stderr, "[vu:native-split] wait=%.1f exec=%.1f end=%.1f ticks/call; exec per pair:",
                                         s_splitWait / n, exec / n, s_splitEnd / n);
                            for (uint32_t j = 0; j < kNative0658Pairs; ++j)
                                std::fprintf(stderr, " %u:%.0f", j, s_splitExec[j] / n);
                            std::fprintf(stderr, "\n");
                        }
                    }
                    if (end == ReplayEnd::ProgramEnded)
                    {
                        programEnded = true;
                        break;
                    }
                    if (end == ReplayEnd::Done)
                    {
                        chainBlock = block;
                        chainSchedule = foundIndex;
                    }
                    nativeBoundary = false;
                    nativeTail = false;
                    continue;
                }
            }
            const DecodedInstructionPair &decoded = m_decodedCodeCache[pairIndex];
            if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
            {
                fastLeft = true;
                break;
            }

            const uint64_t readyCycle = s_cachedReads
                ? fastReadyCycle(decoded, s_fastOps.reads[pairIndex])
                : calculatePairReadyCycle(decoded);
            uint64_t kickDataReady = 0;
            if (s_rec.block && decoded.lowerUsage.pipeline == PipelineXgkick)
            {
                // Replay re-checks the PATH1 wait itself and needs only the operand part.
                const bool kickActive = m_xgkick.active;
                m_xgkick.active = false;
                kickDataReady = fastReadyCycle(decoded, s_fastOps.reads[pairIndex]);
                m_xgkick.active = kickActive;
            }
            if (readyCycle > m_cycle)
            {
                if (readyCycle >= budgetEnd)
                {
                    fastAdvanceTo(budgetEnd);
                    break;
                }
                fastAdvanceTo(readyCycle);
            }
            if (decoded.lowerUsage.pipeline == PipelineXgkick)
            {
                while (m_xgkick.active && m_cycle < budgetEnd)
                    advanceOneCycle();
                if (m_xgkick.active)
                    break;
            }
            if (s_rec.block)
            {
                if (pairIndex != s_rec.block->first + s_rec.count || m_cycle - s_rec.entry > kMaxRecordedCycle)
                    s_rec.block = nullptr;
                else
                {
                    s_rec.issue[s_rec.count] = static_cast<uint16_t>(m_cycle - s_rec.entry);
                    s_rec.dataReady[s_rec.count] =
                        static_cast<uint16_t>(kickDataReady > s_rec.entry ? kickDataReady - s_rec.entry : 0u);
                    ++s_rec.count;
                }
            }
            if (g_vuVerifyTrace)
                recordVerifyPair(m_state, m_cycle, decoded.upperUsage, decoded.lowerUsage);

            if (g_vuMacOnlyFlags && s_fastOps.pairReadsFlags[pairIndex] != 0u)
                applyPendingMac(m_state.mac, m_cycle);
            execPair(decoded, s_fastOps.ops[pairIndex]);
            markWrites(pairIndex, decoded);
            resetZeroRegisters();
            programEnded = finishPc(decoded);
            endPairCycle();
            ++g_vu1PairTotal;
            nativeBoundary = nativeTail;
            nativeTail = (!decoded.iBit && decoded.lowerUsage.pipeline == PipelineBranch) || decoded.eBit;
            if (s_rec.block && s_rec.count == s_rec.block->length)
                finishRecording();
            if (programEnded)
                break;
        }
        s_rec.block = nullptr;
        // The interpreter loop continues after fastLeft and needs MAC values that
        // are not visible yet in the flag pipeline; otherwise run() flushes every
        // pipeline on exit and the latest MAC wins.
        applyPendingMac(m_state.mac, fastLeft ? m_cycle : std::numeric_limits<uint64_t>::max());
        for (; g_vuPendingMacCount != 0u; --g_vuPendingMacCount)
        {
            const PendingMac &pending = g_vuPendingMac[g_vuPendingMacHead];
            g_vuPendingMacHead = (g_vuPendingMacHead + 1u) % kPendingMacSlots;
            for (FlagPipelineEntry &entry : m_flagPipeline)
            {
                if (entry.valid)
                    continue;
                entry = {};
                entry.valid = true;
                entry.issueCycle = pending.readyCycle - kFmacLatency;
                entry.readyCycle = pending.readyCycle;
                noteQueued(this, kPipeFlag, m_flagPipeline, entry);
                entry.mac = pending.mac;
                entry.writesMac = true;
                break;
            }
        }
        g_vuMacOnlyFlags = false;
        g_vuSkipMacFlags = false;
        g_vuFmacLean = false;
        g_vuDirectStores = false;
        fastDrainCycle = std::max(writesReadyAt, g_vuSkippedFlagsReady);
        recompRan = !fastLeft;
        ++g_vuRecompRuns;
    }
}
