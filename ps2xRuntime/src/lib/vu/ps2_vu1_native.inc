// VU1 fast-path operations shared by the fast loop (ps2_vu1_fast_loop.inc)
// and by natively compiled VU1 blocks (ps2_vu1_native_blocks.inc).
// Included once in ps2_vu1_core.cpp, after the file-local helpers.
//
// VU1Native is a friend of VU1Interpreter. Its op classification and the
// fastUpper/fastLower bodies used to live inside VU1Interpreter::run; they
// moved here unchanged so generated code can instantiate them with every
// field folded at compile time.

extern uint64_t g_vu1PairTotal;
// Pairs executed by natively compiled blocks (subset of g_vu1PairTotal).
inline uint64_t g_vu1NativePairs = 0;

struct VU1Native
{
    using Interp = VU1Interpreter;
    using DecodedInstructionPair = VU1Interpreter::DecodedInstructionPair;
    using InstructionUsage = VU1Interpreter::InstructionUsage;
    using VfAccess = VU1Interpreter::VfAccess;

    struct FastOp
    {
        uint8_t kind, fd, fs, ft, dest, bc;
        uint8_t lowerKind;
    };
    // Common lower ops handled inline; everything else goes to execLower.
    enum LowerKind : uint8_t
    {
        kLowerExec = 0,
        kLowerNop,
        kLq, kSq, kIaddiu, kLqi, kSqi, kIbeq, kIbne, kMr32, kMove,
        kIlw, kIsw, kIsubiu, kFcand, kFcor, kFcget, kB, kJr, kIbltz, kIbgtz, kIblez, kIbgez,
        kIadd, kIsub, kIaddi, kIand, kIor, kWaitq, kMtir, kMfir, kMfp,
        kDiv, kEleng,
        // Appended (generated blocks keep their packed kinds).
        kXgkick, kXtop, kXitop, kFcset, kWaitp, kBal, kJalr,
    };
    static constexpr uint8_t classifyLowerWord(uint32_t lower)
    {
        if (lower == 0x8000033Cu || lower == 0u)
            return kLowerNop;
        switch ((lower >> 25) & 0x7Fu)
        {
        case 0x00: return kLq;
        case 0x01: return kSq;
        case 0x04: return kIlw;
        case 0x05: return kIsw;
        case 0x08: return kIaddiu;
        case 0x09: return kIsubiu;
        case 0x11: return kFcset;
        case 0x12: return kFcand;
        case 0x13: return kFcor;
        case 0x1C: return kFcget;
        case 0x20: return kB;
        case 0x21: return kBal;
        case 0x24: return kJr;
        case 0x25: return kJalr;
        case 0x28: return kIbeq;
        case 0x29: return kIbne;
        case 0x2C: return kIbltz;
        case 0x2D: return kIbgtz;
        case 0x2E: return kIblez;
        case 0x2F: return kIbgez;
        case 0x40:
            if ((lower & 0x3Cu) == 0x3Cu)
            {
                switch ((lower & 0x3u) | ((lower >> 4) & 0x7Cu))
                {
                case 0x30: return kMove;
                case 0x31: return kMr32;
                case 0x34: return kLqi;
                case 0x35: return kSqi;
                case 0x38: return kDiv;
                case 0x3B: return kWaitq;
                case 0x3C: return kMtir;
                case 0x3D: return kMfir;
                case 0x64: return kMfp;
                case 0x68: return kXtop;
                case 0x69: return kXitop;
                case 0x6C: return kXgkick;
                case 0x72: return kEleng;
                case 0x7B: return kWaitp;
                default: break;
                }
                return kLowerExec;
            }
            switch (lower & 0x3Fu)
            {
            case 0x30: return kIadd;
            case 0x31: return kIsub;
            case 0x32: return kIaddi;
            case 0x34: return kIand;
            case 0x35: return kIor;
            default: break;
            }
            return kLowerExec;
        default:
            return kLowerExec;
        }
    }
    static uint8_t classifyLower(const DecodedInstructionPair &pair)
    {
        return pair.iBit ? kLowerExec : classifyLowerWord(pair.lower);
    }
    enum UpperKind : uint8_t
    {
        kExec = 0,
        kNop,
        kAddBc, kSubBc, kMaddBc, kMsubBc, kMaxBc, kMiniBc, kMulBc,
        kAdd, kSub, kMul, kMax, kMini,
        kMulQ, kMiniI,
        kMulaBc, kMaddaBc,
        kMulI, kAddI,
        kClip, kFtoi0, kFtoi4, kItof0,
        // Appended, so the packed ops in ps2_vu1_native_blocks.inc keep their kinds.
        kAddaBc, kMula, kSubI, kOpmula, kOpmsub, kItof12, kAbs,
        kSubaBc, kMsubaBc, kMadd, kMsub, kMaddQ, kMadda, kMaxI, kItof4, kFtoi12,
    };
    struct FastReads
    {
        // At most two VF operands in each half, with four lanes each. Unused
        // vf/vi slots stay 0 (vf0.x / vi0, never written, so never stall),
        // which lets the ready check scan fixed-length lists without branches.
        uint8_t vf[16]{}, vi[15]{}, acc[4]{};
        uint8_t vfCount = 0, viCount = 0, accCount = 0;
        // Pair waits on FDIV, EFU, P or a PATH1 transfer.
        bool scalarWait = false;
    };
    // markPairWrites and the drain-cycle bookkeeping, precomputed per pair.
    // Absent writes use register 0 with no lanes, which leaves the ready
    // tables unchanged.
    struct FastWrites
    {
        uint32_t upperLat = 0, lowerLat = 0, viLat = 0, readyOffset = 0;
        uint8_t upperReg = 0, upperLanes = 0, lowerReg = 0, lowerLanes = 0, viReg = 0, accLanes = 0;
        bool multiVi = false;
    };
    static FastWrites classifyWrites(const DecodedInstructionPair &pair)
    {
        FastWrites w{};
        const VfAccess upper = pair.upperUsage.vfWrite;
        const VfAccess lower = pair.lowerUsage.vfWrite;
        const bool hasUpper = upper.reg != 0u;
        const bool hasLower = lower.reg != 0u && pair.suppressedLowerVf != lower.reg;
        const bool distinctLower = hasLower && (!hasUpper || lower.reg != upper.reg);
        w.upperLat = pair.upperUsage.vfLatency != 0u ? pair.upperUsage.vfLatency : pair.upperUsage.latency;
        w.lowerLat = pair.lowerUsage.vfLatency != 0u ? pair.lowerUsage.vfLatency : pair.lowerUsage.latency;
        w.viLat = pair.lowerUsage.viLatency != 0u ? pair.lowerUsage.viLatency : pair.lowerUsage.latency;
        if (hasUpper)
        {
            w.upperReg = upper.reg;
            w.upperLanes = upper.lanes;
        }
        if (hasLower)
        {
            w.lowerReg = lower.reg;
            w.lowerLanes = lower.lanes;
        }
        const uint32_t viBits = pair.lowerUsage.viWrite & 0xFFFEu;
        if (viBits != 0u)
            w.viReg = static_cast<uint8_t>(std::countr_zero(viBits));
        w.multiVi = std::popcount(viBits) > 1;
        w.accLanes = pair.upperUsage.accWrite;
        if (hasUpper)
            w.readyOffset = std::max(w.readyOffset, w.upperLat);
        if (distinctLower)
            w.readyOffset = std::max(w.readyOffset, w.lowerLat);
        if (w.accLanes != 0u)
            w.readyOffset = std::max<uint32_t>(w.readyOffset, Interp::kAccForwardLatency);
        if (w.viReg != 0u)
            w.readyOffset = std::max(w.readyOffset, w.viLat);
        return w;
    }
    enum : uint8_t
    {
        kReadsStatus = 1,
        kReadsMac = 2,
    };
    static uint8_t lowerReadsFlags(const DecodedInstructionPair &pair)
    {
        if (pair.iBit)
            return 0u;
        switch ((pair.lower >> 25) & 0x7Fu)
        {
        case 0x14: case 0x16: case 0x17: // FSEQ, FSAND, FSOR
            return kReadsStatus;
        case 0x18: case 0x1A: case 0x1B: // FMEQ, FMAND, FMOR
            return kReadsMac;
        default:
            return 0u;
        }
    }
    static FastReads classifyReads(const DecodedInstructionPair &pair)
    {
        FastReads result{};
        uint8_t vfLanes[32]{};
        uint16_t viMask = 0;
        uint8_t accMask = 0;
        for (const InstructionUsage *usage : {&pair.upperUsage, &pair.lowerUsage})
        {
            for (uint32_t i = 0; i < usage->vfReadCount; ++i)
                vfLanes[usage->vfRead[i].reg] |= usage->vfRead[i].lanes;
            viMask |= usage->viRead;
            accMask |= usage->accRead;
        }
        // Preserve the reference scheduler's ready-lane indexing exactly:
        // ascending reg * 4 + lane, taken from a 128-bit slot mask.
        uint64_t slots[2] = {0u, 0u};
        for (uint32_t reg = 0; reg < 32; ++reg)
            if (vfLanes[reg] != 0u)
                slots[reg >> 4] |= static_cast<uint64_t>(vfLanes[reg] & 0xFu) << ((reg & 15u) * 4u);
        for (uint32_t half = 0; half < 2u; ++half)
            for (uint64_t bits = slots[half]; bits != 0u; bits &= bits - 1u)
                result.vf[result.vfCount++] = static_cast<uint8_t>(half * 64u + std::countr_zero(bits));
        for (uint32_t bits = viMask & 0xFFFEu; bits != 0u; bits &= bits - 1u)
            result.vi[result.viCount++] = static_cast<uint8_t>(std::countr_zero(bits));
        for (uint32_t lane = 0; lane < 4; ++lane)
            if (accMask & (1u << lane))
                result.acc[result.accCount++] = static_cast<uint8_t>(lane);
        for (uint32_t i = result.accCount; i < 4u; ++i)
            result.acc[i] = result.acc[0];
        const InstructionUsage &lower = pair.lowerUsage;
        result.scalarWait = lower.pipeline == Interp::PipelineFdiv || lower.waitQ || lower.pipeline == Interp::PipelineEfu ||
                            lower.waitP || lower.pipeline == Interp::PipelineXgkick;
        return result;
    }

    static constexpr FastOp classifyUpper(uint32_t up)
    {
        FastOp op{};
        op.dest = static_cast<uint8_t>((up >> 21) & 0xFu);
        op.ft = static_cast<uint8_t>((up >> 16) & 0x1Fu);
        op.fs = static_cast<uint8_t>((up >> 11) & 0x1Fu);
        op.fd = static_cast<uint8_t>((up >> 6) & 0x1Fu);
        const uint32_t low = up & 0x3Fu;
        op.bc = static_cast<uint8_t>(low & 3u);
        if (low <= 0x1Bu)
        {
            constexpr uint8_t kBcKinds[7] = {kAddBc, kSubBc, kMaddBc, kMsubBc, kMaxBc, kMiniBc, kMulBc};
            op.kind = kBcKinds[low >> 2];
        }
        else if (low == 0x1Cu)
            op.kind = kMulQ;
        else if (low == 0x1Du)
            op.kind = kMaxI;
        else if (low == 0x1Eu)
            op.kind = kMulI;
        else if (low == 0x1Fu)
            op.kind = kMiniI;
        else if (low == 0x21u)
            op.kind = kMaddQ;
        else if (low == 0x22u)
            op.kind = kAddI;
        else if (low == 0x26u)
            op.kind = kSubI;
        else if (low == 0x2Eu && (op.dest & 1u) == 0u) // OPMSUB; the interpreter's w is 0, not ACC.w - 0
            op.kind = kOpmsub;
        else if (low == 0x28u)
            op.kind = kAdd;
        else if (low == 0x29u)
            op.kind = kMadd;
        else if (low == 0x2Du)
            op.kind = kMsub;
        else if (low == 0x2Au)
            op.kind = kMul;
        else if (low == 0x2Bu)
            op.kind = kMax;
        else if (low == 0x2Cu)
            op.kind = kSub;
        else if (low == 0x2Fu)
            op.kind = kMini;
        else if (low >= 0x3Cu)
        {
            const uint32_t spec = (up & 3u) | ((up >> 4) & 0x7Cu);
            op.bc = static_cast<uint8_t>(spec & 3u);
            if (spec >= 0x18u && spec <= 0x1Bu)
                op.kind = kMulaBc;
            else if (spec >= 0x08u && spec <= 0x0Bu)
                op.kind = kMaddaBc;
            else if (spec == 0x2Fu)
                op.kind = kNop;
            else if (spec == 0x1Fu)
                op.kind = kClip;
            else if (spec == 0x14u)
                op.kind = kFtoi0;
            else if (spec == 0x15u)
                op.kind = kFtoi4;
            else if (spec == 0x10u)
                op.kind = kItof0;
            else if (spec <= 0x03u)
                op.kind = kAddaBc;
            else if (spec == 0x2Au)
                op.kind = kMula;
            else if (spec == 0x2Eu && (op.dest & 1u) == 0u) // OPMULA, as OPMSUB
                op.kind = kOpmula;
            else if (spec == 0x12u)
                op.kind = kItof12;
            else if (spec == 0x1Du)
                op.kind = kAbs;
            else if (spec >= 0x04u && spec <= 0x07u)
                op.kind = kSubaBc;
            else if (spec >= 0x0Cu && spec <= 0x0Fu)
                op.kind = kMsubaBc;
            else if (spec == 0x29u)
                op.kind = kMadda;
            else if (spec == 0x11u)
                op.kind = kItof4;
            else if (spec == 0x16u)
                op.kind = kFtoi12;
        }
        return op;
    }
    static constexpr uint64_t packFastOp(const FastOp &op)
    {
        return uint64_t{op.kind} | uint64_t{op.fd} << 8 | uint64_t{op.fs} << 16 | uint64_t{op.ft} << 24 |
               uint64_t{op.dest} << 32 | uint64_t{op.bc} << 40 | uint64_t{op.lowerKind} << 48;
    }
    static constexpr FastOp unpackFastOp(uint64_t packed)
    {
        FastOp op{};
        op.kind = static_cast<uint8_t>(packed);
        op.fd = static_cast<uint8_t>(packed >> 8);
        op.fs = static_cast<uint8_t>(packed >> 16);
        op.ft = static_cast<uint8_t>(packed >> 24);
        op.dest = static_cast<uint8_t>(packed >> 32);
        op.bc = static_cast<uint8_t>(packed >> 40);
        op.lowerKind = static_cast<uint8_t>(packed >> 48);
        return op;
    }
    // fastUpper takes a runtime FastOp or std::integral_constant<uint64_t,
    // packFastOp(op)>; the latter makes every field a compile-time constant.
    template <typename Arg>
    static constexpr FastOp asFastOp(const Arg &arg)
    {
        if constexpr (std::is_same_v<Arg, FastOp>)
            return arg;
        else
        {
            constexpr FastOp op = unpackFastOp(Arg::value);
            return op;
        }
    }

    static inline const bool s_fmacZero = std::getenv("ATV_VU_NO_FMAC_ZERO") == nullptr;
    // vuNormalizeOperand (and normalizeResult's value) on the bits, always inline. Only
    // for single-operation uses: in a longer float expression a different inlining can
    // change how /fp:fast contracts it.
    static __forceinline float normalizeBits(float value)
    {
        uint32_t bits = std::bit_cast<uint32_t>(value);
        const uint32_t exponent = bits & 0x7F800000u;
        if (exponent == 0u)
            bits &= 0x80000000u;
        else if (exponent == 0x7F800000u)
            bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
        return std::bit_cast<float>(bits);
    }

    // Cold parts of fastUpper's FMAC path, kept out of line so the hot path
    // needs fewer registers (no spills) in every folded op instance.
    static __declspec(noinline) void fmacStats(__m128 result, __m128 vs, __m128i handled, __m128i destMask,
                                               bool productSum)
    {
        ++g_vuFmacStat[kFmacStatOps];
        const int slowLanes = _mm_movemask_ps(_mm_castsi128_ps(_mm_andnot_si128(handled, destMask)));
        if (slowLanes != 0)
        {
            ++g_vuFmacStat[kFmacStatSlowOps];
            if (g_vuSkipMacFlags)
                ++g_vuFmacStat[kFmacStatSlowNoFlag];
            alignas(16) float r[4];
            alignas(16) float s[4];
            _mm_store_ps(r, result);
            _mm_store_ps(s, vs);
            for (uint32_t c = 0; c < 4u; ++c)
            {
                if (((slowLanes >> c) & 1) == 0)
                    continue;
                uint32_t bits = 0;
                std::memcpy(&bits, &r[c], 4);
                const uint32_t e = (bits >> 23) & 0xFFu;
                if ((bits & 0x7FFFFFFFu) == 0u)
                {
                    ++g_vuFmacStat[kFmacStatZero];
                    if (!productSum && s[c] == 0.0f)
                        ++g_vuFmacStat[kFmacStatZeroFactor];
                }
                else if (e == 0u)
                    ++g_vuFmacStat[kFmacStatDenormal];
                else if (e == 0xFFu)
                    ++g_vuFmacStat[kFmacStatInfNan];
                else if (productSum && e >= 3u && e <= 0xFCu)
                    ++g_vuFmacStat[kFmacStatProduct];
                else
                    ++g_vuFmacStat[kFmacStatEdge];
            }
        }
    }
    static __declspec(noinline) void fmacFlagsNormal(Interp &vu, uint8_t dest, __m128 result, __m128i exactZero)
    {
        const int signs = _mm_movemask_ps(result);
        const int zeros = _mm_movemask_ps(_mm_castsi128_ps(exactZero));
        uint8_t laneFlags[4];
        for (uint32_t c = 0; c < 4u; ++c)
            laneFlags[c] = (dest & laneForComponent(c)) == 0u
                               ? 0u
                               : static_cast<uint8_t>((((signs >> c) & 1) ? 0x2u : 0u) |
                                                      (((zeros >> c) & 1) ? 0x1u : 0u));
        vu.updateFmacFlags(laneFlags, dest, vu.calculateFmacProductSticky(dest));
    }
    static __declspec(noinline) __m128 fmacSlow(Interp &vu, uint32_t upper, uint8_t dest, __m128 result,
                                                __m128i exactZero)
    {
        alignas(16) float lanes[4];
        _mm_store_ps(lanes, result);
        uint8_t laneFlags[4]{};
        vu.normalizeFmacResult(lanes, dest, laneFlags);
        if (g_vuVerifyFast)
        {
            const int zeros = _mm_movemask_ps(_mm_castsi128_ps(exactZero));
            const int signs = _mm_movemask_ps(result);
            for (uint32_t c = 0; c < 4u; ++c)
            {
                if ((dest & laneForComponent(c)) == 0u || ((zeros >> c) & 1) == 0)
                    continue;
                const uint8_t fastFlags = static_cast<uint8_t>(0x1u | (((signs >> c) & 1) ? 0x2u : 0u));
                uint32_t slowBits = 0;
                std::memcpy(&slowBits, &lanes[c], 4);
                if ((laneFlags[c] != fastFlags || slowBits != (((signs >> c) & 1) ? 0x80000000u : 0u)) &&
                    g_vuVerifyFastReports < 16u)
                {
                    ++g_vuVerifyFastReports;
                    std::fprintf(stderr, "[vu-verify] fmac zero mismatch upper=%08x lane=%u slow=%x fast=%x value=%08x\n",
                                 upper, c, laneFlags[c], fastFlags, slowBits);
                }
            }
        }
        if (g_vuMacOnlyFlags)
            vu.updateFmacFlags(laneFlags, dest, 0u);
        else if (!g_vuSkipMacFlags)
            vu.updateFmacFlags(laneFlags, dest, vu.calculateFmacProductSticky(dest));
        result = _mm_load_ps(lanes);
        return result;
    }

    // Lean = g_vuFmacLean holds: no status flags (at most MAC), so an FMAC
    // result only needs fmacSlow when a lane is not provably normal or exactly zero.
    // NoMac: merge_blocks.py proved no instruction can read this op's MAC
    // before a later FMAC replaces it, so the MAC-only push is skipped.
    template <bool Lean = false, bool NoMac = false, typename OpArg>
    static __forceinline void fastUpper(Interp &vu, const OpArg &opArg, uint32_t upper)
    {
        const FastOp op = asFastOp(opArg);
        if (op.kind == kExec)
        {
            vu.execUpper(upper);
            return;
        }
        VU1State &m_state = vu.m_state;
        vu.m_currentUpperInstruction = upper;
        switch (op.kind)
        {
        case kNop:
            return;
        case kClip:
        {
            uint32_t wBits = 0u;
            std::memcpy(&wBits, &m_state.vf[op.ft][3], sizeof(wBits));
            const int32_t limit =
                (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
            uint32_t bits[3];
            std::memcpy(bits, m_state.vf[op.fs], sizeof(bits));
            uint32_t flags = 0u;
            for (uint32_t c = 0; c < 3u; ++c)
            {
                if (static_cast<int32_t>(bits[c]) > limit)
                    flags |= 1u << (2u * c);
                if (static_cast<int32_t>(bits[c] ^ 0x80000000u) > limit)
                    flags |= 2u << (2u * c);
            }
            vu.queueClip(flags);
            return;
        }
        case kFtoi0:
        case kFtoi4:
        case kFtoi12:
        {
            const double scale = op.kind == kFtoi0 ? 1.0 : op.kind == kFtoi4 ? 16.0 : 4096.0;
            float result[4];
            for (uint32_t c = 0; c < 4u; ++c)
            {
                const double scaled = static_cast<double>(vuNormalizeOperand(m_state.vf[op.fs][c])) * scale;
                int32_t value;
                if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
                    value = std::numeric_limits<int32_t>::max();
                else if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
                    value = std::numeric_limits<int32_t>::min();
                else
                    value = static_cast<int32_t>(scaled);
                std::memcpy(&result[c], &value, sizeof(value));
            }
            vu.applyDest(m_state.vf[op.ft], result, op.dest);
            return;
        }
        case kItof0:
        case kItof4:
        case kItof12:
        {
            const float scale = op.kind == kItof0 ? 1.0f : op.kind == kItof4 ? 16.0f : 4096.0f;
            float result[4];
            for (uint32_t c = 0; c < 4u; ++c)
            {
                int32_t value;
                std::memcpy(&value, &m_state.vf[op.fs][c], sizeof(value));
                result[c] = static_cast<float>(value) / scale;
            }
            vu.applyDest(m_state.vf[op.ft], result, op.dest);
            return;
        }
        case kAbs:
        {
            alignas(16) float result[4];
            _mm_store_ps(result, _mm_andnot_ps(_mm_set1_ps(-0.0f), vuNormalize4(_mm_loadu_ps(m_state.vf[op.fs]))));
            vu.applyDest(m_state.vf[op.ft], result, op.dest);
            return;
        }
        default:
            break;
        }
        // The cross products (OPMULA, OPMSUB) take fs.yzx and ft.zxy, so the zero
        // and normal-range proofs below see the operands of each lane.
        __m128 vs = vuNormalize4(_mm_loadu_ps(m_state.vf[op.fs]));
        const __m128 vt = vuNormalize4(_mm_loadu_ps(m_state.vf[op.ft]));
        const auto broadcast = [&]() {
            switch (op.bc)
            {
            case 0: return _mm_shuffle_ps(vt, vt, 0x00);
            case 1: return _mm_shuffle_ps(vt, vt, 0x55);
            case 2: return _mm_shuffle_ps(vt, vt, 0xAA);
            default: return _mm_shuffle_ps(vt, vt, 0xFF);
            }
        };
        const auto acc = [&]() { return vuNormalize4(_mm_loadu_ps(m_state.acc)); };
        __m128 result;
        __m128 right = vt;
        bool fmac = true;
        bool toAcc = false;
        bool productSum = false;
        // How a +-0 result lane is proven exactly zero: vs == -right (add),
        // vs == right (sub), or a zero factor (mul, product-sum).
        enum : uint8_t { kZeroAdd, kZeroSub, kZeroFactor } zeroRule = kZeroFactor;
        switch (op.kind)
        {
        case kAddBc: right = broadcast(); result = _mm_add_ps(vs, right); zeroRule = kZeroAdd; break;
        case kSubBc: right = broadcast(); result = _mm_sub_ps(vs, right); zeroRule = kZeroSub; break;
        case kMaddBc: right = broadcast(); result = _mm_add_ps(acc(), _mm_mul_ps(vs, right)); productSum = true; break;
        case kMsubBc: right = broadcast(); result = _mm_sub_ps(acc(), _mm_mul_ps(vs, right)); productSum = true; break;
        case kMaxBc: result = _mm_max_ps(vs, broadcast()); fmac = false; break;
        case kMiniBc: result = _mm_min_ps(vs, broadcast()); fmac = false; break;
        case kMulBc: right = broadcast(); result = _mm_mul_ps(vs, right); break;
        case kAdd: result = _mm_add_ps(vs, vt); zeroRule = kZeroAdd; break;
        case kSub: result = _mm_sub_ps(vs, vt); zeroRule = kZeroSub; break;
        case kMul: result = _mm_mul_ps(vs, vt); break;
        case kMax: result = _mm_max_ps(vs, vt); fmac = false; break;
        case kMini: result = _mm_min_ps(vs, vt); fmac = false; break;
        case kMulQ:
            if (vu.m_fdiv.valid && vu.m_fdiv.readyCycle <= vu.m_cycle) // lazy Q (g_vuLazyQ)
            {
                m_state.q = vu.m_fdiv.value;
                const uint32_t currentDi = vu.m_fdiv.statusDi & 0x30u;
                m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
                vu.m_fdiv = {};
            }
            right = vuNormalize4(_mm_set1_ps(m_state.q));
            result = _mm_mul_ps(vs, right);
            break;
        case kMiniI: result = _mm_min_ps(vs, vuNormalize4(_mm_set1_ps(m_state.i))); fmac = false; break;
        case kMaxI: result = _mm_max_ps(vs, vuNormalize4(_mm_set1_ps(m_state.i))); fmac = false; break;
        case kMulI: right = vuNormalize4(_mm_set1_ps(m_state.i)); result = _mm_mul_ps(vs, right); break;
        case kAddI: right = vuNormalize4(_mm_set1_ps(m_state.i)); result = _mm_add_ps(vs, right); zeroRule = kZeroAdd; break;
        case kMulaBc: right = broadcast(); result = _mm_mul_ps(vs, right); toAcc = true; break;
        case kMaddaBc: right = broadcast(); result = _mm_add_ps(acc(), _mm_mul_ps(vs, right)); toAcc = true; productSum = true; break;
        case kAddaBc: right = broadcast(); result = _mm_add_ps(vs, right); toAcc = true; zeroRule = kZeroAdd; break;
        case kMula: result = _mm_mul_ps(vs, vt); toAcc = true; break;
        case kSubI: right = vuNormalize4(_mm_set1_ps(m_state.i)); result = _mm_sub_ps(vs, right); zeroRule = kZeroSub; break;
        case kSubaBc: right = broadcast(); result = _mm_sub_ps(vs, right); toAcc = true; zeroRule = kZeroSub; break;
        case kMsubaBc: right = broadcast(); result = _mm_sub_ps(acc(), _mm_mul_ps(vs, right)); toAcc = true; productSum = true; break;
        case kMadd: result = _mm_add_ps(acc(), _mm_mul_ps(vs, vt)); productSum = true; break;
        case kMsub: result = _mm_sub_ps(acc(), _mm_mul_ps(vs, vt)); productSum = true; break;
        case kMadda: result = _mm_add_ps(acc(), _mm_mul_ps(vs, vt)); toAcc = true; productSum = true; break;
        case kMaddQ:
            if (vu.m_fdiv.valid && vu.m_fdiv.readyCycle <= vu.m_cycle) // lazy Q (g_vuLazyQ), as kMulQ
            {
                m_state.q = vu.m_fdiv.value;
                const uint32_t currentDi = vu.m_fdiv.statusDi & 0x30u;
                m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
                vu.m_fdiv = {};
            }
            right = vuNormalize4(_mm_set1_ps(m_state.q));
            result = _mm_add_ps(acc(), _mm_mul_ps(vs, right));
            productSum = true;
            break;
        case kOpmula:
        case kOpmsub:
            vs = _mm_shuffle_ps(vs, vs, _MM_SHUFFLE(3, 0, 2, 1));
            right = _mm_shuffle_ps(vt, vt, _MM_SHUFFLE(3, 1, 0, 2));
            result = _mm_mul_ps(vs, right);
            if (op.kind == kOpmula)
                toAcc = true;
            else
            {
                result = _mm_sub_ps(acc(), result);
                productSum = true;
            }
            break;
        default:
            vu.execUpper(upper);
            return;
        }
        const __m128i destMask = vuDestMask(op.dest);
        if constexpr (Lean)
        {
            if (fmac)
            {
                // The checks of the general path below, without its modes.
                const __m128i exponent = vuExponent4(result);
                __m128i normal;
                if (!productSum)
                    normal = vuInRange4(exponent, 2, 0xFD);
                else
                {
                    const __m128 zero = _mm_setzero_ps();
                    const __m128i zeroFactor =
                        _mm_castps_si128(_mm_or_ps(_mm_cmpeq_ps(vs, zero), _mm_cmpeq_ps(right, zero)));
                    const __m128i productExponent = vuExponent4(_mm_mul_ps(vs, right));
                    const __m128i productNormal =
                        _mm_and_si128(vuInRange4(productExponent, 1, 0xFD),
                                      _mm_cmpgt_epi32(_mm_add_epi32(exponent, _mm_set1_epi32(21)), productExponent));
                    normal = _mm_and_si128(vuInRange4(exponent, 3, 0xFC), _mm_or_si128(zeroFactor, productNormal));
                }
                const __m128 zero = _mm_setzero_ps();
                __m128 proof;
                if (zeroRule == kZeroAdd)
                    proof = _mm_cmpeq_ps(vs, _mm_xor_ps(right, _mm_set1_ps(-0.0f)));
                else if (zeroRule == kZeroSub)
                    proof = _mm_cmpeq_ps(vs, right);
                else
                    proof = _mm_or_ps(_mm_cmpeq_ps(vs, zero), _mm_cmpeq_ps(right, zero));
                const __m128i exactZero = _mm_castps_si128(_mm_and_ps(_mm_cmpeq_ps(result, zero), proof));
                const __m128i handled = _mm_or_si128(normal, exactZero);
                if (_mm_movemask_ps(_mm_castsi128_ps(_mm_andnot_si128(handled, destMask))) != 0)
                    result = fmacSlow(vu, upper, op.dest, result, exactZero);
                else if (!NoMac && g_vuMacOnlyFlags && op.dest != 0u)
                {
                    static constexpr uint8_t kComponentsToLanes[16] = {0, 8, 4, 12, 2, 10, 6, 14,
                                                                       1, 9, 5, 13, 3, 11, 7, 15};
                    pushPendingMac(
                        m_state.mac, vu.m_cycle + Interp::kFmacLatency,
                        (static_cast<uint32_t>(kComponentsToLanes[_mm_movemask_ps(result)] & op.dest) << 4) |
                            (kComponentsToLanes[_mm_movemask_ps(_mm_castsi128_ps(exactZero))] & op.dest));
                }
            }
        }
        else if (fmac && g_vuApprox && (g_vuSkipMacFlags || g_vuMacOnlyFlags))
        {
            // ATV_VU_APPROX (microVU-style): FTZ/DAZ are set for the run, so
            // denormal results are already +-0; +-Inf/NaN clamp to +-max. No
            // exact-result proofs. MAC, when the program reads it, comes from
            // the result's sign and zero lanes.
            const __m128 maxV = _mm_set1_ps(std::numeric_limits<float>::max());
            result = _mm_max_ps(_mm_min_ps(result, maxV), _mm_xor_ps(maxV, _mm_set1_ps(-0.0f)));
            if (g_vuMacOnlyFlags && op.dest != 0u)
            {
                static constexpr uint8_t kComponentsToLanes[16] = {0, 8, 4, 12, 2, 10, 6, 14,
                                                                   1, 9, 5, 13, 3, 11, 7, 15};
                const int zeros = _mm_movemask_ps(_mm_cmpeq_ps(result, _mm_setzero_ps()));
                pushPendingMac(m_state.mac, vu.m_cycle + Interp::kFmacLatency,
                               (static_cast<uint32_t>(kComponentsToLanes[_mm_movemask_ps(result)] & op.dest) << 4) |
                                   (kComponentsToLanes[zeros] & op.dest));
            }
        }
        else if (fmac)
        {
            // Lanes whose result normalizeFmacResult would provably leave
            // unchanged with only the sign flag (same bounds as there).
            const __m128i exponent = vuExponent4(result);
            __m128i normal;
            if (!productSum)
                normal = vuInRange4(exponent, 2, 0xFD);
            else
            {
                const __m128 zero = _mm_setzero_ps();
                const __m128i zeroFactor =
                    _mm_castps_si128(_mm_or_ps(_mm_cmpeq_ps(vs, zero), _mm_cmpeq_ps(right, zero)));
                const __m128i productExponent = vuExponent4(_mm_mul_ps(vs, right));
                const __m128i productNormal =
                    _mm_and_si128(vuInRange4(productExponent, 1, 0xFD),
                                  _mm_cmpgt_epi32(_mm_add_epi32(exponent, _mm_set1_epi32(21)), productExponent));
                normal = _mm_and_si128(vuInRange4(exponent, 3, 0xFC), _mm_or_si128(zeroFactor, productNormal));
            }
            // +-0 lanes whose exact value is provably zero: the slow path
            // keeps the value and reports Z, plus S from the sign the exact
            // zero shares with the float result.
            __m128i exactZero = _mm_setzero_si128();
            if (s_fmacZero)
            {
                const __m128 zero = _mm_setzero_ps();
                __m128 proof;
                if (zeroRule == kZeroAdd)
                    proof = _mm_cmpeq_ps(vs, _mm_xor_ps(right, _mm_set1_ps(-0.0f)));
                else if (zeroRule == kZeroSub)
                    proof = _mm_cmpeq_ps(vs, right);
                else
                    proof = _mm_or_ps(_mm_cmpeq_ps(vs, zero), _mm_cmpeq_ps(right, zero));
                exactZero = _mm_castps_si128(_mm_and_ps(_mm_cmpeq_ps(result, zero), proof));
            }
            const __m128i handled = _mm_or_si128(normal, exactZero);
            const bool allNormal =
                _mm_movemask_ps(_mm_castsi128_ps(_mm_andnot_si128(handled, destMask))) == 0 && !g_vuVerifyFast;
            if (g_vuFmacStats)
                fmacStats(result, vs, handled, destMask, productSum);
            if (allNormal)
            {
                if (g_vuMacOnlyFlags)
                {
                    // Only Z and S can be set; MAC holds Z in bits 0..3 and S in
                    // bits 4..7, x as the high bit of each nibble.
                    static constexpr uint8_t kComponentsToLanes[16] = {0, 8, 4, 12, 2, 10, 6, 14,
                                                                       1, 9, 5, 13, 3, 11, 7, 15};
                    if (op.dest != 0u)
                        pushPendingMac(
                            m_state.mac, vu.m_cycle + Interp::kFmacLatency,
                            (static_cast<uint32_t>(kComponentsToLanes[_mm_movemask_ps(result)] & op.dest) << 4) |
                                (kComponentsToLanes[_mm_movemask_ps(_mm_castsi128_ps(exactZero))] & op.dest));
                }
                else if (!g_vuSkipMacFlags)
                    fmacFlagsNormal(vu, op.dest, result, exactZero);
            }
            else
                result = fmacSlow(vu, upper, op.dest, result, exactZero);
        }
        float *dst = toAcc ? m_state.acc : m_state.vf[op.fd];
        const __m128 keep = _mm_castsi128_ps(destMask);
        _mm_storeu_ps(dst, _mm_or_ps(_mm_and_ps(keep, result), _mm_andnot_ps(keep, _mm_loadu_ps(dst))));
    }

    // queueStore with g_vuDirectStores (always set while the fast loop runs):
    // the masked 16-byte write, without the call.
    static __forceinline void storeQword(Interp &vu, uint32_t address, const uint32_t words[4], uint8_t laneMask)
    {
        if (!g_vuDirectStores)
        {
            vu.queueStore(address, words, laneMask);
            return;
        }
        if (vu.m_activeVuData && address + 16u <= vu.m_activeVuDataSize)
        {
            uint8_t *dst = vu.m_activeVuData + address;
            const __m128 keep = _mm_castsi128_ps(vuDestMask(laneMask));
            const __m128 value = _mm_loadu_ps(reinterpret_cast<const float *>(words));
            _mm_storeu_ps(reinterpret_cast<float *>(dst),
                          _mm_or_ps(_mm_and_ps(keep, value),
                                    _mm_andnot_ps(keep, _mm_loadu_ps(reinterpret_cast<const float *>(dst)))));
        }
    }

    // Same semantics as the matching execLower cases. kind/instr may be
    // std::integral_constant (native blocks) to fold the decode.
    template <typename KindArg, typename InstrArg>
    static __forceinline void fastLower(Interp &vu, KindArg kindArg, InstrArg instrArg, uint32_t upper,
                                        uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory)
    {
        const uint8_t kind = kindArg;
        const uint32_t instr = instrArg;
        VU1State &m_state = vu.m_state;
        switch (kind)
        {
        case kLowerNop:
            return;
        case kLq:
        {
            uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(m_state.vi[VIS(instr)] + IMM11(instr))) * 16u;
            addr &= (dataSize - 1);
            if (addr + 16 <= dataSize)
            {
                float tmp[4];
                std::memcpy(tmp, vuData + addr, 16);
                vu.applyDest(m_state.vf[FT(instr)], tmp, DEST(instr));
            }
            return;
        }
        case kSq:
        {
            uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(m_state.vi[VIT(instr)] + IMM11(instr))) * 16u;
            addr &= (dataSize - 1);
            if (addr + 16 <= dataSize)
            {
                uint32_t words[4];
                std::memcpy(words, m_state.vf[FS(instr)], sizeof(words));
                storeQword(vu, addr, words, DEST(instr));
            }
            return;
        }
        case kIaddiu:
        {
            const uint8_t it = VIT(instr);
            const int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
            if (it != 0)
                m_state.vi[it] = (int16_t)(m_state.vi[VIS(instr)] + imm);
            return;
        }
        case kLqi:
        {
            const uint8_t viS = VIS(instr);
            uint32_t addr = static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[viS])) * 16u;
            addr &= (dataSize - 1);
            if (addr + 16 <= dataSize)
            {
                float tmp[4];
                std::memcpy(tmp, vuData + addr, 16);
                vu.applyDest(m_state.vf[FT(instr)], tmp, DEST(instr));
            }
            if (viS != 0)
                m_state.vi[viS] = (int16_t)(m_state.vi[viS] + 1);
            return;
        }
        case kSqi:
        {
            const uint8_t viT = VIT(instr);
            uint32_t addr = static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[viT])) * 16u;
            addr &= (dataSize - 1);
            if (addr + 16 <= dataSize)
            {
                uint32_t words[4];
                std::memcpy(words, m_state.vf[FS(instr)], sizeof(words));
                storeQword(vu, addr, words, DEST(instr));
            }
            if (viT != 0)
                m_state.vi[viT] = (int16_t)(m_state.vi[viT] + 1);
            return;
        }
        case kIbeq:
        case kIbne:
        {
            const bool equal = (int16_t)vu.readBranchVi(VIS(instr)) == (int16_t)vu.readBranchVi(VIT(instr));
            if (equal == (kind == kIbeq))
            {
                m_state.branchPending = true;
                m_state.branchTarget = (m_state.pc + 8 + IMM11(instr) * 8) & vu.microAddressMask();
                m_state.branchDelay = 1;
            }
            return;
        }
        case kMr32:
        {
            const float *s = m_state.vf[FS(instr)];
            const float tmp[4] = {s[1], s[2], s[3], s[0]};
            vu.applyDest(m_state.vf[FT(instr)], tmp, DEST(instr));
            return;
        }
        case kMove:
        {
            float tmp[4];
            std::memcpy(tmp, m_state.vf[FS(instr)], 16);
            vu.applyDest(m_state.vf[FT(instr)], tmp, DEST(instr));
            return;
        }
        case kIlw:
        {
            uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(m_state.vi[VIS(instr)] + IMM11(instr))) * 16u;
            addr &= (dataSize - 1);
            if (addr + 16 <= dataSize)
            {
                const uint8_t dest = DEST(instr);
                const uint32_t comp = (dest & 0x8u) ? 0u : (dest & 0x4u) ? 1u : (dest & 0x2u) ? 2u : 3u;
                uint32_t v;
                std::memcpy(&v, vuData + addr + comp * 4u, 4);
                if (VIT(instr) != 0)
                    m_state.vi[VIT(instr)] = static_cast<int16_t>(v & 0xFFFFu);
            }
            return;
        }
        case kIsw:
        {
            uint32_t addr = static_cast<uint32_t>(static_cast<int32_t>(m_state.vi[VIS(instr)] + IMM11(instr))) * 16u;
            addr &= (dataSize - 1);
            if (addr + 16 <= dataSize)
            {
                const uint32_t val = static_cast<uint16_t>(m_state.vi[VIT(instr)] & 0xFFFF);
                const uint32_t words[4] = {val, val, val, val};
                storeQword(vu, addr, words, DEST(instr));
            }
            return;
        }
        case kIsubiu:
        {
            const uint8_t it = VIT(instr);
            const int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
            if (it != 0)
                m_state.vi[it] = (int16_t)(m_state.vi[VIS(instr)] - imm);
            return;
        }
        case kFcand:
            if (g_vuPendingClipCount != 0u)
                applyPendingClip(m_state.clip, vu.m_cycle);
            m_state.vi[1] = (m_state.clip & instr & 0xFFFFFFu) != 0u ? 1 : 0;
            return;
        case kFcor:
            if (g_vuPendingClipCount != 0u)
                applyPendingClip(m_state.clip, vu.m_cycle);
            m_state.vi[1] = (m_state.clip | (instr & 0xFFFFFFu)) == 0xFFFFFFu ? 1 : 0;
            return;
        case kFcget:
            if (g_vuPendingClipCount != 0u)
                applyPendingClip(m_state.clip, vu.m_cycle);
            if (VIT(instr) != 0)
                m_state.vi[VIT(instr)] = static_cast<int16_t>(m_state.clip & 0xFFFFu);
            return;
        case kB:
            m_state.branchPending = true;
            m_state.branchTarget = (m_state.pc + 8 + IMM11(instr) * 8) & vu.microAddressMask();
            m_state.branchDelay = 1;
            return;
        case kJr: // reads the current VI (no IBxx-style pre-write backup), as PCSX2 microVU
            m_state.branchPending = true;
            m_state.branchTarget =
                (static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[VIS(instr)])) * 8u) & vu.microAddressMask();
            m_state.branchDelay = 1;
            return;
        case kBal: // execLower without its first-calls log line
        case kJalr:
        {
            const uint32_t target = kind == kBal ? m_state.pc + 8 + IMM11(instr) * 8
                                                 : static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[VIS(instr)])) * 8u;
            if (VIT(instr) != 0)
                m_state.vi[VIT(instr)] = static_cast<int32_t>((m_state.pc + 16) / 8);
            m_state.branchPending = true;
            m_state.branchTarget = target & vu.microAddressMask();
            m_state.branchDelay = 1;
            return;
        }
        case kIbltz:
        case kIbgtz:
        case kIblez:
        case kIbgez:
        {
            const int16_t value = (int16_t)vu.readBranchVi(VIS(instr));
            const bool taken = kind == kIbltz   ? value < 0
                               : kind == kIbgtz ? value > 0
                               : kind == kIblez ? value <= 0
                                                : value >= 0;
            if (taken)
            {
                m_state.branchPending = true;
                m_state.branchTarget = (m_state.pc + 8 + IMM11(instr) * 8) & vu.microAddressMask();
                m_state.branchDelay = 1;
            }
            return;
        }
        case kIadd:
            if (VID(instr) != 0)
                m_state.vi[VID(instr)] = (int16_t)(m_state.vi[VIS(instr)] + m_state.vi[VIT(instr)]);
            return;
        case kIsub:
            if (VID(instr) != 0)
                m_state.vi[VID(instr)] = (int16_t)(m_state.vi[VIS(instr)] - m_state.vi[VIT(instr)]);
            return;
        case kIaddi:
            if (VIT(instr) != 0)
                m_state.vi[VIT(instr)] =
                    (int16_t)(m_state.vi[VIS(instr)] + (int16_t)((int32_t)((instr >> 6) & 0x1F) << 27 >> 27));
            return;
        case kIand:
            if (VID(instr) != 0)
                m_state.vi[VID(instr)] = (int16_t)(m_state.vi[VIS(instr)] & m_state.vi[VIT(instr)]);
            return;
        case kIor:
            if (VID(instr) != 0)
                m_state.vi[VID(instr)] = (int16_t)(m_state.vi[VIS(instr)] | m_state.vi[VIT(instr)]);
            return;
        case kWaitq:
            return;
        case kMtir:
        {
            uint32_t bits;
            std::memcpy(&bits, &m_state.vf[FS(instr)][(instr >> 21) & 0x3u], 4);
            if (VIT(instr) != 0)
                m_state.vi[VIT(instr)] = static_cast<int16_t>(bits & 0xFFFFu);
            return;
        }
        case kMfir:
        {
            const int32_t value = static_cast<int16_t>(m_state.vi[VIS(instr)] & 0xFFFF);
            float result[4];
            std::memcpy(&result[0], &value, 4);
            result[1] = result[0];
            result[2] = result[0];
            result[3] = result[0];
            vu.applyDest(m_state.vf[FT(instr)], result, DEST(instr));
            return;
        }
        case kMfp:
        {
            const float result[4] = {m_state.p, m_state.p, m_state.p, m_state.p};
            vu.applyDest(m_state.vf[FT(instr)], result, DEST(instr));
            return;
        }
        case kDiv:
        {
            const float num = normalizeBits(m_state.vf[FS(instr)][(instr >> 21) & 0x3u]);
            const float den = normalizeBits(m_state.vf[FT(instr)][(instr >> 23) & 0x3u]);
            uint32_t statusDi = 0u;
            float result;
            if (den == 0.0f)
            {
                statusDi = num == 0.0f ? 0x10u : 0x20u;
                result = std::signbit(num) != std::signbit(den) ? -std::numeric_limits<float>::max()
                                                                : std::numeric_limits<float>::max();
            }
            else
                result = num / den;
            vu.queueQ(normalizeBits(result), 7u, statusDi); // normalizeResult's value
            return;
        }
        case kEleng:
        {
            const float x = vuNormalizeOperand(m_state.vf[FS(instr)][0]);
            const float y = vuNormalizeOperand(m_state.vf[FS(instr)][1]);
            const float z = vuNormalizeOperand(m_state.vf[FS(instr)][2]);
            vu.queueP(std::sqrt(x * x + y * y + z * z), 18u);
            return;
        }
        // execLower without its first-calls log lines.
        case kXgkick:
            vu.startXgkick(static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[VIS(instr)])));
            return;
        case kXtop:
            if (VIT(instr) != 0)
                m_state.vi[VIT(instr)] = static_cast<int32_t>(m_state.top & 0x3FFu);
            return;
        case kXitop:
            if (VIT(instr) != 0)
                m_state.vi[VIT(instr)] = static_cast<int32_t>(m_state.itop & 0x3FFu);
            return;
        case kFcset:
            vu.queueFcset(instr & 0xFFFFFFu);
            return;
        case kWaitp:
            return;
        default:
            vu.execLower(instr, vuData, dataSize, gs, memory, upper);
            return;
        }
    }

    // ---- Natively compiled blocks -----------------------------------------
    //
    // A block starts at any pair boundary with no pending branch, E-bit or
    // halt, and runs straight-line pairs up to and including the delay slot
    // of its first branch or E-bit pair. Each pair does exactly what one
    // iteration of the fast loop's per-pair path does (same ready checks,
    // stalls, exec, ready-table marks, PC handling and cycle end), with the
    // decode, register indices and ready-table slots folded in. Blocks are
    // generated from the decoded code cache (ATV_VU_NATIVE_GEN) and selected
    // by their exact code words, so any other code runs through the fast loop.
    struct Ctx
    {
        Interp *vu;
        uint8_t *vuData;
        uint32_t dataSize;
        GS *gs;
        PS2Memory *memory;
        uint64_t budgetEnd;
        uint64_t *writesReadyAt;
        CommitHint *hint;
        // Interpreter state for generated code, which is not a friend.
        uint64_t *cycle;
        uint64_t *vfReady; // m_vfReady flattened: reg * 4 + lane bit
        uint64_t *viReady;
        uint64_t *accReady;
        bool *xgkickActive;
        bool *viBackupValid;
    };
    enum : uint32_t
    {
        kNativeContinue = 0, // block done; m_state.pc is the next pair
        kNativeBail,         // stopped before the pair at m_state.pc
        kNativeEnded,        // program ended (E-bit delay slot)
        kNativePartDone,     // internal: a block part ran to its end
    };
    using BlockFn = uint32_t (*)(Ctx &);
    struct BlockEntry
    {
        uint32_t pc;
        uint32_t pairs;
        const uint32_t *words;
        BlockFn fn;
    };
    // Defined in ps2_vu1_native_blocks.inc (generated).
    static const BlockEntry *registry(size_t &count);

    static inline const bool s_enabled = std::getenv("ATV_VU_NO_NATIVE_BLOCKS") == nullptr;
    static inline const char *const s_genPath = std::getenv("ATV_VU_NATIVE_GEN");
    // ATV_VU_NATIVE_REGEN=1 with ATV_VU_NATIVE_GEN: ignore the compiled blocks, so every block that
    // runs is generated again (after the fast loop learns new ops; merge the new chunks first).
    static inline const bool s_regen = s_genPath != nullptr && std::getenv("ATV_VU_NATIVE_REGEN") != nullptr;
    // Same knob and test as fastAdvanceTo in the fast loop.
    static inline const bool s_idleSkip = std::getenv("ATV_VU_NO_IDLE_SKIP") == nullptr;
    // ATV_VU_NO_INLINE_UPPER=1: merged blocks call the shared upper ops (upperOpGeneral)
    // instead of their inlined lean forms.
    static inline const bool s_inlineUpper = std::getenv("ATV_VU_NO_INLINE_UPPER") == nullptr;
    // Whether a merged block part takes its inlined lean upper ops.
    static __forceinline bool leanInline() { return g_vuFmacLean && s_inlineUpper; }

    static void initCtx(Ctx &c, Interp &vu, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory,
                        uint64_t budgetEnd, uint64_t *writesReadyAt, const CommitHint &hint)
    {
        c.vu = &vu;
        c.vuData = vuData;
        c.dataSize = dataSize;
        c.gs = &gs;
        c.memory = memory;
        c.budgetEnd = budgetEnd;
        c.writesReadyAt = writesReadyAt;
        c.hint = const_cast<CommitHint *>(&hint);
        c.cycle = &vu.m_cycle;
        c.vfReady = vu.m_vfReady[0].data();
        c.viReady = vu.m_viReady.data();
        c.accReady = vu.m_accReady.data();
        c.xgkickActive = &vu.m_xgkick.active;
        c.viBackupValid = &vu.m_viBranchBackupValid;
    }

    // fastAdvanceTo.
    static __declspec(noinline) void advanceTo(Ctx &c, uint64_t target)
    {
        Interp &vu = *c.vu;
        while (vu.m_cycle < target)
        {
            if (s_idleSkip && !vu.m_xgkick.active && !g_vuVerifyFast && c.hint->nextReady > vu.m_cycle + 1u)
            {
                vu.m_cycle = std::min(target, c.hint->nextReady - 1u);
                continue;
            }
            vu.advanceOneCycle();
        }
    }
    // advanceTo(target) then flushPipelines() at the end of a fast-loop run, without
    // stepping: nothing writes VU memory any more (no queued stores), so the PATH1
    // transfer in flight is sent in one progressXgkick call, and the cycle jumps
    // to each pending commit. Each qword costs two cycles of banked credit, so the
    // credit it used gives the cycle the transfer ends on. Same commits on the same
    // cycles, same final cycle and packet as stepping. False (nothing done) when a
    // store is queued or PATH1 has no data memory; the caller then steps.
    static bool drain(Interp &vu, uint64_t target)
    {
        if (g_vuVerifyFast)
            return false;
        for (const Interp::PendingStore &store : vu.m_storePipeline)
            if (store.valid)
                return false;
        const uint64_t start = vu.m_cycle;
        uint64_t end = std::max(start, target);
        if (vu.m_xgkick.active)
        {
            if (!vu.m_activeVuData || vu.m_activeVuDataSize == 0u || vu.m_xgkick.cycleCredit > 1u)
                return false;
            const uint32_t credit = vu.m_xgkick.cycleCredit;
            constexpr uint32_t kBank = 1u << 30; // > two cycles per qword of the 64 KB buffer
            vu.m_xgkick.cycleCredit = kBank;
            vu.progressXgkick(); // ++credit, then a qword per two credits until the end or a fault
            if (vu.m_xgkick.active)
            {
                // Cannot happen (the buffer bounds every packet); keep the per-cycle meaning.
                vu.m_xgkick.cycleCredit = credit;
                return false;
            }
            const uint64_t steps = (kBank + 1u - vu.m_xgkick.cycleCredit) / 2u;
            vu.m_xgkick.cycleCredit = 0u;
            end = std::max(end, start + 2u * steps - credit);
        }
        uint64_t lastReady = 0;
        bool pending = false;
        const auto note = [&](bool valid, uint64_t readyCycle) {
            if (valid)
            {
                pending = true;
                lastReady = std::max(lastReady, readyCycle);
            }
        };
        for (const auto &entry : vu.m_flagPipeline)
            note(entry.valid, entry.readyCycle);
        note(vu.m_fdiv.valid, vu.m_fdiv.readyCycle);
        for (const auto &entry : vu.m_efu)
            note(entry.valid, entry.readyCycle);
        for (const auto &entry : vu.m_vfWritePipeline)
            note(entry.valid, entry.readyCycle);
        for (const auto &entry : vu.m_viWritePipeline)
            note(entry.valid, entry.readyCycle);
        for (const auto &entry : vu.m_accWritePipeline)
            note(entry.valid, entry.readyCycle);
        if (pending)
        {
            end = std::max({end, start + 1u, lastReady});
            // commitReadyPipelines changes state only on cycles where an entry is due,
            // and hint.nextReady is a lower bound on the next such cycle.
            const CommitHint &hint = commitHintFor(&vu);
            while (vu.m_cycle < end)
            {
                vu.m_cycle = std::min(end, std::max(vu.m_cycle + 1u, hint.nextReady));
                vu.commitReadyPipelines();
            }
        }
        vu.m_cycle = end;
        vu.m_state.cycles = end;
        return true;
    }
    // The XGKICK wait for the previous PATH1 transfer; false at the budget.
    static __declspec(noinline) bool waitXgkick(Ctx &c)
    {
        Interp &vu = *c.vu;
        while (vu.m_xgkick.active && vu.m_cycle < c.budgetEnd)
        {
            // advanceOneCycle, with endCycle's skipped commit check and banked credit.
            if (g_vuVerifyFast)
                vu.advanceOneCycle();
            else
                endCycle(c, vu.m_state);
        }
        return !vu.m_xgkick.active;
    }
    // endPairCycle after the cycle increment, when a commit is due or PATH1 is live.
    static __declspec(noinline) void endCycleSlow(Ctx &c)
    {
        Interp &vu = *c.vu;
        if (vu.m_cycle >= c.hint->nextReady)
            vu.commitReadyPipelines();
        if (vu.m_xgkick.active)
            vu.progressXgkick();
    }
    // progressXgkick on a cycle that sends a qword (banked credit 1). The common case,
    // a qword inside a GIF tag that is not its last, is done here without the call's
    // spills; anything else (a tag boundary, wrap-around, faults) goes to progressXgkick.
    static __declspec(noinline) void kickStep(Interp &vu)
    {
        Interp::XgkickPipeline &kick = vu.m_xgkick;
        const uint32_t offset = kick.copiedBytes;
        const uint32_t size = vu.m_activeVuDataSize;
        const uint32_t source = (kick.sourceAddress + offset) & (size - 1u);
        if (kick.cycleCredit == 1u && vu.m_activeVuData && (size & (size - 1u)) == 0u && source + 16u <= size &&
            offset <= Interp::XgkickPipeline::kBufferSize - 16u && kick.currentTagEnd != 0u &&
            offset + 16u < kick.currentTagEnd)
        {
            std::memcpy(kick.packet.data() + offset, vu.m_activeVuData + source, 16u);
            kick.copiedBytes = offset + 16u;
            kick.cycleCredit = 0u;
            return;
        }
        vu.progressXgkick();
    }
    // endPairCycle: the cycle increment, due commits, and PATH1 progress.
    // progressXgkick moves a qword every second cycle; the other cycles only
    // bank the credit, which is done here without the call.
    // VU1State::cycles is only read after run(), which copies m_cycle into it.
    static __forceinline void endCycle(Ctx &c, VU1State &)
    {
        Interp &vu = *c.vu;
        const uint64_t cyc = ++vu.m_cycle;
        if (cyc >= c.hint->nextReady)
            vu.commitReadyPipelines();
        if (vu.m_xgkick.active)
        {
            if (vu.m_xgkick.cycleCredit == 0u && vu.m_activeVuData && vu.m_activeVuDataSize != 0u)
                vu.m_xgkick.cycleCredit = 1u;
            else
                kickStep(vu);
        }
    }
    // Interpreter state at fixed offsets from `vu`, which merged blocks keep
    // in a register (reading them through Ctx pointers reloads the pointer
    // after every call).
    static __forceinline uint64_t *vfReadyOf(Interp &vu) { return vu.m_vfReady[0].data(); }
    static __forceinline uint64_t *viReadyOf(Interp &vu) { return vu.m_viReady.data(); }
    static __forceinline uint64_t *accReadyOf(Interp &vu) { return vu.m_accReady.data(); }
    static __forceinline uint64_t cycleOf(const Interp &vu) { return vu.m_cycle; }
    static __forceinline void clearViBackup(Interp &vu) { vu.m_viBranchBackupValid = false; }
    static __forceinline void recordVi(Interp &vu, uint8_t reg, int32_t oldValue)
    {
        vu.recordViWriteForBranch(reg, oldValue);
    }
    // Four lanes of a VF or ACC ready entry set to one cycle (merged blocks).
    static __forceinline void setReady4(uint64_t *slots, uint64_t value)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(slots), _mm256_set1_epi64x(static_cast<long long>(value)));
    }
    // endCycle for merged blocks, which keep the cycle in a local `cyc`:
    // m_cycle is stored on every increment (helpers read it), never reloaded
    // except after advanceTo/waitXgkick, the only calls that move it. Blocks only
    // run with VU1's data memory active (nativeOn), so banking needs no data check.
    static __forceinline void endCycleLocal(Interp &vu, uint64_t &cyc, const uint64_t *nextReady)
    {
        vu.m_cycle = ++cyc;
        if (cyc >= *nextReady)
            vu.commitReadyPipelines();
        if (vu.m_xgkick.active)
        {
            if (vu.m_xgkick.cycleCredit == 0u)
                vu.m_xgkick.cycleCredit = 1u;
            else
                kickStep(vu);
        }
    }
    // One folded op per instantiation, shared by every block that uses the
    // same code word; keeps generated block functions small for the compiler.
    template <uint64_t Packed, uint32_t Upper>
    static __declspec(noinline) void upperOpGeneral(Interp &vu)
    {
        fastUpper(vu, std::integral_constant<uint64_t, Packed>{}, Upper);
    }
    // The lean path inline, every other flag mode out of line, so the common
    // case pays one check and needs fewer saved registers.
    template <uint64_t Packed, uint32_t Upper, bool NoMac = false>
    static __declspec(noinline) void upperOp(Interp &vu)
    {
        if (g_vuFmacLean)
            fastUpper<true, NoMac>(vu, std::integral_constant<uint64_t, Packed>{}, Upper);
        else
            upperOpGeneral<Packed, Upper>(vu);
    }
    template <uint8_t Kind, uint32_t Lower, uint32_t Upper>
    static __declspec(noinline) void lowerOp(Ctx &c)
    {
        fastLower(*c.vu, std::integral_constant<uint8_t, Kind>{}, std::integral_constant<uint32_t, Lower>{}, Upper,
                  c.vuData, c.dataSize, *c.gs, c.memory);
    }

    // ---- Ops inlined into merged blocks (merge_blocks.py) --------------------
    //
    // In the lean flag mode (g_vuFmacLean, every race program) an upper op needs,
    // besides its result, only the normal/exact-zero proof of its dest lanes, so
    // merged blocks inline that form. They keep vuNormalize4 of the registers they
    // read in locals while a register is unchanged (operands arrive normalized),
    // and with a constant DEST the dest-lane store is a blend. Every other flag
    // mode calls upperOpGeneral, as before. Each form has fastUpper's effects.

    // DEST as movemask/blend lane bits (x = bit 0).
    static constexpr int laneBits(uint32_t dest)
    {
        return static_cast<int>(((dest >> 3) & 1u) | ((dest >> 1) & 2u) | ((dest << 1) & 4u) | ((dest << 3) & 8u));
    }
    template <uint32_t Dest>
    static __forceinline void storeDest(float *dst, __m128 value)
    {
        if constexpr (Dest == 0xFu)
            _mm_storeu_ps(dst, value);
        else if constexpr (Dest != 0u)
            _mm_storeu_ps(dst, _mm_blend_ps(_mm_loadu_ps(dst), value, laneBits(Dest)));
    }
    // A register's normalized value after storeDest<Dest> of a value normalizing to `norm`.
    template <uint32_t Dest>
    static __forceinline __m128 mergeDest(__m128 cached, __m128 norm)
    {
        if constexpr (Dest == 0xFu)
            return norm;
        else
            return _mm_blend_ps(cached, norm, laneBits(Dest));
    }
    // Operands come from memory, never compile-time constants: under /fp:fast the
    // compiler folds x * 0 and x + 0 and loses the sign of zero results.
    static __forceinline __m128 normLoad(const float *reg) { return vuNormalize4(_mm_loadu_ps(reg)); }

    static __declspec(noinline) __m128 __vectorcall fmacSlowLean(Interp &vu, __m128 result, __m128 exactZero,
                                                                 uint32_t upper, uint32_t dest)
    {
        vu.m_currentUpperInstruction = upper; // read by normalizeFmacResult
        return fmacSlow(vu, upper, static_cast<uint8_t>(dest), result, _mm_castps_si128(exactZero));
    }
    static __declspec(noinline) void macPushLean(Interp &vu, int signs, int zeros, uint32_t dest)
    {
        static constexpr uint8_t kComponentsToLanes[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
        pushPendingMac(vu.m_state.mac, vu.m_cycle + Interp::kFmacLatency,
                       (static_cast<uint32_t>(kComponentsToLanes[signs] & dest) << 4) |
                           (kComponentsToLanes[zeros] & dest));
    }
    // The lazily committed DIV result, as fastUpper's kMulQ.
    static __forceinline float leanQ(Interp &vu)
    {
        if (vu.m_fdiv.valid && vu.m_fdiv.readyCycle <= vu.m_cycle)
        {
            vu.m_state.q = vu.m_fdiv.value;
            const uint32_t currentDi = vu.m_fdiv.statusDi & 0x30u;
            vu.m_state.status = (vu.m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            vu.m_fdiv = {};
        }
        return vu.m_state.q;
    }
    // fastUpper<true, NoMac> for every kind but kExec, kNop and kClip, with vs, vt
    // and accIn the normalized FS, FT and ACC (each only if the kind reads it; ITOF
    // takes FS unnormalized in vs) and macOnly g_vuMacOnlyFlags. Returns the value for
    // the dest lanes; norm gets its normalized form (left unset for FTOI, whose
    // integers the caller re-reads).
    template <uint64_t Packed, uint32_t Upper, bool NoMac>
    static __forceinline __m128 leanUpper(Interp &vu, bool macOnly, __m128 vs, __m128 vt, __m128 accIn, __m128 &norm)
    {
        constexpr FastOp op = unpackFastOp(Packed);
        constexpr int lanes = laneBits(op.dest);
        constexpr uint8_t k = op.kind;
        if constexpr (k == kItof0 || k == kItof4 || k == kItof12)
        {
            // static_cast<float>(int) / 2^n: the conversion rounds as cvtdq2ps (MXCSR),
            // the power-of-two scale is exact.
            __m128 r = _mm_cvtepi32_ps(_mm_castps_si128(vs));
            if constexpr (k == kItof4)
                r = _mm_mul_ps(r, _mm_set1_ps(1.0f / 16.0f));
            else if constexpr (k == kItof12)
                r = _mm_mul_ps(r, _mm_set1_ps(1.0f / 4096.0f));
            norm = r;
            return r;
        }
        else if constexpr (k == kFtoi0 || k == kFtoi4 || k == kFtoi12)
        {
            // (int32)((double)v * 2^n), saturating: v * 2^n is exact in float (or
            // +-inf), cvttps2dq truncates like the cast and gives INT32_MIN below the
            // range; v >= 2^31 / 2^n is exactly the scaled >= INT32_MAX case.
            constexpr float scale = k == kFtoi0 ? 1.0f : k == kFtoi4 ? 16.0f : 4096.0f;
            const __m128 scaled = k == kFtoi0 ? vs : _mm_mul_ps(vs, _mm_set1_ps(scale));
            const __m128i r = _mm_blendv_epi8(_mm_cvttps_epi32(scaled), _mm_set1_epi32(0x7FFFFFFF),
                                              _mm_castps_si128(_mm_cmpge_ps(vs, _mm_set1_ps(2147483648.0f / scale))));
            return _mm_castsi128_ps(r);
        }
        else if constexpr (k == kAbs)
        {
            const __m128 r = _mm_andnot_ps(_mm_set1_ps(-0.0f), vs);
            norm = r;
            return r;
        }
        else
        {
            const auto broadcast = [&]() {
                if constexpr (op.bc == 0)
                    return _mm_shuffle_ps(vt, vt, 0x00);
                else if constexpr (op.bc == 1)
                    return _mm_shuffle_ps(vt, vt, 0x55);
                else if constexpr (op.bc == 2)
                    return _mm_shuffle_ps(vt, vt, 0xAA);
                else
                    return _mm_shuffle_ps(vt, vt, 0xFF);
            };
            constexpr bool isMinMax = k == kMaxBc || k == kMiniBc || k == kMax || k == kMini || k == kMiniI || k == kMaxI;
            if constexpr (isMinMax)
            {
                __m128 r;
                if constexpr (k == kMaxBc)
                    r = _mm_max_ps(vs, broadcast());
                else if constexpr (k == kMiniBc)
                    r = _mm_min_ps(vs, broadcast());
                else if constexpr (k == kMax)
                    r = _mm_max_ps(vs, vt);
                else if constexpr (k == kMini)
                    r = _mm_min_ps(vs, vt);
                else if constexpr (k == kMiniI)
                    r = _mm_min_ps(vs, vuNormalize4(_mm_set1_ps(vu.m_state.i)));
                else
                    r = _mm_max_ps(vs, vuNormalize4(_mm_set1_ps(vu.m_state.i)));
                norm = r; // one of two normalized operands
                return r;
            }
            else
            {
                constexpr bool productSum = k == kMaddBc || k == kMsubBc || k == kMaddaBc || k == kMsubaBc ||
                                            k == kOpmsub || k == kMadd || k == kMsub || k == kMadda || k == kMaddQ;
                constexpr int zeroRule = (k == kAddBc || k == kAdd || k == kAddI || k == kAddaBc) ? 0
                                         : (k == kSubBc || k == kSub || k == kSubI || k == kSubaBc) ? 1
                                                                                                   : 2;
                static_assert(k != kExec && k != kNop && k != kClip, "leanUpper: no lean form");
                __m128 left = vs;
                __m128 right;
                __m128 result;
                if constexpr (k == kAddBc || k == kAddaBc)
                    { right = broadcast(); result = _mm_add_ps(left, right); }
                else if constexpr (k == kSubBc || k == kSubaBc)
                    { right = broadcast(); result = _mm_sub_ps(left, right); }
                else if constexpr (k == kMaddBc || k == kMaddaBc)
                    { right = broadcast(); result = _mm_add_ps(accIn, _mm_mul_ps(left, right)); }
                else if constexpr (k == kMsubBc || k == kMsubaBc)
                    { right = broadcast(); result = _mm_sub_ps(accIn, _mm_mul_ps(left, right)); }
                else if constexpr (k == kMulBc || k == kMulaBc)
                    { right = broadcast(); result = _mm_mul_ps(left, right); }
                else if constexpr (k == kAdd)
                    { right = vt; result = _mm_add_ps(left, right); }
                else if constexpr (k == kSub)
                    { right = vt; result = _mm_sub_ps(left, right); }
                else if constexpr (k == kMul || k == kMula)
                    { right = vt; result = _mm_mul_ps(left, right); }
                else if constexpr (k == kMadd || k == kMadda)
                    { right = vt; result = _mm_add_ps(accIn, _mm_mul_ps(left, right)); }
                else if constexpr (k == kMsub)
                    { right = vt; result = _mm_sub_ps(accIn, _mm_mul_ps(left, right)); }
                else if constexpr (k == kMulQ)
                    { right = vuNormalize4(_mm_set1_ps(leanQ(vu))); result = _mm_mul_ps(left, right); }
                else if constexpr (k == kMaddQ)
                    { right = vuNormalize4(_mm_set1_ps(leanQ(vu))); result = _mm_add_ps(accIn, _mm_mul_ps(left, right)); }
                else if constexpr (k == kMulI)
                    { right = vuNormalize4(_mm_set1_ps(vu.m_state.i)); result = _mm_mul_ps(left, right); }
                else if constexpr (k == kAddI)
                    { right = vuNormalize4(_mm_set1_ps(vu.m_state.i)); result = _mm_add_ps(left, right); }
                else if constexpr (k == kSubI)
                    { right = vuNormalize4(_mm_set1_ps(vu.m_state.i)); result = _mm_sub_ps(left, right); }
                else if constexpr (k == kOpmula || k == kOpmsub)
                {
                    left = _mm_shuffle_ps(vs, vs, _MM_SHUFFLE(3, 0, 2, 1));
                    right = _mm_shuffle_ps(vt, vt, _MM_SHUFFLE(3, 1, 0, 2));
                    result = _mm_mul_ps(left, right);
                    if constexpr (k == kOpmsub)
                        result = _mm_sub_ps(accIn, result);
                }
                else
                    static_assert(k == kAddBc, "leanUpper: unhandled kind");
                // fastUpper's lean proofs.
                __m128i normal;
                if constexpr (!productSum)
                {
                    // Exponent in [2, 0xFD]: the magnitude bits in [0x01000000, 0x7EFFFFFF],
                    // which + 0x7F000000 maps (wrapping) to the signed range below -0x02000000.
                    const __m128i magnitude = _mm_and_si128(_mm_castps_si128(result), _mm_set1_epi32(0x7FFFFFFF));
                    normal = _mm_cmpgt_epi32(_mm_set1_epi32(-0x02000000),
                                             _mm_add_epi32(magnitude, _mm_set1_epi32(0x7F000000)));
                }
                else
                {
                    const __m128i exponent = vuExponent4(result);
                    const __m128 zero = _mm_setzero_ps();
                    const __m128i zeroFactor =
                        _mm_castps_si128(_mm_or_ps(_mm_cmpeq_ps(left, zero), _mm_cmpeq_ps(right, zero)));
                    const __m128i productExponent = vuExponent4(_mm_mul_ps(left, right));
                    const __m128i productNormal =
                        _mm_and_si128(vuInRange4(productExponent, 1, 0xFD),
                                      _mm_cmpgt_epi32(_mm_add_epi32(exponent, _mm_set1_epi32(21)), productExponent));
                    normal = _mm_and_si128(vuInRange4(exponent, 3, 0xFC), _mm_or_si128(zeroFactor, productNormal));
                }
                const __m128 zero = _mm_setzero_ps();
                __m128 proof;
                if constexpr (zeroRule == 0)
                    proof = _mm_cmpeq_ps(left, _mm_xor_ps(right, _mm_set1_ps(-0.0f)));
                else if constexpr (zeroRule == 1)
                    proof = _mm_cmpeq_ps(left, right);
                else
                    proof = _mm_or_ps(_mm_cmpeq_ps(left, zero), _mm_cmpeq_ps(right, zero));
                const __m128 exactZero = _mm_and_ps(_mm_cmpeq_ps(result, zero), proof);
                const int handled = _mm_movemask_ps(_mm_or_ps(_mm_castsi128_ps(normal), exactZero));
                if ((~handled & lanes) != 0) [[unlikely]]
                {
                    result = fmacSlowLean(vu, result, exactZero, Upper, op.dest);
                    norm = vuNormalize4(result);
                }
                else
                {
                    if constexpr (!NoMac && op.dest != 0u)
                        if (macOnly)
                            macPushLean(vu, _mm_movemask_ps(result), _mm_movemask_ps(exactZero), op.dest);
                    norm = result; // dest lanes normal or exactly zero
                }
                return result;
            }
        }
    }

    // Lower ops of merged blocks. Data memory is 16 KB there (nativeOn), so every
    // masked quadword address is in range, and stores are direct (g_vuDirectStores
    // holds while the fast loop runs). Same effects as fastLower.
    template <uint32_t Instr>
    static __forceinline uint32_t qwordAddress(int32_t base)
    {
        constexpr int32_t imm = static_cast<int32_t>(Instr << 21) >> 21;
        return (static_cast<uint32_t>(base + imm) * 16u) & 0x3FF0u;
    }
    template <uint32_t Instr>
    static __forceinline void lq(VU1State &s, const uint8_t *vuData)
    {
        const uint32_t addr = qwordAddress<Instr>(s.vi[(Instr >> 11) & 0xFu]);
        storeDest<(Instr >> 21) & 0xFu>(s.vf[(Instr >> 16) & 0x1Fu],
                                        _mm_loadu_ps(reinterpret_cast<const float *>(vuData + addr)));
    }
    template <uint32_t Instr>
    static __forceinline void sq(VU1State &s, uint8_t *vuData)
    {
        const uint32_t addr = qwordAddress<Instr>(s.vi[(Instr >> 16) & 0xFu]);
        storeDest<(Instr >> 21) & 0xFu>(reinterpret_cast<float *>(vuData + addr), _mm_loadu_ps(s.vf[(Instr >> 11) & 0x1Fu]));
    }
    template <uint32_t Instr>
    static __forceinline void lqi(VU1State &s, const uint8_t *vuData)
    {
        constexpr uint32_t is = (Instr >> 11) & 0xFu;
        const uint32_t addr = (static_cast<uint32_t>(static_cast<uint16_t>(s.vi[is])) * 16u) & 0x3FF0u;
        storeDest<(Instr >> 21) & 0xFu>(s.vf[(Instr >> 16) & 0x1Fu],
                                        _mm_loadu_ps(reinterpret_cast<const float *>(vuData + addr)));
        if constexpr (is != 0u)
            s.vi[is] = static_cast<int16_t>(s.vi[is] + 1);
    }
    template <uint32_t Instr>
    static __forceinline void sqi(VU1State &s, uint8_t *vuData)
    {
        constexpr uint32_t it = (Instr >> 16) & 0xFu;
        const uint32_t addr = (static_cast<uint32_t>(static_cast<uint16_t>(s.vi[it])) * 16u) & 0x3FF0u;
        storeDest<(Instr >> 21) & 0xFu>(reinterpret_cast<float *>(vuData + addr), _mm_loadu_ps(s.vf[(Instr >> 11) & 0x1Fu]));
        if constexpr (it != 0u)
            s.vi[it] = static_cast<int16_t>(s.vi[it] + 1);
    }
    template <uint32_t Instr>
    static __forceinline void moveVf(VU1State &s)
    {
        storeDest<(Instr >> 21) & 0xFu>(s.vf[(Instr >> 16) & 0x1Fu], _mm_loadu_ps(s.vf[(Instr >> 11) & 0x1Fu]));
    }
    template <uint32_t Instr>
    static __forceinline void mr32(VU1State &s)
    {
        const __m128 v = _mm_loadu_ps(s.vf[(Instr >> 11) & 0x1Fu]);
        storeDest<(Instr >> 21) & 0xFu>(s.vf[(Instr >> 16) & 0x1Fu], _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 3, 2, 1)));
    }
    static __forceinline uint64_t fdivReady(Ctx &c)
    {
        return c.vu->m_fdiv.valid ? c.vu->m_fdiv.readyCycle : 0u;
    }
    static __forceinline uint64_t efuReady(Ctx &c) { return c.vu->m_efuResourceReady; }
    static __forceinline uint64_t pReady(Ctx &c)
    {
        uint64_t ready = 0;
        for (const Interp::ScalarPipelineEntry &entry : c.vu->m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
        return ready;
    }
    static __forceinline uint32_t bail(Ctx &c, uint32_t pc)
    {
        c.vu->m_state.pc = pc;
        return kNativeBail;
    }
    static __forceinline void recordVi(Ctx &c, uint8_t reg, int32_t oldValue)
    {
        c.vu->recordViWriteForBranch(reg, oldValue);
    }
    static __forceinline void resetZero(VU1State &s)
    {
        _mm_storeu_ps(s.vf[0], _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f));
        s.vi[0] = 0;
    }
    static __declspec(noinline) void markGeneric(Ctx &c, uint32_t pairIndex)
    {
        c.vu->markPairWrites(c.vu->m_decodedCodeCache[pairIndex]);
    }
    // ATV_VU_COUNTS=1: execution counts for tuning the native blocks (generic pairs by code
    // words, fast-loop pairs by PC, drain cycles), each table printed every 2^20 counts.
    static inline const bool s_counts = std::getenv("ATV_VU_COUNTS") != nullptr;
    // Fixed open-addressed table: no new template instantiations (this object is near
    // the COFF section limit).
    struct CountTable
    {
        static constexpr uint32_t kSlots = 8192u;
        const char *name;
        uint64_t keys[kSlots]{};
        uint64_t hits[kSlots]{};
        uint64_t total = 0;
        uint32_t used = 0;
    };
    static __declspec(noinline) void count(CountTable &table, uint64_t key, uint64_t n = 1u)
    {
        uint32_t slot = static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ull) >> 51) & (CountTable::kSlots - 1u);
        while (table.hits[slot] != 0u && table.keys[slot] != key)
            slot = (slot + 1u) & (CountTable::kSlots - 1u);
        if (table.hits[slot] == 0u)
        {
            if (table.used + 1u >= CountTable::kSlots)
                return;
            ++table.used;
            table.keys[slot] = key;
        }
        table.hits[slot] += n;
        const uint64_t before = table.total;
        table.total += n;
        if ((before >> 20) == (table.total >> 20))
            return;
        std::fprintf(stderr, "[vu-counts] %s total=%llu keys=%u\n", table.name,
                     static_cast<unsigned long long>(table.total), table.used);
        uint64_t below = ~0ull;
        for (uint32_t shown = 0; shown < 24u; ++shown)
        {
            uint32_t best = CountTable::kSlots;
            for (uint32_t i = 0; i < CountTable::kSlots; ++i)
                if (table.hits[i] < below && (best == CountTable::kSlots || table.hits[i] > table.hits[best]))
                    best = i;
            if (best == CountTable::kSlots || table.hits[best] == 0u)
                break;
            below = table.hits[best];
            std::fprintf(stderr, "[vu-counts]   %5.1f%% %016llx %llu\n", 100.0 * table.hits[best] / table.total,
                         static_cast<unsigned long long>(table.keys[best]),
                         static_cast<unsigned long long>(table.hits[best]));
        }
    }
    static inline CountTable s_genericCounts{"generic pairs (lower<<32|upper)"};
    static inline CountTable s_loopCounts{"fast-loop pairs (pc)"};
    static inline CountTable s_drainCounts{"drain cycles (kick active at drain)"};
    // The fast loop's execPair for pairs without a folded form (I-bit or
    // shadowed upper writes, or an upper/lower op without a fast case).
    static __declspec(noinline) void execPairGeneric(Ctx &c, uint32_t pairIndex)
    {
        Interp &vu = *c.vu;
        VU1State &m_state = vu.m_state;
        const DecodedInstructionPair &decoded = vu.m_decodedCodeCache[pairIndex];
        if (s_counts)
            count(s_genericCounts, uint64_t{decoded.lower} << 32 | decoded.upper);
        FastOp op = classifyUpper(decoded.upper);
        op.lowerKind = classifyLower(decoded);
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
            fastUpper(vu, op, decoded.upper);
            float immediate = 0.0f;
            std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
            m_state.i = vu.normalizeOperand(immediate);
        }
        else if (decoded.upperVfShadowReg != 0u)
        {
            float oldVf[4];
            float upperVf[4];
            std::memcpy(oldVf, m_state.vf[decoded.upperVfShadowReg], sizeof(oldVf));
            fastUpper(vu, op, decoded.upper);
            std::memcpy(upperVf, m_state.vf[decoded.upperVfShadowReg], sizeof(upperVf));
            std::memcpy(m_state.vf[decoded.upperVfShadowReg], oldVf, sizeof(oldVf));
            vu.execLower(decoded.lower, c.vuData, c.dataSize, *c.gs, c.memory, decoded.upper);
            std::memcpy(m_state.vf[decoded.upperVfShadowReg], upperVf, sizeof(upperVf));
        }
        else
        {
            if (op.kind != kNop)
                fastUpper(vu, op, decoded.upper);
            if (op.lowerKind != kLowerNop)
                fastLower(vu, op.lowerKind, decoded.lower, decoded.upper, c.vuData, c.dataSize, *c.gs, c.memory);
        }

        vu.m_viBranchBackupValid = false;

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
                vu.recordViWriteForBranch(writtenVi, oldVi);
        }
    }
    // The fast loop's finishPc; true when the program ends.
    static __forceinline bool finishPc(Ctx &c, uint32_t nextPc, bool eBit, bool dBit, bool tBit, bool branch)
    {
        VU1State &m_state = c.vu->m_state;
        m_state.pc = nextPc;
        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0u)
            {
                m_state.pc = m_state.branchTarget & c.vu->microAddressMask();
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        const bool dHalt = dBit && m_state.dBitEnabled;
        const bool tHalt = tBit && m_state.tBitEnabled;
        const bool haltBit = dHalt || tHalt;
        const bool haltBranch = haltBit && branch;

        bool ended = false;
        if (m_state.haltAfterDelaySlot)
        {
            m_state.stoppedByD = c.vu->m_pendingHaltD;
            m_state.stoppedByT = c.vu->m_pendingHaltT;
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
        else if (eBit)
            m_state.ebit = true;
        else if (haltBranch)
        {
            m_state.haltAfterDelaySlot = true;
            c.vu->m_pendingHaltD = dHalt;
            c.vu->m_pendingHaltT = tHalt;
        }
        return ended;
    }

    // Block for the pair at pairIndex, valid for the code image `generation`.
    struct Resolved
    {
        uint64_t generation; // generation + 1; 0 = never resolved
        BlockFn fn;
    };
    static inline std::array<Resolved, Interp::kMaxDecodedPairs> s_resolved{};
    static __forceinline BlockFn lookup(const uint8_t *vuCode, uint64_t generation, uint32_t pairIndex)
    {
        const Resolved &r = s_resolved[pairIndex];
        if (r.generation == generation + 1u)
            return r.fn;
        return resolve(vuCode, generation, pairIndex);
    }
    static __declspec(noinline) BlockFn resolve(const uint8_t *vuCode, uint64_t generation, uint32_t pairIndex)
    {
        static std::vector<std::vector<uint32_t>> s_byPair;
        static size_t s_count = 0;
        static const BlockEntry *s_entries = nullptr;
        if (!s_entries)
        {
            s_entries = registry(s_count);
            s_byPair.resize(Interp::kMaxDecodedPairs);
            for (uint32_t i = 0; i < s_count; ++i)
                if ((s_entries[i].pc >> 3) < Interp::kMaxDecodedPairs)
                    s_byPair[s_entries[i].pc >> 3].push_back(i);
            std::fprintf(stderr, "[vu-native] %zu compiled VU1 blocks%s\n", s_count,
                         s_enabled ? "" : " (disabled by ATV_VU_NO_NATIVE_BLOCKS)");
        }
        Resolved &r = s_resolved[pairIndex];
        r.generation = generation + 1u;
        r.fn = nullptr;
        for (const uint32_t index : s_byPair[pairIndex])
        {
            const BlockEntry &e = s_entries[index];
            if ((e.pc >> 3) + e.pairs <= Interp::kMaxDecodedPairs &&
                std::memcmp(vuCode + e.pc, e.words, e.pairs * 8u) == 0)
            {
                r.fn = e.fn;
                break;
            }
        }
        return r.fn;
    }

    // ATV_VU_NATIVE_GEN=file: append C++ for the block starting at `first`
    // (tools/vu1gen/merge_blocks.py turns the chunks into
    // ps2_vu1_native_blocks.inc). Uses the decoded code cache and the fast
    // loop's own classification, so the folded code matches it pair by pair.
    // `maxPairs` ends the block early (buildFromWords, whose code memory holds
    // only the block).
    static void generate(Interp &vu, uint32_t first, uint32_t codePairs, uint32_t maxPairs = ~0u)
    {
        static constexpr uint32_t kMaxPairs = 256;
        static FILE *s_file = nullptr;
        static std::vector<std::pair<uint32_t, std::vector<uint32_t>>> s_done;
        if (!s_genPath)
            return;

        // Pairs in the block: body..., terminator (branch or E-bit), delay slot.
        std::vector<uint32_t> pairs;
        bool tail = false;
        bool complete = false;
        for (uint32_t i = first; i < codePairs && i + 1u < codePairs; ++i)
        {
            const DecodedInstructionPair &d = vu.m_decodedCodeCache[i];
            const bool branch = !d.iBit && d.lowerUsage.pipeline == Interp::PipelineBranch;
            if (d.upperUsage.reserved || d.lowerUsage.reserved || d.dBit || d.tBit ||
                (tail && (branch || d.eBit)) || (!tail && pairs.size() >= kMaxPairs) || pairs.size() >= maxPairs)
                break;
            pairs.push_back(i);
            if (tail)
            {
                complete = true;
                break;
            }
            tail = branch || d.eBit;
        }
        if (pairs.empty())
            return;
        std::vector<uint32_t> words(pairs.size() * 2u);
        for (size_t n = 0; n < pairs.size(); ++n)
        {
            words[n * 2u] = vu.m_decodedCodeCache[pairs[n]].lower;
            words[n * 2u + 1u] = vu.m_decodedCodeCache[pairs[n]].upper;
        }
        for (const auto &done : s_done)
            if (done.first == first && done.second == words)
                return;
        s_done.emplace_back(first, words);
        if (!s_file)
        {
            s_file = std::fopen(s_genPath, "ab");
            if (!s_file)
            {
                std::fprintf(stderr, "[vu-native] cannot open %s\n", s_genPath);
                s_done.clear();
                return;
            }
        }
        uint64_t key = 1469598103934665603ull;
        for (const uint32_t word : words)
            for (uint32_t b = 0; b < 4u; ++b)
                key = (key ^ ((word >> (8u * b)) & 0xFFu)) * 1099511628211ull;

        std::string out;
        char buf[512];
        const auto line = [&](const char *fmt, auto... args) {
            std::snprintf(buf, sizeof(buf), fmt, args...);
            out += buf;
            out += '\n';
        };
        const uint32_t pc0 = first * 8u;
        line("// ATV-VU1-BLOCK pc=0x%04x pairs=%zu key=%016llx complete=%d", pc0, pairs.size(),
             static_cast<unsigned long long>(key), complete ? 1 : 0);
        line("static const uint32_t vu1nbw_%04x_%016llx[] = {", pc0, static_cast<unsigned long long>(key));
        for (size_t n = 0; n < words.size(); n += 2u)
            line("    0x%08xu, 0x%08xu,", words[n], words[n + 1u]);
        line("};");
        line("static uint32_t vu1nb_%04x_%016llx(VU1Native::Ctx &c)", pc0, static_cast<unsigned long long>(key));
        line("{");
        line("    VU1Interpreter &vu = *c.vu;");
        line("    VU1State &s = vu.state();");
        line("    uint64_t &cyc = *c.cycle;");
        line("    (void)vu;");
        for (size_t n = 0; n < pairs.size(); ++n)
        {
            const uint32_t p = pairs[n];
            const uint32_t pc = p * 8u;
            const DecodedInstructionPair &d = vu.m_decodedCodeCache[p];
            const InstructionUsage &lu = d.lowerUsage;
            FastOp op = classifyUpper(d.upper);
            op.lowerKind = classifyLower(d);
            const FastReads rd = classifyReads(d);
            const FastWrites w = classifyWrites(d);
            const bool branch = !d.iBit && lu.pipeline == Interp::PipelineBranch;
            const bool last = n + 1u == pairs.size();
            const bool terminator = branch || d.eBit;
            line("    // 0x%04x lower=%08x upper=%08x", pc, d.lower, d.upper);
            line("    {");
            line("        uint64_t t = cyc;");
            for (uint32_t k = 0; k < rd.vfCount; ++k)
                line("        t = std::max(t, c.vfReady[%u]);", rd.vf[k]);
            for (uint32_t k = 0; k < rd.viCount; ++k)
                line("        t = std::max(t, c.viReady[%u]);", rd.vi[k]);
            for (uint32_t k = 0; k < rd.accCount; ++k)
                line("        t = std::max(t, c.accReady[%u]);", rd.acc[k]);
            if (rd.scalarWait)
            {
                if (lu.pipeline == Interp::PipelineFdiv || lu.waitQ)
                    line("        t = std::max(t, VU1Native::fdivReady(c));");
                if (lu.pipeline == Interp::PipelineEfu)
                    line("        t = std::max(t, VU1Native::efuReady(c));");
                if (lu.waitP)
                    line("        t = std::max(t, VU1Native::pReady(c));");
                if (lu.pipeline == Interp::PipelineXgkick)
                    line("        if (*c.xgkickActive) t = std::max(t, cyc + 1u);");
            }
            line("        if (t > cyc)");
            line("        {");
            line("            if (t >= c.budgetEnd) return VU1Native::bail(c, 0x%04xu);", pc);
            line("            VU1Native::advanceTo(c, t);");
            line("        }");
            if (lu.pipeline == Interp::PipelineXgkick)
                line("        if (!VU1Native::waitXgkick(c)) return VU1Native::bail(c, 0x%04xu);", pc);
            line("        s.pc = 0x%04xu;", pc);
            if (lowerReadsFlags(d) != 0u)
                line("        if (g_vuMacOnlyFlags) applyPendingMac(s.mac, cyc);");

            const bool viRecord = lu.delaysNextBranchRead && w.viReg != 0u;
            const bool ibitFast = d.iBit && d.upperVfShadowReg == 0u && op.kind != kExec;
            // Both halves fast: execPairGeneric's shadow form (the lower half sees the
            // upper's register as it was before the pair) needs no generic call either.
            const bool fast = !d.iBit && op.kind != kExec && op.lowerKind != kLowerExec;
            const uint32_t shadow = d.upperVfShadowReg;
            const unsigned long long packed = packFastOp(op);
            if (ibitFast)
            {
                if (op.kind != kNop)
                    line("        VU1Native::upperOp<0x%llxull, 0x%08xu>(vu);", packed, d.upper);
                const uint32_t immediate = std::bit_cast<uint32_t>(vuNormalizeOperand(std::bit_cast<float>(d.lower)));
                line("        s.i = std::bit_cast<float>(0x%08xu);", immediate);
                line("        *c.viBackupValid = false;");
            }
            else if (fast)
            {
                if (viRecord)
                    line("        const int32_t oldVi = s.vi[%u];", w.viReg);
                if (shadow != 0u)
                    line("        const __m128 shadowOld = _mm_loadu_ps(s.vf[%u]);", shadow);
                if (op.kind != kNop)
                    line("        VU1Native::upperOp<0x%llxull, 0x%08xu>(vu);", packed, d.upper);
                if (shadow != 0u)
                {
                    line("        const __m128 shadowNew = _mm_loadu_ps(s.vf[%u]);", shadow);
                    line("        _mm_storeu_ps(s.vf[%u], shadowOld);", shadow);
                }
                if (op.lowerKind != kLowerNop)
                    line("        VU1Native::lowerOp<%u, 0x%08xu, 0x%08xu>(c);", op.lowerKind, d.lower, d.upper);
                if (shadow != 0u)
                    line("        _mm_storeu_ps(s.vf[%u], shadowNew);", shadow);
                line("        *c.viBackupValid = false;");
                if (viRecord)
                    line("        VU1Native::recordVi(c, %u, oldVi);", w.viReg);
            }
            else
                line("        VU1Native::execPairGeneric(c, %u);", p);

            if (w.multiVi)
                line("        VU1Native::markGeneric(c, %u);", p);
            else
            {
                for (uint32_t b = 0; b < 4u; ++b)
                    if ((w.lowerLanes & (1u << b)) != 0u)
                        line("        c.vfReady[%u] = cyc + %uu;", w.lowerReg * 4u + b, w.lowerLat);
                for (uint32_t b = 0; b < 4u; ++b)
                    if ((w.upperLanes & (1u << b)) != 0u)
                        line("        c.vfReady[%u] = cyc + %uu;", w.upperReg * 4u + b, w.upperLat);
                if (w.viReg != 0u)
                    line("        c.viReady[%u] = cyc + %uu;", w.viReg, w.viLat);
                for (uint32_t b = 0; b < 4u; ++b)
                    if ((w.accLanes & (1u << b)) != 0u)
                        line("        c.accReady[%u] = cyc + %uu;", b, Interp::kAccForwardLatency);
            }
            line("        *c.writesReadyAt = std::max(*c.writesReadyAt, cyc + %uu);", w.readyOffset);
            line("        VU1Native::resetZero(s);");
            const bool finish = terminator || last;
            if (finish)
            {
                uint32_t nextPc = pc + 8u;
                if (nextPc >= codePairs * 8u)
                    nextPc = 0u;
                line("        const bool ended = VU1Native::finishPc(c, 0x%04xu, %s, %s, %s, %s);", nextPc,
                     d.eBit ? "true" : "false", d.dBit ? "true" : "false", d.tBit ? "true" : "false",
                     branch ? "true" : "false");
            }
            line("        VU1Native::endCycle(c, s);");
            line("        ++g_vu1PairTotal;");
            if (finish)
                line("        if (ended) return VU1Native::kNativeEnded;");
            line("    }");
        }
        line("    return VU1Native::kNativeContinue;");
        line("}");
        line("// ATV-VU1-END");
        std::fwrite(out.data(), 1, out.size(), s_file);
        std::fflush(s_file);
    }

    // ATV_VU_NATIVE_BUILD=file with ATV_VU_NATIVE_GEN: generate the block of each
    // "<pc hex> <pairs> <word hex>..." entry (tools/vu1gen/make_vu1_blocks.py
    // reads the words from the game's ELF), with its code alone in an empty code
    // memory. A pair's decode depends only on its own words and the pair count
    // bounds the block, so the chunks match the ones the game's runs produce.
    static int buildFromWords(const char *wordsPath)
    {
        if (!s_genPath)
        {
            std::fprintf(stderr, "[vu-native] ATV_VU_NATIVE_BUILD needs ATV_VU_NATIVE_GEN=<output file>\n");
            return 1;
        }
        FILE *in = std::fopen(wordsPath, "r");
        if (!in)
        {
            std::fprintf(stderr, "[vu-native] cannot open %s\n", wordsPath);
            return 1;
        }
        auto vu = std::make_unique<Interp>(Interp::Unit::VU1);
        std::vector<uint8_t> code(Interp::kMaxDecodedPairs * 8u);
        uint64_t generation = 0;
        size_t blocks = 0;
        unsigned pc = 0, pairs = 0;
        while (std::fscanf(in, "%x %u", &pc, &pairs) == 2)
        {
            if ((pc & 7u) != 0u || pairs == 0u || pc + pairs * 8ull > code.size())
            {
                std::fprintf(stderr, "[vu-native] bad block pc=0x%04x pairs=%u in %s\n", pc, pairs, wordsPath);
                std::fclose(in);
                return 1;
            }
            std::fill(code.begin(), code.end(), uint8_t{0});
            for (unsigned n = 0; n < pairs * 2u; ++n)
            {
                unsigned word = 0;
                if (std::fscanf(in, "%x", &word) != 1)
                {
                    std::fprintf(stderr, "[vu-native] short block pc=0x%04x in %s\n", pc, wordsPath);
                    std::fclose(in);
                    return 1;
                }
                std::memcpy(code.data() + pc + n * 4u, &word, 4u);
            }
            vu->rebuildDecodedCodeCache(code.data(), static_cast<uint32_t>(code.size()), nullptr, ++generation);
            generate(*vu, pc >> 3, Interp::kMaxDecodedPairs, pairs);
            ++blocks;
        }
        std::fclose(in);
        std::fprintf(stderr, "[vu-native] generated %zu block(s) into %s\n", blocks, s_genPath);
        return 0;
    }
};
