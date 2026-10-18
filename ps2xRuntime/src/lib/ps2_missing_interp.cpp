// Fallback MIPS R5900 interpreter for guest code that has no exact recompiled entry.
//
// The analyzer sometimes merges a small function into its neighbour, so the merged function's
// start address has no table slot and no re-entry case. Calls to it (mostly JALR through vtables
// and array-constructor helpers) used to be skipped. Guest state lives entirely in R5900Context
// and guest RAM, so execution can start anywhere: interpret from the missing address until the
// pc reaches an address that does have a recompiled entry (typically the return address of the
// first JAL, or a loop header), then hand off to generated code.

#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>

extern const uint32_t g_ps2RecompiledFunctionTableBase;
extern const uint32_t g_ps2RecompiledFunctionTableEnd;

namespace
{
    constexpr uint32_t kFpuCondBit = 0x00800000u;

    float asFloat(uint32_t bits)
    {
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    uint32_t asBits(float f)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return bits;
    }

    void logOnce(const char *what, uint32_t key, uint32_t pc, uint32_t extra)
    {
        static std::mutex s_mutex;
        static uint32_t s_seen[256] = {};
        static uint32_t s_count = 0u;
        std::lock_guard<std::mutex> lock(s_mutex);
        for (uint32_t i = 0; i < s_count; ++i)
        {
            if (s_seen[i] == key)
                return;
        }
        if (s_count >= 256u)
            return;
        s_seen[s_count++] = key;
        std::cerr << "[interp] " << what << " pc=0x" << std::hex << pc << " 0x" << extra << std::dec << std::endl;
    }

    struct Interp
    {
        PS2Runtime *runtime;
        uint8_t *rdram;
        R5900Context *ctx;

        uint32_t fetch(uint32_t pc) { return READ32(pc); }

        uint32_t fbits(uint32_t r) { return asBits(ctx->f[r]); }
        void setFbits(uint32_t r, uint32_t v) { ctx->f[r] = asFloat(v); }

        // Executes one non-control-flow instruction. Returns false if unsupported.
        bool execSimple(uint32_t instr, uint32_t pc)
        {
            const uint32_t op = instr >> 26;
            const uint32_t rs = (instr >> 21) & 31u;
            const uint32_t rt = (instr >> 16) & 31u;
            const uint32_t rd = (instr >> 11) & 31u;
            const uint32_t sa = (instr >> 6) & 31u;
            const uint32_t funct = instr & 63u;
            const int32_t imm = static_cast<int16_t>(instr & 0xFFFFu);
            const uint32_t uimm = instr & 0xFFFFu;
            const uint32_t addr = GPR_U32(ctx, rs) + static_cast<uint32_t>(imm);

            switch (op)
            {
            case 0x00:
                switch (funct)
                {
                case 0x00: SET_GPR_S32(ctx, rd, GPR_U32(ctx, rt) << sa); return true;
                case 0x02: SET_GPR_S32(ctx, rd, GPR_U32(ctx, rt) >> sa); return true;
                case 0x03: SET_GPR_S32(ctx, rd, GPR_S32(ctx, rt) >> sa); return true;
                case 0x04: SET_GPR_S32(ctx, rd, GPR_U32(ctx, rt) << (GPR_U32(ctx, rs) & 31u)); return true;
                case 0x06: SET_GPR_S32(ctx, rd, GPR_U32(ctx, rt) >> (GPR_U32(ctx, rs) & 31u)); return true;
                case 0x07: SET_GPR_S32(ctx, rd, GPR_S32(ctx, rt) >> (GPR_U32(ctx, rs) & 31u)); return true;
                case 0x0A: if (GPR_U64(ctx, rt) == 0u) SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs)); return true;
                case 0x0B: if (GPR_U64(ctx, rt) != 0u) SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs)); return true;
                case 0x0F: return true; // sync
                case 0x10: SET_GPR_U64(ctx, rd, ctx->hi); return true;
                case 0x11: ctx->hi = GPR_U64(ctx, rs); return true;
                case 0x12: SET_GPR_U64(ctx, rd, ctx->lo); return true;
                case 0x13: ctx->lo = GPR_U64(ctx, rs); return true;
                case 0x14: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) << (GPR_U32(ctx, rs) & 63u)); return true;
                case 0x16: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) >> (GPR_U32(ctx, rs) & 63u)); return true;
                case 0x17: SET_GPR_S64(ctx, rd, GPR_S64(ctx, rt) >> (GPR_U32(ctx, rs) & 63u)); return true;
                case 0x18:
                {
                    const int64_t p = static_cast<int64_t>(GPR_S32(ctx, rs)) * GPR_S32(ctx, rt);
                    ctx->lo = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(p)));
                    ctx->hi = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(p >> 32)));
                    SET_GPR_U64(ctx, rd, ctx->lo);
                    return true;
                }
                case 0x19:
                {
                    const uint64_t p = static_cast<uint64_t>(GPR_U32(ctx, rs)) * GPR_U32(ctx, rt);
                    ctx->lo = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(p)));
                    ctx->hi = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(p >> 32)));
                    SET_GPR_U64(ctx, rd, ctx->lo);
                    return true;
                }
                case 0x1A:
                {
                    const int32_t a = GPR_S32(ctx, rs), b = GPR_S32(ctx, rt);
                    if (b != 0 && !(a == INT32_MIN && b == -1))
                    {
                        ctx->lo = static_cast<uint64_t>(static_cast<int64_t>(a / b));
                        ctx->hi = static_cast<uint64_t>(static_cast<int64_t>(a % b));
                    }
                    return true;
                }
                case 0x1B:
                {
                    const uint32_t a = GPR_U32(ctx, rs), b = GPR_U32(ctx, rt);
                    if (b != 0u)
                    {
                        ctx->lo = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a / b)));
                        ctx->hi = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a % b)));
                    }
                    return true;
                }
                case 0x20: case 0x21: SET_GPR_S32(ctx, rd, GPR_U32(ctx, rs) + GPR_U32(ctx, rt)); return true;
                case 0x22: case 0x23: SET_GPR_S32(ctx, rd, GPR_U32(ctx, rs) - GPR_U32(ctx, rt)); return true;
                case 0x24: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) & GPR_U64(ctx, rt)); return true;
                case 0x25: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) | GPR_U64(ctx, rt)); return true;
                case 0x26: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) ^ GPR_U64(ctx, rt)); return true;
                case 0x27: SET_GPR_U64(ctx, rd, ~(GPR_U64(ctx, rs) | GPR_U64(ctx, rt))); return true;
                case 0x28: SET_GPR_U64(ctx, rd, ctx->sa); return true;
                case 0x29: ctx->sa = GPR_U32(ctx, rs); return true;
                case 0x2A: SET_GPR_U64(ctx, rd, GPR_S64(ctx, rs) < GPR_S64(ctx, rt) ? 1u : 0u); return true;
                case 0x2B: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) < GPR_U64(ctx, rt) ? 1u : 0u); return true;
                case 0x2C: case 0x2D: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) + GPR_U64(ctx, rt)); return true;
                case 0x2E: case 0x2F: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) - GPR_U64(ctx, rt)); return true;
                case 0x38: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) << sa); return true;
                case 0x3A: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) >> sa); return true;
                case 0x3B: SET_GPR_S64(ctx, rd, GPR_S64(ctx, rt) >> sa); return true;
                case 0x3C: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) << (sa + 32u)); return true;
                case 0x3E: SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) >> (sa + 32u)); return true;
                case 0x3F: SET_GPR_S64(ctx, rd, GPR_S64(ctx, rt) >> (sa + 32u)); return true;
                default: return false;
                }
            case 0x08: case 0x09: SET_GPR_S32(ctx, rt, GPR_U32(ctx, rs) + static_cast<uint32_t>(imm)); return true;
            case 0x0A: SET_GPR_U64(ctx, rt, GPR_S64(ctx, rs) < static_cast<int64_t>(imm) ? 1u : 0u); return true;
            case 0x0B: SET_GPR_U64(ctx, rt, GPR_U64(ctx, rs) < static_cast<uint64_t>(static_cast<int64_t>(imm)) ? 1u : 0u); return true;
            case 0x0C: SET_GPR_U64(ctx, rt, GPR_U64(ctx, rs) & uimm); return true;
            case 0x0D: SET_GPR_U64(ctx, rt, GPR_U64(ctx, rs) | uimm); return true;
            case 0x0E: SET_GPR_U64(ctx, rt, GPR_U64(ctx, rs) ^ uimm); return true;
            case 0x0F: SET_GPR_S32(ctx, rt, uimm << 16); return true;
            case 0x18: case 0x19: SET_GPR_U64(ctx, rt, GPR_U64(ctx, rs) + static_cast<uint64_t>(static_cast<int64_t>(imm))); return true;
            case 0x1E: SET_GPR_VEC(ctx, rt, READ128(addr & ~15u)); return true;
            case 0x1F: WRITE128(addr & ~15u, GPR_VEC(ctx, rt)); return true;
            case 0x20: SET_GPR_S64(ctx, rt, static_cast<int64_t>(static_cast<int8_t>(READ8(addr)))); return true;
            case 0x21: SET_GPR_S64(ctx, rt, static_cast<int64_t>(static_cast<int16_t>(READ16(addr)))); return true;
            case 0x23: SET_GPR_S32(ctx, rt, READ32(addr)); return true;
            case 0x24: SET_GPR_U64(ctx, rt, READ8(addr)); return true;
            case 0x25: SET_GPR_U64(ctx, rt, READ16(addr)); return true;
            case 0x27: SET_GPR_U64(ctx, rt, READ32(addr)); return true;
            case 0x28: WRITE8(addr, static_cast<uint8_t>(GPR_U32(ctx, rt))); return true;
            case 0x29: WRITE16(addr, static_cast<uint16_t>(GPR_U32(ctx, rt))); return true;
            case 0x2B: WRITE32(addr, GPR_U32(ctx, rt)); return true;
            case 0x2F: return true; // cache
            case 0x31: setFbits(rt, READ32(addr)); return true;
            case 0x33: return true; // pref
            case 0x36: { const __m128i v = READ128(addr & ~15u); ctx->vu0_vf[rt] = _mm_castsi128_ps(v); return true; }
            case 0x37: SET_GPR_U64(ctx, rt, READ64(addr)); return true;
            case 0x39: WRITE32(addr, fbits(rt)); return true;
            case 0x3E: WRITE128(addr & ~15u, _mm_castps_si128(ctx->vu0_vf[rt])); return true;
            case 0x3F: WRITE64(addr, GPR_U64(ctx, rt)); return true;
            case 0x11: return execCop1(instr);
            default: (void)pc; return false;
            }
        }

        bool execCop1(uint32_t instr)
        {
            const uint32_t fmt = (instr >> 21) & 31u;
            const uint32_t ft = (instr >> 16) & 31u;
            const uint32_t fs = (instr >> 11) & 31u;
            const uint32_t fd = (instr >> 6) & 31u;
            const uint32_t funct = instr & 63u;
            switch (fmt)
            {
            case 0x00: SET_GPR_S32(ctx, ft, fbits(fs)); return true;
            case 0x02: SET_GPR_S32(ctx, ft, fs == 31u ? ctx->fcr31 : 0u); return true;
            case 0x04: setFbits(fs, GPR_U32(ctx, ft)); return true;
            case 0x06: if (fs == 31u) ctx->fcr31 = GPR_U32(ctx, ft); return true;
            case 0x10:
            {
                const float a = ctx->f[fs], b = ctx->f[ft];
                switch (funct)
                {
                case 0x00: ctx->f[fd] = a + b; return true;
                case 0x01: ctx->f[fd] = a - b; return true;
                case 0x02: ctx->f[fd] = a * b; return true;
                case 0x03: ctx->f[fd] = (b != 0.0f) ? a / b : 0.0f; return true;
                case 0x04: ctx->f[fd] = std::sqrt(std::fabs(b)); return true;
                case 0x05: ctx->f[fd] = std::fabs(a); return true;
                case 0x06: ctx->f[fd] = a; return true;
                case 0x07: ctx->f[fd] = -a; return true;
                case 0x24: setFbits(fd, static_cast<uint32_t>(static_cast<int32_t>(a))); return true;
                case 0x28: ctx->f[fd] = a > b ? a : b; return true;
                case 0x29: ctx->f[fd] = a < b ? a : b; return true;
                case 0x30: ctx->fcr31 &= ~kFpuCondBit; return true;
                case 0x32: ctx->fcr31 = (a == b) ? (ctx->fcr31 | kFpuCondBit) : (ctx->fcr31 & ~kFpuCondBit); return true;
                case 0x34: ctx->fcr31 = (a < b) ? (ctx->fcr31 | kFpuCondBit) : (ctx->fcr31 & ~kFpuCondBit); return true;
                case 0x36: ctx->fcr31 = (a <= b) ? (ctx->fcr31 | kFpuCondBit) : (ctx->fcr31 & ~kFpuCondBit); return true;
                default: return false;
                }
            }
            case 0x14:
                if (funct == 0x20)
                {
                    ctx->f[fd] = static_cast<float>(static_cast<int32_t>(fbits(fs)));
                    return true;
                }
                return false;
            default: return false;
            }
        }

        bool isControl(uint32_t instr)
        {
            const uint32_t op = instr >> 26;
            if (op == 0x00)
            {
                const uint32_t funct = instr & 63u;
                return funct == 0x08 || funct == 0x09;
            }
            if (op == 0x01)
                return true;
            if (op == 0x11 && ((instr >> 21) & 31u) == 0x08)
                return true;
            return (op >= 0x02 && op <= 0x07) || (op >= 0x14 && op <= 0x17);
        }

        bool execDelaySlot(uint32_t pc)
        {
            const uint32_t instr = fetch(pc + 4u);
            if (isControl(instr) || !execSimple(instr, pc + 4u))
            {
                logOnce("unsupported delay-slot instruction", pc + 4u, pc + 4u, instr);
                return false;
            }
            return true;
        }

        // Unsupported instruction: undo this invocation's register changes and return to the
        // caller, which is what the old skip-missing-call policy did.
        void bail(const R5900Context &entry)
        {
            *ctx = entry;
            ctx->pc = GPR_U32(ctx, 31);
        }

        // Runs from ctx->pc. Returns when control leaves through JR, hands off to generated code,
        // or a call suspends; ctx->pc then holds the next guest pc.
        void run()
        {
            const R5900Context entry = *ctx;
            const uint32_t entryPc = ctx->pc;
            uint32_t pc = entryPc;
            for (uint64_t steps = 0;; ++steps)
            {
                if (pc != entryPc && runtime->hasFunction(pc))
                {
                    ctx->pc = pc;
                    runtime->lookupFunction(pc)(rdram, ctx, runtime);
                    return;
                }
                if ((steps & 0xFFFFu) == 0xFFFFu && runtime->isStopRequested())
                {
                    ctx->pc = pc;
                    return;
                }

                const uint32_t instr = fetch(pc);
                const uint32_t op = instr >> 26;
                const uint32_t rs = (instr >> 21) & 31u;
                const uint32_t rt = (instr >> 16) & 31u;
                const int32_t imm = static_cast<int16_t>(instr & 0xFFFFu);
                const uint32_t branchTarget = pc + 4u + (static_cast<uint32_t>(imm) << 2);

                if (!isControl(instr))
                {
                    if (!execSimple(instr, pc))
                    {
                        logOnce("unsupported instruction", pc, pc, instr);
                        bail(entry);
                        return;
                    }
                    pc += 4u;
                    continue;
                }

                bool taken = false;
                bool likely = false;
                bool isCall = false;
                bool isRegJump = false;
                uint32_t target = branchTarget;

                if (op == 0x00)
                {
                    target = GPR_U32(ctx, rs);
                    isRegJump = true;
                    if ((instr & 63u) == 0x09)
                    {
                        isCall = true;
                        SET_GPR_U32(ctx, (instr >> 11) & 31u, pc + 8u);
                    }
                    taken = true;
                }
                else if (op == 0x02 || op == 0x03)
                {
                    target = ((pc + 4u) & 0xF0000000u) | ((instr & 0x03FFFFFFu) << 2);
                    taken = true;
                    if (op == 0x03)
                    {
                        isCall = true;
                        SET_GPR_U32(ctx, 31, pc + 8u);
                    }
                }
                else if (op == 0x01)
                {
                    const int64_t v = GPR_S64(ctx, rs);
                    switch (rt)
                    {
                    case 0x00: taken = v < 0; break;
                    case 0x01: taken = v >= 0; break;
                    case 0x02: taken = v < 0; likely = true; break;
                    case 0x03: taken = v >= 0; likely = true; break;
                    default:
                        logOnce("unsupported regimm", pc, pc, instr);
                        bail(entry);
                        return;
                    }
                }
                else if (op == 0x11)
                {
                    const bool cond = (ctx->fcr31 & kFpuCondBit) != 0u;
                    taken = (rt & 1u) ? cond : !cond;
                    likely = (rt & 2u) != 0u;
                }
                else
                {
                    const int64_t a = GPR_S64(ctx, rs), b = GPR_S64(ctx, rt);
                    switch (op & 0x03u)
                    {
                    case 0x00: taken = a == b; break;
                    case 0x01: taken = a != b; break;
                    case 0x02: taken = a <= 0; break;
                    default: taken = a > 0; break;
                    }
                    likely = op >= 0x14;
                }

                if (!likely || taken)
                {
                    if (!execDelaySlot(pc))
                    {
                        bail(entry);
                        return;
                    }
                }

                if (!taken)
                {
                    pc += 8u;
                    continue;
                }

                if (isCall)
                {
                    const uint32_t returnPc = pc + 8u;
                    ctx->pc = target;
                    if (!runtime->dispatchGuestBranch(rdram, ctx, target, pc, returnPc,
                                                      isRegJump ? PS2Runtime::GuestBranchKind::IndirectCall
                                                                : PS2Runtime::GuestBranchKind::DirectCall,
                                                      "interp"))
                        return;
                    pc = ctx->pc;
                    continue;
                }

                if (isRegJump)
                {
                    ctx->pc = target;
                    return;
                }

                pc = target;
            }
        }
    };
}

bool ps2InterpCanHandle(uint32_t address)
{
    static const bool enabled = []()
    {
        const char *v = std::getenv("PS2_MISSING_INTERP");
        return v == nullptr || v[0] != '0';
    }();
    return enabled && (address & 3u) == 0u && address >= g_ps2RecompiledFunctionTableBase &&
           address < g_ps2RecompiledFunctionTableEnd;
}

void ps2InterpretGuest(PS2Runtime *runtime, uint8_t *rdram, R5900Context *ctx)
{
    logOnce("enter", ctx->pc, ctx->pc, static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)));
    Interp interp{runtime, rdram, ctx};
    interp.run();
}
