"""Build ps2_vu1_native_blocks.inc from VU1 native block chunks.

The runtime writes one chunk per new block when ATV_VU_NATIVE_GEN=<file> is set
(VU1Native::generate in ps2_vu1_native.inc). Chunks are the plain per-pair
form: every pair reads and writes the interpreter's ready tables and calls one
noinline function per op. This script keeps all chunks in a flat store
(deduplicated by start PC and code words) and emits the compiled form:

  - each block is split into parts of PART_PAIRS pairs (separate noinline
    functions, so MSVC never sees one huge function);
  - upper ops are inlined in their lean form (VU1Native::leanUpper, the flag
    mode of every race program) next to a call of the shared general op for
    the other modes; a part keeps vuNormalize4 of each register it reads in a
    local (n<reg>, nacc) while the register is unchanged, so most operands
    are not normalized again;
  - the ready-table entries a part touches live in locals, loaded once on
    entry and stored on every exit (unwritten ones store back their own value),
    and so do writesReadyAt and the pair counter;
  - VF0/VI0 are reset only after pairs that can write them.

Nothing a part calls reads the ready tables except markPairWrites (multi-VI
pairs), around which the locals are flushed and reloaded. Nothing a part calls
writes a VF register or ACC except execPairGeneric (all caches dropped) and
the ops whose writes the caches track.

    python -I tools/vu1gen/merge_blocks.py <flat-store.txt> <out.inc> [chunks.txt...]
"""
import re
import sys

PART_PAIRS = 128

# VU1Native::LowerKind values.
L_LQ, L_SQ, L_LQI, L_SQI, L_MR32, L_MOVE, L_MFIR, L_MFP = 2, 3, 5, 6, 9, 10, 30, 31
L_XGKICK, L_FCSET, L_BAL, L_JALR = 34, 37, 39, 40
# Lower kinds folded into the block (VU1Native::fastLower inline): small integer,
# load/store, move and branch ops, DIV, XTOP/XITOP, XGKICK, FCSET, WAITP, BAL and
# JALR. ELENG and generic ops stay calls.
INLINE_LOWER = set(range(2, 33)) | set(range(34, 41))
# Inline lower kinds that read the pair's PC: IBEQ, IBNE, B, JR, IBLTZ, IBGTZ,
# IBLEZ, IBGEZ, BAL, JALR (branch targets and links); XGKICK and FCSET (their
# fault logs).
PC_LOWER = {7, 8, 17, 18, 19, 20, 21, 22, L_BAL, L_JALR, L_XGKICK, L_FCSET}
# Lower kinds that write VF[FT] (all others write no VF register, and no fast
# lower op writes VI0).
VF_LOWER = {L_LQ, L_LQI, L_MR32, L_MOVE, L_MFIR, L_MFP}

# VU1Native::UpperKind values.
(K_EXEC, K_NOP, K_ADDBC, K_SUBBC, K_MADDBC, K_MSUBBC, K_MAXBC, K_MINIBC, K_MULBC, K_ADD, K_SUB, K_MUL, K_MAX,
 K_MINI, K_MULQ, K_MINII, K_MULABC, K_MADDABC, K_MULI, K_ADDI, K_CLIP, K_FTOI0, K_FTOI4, K_ITOF0, K_ADDABC,
 K_MULA, K_SUBI, K_OPMULA, K_OPMSUB, K_ITOF12, K_ABS, K_SUBABC, K_MSUBABC, K_MADD, K_MSUB, K_MADDQ, K_MADDA,
 K_MAXI, K_ITOF4, K_FTOI12) = range(40)
BC_KINDS = {K_ADDBC, K_SUBBC, K_MADDBC, K_MSUBBC, K_MAXBC, K_MINIBC, K_MULBC, K_MULABC, K_MADDABC, K_ADDABC,
            K_SUBABC, K_MSUBABC}
VECTOR_KINDS = {K_ADD, K_SUB, K_MUL, K_MAX, K_MINI, K_MULA, K_OPMULA, K_OPMSUB, K_MADD, K_MSUB, K_MADDA}
ACC_READ_KINDS = {K_MADDBC, K_MSUBBC, K_MADDABC, K_MSUBABC, K_OPMSUB, K_MADD, K_MSUB, K_MADDA, K_MADDQ}
TO_ACC_KINDS = {K_MULABC, K_MADDABC, K_ADDABC, K_MULA, K_OPMULA, K_SUBABC, K_MSUBABC, K_MADDA}
ITOF_KINDS = {K_ITOF0, K_ITOF4, K_ITOF12}
FTOI_KINDS = {K_FTOI0, K_FTOI4, K_FTOI12}
FT_DEST_KINDS = ITOF_KINDS | FTOI_KINDS | {K_ABS}
# Kinds without a lean form (VU1Native::leanUpper).
NOT_LEAN = {K_EXEC, K_NOP, K_CLIP}


def needs_pc(lines):
    """Whether the pair's code may read s.pc: branches, generic and
    out-of-line lower ops, and upper ops on the general execUpper path
    (kind 0; it prints the PC in diagnostics). Exits set the PC themselves."""
    for line in lines:
        if line.startswith(('VU1Native::execPairGeneric(', 'VU1Native::markGeneric(')):
            return True
        m = LOWER_RE.match(line)
        if m and (int(m.group(1)) in PC_LOWER or int(m.group(1)) not in INLINE_LOWER):
            return True
        m = UPPER_RE.match(line) or UPPER_NOMAC_RE.match(line)
        if m and int(m.group(1)[:-3], 16) & 0xFF == 0:
            return True
    return False


BLOCK_RE = re.compile(r'^// ATV-VU1-BLOCK pc=0x([0-9a-f]+) pairs=(\d+) key=([0-9a-f]+).*?^// ATV-VU1-END\n',
                      re.M | re.S)
WORDS_RE = re.compile(r'static const uint32_t (vu1nbw_\w+)\[\] = \{(.*?)\};', re.S)
PAIR_RE = re.compile(r'^    // (0x[0-9a-f]+ lower=\w+ upper=\w+)\n    \{\n(.*?)^    \}\n', re.M | re.S)

READ_RE = re.compile(r'^t = std::max\(t, c\.(vf|vi|acc)Ready\[(\d+)\]\);$')
WRITE_RE = re.compile(r'^c\.(vf|vi|acc)Ready\[(\d+)\] = cyc \+ (\d+)u;$')
UPPER_RE = re.compile(r'^VU1Native::upperOp<(0x[0-9a-f]+ull), (0x[0-9a-f]+u)>\(vu\);$')
# An upper op whose MAC push dead_mac_pairs proved unobservable.
UPPER_NOMAC_RE = re.compile(r'^VU1Native::upperOpNoMac<(0x[0-9a-f]+ull), (0x[0-9a-f]+u)>\(vu\);$')
LOWER_RE = re.compile(r'^VU1Native::lowerOp<(\d+), (0x[0-9a-f]+u), (0x[0-9a-f]+u)>\(c\);$')
WRA_RE = re.compile(r'^\*c\.writesReadyAt = std::max\(\*c\.writesReadyAt, cyc \+ (\d+)u\);$')
RETURN_RE = re.compile(r'return (VU1Native::[\w:]+(?:\(c, 0x[0-9a-f]+u\))?);')
TABLE = {'vf': 'vfReady', 'vi': 'viReady', 'acc': 'accReady'}


def table(kind):
    """The ready table, addressed from `vu` (kept in a register)."""
    return f'VU1Native::{TABLE[kind]}Of(vu)'

HEADER = """// Natively compiled VU1 blocks. Generated by tools/vu1gen/merge_blocks.py
// from ATV_VU_NATIVE_GEN output; do not edit by hand.
"""


def flat_blocks(text):
    for m in BLOCK_RE.finditer(text):
        body = m.group(0)
        words = WORDS_RE.search(body).group(2).split()
        yield int(m.group(1), 16), int(m.group(2)), m.group(3), words, body


def local(kind, index):
    return f'r{kind}{index}'


READY4_RE = re.compile(r'^@ready4 (vf|acc) (\d+) (\d+)$')


def collapse_ready4(lines, slots):
    """Four consecutive table writes of lanes 0-3 of one VF register (or ACC)
    with one latency, none held in a local, become one '@ready4' line."""
    out = []
    i = 0
    while i < len(lines):
        m = WRITE_RE.match(lines[i])
        if m and m.group(1) in ('vf', 'acc') and int(m.group(2)) % 4 == 0 and i + 3 < len(lines):
            kind, base, lat = m.group(1), int(m.group(2)), m.group(3)
            group = [WRITE_RE.match(lines[i + k]) for k in range(4)]
            if all(g and g.group(1) == kind and int(g.group(2)) == base + k and g.group(3) == lat and
                   (kind, base + k) not in slots for k, g in enumerate(group)):
                out.append(f'@ready4 {kind} {base} {lat}')
                i += 4
                continue
        out.append(lines[i])
        i += 1
    return out


def unpack(packed):
    """VU1Native::unpackFastOp: kind, fd, fs, ft, dest, bc."""
    p = int(packed[:-3], 16)
    return p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF, (p >> 24) & 0xFF, (p >> 32) & 0xFF, (p >> 40) & 0xFF


def cache_name(reg):
    return 'nacc' if reg == 'acc' else f'n{reg}'


# VF0 as every pair starts with it (already normalized). Read from memory, never
# a constant: under /fp:fast the compiler would fold x * 0 and x + 0 and lose the
# sign of zero results.
VF0 = '_mm_loadu_ps(s.vf[0])'


class Caches:
    """Which registers have vuNormalize4 of their current value in a local
    (never VF0, see VF0)."""

    def __init__(self):
        self.valid = set()
        self.used = set()

    def operand(self, reg, out):
        """The normalized-register expression, loading the cache if needed."""
        if reg == 0:
            return VF0
        name = cache_name(reg)
        self.used.add(reg)
        if reg not in self.valid:
            src = 's.acc' if reg == 'acc' else f's.vf[{reg}]'
            out.append(f'{name} = VU1Native::normLoad({src});')
            self.valid.add(reg)
        return name

    def drop(self, reg):
        self.valid.discard(reg)


def lean_upper(packed, upper, nomac, caches):
    """The inline lean form of an upper op line, or None (no lean form).
    Returns (lines, reg it writes or None)."""
    kind, fd, fs, ft, dest, _ = unpack(packed)
    if kind in NOT_LEAN:
        return None
    body = []
    if kind in ITOF_KINDS:
        vs = f'_mm_loadu_ps(s.vf[{fs}])'
    else:
        vs = caches.operand(fs, body)
    vt = caches.operand(ft, body) if kind in BC_KINDS or kind in VECTOR_KINDS else '_mm_setzero_ps()'
    va = caches.operand('acc', body) if kind in ACC_READ_KINDS else '_mm_setzero_ps()'
    if kind in TO_ACC_KINDS:
        reg, dst = 'acc', 's.acc'
    else:
        reg = ft if kind in FT_DEST_KINDS else fd
        dst = f's.vf[{reg}]'
    body.append('__m128 nr;')
    body.append(f'const __m128 r = VU1Native::leanUpper<{packed}, {upper}, {"true" if nomac else "false"}>'
                f'(vu, macOnly, {vs}, {vt}, {va}, nr);')
    if dest != 0:
        body.append(f'VU1Native::storeDest<{dest}>({dst}, r);')
        if reg != 0:
            if kind in FTOI_KINDS:
                caches.drop(reg)
            elif dest == 0xF:
                body.append(f'{cache_name(reg)} = nr;')
                caches.used.add(reg)
                caches.valid.add(reg)
            elif reg in caches.valid:
                body.append(f'{cache_name(reg)} = VU1Native::mergeDest<{dest}>({cache_name(reg)}, nr);')
    out = ['if (lean)', '{'] + ['    ' + b for b in body] + ['}', 'else', f'    VU1Native::upperOpGeneral<{packed}, {upper}>(vu);']
    return out, (reg if dest != 0 else None)


def lower_inline(kind, lower, upper, caches, shadow):
    """The inline form of a fast lower op, and the VF register it writes (or None).
    shadow: the pair's lower half sees the old value of a register the upper half
    wrote (and whose cache it already updated), so copies keep no cache."""
    lw = int(lower[:-1], 16)
    ft, fs, dest = (lw >> 16) & 0x1F, (lw >> 11) & 0x1F, (lw >> 21) & 0xF
    special = {L_LQ: 'lq', L_SQ: 'sq', L_LQI: 'lqi', L_SQI: 'sqi', L_MOVE: 'moveVf', L_MR32: 'mr32'}
    if kind in special:
        name = special[kind]
        args = 's, vuData' if kind in (L_LQ, L_SQ, L_LQI, L_SQI) else 's'
        out = [f'VU1Native::{name}<{lower}>({args});']
    else:
        out = [f'VU1Native::fastLower(vu, std::integral_constant<uint8_t, {kind}>{{}}, '
               f'std::integral_constant<uint32_t, {lower}>{{}}, {upper}, '
               'c.vuData, c.dataSize, *c.gs, c.memory);']
    if kind not in VF_LOWER:
        return out, None
    if ft != 0:
        if kind in (L_MOVE, L_MR32) and dest == 0xF and not shadow and fs != 0 and fs in caches.valid:
            src = caches.operand(fs, out)
            # A copy of a register normalizes to the copy of its normalized value.
            value = src if kind == L_MOVE else f'_mm_shuffle_ps({src}, {src}, _MM_SHUFFLE(0, 3, 2, 1))'
            out.append(f'{cache_name(ft)} = {value};')
            caches.used.add(ft)
            caches.valid.add(ft)
        else:
            caches.drop(ft)
    return out, ft


DEAD_STATS = [0, 0, 0, 0]


def drop_dead_stores(pairs, slots):
    """Within a part, drops a ready-table write (of a slot the part keeps no local
    for) that a later pair rewrites, and a writesReadyAt update a later pair's
    dominates (each pair is at least a cycle later), when nothing between can
    leave the part: a pair's exits (stall bail, XGKICK wait) come before its own
    writes, the program-end exit after them. Generic pairs count as exits."""
    def leaves(lines):
        return any('return' in l or l.startswith(('VU1Native::execPairGeneric(', 'VU1Native::markGeneric('))
                   for l in lines)
    def ends(lines):
        return any(l.startswith('if (ended) return') for l in lines)
    writes = [{(m.group(1), int(m.group(2))) for m in map(WRITE_RE.match, lines) if m} for _, lines in pairs]
    wra = [max([int(m.group(1)) for m in map(WRA_RE.match, lines) if m] or [0]) for _, lines in pairs]
    out = []
    for i, (comment, lines) in enumerate(pairs):
        dead = set()
        wra_dead = False
        if not ends(lines):
            pending = {w for w in writes[i] if w not in slots}
            for j in range(i + 1, len(pairs)):
                if leaves(pairs[j][1]):
                    break
                dead |= pending & writes[j]
                pending -= writes[j]
                wra_dead |= wra[i] > 1 and wra[j] >= wra[i] - (j - i)
                if ends(pairs[j][1]):
                    break
        kept = []
        for line in lines:
            m = WRITE_RE.match(line)
            if m:
                DEAD_STATS[1] += 1
                if (m.group(1), int(m.group(2))) in dead:
                    DEAD_STATS[0] += 1
                    continue
            m = WRA_RE.match(line)
            if m and int(m.group(1)) > 1:
                DEAD_STATS[3] += 1
                if wra_dead:
                    DEAD_STATS[2] += 1
                    continue
            kept.append(line)
        out.append((comment, kept))
    return out


def emit_part(name, pairs, last):
    """pairs: list of (comment, [stripped body lines])."""
    # Only slots the part still checks live in locals; slots it only writes
    # (most, once stall checks are pruned) are stored straight to the table.
    slots = []
    for _, lines in pairs:
        for line in lines:
            m = READ_RE.match(line)
            if m and (m.group(1), int(m.group(2))) not in slots:
                slots.append((m.group(1), int(m.group(2))))
    pairs = drop_dead_stores(pairs, set(slots))
    written = set()
    for _, lines in pairs:
        for line in lines:
            m = WRITE_RE.match(line)
            if m and (m.group(1), int(m.group(2))) in slots:
                written.add((m.group(1), int(m.group(2))))
    out = [f'static __declspec(noinline) uint32_t {name}(VU1Native::Ctx &c)', '{',
           '    VU1Interpreter &vu = *c.vu;', '    VU1State &s = vu.state();', '    uint64_t cyc = *c.cycle;',
           '    const uint64_t *const nextReady = &c.hint->nextReady;',
           '    uint8_t *const vuData = c.vuData;',
           '    const bool lean = VU1Native::leanInline();',
           '    const bool macOnly = g_vuMacOnlyFlags;',
           '    (void)vu;',
           '    (void)vuData;',
           '    (void)lean;',
           '    (void)macOnly;',
           f'    uint32_t rc = {"VU1Native::kNativeContinue" if last else "VU1Native::kNativePartDone"};',
           '    uint64_t wra = *c.writesReadyAt;', '    uint64_t np = 0;']
    cache_decl_at = len(out)
    for kind, index in slots:
        out.append(f'    uint64_t {local(kind, index)} = {table(kind)}[{index}];')
    flush = [f'{table(k)}[{i}] = {local(k, i)};' for k, i in slots if (k, i) in written]
    reload = [f'{local(k, i)} = {table(k)}[{i}];' for k, i in slots]
    caches = Caches()
    # Whether viBackupValid is known false here: only recordVi and the
    # out-of-line generic/lower calls can set it, so repeated clears between
    # them are dropped (branch ops read it before their pair clears it).
    backup_clear = False
    for pair_index, (comment, lines) in enumerate(pairs):
        out.append(f'    // {comment}')
        out.append('    {')
        keep_pc = needs_pc(lines)
        shadow = any(l.startswith('const __m128 shadowOld = ') for l in lines)
        counted = False  # past this pair's pair-count line
        writes_zero = False  # the pair may write VF0 or VI0
        for line in collapse_ready4(lines, slots):
            m = READY4_RE.match(line)
            if m:
                out.append(f'        VU1Native::setReady4({table(m.group(1))} + {m.group(2)}, cyc + {m.group(3)}u);')
                continue
            if line.startswith('s.pc = ') and not keep_pc:
                continue
            if line == '*c.viBackupValid = false;':
                # recordVi later in the pair sets it again (it is only emitted
                # for a VI register other than vi0).
                if backup_clear or any(l.startswith('VU1Native::recordVi(') for l in lines):
                    continue
                backup_clear = True
            elif line.startswith(('VU1Native::recordVi(', 'VU1Native::execPairGeneric(', 'VU1Native::markGeneric(',
                                  'VU1Native::lowerOp<')):
                backup_clear = False
            m = READ_RE.match(line)
            if m:
                out.append(f'        t = std::max(t, {local(m.group(1), int(m.group(2)))});')
                continue
            m = WRITE_RE.match(line)
            if m:
                key = (m.group(1), int(m.group(2)))
                if key in slots:
                    out.append(f'        {local(*key)} = cyc + {m.group(3)}u;')
                else:
                    out.append(f'        {table(key[0])}[{key[1]}] = cyc + {m.group(3)}u;')
                continue
            m = UPPER_NOMAC_RE.match(line) or UPPER_RE.match(line)
            if m:
                nomac = UPPER_NOMAC_RE.match(line) is not None
                lean = lean_upper(m.group(1), m.group(2), nomac, caches)
                if lean is None:
                    out.append(f'        VU1Native::upperOp<{m.group(1)}, {m.group(2)}{", true" if nomac else ""}>(vu);')
                    kind, fd, _, ft, dest, _ = unpack(m.group(1))
                    writes_zero |= kind == K_EXEC
                else:
                    out.extend('        ' + l for l in lean[0])
                    writes_zero |= lean[1] == 0
                continue
            m = LOWER_RE.match(line)
            if m:
                kind = int(m.group(1))
                if kind in INLINE_LOWER:
                    code, reg = lower_inline(kind, m.group(2), m.group(3), caches, shadow)
                    out.extend('        ' + l for l in code)
                    writes_zero |= reg == 0
                else:
                    out.append(f'        VU1Native::lowerOp<{m.group(1)}, {m.group(2)}, {m.group(3)}>(c);')
                continue
            m = WRA_RE.match(line)
            if m:
                # writesReadyAt only feeds the drain at the end of run(), which
                # advances to it; after this pair the cycle is already >= cyc + 1.
                if int(m.group(1)) > 1:
                    out.append(f'        wra = std::max(wra, cyc + {m.group(1)}u);')
                continue
            if line == '++g_vu1PairTotal;':
                counted = True
                continue
            if line == 'VU1Native::resetZero(s);':
                if writes_zero:
                    out.append('        ' + line)
                continue
            if line.startswith('VU1Native::execPairGeneric('):
                caches.valid.clear()
                writes_zero = True
            if line.startswith('VU1Native::markGeneric('):
                out.extend('        ' + f for f in flush)
                out.append('        ' + line)
                out.extend('        ' + r for r in reload)
                continue
            if line == 'VU1Native::endCycle(c, s);':
                out.append('        VU1Native::endCycleLocal(vu, cyc, nextReady);')
                continue
            # Interpreter state through `vu` rather than Ctx pointers.
            if line == '*c.viBackupValid = false;':
                line = 'VU1Native::clearViBackup(vu);'
            line = line.replace('VU1Native::recordVi(c, ', 'VU1Native::recordVi(vu, ')
            done = pair_index + (1 if counted else 0)
            line = RETURN_RE.sub(lambda r: '{ rc = ' + r.group(1) + f'; np = {done}; goto out; }}', line)
            if 'return' in line:
                raise SystemExit(f'unhandled return in {name}: {line}')
            out.append('        ' + line)
            # The cycle lives in the local `cyc`; these calls move m_cycle.
            if line == 'VU1Native::advanceTo(c, t);' or line.startswith('if (!VU1Native::waitXgkick(c))'):
                out.append('        cyc = VU1Native::cycleOf(vu);')
        out.append('    }')
    out[cache_decl_at:cache_decl_at] = [f'    __m128 {cache_name(r)} = _mm_setzero_ps();'
                                        for r in sorted(caches.used, key=lambda r: (r == 'acc', r if r != 'acc' else 0))]
    out.append(f'    np = {len(pairs)};')
    out.append('out:')
    out.extend('    ' + f for f in flush)
    out += ['    *c.writesReadyAt = wra;', '    g_vu1PairTotal += np;', '    g_vu1NativePairs += np;', '    return rc;', '}']
    return out


# Longest latency of any VF/VI/ACC ready-table write (FMAC/LQ 4, VI/ACC 1;
# FDIV/EFU write Q/P, not the tables). VU1Interpreter::run resets the tables
# with the cycle, so a slot is never further ahead than this.
MAX_WRITE_LATENCY = 4

STALL_BLOCK = ['if (t > cyc)', '{', None, 'VU1Native::advanceTo(c, t);', '}']


def prune_stalls(pairs):
    """Drops ready checks that cannot stall.

    Every pair ends in endCycle (cycle + 1) and stalls only move the cycle
    forward, so when pair i sets a slot to cyc + L, pair j >= i + L of the
    same block already runs at or past that cycle: its check of the slot is a
    no-op. Parts of a block run in order, so this holds across parts.
    A slot not written in the block was last set before the block's first
    pair (cycle <= entry - 1) with a latency <= MAX_WRITE_LATENCY, so pair
    j >= MAX_WRITE_LATENCY - 1 cannot stall on it either. markGeneric
    rewrites the tables, which counts as writing every slot at that pair.
    A pair left without checks also loses its `t` and stall branch.
    """
    known = {}
    base = -1  # pair whose writes may have set any slot not in `known`
    pruned = 0
    out = []
    for index, (comment, lines) in enumerate(pairs):
        kept = []
        for line in lines:
            m = READ_RE.match(line)
            if m:
                slot = (m.group(1), int(m.group(2)))
                if slot in known:
                    if index - known[slot][0] >= known[slot][1]:
                        pruned += 1
                        continue
                elif index - base >= MAX_WRITE_LATENCY:
                    pruned += 1
                    continue
            kept.append(line)
            m = WRITE_RE.match(line)
            if m:
                latency = int(m.group(3))
                if latency > MAX_WRITE_LATENCY:
                    raise SystemExit(f'ready-table latency {latency} > MAX_WRITE_LATENCY: {line}')
                known[(m.group(1), int(m.group(2)))] = (index, latency)
            elif line.startswith('VU1Native::markGeneric('):
                known.clear()
                base = index
        if kept and kept[0] == 'uint64_t t = cyc;' and not any(l.startswith('t = std::max(t,') or
                                                                  l.startswith('if (*c.xgkickActive) t =')
                                                                  for l in kept):
            at = kept.index('if (t > cyc)')
            block = kept[at:at + len(STALL_BLOCK)]
            if (len(block) == len(STALL_BLOCK) and block[2].startswith('if (t >= c.budgetEnd) return') and
                    all(b == s for b, s in zip(block, STALL_BLOCK) if s is not None)):
                kept = kept[1:at] + kept[at + len(STALL_BLOCK):]
        out.append((comment, kept))
    return out, pruned


# UpperKind values whose fast form pushes a MAC value in MAC-only mode when
# dest != 0: every FMAC kind (not MAX/MINI, ITOF/FTOI, ABS or CLIP). 0 (general
# execUpper) is unknown.
MAC_WRITER_KINDS = {K_ADDBC, K_SUBBC, K_MADDBC, K_MSUBBC, K_MULBC, K_ADD, K_SUB, K_MUL, K_MULQ, K_MULABC, K_MADDABC,
                    K_MULI, K_ADDI, K_ADDABC, K_MULA, K_SUBI, K_OPMULA, K_OPMSUB, K_SUBABC, K_MSUBABC, K_MADD, K_MSUB,
                    K_MADDQ, K_MADDA}
DEADMAC_STATS = [0, 0]


def dead_mac_pairs(pairs):
    """Pairs whose MAC push no instruction can see.

    MAC pushed at pair i is visible from cyc_i + 4; the next pusher j is
    visible from cyc_j + 4 <= cyc_k for any pair k > j + 3. So if pairs
    i+1 .. j+3 (all in this block) hold no MAC reader, no generic or
    general-path op (unknown flag use) and no exit (bail, XGKICK wait, end),
    nothing can read i's value before j's replaces it."""
    def classify(lines):
        writer = barrier = False
        for line in lines:
            m = UPPER_RE.match(line)
            if m:
                packed = int(m.group(1)[:-3], 16)
                kind, dest = packed & 0xFF, (packed >> 32) & 0xFF
                if kind == 0:
                    barrier = True
                elif kind in MAC_WRITER_KINDS and dest != 0:
                    writer = True
            # MAC readers (FMEQ/FMAND/FMOR) only run through execPairGeneric;
            # fast lower kinds (lowerOp lines here) never read MAC.
            if line.startswith(('VU1Native::execPairGeneric(', 'VU1Native::markGeneric(')):
                barrier = True
            if 'return' in line or 'applyPendingMac' in line:
                barrier = True
        return writer, barrier

    info = [classify(lines) for _, lines in pairs]
    dead = set()
    for i, (writer, _) in enumerate(info):
        if not writer:
            continue
        j = next((k for k in range(i + 1, len(pairs)) if info[k][0]), None)
        if j is None or j + 3 >= len(pairs):
            continue
        if not any(info[k][1] for k in range(i + 1, j + 4)):
            dead.add(i)
    return dead


PRUNE_STATS = [0, 0]
# Lower kinds that read the VI branch backup (VU1Interpreter::readBranchVi): IBEQ,
# IBNE, IBLTZ, IBGTZ, IBLEZ, IBGEZ.
BACKUP_READERS = {7, 8, 19, 20, 21, 22}
RECORD_RE = re.compile(r'^VU1Native::recordVi\(c, (\d+), oldVi\);$')
RECORD_STATS = [0, 0]


def drop_dead_records(pairs):
    """A pair's VI branch backup is only read by the next pair (the one after
    clears it), and only by a conditional branch or a generic pair. Drops the
    record (and its saved value) when the next pair in the block is neither; the
    pair then clears the backup like any other. The last pair keeps it: the next
    block may start with a branch."""
    out = []
    for n, (comment, lines) in enumerate(pairs):
        if n + 1 < len(pairs) and any(RECORD_RE.match(l) for l in lines):
            RECORD_STATS[1] += 1
            following = pairs[n + 1][1]
            reads = any(l.startswith('VU1Native::execPairGeneric(') or
                        (LOWER_RE.match(l) and int(LOWER_RE.match(l).group(1)) in BACKUP_READERS)
                        for l in following)
            if not reads:
                RECORD_STATS[0] += 1
                lines = [l for l in lines if not RECORD_RE.match(l) and not l.startswith('const int32_t oldVi = ')]
        out.append((comment, lines))
    return out


def emit_block(pc, key, words, body):
    pairs = []
    for m in PAIR_RE.finditer(body):
        lines = [l.strip() for l in m.group(2).split('\n') if l.strip()]
        pairs.append((m.group(1), lines))
    PRUNE_STATS[1] += sum(1 for _, ls in pairs for l in ls if READ_RE.match(l))
    pairs, pruned = prune_stalls(pairs)
    PRUNE_STATS[0] += pruned
    pairs = drop_dead_records(pairs)
    # Mark dead MAC pushes (after pruning: fewer stall exits).
    dead = dead_mac_pairs(pairs)
    DEADMAC_STATS[0] += len(dead)
    DEADMAC_STATS[1] += sum(1 for _, ls in pairs for l in ls if UPPER_RE.match(l))
    pairs = [(c, [l.replace('VU1Native::upperOp<', 'VU1Native::upperOpNoMac<') if (n in dead and UPPER_RE.match(l)) else l
                  for l in ls])
             for n, (c, ls) in enumerate(pairs)]
    if len(pairs) * 2 != len(words) // 2 * 2 or len(pairs) != len(words) // 2:
        raise SystemExit(f'pair parse mismatch at pc=0x{pc:04x}')
    base = f'vu1nb_{pc:04x}_{key}'
    out = [f'// block pc=0x{pc:04x} pairs={len(pairs)}',
           f'static const uint32_t vu1nbw_{pc:04x}_{key}[] = {{']
    for n in range(0, len(words), 2):
        out.append(f'    {words[n]} {words[n + 1]}')
    out.append('};')
    parts = [pairs[i:i + PART_PAIRS] for i in range(0, len(pairs), PART_PAIRS)]
    for n, part in enumerate(parts):
        out += emit_part(f'{base}_p{n}', part, n + 1 == len(parts))
    out += [f'static uint32_t {base}(VU1Native::Ctx &c)', '{']
    for n in range(len(parts) - 1):
        out.append(f'    if (const uint32_t r = {base}_p{n}(c); r != VU1Native::kNativePartDone) return r;')
    out += [f'    return {base}_p{len(parts) - 1}(c);', '}', '']
    return '\n'.join(out) + '\n'


def main():
    flat_path, out_path = sys.argv[1], sys.argv[2]
    blocks = {}
    texts = []
    for path in [flat_path] + sys.argv[3:]:
        try:
            with open(path, encoding='utf-8') as f:
                texts.append(f.read())
        except FileNotFoundError:
            continue
    for text in texts:
        for pc, pairs, key, words, body in flat_blocks(text):
            ident = (pc, key)
            if ident in blocks:
                if blocks[ident][1] != words:
                    raise SystemExit(f'key collision at pc=0x{pc:04x} key={key}')
                continue
            blocks[ident] = (pairs, words, body)
    order = sorted(blocks)
    with open(flat_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(''.join(blocks[i][2] for i in order))
    parts = [HEADER]
    for pc, key in order:
        parts.append(emit_block(pc, key, blocks[(pc, key)][1], blocks[(pc, key)][2]))
    parts.append('const VU1Native::BlockEntry *VU1Native::registry(size_t &count)\n{\n')
    parts.append('    static const BlockEntry kEntries[] = {\n')
    for pc, key in order:
        parts.append(f'        {{0x{pc:04x}u, {blocks[(pc, key)][0]}u, vu1nbw_{pc:04x}_{key}, vu1nb_{pc:04x}_{key}}},\n')
    parts.append('        {0xFFFFFFFFu, 0u, nullptr, nullptr}, // sentinel\n')
    parts.append('    };\n')
    parts.append('    count = sizeof(kEntries) / sizeof(kEntries[0]) - 1u;\n')
    parts.append('    return kEntries;\n}\n')
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(''.join(parts))
    print(f'{len(order)} blocks, {sum(blocks[i][0] for i in order)} pairs -> {out_path}')
    print(f'ready checks pruned: {PRUNE_STATS[0]} of {PRUNE_STATS[1]}')
    print(f'dead MAC pushes: {DEADMAC_STATS[0]} of {DEADMAC_STATS[1]} upper ops')
    print(f'dead VI branch records: {RECORD_STATS[0]} of {RECORD_STATS[1]}')
    print(f'dead ready-table writes: {DEAD_STATS[0]} of {DEAD_STATS[1]}, writesReadyAt updates: {DEAD_STATS[2]} of {DEAD_STATS[3]}')


if __name__ == '__main__':
    main()
