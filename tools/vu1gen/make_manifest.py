"""Write vu1_blocks_manifest.txt: where each VU1 native block's code is in SCUS_972.11.

The compiled blocks (ps2_vu1_native_blocks.inc) are translations of the game's
VU1 microcode, so the repository keeps only this manifest, and
make_vu1_blocks.py rebuilds the blocks from the user's own ELF. Run this after
adding blocks to the flat store (ATV_VU_NATIVE_GEN runs merged with
merge_blocks.py):

    python -I tools/vu1gen/make_manifest.py <vu1_blocks_flat.txt> <SCUS_972.11> [manifest]

Each line is "<pc> <pairs> <key> <offset>:<pairs>...": the block's start PC,
pair count, FNV-1a key of its code words (the key in its name), and the ELF
file offset and pair count of each run of its code. The game uploads
microcode in MPG packets, so one block can span two of them.
"""
import hashlib
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
from merge_blocks import WORDS_RE, flat_blocks  # noqa: E402


def find_runs(data, code, pairs):
    """Splits the block's code into the longest runs found in the ELF."""
    runs, i, prev_end = [], 0, None
    while i < pairs:
        lo, hi, best = 1, pairs - i, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if data.find(code[i * 8:(i + mid) * 8]) >= 0:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if not best:
            return None
        seg = code[i * 8:(i + best) * 8]
        hits, at = [], data.find(seg)
        while at >= 0:
            hits.append(at)
            at = data.find(seg, at + 1)
        # Prefer the hit nearest the previous run (the next MPG packet).
        off = hits[0] if prev_end is None else min(hits, key=lambda h: abs(h - prev_end))
        runs.append((off, best))
        prev_end = off + best * 8
        i += best
    return runs


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    flat_path, elf_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, 'vu1_blocks_manifest.txt')
    with open(flat_path, encoding='utf-8') as f:
        text = f.read()
    with open(elf_path, 'rb') as f:
        data = f.read()
    lines = ['# VU1 native blocks: <pc> <pairs> <key> <elf offset>:<pairs>... (tools/vu1gen/make_manifest.py)',
             f'# sha1 {hashlib.sha1(data).hexdigest()} {os.path.basename(elf_path)}']
    missing = 0
    for pc, pairs, key, _, body in flat_blocks(text):
        words = [int(w, 16) for w in re.findall(r'0x([0-9a-f]{8})u', WORDS_RE.search(body).group(2))]
        runs = find_runs(data, struct.pack(f'<{len(words)}I', *words), pairs)
        if runs is None:
            missing += 1
            print(f'pc=0x{pc:04x} key={key}: code not found in {elf_path}', file=sys.stderr)
            continue
        lines.append(f'{pc:04x} {pairs} {key} ' + ' '.join(f'{off:x}:{n}' for off, n in runs))
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{len(lines) - 2} blocks -> {out_path}' + (f' ({missing} not found, left out)' if missing else ''))


if __name__ == '__main__':
    main()
