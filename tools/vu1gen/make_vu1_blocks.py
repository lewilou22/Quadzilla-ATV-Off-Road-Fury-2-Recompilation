"""Build ps2_vu1_native_blocks.inc from your own copy of the game.

The compiled VU1 blocks are translations of the game's VU1 microcode, so they
are not in the repository. This reads each block's code from your SCUS_972.11
at the places vu1_blocks_manifest.txt lists, checks it against the block's key,
has the runner generate the C++ (ATV_VU_NATIVE_BUILD) and merges it with
merge_blocks.py. Build the runner once first, then:

    python -I tools/vu1gen/make_vu1_blocks.py <SCUS_972.11> [ps2EntryRunner.exe]

and build it again. The runner builds without the file too, but every VU1
program then runs on the slower fast loop.
"""
import argparse
import filecmp
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VU_DIR = os.path.join(ROOT, 'ps2xRuntime', 'src', 'lib', 'vu')


def fnv1a(words):
    """VU1Native::generate's block key."""
    key = 1469598103934665603
    for w in words:
        for b in range(4):
            key = ((key ^ ((w >> (8 * b)) & 0xFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return key


def read_manifest(path):
    sha1, entries = None, []
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.split()
            if fields[:2] == ['#', 'sha1']:
                sha1 = fields[2]
            if not fields or fields[0].startswith('#'):
                continue
            runs = [tuple(int(v, 16 if i == 0 else 10) for i, v in enumerate(r.split(':'))) for r in fields[3:]]
            entries.append((int(fields[0], 16), int(fields[1]), int(fields[2], 16), runs))
    return sha1, entries


def main():
    ap = argparse.ArgumentParser(description='Build ps2_vu1_native_blocks.inc from your SCUS_972.11.')
    ap.add_argument('elf', help='the game ELF (SCUS_972.11 from your disc)')
    ap.add_argument('runner', nargs='?', default=os.path.join(ROOT, 'build64', 'ps2xRuntime', 'ps2EntryRunner.exe'),
                    help='a built ps2EntryRunner (default: build64)')
    ap.add_argument('--manifest', default=os.path.join(HERE, 'vu1_blocks_manifest.txt'))
    ap.add_argument('--out', default=os.path.join(VU_DIR, 'ps2_vu1_native_blocks.inc'))
    ap.add_argument('--flat', default=os.path.join(HERE, 'vu1_blocks_flat.txt'),
                    help='flat block store to merge into (kept for ATV_VU_NATIVE_GEN work)')
    args = ap.parse_args()

    with open(args.elf, 'rb') as f:
        data = f.read()
    sha1, entries = read_manifest(args.manifest)
    if sha1 and hashlib.sha1(data).hexdigest() != sha1:
        print(f'warning: {args.elf} is not the ELF the manifest was made from (sha1 {sha1})', file=sys.stderr)
    lines, bad = [], 0
    for pc, pairs, key, runs in entries:
        code = b''.join(data[off:off + n * 8] for off, n in runs)
        words = struct.unpack(f'<{pairs * 2}I', code) if len(code) == pairs * 8 else None
        if words is None or fnv1a(words) != key:
            bad += 1
            continue
        lines.append(f'{pc:x} {pairs} ' + ' '.join(f'{w:x}' for w in words))
    if bad:
        raise SystemExit(f'{bad} of {len(entries)} blocks do not match {args.elf}: use the SCUS_972.11 '
                         'the recomp is built from')
    if not os.path.isfile(args.runner):
        raise SystemExit(f'{args.runner} not found: build the runner first')

    with tempfile.TemporaryDirectory() as tmp:
        words_path = os.path.join(tmp, 'words.txt')
        chunks_path = os.path.join(tmp, 'chunks.txt')
        inc_path = os.path.join(tmp, 'blocks.inc')
        with open(words_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        env = dict(os.environ, ATV_VU_NATIVE_BUILD=words_path, ATV_VU_NATIVE_GEN=chunks_path)
        if subprocess.run([args.runner], env=env).returncode != 0 or not os.path.isfile(chunks_path):
            raise SystemExit('the runner did not generate the blocks (is it built from this source?)')
        if subprocess.run([sys.executable, '-I', os.path.join(HERE, 'merge_blocks.py'), args.flat, inc_path,
                           chunks_path]).returncode != 0:
            raise SystemExit('merge_blocks.py failed')
        if os.path.isfile(args.out) and filecmp.cmp(inc_path, args.out, shallow=False):
            print(f'{args.out} is up to date')
            return
        shutil.copyfile(inc_path, args.out)
    # The build that ran without the file never listed it as a dependency.
    if os.path.dirname(os.path.abspath(args.out)) == VU_DIR:
        os.utime(os.path.join(VU_DIR, 'ps2_vu1_core.cpp'))
    print(f'wrote {args.out}: build the runner again')


if __name__ == '__main__':
    main()
