"""Build ATV Offroad Fury 2 for PC from your own disc image.

    setup.bat "<your ATV Offroad Fury 2 (USA) (v2.00).iso>"

or, with the same options:

    python -I tools/setup/build_atv2.py <iso> [--out DIR] [--build-dir DIR] [--jobs N]

Steps: check the tools (Visual Studio 2026 with C++ and CMake, git), extract
SCUS_972.11 from the ISO and check it is the USA v2.00 program, build the
recompiler, recompile SCUS_972.11 into ps2xRuntime/src/runner (two passes,
merged: see recompile()), build ps2EntryRunner, build the VU1 native blocks
from the ELF (tools/vu1gen/make_vu1_blocks.py), build again, and lay out a
game folder with launchers (default: ATV2-Recomp in the repository folder).
Everything generated from the game lands in gitignored paths.

Running it again picks up where it left off: unchanged generated files are not
rewritten, so the builds are incremental.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_TOML = ROOT / 'config' / 'SCUS_972.11.toml'
CTOR_MAP = ROOT / 'config' / 'SCUS_972.11_ctor_map.csv'
VU1_MANIFEST = ROOT / 'tools' / 'vu1gen' / 'vu1_blocks_manifest.txt'
MAKE_VU1_BLOCKS = ROOT / 'tools' / 'vu1gen' / 'make_vu1_blocks.py'
RUNNER_DIR = ROOT / 'ps2xRuntime' / 'src' / 'runner'
INCLUDE_DIR = ROOT / 'ps2xRuntime' / 'include'
FUNCTIONS_H, STUBS_H = 'ps2_recompiled_functions.h', 'ps2_recompiled_stubs.h'
REGISTER_CPP = 'register_functions.cpp'
ELF_NAME = 'SCUS_972.11'
GAME_TITLE = 'ATV Offroad Fury 2 (USA) (v2.00)'
MIN_FREE_GB = 6  # a full build plus the game folder takes ~2.5 GB

LAUNCHERS = {
    'Play ATV2 (GPU 2x).bat': ('gpu', '2', '', 'GPU renderer, 2x resolution'),
    'Play ATV2 (GPU 2x widescreen).bat': ('gpu', '2', '1', 'GPU renderer, 2x resolution, widescreen'),
    'Play ATV2 (CPU renderer).bat': ('', '', '', 'CPU renderer (original resolution)'),
}

_step_no = 0
_log_dir = None


def fail(msg):
    print(f'\n[ERROR] {msg}', flush=True)
    sys.exit(1)


def step(title):
    global _step_no
    _step_no += 1
    print(f'\n=== {_step_no}. {title} ({datetime.datetime.now():%H:%M:%S})', flush=True)


def elapsed(t0):
    s = int(time.time() - t0)
    return f'{s // 60} min {s % 60} s' if s >= 60 else f'{s} s'


def read_text(path):
    """A generated file's text with its line endings as they are."""
    with open(path, encoding='utf-8', newline='') as f:
        return f.read()


def run(cmd, log_name, cwd=None, env=None, ok_codes=(0,), show=None):
    """Runs cmd with all of its output in <build>/setup_logs/<log_name>.log. On the console:
    everything (show=None), a build's progress line every 15 s plus its error lines
    (show='build'), or the lines matching the regex show."""
    log_path = _log_dir / f'{log_name}.log'
    tail = collections.deque(maxlen=40)
    last_progress, pending = 0.0, None
    with open(log_path, 'w', encoding='utf-8', errors='replace') as log:
        log.write('> ' + subprocess.list2cmdline([str(c) for c in cmd]) + '\n')
        try:
            p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        except OSError as e:
            fail(f'could not start {cmd[0]}: {e}')
        for line in p.stdout:
            log.write(line)
            tail.append(line)
            if show == 'build':
                if re.match(r'\[\d+/\d+\] ', line):
                    pending = line
                    if time.time() - last_progress >= 15:
                        sys.stdout.write(line[:150].rstrip() + '\n')
                        last_progress, pending = time.time(), None
                elif re.search(r'\berror\b|FAILED|fatal', line, re.I):
                    sys.stdout.write(line)
            elif show is None or re.search(show, line):
                sys.stdout.write(line)
            sys.stdout.flush()
        rc = p.wait()
    if pending:
        sys.stdout.write(pending[:150].rstrip() + '\n')
    if rc not in ok_codes:
        print('--- last lines of the log:')
        sys.stdout.write(''.join(tail))
        fail(f'{Path(str(cmd[0])).name} failed (exit code {rc}). Full log: {log_path}')
    return rc


# --- tools ---------------------------------------------------------------------------------------

def find_visual_studio():
    vswhere = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / \
        'Microsoft Visual Studio' / 'Installer' / 'vswhere.exe'
    need = ('Visual Studio 2026 with the "Desktop development with C++" workload (it includes the '
            'MSVC x64/x86 build tools and "C++ CMake tools for Windows"). Get it from '
            'https://visualstudio.microsoft.com/ (Community is free), then run this again.')
    if not vswhere.is_file():
        fail(f'Visual Studio was not found. Install {need}')
    query = [str(vswhere), '-products', '*', '-prerelease', '-version', '[18.0,19.0)', '-format', 'json',
             '-utf8']
    try:
        found = json.loads(subprocess.run(query, capture_output=True, text=True, encoding='utf-8').stdout or '[]')
        with_cpp = json.loads(subprocess.run(
            query + ['-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'],
            capture_output=True, text=True, encoding='utf-8').stdout or '[]')
    except (OSError, ValueError) as e:
        fail(f'vswhere failed ({e}). Install {need}')
    if not found:
        fail(f'Visual Studio 2026 was not found (older versions are not supported). Install {need}')
    if not with_cpp:
        fail(f'Visual Studio 2026 at {found[0]["installationPath"]} has no C++ compiler. In the Visual Studio '
             f'Installer choose Modify and add the "Desktop development with C++" workload.')
    vs = Path(sorted(with_cpp, key=lambda i: i.get('installationVersion', ''))[-1]['installationPath'])
    vcvars = vs / 'VC' / 'Auxiliary' / 'Build' / 'vcvars64.bat'
    cmake_dir = vs / 'Common7' / 'IDE' / 'CommonExtensions' / 'Microsoft' / 'CMake'
    cmake, ninja = cmake_dir / 'CMake' / 'bin' / 'cmake.exe', cmake_dir / 'Ninja' / 'ninja.exe'
    if not vcvars.is_file():
        fail(f'{vcvars} is missing: repair Visual Studio 2026 in the Visual Studio Installer.')
    if not cmake.is_file() or not ninja.is_file():
        fail(f'Visual Studio 2026 at {vs} has no CMake. In the Visual Studio Installer choose Modify, '
             f'"Desktop development with C++", and tick "C++ CMake tools for Windows".')
    return vs, vcvars, cmake, ninja, vswhere.parent


def vs_environment(vcvars, extra_path):
    """The environment vcvars64.bat sets up, read back as JSON (safe for any characters)."""
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join(str(p) for p in extra_path) + os.pathsep + env.get('PATH', '')
    env['VSCMD_SKIP_SENDTELEMETRY'] = '1'
    dump = f'"{sys.executable}" -I -c "import json,os;print(json.dumps(dict(os.environ)))"'
    r = subprocess.run(f'cmd /d /s /c ""{vcvars}" >nul 2>&1 && {dump}"', capture_output=True, text=True,
                       env=env)
    try:
        vsenv = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        fail(f'could not set up the Visual Studio environment with {vcvars}:\n{r.stdout}{r.stderr}')
    if not shutil.which('cl.exe', path=vsenv.get('PATH', '')):
        fail(f'{vcvars} did not put the C++ compiler (cl.exe) on PATH: repair Visual Studio 2026.')
    return vsenv


def check_tools(build_dir):
    if sys.platform != 'win32':
        fail('this script builds the Windows version; run it on Windows 10/11 (64-bit).')
    if sys.version_info < (3, 8):
        fail(f'Python 3.8 or newer is needed (this is {sys.version.split()[0]}).')
    vs, vcvars, cmake, ninja, installer = find_visual_studio()
    print(f'Visual Studio: {vs}')
    git_dirs = []
    if not shutil.which('git'):
        vs_git = vs / 'Common7' / 'IDE' / 'CommonExtensions' / 'Microsoft' / 'TeamFoundation' / \
            'Team Explorer' / 'Git' / 'cmd'
        if not (vs_git / 'git.exe').is_file():
            fail('git was not found (CMake downloads raylib and Dear ImGui with it). Install it from '
                 'https://git-scm.com/download/win, or add "Git for Windows" in the Visual Studio Installer.')
        git_dirs.append(vs_git)
    env = vs_environment(vcvars, [cmake.parent, ninja.parent, installer] + git_dirs)
    # The release flags ask for /GL (whole-program optimization), which takes hours and far too much
    # memory on ~6,000 recompiled files; build with it off, as the project's own builds do.
    env['_CL_'] = '/GL-'
    build_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(build_dir).free / 2 ** 30
    if free_gb < MIN_FREE_GB:
        fail(f'only {free_gb:.0f} GB free on the drive of {build_dir}; the build needs about {MIN_FREE_GB} GB.')
    print(f'cmake: {cmake}\nninja: {ninja}\nfree disk space: {free_gb:.0f} GB')
    return env, cmake, ninja, vs


def default_jobs():
    cpus = os.cpu_count() or 4
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        ram_gb = status.ullTotalPhys / 2 ** 30
    except (AttributeError, OSError):
        ram_gb = 16
    # One compiler process on a batch of recompiled files can take ~3 GB.
    return max(1, min(cpus, int(ram_gb // 3)))


# --- ISO -------------------------------------------------------------------------------------------

class Iso:
    """Just enough ISO 9660 to read files from the root directory of a PS2 disc image
    (2048-byte sectors, or raw 2352-byte sectors)."""

    def __init__(self, path):
        self.f = open(path, 'rb')
        for sector_size, data_offset in ((2048, 0), (2352, 24), (2352, 16)):
            self.f.seek(16 * sector_size + data_offset)
            pvd = self.f.read(2048)
            if pvd[0:1] == b'\x01' and pvd[1:6] == b'CD001':
                self.sector_size, self.data_offset = sector_size, data_offset
                break
        else:
            fail(f'{path} is not a disc image this script can read (an ISO 9660 .iso is needed; '
                 f'convert .chd/.cso/.nrg/.mdf images to .iso first).')
        self.root = self._record(pvd[156:190])

    @staticmethod
    def _record(r):
        return {'lba': struct.unpack_from('<I', r, 2)[0], 'size': struct.unpack_from('<I', r, 10)[0],
                'dir': bool(r[25] & 2), 'name': r[33:33 + r[32]].decode('ascii', 'replace')}

    def read(self, lba, size):
        out = bytearray()
        for i in range((size + 2047) // 2048):
            self.f.seek((lba + i) * self.sector_size + self.data_offset)
            out += self.f.read(2048)
        return bytes(out[:size])

    def root_files(self):
        data, files, pos = self.read(self.root['lba'], self.root['size']), {}, 0
        while pos < len(data):
            n = data[pos]
            if n == 0:
                pos = (pos // 2048 + 1) * 2048
                continue
            rec = self._record(data[pos:pos + n])
            if not rec['dir']:
                files[rec['name'].split(';')[0].upper()] = rec
            pos += n
        return files


def manifest_sha1():
    for line in VU1_MANIFEST.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if fields[:2] == ['#', 'sha1']:
            return fields[2]
    fail(f'{VU1_MANIFEST} has no "# sha1" line')


def extract_elf(iso_path, elf_path):
    iso = Iso(iso_path)
    files = iso.root_files()
    cnf = files.get('SYSTEM.CNF')
    if not cnf:
        fail(f'{iso_path} has no SYSTEM.CNF: it is not a PlayStation 2 game disc.')
    boot = re.search(r'BOOT2\s*=\s*cdrom0:\\?([^;\s]+)', iso.read(cnf['lba'], cnf['size']).decode('ascii', 'replace'))
    boot_name = boot.group(1).upper() if boot else '?'
    if boot_name != ELF_NAME or ELF_NAME not in files:
        fail(f'{iso_path} boots {boot_name}, not {ELF_NAME}: this recompilation needs {GAME_TITLE}.')
    data = iso.read(files[ELF_NAME]['lba'], files[ELF_NAME]['size'])
    sha1, want = hashlib.sha1(data).hexdigest(), manifest_sha1()
    if sha1 != want:
        fail(f'{ELF_NAME} on this disc (sha1 {sha1}) is not the program this recompilation is made from '
             f'(sha1 {want}, {GAME_TITLE}). Other versions of the game are not supported.')
    elf_path.parent.mkdir(parents=True, exist_ok=True)
    if not elf_path.is_file() or elf_path.read_bytes() != data:
        elf_path.write_bytes(data)
    print(f'{ELF_NAME}: {len(data)} bytes, sha1 {sha1} ({GAME_TITLE}) -> {elf_path}')


# --- recompile -------------------------------------------------------------------------------------

def toml_with_paths(elf, ghidra_map, out_dir):
    text = CONFIG_TOML.read_text(encoding='utf-8-sig')
    for key, value in (('input', elf), ('ghidra_output', ghidra_map), ('output', out_dir)):
        line = f'{key} = "{value.as_posix()}"' if value else f'# {key} = ""'
        text, n = re.subn(rf'^#?\s*{key}\s*=.*$', lambda m: line, text, count=1, flags=re.M)
        if n != 1:
            fail(f'{CONFIG_TOML}: no "{key} = ..." line')
    return text


def run_recompiler(recomp_exe, name, elf, ghidra_map, work, ok_codes=(0,)):
    out_dir = work / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    toml_path = work / f'{name}.toml'
    toml_path.write_text(toml_with_paths(elf, ghidra_map, out_dir), encoding='utf-8')
    run([recomp_exe, toml_path], f'recomp_{name}', cwd=work, ok_codes=ok_codes,
        show=r'^\[recompiler\]|error|Recompilation completed|Functions discovered|Generated functions')
    return out_dir


FUNC_FILE = re.compile(r'^\w+_0x([0-9a-f]+)\.cpp$')
FUNC_DEF = re.compile(r'^void (\w+)\(uint8_t\* rdram, R5900Context\* ctx, PS2Runtime \*runtime\) \{\r?\n', re.M)
REG_LINE = re.compile(r'^        g_ps2RecompiledFunctionTable\[(\d+)\] = (\w+); // 0x([0-9a-f]+)\r?$', re.M)
ADDRESS_COMMENT = re.compile(r'^// Address: 0x([0-9a-f]+) - 0x([0-9a-f]+)', re.M)
DIRECT_BRANCH = re.compile(r'dispatchGuestBranch\(rdram, ctx, 0x([0-9A-Fa-f]+)u, 0x[0-9A-Fa-f]+u, '
                           r'0x[0-9A-Fa-f]+u, PS2Runtime::GuestBranchKind::Direct(?:Call|Jump)')
ENTRY_SWITCH = re.compile(r'(?:#ifdef PS2_FUNCTION_LOG_TRACKER\r?\n.*\r?\n#endif\r?\n)?\r?\n'
                          r'    switch \(ctx->pc\) \{\r?\n((?:        case 0x[0-9a-f]+u: goto label_[0-9a-f]+;\r?\n)*)'
                          r'        default: break;')


def func_name(text, path):
    m = FUNC_DEF.search(text)
    if not m:
        fail(f'{path}: no function definition')
    return m


def entry_cases(text, path):
    """The resume entry addresses in a generated function's entry switch."""
    m = ENTRY_SWITCH.match(text, func_name(text, path).end())
    return [int(a, 16) for a in re.findall(r'case 0x([0-9a-f]+)u:', m.group(1))] if m else []


def add_resume_entry(text, path, owner_start, target):
    """Adds a resume entry at target to a generated function: an entry-switch case and a label in
    front of the instruction, as the recompiler emits for the entries it finds itself. Returns
    None when target is not a plain instruction of the function (e.g. a delay slot)."""
    nl = '\r\n' if '\r\n' in text else '\n'
    insn = re.compile(rf'^    // 0x{target:x}: [^\r\n]*', re.M)
    hits = insn.findall(text)
    if len(hits) != 1 or 'Delay Slot' in hits[0]:
        return None
    if not re.search(rf'^label_{target:x}:', text, re.M):
        text = insn.sub(lambda m: f'label_{target:x}:{nl}' + m.group(0), text, count=1)
    cases = entry_cases(text, path)
    if target in cases:
        return text
    case = f'        case 0x{target:x}u: goto label_{target:x};{nl}'
    body = func_name(text, path).end()
    if cases:
        later = [a for a in cases if a > target]
        i = text.index(f'        case 0x{later[0]:x}u:' if later else '        default: break;', body)
        return text[:i] + case + text[i:]
    i = text.index(f'    ctx->pc = 0x{owner_start:x}u;{nl}', body)
    return text[:i] + f'    switch (ctx->pc) {{{nl}{case}        default: break;{nl}    }}{nl}{nl}' + text[i:]


def merge_outputs(plain, ctor, merged):
    """plain: the full analysis (every function reached from the entry point and the JAL targets);
    ctor: the ctor-map pass, whose static initializers (called through the .ctors list only) the
    analysis never finds. The result is plain plus those functions, their table entries and
    declarations, plus a resume entry for each mid-function address they branch to."""
    reg_text = read_text(plain / REGISTER_CPP)
    registered = {int(a, 16): n for _, n, a in REG_LINE.findall(reg_text)}
    if not registered:
        fail(f'{plain / REGISTER_CPP}: no function table entries')
    plain_funcs, other = {}, []
    for f in plain.iterdir():
        m = FUNC_FILE.match(f.name)
        if m:
            plain_funcs[int(m[1], 16)] = f
        elif f.name not in (REGISTER_CPP, FUNCTIONS_H, STUBS_H):
            other.append(f.name)
    if other:
        fail(f'unexpected files in the recompiler output {plain}: {", ".join(other[:5])}')
    taken = {}
    for f in sorted(ctor.iterdir()):
        m = FUNC_FILE.match(f.name)
        if m and int(m[1], 16) not in registered:
            taken[int(m[1], 16)] = f
    if not taken:
        fail(f'the ctor-map pass generated no new functions in {ctor}')

    added, patched, taken_names = {}, {}, set()
    for start, f in taken.items():
        text = read_text(f)
        name = func_name(text, f)[1]
        taken_names.add(name)
        for addr in [start] + entry_cases(text, f):
            if addr not in registered:
                added[addr] = name
    known = {**registered, **added}
    owners = []
    for f in plain_funcs.values():
        with open(f, encoding='utf-8') as fh:
            m = ADDRESS_COMMENT.search(fh.read(600))
        if m:
            owners.append((int(m[1], 16), int(m[2], 16), f))
    unresolved = []
    for f in taken.values():
        for target in sorted({int(t, 16) for t in DIRECT_BRANCH.findall(read_text(f))}):
            if target in known:
                continue
            owner = next((o for o in owners if o[0] < target < o[1]), None)
            text = None
            if owner:
                text = add_resume_entry(patched.get(owner[2]) or read_text(owner[2]), owner[2], owner[0], target)
            if text is None:
                unresolved.append(target)
                continue
            patched[owner[2]] = text
            known[target] = added[target] = func_name(text, owner[2])[1]

    if merged.exists():
        shutil.rmtree(merged)
    merged.mkdir(parents=True)
    for f in list(plain_funcs.values()) + list(taken.values()):
        if f in patched:
            (merged / f.name).write_bytes(patched[f].encode('utf-8'))
        else:
            shutil.copyfile(f, merged / f.name)

    # The function table: one slot per instruction word from the lowest entry to the highest.
    entries = sorted(known.items())
    base, end = entries[0][0], entries[-1][0] + 4
    slots = (end - base) // 4
    nl = '\r\n' if '\r\n' in reg_text else '\n'
    lines = ''.join(f'        g_ps2RecompiledFunctionTable[{(a - base) // 4}] = {n}; // 0x{a:x}{nl}'
                    for a, n in entries)
    first, last = REG_LINE.search(reg_text), list(REG_LINE.finditer(reg_text))[-1]
    body = reg_text[:first.start()] + lines + reg_text[reg_text.index('\n', last.start()) + 1:]
    for key, value in (('Base', f'0x{base:x}u'), ('End', f'0x{end:x}u'), ('SlotCount', f'{slots}u')):
        body, n = re.subn(rf'(g_ps2RecompiledFunctionTable{key} = )[^;]+;', lambda m: m[1] + value + ';', body)
        if n != 1:
            fail(f'{REGISTER_CPP}: no g_ps2RecompiledFunctionTable{key}')
    body, n = re.subn(r'(g_ps2RecompiledFunctionTable\[)\d+u(\] = \{\};)', lambda m: f'{m[1]}{slots}u{m[2]}', body)
    if n != 1:
        fail(f'{REGISTER_CPP}: no function table definition')
    (merged / REGISTER_CPP).write_bytes(body.encode('utf-8'))

    header = read_text(plain / FUNCTIONS_H)
    decl = re.compile(r'^void (\w+)\(.*\r?\n', re.M)
    decls = ''.join(m[0] for m in decl.finditer(read_text(ctor / FUNCTIONS_H)) if m[1] in taken_names)
    last_decl = list(decl.finditer(header))[-1]
    header = header[:last_decl.end()] + decls + header[last_decl.end():]
    (merged / FUNCTIONS_H).write_bytes(header.encode('utf-8'))
    shutil.copyfile(plain / STUBS_H, merged / STUBS_H)

    print(f'merged: {len(plain_funcs)} functions + {len(taken)} static initializers; {len(entries)} '
          f'function table entries ({len(added)} added); resume entries added to {len(patched)} function(s)')
    for t in unresolved:
        print(f'  note: 0x{t:x} (a static initializer branches there) has no compiled entry; '
              f'it runs on the interpreter fallback')


def sync_generated(merged, backup_root):
    """Copies the merged output into ps2xRuntime (src/runner and the two headers), writing only the
    files that changed so that the build stays incremental. The files it replaces or removes are
    kept in backup_root."""
    targets = {RUNNER_DIR / f.name: f for f in merged.iterdir() if f.name not in (FUNCTIONS_H, STUBS_H)}
    targets.update({INCLUDE_DIR / h: merged / h for h in (FUNCTIONS_H, STUBS_H)})
    stale = [f for f in RUNNER_DIR.iterdir() if f.is_file() and f not in targets] if RUNNER_DIR.is_dir() else []
    backup = backup_root / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    written = 0
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    for dst, src in targets.items():
        data = src.read_bytes()
        if dst.is_file():
            if dst.read_bytes() == data:
                continue
            (backup / dst.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, backup / dst.parent.name / dst.name)
        dst.write_bytes(data)
        written += 1
    for f in stale:
        (backup / RUNNER_DIR.name).mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(backup / RUNNER_DIR.name / f.name))
    print(f'{RUNNER_DIR}: {written} file(s) written, {len(targets) - written} unchanged, {len(stale)} removed')
    if backup.exists():
        print(f'  the previous versions of the replaced files are in {backup}')


def recompile(recomp_exe, elf, work):
    """Two passes over the same config: the plain analysis, and the ctor map. With a function map
    the recompiler uses only the map's functions, so that pass generates just the static
    initializers, and then stops before writing a function table because the map has no entry
    point (expected). merge_outputs() puts the two together."""
    plain = run_recompiler(recomp_exe, 'plain', elf, None, work)
    ctor = run_recompiler(recomp_exe, 'ctor', elf, CTOR_MAP, work, ok_codes=(0, 1))
    wanted = sum(1 for line in CTOR_MAP.read_text(encoding='utf-8-sig').splitlines()[1:] if line.strip())
    got = sum(1 for f in ctor.glob('*.cpp') if f.name != REGISTER_CPP)
    if got != wanted or not (ctor / FUNCTIONS_H).is_file():
        fail(f'the ctor-map pass generated {got} of {wanted} functions; see {_log_dir / "recomp_ctor.log"}')
    print('(the ctor-map pass stopping at "No entry function name available for registration" is expected)')
    merge_outputs(plain, ctor, work / 'merged')
    sync_generated(work / 'merged', work / 'replaced')


# --- build and package --------------------------------------------------------------------------------

def check_not_running(exe):
    if exe.is_file():
        try:
            with open(exe, 'r+b'):
                pass
        except PermissionError:
            fail(f'{exe} is in use: close the game (and anything else running it) first.')


def configure(cmake, ninja, build_dir, env):
    args = [cmake, '-S', ROOT, '-B', build_dir]
    if not (build_dir / 'CMakeCache.txt').is_file():
        args += ['-G', 'Ninja', f'-DCMAKE_MAKE_PROGRAM={ninja}', '-DCMAKE_BUILD_TYPE=Release',
                 '-DCMAKE_C_COMPILER=cl.exe', '-DCMAKE_CXX_COMPILER=cl.exe',
                 '-DPS2X_BUILD_ANALYZER=OFF', '-DPS2X_BUILD_TEST=OFF', '-DPS2X_BUILD_STUDIO=OFF']
    run(args, 'configure', env=env, show=r'^-- Configuring done|^-- Build files have been written to: (?!.*-subbuild)|'
                                         r'CMake Error|\berror\b')


def build(cmake, build_dir, target, jobs, env, log_name):
    run([cmake, '--build', build_dir, '--target', target, '-j', str(jobs)], log_name, env=env, show='build')


def vc_runtime_dlls(vs, exe, env):
    """The Visual C++ runtime DLLs the game imports, from Visual Studio's redistributable folder, so
    that the game folder also runs on a PC without the runtime installed."""
    crt_dirs = sorted((vs / 'VC' / 'Redist' / 'MSVC').glob('*/x64/Microsoft.VC*.CRT'))
    dumpbin = shutil.which('dumpbin.exe', path=env.get('PATH', ''))
    if not crt_dirs or not dumpbin:
        return []
    deps = subprocess.run([dumpbin, '/nologo', '/dependents', str(exe)], capture_output=True, text=True,
                          env=env, errors='replace').stdout.lower()
    return [f for f in crt_dirs[-1].glob('*.dll') if f.name.lower() in deps]


def launch_bat(iso):
    iso_line = str(iso).replace('%', '%%')
    return f'''@echo off
setlocal
cd /d "%~dp0"
rem Game files: {ELF_NAME} in the game folder here; the ISO from the game folder here, else the
rem one this folder was built from.
set "ELF=%~dp0game\\{ELF_NAME}"
set "ISO="
for %%I in ("%~dp0game\\*.iso") do if not defined ISO set "ISO=%%~fI"
if not defined ISO set "ISO={iso_line}"
if not exist "%ELF%" goto noelf
if not exist "%ISO%" goto noiso
echo Starting ATV Offroad Fury 2: %~1
echo Game log: "%~dp0logs\\last_run.log"
echo (F1 shows or hides the debug panel. Close the game window to quit.)
if not exist "%~dp0logs" mkdir "%~dp0logs"
"%~dp0ps2EntryRunner.exe" "%ELF%" --cd "%ISO%" > "%~dp0logs\\last_run.log" 2>&1
echo Game exited with code %ERRORLEVEL%.
timeout /t 3 >nul
exit /b 0
:noelf
echo [ERROR] {ELF_NAME} not found. Run setup.bat again, or copy it from the disc into "%~dp0game".
pause
exit /b 1
:noiso
echo [ERROR] The game ISO was not found at "%ISO%". Put the ATV Offroad Fury 2 ISO into "%~dp0game".
pause
exit /b 1
'''


def package(build_dir, out, elf, iso, vs, env):
    exe = build_dir / 'ps2xRuntime' / 'ps2EntryRunner.exe'
    check_not_running(out / exe.name)
    for d in ('game', 'game/mc0', 'logs'):
        (out / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe, out / exe.name)
    dlls = sorted(exe.parent.glob('*.dll')) + vc_runtime_dlls(vs, exe, env)
    for f in dlls:
        shutil.copy2(f, out / f.name)
    shutil.copy2(elf, out / 'game' / ELF_NAME)
    (out / '_launch.bat').write_text(launch_bat(iso), encoding='utf-8', newline='\r\n')
    for name, (backend, scale, wide, desc) in LAUNCHERS.items():
        (out / name).write_text(f'@echo off\nset "ATV_GS_BACKEND={backend}"\nset "ATV_GS_SCALE={scale}"\n'
                                f'set "ATV2_WIDESCREEN={wide}"\ncall "%~dp0_launch.bat" "{desc}"\n',
                                encoding='utf-8', newline='\r\n')
    git = shutil.which('git', path=env.get('PATH', ''))
    commit = subprocess.run([git, '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                            text=True).stdout.strip() if git else ''
    (out / 'README.txt').write_text(f'''ATV Offroad Fury 2 - PC recompilation (built {datetime.date.today()}{f", commit {commit}" if commit else ""})

Double-click one of the launchers:
  Play ATV2 (GPU 2x).bat             GPU renderer at twice the PS2's resolution
  Play ATV2 (GPU 2x widescreen).bat  the same, with the game's own 16:9 mode
  Play ATV2 (CPU renderer).bat       the original software renderer (slower, 1x)

Game files: game\\{ELF_NAME} (from your disc) and your disc image at
  {iso}
If you move the ISO, put it (or a copy) into the "game" folder here.
Saves are in game\\mc0.

Keys: F1 shows or hides the debug panel. Widescreen can also be toggled in-game
with the cheat code WIDESCRN. The game's log is written to logs\\last_run.log.

Sharper GPU picture: edit the launcher and set ATV_GS_SCALE=3 or 4 (slower).

This folder contains the game program recompiled from your disc; keep it to
yourself (do not share it publicly).
''', encoding='utf-8', newline='\r\n')
    print(f'{out}: {exe.name}, {len(dlls)} DLLs, game\\{ELF_NAME}, launchers')


def main():
    ap = argparse.ArgumentParser(description='Build ATV Offroad Fury 2 for PC from your own disc image.')
    ap.add_argument('iso', help=f'your {GAME_TITLE} disc image (.iso)')
    ap.add_argument('--out', default=str(ROOT / 'ATV2-Recomp'), help='game folder to create (default: %(default)s)')
    ap.add_argument('--build-dir', default=str(ROOT / 'build64'), help='CMake build folder (default: %(default)s)')
    ap.add_argument('--jobs', type=int, default=default_jobs(),
                    help='parallel compile jobs (default for this PC: %(default)s; lower it if the PC runs out '
                         'of memory)')
    args = ap.parse_args()
    iso = Path(args.iso.strip('"')).resolve()
    build_dir, out = Path(args.build_dir).resolve(), Path(args.out).resolve()
    if not iso.is_file():
        fail(f'{iso} does not exist. Pass the path of your {GAME_TITLE} .iso.')

    global _log_dir
    _log_dir = build_dir / 'setup_logs'
    t0 = time.time()
    step('Checking tools')
    env, cmake, ninja, vs = check_tools(build_dir)
    _log_dir.mkdir(parents=True, exist_ok=True)
    print(f'compile jobs: {args.jobs}; logs: {_log_dir}')

    step(f'Extracting {ELF_NAME} from {iso.name}')
    elf = build_dir / 'game' / ELF_NAME
    extract_elf(iso, elf)

    exe = build_dir / 'ps2xRuntime' / 'ps2EntryRunner.exe'
    check_not_running(exe)
    step('Configuring the build (the first time downloads raylib, Dear ImGui and FFmpeg)')
    configure(cmake, ninja, build_dir, env)
    step('Building the recompiler')
    build(cmake, build_dir, 'ps2_recomp', args.jobs, env, 'build_recompiler')

    step(f'Recompiling {ELF_NAME} into {RUNNER_DIR.relative_to(ROOT)}')
    recompile(build_dir / 'ps2xRecomp' / 'ps2_recomp.exe', elf, build_dir / 'recomp')

    step('Building ps2EntryRunner (the first build takes a long time)')
    t1 = time.time()
    configure(cmake, ninja, build_dir, env)
    build(cmake, build_dir, 'ps2EntryRunner', args.jobs, env, 'build_runner')
    print(f'built in {elapsed(t1)}')

    step('Building the VU1 native blocks from your ELF')
    run([sys.executable, '-I', MAKE_VU1_BLOCKS, elf, exe], 'vu1_blocks', cwd=build_dir, env=env)
    step('Building ps2EntryRunner with the VU1 blocks')
    build(cmake, build_dir, 'ps2EntryRunner', args.jobs, env, 'build_runner_vu1')

    step('Laying out the game folder')
    package(build_dir, out, elf, iso, vs, env)

    print(f'\nDone in {elapsed(t0)}. To play, double-click:\n  {out / "Play ATV2 (GPU 2x).bat"}\n'
          f'(or the widescreen / CPU renderer launchers next to it). See {out / "README.txt"}.')


if __name__ == '__main__':
    main()
