# ATV-Off-Road-Fury-Quadzilla-Engine-recomp
ATV Off Road Fury Recomp

We are using https://github.com/hkmodd/ps2-recomp-Agent-SKILL  to do the heavy lifting . 

WIP , Findings can be found below 

## Build it from your own disc

The repository holds no game code or data. You need your own copy of **ATV Offroad Fury 2 (USA) (v2.00)** as an `.iso`: one script recompiles the game program from it on your PC and lays out a folder you play from.

### Prerequisites (Windows 10/11, 64-bit)

- **Visual Studio 2026** (Community is free) with the **Desktop development with C++** workload. It includes the MSVC x64 compiler and "C++ CMake tools for Windows" (CMake and Ninja), which the script uses.
- **Python 3.8 or newer** from [python.org](https://www.python.org/downloads/) (tick "Add python.exe to PATH").
- **git** ([git-scm.com](https://git-scm.com/download/win), or "Git for Windows" in the Visual Studio Installer).
- An internet connection for the first build (CMake downloads raylib, Dear ImGui and an FFmpeg build) and about 5 GB of free disk space.

### One command

```
setup.bat "D:\Games\ATV Offroad Fury 2 (USA) (v2.00).iso"
```

(or drag the ISO onto `setup.bat`). It checks the tools, extracts `SCUS_972.11` from the ISO and checks that it is the v2.00 program, builds the recompiler, recompiles the game into `ps2xRuntime/src/runner`, builds `ps2EntryRunner`, builds the VU1 native blocks from your ELF and builds again. The first build compiles ~6,000 generated C++ files; the whole run took 16 minutes on a Core Ultra 7 270K Plus with 6 compile jobs, so expect longer on fewer cores. Then play from the `ATV2-Recomp` folder:

| Launcher | |
| --- | --- |
| `Play ATV2 (GPU 2x).bat` | GPU renderer at twice the PS2's resolution |
| `Play ATV2 (GPU 2x widescreen).bat` | the same, with the game's own 16:9 mode |
| `Play ATV2 (CPU renderer).bat` | the original software renderer (slower, 1x) |

Saves go to `ATV2-Recomp\game\mc0`, the game's log to `ATV2-Recomp\logs\last_run.log`. F1 shows the debug panel. The folder contains the program recompiled from your disc, so keep it to yourself.

Options (`setup.bat --help`): `--out <folder>` for the game folder, `--jobs N` to compile with fewer processes if the PC runs out of memory, `--build-dir <folder>` (default `build64`). Running it again, for example after a `git pull`, rebuilds only what changed. Build logs are in `build64\setup_logs`.

### What the script does

- `config/SCUS_972.11.toml` is the recompiler config (stubs, MMIO and jump-table addresses, options) and `config/SCUS_972.11_ctor_map.csv` lists the game's static initializers (0x44C100..), which nothing calls directly, so the analysis does not find them. The recompiler runs twice, without and with the map, and `tools/setup/build_atv2.py` merges the two outputs into `ps2xRuntime/src/runner` and the generated headers in `ps2xRuntime/include` (all gitignored).
- The VU1 native blocks are translations of the game's VU1 microcode, so they are built from your ELF too: `python -I tools/vu1gen/make_vu1_blocks.py <SCUS_972.11> build64/ps2xRuntime/ps2EntryRunner.exe`, then build `ps2EntryRunner` again (races run noticeably slower without them).
- The builds are `cmake --build build64 --target ps2EntryRunner` in a Visual Studio x64 environment with `_CL_=/GL-` (whole-program optimization off).



https://www.youtube.com/watch?v=Tpdy3xvcvYQ&lc=Ugy6IkaXspRC5gbXddV4AaABAg
