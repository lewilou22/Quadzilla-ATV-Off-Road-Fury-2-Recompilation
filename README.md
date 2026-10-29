# ATV-Off-Road-Fury-Quadzilla-Engine-recomp
ATV Off Road Fury Recomp

We are using https://github.com/hkmodd/ps2-recomp-Agent-SKILL  to do the heavy lifting . 

WIP , Findings can be found below 

## Game files are not included

You need your own copy of the game (`SCUS_972.11` from the disc). The recompiled game code (`ps2xRuntime/src/runner`) and the compiled VU1 blocks are generated from it on your machine:

1. Build `ps2EntryRunner` once (it builds without the VU1 blocks).
2. Build the VU1 blocks from your ELF: `python -I tools/vu1gen/make_vu1_blocks.py <path to SCUS_972.11> build64/ps2xRuntime/ps2EntryRunner.exe`
3. Build `ps2EntryRunner` again. Races run noticeably slower without step 2.



