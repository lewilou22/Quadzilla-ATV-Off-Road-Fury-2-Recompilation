"""Convert a PCSX2 DDS (BC1-BC7) texture pack to PNG for ATV2_TEXTURE_PACK, plus the thumbnail
index the recomp uses to match textures whose bytes differ from PCSX2's (IPU-decoded ones).

usage: python convert_dds_pack.py <dds replacements dir> <out pack dir>
Writes <out>/replacements/<name>.png (base level; the recomp builds its own mips) and
<out>/fuzzy_index.bin: for each direct-colour texture (CT32/CT24/CT16) its name and a 16x16 RGB
box thumbnail. Already converted files are skipped.
"""
import os
import struct
import sys
from multiprocessing import Pool

from PIL import Image

THUMB = 16
PALETTE_PSMS = {0x13, 0x14, 0x1B, 0x24, 0x2C}


def convert(job):
    src, dst = job
    name = os.path.basename(dst)[:-4]
    psm = int(name.split("-")[-1], 16) & 0x3F
    try:
        im = Image.open(src).convert("RGBA")
        if not os.path.exists(dst):
            im.save(dst, compress_level=1)
    except Exception as e:  # keep going; report at the end
        return name, None, str(e)
    if psm in PALETTE_PSMS:
        return name, None, None
    thumb = im.convert("RGB").resize((THUMB, THUMB), Image.BOX).tobytes()
    return name, thumb, None


def main():
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    rep = os.path.join(out_dir, "replacements")
    os.makedirs(rep, exist_ok=True)
    jobs = [(os.path.join(src_dir, f), os.path.join(rep, f[:-4] + ".png"))
            for f in sorted(os.listdir(src_dir)) if f.lower().endswith(".dds")]
    entries, errors = [], []
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        for i, (name, thumb, err) in enumerate(pool.imap_unordered(convert, jobs, chunksize=8), 1):
            if err:
                errors.append((name, err))
            elif thumb:
                entries.append((name, thumb))
            if i % 500 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    with open(os.path.join(out_dir, "fuzzy_index.bin"), "wb") as fh:
        fh.write(b"ATV2FZ1\0" + struct.pack("<II", len(entries), THUMB))
        for name, thumb in sorted(entries):
            raw = name.encode("ascii")
            fh.write(struct.pack("<H", len(raw)) + raw + thumb)
    print(f"converted {len(jobs) - len(errors)} of {len(jobs)}; fuzzy index: {len(entries)} direct-colour textures")
    for name, err in errors[:20]:
        print("failed:", name, err)


if __name__ == "__main__":
    main()
