"""AI-upscale ATV2 texture dumps (ATV2_TEXTURE_DUMP) into a PCSX2-layout replacement pack.

usage: python make_hd_pack.py <dumps_dir> <pack_dir> --exe <realesrgan-ncnn-vulkan.exe> --models <its models dir>
       [--model realesrgan-x4plus] [--text-model realesrgan-x4plus-anime]
       (defaults: dumps/, atv2_ai_hd_pack/ and realesrgan/ next to this script). Real-ESRGAN is not
       in the repo: realesrgan-ncnn-vulkan-20220424-windows.zip from github.com/xinntao/Real-ESRGAN.
Textures already in the pack are skipped, so it can be re-run after dumping more of the game.

Per texture: colour under fully transparent texels is filled from its neighbours, then wrapped-
padded (repeating textures stay seamless), upscaled 4x by Real-ESRGAN, cropped and back-projected
(shrinking it gives the original again: colours and grain kept). Two-level alpha (cut-outs, glyphs)
goes through the text model; smooth alpha (blend weights) is only resized and back-projected.
Font sheets (mostly transparent, near-white glyphs) use the text model for colour too.
Writes <pack_dir>/replacements/<same file name>.
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

SCALE = 4


def is_text_sheet(rgba):
    a = rgba[..., 3]
    opaque = a > 0
    if opaque.mean() > 0.45 or opaque.sum() < 64:
        return False
    rgb = rgba[..., :3][opaque].astype(np.float32)
    return rgb.mean() > 170 and rgb.std(axis=0).mean() < 60


def bleed(rgb, alpha, iterations=32):
    """Fill fully transparent texels with their opaque neighbours' colour, so the upscaler
    doesn't sharpen the (usually black) colour under transparency into dark fringes."""
    known = alpha > 0
    if known.all() or not known.any():
        return rgb
    out = rgb.astype(np.float32)
    for _ in range(iterations):
        acc = np.zeros_like(out)
        cnt = np.zeros(alpha.shape, np.float32)
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(out * known[..., None], shift, (0, 1))
            cnt += np.roll(known, shift, (0, 1))
        new = ~known & (cnt > 0)
        if not new.any():
            break
        out[new] = acc[new] / cnt[new][:, None]
        known |= new
    return out.round().astype(np.uint8)


def back_project(hd, original, iterations=2):
    """Make the upscale shrink back to the original: add the upscaled difference between the
    original and the shrunk upscale. Keeps the original's colours and fine grain (which the
    model smooths away as noise) and keeps the model's sharper edges where they agree."""
    h, w = original.shape[:2]
    big = hd.shape[1], hd.shape[0]
    out = hd.astype(np.float32)
    ref = original.astype(np.float32)
    channels = 1 if out.ndim == 2 else out.shape[2]

    def resize(img, size, method):
        if channels == 1:
            return np.asarray(Image.fromarray(img, "F").resize(size, method))
        return np.stack([np.asarray(Image.fromarray(img[..., c], "F").resize(size, method)) for c in range(channels)], -1)

    for _ in range(iterations):
        residual = ref - resize(out, (w, h), Image.BOX)
        out += resize(residual, big, Image.BICUBIC)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def run(exe, models, model, src, dst):
    os.makedirs(dst, exist_ok=True)
    if not os.path.isdir(src) or not os.listdir(src):
        return
    subprocess.run([exe, "-i", src, "-o", dst, "-n", model, "-m", models, "-s", str(SCALE), "-f", "png"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="?", default=os.path.join(here, "dumps"))
    ap.add_argument("pack", nargs="?", default=os.path.join(here, "atv2_ai_hd_pack"))
    ap.add_argument("--exe", default=os.path.join(here, "realesrgan", "realesrgan-ncnn-vulkan.exe"))
    ap.add_argument("--models", default=os.path.join(here, "realesrgan", "models"))
    ap.add_argument("--model", default="realesrgan-x4plus")
    ap.add_argument("--text-model", default="realesrgan-x4plus-anime")
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--max-size", type=int, default=512)
    args = ap.parse_args()

    out = os.path.join(args.pack, "replacements")
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="hdpack_")
    jobs = {}
    for f in sorted(os.listdir(args.dumps)):
        if not f.lower().endswith(".png") or os.path.exists(os.path.join(out, f)):
            continue
        im = Image.open(os.path.join(args.dumps, f)).convert("RGBA")
        w, h = im.size
        if min(w, h) < args.min_size or max(w, h) > args.max_size:
            continue
        rgba = np.asarray(im)
        pad = max(4, min(w, h) // 8)
        filled = bleed(rgba[..., :3], rgba[..., 3])
        rgb = np.pad(filled, ((pad, pad), (pad, pad), (0, 0)), mode="wrap")
        model = args.text_model if is_text_sheet(rgba) else args.model
        src = os.path.join(work, "in_" + model)
        os.makedirs(src, exist_ok=True)
        Image.fromarray(rgb, "RGB").save(os.path.join(src, f))
        alpha = rgba[..., 3]
        if len(np.unique(alpha)) == 2:
            # Two-level masks (cut-outs, glyphs) go through the text model too: crisp shapes, not
            # Lanczos blobs. Smooth alpha is a blend weight: it is only resized (see below).
            src = os.path.join(work, "in_alpha")
            os.makedirs(src, exist_ok=True)
            mask = np.pad(alpha, ((pad, pad), (pad, pad)), mode="wrap").astype(np.float32) * (255.0 / max(1, int(alpha.max())))
            Image.fromarray(mask.round().astype(np.uint8), "L").convert("RGB").save(os.path.join(src, f))
        jobs[f] = (model, pad, w, h, alpha, filled)

    for model in sorted({j[0] for j in jobs.values()}):
        run(args.exe, args.models, model, os.path.join(work, "in_" + model), os.path.join(work, "out_" + model))
    run(args.exe, args.models, args.text_model, os.path.join(work, "in_alpha"), os.path.join(work, "out_alpha"))

    done = 0
    for f, (model, pad, w, h, alpha, filled) in jobs.items():
        up = Image.open(os.path.join(work, "out_" + model, f)).convert("RGB")
        p = pad * SCALE
        box = (p, p, p + w * SCALE, p + h * SCALE)
        rgb = Image.fromarray(back_project(np.asarray(up.crop(box)), filled), "RGB")
        lo, hi = int(alpha.min()), int(alpha.max())
        levels = len(np.unique(alpha))
        if levels == 2:
            mask = np.asarray(Image.open(os.path.join(work, "out_alpha", f)).convert("L").crop(box), np.float32)
            a = np.clip(mask * (hi / 255.0), lo, hi).round().astype(np.uint8)
        elif levels > 2:
            smooth = np.asarray(Image.fromarray(alpha.astype(np.float32), "F").resize((w * SCALE, h * SCALE), Image.BICUBIC))
            a = np.clip(back_project(smooth, alpha), lo, hi).astype(np.uint8)
        else:
            a = np.full((h * SCALE, w * SCALE), lo, np.uint8)
        Image.merge("RGBA", (*rgb.split(), Image.fromarray(a, "L"))).save(os.path.join(out, f), compress_level=3)
        done += 1
    shutil.rmtree(work, ignore_errors=True)
    text = sum(1 for j in jobs.values() if j[0] == args.text_model)
    print(f"wrote {done} replacements to {out} ({text} with {args.text_model}, the rest {args.model})")


if __name__ == "__main__":
    main()
