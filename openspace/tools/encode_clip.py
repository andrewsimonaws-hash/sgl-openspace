"""Turn the latest exported OpenSpace frames into an MP4 (and optionally a preview GIF).

    python encode_clip.py out.mp4 [--runs 2] [--width 1920] [--crf 20] [--gif preview.gif] [--fps 30]
                          [--gif-width 600] [--gif-step 4]

Takes the newest folder in OpenSpace's user/screenshots (see ../paths.py), or --folder, and
inside it the last unbroken run of frames (a gap of more than 20 s between files separates
runs). --runs 2 joins the last two runs in order, for the intro exported straight before the
reel. --width scales supersampled frames down (Lanczos) to the clip's width.
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402


def latest_runs(n, folder=None):
    if folder is None:
        shots = paths.screenshots()
        folder = max((os.path.join(shots, d) for d in os.listdir(shots)), key=os.path.getmtime)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    times = [os.path.getmtime(os.path.join(folder, f)) for f in files]
    start, runs = len(files) - 1, 0
    while start > 0:
        if times[start] - times[start - 1] >= 20:
            runs += 1
            if runs == n:
                break
        start -= 1
    return [os.path.join(folder, f) for f in files[start:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--runs", type=int, default=1, help="join the last n runs of frames")
    ap.add_argument("--width", type=int, help="scale the frames down to this width")
    ap.add_argument("--folder", help="frame folder (default: the newest in user/screenshots)")
    ap.add_argument("--gif")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=20, help="x264 quality (lower is better)")
    ap.add_argument("--gif-width", type=int, default=600)
    ap.add_argument("--gif-step", type=int, default=4, help="keep every n-th frame in the GIF")
    args = ap.parse_args()

    frames = latest_runs(args.runs, args.folder)
    scale = ["-vf", f"scale={args.width}:-2:flags=lanczos"] if args.width else []
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=None, macro_block_size=1,
                            output_params=scale + ["-crf", str(args.crf), "-pix_fmt", "yuv420p",
                                                   "-movflags", "+faststart"]) as w:
        for f in frames:
            img = np.asarray(Image.open(f).convert("RGB"))
            h, wd = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2      # x264 needs even sizes
            w.append_data(img[:h, :wd])
    print(f"wrote {args.out}: {len(frames)} frames, {len(frames) / args.fps:.1f} s")

    if args.gif:
        small = []
        for f in frames[::args.gif_step]:
            im = Image.open(f).convert("RGB")
            small.append(im.resize((args.gif_width, int(im.height * args.gif_width / im.width)), Image.LANCZOS))
        # One shared palette, built from frames across the clip, and no dithering:
        # dithered star fields compress badly.
        picks = small[:: max(1, len(small) // 8)][:8]
        strip = Image.new("RGB", (small[0].width, small[0].height * len(picks)))
        for i, im in enumerate(picks):
            strip.paste(im, (0, i * small[0].height))
        palette = strip.quantize(256, method=Image.Quantize.MEDIANCUT)
        small = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in small]
        small[0].save(args.gif, save_all=True, append_images=small[1:],
                      duration=int(1000 * args.gif_step / args.fps), loop=0, optimize=True)
        print(f"wrote {args.gif}: {len(small)} frames")


if __name__ == "__main__":
    main()
