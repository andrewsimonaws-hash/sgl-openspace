"""Turn the latest exported OpenSpace frames into an MP4 (and optionally a preview GIF).

    python encode_clip.py out.mp4 [--gif preview.gif] [--fps 30] [--gif-width 600] [--gif-step 4]

Takes the newest folder in user/screenshots and, inside it, the last unbroken run of
frames (a gap of more than 30 s between files separates runs).
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

SHOTS = r"C:\Users\andre\OpenSpace\v0.22.0\user\screenshots"


def latest_run():
    folder = max((os.path.join(SHOTS, d) for d in os.listdir(SHOTS)), key=os.path.getmtime)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    times = [os.path.getmtime(os.path.join(folder, f)) for f in files]
    start = len(files) - 1
    while start > 0 and times[start] - times[start - 1] < 30:
        start -= 1
    return [os.path.join(folder, f) for f in files[start:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gif")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-width", type=int, default=600)
    ap.add_argument("--gif-step", type=int, default=4, help="keep every n-th frame in the GIF")
    args = ap.parse_args()

    frames = latest_run()
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=None,
                            output_params=["-crf", "18", "-pix_fmt", "yuv420p",
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
        # One shared palette and no dithering: dithered star fields compress badly.
        palette = small[len(small) // 2].quantize(256, method=Image.Quantize.MEDIANCUT)
        small = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in small]
        small[0].save(args.gif, save_all=True, append_images=small[1:],
                      duration=int(1000 * args.gif_step / args.fps), loop=0, optimize=True)
        print(f"wrote {args.gif}: {len(small)} frames")


if __name__ == "__main__":
    main()
