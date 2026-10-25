"""Turn the latest exported OpenSpace frames into an MP4 (and optionally a preview GIF).

    python encode_clip.py out.mp4 [--gif preview.gif] [--fps 30]

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
        for f in frames[::3]:
            im = Image.open(f).convert("RGB")
            small.append(im.resize((640, int(im.height * 640 / im.width))))
        small[0].save(args.gif, save_all=True, append_images=small[1:],
                      duration=int(3000 / args.fps), loop=0, optimize=True)
        print(f"wrote {args.gif}: {len(small)} frames")


if __name__ == "__main__":
    main()
