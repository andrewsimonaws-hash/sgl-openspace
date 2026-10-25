"""Draw the three trajectory scenarios over the image-plane texture as a quick check.

    python preview.py    # writes out/preview.png
"""
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
COLORS = [(255, 170, 60), (90, 160, 255), (120, 220, 140), (240, 110, 200)]


def main():
    with open(os.path.join(OUT, "textures", "manifest.json")) as f:
        tex = json.load(f)
    pitch = tex["geometry"]["pitch_m"]
    base = Image.open(os.path.join(OUT, "textures", "image_plane_stretched.png")).convert("RGB")
    size = base.size[0]
    c = (size - 1) / 2

    def to_px(x, y):
        return (x / pitch + c, c - y / pitch)

    panels = []
    for name in ("m0_spiral", "raster_single", "formation12"):
        img = base.copy()
        d = ImageDraw.Draw(img)
        paths = {}
        with open(os.path.join(OUT, "trajectories", f"{name}.csv")) as f:
            for row in csv.DictReader(f):
                paths.setdefault(int(row["sc"]), []).append(to_px(float(row["x_m"]), float(row["y_m"])))
        for sc, pts in paths.items():
            d.line(pts, fill=COLORS[sc % len(COLORS)], width=1)
        d.text((6, 6), name, fill=(255, 255, 255))
        panels.append(img)
    sheet = Image.new("RGB", (size * len(panels), size))
    for i, p in enumerate(panels):
        sheet.paste(p, (i * size, 0))
    sheet.save(os.path.join(OUT, "preview.png"))
    print("wrote", os.path.join(OUT, "preview.png"))


if __name__ == "__main__":
    main()
