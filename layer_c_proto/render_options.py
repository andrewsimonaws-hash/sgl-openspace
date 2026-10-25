"""Render the telescope-view options for review (layer C prototype).

    python render_options.py [openspace_screenshot.png]

Writes out/visuals/: a sequence as the telescope crosses the image plane, three options each
for planet exaggeration, optics, and Sun treatment, and (given a screenshot) three inset
layouts mocked up on the OpenSpace scene.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ring_view as R

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "layer_a", "out", "textures")
OUT = os.path.join(HERE, "out", "visuals")
GAIN = 2.2              # same display gain as the layer-A colour textures
SIZE = 640
BASE_U = (0.35, 0.2)    # telescope position for the option panels (image radii)


def to_png(img, name, gain=GAIN):
    a = (np.clip(img * gain, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(a).save(os.path.join(OUT, name))
    return os.path.join(OUT, name)


def minimap(glow, u, size=150):
    """Small image-plane map with the telescope's position marked."""
    m = glow.resize((size, size))
    d = ImageDraw.Draw(m)
    # The glow texture spans 1.5 image diameters, so one image radius is size / 3 pixels.
    cx = size / 2 + u[0] * size / 3
    cy = size / 2 - u[1] * size / 3
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=(255, 170, 60), width=2)
    d.rectangle([0, 0, size - 1, size - 1], outline=(90, 90, 90))
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    rgb = np.load(os.path.join(TEX, "truth_rgb.npy")).astype(np.float64)
    gray = np.load(os.path.join(TEX, "truth.npy")).astype(np.float64)
    glow = Image.open(os.path.join(TEX, "image_plane_stretched_rgb.png")).convert("RGB")

    # Sequence: the telescope walks from the centre of the image to beyond its edge.
    frames = []
    for u in [(0.0, 0.0), (0.5, 0.0), (0.95, 0.0), (1.35, 0.0)]:
        img = R.render(rgb, u, 0.05, size=420, occulter=1.02)
        im = Image.fromarray((np.clip(img * GAIN, 0, 1) * 255).astype(np.uint8))
        im.paste(minimap(glow, u, 110), (420 - 116, 6))
        frames.append(im)
    strip = Image.new("RGB", (420 * 4 + 30, 420), (0, 0, 0))
    for i, f in enumerate(frames):
        strip.paste(f, (i * 430, 0))
    strip.save(os.path.join(OUT, "sequence.png"))

    # A. How much to magnify the planet.
    for k, name in [(0.02, "a_k02"), (0.05, "a_k05"), (0.12, "a_k12")]:
        to_png(R.render(rgb, BASE_U, k, size=SIZE), f"{name}.png")
        print(name, f"planet drawn {R.exaggeration(k):,.0f}x larger than real")

    # B. Optics: sharp geometric view, diffraction-limited 1 m telescope at 443 nm, or both.
    sharp = R.render(rgb, BASE_U, 0.05, size=SIZE)
    real = R.diffraction_view(gray, rgb, BASE_U, 0.05, size=SIZE)
    to_png(sharp, "b_sharp.png")
    to_png(real, "b_diffraction.png", gain=1.0)
    split = np.clip(sharp * GAIN, 0, 1).copy()
    split[:, SIZE // 2:] = np.clip(real[:, SIZE // 2:], 0, 1)
    split[:, SIZE // 2 - 1:SIZE // 2 + 1] = 0.35
    to_png(split, "b_split.png", gain=1.0)

    # C. The Sun: coronagraph occulter only, occulter plus a stylised corona, or a dimmed disk.
    to_png(R.render(rgb, BASE_U, 0.05, size=SIZE, occulter=1.02), "c_occulter.png")
    to_png(R.render(rgb, BASE_U, 0.05, size=SIZE, occulter=1.02, corona=0.06), "c_corona.png")
    to_png(R.render(rgb, BASE_U, 0.05, size=SIZE, occulter=0), "c_sun.png")

    # D. Inset layouts on an OpenSpace screenshot.
    if len(sys.argv) > 1:
        shot = Image.open(sys.argv[1]).convert("RGB")
        W = 1600
        shot = shot.resize((W, int(shot.height * W / shot.width)))
        H = shot.height
        view = Image.open(os.path.join(OUT, "c_corona.png"))
        try:
            font = ImageFont.truetype("segoeui.ttf", 18)
        except OSError:
            font = ImageFont.load_default()

        sq = shot.copy()
        s = int(H * 0.42)
        sq.paste(view.resize((s, s)), (W - s - 24, H - s - 24))
        d = ImageDraw.Draw(sq)
        d.rectangle([W - s - 25, H - s - 25, W - 23, H - 23], outline=(200, 200, 200), width=2)
        d.text((W - s - 20, H - s - 52), "Telescope view - planet x57,000", fill=(230, 230, 230), font=font)
        sq.save(os.path.join(OUT, "d_square.jpg"), quality=88)

        ep = shot.copy()
        s = int(H * 0.46)
        circ = view.resize((s, s))
        mask = Image.new("L", (s, s), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, s - 1, s - 1], fill=255)
        ep.paste(circ, (W - s - 24, H - s - 24), mask)
        d = ImageDraw.Draw(ep)
        x0, y0 = W - s - 24, H - s - 24
        d.ellipse([x0, y0, x0 + s, y0 + s], outline=(200, 200, 200), width=3)
        d.line([x0 + s / 2, y0 + 8, x0 + s / 2, y0 + 30], fill=(200, 200, 200), width=2)
        d.line([x0 + s / 2, y0 + s - 30, x0 + s / 2, y0 + s - 8], fill=(200, 200, 200), width=2)
        d.line([x0 + 8, y0 + s / 2, x0 + 30, y0 + s / 2], fill=(200, 200, 200), width=2)
        d.line([x0 + s - 30, y0 + s / 2, x0 + s - 8, y0 + s / 2], fill=(200, 200, 200), width=2)
        ep.save(os.path.join(OUT, "d_eyepiece.jpg"), quality=88)

        sp = Image.new("RGB", (W, H), (0, 0, 0))
        left = shot.crop((W // 4, 0, W // 4 + W // 2, H))
        sp.paste(left, (0, 0))
        s = min(W // 2, H) - 40
        sp.paste(view.resize((s, s)), (W // 2 + (W // 2 - s) // 2, (H - s) // 2))
        ImageDraw.Draw(sp).line([W // 2, 0, W // 2, H], fill=(80, 80, 80), width=2)
        sp.save(os.path.join(OUT, "d_split.jpg"), quality=88)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
