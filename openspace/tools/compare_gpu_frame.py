"""Compare the GPU telescope view (the Sgl module's SaveFrame output) with the NumPy reference.

    python compare_gpu_frame.py                    # compare the committed GPU frame
    python compare_gpu_frame.py --capture out.png  # save a fresh frame from a running OpenSpace first

The reference is layer_c_proto/out/visuals/c_occulter.png, rendered by render_options.py with
ring_view.render(rgb, u=(0.35, 0.2), kappa=0.05, size=640, occulter=1.02). The GPU frame must
use the same settings, which --capture sets on ScreenSpace.SGL_Telescope: no telescope node, a
manual offset of (0.35, 0.2), the medium planet size (kappa = 0.05), the default 1.02 R_sun
occulter, no corona, no eyepiece, 640 px and gain 1 (the PNG texture already carries its display
gain). Saved textures come out bottom-up, so the GPU frame is flipped before comparing.

A correlation alone is a weak check, so this also prints the mean difference on the ring, the
share of pixels off by more than 16 and 32 levels, and the correlation of mirrored and rotated
copies of the GPU frame, which should score far lower if the orientation is right.
"""
import argparse
import os
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO = os.path.normpath(os.path.join(HERE, "..", ".."))
REFERENCE = os.path.join(PORTFOLIO, "layer_c_proto", "out", "visuals", "c_occulter.png")
GPU_FRAME = os.path.join(PORTFOLIO, "layer_c_proto", "out", "gpu_frame_c_occulter.png")
P = "ScreenSpace.SGL_Telescope."


def capture(path):
    from oslua import send
    path = os.path.abspath(path).replace("\\", "/")
    send([f"openspace.setPropertyValueSingle('{P}TelescopeNode', '')",
          f"openspace.setPropertyValueSingle('{P}ManualOffset', {{0.35, 0.2}})",
          f"openspace.setPropertyValueSingle('{P}PlanetSize', 1)",
          f"openspace.setPropertyValueSingle('{P}OcculterRadius', 1.02)",
          f"openspace.setPropertyValueSingle('{P}CoronaStrength', 0.0)",
          f"openspace.setPropertyValueSingle('{P}Eyepiece', false)",
          f"openspace.setPropertyValueSingle('{P}Resolution', 640)",
          f"openspace.setPropertyValueSingle('{P}Gain', 1.0)",
          f"openspace.setPropertyValueSingle('{P}SaveFramePath', '{path}')"], timeout=60)
    time.sleep(2)
    send([f"openspace.setPropertyValueSingle('{P}SaveFrame', nil)"], timeout=60)
    time.sleep(2)
    return path


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", metavar="PNG", help="save a fresh GPU frame from a running OpenSpace to this path")
    ap.add_argument("--gpu", default=GPU_FRAME, help="GPU frame to compare (ignored with --capture)")
    ap.add_argument("--reference", default=REFERENCE)
    args = ap.parse_args()

    gpu_path = capture(args.capture) if args.capture else args.gpu
    gpu = load(gpu_path)[::-1]                  # saved textures are bottom-up
    ref = load(args.reference)
    assert gpu.shape == ref.shape, (gpu.shape, ref.shape)

    diff = np.abs(gpu - ref)
    ring = np.maximum(gpu.max(axis=2), ref.max(axis=2)) > 8
    worst = diff.max(axis=2)
    r = np.corrcoef(gpu.ravel(), ref.ravel())[0, 1]
    print(f"correlation r = {r:.4f}")
    print(f"mean |difference|: {diff.mean():.2f}/255 overall, {diff[ring].mean():.2f}/255 on the ring")
    print(f"pixels off by more than 16/255: {100 * (worst > 16).mean():.3f}%, more than 32/255: {100 * (worst > 32).mean():.3f}%")
    for name, g in {"mirrored left-right": gpu[:, ::-1], "rotated 180 degrees": gpu[::-1, ::-1]}.items():
        print(f"orientation check, {name}: r = {np.corrcoef(g.ravel(), ref.ravel())[0, 1]:.4f}")


if __name__ == "__main__":
    main()
