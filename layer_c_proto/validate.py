"""Checks for the telescope-view math before it becomes a shader.

1. A point source behind the Sun appears as a ring of radius theta_E = 1.6076" (650 AU).
2. Integrating the rendered ring gives the light a telescope collects. Across the image plane
   that must follow the layer-A image-plane brightness (built from the wave-optics PSF). If the
   ray trace or a sign were wrong, it would follow a flipped copy instead, so the flipped
   copies are tested too.
3. The integrated light doesn't depend on the exaggeration kappa (so drawing the planet big
   doesn't change what the telescope measures).

    python validate.py      # prints the results, writes out/validation.json, exits 1 on a failure
"""
import json
import os
import sys

import numpy as np

import ring_view as R

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "layer_a", "out", "textures")
OUT = os.path.join(HERE, "out")


def ring_flux(tex, u, kappa, n_r=1600, n_az=1440):
    """Total planet light in the telescope's view, by integrating over the ring region."""
    phi = (np.arange(n_az) + 0.5) / n_az * 2 * np.pi
    span = 1.5 * kappa * (1 + np.hypot(*u))
    rr = np.linspace(max(1 - span, 1e-3), 1 + span, n_r)
    P, Rr = np.meshgrid(phi, rr)
    val = R.lensed_planet(tex, Rr * np.cos(P), Rr * np.sin(P), u, kappa)
    return float((val * Rr).sum() * (rr[1] - rr[0]) * (phi[1] - phi[0]))


def main():
    os.makedirs(OUT, exist_ok=True)
    truth = np.load(os.path.join(TEX, "truth.npy")).astype(np.float64)
    plane = np.load(os.path.join(TEX, "image_plane.npy")).astype(np.float64)
    with open(os.path.join(TEX, "manifest.json")) as f:
        geo = json.load(f)["geometry"]
    n_ext = plane.shape[0]
    radius_px = geo["pixels_across_image"] / 2

    def plane_at(ux, uy):
        """Layer-A image-plane brightness at image-plane position (ux, uy) in image radii."""
        c = (n_ext - 1) / 2
        return float(R.bilinear(plane, ux * radius_px / c, uy * radius_px / c))

    results = {}

    # 1. Ring radius from a point-like source (tiny planet, telescope on axis).
    tiny = np.ones((8, 8))
    img = R.render(tiny, (0.0, 0.0), 0.004, size=1024, fov=1.2, supersample=1, occulter=0)
    s = (np.arange(1024) + 0.5) / 1024 * 2 - 1
    tx, ty = np.meshgrid(s * 1.2, -s * 1.2)
    w = img[..., 0] * (np.hypot(tx, ty) > 1.0 * R.THETA_SUN / R.THETA_E)
    r_ring = float((np.hypot(tx, ty) * w).sum() / w.sum())
    results["ring_radius_arcsec"] = r_ring * np.degrees(R.THETA_E) * 3600
    results["theta_E_arcsec"] = float(np.degrees(R.THETA_E) * 3600)
    results["ring_in_solar_radii"] = r_ring * R.THETA_E / R.THETA_SUN
    print(f"[1] ring radius {results['ring_radius_arcsec']:.4f}\" (theta_E {results['theta_E_arcsec']:.4f}\"),"
          f" {results['ring_in_solar_radii']:.4f} solar radii")

    # 2. Integrated ring light vs layer-A brightness, on a grid of telescope positions.
    kappa = 0.05
    pts = [(x, y) for x in np.linspace(-1.2, 1.2, 9) for y in np.linspace(-1.2, 1.2, 9)]
    flux = np.array([ring_flux(truth, p, kappa) for p in pts])
    tests = {
        "as built": [plane_at(x, y) for x, y in pts],
        "flipped x": [plane_at(-x, y) for x, y in pts],
        "flipped y": [plane_at(x, -y) for x, y in pts],
        "rotated 180": [plane_at(-x, -y) for x, y in pts],
    }
    corr = {k: float(np.corrcoef(flux, v)[0, 1]) for k, v in tests.items()}
    spread = {k: float(100 * np.std(flux / np.array(v)) / np.mean(flux / np.array(v))) for k, v in tests.items()}
    results["flux_vs_image_plane"] = {
        "positions": len(pts), "kappa": kappa, "correlation": corr, "ratio_spread_percent": spread,
    }
    print("[2] integrated ring light vs layer-A brightness (correlation, spread of the ratio):")
    for k in tests:
        print(f"      {k:12s} r = {corr[k]:.4f}   ratio spread {spread[k]:5.1f}%")

    # A transect for plotting: telescope moving along x through the centre.
    xs = np.linspace(-1.4, 1.4, 57)
    f_line = np.array([ring_flux(truth, (x, 0.0), kappa) for x in xs])
    b_line = np.array([plane_at(x, 0.0) for x in xs])
    scale = (f_line @ b_line) / (f_line @ f_line)
    results["transect"] = {"u": xs.tolist(), "ring_flux_scaled": (f_line * scale).tolist(),
                           "image_plane": b_line.tolist()}

    # 3. Exaggeration: magnifying the planet by E multiplies its area by E^2 and divides the
    # lens magnification by E, so the ring's total light grows as kappa. Light / kappa should
    # not change, which means the pattern across the image plane is unaffected.
    ks = [0.02, 0.05, 0.12]
    probe = [(0.0, 0.0), (0.5, 0.3), (-0.8, 0.1)]
    table = {k: [ring_flux(truth, p, k) / k for p in probe] for k in ks}
    spread = [float(np.ptp([table[k][i] for k in ks]) / np.mean([table[k][i] for k in ks]) * 100)
              for i in range(len(probe))]
    results["kappa_scaling"] = {"kappas": ks, "light_over_kappa_spread_percent": spread}
    print(f"[3] ring light / kappa for kappa {ks}: spread {', '.join(f'{v:.2f}%' for v in spread)}")

    with open(os.path.join(OUT, "validation.json"), "w") as f:
        json.dump(results, f, indent=2)

    fit = results["flux_vs_image_plane"]["ratio_spread_percent"]
    checks = {
        "ring radius within 0.1% of theta_E":
            abs(results["ring_radius_arcsec"] / results["theta_E_arcsec"] - 1) < 1e-3,
        "ring light follows the image plane (r > 0.9999, ratio spread < 1%)":
            corr["as built"] > 0.9999 and fit["as built"] < 1.0,
        "flipped and rotated copies fit at least 5x worse":
            min(v for k, v in fit.items() if k != "as built") > 5 * fit["as built"],
        "ring light / kappa varies by less than 1%": max(spread) < 1.0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        sys.exit("FAILED: " + "; ".join(failed))
    print("all checks passed")


if __name__ == "__main__":
    main()
