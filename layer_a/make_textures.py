"""Layer A: image-plane textures for the OpenSpace SGL scene.

Takes the real DSCOVR EPIC Earth (the thesis ground truth), places it at 30 pc, and models
what a telescope sweeping the image plane at 650 AU records, using the physical
aperture-averaged PSF from sgl_psf.py. Writes PNG textures, float32 .npy arrays and a
manifest of every parameter and measured number to out/.

Every picture is in color: EPIC's 680, 551 and 443 nm bands, each blurred, measured and
recovered on its own, then stacked (*_rgb.png). The averaged PSF doesn't depend on
wavelength, so every band blurs the same way. The 443 nm band the thesis used is also run
alone, for the measured numbers in the manifest and for the float arrays truth.npy and
image_plane.npy, which the Colab rerun and layer C's validation read.

    python make_textures.py              # N = 256 (about 5 minutes)
    python make_textures.py --n 128      # quicker
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

import sgl_psf as S

HERE = os.path.dirname(os.path.abspath(__file__))
EPIC_H5 = os.path.join(HERE, "..", "..", "code", "earth access", "epic_1b_20150613110250_03.h5")
EPIC_NPY = os.path.join(HERE, "..", "..", "code", "revised", "data", "epic_443nm_image.npy")
OUT = os.path.join(HERE, "out", "textures")

Z_OBS = 650 * S.AU
TARGET_DISTANCE = 30 * S.PC
PLANET_DIAMETER = 2 * S.R_EARTH_EQ
WAVELENGTH = 443e-9                 # only used for the thesis-PSF comparison
SNR_TARGETS = (1, 3, 10, 30)        # recovered-image SNRs to illustrate
EXTENT = 1.5                        # blurred image plane exported over 1.5 planet diameters
RGB_BANDS = ("Band680nm", "Band551nm", "Band443nm")
RGB_GAIN = 2.2                      # display gain, as in the thesis RGB figure (white = 45% of max)


def thesis_orientation(raw):
    raw = np.where(np.isfinite(raw), raw, 0.0).astype(np.float64)
    return np.fliplr(np.rot90(raw, k=-1))


def disk_box(raws):
    """Bounding box (left, top, right, bottom) of the Earth's disk in full-resolution pixels."""
    peak = max(r.max() for r in raws)
    ys, xs = np.nonzero(np.max([r / peak for r in raws], axis=0) > 0.02)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    r = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2
    return (int(round(cx - r)), int(round(cy - r)), int(round(cx + r)), int(round(cy + r)))


def disk_mask(n):
    yy, xx = np.mgrid[:n, :n]
    return np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) <= n / 2 - 0.5


def resample(raw, n, box, mask):
    """Area-average a full-resolution array over box down to n x n, zero outside the disk."""
    small = Image.fromarray(raw.astype(np.float32), mode="F").resize((n, n), Image.BOX, box=box)
    return np.where(mask, np.clip(np.asarray(small, dtype=np.float64), 0, None), 0.0)


def epic_box():
    """The crop the color and biosignature products share, from the three RGB bands."""
    import h5py
    with h5py.File(EPIC_H5, "r") as f:
        return disk_box([thesis_orientation(np.array(f[b]["Image"])) for b in RGB_BANDS])


def load_bands(n, color):
    """EPIC band(s) cropped to the disk's bounding box and area-averaged to n x n.

    Gray: the thesis 443 nm array, normalized to its own max. Color: three bands from the
    EPIC file, normalized by their common max so the colour balance survives.
    """
    if color:
        import h5py
        with h5py.File(EPIC_H5, "r") as f:
            raws = [thesis_orientation(np.array(f[b]["Image"])) for b in RGB_BANDS]
    else:
        raws = [thesis_orientation(np.load(EPIC_NPY))]
    peak = max(r.max() for r in raws)
    raws = [r / peak for r in raws]
    box = disk_box(raws)
    mask = disk_mask(n)
    bands = [resample(raw, n, box, mask) for raw in raws]
    top = max(b.max() for b in bands)
    return [b / top for b in bands], mask


def pad_center(a, size):
    out = np.zeros((size, size), dtype=a.dtype)
    o = (size - a.shape[0]) // 2
    out[o:o + a.shape[0], o:o + a.shape[1]] = a
    return out


def simulate(earth, mask, n, pitch, rng):
    """Blur, measure and recover one band. Returns arrays and numbers for that band."""
    n_ext = int(round(EXTENT * n / 2)) * 2
    thesis_kernel = S.thesis_kernel(pitch, Z_OBS, WAVELENGTH)
    res = {
        "n_ext": n_ext,
        "blurred_ext": S.SGLBlur(n_ext).apply(pad_center(earth, n_ext)),
        "thesis_ext": S.SGLBlur(n_ext, kernel_fn=thesis_kernel).apply(pad_center(earth, n_ext)),
    }
    op = S.SGLBlur(n)
    clean = op.apply(earth) * mask
    c = n // 2
    res["direct_share"] = earth[c, c] / clean[c, c]
    res["thesis_direct"] = earth[c, c] / S.SGLBlur(n, kernel_fn=thesis_kernel).apply(earth)[c, c]
    res["noiseless"], res["noiseless_its"] = S.solve_cg(op, clean, mask, tol=1e-10)

    # Measure this image's penalty once, then pick SNR_C to hit each target SNR_R.
    probe_snr_c = 1e4
    noisy = clean + clean[mask].mean() / probe_snr_c * rng.standard_normal(clean.shape) * mask
    rec, _ = S.solve_cg(op, noisy, mask, tol=1e-10)
    res["penalty"] = S.snr(earth, rec, mask) / probe_snr_c

    res["recoveries"] = []
    for target in SNR_TARGETS:
        snr_c = target / res["penalty"]
        sigma = clean[mask].mean() / snr_c
        noisy = (clean + sigma * rng.standard_normal(clean.shape)) * mask
        plain, its_p = S.solve_cg(op, noisy, mask, tol=1e-10)
        # Tikhonov filter, strength picked against the truth: an illustration of what
        # filtering buys, not a blind reconstruction method.
        best = None
        for reg in np.geomspace(1e-3, 30, 16) * np.sqrt(mask.sum()) / snr_c:
            f, _ = S.solve_cg(op, noisy, mask, reg=reg, tol=1e-8)
            e = S.snr(earth, f, mask)
            if best is None or e > best[0]:
                best = (e, reg, f)
        res["recoveries"].append({
            "target": target, "snr_c": snr_c, "noisy": noisy,
            "plain": plain, "plain_snr": S.snr(earth, plain, mask), "plain_its": its_p,
            "filtered": best[2], "filtered_snr": best[0], "reg": best[1],
        })
    return res


def save_png(name, img, lo=None, hi=None, mask=None, alpha=None):
    """8-bit grayscale or RGB PNG (img is 2-D or n x n x 3) with an explicit stretch.

    Returns the stretch for the manifest. With a mask, lo/hi default to the masked range.
    """
    vals = img[mask] if mask is not None else img
    lo = float(vals.min()) if lo is None else lo
    hi = float(vals.max()) if hi is None else hi
    g = (np.clip((img - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)
    if alpha is not None:
        a = (alpha * 255).astype(np.uint8)
        rgb = g if g.ndim == 3 else np.dstack([g, g, g])
        Image.fromarray(np.dstack([rgb, a]), mode="RGBA").save(os.path.join(OUT, name))
    else:
        Image.fromarray(g, mode="RGB" if g.ndim == 3 else "L").save(os.path.join(OUT, name))
    return {"file": name, "black": lo, "white": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="pixels across the planet's image")
    args = ap.parse_args()
    n = args.n
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)

    image_d = S.image_diameter(PLANET_DIAMETER, Z_OBS, TARGET_DISTANCE)
    pitch = image_d / n

    # ---- The thesis 443 nm band, for the numbers and the float arrays ----
    (earth,), mask = load_bands(n, color=False)
    r = simulate(earth, mask, n, pitch, rng)
    n_ext = r["n_ext"]

    manifest = {
        "description": "SGL image-plane textures for the OpenSpace portfolio scene (layer A).",
        "model": "Aperture-averaged SGL PSF mu(rho), Toth & Turyshev 2021 (arXiv:2012.05477) eq. 2-3",
        "ground_truth": "DSCOVR EPIC L1B, 2015-06-13 11:00:36 UTC (code/earth access/epic_1b_20150613110250_03.h5)",
        "geometry": {
            "z_obs_au": 650, "target_distance_pc": 30, "planet_diameter_m": PLANET_DIAMETER,
            "focal_distance_au": S.focal_distance() / S.AU,
            "einstein_angle_arcsec": float(np.degrees(S.einstein_angle(Z_OBS)) * 3600),
            "einstein_ring_in_solar_radii": float(S.einstein_angle(Z_OBS) * Z_OBS / S.R_SUN),
            "image_diameter_m": image_d,
            "pixels_across_image": n, "pitch_m": pitch, "telescope_aperture_m": pitch,
            "sampling": "fully sampled (aperture = pitch)",
            "extended_texture_span_m": n_ext * pitch,
        },
        "measured": {
            "band": "443 nm",
            "direct_share_center_physical": float(r["direct_share"]),
            "direct_share_center_thesis_psf": float(r["thesis_direct"]),
            "thesis_overweights_direct_by": float(r["thesis_direct"] / r["direct_share"]),
            "noiseless_recovery_rmse": float(np.sqrt(np.mean((r["noiseless"] - earth)[mask] ** 2))),
            "noiseless_cg_iterations": r["noiseless_its"],
            "penalty_snr_r_over_snr_c": float(r["penalty"]),
            "penalty_times_sqrt_n_disk": float(r["penalty"] * np.sqrt(mask.sum())),
            "disk_pixels": int(mask.sum()),
        },
        "recoveries": [],
        "arrays": {
            "truth.npy": "443 nm planet, n x n, max 1",
            "image_plane.npy": "443 nm image plane under the physical PSF, extended span",
            "truth_rgb.npy": "680 / 551 / 443 nm planet, n x n x 3, common max 1",
            "image_plane_rgb.npy": "680 / 551 / 443 nm image plane, extended span, n_ext x n_ext x 3",
        },
    }
    np.save(os.path.join(OUT, "truth.npy"), earth.astype(np.float32))
    np.save(os.path.join(OUT, "image_plane.npy"), r["blurred_ext"].astype(np.float32))
    for rec in r["recoveries"]:
        manifest["recoveries"].append({
            "target_snr_r": rec["target"], "snr_c_required": float(rec["snr_c"]),
            "plain_snr_r": float(rec["plain_snr"]), "plain_cg_iterations": rec["plain_its"],
            "filtered_snr_r": float(rec["filtered_snr"]), "filter_reg": float(rec["reg"]),
        })
        print(f"443 nm  SNR_R {rec['target']:>2}: needs SNR_C {rec['snr_c']:8.0f} | plain {rec['plain_snr']:.2f}"
              f" | filtered {rec['filtered_snr']:.2f}")

    # ---- Color: 680 / 551 / 443 nm, each band simulated on its own, then stacked ----
    bands, mask_c = load_bands(n, color=True)
    runs = []
    for name, band in zip(RGB_BANDS, bands):
        runs.append(simulate(band, mask_c, n, pitch, rng))
        print(f"{name}: penalty x sqrt(N) = {runs[-1]['penalty'] * np.sqrt(mask_c.sum()):.3f}")
    stack = lambda key: np.dstack([run[key] for run in runs])
    white = 1.0 / RGB_GAIN
    ext_mask_c = pad_center(mask_c, n_ext)
    glow = stack("blurred_ext")
    glow_lo, glow_hi = float(glow[ext_mask_c].min()), float(glow[ext_mask_c].max())
    thesis = stack("thesis_ext")
    color = {
        "bands": list(RGB_BANDS),
        "display": f"linear, white point {white:.3f} of the brightest band (gain {RGB_GAIN})",
        "penalty_times_sqrt_n_disk": {b: float(run["penalty"] * np.sqrt(mask_c.sum()))
                                      for b, run in zip(RGB_BANDS, runs)},
        "textures": {
            "truth": save_png("truth_rgb.png", np.dstack(bands), 0.0, white),
            "truth_rgba": save_png("truth_rgba_rgb.png", np.dstack(bands), 0.0, white,
                                   alpha=mask_c.astype(float)),
            "image_plane_stretched": save_png("image_plane_stretched_rgb.png", glow, glow_lo, glow_hi),
            "image_plane_thesis_psf": save_png("image_plane_thesis_psf_rgb.png", thesis,
                                               float(thesis[ext_mask_c].min()), float(thesis[ext_mask_c].max())),
            "recovered_noiseless": save_png("recovered_noiseless_rgb.png", stack("noiseless"), 0.0, white),
        },
        "recoveries": [],
    }
    for i, target in enumerate(SNR_TARGETS):
        tag = f"snr{target:02d}"
        noisy = np.dstack([run["recoveries"][i]["noisy"] for run in runs])
        plain = np.dstack([run["recoveries"][i]["plain"] for run in runs])
        filt = np.dstack([run["recoveries"][i]["filtered"] for run in runs])
        color["recoveries"].append({
            "target_snr_r": target,
            "snr_c_required": {b: float(run["recoveries"][i]["snr_c"]) for b, run in zip(RGB_BANDS, runs)},
            "filtered_snr_r": {b: float(run["recoveries"][i]["filtered_snr"]) for b, run in zip(RGB_BANDS, runs)},
            "measurements": save_png(f"measurements_{tag}_rgb.png", noisy, mask=mask_c),
            "plain": save_png(f"recovered_plain_{tag}_rgb.png", plain, 0.0, white),
            "filtered": save_png(f"recovered_filtered_{tag}_rgb.png", filt, 0.0, white),
        })
    np.save(os.path.join(OUT, "truth_rgb.npy"), np.dstack(bands).astype(np.float32))
    np.save(os.path.join(OUT, "image_plane_rgb.npy"), glow.astype(np.float32))
    manifest["color"] = color

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    m = manifest["measured"]
    g = manifest["geometry"]
    print(f"image diameter {g['image_diameter_m']:.0f} m, pitch {g['pitch_m']:.2f} m, ring {g['einstein_angle_arcsec']:.2f}\"")
    print(f"direct share at center: physical {m['direct_share_center_physical']:.5f},"
          f" thesis PSF {m['direct_share_center_thesis_psf']:.4f} ({m['thesis_overweights_direct_by']:.0f}x)")
    print(f"noiseless recovery RMSE {m['noiseless_recovery_rmse']:.2e} in {m['noiseless_cg_iterations']} CG iterations")
    print(f"penalty SNR_R/SNR_C = {m['penalty_snr_r_over_snr_c']:.5f} (x sqrt(N_disk) = {m['penalty_times_sqrt_n_disk']:.3f})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
