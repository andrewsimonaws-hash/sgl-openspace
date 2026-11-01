"""Layer A: would the planet look alive through the lens? Vegetation and oxygen maps.

Two classic biosignatures, from three of EPIC's calibrated bands (reflectance = K x counts/s,
with the factors from epic.gsfc.nasa.gov/science/calibration; 680 and 780 nm from
Geogdzhayev & Marshak 2018, 764 nm from the EPIC team's lunar-ratio O2 calibration):

  vegetation  NDVI = (R780 - R680) / (R780 + R680). Chlorophyll absorbs red light and leaves
              reflect near-infrared, so NDVI is high over forest and grassland and low over
              desert, ocean and cloud.
  oxygen      depth of the O2 A-band at 764 nm against the 780 nm continuum,
              1 - R764 / R780. Oxygen is well mixed, so the band shows everywhere; its depth
              tracks the light path through the air: shallow over high cloud, deep toward
              the limb. (EPIC uses the band to measure cloud height.)

Each band is blurred with the same aperture-averaged SGL PSF as make_textures.py, given
Gaussian noise and recovered with the (A + aI) filter of sgl_psf.solve_cg. The strength a
is set by the discrepancy principle (the residual matches the noise), which needs the noise
level but not the true planet. The truth-tuned strength is reported next to it as an upper
bound. NDVI and depth are computed from the recovered bands and scored against the same
quantities from the true bands, per pixel and over 32 x 32 blocks (about 400 km on Earth).

Cases:
  snr03, snr10, snr30  256 x 256 with aperture = pitch (5.23 m). The noise gives each band
                       the plain-deconvolution SNR in the name, as in make_textures.py.
  tt_fig4              128 x 128 with a 1 m telescope (pitch 10.5 m), the regime of Toth &
                       Turyshev (2021) Fig. 4. The noise gives plain deconvolution the SNR
                       their simulation reached there, 4.1. (Their SNR_C of 50 would give 5.4
                       here: they count the light of a circular aperture, a factor pi/4, and
                       all 128^2 pixels rather than the disk.)

The 680 and 780 nm images were taken 54 s apart, so the Earth turns about half a 256-grid
pixel between them; NDVI picks up a thin rim along sharp cloud edges from that.

    python make_biosignatures.py      # about 5 minutes; writes out/biosignatures/
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import make_textures as T
import sgl_psf as S

OUT = os.path.join(T.HERE, "out", "biosignatures")
CAL = {"Band680nm": 9.300e-6, "Band764nm": 2.360e-5, "Band780nm": 1.435e-5}
# (lat range, lon range) in degrees; all three are in view at 11:00 UTC
REGIONS = {
    "Congo basin": ((-5, 5), (12, 30)),
    "Sahara": ((18, 28), (-5, 25)),
    "Europe": ((45, 55), (0, 20)),
}
CASES = [
    {"name": "snr30", "n": 256, "aperture_m": None, "snr_r": 30},
    {"name": "snr10", "n": 256, "aperture_m": None, "snr_r": 10},
    {"name": "snr03", "n": 256, "aperture_m": None, "snr_r": 3},
    {"name": "tt_fig4", "n": 128, "aperture_m": 1.0, "snr_r": 4.1},
]
BLOCKS = 32                  # blocks across the disk for the continent-scale scores
VEGETATED = 0.3              # NDVI above this counts as vegetated

# Display colormaps: (value, RGB) stops, linear in between.
NDVI_STOPS = [(-0.3, (25, 55, 120)), (-0.05, (95, 150, 205)), (0.02, (232, 230, 222)),
              (0.12, (205, 170, 110)), (0.3, (140, 185, 70)), (0.6, (25, 105, 45))]
DEPTH_STOPS = [(0.35, (250, 245, 215)), (0.55, (120, 190, 200)), (0.75, (40, 90, 160)),
               (0.9, (20, 25, 80))]


def load(n):
    """Reflectance per band, the disk mask and region masks on the n x n grid."""
    import h5py
    box = T.epic_box()
    mask = T.disk_mask(n)
    with h5py.File(T.EPIC_H5, "r") as f:
        refl = {b: k * T.resample(T.thesis_orientation(np.array(f[b]["Image"])), n, box, mask)
                for b, k in CAL.items()}
        geo = f["Band780nm"]["Geolocation"]["Earth"]
        lat, lon = np.array(geo["Latitude"]), np.array(geo["Longitude"])
    valid = np.isfinite(lat) & np.isfinite(lon)
    regions = {}
    for name, ((la0, la1), (lo0, lo1)) in REGIONS.items():
        inside = valid & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        frac = T.resample(T.thesis_orientation(inside.astype(np.float64)), n, box, mask)
        regions[name] = frac > 0.5
    return refl, mask, regions


def ndvi(r680, r780):
    s = r780 + r680
    return np.clip((r780 - r680) / np.where(s > 0, s, np.inf), -1, 1)


def depth(r764, r780):
    return np.clip(1 - r764 / np.where(r780 > 0, r780, np.inf), 0, 1)


def blocks(img, mask, k):
    """Sum img over k x k blocks; also return which blocks lie fully inside the disk."""
    n = img.shape[0]
    b = n // k
    sums = img[:b * k, :b * k].reshape(k, b, k, b).sum(axis=(1, 3))
    full = mask[:b * k, :b * k].reshape(k, b, k, b).all(axis=(1, 3))
    return sums, full


def recover(op, band, mask, sigma, rng):
    """Noisy measurement of one band, then blind (discrepancy) and truth-tuned recoveries."""
    clean = op.apply(band) * mask
    noisy = (clean + sigma * rng.standard_normal(clean.shape)) * mask
    noise_norm = sigma * np.sqrt(mask.sum())
    cache = {}

    def solve(a):
        if a not in cache:
            x, _ = S.solve_cg(op, noisy, mask, reg=a, tol=1e-7)
            # (A + aI) x = b, so the data residual A x - b is just -a x
            cache[a] = (x, a * np.sqrt(np.sum(x[mask] ** 2)))
        return cache[a]

    grid = op.kernel_sum * np.geomspace(1e-6, 1, 13)
    resid = [solve(a)[1] for a in grid]
    # Discrepancy principle: the strongest filter whose residual stays within the noise.
    i = max([k for k, r in enumerate(resid) if r <= noise_norm], default=0)
    lo, hi = np.log(grid[i]), np.log(grid[min(i + 1, len(grid) - 1)])
    for _ in range(5):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if solve(np.exp(mid))[1] <= noise_norm else (lo, mid)
    blind_a = np.exp(lo)
    oracle_a = max(cache, key=lambda a: S.snr(band, cache[a][0], mask))
    plain, _ = S.solve_cg(op, noisy, mask, tol=1e-8)
    return {
        "noisy": noisy, "blind": cache[blind_a][0], "oracle": cache[oracle_a][0],
        "numbers": {
            "sigma": float(sigma), "snr_c": float(clean[mask].mean() / sigma),
            "plain_snr_r": float(S.snr(band, plain, mask)),
            "blind_snr_r": float(S.snr(band, cache[blind_a][0], mask)),
            "oracle_snr_r": float(S.snr(band, cache[oracle_a][0], mask)),
            "blind_reg": float(blind_a), "oracle_reg": float(oracle_a),
        },
    }


def score(refl, rec, mask, regions):
    """Compare NDVI and A-band depth from recovered bands with those from the true bands."""
    t680, t764, t780 = (refl[b] for b in CAL)
    r680, r764, r780 = (rec[b] for b in CAL)
    out = {}
    for key, f, a, b, ta, tb in [("ndvi", ndvi, r680, r780, t680, t780),
                                 ("o2_depth", depth, r764, r780, t764, t780)]:
        truth, est = f(ta, tb)[mask], f(a, b)[mask]
        k = BLOCKS
        sa, full = blocks(a, mask, k)
        sb, _ = blocks(b, mask, k)
        ta_b, _ = blocks(ta, mask, k)
        tb_b, _ = blocks(tb, mask, k)
        bt, be = f(ta_b, tb_b)[full], f(sa, sb)[full]
        out[key] = {
            "pixel_r": float(np.corrcoef(truth, est)[0, 1]),
            "pixel_rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
            "block_r": float(np.corrcoef(bt, be)[0, 1]),
            "block_rmse": float(np.sqrt(np.mean((be - bt) ** 2))),
        }
        if key == "ndvi":
            veg, found = bt > VEGETATED, be > VEGETATED
            out[key]["vegetated_blocks"] = int(veg.sum())
            out[key]["vegetated_found"] = float((veg & found).sum() / max(veg.sum(), 1))
            out[key]["false_alarms"] = float((found & ~veg).sum() / max((~veg).sum(), 1))
    out["regions_ndvi"] = {
        name: {"true": float(ndvi(t680[m].sum(), t780[m].sum())),
               "recovered": float(ndvi(r680[m].sum(), r780[m].sum()))}
        for name, m in regions.items()}
    return out


def colorize(values, mask, stops):
    """RGBA map, black and transparent off the disk (the eyepiece shader ignores alpha)."""
    xs = [s[0] for s in stops]
    rgb = np.stack([np.interp(values, xs, [s[1][c] for s in stops]) for c in range(3)], axis=-1)
    return np.dstack([rgb * mask[..., None], mask * 255.0]).round().astype(np.uint8)


def save_map(name, values, mask, stops):
    img = colorize(np.nan_to_num(values), mask, stops)
    Image.fromarray(img, mode="RGBA").save(os.path.join(OUT, name))
    return name


LEGENDS = [
    ("Vegetation (NDVI)", NDVI_STOPS,
     [(-0.2, "ocean"), (0.02, "cloud"), (0.12, "desert"), (0.45, "plants")]),
    ("O2 A-band depth", DEPTH_STOPS,
     [(0.4, "high cloud"), (0.62, "longer path through air"), (0.86, "limb")]),
]


def panel(columns, path):
    """Rows of maps with a title over each column and a colorbar per row, for the README."""
    size, pad, head, foot = 256, 8, 40, 46
    rows = len(columns[0]["maps"])
    width = len(columns) * (size + pad) + pad
    sheet = Image.new("RGB", (width, rows * (size + pad) + head + foot))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for i, col in enumerate(columns):
        x = pad + i * (size + pad)
        for j, line in enumerate(col["title"].split("\n")):
            d.text((x, 4 + 17 * j), line, fill=(235, 235, 235), font=font)
        for j, m in enumerate(col["maps"]):
            img = Image.fromarray(m, mode="RGBA").resize((size, size), Image.NEAREST)
            sheet.paste(img, (x, head + j * (size + pad)), img)
    # One colorbar per row, side by side under the maps.
    y = head + rows * (size + pad) + 2
    half = width // 2
    for k, (name, stops, labels) in enumerate(LEGENDS):
        x0, bar_x = pad + k * half, pad + k * half + 150
        bar_w = half - 150 - 4 * pad
        lo, hi = stops[0][0], stops[-1][0]
        strip = colorize(np.tile(np.linspace(lo, hi, bar_w), (12, 1)), np.ones((12, bar_w), bool), stops)
        d.text((x0, y), name, fill=(235, 235, 235), font=font)
        sheet.paste(Image.fromarray(strip[..., :3]), (bar_x, y + 2))
        for v, text in labels:
            d.text((bar_x + (v - lo) / (hi - lo) * bar_w, y + 18), text,
                   fill=(200, 200, 200), font=font, anchor="ma")
    sheet.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    image_d = S.image_diameter(T.PLANET_DIAMETER, T.Z_OBS, T.TARGET_DISTANCE)
    data = {n: load(n) for n in (256, 128)}

    refl, mask, regions = data[256]
    t = {"ndvi": ndvi(refl["Band680nm"], refl["Band780nm"]),
         "o2_depth": depth(refl["Band764nm"], refl["Band780nm"])}
    manifest = {
        "description": "Vegetation (NDVI) and O2 A-band maps through the SGL (layer A).",
        "bands": {b: {"calibration_K": k} for b, k in CAL.items()},
        "truth": {
            "disk_ndvi": float(ndvi(refl["Band680nm"][mask].sum(), refl["Band780nm"][mask].sum())),
            "disk_o2_depth": float(depth(refl["Band764nm"][mask].sum(), refl["Band780nm"][mask].sum())),
            "regions_ndvi": {name: float(ndvi(refl["Band680nm"][m].sum(), refl["Band780nm"][m].sum()))
                             for name, m in regions.items()},
            "region_pixels": {name: int(m.sum()) for name, m in regions.items()},
        },
        "method": ("blur with the aperture-averaged SGL PSF, Gaussian noise, (A + aI) recovery with a "
                   "set by the discrepancy principle; scores compare NDVI / depth from recovered bands "
                   f"with the truth, per pixel and over {BLOCKS} x {BLOCKS} blocks"),
        "textures": {
            "ndvi_truth": save_map("bio_ndvi_truth.png", t["ndvi"], mask, NDVI_STOPS),
            "o2_truth": save_map("bio_o2_truth.png", t["o2_depth"], mask, DEPTH_STOPS),
        },
        "cases": [],
    }
    columns = [{"title": "Planet\n(EPIC, 30 pc)", "maps": [colorize(t["ndvi"], mask, NDVI_STOPS),
                                                          colorize(t["o2_depth"], mask, DEPTH_STOPS)]}]

    for case in CASES:
        n = case["n"]
        refl, mask, regions = data[n]
        pitch = image_d / n
        ratio = case["aperture_m"] / pitch if case["aperture_m"] else 1.0
        op = S.SGLBlur(n, aperture_over_pitch=ratio)
        bands, numbers, recorded = {}, {}, {}
        for b in CAL:
            clean = op.apply(refl[b]) * mask
            # Penalty for this band, then the SNR_C that gives the target plain SNR_R.
            probe = 1e4
            noisy = clean + clean[mask].mean() / probe * rng.standard_normal(clean.shape) * mask
            rec, _ = S.solve_cg(op, noisy, mask, tol=1e-10)
            snr_c = case["snr_r"] / (S.snr(refl[b], rec, mask) / probe)
            r = recover(op, refl[b], mask, clean[mask].mean() / snr_c, rng)
            bands[b], numbers[b], recorded[b] = r["blind"], r["numbers"], r["noisy"]
        scores = score(refl, bands, mask, regions)
        # O2 needs no image: the band shows in the summed light of the whole disk.
        meas = {b: recorded[b][mask].sum() for b in CAL}
        rel = [np.sqrt(mask.sum()) * numbers[b]["sigma"] / meas[b] for b in ("Band764nm", "Band780nm")]
        d_meas = float(depth(meas["Band764nm"], meas["Band780nm"]))
        scores["disk_o2_depth_measured"] = d_meas
        scores["disk_o2_depth_uncertainty"] = float((1 - d_meas) * np.hypot(*rel))
        rec_ndvi = ndvi(bands["Band680nm"], bands["Band780nm"])
        rec_depth = depth(bands["Band764nm"], bands["Band780nm"])
        entry = {**case, "pitch_m": pitch, "aperture_over_pitch": ratio, "bands": numbers, "scores": scores,
                 "textures": {"ndvi": save_map(f"bio_ndvi_{case['name']}.png", rec_ndvi, mask, NDVI_STOPS),
                              "o2": save_map(f"bio_o2_{case['name']}.png", rec_depth, mask, DEPTH_STOPS)}}
        if case["name"] == "snr10":
            rec_ndvi_rec = ndvi(recorded["Band680nm"], recorded["Band780nm"])
            rec_depth_rec = depth(recorded["Band764nm"], recorded["Band780nm"])
            entry["textures"]["ndvi_recorded"] = save_map("bio_ndvi_recorded.png", rec_ndvi_rec, mask, NDVI_STOPS)
            entry["textures"]["o2_recorded"] = save_map("bio_o2_recorded.png", rec_depth_rec, mask, DEPTH_STOPS)
            columns.insert(1, {"title": "Image plane\n(as recorded)",
                               "maps": [colorize(rec_ndvi_rec, mask, NDVI_STOPS),
                                        colorize(rec_depth_rec, mask, DEPTH_STOPS)]})
        manifest["cases"].append(entry)
        title = (f"Recovered, SNR {case['snr_r']}\n(256 px, 5.2 m telescopes)" if n == 256
                 else f"Recovered, SNR {case['snr_r']}\n(128 px, 1 m, T&T Fig. 4)")
        columns.append({"title": title, "maps": [colorize(rec_ndvi, mask, NDVI_STOPS),
                                                 colorize(rec_depth, mask, DEPTH_STOPS)]})
        s, nd = scores, scores["ndvi"]
        print(f"{case['name']:8s} SNR_R blind " + " ".join(f"{numbers[b]['blind_snr_r']:5.1f}" for b in CAL)
              + f" | NDVI r pixel {nd['pixel_r']:.2f} block {nd['block_r']:.2f}, vegetated found"
              f" {nd['vegetated_found']:.0%}, false alarms {nd['false_alarms']:.0%}"
              f" | O2 r block {s['o2_depth']['block_r']:.2f}, disk depth {s['disk_o2_depth_measured']:.4f}"
              f" +- {s['disk_o2_depth_uncertainty']:.1e}")
        print("         regions " + ", ".join(f"{k} {v['true']:.2f} -> {v['recovered']:.2f}"
                                           for k, v in s["regions_ndvi"].items()))

    panel(columns, os.path.join(OUT, "biosignatures_panel.png"))
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    tr = manifest["truth"]
    print(f"truth: disk NDVI {tr['disk_ndvi']:.3f}, disk O2 depth {tr['disk_o2_depth']:.4f}, regions "
          + ", ".join(f"{k} {v:.2f}" for k, v in tr["regions_ndvi"].items()))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
