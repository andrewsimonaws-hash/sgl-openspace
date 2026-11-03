# Layer A: SGL image-plane data

Offline Python that produces the data the OpenSpace scene displays. It covers what the solar gravitational lens does to an exoplanet's image, what a telescope sweeping the image plane records, and the paths the spacecraft fly to record it.

This layer replaces the thesis PSF with the physical one. The thesis put 65% of the light at the center of a 256² image into the directly imaged pixel. The physical share is **0.4%**, 161 times less. That gap is what makes SGL imaging hard, and the visualization is built to show it.

## Run it

The outputs in `out/` are committed, so this is only needed to regenerate them. `make_textures.py` and `make_biosignatures.py` read one input, the DSCOVR EPIC Level 1B granule [`epic_1b_20150613110250_03.h5`](https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/DSCOVR/DSCOVR_EPIC_L1B_3/2015.06/epic_1b_20150613110250_03.h5) (version 3, 426 MB). Download it with a free NASA Earthdata login into `layer_a/data/`, which git ignores, or set `EPIC_H5` to its path.

```bash
python -m venv ../.venv
../.venv/Scripts/python -m pip install -r requirements.txt
../.venv/Scripts/python check_psf.py          # physics checks against the paper
../.venv/Scripts/python make_textures.py      # ~5 min, writes out/textures
../.venv/Scripts/python make_trajectories.py  # seconds, writes out/trajectories
../.venv/Scripts/python make_biosignatures.py # ~1 min, writes out/biosignatures
../.venv/Scripts/python preview.py            # out/preview.png, paths over the image plane
```

The ground truth is the DSCOVR EPIC Earth from 13 June 2015, 11:00 UTC, from that granule.
- **Every picture is in color** (`*_rgb.png`): the 680, 551 and 443 nm bands run through the pipeline separately, then stack.
  - The averaged PSF doesn't depend on wavelength, so each band blurs identically. Their penalties come out 1.00, 1.02 and 1.03 × 1/√N.
  - Colors use the thesis figure's display: L1B counts normalized by their common maximum, gain 2.2. They aren't radiometrically calibrated.
  - EPIC takes its bands minutes apart, so cloud edges can show slight color fringes.
- **The 443 nm band alone**, the one the thesis used, still runs for the numbers in the manifest and for `truth.npy` and `image_plane.npy`, which the Colab rerun and layer C's validation read. It is the granule's `Band443nm`, byte for byte the array the thesis saved.

## Physics

`sgl_psf.py` implements Toth & Turyshev (2021), [arXiv:2012.05477](https://arxiv.org/abs/2012.05477):

- The aperture-averaged PSF μ(ρ) comes from eq. 3, an elliptic integral. It is rewritten with the reciprocal-parameter transformation so that SciPy's complete integrals evaluate it. μ(0) = 1, and μ falls to d/(4ρ) far out.
- The blur is a zero-padded FFT convolution. Restricted to the planet's disk it is symmetric, so plain conjugate gradients invert it exactly when there is no noise. Recovery RMSE is 1.5×10⁻⁹ after 74 iterations.

`check_psf.py` verifies the two results the visualization relies on:

| Check | This code | Paper |
|---|---|---|
| μ(0), μ(d/2), μ(100d) | 1, 2/π, 0.002500 | 1, 2/π, d/4ρ = 0.0025 |
| Deconvolution penalty × √N (square image, fully sampled) | 1.05, constant for N = 16², 32², 64² | 1.135 matrix-only estimate (eq. 17) |
| Same, × π/4 for a circular aperture | 0.83 | 0.891 analytic, 0.878 ± 0.003 simulated |

The 1/√N scaling matches exactly. The constant is 6% below the paper's simulation, which is within the content dependence the paper notes for its own estimate.

## Geometry (from `out/textures/manifest.json`)

| Quantity | Value |
|---|---|
| Observer distance | 650 AU (focal line starts at 547.7 AU) |
| Target | Earth-size planet (EPIC Earth) at 30 pc |
| Einstein ring | 1.61″ radius, 1.09 solar radii |
| Image of the planet | 1,340 m across |
| Grid | 256 px across the image, 5.23 m pitch, aperture = pitch (fully sampled) |

Real missions would use a 1–2 m telescope and undersample the image. The penalty then scales as D/(N d) (paper eq. 18). The code keeps aperture = pitch for clarity.

## Textures (`out/textures/`)

Each PNG's black and white levels are recorded in the manifest, so the stretch can be undone.

| File | What it shows |
|---|---|
| `truth_rgb.png`, `truth_rgba_rgb.png` | The planet itself, 256², the second with alpha outside the disk. The telescope view in layer C uses the first. |
| `image_plane_stretched_rgb.png` | Brightness across the image plane, 1.5 planet diameters wide, contrast stretched over the disk so the faint structure shows. It's a soft glow. |
| `image_plane_thesis_psf_rgb.png` | What the thesis PSF predicted, which looks almost sharp. Use it for the before/after shot. |
| `measurements_snrXX_rgb.png` | Noisy measurements inside the disk |
| `recovered_plain_snrXX_rgb.png` | Plain deconvolution |
| `recovered_filtered_snrXX_rgb.png` | Tikhonov-filtered deconvolution. The strength is tuned against the truth, so it's an illustration, not a blind method. |
| `recovered_noiseless_rgb.png` | Exact recovery without noise |
| `truth.npy`, `image_plane.npy` | float32, 443 nm |
| `truth_rgb.npy`, `image_plane_rgb.npy` | float32, the three bands |

The measurement SNR each recovered image needs at 443 nm, 256² (measured penalty 0.0046):

| Recovered SNR | Measurement SNR needed | Plain | Filtered |
|---|---|---|---|
| 1 | 218 | 1.0 | 4.0 |
| 3 | 653 | 3.0 | 6.0 |
| 10 | 2,177 | 10.0 | 11.8 |
| 30 | 6,530 | 29.8 | 30.6 |

## Trajectories (`out/trajectories/`)

All positions are in image-plane metres, with the origin on the optical axis, x right and y up. Every hop is a bang-coast-bang transfer, the analytical profile the thesis policies were trained to imitate, at the thesis's "balanced" setting (T = 2.35 T_min). The thrust is 6×10⁻⁵ m/s² (the thesis's thrust), and each stop dwells 3,000 s. These are **reference paths, not outputs of the trained networks.**

| Scenario | Spacecraft | Samples in disk | Duration | Notes |
|---|---|---|---|---|
| `m0_spiral` | 1 | 160 | 15.8 days | The thesis's acquisition spiral, 20 arms × 8 radii |
| `raster_single` | 1 | 1,569 | 115.1 days | Boustrophedon raster, 30 m spacing |
| `formation12` | 12 | 1,568 | 13.4 days | Rigid vertical line, 120 m apart, sweeping together. Minimum separation is 120 m (thesis safety distance: 50 m). |

Each scenario has two CSVs:
- `<name>.csv` is the dense path: `t_s, sc, x_m, y_m, state`, where state is hop or dwell.
- `<name>_samples.csv` has one row per stop: `t_s, sc, x_m, y_m, in_disk, brightness, r, g, b`. The color is the noiseless image-plane value there in the three bands, read from `image_plane_rgb.npy` with the stretch of `image_plane_stretched_rgb.png`, so each sample matches the glow under it. Brightness is the mean of r, g and b.

## Biosignatures (`out/biosignatures/`)

`make_biosignatures.py` asks whether the lens would show that the planet is alive. It uses three calibrated EPIC bands (reflectance = K × counts/s, with the factors from [EPIC's calibration pages](https://epic.gsfc.nasa.gov/science/calibration)):
- **Vegetation:** NDVI = (R780 − R680)/(R780 + R680)
- **Oxygen:** A-band depth = 1 − R764/R780

Each band goes through the same blur and gets Gaussian noise. It's recovered with the (A + aI) filter, where a comes from the discrepancy principle: the strongest filter whose residual stays within the noise. That uses the noise level but not the true planet; the truth-tuned optimum is in the manifest as an upper bound. Scores compare NDVI and depth from the recovered bands with the same quantities from the true bands, per pixel and over 32 × 32 blocks across the disk (about 400 km each). The 732 blocks fully inside the disk count; 43 of them have NDVI above 0.3.

| Case | Grid, aperture | Blind SNR per band | NDVI r, pixel / block | Vegetated blocks found | A-band depth r, pixel / block |
|---|---|---|---|---|---|
| `snr30` | 256², 5.23 m | 30.4–30.5 | 0.93 / 1.00 | 43 | 0.88 / 1.00 |
| `snr10` | 256², 5.23 m | 10.7–10.9 | 0.68 / 1.00 | 43 | 0.62 / 1.00 |
| `snr03` | 256², 5.23 m | 3.8–3.9 | 0.38 / 0.99 | 42 | 0.34 / 0.99 |
| `tt_fig4` | 128², 1 m | 4.6–4.7 | 0.43 / 0.92 | 36 | 0.40 / 0.91 |

- **Noise levels:** the 256² cases set the noise so plain deconvolution reaches the SNR in the name, as in `make_textures.py`. `tt_fig4` is Toth & Turyshev's Fig. 4 regime, with the noise set so plain deconvolution reaches the 4.1 their simulation did. Their SNR_C of 50 would give 5.4 here, because they count the light of a circular aperture (a factor π/4) and all 128² pixels rather than the disk.
- **Regions:** NDVI from summed reflectance over the Congo basin, the Sahara and Europe (from EPIC's geolocation) is 0.21, 0.075 and 0.24. Every case recovers those within 0.012. The whole disk's NDVI is 0.082.
- **Oxygen from the summed light:** 1 − ΣR764/ΣR780 over the recorded disk is 0.630 against 0.633 true. Noise adds 3×10⁻⁷ to 1.2×10⁻⁴. The offset comes from the blur, which keeps more of the center's light inside the disk than the limb's.
- **Band timing:** the 680 and 780 nm images were taken 54 s apart, so the Earth turns about half a 256-grid pixel between them. NDVI picks up a thin rim along sharp cloud edges from that.

| File | What it shows |
|---|---|
| `bio_ndvi_truth.png`, `bio_o2_truth.png` | NDVI and A-band depth of the planet, 256², transparent and black outside the disk |
| `bio_ndvi_recorded.png`, `bio_o2_recorded.png` | The same, computed from the image-plane measurements before recovery (the `snr10` case) |
| `bio_ndvi_<case>.png`, `bio_o2_<case>.png` | From the recovered bands |
| `biosignatures_panel.png` | All of the above side by side, with color scales |
| `manifest.json` | Every number above, per band and per case |

## Not modeled yet

- **Image-plane drift (0.1 m/s) and corona noise.** Neither is in these paths.

The reconstruction rerun under the physical PSF is in [`colab/`](../colab).
