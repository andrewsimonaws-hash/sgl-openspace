# Layer A: SGL image-plane data

Offline Python that produces the data the OpenSpace scene displays. It covers what the solar gravitational lens does to an exoplanet's image, what a telescope sweeping the image plane records, and the paths the spacecraft fly to record it.

This layer replaces the thesis PSF with the physical one. The thesis put 65% of the light at the center of a 256² image into the directly imaged pixel. The physical share is **0.4%**, 161 times less. That gap is what makes SGL imaging hard, and the visualization is built to show it.

## Run it

```bash
python -m venv ../.venv
../.venv/Scripts/python -m pip install -r requirements.txt
../.venv/Scripts/python check_psf.py          # physics checks against the paper
../.venv/Scripts/python make_textures.py      # ~1 min, writes out/textures
../.venv/Scripts/python make_trajectories.py  # seconds, writes out/trajectories
../.venv/Scripts/python preview.py            # out/preview.png, paths over the image plane
```

The ground truth is the DSCOVR EPIC Earth from 13 June 2015, 11:00 UTC (`code/earth access/epic_1b_20150613110250_03.h5`).
- **Gray outputs** use the 443 nm band alone, as the thesis did. `code/revised/data/epic_443nm_image.npy` is that band, byte for byte.
- **Color outputs** (`*_rgb.png`) run the 680, 551 and 443 nm bands through the same pipeline separately, then stack them.
  - The averaged PSF doesn't depend on wavelength, so each band blurs identically. Their penalties come out 1.00, 1.02 and 1.03 × 1/√N.
  - Colors use the thesis figure's display: L1B counts normalized by their common maximum, gain 2.2. They aren't radiometrically calibrated.
  - EPIC takes its bands minutes apart, so cloud edges can show slight color fringes.

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
| `truth.png`, `truth_rgba.png` | The planet itself, 256², with alpha outside the disk (for the telescope view in layer C) |
| `image_plane.png` | Brightness across the image plane, 1.5 planet diameters wide, on a 0 to max scale. It's a soft glow. |
| `image_plane_stretched.png` | The same, contrast stretched over the disk so the faint structure shows |
| `image_plane_thesis_psf.png` | What the thesis PSF predicted, which looks almost sharp. Use it for the before/after shot. |
| `measurements_snrXX.png` | Noisy measurements inside the disk |
| `recovered_plain_snrXX.png` | Plain deconvolution |
| `recovered_filtered_snrXX.png` | Tikhonov-filtered deconvolution. The strength is tuned against the truth, so it's an illustration, not a blind method. |
| `recovered_noiseless.png` | Exact recovery without noise |
| `truth.npy`, `image_plane.npy` | float32 versions |

The measurement SNR each recovered image needs, at 256² (measured penalty 0.0046):

| Recovered SNR | Measurement SNR needed | Plain | Filtered |
|---|---|---|---|
| 1 | 218 | 1.0 | 4.0 |
| 3 | 653 | 3.0 | 6.0 |
| 10 | 2,177 | 10.0 | 11.8 |
| 30 | 6,530 | 29.8 | 30.6 |

## Trajectories (`out/trajectories/`)

All positions are in image-plane metres, with the origin on the optical axis, x right and y up. Every hop is a bang-coast-bang transfer, the analytical profile the thesis policies were trained to imitate, at the thesis's "balanced" setting (T = 2.35 T_min). The thrust is 6×10⁻⁵ m/s² (the thesis Module 4 value), and each stop dwells 3,000 s. These are **reference paths, not outputs of the trained networks.**

| Scenario | Spacecraft | Samples in disk | Duration | Notes |
|---|---|---|---|---|
| `m0_spiral` | 1 | 160 | 15.8 days | Thesis M0 acquisition spiral, 20 arms × 8 radii |
| `raster_single` | 1 | 1,569 | 115.1 days | Boustrophedon raster, 30 m spacing |
| `formation12` | 12 | 1,568 | 13.4 days | Rigid vertical line, 120 m apart, sweeping together. Minimum separation is 120 m (thesis safety distance: 50 m). |

Each scenario has two CSVs:
- `<name>.csv` is the dense path: `t_s, sc, x_m, y_m, state`, where state is hop or dwell.
- `<name>_samples.csv` has one row per stop: `t_s, sc, x_m, y_m, in_disk, brightness`. Brightness is the noiseless image-plane value there, read from `image_plane.npy`.

## Not modeled yet

- **Image-plane drift (0.1 m/s) and corona noise.** Neither is in these paths.

The FBPINN rerun under the corrected PSF is in [`colab/`](../colab).
