# %% [markdown]
# # Module 1 FBPINN under the corrected SGL PSF
#
# This reruns the thesis Module 1 reconstruction with the physical solar gravitational
# lens blur. The aperture-averaged PSF is from Toth & Turyshev 2021, arXiv:2012.05477,
# eq. 3. It's the same model as `portfolio/layer_a/sgl_psf.py`. It fixes four problems
# from REVISION_NOTES:
#
# - **R4:** the physical PSF puts about 0.4% of the light in the directly imaged spot,
#   not 65%.
# - **R3:** the data and the FBPINN use the *same* forward operator, a zero-padded linear
#   convolution, so there's no pixel offset.
# - **R5:** the corona noise is uniform across the image plane.
# - **R1/R2:** there's no oracle fill and no oracle acquisition. Every method sees only
#   the noisy measurements.
#
# It compares five methods on the same measurements:
#
# | Method | What it is |
# |---|---|
# | `plain` | Exact deconvolution with conjugate gradients (100% coverage only) |
# | `tikhonov` | Filtered deconvolution |
# | `grid_tv` | The FBPINN's own loss, optimized directly on a pixel grid |
# | `fbpinn` | The thesis architecture: 5×5 subdomains, 48 Fourier features, [128, 128] heads |
# | `pinn` | The thesis standard-PINN baseline |
#
# **Every regularization strength is picked against the true image.** That's an
# upper bound for each method, and it's the same advantage for all of them, so the
# comparison stays fair.
#
# **To run:** Runtime → Change runtime type → GPU. Upload
# `portfolio/layer_a/out/textures/truth.npy` when asked, then Runtime → Run all.
# Set `QUICK = True` first for a few-minute check. A full run is 72 training runs of
# 2,000 steps; a rough estimate is 15–25 min on an A100, 25–40 on an L4 and about an
# hour on a T4. At the end it downloads `fbpinn_corrected_psf_results.zip`.

# %%
# ---- Settings ----
QUICK = False                  # True: 128x128, short training, one SNR, for a first check
N = 128 if QUICK else 256      # pixels across the planet's image
EPOCHS = 300 if QUICK else 2000
LR_NET = 1e-3                  # thesis value
LR_GRID = 1e-2                 # pixel grids need a larger step
SNR_TARGETS = (10,) if QUICK else (3, 10, 30)      # recovered SNR of *plain* deconvolution
COVERAGES = (1.0, 0.10)        # fraction of disk pixels measured
TV_WEIGHTS = (1e-5, 1e-4, 1e-3, 5e-3)              # thesis used 5e-3
BOUNDARY_W = 0.1               # thesis value
NS1, NF, FFS = 5, 48, 4.0      # thesis FBPINN at 256x256
HIRES = 1024                   # continuous-resolution render of the best FBPINN
SEED = 2026

# %%
import io
import json
import os
import time
import zipfile
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.special import ellipe, ellipk

print("JAX", jax.__version__, "devices:", jax.devices())
if jax.devices()[0].platform == "cpu" and not QUICK:
    print("WARNING: no GPU found. Runtime -> Change runtime type -> GPU, or set QUICK = True.")

# %%
# ---- Ground truth: the EPIC Earth from layer A (256x256, disk filling the frame) ----
TRUTH = os.environ.get("TRUTH_NPY", "truth.npy")
if not os.path.exists(TRUTH):
    from google.colab import files
    files.upload()                       # choose portfolio/layer_a/out/textures/truth.npy
truth_src = np.load(TRUTH).astype(np.float64)
f = truth_src.shape[0] // N
truth = truth_src.reshape(N, f, N, f).mean(axis=(1, 3)) if f > 1 else truth_src
yy, xx = np.mgrid[:N, :N]
mask = np.hypot(xx - (N - 1) / 2, yy - (N - 1) / 2) <= N / 2 - 0.5
truth = np.where(mask, truth, 0.0)
truth /= truth.max()
print(f"truth {truth.shape}, {mask.sum()} disk pixels")

# %%
# ---- Physical SGL blur (Toth & Turyshev 2021 eq. 3), aperture = pixel pitch ----
def mu(q):
    """Aperture-averaged PSF at distance q = rho/d, normalized to mu(0) = 1."""
    q = np.abs(np.asarray(q, dtype=float))
    m = (2 * q) ** 2
    out = np.empty_like(m)
    inside = m <= 1.0
    out[inside] = (2 / np.pi) * ellipe(m[inside])
    mo = m[~inside]
    out[~inside] = (2 / np.pi) * (np.sqrt(mo) * ellipe(1 / mo) - (mo - 1) / np.sqrt(mo) * ellipk(1 / mo))
    return out

off = np.arange(-(N - 1), N)
kernel = mu(np.hypot(*np.meshgrid(off, off)))
padded = np.zeros((2 * N, 2 * N))
padded[: 2 * N - 1, : 2 * N - 1] = kernel
padded = np.roll(padded, (-(N - 1), -(N - 1)), axis=(0, 1))
K_HAT = jnp.asarray(np.fft.rfft2(padded), dtype=jnp.complex64)
MASK = jnp.asarray(mask, dtype=jnp.float32)


@jax.jit
def blur_raw(s):
    p = jnp.zeros((2 * N, 2 * N), jnp.float32).at[:N, :N].set(s)
    return jnp.fft.irfft2(jnp.fft.rfft2(p) * K_HAT, s=(2 * N, 2 * N))[:N, :N]

# Scale so a uniform unit disk reads 1.0 at the center: data stay O(1) as in the thesis.
SCALE = float(blur_raw(MASK)[N // 2, N // 2])


@jax.jit
def blur(s):
    return blur_raw(s) / SCALE

clean = np.asarray(blur(jnp.asarray(truth, jnp.float32)), dtype=np.float64) * mask
c = N // 2
print(f"direct share at center: {truth[c, c] / SCALE / clean[c, c]:.5f}  (thesis PSF: ~0.65)")

# %%
# ---- Linear solvers: plain deconvolution and Tikhonov, by conjugate gradients ----
def cg(apply_a, b, iters=3000, tol=1e-7):
    @jax.jit
    def run(b):
        def body(state):
            x, r, p, rs, k = state
            ap = apply_a(p)
            alpha = rs / jnp.sum(p * ap)
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = jnp.sum(r * r)
            return x, r, r + (rs_new / rs) * p, rs_new, k + 1

        def cond(state):
            return (jnp.sqrt(state[3]) > tol * jnp.sqrt(jnp.sum(b * b))) & (state[4] < iters)

        x0 = jnp.zeros_like(b)
        out = jax.lax.while_loop(cond, body, (x0, b, b, jnp.sum(b * b), 0))
        return out[0], out[4]
    return run(b)


def solve_plain(b):
    return cg(lambda v: blur(v * MASK) * MASK, b * MASK)[0]


def solve_tikhonov(b, sel, lam):
    """min ||P(A s) - b||^2 + lam ||s||^2 over the disk; sel is the measured-pixel mask."""
    rhs = blur(b * sel) * MASK
    return cg(lambda v: blur(blur(v * MASK) * sel) * MASK + lam * v, rhs, tol=1e-6)[0]


def snr_r(rec):
    err = (np.asarray(rec) - truth)[mask]
    return truth[mask].mean() / np.sqrt(np.mean(err**2))


def disk_mse(rec):
    return float(np.mean((np.asarray(rec) - truth)[mask] ** 2))

# Measure this image's deconvolution penalty, then pick the measurement SNR per target.
rng = np.random.default_rng(SEED)
probe = 1e4
noisy = clean + clean[mask].mean() / probe * rng.standard_normal(clean.shape) * mask
PENALTY = snr_r(solve_plain(jnp.asarray(noisy, jnp.float32))) / probe
print(f"penalty SNR_R/SNR_C = {PENALTY:.5f}  (x sqrt(N_disk) = {PENALTY * np.sqrt(mask.sum()):.3f}; paper ~0.83-0.89)")

# %%
# ---- The thesis FBPINN (module1_v7 cell 5), built for an arbitrary set of coordinates ----
def fb_features(xs, ys, centers, half_width, B):
    """Partition-of-unity weights and per-subdomain Fourier features at points (xs, ys)."""
    mx, my = centers
    lx = (xs[None, :] - mx[:, None]) / half_width
    ly = (ys[None, :] - my[:, None]) / half_width
    wx = jnp.where(jnp.abs(lx) <= 1, (1 + jnp.cos(jnp.pi * lx)) ** 2, 0.0)
    wy = jnp.where(jnp.abs(ly) <= 1, (1 + jnp.cos(jnp.pi * ly)) ** 2, 0.0)
    w = wx * wy
    pou = w / (jnp.sum(w, axis=0, keepdims=True) + 1e-8)
    lc = jnp.stack([lx, ly], axis=-1)                       # sub x pts x 2
    proj = lc @ B.T
    feats = jnp.concatenate([jnp.cos(2 * jnp.pi * proj), jnp.sin(2 * jnp.pi * proj), lc], axis=-1)
    return pou, feats

c1d = jnp.linspace(-1, 1, NS1)
CENTERS = [a.flatten() for a in jnp.meshgrid(c1d, c1d)]
HALF_W = float((c1d[1] - c1d[0]) * 1.5 / 2.0)
B_FF = random.normal(random.PRNGKey(0), (NF, 2)) * FFS
coords = jnp.linspace(-1, 1, N).astype(jnp.float32)
GX, GY = jnp.meshgrid(coords, coords)
POU, LFF = fb_features(GX.flatten(), GY.flatten(), CENTERS, HALF_W, B_FF)
FB_LAYERS = [LFF.shape[-1], 128, 128, 1]
PN_LAYERS = [2, 128, 128, 128, 128, 1]
COORDS = jnp.stack([GX.flatten(), GY.flatten()], axis=1)


def init_layers(key, sizes, n_sub=None):
    params = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, wk = random.split(key)
        shape = (m, n) if n_sub is None else (n_sub, m, n)
        bias = (n,) if n_sub is None else (n_sub, n)
        params.append((random.normal(wk, shape) * jnp.sqrt(2.0 / (m + n)), jnp.zeros(bias)))
    return params


def fb_apply(p, pou, lff):
    h = lff
    for w, b in p[:-1]:
        h = jnp.tanh(jnp.einsum("spi,sio->spo", h, w) + b[:, None, :])
    w, b = p[-1]
    return jnp.sum(pou * (jnp.einsum("spi,sio->spo", h, w) + b[:, None, :]).squeeze(-1), axis=0)


def pn_apply(p, crd):
    h = crd
    for w, b in p[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = p[-1]
    return (h @ w + b).squeeze(-1)


MODELS = {
    "fbpinn": (lambda k: init_layers(k, FB_LAYERS, NS1 * NS1), lambda p: fb_apply(p, POU, LFF).reshape(N, N)),
    "pinn": (lambda k: init_layers(k, PN_LAYERS), lambda p: pn_apply(p, COORDS).reshape(N, N)),
    # The same loss on a raw pixel grid: the fair non-neural baseline (REVISION_NOTES R1).
    "grid_tv": (None, lambda p: p),
}

# %%
# ---- Training: thesis loss (data + TV + off-disk penalty, clip to [0, 1.5]), Adam ----
@partial(jax.jit, static_argnames=("model",))
def fit(model, params, b, sel, tv_w, lr):
    """EPOCHS Adam steps in one compiled loop. The data and weights are arguments, so each
    model compiles once and every SNR / coverage / TV setting reuses it."""
    render = MODELS[model][1]
    n_meas = jnp.sum(sel)

    def loss(p):
        s_raw = render(p)
        s = jnp.clip(s_raw, 0, 1.5)
        data = jnp.sum(((blur(s) - b) * sel) ** 2) / n_meas
        tv = jnp.mean(jnp.abs(jnp.diff(s, axis=1))) + jnp.mean(jnp.abs(jnp.diff(s, axis=0)))
        off_disk = jnp.mean(s_raw**2 * (1.0 - MASK))
        return data + tv_w * tv + BOUNDARY_W * off_disk

    def step(t, carry):
        p, m, v, _ = carry
        l, g = jax.value_and_grad(loss)(p)
        k = (t + 1).astype(jnp.float32)
        m = jax.tree_util.tree_map(lambda a, x: 0.9 * a + 0.1 * x, m, g)
        v = jax.tree_util.tree_map(lambda a, x: 0.999 * a + 0.001 * x**2, v, g)
        p = jax.tree_util.tree_map(
            lambda a, mm, vv: a - lr * (mm / (1 - 0.9**k)) / (jnp.sqrt(vv / (1 - 0.999**k)) + 1e-8), p, m, v)
        return p, m, v, l

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return jax.lax.fori_loop(0, EPOCHS, step, (params, zeros, zeros, jnp.float32(0)))


def train(model, b, sel, tv_w, init_value):
    init_fn, render = MODELS[model]
    params = jnp.full((N, N), init_value, jnp.float32) * MASK if init_fn is None else init_fn(random.PRNGKey(42))
    lr = LR_GRID if init_fn is None else LR_NET
    params, _, _, l = fit(model, params, jnp.asarray(b, jnp.float32), jnp.asarray(sel, jnp.float32),
                          jnp.float32(tv_w), jnp.float32(lr))
    return params, np.clip(np.asarray(render(params)), 0, 1.5) * mask, float(l)

# %%
# ---- The sweep ----
results, best_images, best_params = [], {}, {}
t_start = time.time()
for target in SNR_TARGETS:
    snr_c = target / PENALTY
    sigma = clean[mask].mean() / snr_c
    noise = np.random.default_rng(SEED + target).standard_normal(clean.shape)
    meas = (clean + sigma * noise) * mask
    for cov in COVERAGES:
        sel = mask & (np.random.default_rng(SEED + 1).random(mask.shape) < cov) if cov < 1 else mask
        b = meas * sel
        key = f"snr{target:02d}_cov{int(cov * 100):03d}"
        row = {"target_snr_r_plain": target, "snr_c": snr_c, "coverage": cov, "n_measured": int(sel.sum())}
        if cov == 1.0:
            rec = np.asarray(solve_plain(jnp.asarray(b, jnp.float32)))
            row["plain"] = {"snr_r": snr_r(rec), "disk_mse": disk_mse(rec)}
            best_images[f"{key}_plain"] = rec
        best = None
        for lam in np.geomspace(1e-4, 10, 12) * np.sqrt(mask.sum()) / snr_c:
            rec = np.asarray(solve_tikhonov(jnp.asarray(b, jnp.float32), jnp.asarray(sel, jnp.float32), lam))
            if best is None or snr_r(rec) > best[0]:
                best = (snr_r(rec), lam, rec)
        row["tikhonov"] = {"snr_r": best[0], "disk_mse": disk_mse(best[2]), "lambda": float(best[1])}
        best_images[f"{key}_tikhonov"] = best[2]
        init_value = float(b[sel].mean())
        for model in ("grid_tv", "fbpinn", "pinn"):
            best = None
            for tv_w in TV_WEIGHTS:
                t0 = time.time()
                p, rec, final_loss = train(model, b, sel, tv_w, init_value)
                s = snr_r(rec)
                print(f"  {key} {model:8s} tv={tv_w:.0e}: SNR_R {s:6.2f}  MSE {disk_mse(rec):.5f}  ({time.time() - t0:.0f}s)")
                if best is None or s > best[0]:
                    best = (s, tv_w, rec, p)
            row[model] = {"snr_r": best[0], "disk_mse": disk_mse(best[2]), "tv_weight": best[1]}
            best_images[f"{key}_{model}"] = best[2]
            if model == "fbpinn":
                best_params[key] = best[3]
        results.append(row)
        summary = "  ".join(f"{k} {row[k]['snr_r']:.2f}" for k in ("plain", "tikhonov", "grid_tv", "fbpinn", "pinn") if k in row)
        print(f"{key}: SNR_C {snr_c:.0f} | recovered SNR: {summary}  [{(time.time() - t_start) / 60:.1f} min]")

# %%
# ---- Results table ----
methods = ("plain", "tikhonov", "grid_tv", "fbpinn", "pinn")
print(f"{'case':<16}{'SNR_C':>8}" + "".join(f"{m:>10}" for m in methods))
for r in results:
    key = f"snr{r['target_snr_r_plain']:02d}_cov{int(r['coverage'] * 100):03d}"
    print(f"{key:<16}{r['snr_c']:>8.0f}" + "".join(f"{r[m]['snr_r']:>10.2f}" if m in r else f"{'-':>10}" for m in methods))
print("\nCells are recovered-image SNR (mean truth / RMS error over the disk); higher is better.")

# %%
# ---- Continuous-resolution render of the best FBPINN (the thesis's resolution claim) ----
best_key = max(best_params, key=lambda k: next(r["fbpinn"]["snr_r"] for r in results
                                               if f"snr{r['target_snr_r_plain']:02d}_cov{int(r['coverage'] * 100):03d}" == k))
hi_coords = jnp.linspace(-1, 1, HIRES).astype(jnp.float32)
hi_rows = []
for row_y in np.array_split(np.arange(HIRES), max(1, HIRES // 64)):
    gx, gy = jnp.meshgrid(hi_coords, hi_coords[row_y])
    pou, lff = fb_features(gx.flatten(), gy.flatten(), CENTERS, HALF_W, B_FF)
    hi_rows.append(np.asarray(fb_apply(best_params[best_key], pou, lff)).reshape(len(row_y), HIRES))
hires = np.clip(np.concatenate(hi_rows), 0, 1.5)
hy, hx = np.mgrid[:HIRES, :HIRES]
hires *= np.hypot(hx - (HIRES - 1) / 2, hy - (HIRES - 1) / 2) <= HIRES / 2 - 0.5
print(f"rendered {best_key} at {HIRES}x{HIRES}")

# %%
# ---- Save everything and download ----
from PIL import Image


def png_bytes(img):
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()

out = "fbpinn_corrected_psf_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("results.json", json.dumps({
        "settings": {"N": N, "epochs": EPOCHS, "lr_net": LR_NET, "lr_grid": LR_GRID, "tv_weights": TV_WEIGHTS,
                     "boundary_w": BOUNDARY_W, "fbpinn": {"ns1": NS1, "nf": NF, "ffs": FFS}, "quick": QUICK},
        "penalty_snr_r_over_snr_c": PENALTY,
        "note": "Regularization strengths chosen against the truth for every method (upper bounds).",
        "results": results, "hires_from": best_key}, indent=2, default=float))
    z.writestr("truth.png", png_bytes(truth))
    for name, img in best_images.items():
        z.writestr(f"recon_{name}.png", png_bytes(img))
        z.writestr(f"recon_{name}.npy", (lambda b: (np.save(b, img.astype(np.float32)), b.getvalue())[1])(io.BytesIO()))
    z.writestr(f"fbpinn_{best_key}_{HIRES}.png", png_bytes(hires))
print("wrote", out)
try:
    from google.colab import files
    files.download(out)
except ImportError:
    pass
