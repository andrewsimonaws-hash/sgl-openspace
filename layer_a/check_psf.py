"""Check the SGL model against Toth & Turyshev (2021) before using it for textures.

1. mu(rho) matches its limits: mu(0) = 1, mu(d/2) = 2/pi, mu -> d/(4 rho) far out.
2. Plain deconvolution of a fully sampled square image loses SNR by ~0.878/sqrt(N)
   (their simulations; eq. 18 predicts 0.891/sqrt(N)).

    python check_psf.py     # exits 1 if a check fails
"""
import sys

import numpy as np

import sgl_psf as S

print("mu(0) =", float(S.mu(0.0)), "(expect 1)")
print("mu(0.5) =", float(S.mu(0.5)), f"(expect 2/pi = {2 / np.pi:.6f})")
for q in (2.0, 10.0, 100.0):
    print(f"mu({q}) = {float(S.mu(q)):.6f}  d/4rho = {1 / (4 * q):.6f}")

print("\nDeconvolution penalty, fully sampled square image, plain CG inversion:")
rng = np.random.default_rng(0)
penalties = []
for n in (16, 32, 64):
    op = S.SGLBlur(n, aperture_over_pitch=1.0)
    mask = np.ones((n, n), bool)
    truth = rng.uniform(0.2, 1.0, (n, n))
    clean = op.apply(truth)
    ratios = []
    for snr_c in (1e3, 1e4):
        sigma = clean.mean() / snr_c
        noisy = clean + sigma * rng.standard_normal(clean.shape)
        rec, iters = S.solve_cg(op, noisy, mask, tol=1e-10, max_iter=5000)
        ratios.append(S.snr(truth, rec, mask) / snr_c)
    r = np.mean(ratios)
    # The paper's 0.891 includes a pi/4 factor for the light a circular aperture collects
    # compared with a square pixel; without it their eq. 17 gives 1/ln(1+sqrt 2) = 1.135.
    print(f"  N = {n}^2: SNR_R/SNR_C x sqrt(N) = {r * n:.3f} (matrix only; paper's estimate 1.135)"
          f" -> x pi/4 = {r * n * np.pi / 4:.3f} (paper: 0.891 analytic, 0.878 simulated)  [{iters} CG its]")
    penalties.append(r * n * np.pi / 4)

checks = {
    "mu(0) = 1": abs(S.mu(0.0) - 1) < 1e-9,
    "mu(d/2) = 2/pi": abs(S.mu(0.5) - 2 / np.pi) < 1e-6,
    "mu -> d/(4 rho) within 0.1% from 10 d out": all(abs(S.mu(q) * 4 * q - 1) < 1e-3 for q in (10.0, 100.0)),
    "penalty scales as 1/sqrt(N) (flat within 3%)": np.ptp(penalties) / np.mean(penalties) < 0.03,
    "penalty within 10% of the paper's simulated 0.878": abs(np.mean(penalties) / 0.878 - 1) < 0.10,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    sys.exit("FAILED: " + "; ".join(failed))
print("\nall checks passed")
