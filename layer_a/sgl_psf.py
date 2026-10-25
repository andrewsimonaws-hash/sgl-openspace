"""Solar gravitational lens (SGL) image-plane model for the portfolio visualization.

Follows Toth & Turyshev (2021), "Image recovery with the solar gravitational lens",
arXiv:2012.05477. A telescope of aperture d sits in the image plane and records one
brightness value per position: the light of the Einstein ring it sees. Every point of
the planet contributes to that value, weighted by the aperture-averaged PSF mu(rho)
(their eq. 2-3), which is 1 for the directly imaged spot and falls off as d/(4 rho).

This replaces the thesis PSF (REVISION_NOTES R4), which put 34-67% of the light in the
directly imaged pixel; the physical share is a fraction of a percent.
"""
import numpy as np
from scipy.special import ellipe, ellipk

G = 6.67430e-11
M_SUN = 1.98847e30
C_LIGHT = 299792458.0
R_SUN = 6.957e8
AU = 1.495978707e11
PC = 3.0856775814913673e16
R_EARTH_EQ = 6.3781e6

R_G = 2 * G * M_SUN / C_LIGHT**2          # Schwarzschild radius of the Sun, ~2953 m


def focal_distance():
    """Heliocentric distance where grazing rays first meet the optical axis, R_sun^2 / 2r_g."""
    return R_SUN**2 / (2 * R_G)


def einstein_angle(z):
    """Angular radius of the Einstein ring seen from heliocentric distance z (rad)."""
    return np.sqrt(2 * R_G / z)


def image_diameter(planet_diameter, z, target_distance):
    """Diameter of the planet's image in the image plane at heliocentric distance z."""
    return planet_diameter * z / target_distance


def mu(rho_over_d):
    """Aperture-averaged SGL PSF mu(rho), normalized to mu(0) = 1 (Toth & Turyshev eq. 3).

    For rho <= d/2 this is (2/pi) E(m) with m = (2 rho/d)^2. For rho > d/2 the incomplete
    integral E(arcsin(d/2rho), m) is rewritten with the reciprocal-parameter transformation,
    which gives complete integrals of parameter 1/m. Large rho tends to d/(4 rho).
    """
    q = np.abs(np.asarray(rho_over_d, dtype=float))
    m = (2 * q) ** 2
    out = np.empty_like(m)
    inside = m <= 1.0
    out[inside] = (2 / np.pi) * ellipe(m[inside])
    mo = m[~inside]
    mi = 1.0 / mo
    out[~inside] = (2 / np.pi) * (np.sqrt(mo) * ellipe(mi) - (mo - 1) / np.sqrt(mo) * ellipk(mi))
    return out


class SGLBlur:
    """Linear SGL blur on an n x n grid of image-plane positions with pitch p.

    `aperture_over_pitch` is d/p: 1.0 means the telescope aperture equals the sample
    spacing (a fully sampled image plane, D = sqrt(N) d in the paper's notation).
    `apply` is a zero-padded (non-circular) convolution, so the operator is exact for a
    finite planet; restricted to a mask it is symmetric, which lets us solve it with CG.
    """

    def __init__(self, n, aperture_over_pitch=1.0, kernel_fn=None):
        self.n = n
        self.size = 2 * n
        off = np.arange(-(n - 1), n)
        dx, dy = np.meshgrid(off, off)
        dist = np.hypot(dx, dy)
        # kernel_fn maps distance in pixels to weight; the default is the physical SGL PSF.
        kernel = kernel_fn(dist) if kernel_fn else mu(dist / aperture_over_pitch)
        padded = np.zeros((self.size, self.size))
        padded[: 2 * n - 1, : 2 * n - 1] = kernel
        # Shift so the kernel's center (offset 0) sits at index (0, 0) for FFT convolution.
        padded = np.roll(padded, (-(n - 1), -(n - 1)), axis=(0, 1))
        self.kernel_hat = np.fft.rfft2(padded)
        self.kernel_sum = kernel.sum()

    def apply(self, img):
        padded = np.zeros((self.size, self.size))
        padded[: self.n, : self.n] = img
        out = np.fft.irfft2(np.fft.rfft2(padded) * self.kernel_hat, s=padded.shape)
        return out[: self.n, : self.n]


def solve_cg(op, b, mask, reg=0.0, tol=1e-8, max_iter=2000):
    """Solve (M A M + reg I) s = b on the masked pixels with conjugate gradients.

    reg = 0 is the plain deconvolution the paper analyzes; reg > 0 is Tikhonov filtering.
    Returns the solution (zero off the mask) and the number of iterations used.
    """
    def a_m(v):
        return op.apply(v * mask) * mask + reg * v

    b = b * mask
    x = np.zeros_like(b)
    r = b - a_m(x)
    p = r.copy()
    rs = np.sum(r * r)
    b_norm = np.sqrt(np.sum(b * b))
    for it in range(1, max_iter + 1):
        ap = a_m(p)
        alpha = rs / np.sum(p * ap)
        x += alpha * p
        r -= alpha * ap
        rs_new = np.sum(r * r)
        if np.sqrt(rs_new) < tol * b_norm:
            return x, it
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, max_iter


def snr(signal, estimate, mask):
    """Mean signal over the mask divided by the RMS error (the paper's SNR definition)."""
    err = (estimate - signal)[mask]
    return signal[mask].mean() / np.sqrt(np.mean(err**2))


def thesis_kernel(pitch, z, wavelength):
    """The thesis PSF as a kernel_fn: 1.0 at the center, 1/(pi k r sqrt(2 r_g/z)) elsewhere.

    It skips integrating the J0^2 core over the pixel, which over-weights the center ~190x.
    """
    k = 2 * np.pi / wavelength
    b = k * np.sqrt(2 * R_G / z)

    def kernel(dist_pixels):
        r = dist_pixels * pitch
        with np.errstate(divide="ignore"):
            return np.where(dist_pixels < 0.5, 1.0, 1.0 / (np.pi * b * np.maximum(r, 1e-9)))

    return kernel
