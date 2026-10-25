"""Layer C prototype: what a telescope in the SGL image plane sees when it looks back at the Sun.

This is the per-pixel math the OpenSpace C++/GLSL telescope view will run, written in numpy
first so it can be checked. For each sky direction theta (angle from the Sun's centre, as seen
by the telescope) the point-mass lens equation gives where that ray came from:

    beta = theta - theta_E^2 * theta / |theta|^2,      theta_E = sqrt(2 r_g / z)

Surface brightness is conserved by lensing, so the pixel just takes the planet's brightness at
beta. A telescope offset by rho in the image plane sees the planet centre shifted by rho/z.

The real planet is ~10^-11 rad across against a ~10^-5 rad ring, so no telescope resolves it.
To show anything the planet (and the telescope's offset with it) is magnified by a factor E;
kappa = E * alpha_planet / theta_E is the drawn planet radius in units of the ring radius.
"""
import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "layer_a"))
import sgl_psf as S  # noqa: E402

Z_OBS = 650 * S.AU
TARGET_DISTANCE = 30 * S.PC
THETA_E = S.einstein_angle(Z_OBS)                     # rad, ~7.79e-6
THETA_SUN = S.R_SUN / Z_OBS                           # rad, ~7.16e-6
ALPHA_PLANET = S.R_EARTH_EQ / TARGET_DISTANCE         # rad, planet's true angular radius


def exaggeration(kappa):
    """Magnification of the planet needed for a drawn radius of kappa * theta_E."""
    return kappa * THETA_E / ALPHA_PLANET


def bilinear(tex, u, v):
    """Sample tex (n x n or n x n x c, row 0 = +v) at normalized coords u, v in [-1, 1].

    Texel centres sit at (i + 0.5) / n, as in OpenGL and in layer A's pixel grid, and
    coordinates outside clamp to the edge texel (GL_CLAMP_TO_EDGE).
    """
    n = tex.shape[0]
    col = (u + 1) / 2 * n - 0.5
    row = (1 - v) / 2 * n - 0.5
    inside = (col >= -0.5) & (col <= n - 0.5) & (row >= -0.5) & (row <= n - 0.5)
    c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
    r0 = np.clip(np.floor(row).astype(int), 0, n - 2)
    fc = np.clip(col - c0, 0, 1)
    fr = np.clip(row - r0, 0, 1)
    if tex.ndim == 3:
        fc, fr, inside = fc[..., None], fr[..., None], inside[..., None]
    out = ((1 - fr) * ((1 - fc) * tex[r0, c0] + fc * tex[r0, c0 + 1])
           + fr * ((1 - fc) * tex[r0 + 1, c0] + fc * tex[r0 + 1, c0 + 1]))
    return np.where(inside, out, 0.0)


def lensed_planet(tex, tx, ty, u, kappa):
    """Planet brightness seen along sky directions (tx, ty), in units of theta_E.

    u: telescope position in the image plane, in units of the planet image's radius (x, y).
    tex is the image-plane texture (x right, y up), as in layer A.
    """
    r2 = tx * tx + ty * ty
    bx = tx - tx / r2                                  # beta / theta_E
    by = ty - ty / r2
    # Planet coordinates q (units of its drawn radius), measured from its apparent centre.
    qx = (bx - kappa * u[0]) / kappa
    qy = (by - kappa * u[1]) / kappa
    # The image plane holds the planet inverted, so sky offset q is image-plane point -q.
    val = bilinear(tex, -qx, -qy)
    disk = (qx * qx + qy * qy) <= 1.0
    return np.where(disk[..., None] if val.ndim == 3 else disk, val, 0.0)


def render(tex, u, kappa, size=768, fov=1.45, supersample=2, occulter=1.02, corona=0.0):
    """Telescope view centred on the Sun, field of view +/- fov * theta_E.

    occulter: coronagraph radius in solar radii (0 = no occulter, the Sun is drawn dimmed).
    corona:   strength of a stylised r^-3 corona glow outside the occulter (0 = none).
    """
    n = size * supersample
    s = (np.arange(n) + 0.5) / n * 2 - 1
    tx, ty = np.meshgrid(s * fov, -s * fov)
    img = lensed_planet(tex, tx, ty, u, kappa)
    r = np.hypot(tx, ty) * THETA_E / THETA_SUN          # radius in solar radii
    if img.ndim == 2:
        img = img[..., None].repeat(3, axis=-1)
    if corona > 0:
        glow = corona * np.clip(r, 1, None) ** -3.0
        img = img + glow[..., None] * np.array([1.0, 0.82, 0.6])
    if occulter > 0:
        img = np.where((r <= occulter)[..., None], 0.0, img)
    else:
        img = np.where((r <= 1.0)[..., None], np.array([1.0, 0.85, 0.55]) * 0.35, img)
    img = img.reshape(size, supersample, size, supersample, 3).mean(axis=(1, 3))
    return img


def diffraction_view(tex_gray, tex_rgb, u, kappa, wavelength=443e-9, aperture=1.0,
                     size=768, fov=1.45, n_az=1440):
    """What a real telescope would record: the true ring is far thinner than the diffraction
    limit, so it is drawn at theta_E with a Gaussian of the Airy core's width, and its brightness
    around the ring follows the light the lensed planet sends into each azimuth.
    """
    sigma = 0.42 * wavelength / aperture / THETA_E     # Airy core as a Gaussian, units of theta_E
    # Light per azimuth: integrate the exaggerated render radially (shape doesn't depend on kappa
    # for small kappa, which is checked in validate.py).
    phi = (np.arange(n_az) + 0.5) / n_az * 2 * np.pi
    rr = np.linspace(1 - 4 * kappa, 1 + 4 * kappa, 800)
    P, R = np.meshgrid(phi, rr)
    col = lensed_planet(tex_rgb, R * np.cos(P), R * np.sin(P), u, kappa)
    prof = (col * R[..., None]).sum(axis=0)              # n_az x 3
    s = (np.arange(size) + 0.5) / size * 2 - 1
    tx, ty = np.meshgrid(s * fov, -s * fov)
    r = np.hypot(tx, ty)
    az = np.mod(np.arctan2(ty, tx), 2 * np.pi)
    idx = np.clip((az / (2 * np.pi) * n_az).astype(int), 0, n_az - 1)
    radial = np.exp(-0.5 * ((r - 1) / sigma) ** 2)
    img = prof[idx] * radial[..., None]
    img /= max(img.max(), 1e-12)
    rs = r * THETA_E / THETA_SUN
    return np.where((rs <= 1.02)[..., None], 0.0, img)
