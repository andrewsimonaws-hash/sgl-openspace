"""Coronal loops for the opening's close pass: magnetic field lines traced from sources under
the Sun's surface, so they share footpoints, bend around each other and bundle near the Sun.

The sources are a stylized solar-minimum pattern, not a prediction: bipoles in belts around the
equator and a weak network elsewhere, placed from a fixed seed so every build is identical. The
field is a sum of point sources, a simple stand-in for a full potential-field model.

Output is float32: the solar pole (x, y, z, 0) in SGL_Frame, then pairs of vertices for GL_LINES,
each (x, y, z, weight), with positions in solar radii. module_sgl/rendering/renderablesglcorona.cpp
draws it.
"""
import math

import numpy as np

# The Sun's rotation axis (IAU: RA 286.13, Dec 63.87 degrees, J2000) and J2000 to galactic
POLE_RA, POLE_DEC = math.radians(286.13), math.radians(63.87)
J2000_TO_GALACTIC = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                              [0.4941094279, -0.4448296300, 0.7469822445],
                              [-0.8676661490, -0.1980763734, 0.4559837762]])
SOURCE_DEPTH = 0.97     # solar radii: just under the surface
MAX_HEIGHT = 1.75       # closed loops stop about here; taller ones are dropped
STEP = 0.005            # solar radii per tracing step
KEEP_EVERY = 3          # of the traced points


def solar_pole(galactic_to_frame):
    """The solar north pole as a unit vector in SGL_Frame."""
    equatorial = np.array([math.cos(POLE_DEC) * math.cos(POLE_RA),
                           math.cos(POLE_DEC) * math.sin(POLE_RA), math.sin(POLE_DEC)])
    return galactic_to_frame @ (J2000_TO_GALACTIC @ equatorial)


def sources(pole, seed=11):
    rng = np.random.default_rng(seed)
    a = np.cross(pole, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(pole, a)

    def on_sphere(lat, lon):
        return math.cos(lat) * (math.cos(lon) * a + math.sin(lon) * b) + math.sin(lat) * pole

    xs, qs = [], []
    for _ in range(26):                             # active-region bipoles in the belts
        lat, lon = rng.normal(0, math.radians(24)), rng.uniform(0, 2 * math.pi)
        center = on_sphere(lat, lon)
        east = np.cross(pole, center)
        east /= np.linalg.norm(east)
        north = np.cross(center, east)
        tilt = rng.normal(0, math.radians(25))
        separation = rng.uniform(0.06, 0.2)
        d = math.cos(tilt) * east + math.sin(tilt) * north
        q = rng.lognormal(0, 0.6) * separation ** 2 * 40
        for sign in (1, -1):
            x = center + sign * d * separation
            xs.append(x / np.linalg.norm(x) * SOURCE_DEPTH)
            qs.append(sign * q * rng.uniform(0.75, 1.25))
    for _ in range(50):                             # the weak network field everywhere
        lat, lon = math.asin(rng.uniform(-1, 1)), rng.uniform(0, 2 * math.pi)
        xs.append(on_sphere(lat, lon) * SOURCE_DEPTH)
        qs.append(rng.choice([-1, 1]) * rng.uniform(0.05, 0.2))
    return np.array(xs), np.array(qs)


def trace(xs, qs, seed=3, max_steps=1400):
    """Trace every line at once (midpoint method), from footpoints around each positive source.
    Returns the closed ones, as (points, weight)."""
    rng = np.random.default_rng(seed)
    starts, weights = [], []
    for x, q in zip(xs, qs):
        if q <= 0:
            continue
        radial = x / np.linalg.norm(x)
        t1 = np.cross(radial, [0.3, 0.5, 0.8])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(radial, t1)
        for _ in range(int(np.clip(round(q * 40), 3, 26))):
            angle, spread = rng.uniform(0, 2 * math.pi), 0.012 + 0.03 * rng.random()
            p = radial + spread * (math.cos(angle) * t1 + math.sin(angle) * t2)
            starts.append(p / np.linalg.norm(p) * 1.001)
            weights.append(rng.uniform(0.45, 1.0))
    pos = np.array(starts)

    def direction(p):
        d = p[:, None] - xs[None]
        field = np.sum(qs[None, :, None] * d / np.sum(d * d, axis=-1, keepdims=True) ** 1.5, axis=1)
        return field / np.linalg.norm(field, axis=1, keepdims=True)

    alive = np.ones(len(pos), bool)
    closed = np.zeros(len(pos), bool)
    tracks = [[p.copy()] for p in pos]
    for _ in range(max_steps):
        i = np.nonzero(alive)[0]
        if len(i) == 0:
            break
        midpoint = pos[i] + 0.5 * STEP * direction(pos[i])
        pos[i] = pos[i] + STEP * direction(midpoint)
        r = np.linalg.norm(pos[i], axis=1)
        for j, p in zip(i, pos[i]):
            tracks[j].append(p.copy())
        closed[i[r < 1.0]] = True
        alive[i[(r < 1.0) | (r > MAX_HEIGHT)]] = False
    return [(np.array(t), w) for t, w, c in zip(tracks, weights, closed) if c and len(t) > 6]


def write_loops(path, galactic_to_frame):
    pole = solar_pole(galactic_to_frame)
    xs, qs = sources(pole)
    lines = trace(xs, qs)
    vertices = [*pole, 0.0]
    for points, weight in lines:
        points = np.concatenate([points[:-1:KEEP_EVERY], points[-1:]])
        for p0, p1 in zip(points[:-1], points[1:]):
            vertices += [*p0, weight, *p1, weight]
    np.array(vertices, dtype=np.float32).tofile(path)
    tops = [np.linalg.norm(p, axis=1).max() for p, _ in lines]
    print(f"corona loops: {len(lines)} closed field lines from {len(xs)} sources, "
          f"{(len(vertices) - 4) // 8} segments, median top {np.median(tops):.2f} R")
