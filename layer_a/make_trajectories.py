"""Layer A: spacecraft paths across the SGL image plane, as CSVs for OpenSpace.

Three scenarios, all in image-plane metres (origin on the optical axis, x right, y up):

  m0_spiral     one spacecraft probing the thesis M0 acquisition spiral (20 arms x 8 radii)
  raster_single one spacecraft running the Turyshev boustrophedon raster over the disk
  formation12   twelve spacecraft in a rigid vertical line sweeping the disk together

Every hop is a bang-coast-bang transfer: the analytical double-integrator profile the
thesis policies were trained to imitate (thesis section 2.2), at the "balanced" Pareto
setting T = 2.35 T_min. These are reference paths, not outputs of the trained networks.
Each stop also gets the image-plane brightness there, read from out/textures.

    python make_trajectories.py      # run make_textures.py first
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "out", "textures")
OUT = os.path.join(HERE, "out", "trajectories")

THRUST = 6e-5            # m/s^2, the thesis Module 4 value, used for every scenario
PARETO_STRETCH = 2.35    # T = T_min (1.2 + 2.3 r) at r = 0.5, as in thesis Module 5
DWELL = 3000.0           # s per stop: ~1 year of integration spread over a 100 x 100 image
ROW_SPACING = 30.0       # m between raster rows and between stops along a row
N_AGENTS = 12            # thesis Module 4 constellation size
AGENT_SPACING = 120.0    # m between neighbours in the formation line (thesis safety: 50 m)


def bcb_position(p0, p1, t, duration):
    """Bang-coast-bang position at time t of a hop from p0 to p1 lasting `duration`."""
    d = p1 - p0
    length = np.linalg.norm(d)
    if length == 0:
        return p0.copy()
    ta = (duration - np.sqrt(duration**2 - 4 * length / THRUST)) / 2
    if t < ta:
        s = 0.5 * THRUST * t**2
    elif t <= duration - ta:
        s = 0.5 * THRUST * ta**2 + THRUST * ta * (t - ta)
    else:
        s = length - 0.5 * THRUST * (duration - t) ** 2
    return p0 + d * (s / length)


def hop_time(length):
    return PARETO_STRETCH * 2 * np.sqrt(length / THRUST) if length > 0 else 0.0


def schedule(stops, t0=0.0):
    """Hop-then-dwell segments through a list of stops. Returns segments and stop times."""
    segs, stop_times = [], []
    t = t0
    for i, p in enumerate(stops):
        if i > 0:
            T = hop_time(np.linalg.norm(p - stops[i - 1]))
            segs.append(("hop", t, t + T, stops[i - 1], p))
            t += T
        stop_times.append(t)
        segs.append(("dwell", t, t + DWELL, p, p))
        t += DWELL
    return segs, stop_times


def synced_schedule(stop_lists):
    """Formation flying: every agent hops at once and waits for the slowest hop."""
    n_stops = len(stop_lists[0])
    segs = [[] for _ in stop_lists]
    stop_times = []
    t = 0.0
    for i in range(n_stops):
        if i > 0:
            T = max(hop_time(np.linalg.norm(s[i] - s[i - 1])) for s in stop_lists)
            for a, s in enumerate(stop_lists):
                segs[a].append(("hop", t, t + T, s[i - 1], s[i]))
            t += T
        stop_times.append(t)
        for a, s in enumerate(stop_lists):
            segs[a].append(("dwell", t, t + DWELL, s[i], s[i]))
        t += DWELL
    return segs, stop_times


def sample(segs, times):
    """Evaluate a segment list at the given times."""
    out = np.zeros((len(times), 2))
    states = []
    k = 0
    for i, t in enumerate(times):
        while k < len(segs) - 1 and t > segs[k][2]:
            k += 1
        kind, a, b, p0, p1 = segs[k]
        tt = min(max(t - a, 0.0), b - a)
        out[i] = bcb_position(p0, p1, tt, b - a) if kind == "hop" else p0
        states.append(kind)
    return out, states


class Brightness:
    """Bilinear lookup of the noiseless image-plane brightness at (x, y) in metres."""

    def __init__(self, manifest):
        self.img = np.load(os.path.join(TEX, "image_plane.npy")).astype(np.float64)
        self.img /= self.img.max()
        self.pitch = manifest["geometry"]["pitch_m"]
        self.c = (self.img.shape[0] - 1) / 2

    def __call__(self, x, y):
        col = x / self.pitch + self.c
        row = self.c - y / self.pitch
        r0, c0 = int(np.floor(row)), int(np.floor(col))
        fr, fc = row - r0, col - c0
        i = self.img
        return float((1 - fr) * ((1 - fc) * i[r0, c0] + fc * i[r0, c0 + 1])
                     + fr * ((1 - fc) * i[r0 + 1, c0] + fc * i[r0 + 1, c0 + 1]))


def raster_rows(radius, spacing):
    ys = np.arange(radius, -radius - 1e-9, -spacing)
    rows = []
    for i, y in enumerate(ys):
        half = np.sqrt(max(radius**2 - y**2, 0.0))
        xs = np.arange(-np.floor(half / spacing) * spacing, half + 1e-9, spacing)
        rows.append([np.array([x, y]) for x in (xs if i % 2 == 0 else xs[::-1])])
    return [r for r in rows if r]


def write_scenario(name, agent_segs, agent_stops, stop_times, dt, brightness, radius):
    os.makedirs(OUT, exist_ok=True)
    end = max(segs[-1][2] for segs in agent_segs)
    times = np.arange(0.0, end + dt, dt)
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "sc", "x_m", "y_m", "state"])
        paths = []
        for a, segs in enumerate(agent_segs):
            pos, states = sample(segs, times)
            paths.append(pos)
            for t, p, s in zip(times, pos, states):
                w.writerow([f"{t:.0f}", a, f"{p[0]:.3f}", f"{p[1]:.3f}", s])
    n_samples = 0
    with open(os.path.join(OUT, f"{name}_samples.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "sc", "x_m", "y_m", "in_disk", "brightness"])
        for a, stops in enumerate(agent_stops):
            times_a = stop_times[a] if isinstance(stop_times[0], list) else stop_times
            for t, p in zip(times_a, stops):
                inside = bool(np.hypot(*p) <= radius)
                n_samples += inside
                w.writerow([f"{t:.0f}", a, f"{p[0]:.3f}", f"{p[1]:.3f}", int(inside),
                            f"{brightness(*p):.5f}"])
    min_sep = None
    if len(paths) > 1:
        stack = np.stack(paths)                     # agents x times x 2
        diffs = stack[:, None] - stack[None, :]
        dist = np.hypot(diffs[..., 0], diffs[..., 1])
        dist[np.arange(len(paths)), np.arange(len(paths))] = np.inf
        min_sep = float(dist.min())
    return {"csv": f"{name}.csv", "samples_csv": f"{name}_samples.csv", "agents": len(agent_segs),
            "duration_days": end / 86400, "dense_dt_s": dt, "samples_in_disk": int(n_samples),
            "min_separation_m": min_sep}


def main():
    with open(os.path.join(TEX, "manifest.json")) as f:
        tex = json.load(f)
    radius = tex["geometry"]["image_diameter_m"] / 2
    bright = Brightness(tex)
    scenarios = {}

    # M0: thesis acquisition spiral, 20 arms x 8 radii, ring by ring.
    stops = []
    for k in range(8):
        r = (k + 1) / 9 * 0.85 * radius
        for j in range(20):
            th = 2 * np.pi * j / 20 + k * np.pi / 20 * 0.5
            stops.append(np.array([r * np.cos(th), r * np.sin(th)]))
    segs, times = schedule(stops)
    scenarios["m0_spiral"] = write_scenario("m0_spiral", [segs], [stops], times, 300.0, bright, radius)

    # Single-spacecraft raster over the whole disk.
    stops = [p for row in raster_rows(radius, ROW_SPACING) for p in row]
    segs, times = schedule(stops)
    scenarios["raster_single"] = write_scenario("raster_single", [segs], [stops], times, 1800.0, bright, radius)

    # Twelve spacecraft in a vertical line, AGENT_SPACING apart, sweeping left-right-left.
    # Each sweep stops every ROW_SPACING in x; between sweeps the line steps down one row.
    passes = int(round(AGENT_SPACING / ROW_SPACING))
    top = (N_AGENTS - 1) / 2 * AGENT_SPACING
    xs = np.arange(-np.ceil(radius / ROW_SPACING) * ROW_SPACING, radius + 1e-9, ROW_SPACING)
    agent_stops = []
    for a in range(N_AGENTS):
        y0 = top - a * AGENT_SPACING + ROW_SPACING * (passes - 1) / 2
        path = []
        for j in range(passes):
            row_x = xs if j % 2 == 0 else xs[::-1]
            path += [np.array([x, y0 - j * ROW_SPACING]) for x in row_x]
        agent_stops.append(path)
    segs, times = synced_schedule(agent_stops)
    scenarios["formation12"] = write_scenario("formation12", segs, agent_stops, times, 300.0, bright, radius)

    manifest = {
        "description": "Bang-coast-bang reference paths across the SGL image plane (layer A).",
        "frame": "image-plane metres, origin on the optical axis, x right, y up (matches texture columns; rows count down from +y)",
        "image_radius_m": radius,
        "thrust_m_s2": THRUST, "pareto_stretch": PARETO_STRETCH, "dwell_s": DWELL,
        "row_spacing_m": ROW_SPACING, "formation": {"agents": N_AGENTS, "spacing_m": AGENT_SPACING},
        "brightness": "noiseless image-plane value at each stop, from textures/image_plane.npy, max = 1",
        "scenarios": scenarios,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for k, v in scenarios.items():
        sep = f", min separation {v['min_separation_m']:.0f} m" if v["min_separation_m"] else ""
        print(f"{k:14s} {v['agents']:2d} sc, {v['samples_in_disk']:5d} samples in disk, {v['duration_days']:6.1f} days{sep}")


if __name__ == "__main__":
    main()
