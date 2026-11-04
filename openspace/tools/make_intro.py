"""Write the reel's opening as a session recording, frame by frame.

Overview of the planet 30 pc away and the Sun, a flight out to the planet, then its light:
across 30 pc to the Sun, past the limb where it bends by 1.6 arcseconds, and on to the
image plane 650 AU out, ending exactly on the first frame of the flat reel.

Each frame gets its own camera keyframe and scripts (simulation time, the light pulse's
position and size, captions), so the result doesn't depend on how fast OpenSpace runs live.
The camera is anchored to whatever it follows (planet, light pulse, image plane), which
keeps lines precise across 30 pc. Travel along the light uses a logarithmic time-lapse:
distances shrink by a steady factor per second, like a powers-of-ten zoom.

    python make_intro.py                    # writes user/recordings/sgl_intro_flat.osrectxt
    python export_frames.py <recording> 30  # then encode_clip.py joins it to the reel
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import intro as I                                    # noqa: E402
import paths                                         # noqa: E402
from record_reel import FRAMING, LABEL, P, SUN_LABEL, label_text  # noqa: E402

TEXTURES_MANIFEST = os.path.join(HERE, "..", "..", "layer_a", "out", "textures", "manifest.json")
FPS = 30
BEATS = {"overview": 6.5, "fly": 9.0, "travel": 11.0, "bend": 7.5, "converge": 7.0, "arrive": 4.0}
UP = np.array([0.0, 1.0, 0.0])
U_SLOW = 3.75e8         # m either side of the limb covered at a steady speed
V_SLOW = 2 * U_SLOW / BEATS["bend"]     # m per clip second there: a third of the speed of light
FLY_NEAR = 5.0          # s of the flight spent closing from NEAR_DIST, as the planet grows from a dot
NEAR_DIST = 1e10        # m from the planet
DOT_FADE = (1.5, 5.0)   # planet radius in pixels over which its marker fades out
GLARE = 0.65            # the Sun glare's own opacity (glare.asset)
# Faded while the glare is still small, so it never balloons into a halo before it goes
GLARE_FADE = (1e12, 2e11)       # m from the Sun: full glare beyond the first, none inside the second
CORONA_FADE = (8e11, 1.5e11)    # the close-pass glow: none beyond the first, full inside the second
OVERVIEW_FADE = 15              # frames over which the overview's diagram and labels fade out
# OpenSpace's SunStar (the Sun as a star from outside the solar system) swells into a halo
# and only fades between 0.1 and 0.0075 pc, a flash at this speed; fade it while it's small
SUNSTAR_FADE = (1e17, 1e16)
D_PLANET, D_SUN, D_IMAGE = I.D_PLANET, 2.5e9, 3.2e3  # closest camera distances on each leg
# How the opening sits on screen. Screen-space positions are in the camera's view space, where
# z = -2 is the flat screen. On the dome master the circle's center is straight ahead and a
# flat-sized overview would fill only its middle, so the overview is drawn `scale` times
# larger, and the clock and a two-line caption sit above and below it, turned to the camera.
LAYOUT = {
    "flat": dict(scale=1.0, overview_dist=40 * I.PC,      # camera to the planet-Sun midpoint
                 px_per_rad=I.OUTPUT[0] / I.SCREEN[0]),   # clip pixels per unit of tan(angle)
    "dome": dict(scale=1.6, overview_dist=27 * I.PC, px_per_rad=1024 / math.pi,
                 clock=((0.0, 2.86, -2.0), 2.93),          # (position, scale)
                 caption=(((0.0, -2.2, -2.0), 2.28), ((0.0, -2.75, -2.0), 2.61))),
}
FOLLOW = 0.05           # camera distance as a fraction of the distance to the nearer end
PHOTON_SIZE = 0.03      # light-pulse half-width as a fraction of the camera distance
PHOTON_PULSE = 1.2, 0.2  # the pulse throbs: period (s), and depth as a fraction of its size
HOLD = 12               # frames held on the reel's first view at the end
REEL_START = datetime(2075, 1, 1, 6, 0, 0)            # UTC of the reel's first frame
SP = "openspace.setPropertyValueSingle"
GLARE_OPACITY = "Scene.SunGlare.Renderable.Opacity"
CORONA_OPACITY = "Scene.SGL_IntroCorona.Renderable.Opacity"
DOT_OPACITY = "Scene.SGL_IntroPlanetDot.Renderable.Opacity"
SUNSTAR_OPACITY = "Scene.SunStar.Renderable.Opacity"


def unit(v):
    return v / np.linalg.norm(v)


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def slerp(a, b, t):
    a, b = unit(a), unit(b)
    om = math.acos(min(max(float(a @ b), -1.0), 1.0))
    if om < 1e-9:
        return a
    return (math.sin((1 - t) * om) * a + math.sin(t * om) * b) / math.sin(om)


def look_quat(forward, up=UP):
    """Quaternion (w, x, y, z) turning the camera's -z onto forward, with +y toward up."""
    f = unit(forward)
    r = unit(np.cross(f, up))
    u = np.cross(r, f)
    m = np.column_stack([r, u, -f])
    tr = m.trace()
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def rotate(q, v):
    w, u = q[0], np.array(q[1:])
    v = np.asarray(v, dtype=float)
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


def qslerp(a, b, t):
    if a @ b < 0:
        b = -b
    om = math.acos(min(float(a @ b), 1.0))
    if om < 1e-9:
        return a
    return (math.sin((1 - t) * om) * a + math.sin(t * om) * b) / math.sin(om)


def hold_on_screen(cam, target, xy, fwd):
    """Forward direction, starting from fwd, that puts target at screen position xy with the
    camera's up still toward UP (Newton's method on yaw and pitch)."""
    def direction(a):
        return np.array([math.cos(a[1]) * math.cos(a[0]), math.sin(a[1]), math.cos(a[1]) * math.sin(a[0])])

    def miss(a):
        return np.array(screen_xy(dict(quat=look_quat(direction(a)), cam=cam), target, 0.0)) - xy

    f = unit(fwd)
    a = np.array([math.atan2(f[2], f[0]), math.asin(f[1])])
    for _ in range(20):
        r = miss(a)
        if np.abs(r).max() < 1e-12:
            break
        jac = np.column_stack([(miss(a + da) - r) / 1e-7 for da in np.eye(2) * 1e-7])
        a = a - np.linalg.solve(jac, r)
    return direction(a)


def smooth_max(a, b, p=4):
    return (a ** p + b ** p) ** (1 / p)


def fade(x, a, b):
    """1 at x = a, 0 at x = b (either order), smooth in log x between."""
    return 1 - smooth((math.log(x) - math.log(a)) / (math.log(b) - math.log(a)))


def ramped(lam0, lam1, duration, r0, r1, ramp_in, ramp_out):
    """lam0 to lam1 at a steady rate, eased in from r0 and out to r1 (per second) over the
    ramps. Returns the value at the end of each frame."""
    a, b = ramp_in, ramp_out
    peak = (lam1 - lam0 - (r0 * a + r1 * b) / 2) / (duration - (a + b) / 2)

    def rate(t):
        if t < a:
            return r0 + (peak - r0) * smooth(t / a)
        if t > duration - b:
            return peak + (r1 - peak) * smooth((t - duration + b) / b)
        return peak

    n, sub = int(round(duration * FPS)), 20
    lam, out = lam0, []
    for k in range(n):
        for j in range(sub):
            lam += rate((k + (j + 0.5) / sub) / FPS) / (FPS * sub)
        out.append(lam)
    return lam0 + (np.array(out) - lam0) * (lam1 - lam0) / (lam - lam0)   # land exactly on lam1


def log_leg(length, s0, u_end, duration, r0, r1, ramp_in, ramp_out):
    """Distance along a leg at the end of each frame, steady in log space, eased at both ends.

    The log coordinate counts decades out from the start until halfway, then decades down
    to the end, so leaving one end feels the same as arriving at the other. r0 and r1 are
    the starting and ending rates in decades per second.
    """
    mid = math.log10(length / 2)
    out = ramped(math.log10(s0), 2 * mid - math.log10(u_end), duration, r0, r1, ramp_in, ramp_out)
    return np.where(out <= mid, 10 ** out, length - 10 ** (2 * mid - out))


# unit, metres per unit, used from this many metres up, width of the whole part, decimals
UNITS = (("light-years", I.LY, 0.1 * I.LY, 2, 2), ("AU", I.AU, 0.1 * I.AU, 5, 1),
         ("km", 1e3, 1e4, 10, 0), ("m", 1.0, 0.0, 5, 0))


def distance_text(m):
    """The distance with its whole part right-aligned, so in the monospace clock the decimal
    point and the unit hold still while the number counts."""
    for unit, scale, above, whole, decimals in UNITS:
        if m >= above:
            w, _, f = f"{m / scale:,.{decimals}f}".partition(".")
            return f"{w:>{whole}}" + (f".{f}" if decimals else "") + f" {unit}"


def clock_text(et, et_arrive, fmt, rest):
    """Date right-aligned (so the year holds still as month names change length), then the rest."""
    return f"{date_text(et, et_arrive, fmt):>16}  -  {rest}"


def date_text(et, et_arrive, fmt):
    return (REEL_START - timedelta(seconds=et_arrive - et)).strftime(fmt).lstrip("0")


def build(reel_path, layout):
    # The reel's first camera keyframe: where and when the intro has to end.
    with open(reel_path, encoding="utf-8") as fh:
        first = next(ln.split() for ln in fh if ln.startswith("camera"))
    et_arrive = float(first[2])
    reel_pos = np.array([float(v) for v in first[3:6]])
    reel_quat = np.array([float(v) for v in first[6:10]])

    with open(TEXTURES_MANIFEST) as fh:
        geo = json.load(fh)["geometry"]
    b = geo["einstein_ring_in_solar_radii"] * I.R_SUN
    planet, image = np.array(I.planet_pos()), np.array(I.image_pos())
    limb = np.array(I.limb_points(b)[0])
    f1, f2 = unit(limb - planet), unit(image - limb)
    n_out = unit(limb)                                  # outward from the axis at the limb
    len1, len2 = np.linalg.norm(limb - planet), np.linalg.norm(image - limb)
    et_limb = et_arrive - len2 / I.C
    et_emit = et_limb - len1 / I.C
    s_start = I.PULSE_START                             # the pulse sets off from the surface

    side = unit(np.cross(f1, n_out))                     # across the ray, for the swing at the Sun

    def rig(d, f, swing=0.0):
        """Camera offset from the light pulse: behind it and a little outside the axis, or
        swung round to the side by `swing` (0 to 1)."""
        a = swing * math.pi / 2
        back = math.cos(a) * -I.RIG_BACK * f + math.sin(a) * I.RIG_BACK * side
        return d * (back + I.RIG_LIFT * n_out)

    def follow(leg, s, d, f, et, clock, swing=0.0):
        """Frame following the pulse at distance s along leg 1 (planet to limb) or 2 (limb to
        image plane; negative s means still before the limb). The camera is anchored to the
        fixed end nearer the pulse, so the camera never trails a node that moves each frame."""
        off = rig(d, f, swing)
        if leg == 1 and s < len1 / 2:
            anchor, from_anchor = "SGL_IntroPlanet", f1 * s
        elif leg == 1:
            anchor, from_anchor = "SGL_RayLimb0", -f1 * (len1 - s)
        elif s < len2 / 2:
            anchor, from_anchor = "SGL_RayLimb0", (f1 if s < 0 else f2) * s
        else:
            anchor, from_anchor = "SGL_ImagePlane", -f2 * (len2 - s)
        pos = (planet + f1 * s) if leg == 1 else (limb + (f1 if s < 0 else f2) * s)
        look = f * 1.6 * d - off if swing == 0.0 else (f * 0.6 * d - off)
        return dict(anchor=anchor, rel=from_anchor + off, cam=pos + off, quat=look_quat(look),
                    et=et, photon=pos, d=d, clock=clock)

    frames, marks = [], {}

    # ---- Overview: the planet on the left, the Sun held still in the diagram's box ----
    mid = planet / 2
    dist_a, el = layout["overview_dist"], math.radians(18)
    sun_xy = np.array(I.SUN_XY) * layout["scale"]
    fwd = None
    for k in range(int(BEATS["overview"] * FPS)):
        az = math.radians(-8 + 6 * k / (BEATS["overview"] * FPS))
        cam = mid + dist_a * np.array([-math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])
        fwd = hold_on_screen(cam, np.zeros(3), sun_xy, mid - cam if fwd is None else fwd)
        frames.append(dict(anchor="SGL_IntroPlanet", rel=cam - planet, cam=cam, quat=look_quat(fwd),
                           et=et_emit))
    planet_xy = np.array(screen_xy(frames[-1], planet, 0.0))    # where the overview leaves the planet

    # ---- Fly to the planet: log distance, swinging round behind it ----
    marks["fly"] = len(frames)
    start = planet + f1 * s_start
    v0, v1 = frames[-1]["rel"], f1 * s_start + rig(D_PLANET, f1)
    # The globe's day side is turned toward this point (intro.close_pass_direction)
    assert np.allclose(unit(v1), I.close_pass_direction(b)), "intro's globe no longer faces the close pass"
    look_end = start + f1 * 1.6 * D_PLANET
    n_fly = int(BEATS["fly"] * FPS)
    # Log distance: fast across the parsecs, then a slow approach from NEAR_DIST that eases
    # to a stop, so the planet has time to grow from a dot.
    lam0, lam1, lam_near = (math.log10(x) for x in (np.linalg.norm(v0), np.linalg.norm(v1), NEAR_DIST))
    r_near = 2 * (lam_near - lam1) / FLY_NEAR
    lams = np.concatenate([ramped(lam0, lam_near, BEATS["fly"] - FLY_NEAR, 0.0, -r_near, 1.5, 1.0),
                           ramped(lam_near, lam1, FLY_NEAR, -r_near, 0.0, 0.0, FLY_NEAR)])
    for k in range(1, n_fly + 1):
        t = k / n_fly
        dist = 10 ** lams[k - 1]
        rel = dist * slerp(v0, v1, smooth(t))
        cam = planet + rel
        if t < 0.35:
            # Turn so the planet glides from where the overview left it to the center, on
            # screen the whole way (turning the view alone let it swing off the left edge)
            fwd = hold_on_screen(cam, planet, planet_xy * (1 - smooth(t / 0.35)), fwd)
        elif t < 0.8:
            fwd = planet - cam
        else:
            fwd = slerp(planet - cam, look_end - cam, smooth((t - 0.8) / 0.2))
        frames.append(dict(anchor="SGL_IntroPlanet", rel=rel, cam=cam, quat=look_quat(fwd), et=et_emit,
                           dist=dist))

    # ---- Travel: the light pulse crosses 30 pc to the Sun's limb ----
    marks["travel"] = len(frames)
    r_slow = V_SLOW / (U_SLOW * math.log(10))
    for s in log_leg(len1, s_start, U_SLOW, BEATS["travel"], 0.0, r_slow, 2.2, 2.5):
        d = smooth_max(D_PLANET if s < len1 / 2 else D_SUN, FOLLOW * min(s, len1 - s))
        et = et_emit + s / I.C
        fr = follow(1, s, d, f1, et, clock_text(et, et_arrive, "%B %Y", f"{distance_text(s):>17} from the planet"))
        fr["to_sun"] = len1 - s
        frames.append(fr)

    # ---- Bend: past the limb at half the speed of light, the camera swinging out to the side ----
    marks["bend"] = len(frames)
    n_bend = int(BEATS["bend"] * FPS)
    for k in range(1, n_bend + 1):
        x = -U_SLOW + 2 * U_SLOW * k / n_bend                  # signed distance past the limb
        f = slerp(f1, f2, smooth(0.5 + x / (2 * U_SLOW)))
        et = et_limb + x / I.C
        swing = math.sin(math.pi * k / n_bend) ** 2
        frames.append(follow(2, x, D_SUN, f, et, clock_text(et, et_arrive, "%d %B %Y", "passing the Sun"),
                             swing=swing))

    # ---- Converge: from the limb to the image plane, 650 AU and 3.75 days later ----
    marks["converge"] = len(frames)
    for s in log_leg(len2, U_SLOW, 2 * D_IMAGE, BEATS["converge"], r_slow, 0.0, 2.0, 2.2):
        d = smooth_max(D_SUN if s < len2 / 2 else D_IMAGE, FOLLOW * min(s, len2 - s))
        et = et_limb + s / I.C
        fr = follow(2, s, d, f2, et,
                    clock_text(et, et_arrive, "%d %B %Y", f"{distance_text(len2 - s):>13} to the image plane"))
        fr["past_limb"] = s
        frames.append(fr)

    # ---- Arrive: the pulse reaches the image, the camera swings round to the reel's view ----
    marks["arrive"] = len(frames)
    last = frames[-1]
    p0, fwd0 = last["photon"], rotate(last["quat"], [0, 0, -1])
    c0 = last["rel"]                                       # already relative to the image plane
    n_arr = int(BEATS["arrive"] * FPS)
    for k in range(1, n_arr + 1):
        t = k / n_arr
        e = smooth(t)
        rad = math.exp((1 - e) * math.log(np.linalg.norm(c0)) + e * math.log(np.linalg.norm(reel_pos)))
        rel = rad * slerp(c0, reel_pos, e)
        q = look_quat(slerp(fwd0, -rel, smooth(t / 0.4)))
        if t > 0.6:
            q = qslerp(q, reel_quat, smooth((t - 0.6) / 0.4))
        tp = smooth(t / 0.5)
        frames.append(dict(anchor="SGL_ImagePlane", rel=rel, cam=image + rel, quat=q,
                           et=et_arrive - (1 - tp) * np.linalg.norm(image - p0) / I.C,
                           photon=p0 + (image - p0) * tp, d=last["d"],
                           clock=clock_text(et_arrive, et_arrive, "%d %B %Y", "at the image plane")))
    frames[-1].update(rel=reel_pos, cam=image + reel_pos, quat=reel_quat, et=et_arrive)
    # Hold on the reel's first view while the eyepiece and labels come back (playback drops
    # the last frame or two, and scripts land a frame late).
    marks["hold"] = len(frames)
    for _ in range(HOLD):
        frames.append(dict(frames[-1], photon=None, clock=None))
    return frames, marks, dict(planet=planet, image=image, et_emit=et_emit, et_arrive=et_arrive)


def events_for(frames, marks, mode):
    """Scripts to run on given frames: what is shown, captions, and the hand-over to the reel."""
    layout = LAYOUT[mode]
    vec = lambda v: "{ " + ", ".join(str(x) for x in v) + " }"

    def enable(uri, on):
        return f"{SP}('{uri}', {'true' if on else 'false'})"

    def caption(key):
        if mode == "flat":
            return [f'{SP}("ScreenSpace.SGL_IntroCaption.Text", "{I.pad(I.CAPTIONS[key], I.caption_width())}")']
        return [f'{SP}("ScreenSpace.SGL_IntroDomeCaption{i + 1}.Text", "{I.pad(line, I.dome_caption_width())}")'
                for i, line in enumerate(I.dome_caption_lines(I.CAPTIONS[key]))]

    def dome_layout():
        """Screen-space placement on the dome; the flat layout is the asset's own."""
        k = layout["scale"]
        x0, x1, y0, y1 = I.overlay_rect()
        center = ((x0 + x1) / 2 * k, (y0 + y1) / 2 * k, -2.0)
        out = [f"{SP}('ScreenSpace.SGL_IntroDiagram.CartesianPosition', {vec(center)})",
               f"{SP}('ScreenSpace.SGL_IntroDiagram.Scale', {(x1 - x0) / 2 * k})"]
        for n, key in (("PlanetLabel", "planet"), ("SunLabel", "sun"), ("DistanceLabel", "distance")):
            out += [f"{SP}('ScreenSpace.SGL_Intro{n}.Scale', {0.018 * len(I.LABELS[key]) * k})",
                    f"{SP}('ScreenSpace.SGL_Intro{n}.FaceCamera', true)"]
        pos, scale = layout["clock"]
        out += [f"{SP}('ScreenSpace.SGL_IntroClock.CartesianPosition', {vec(pos)})",
                f"{SP}('ScreenSpace.SGL_IntroClock.Scale', {scale})",
                f"{SP}('ScreenSpace.SGL_IntroClock.FaceCamera', true)"]
        for i, (pos, scale) in enumerate(layout["caption"]):
            out += [f"{SP}('ScreenSpace.SGL_IntroDomeCaption{i + 1}.CartesianPosition', {vec(pos)})",
                    f"{SP}('ScreenSpace.SGL_IntroDomeCaption{i + 1}.Scale', {scale})"]
        return out

    rays = [f"Scene.SGL_Ray{leg}{k}.Renderable.Enabled" for k in range(I.N_RAYS) for leg in ("In", "Out")]
    planet_dot, sun_dot = (f"Scene.SGL_Intro{n}.Renderable.Enabled" for n in ("PlanetDot", "SunDot"))
    planet_label, sun_label, distance_label = (f"ScreenSpace.SGL_Intro{n}.Enabled"
                                               for n in ("PlanetLabel", "SunLabel", "DistanceLabel"))
    photon, planet_img = "Scene.SGL_Photon.Renderable.Enabled", "Scene.SGL_IntroGlobe.Renderable.Enabled"
    sun_globe, sun_glare = "Scene.Sun.Renderable.Enabled", "Scene.SunGlare.Renderable.Enabled"
    corona = "Scene.SGL_IntroCorona.Renderable.Enabled"
    clock, diagram = (f"ScreenSpace.SGL_Intro{n}.Enabled" for n in ("Clock", "Diagram"))
    caps = (["ScreenSpace.SGL_IntroCaption.Enabled"] if mode == "flat" else
            [f"ScreenSpace.SGL_IntroDomeCaption{i}.Enabled" for i in (1, 2)])
    reel_ui = [f"{P}Enabled", f"{LABEL}Enabled", f"{SUN_LABEL}Renderable.Enabled",
               "Scene.SGL_SunMarker.Renderable.Enabled"]

    ev = {}

    def at(k, *scripts):
        ev.setdefault(k, []).extend(scripts)

    # The Sun's globe is on from the start, far too small to see, so its texture has loaded
    # by the close pass (switched on there, it showed grey for a moment).
    at(0, "openspace.time.setPause(true)",
       *[enable(u, False) for u in reel_ui + ["ScreenSpace.SGL_BioCaption.Enabled", photon, planet_img, clock,
                                              corona]],
       *[enable(u, True) for u in rays + caps + [planet_dot, sun_dot, planet_label, sun_label, distance_label,
                                                 diagram, sun_globe, sun_glare]],
       f"{SP}('{GLARE_OPACITY}', {GLARE})", f"{SP}('{DOT_OPACITY}', 1.0)", f"{SP}('{SUNSTAR_OPACITY}', 1.0)",
       *[f"{SP}('ScreenSpace.SGL_Intro{n}.Opacity', 1.0)" for n in ("Diagram", "SunLabel", "DistanceLabel")],
       *(dome_layout() if mode == "dome" else []), *caption("overview"))

    # Fly: from the overview markers to the planet itself as the camera closes in.
    k0 = marks["fly"]
    # write() fades the overview's diagram, Sun name and distance out over the first frames
    at(k0, enable(planet_img, True))
    at(k0 + OVERVIEW_FADE, *caption("planet"), *[enable(u, False) for u in (diagram, sun_label, distance_label)])
    for name, uris, limit in (("label", [planet_label], 0.05 * I.PC), ("dot", [planet_dot], 3e8)):
        k = next(i for i in range(k0, marks["travel"]) if frames[i]["dist"] < limit)
        at(k, *[enable(u, False) for u in uris])

    k = marks["travel"]
    # Near the Sun, write() fades its glare out and the close-pass glow in, and back again.
    at(k, enable(photon, True), enable(clock, True), enable(sun_label, True), enable(corona, True), *caption("travel"),
       f"{SP}('ScreenSpace.SGL_IntroSunLabel.Opacity', 1.0)")
    k = next(i for i in range(k, marks["bend"]) if frames[i]["to_sun"] < 2e12)
    at(k, enable(sun_dot, False), enable(sun_label, False))
    at(marks["bend"], *caption("bend"))
    at(marks["converge"], *caption("converge"))
    at(marks["arrive"], *caption("image"))
    at(marks["arrive"] + int(0.7 * BEATS["arrive"] * FPS), enable(photon, False))

    # First held frame: hand over to the reel's state (eyepiece, labels) and hide the intro.
    f = FRAMING[mode]
    sun = f["sun_label"]
    face = "true" if f["face_camera"] else "false"
    at(marks["hold"],
       *[enable(u, False) for u in rays + caps + [clock, photon, sun_globe, corona]], enable(sun_glare, True),
       f"{SP}('{GLARE_OPACITY}', {GLARE})", f"{SP}('{SUNSTAR_OPACITY}', 1.0)",
       f"{SP}('{P}PlanetSize', 1)", f"{SP}('{LABEL}Text', '{label_text('56,600')}')",
       f"{SP}('{P}Scale', {f['scale']})", f"{SP}('{P}CartesianPosition', {vec(f['eyepiece'])})",
       f"{SP}('{LABEL}CartesianPosition', {vec(f['label'])})", f"{SP}('{LABEL}Scale', {f['label_scale']})",
       f"{SP}('{P}FaceCamera', {face})", f"{SP}('{LABEL}FaceCamera', {face})",
       f"{SP}('{SUN_LABEL}Renderable.OrientationOption', {sun['orient']})",
       f"{SP}('{SUN_LABEL}Translation.Position', {vec(sun['offset'])})",
       f"{SP}('{SUN_LABEL}Renderable.Size', {sun['size']})",
       # in pixels, so scaled for the supersampled frames
       f"{SP}('{SUN_LABEL}Renderable.MinMaxSize', {vec([v * I.SSAA for v in sun['minmax']])})",
       *[enable(u, True) for u in reel_ui])
    return ev


def screen_xy(frame, target, lift):
    """Where a point appears, in ScreenSpace coordinates at z = -2 (None if behind)."""
    v = rotate(frame["quat"] * np.array([1, -1, -1, -1]), target - frame["cam"])
    if v[2] >= 0:
        return None
    return 2 * v[0] / -v[2], 2 * v[1] / -v[2] + lift


def write(path, frames, marks, events, info, layout):
    # (target, lift until the trip, lift on the trip); the Sun's name sits under its box at first
    k_ = layout["scale"]
    labels = {"PlanetLabel": (info["planet"], 0.07 * k_, 0.07 * k_),
              "SunLabel": (np.zeros(3), (-I.BOX[1] - 0.07) * k_, 0.07 * k_),
              "DistanceLabel": (info["planet"] / 2, -0.08 * k_, -0.08 * k_)}
    entries = []          # (timestamp, order, line); playback needs them in time order
    clock = None
    # Every clock line starts in the same column (the longest one centered), so the year
    # never moves; the fields inside keep the decimal point and the unit still too
    margin = (I.CLOCK_WIDTH - max(len(fr["clock"]) for fr in frames if fr.get("clock"))) // 2
    shown = {}            # opacities last set, so each is sent only when it changes

    def opacity(uri, value):
        value = round(value, 3)
        if shown.get(uri) != value:
            shown[uri] = value
            return [f"{SP}('{uri}', {value})"]
        return []

    for k, fr in enumerate(frames):
        ts, et = k / FPS, float(fr["et"])
        # Playback queues a script on the frame it's reached and runs it on the next, so
        # each frame's scripts go in a frame early (and half a frame more, for rounding).
        sts = max(0.0, (k - 1.5) / FPS)
        scripts = [f"openspace.time.setTime({et!r})"] + events.get(k, [])
        if marks["fly"] <= k < marks["fly"] + OVERVIEW_FADE:
            for name in ("Diagram", "SunLabel", "DistanceLabel"):
                scripts += opacity(f"ScreenSpace.SGL_Intro{name}.Opacity",
                                   1 - smooth((k - marks["fly"] + 1) / OVERVIEW_FADE))
        if marks["fly"] <= k < marks["travel"]:
            # The planet's marker gives way to its image as the image grows past a few pixels
            r_px = layout["px_per_rad"] * I.R_EARTH / fr["dist"]
            scripts += opacity(DOT_OPACITY, fade(r_px, *DOT_FADE))
        if marks["travel"] <= k < marks["hold"]:
            to_sun = float(np.linalg.norm(fr["cam"]))
            scripts += opacity(GLARE_OPACITY, GLARE * fade(to_sun, *GLARE_FADE))
            scripts += opacity(CORONA_OPACITY, 1 - fade(to_sun, *CORONA_FADE))
            scripts += opacity(SUNSTAR_OPACITY, fade(to_sun, *SUNSTAR_FADE))
        if fr.get("photon") is not None:
            scripts.append(f"{SP}('Scene.SGL_Photon.Translation.Position', {I.lua_vec(fr['photon'])})")
            throb = math.sin(2 * math.pi * k / (PHOTON_PULSE[0] * FPS))
            size = PHOTON_SIZE * fr["d"] * (1 + PHOTON_PULSE[1] * throb)
            scripts += opacity("Scene.SGL_Photon.Renderable.Opacity", 0.8 + 0.2 * throb)
            scripts.append(f"{SP}('Scene.SGL_Photon.Renderable.Size', {{ {size:.6g}, {size:.6g} }})")
        if k < marks["bend"]:
            for name, (target, lift, lift_trip) in labels.items():
                xy = screen_xy(fr, target, lift if k < marks["travel"] else lift_trip) or (0.0, 9.0)
                scripts.append(f"{SP}('ScreenSpace.SGL_Intro{name}.CartesianPosition', "
                               f"{{ {xy[0]:.5f}, {xy[1]:.5f}, -2.0 }})")
        if fr.get("clock") and fr["clock"] != clock:
            clock = fr["clock"]
            scripts.append(f"{SP}('ScreenSpace.SGL_IntroClock.Text', '{(' ' * margin + clock).ljust(I.CLOCK_WIDTH)}')")
        entries += [(sts, len(entries), f"script {sts!r} {et!r} 1 {s}") for s in scripts]
        anchor = fr["anchor"]
        rel = [float(v) for v in fr["rel"]]
        q = [float(v) for v in fr["quat"]]
        # View scale as the orbital navigator sets it, from the distance to what the camera
        # is looking at (the pulse when following it), not to the anchor.
        subject = float(fr["d"]) if fr.get("d") else math.hypot(*rel)
        scale = 50.0 / max(subject, 1e-3)
        entries.append((ts, len(entries), f"camera {ts!r} {et!r} {rel[0]!r} {rel[1]!r} {rel[2]!r} "
                                          f"{q[0]!r} {q[1]!r} {q[2]!r} {q[3]!r} {scale!r} F {anchor}"))
    lines = ["OpenSpace_record/playback03.00A"] + [e[2] for e in sorted(entries)]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dome", action="store_true", help="lay the opening out for the dome master")
    ap.add_argument("--out")
    ap.add_argument("--reel")
    args = ap.parse_args()
    mode = "dome" if args.dome else "flat"
    out = args.out or os.path.join(paths.recordings(), f"sgl_intro_{mode}.osrectxt")
    reel = args.reel or os.path.join(paths.recordings(), f"sgl_reel_{mode}.osrectxt")
    frames, marks, info = build(reel, LAYOUT[mode])
    write(out, frames, marks, events_for(frames, marks, mode), info, LAYOUT[mode])
    left = date_text(info["et_emit"], info["et_arrive"], "%B %Y")
    print(f"wrote {out}: {len(frames)} frames, {len(frames) / FPS:.1f} s; the light left in {left}")
    print("beats start at frames", marks)


if __name__ == "__main__":
    main()
