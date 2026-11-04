"""Record the reel's session in a running OpenSpace (source build, sgl_telescope profile).

About 20 s of real time at 4 hours per second, starting 6 hours into the mission, while
spacecraft 6 is inside the planet's image: its eyepiece shows the full Einstein ring as the
planet size steps medium -> bold -> thin -> medium, the same as pressing M. (Earlier in the
mission, outside the image, the eyepiece shows two arcs instead.) Playback
(export_frames.py) then renders it frame by frame.

The flat take looks back toward the Sun from just behind the image plane, slightly off the
focal axis, so the image plane, the formation, the focal line and the Sun at its far end
are all in frame. The camera drifts slowly enough that the Sun stays in view.

    python record_reel.py                 # flat framing
    python record_reel.py --dome          # fisheye framing, for a dome master
"""
import argparse
import os
import sys
import time

from oslua import send

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402

DURATION = 20.0
SIZE_SWITCHES = (9.0, 13.0, 17.0)      # seconds into the clip
P = "ScreenSpace.SGL_Telescope."
LABEL = "ScreenSpace.SGL_TelescopeLabel."
# The eyepiece label, padded to its longest form (x135,700): the text box is sized from the
# first text it shows. build_asset.py writes the same text into the asset.
LABEL_TEXT, LABEL_WIDTH = "Telescope: Einstein ring of planet x{} (M)", 47

# camera position and up vector in the image plane's frame (m), camera pitch and yaw (rad;
# up and right are positive, measured from looking at the image plane), orbit rate per step, eyepiece scale and position, label position and scale,
# whether the screen-space planes turn toward the camera (which foreshortens them when far
# off-axis), and the Sun label: orientation (0 = camera view direction, 1 = camera
# position normal), offset from the marker (m, 1e9 m away), size exponent, min-max pixels
FRAMING = {
    "flat": {
        "camera": (970, 300, 3000), "up": (0, 1, 0), "pitch": 0.0, "yaw": 0.0, "orbit": 0.04,
        "scale": 0.38, "eyepiece": (-1.05, 0.42, -2.0),
        "label": (-1.05, -0.03, -2.0), "label_scale": 0.55, "face_camera": False,
        "sun_label": {"orient": 0, "offset": (1.1e7, 1.3e7, 0), "size": 6.6,
                      "minmax": (14, 28)},
    },
    # In fisheye the camera sits 2 km from the image plane, off to the side and sunward of it,
    # and looks across the focal axis: the image plane on the left of the dome master, the Sun
    # (650 AU away, along the axis) on the right, the axis running across between them, just
    # below center. The eyepiece and its label sit in the open sky above. The Sun label faces
    # the camera and keeps its world size (min-max wide open): each cube face applies the
    # pixel clamp on its own, so a clamped label can split where it crosses a face seam.
    "dome": {
        "camera": (1640, 0, -1150), "up": (0, 1, 0), "pitch": 0.15, "yaw": 1.09, "orbit": 0.03,
        "scale": 0.68, "eyepiece": (0.0, 1.15, -1.64),
        "label": (0.0, 0.39, -1.96), "label_scale": 1.03, "face_camera": True,
        "sun_label": {"orient": 1, "offset": (-3.6e8, 6e7, 0), "size": 6.0,
                      "minmax": (1, 100)},
    },
}
SUN_LABEL = "Scene.SGL_SunLabel."


def lua_vec(v):
    return "{ " + ", ".join(str(x) for x in v) + " }"


def label_text(magnification):
    return LABEL_TEXT.format(magnification).ljust(LABEL_WIDTH)


def framing_scripts(mode):
    """Scripts that set up the reel's first view: time, eyepiece, labels and camera."""
    f = FRAMING[mode]
    face = "true" if f["face_camera"] else "false"
    sun = f["sun_label"]
    return [
        "openspace.time.setPause(true)",
        "openspace.time.setTime('2075-01-01T06:00:00')",
        f"openspace.setPropertyValueSingle('{P}PlanetSize', 1)",
        f"openspace.setPropertyValueSingle('{LABEL}Text', '{label_text('56,600')}')",
        f"openspace.setPropertyValueSingle('{P}Scale', {f['scale']})",
        f"openspace.setPropertyValueSingle('{P}CartesianPosition', {lua_vec(f['eyepiece'])})",
        f"openspace.setPropertyValueSingle('{LABEL}CartesianPosition', {lua_vec(f['label'])})",
        f"openspace.setPropertyValueSingle('{LABEL}Scale', {f['label_scale']})",
        f"openspace.setPropertyValueSingle('{P}FaceCamera', {face})",
        f"openspace.setPropertyValueSingle('{LABEL}FaceCamera', {face})",
        f"openspace.setPropertyValueSingle('{SUN_LABEL}Renderable.OrientationOption', {sun['orient']})",
        f"openspace.setPropertyValueSingle('{SUN_LABEL}Translation.Position', {lua_vec(sun['offset'])})",
        f"openspace.setPropertyValueSingle('{SUN_LABEL}Renderable.Size', {sun['size']})",
        f"openspace.setPropertyValueSingle('{SUN_LABEL}Renderable.MinMaxSize', {lua_vec(sun['minmax'])})",
        "openspace.navigation.setNavigationState({ Anchor = 'SGL_ImagePlane', "
        f"ReferenceFrame = 'SGL_ImagePlane', Position = {lua_vec(f['camera'])}, "
        f"Up = {lua_vec(f['up'])}, Pitch = {f['pitch']}, Yaw = {f['yaw']} }})",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dome", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    mode = "dome" if args.dome else "flat"
    out = (args.out or os.path.join(paths.recordings(), f"sgl_reel_{mode}.osrectxt")).replace("\\", "/")
    f = FRAMING[mode]

    send(framing_scripts(mode), timeout=60)
    time.sleep(2)

    send(["openspace.sessionRecording.startRecording()",
          "openspace.time.setDeltaTime(14400)",
          "openspace.time.setPause(false)"], timeout=60)
    t0 = time.time()
    switches = list(SIZE_SWITCHES)
    while (elapsed := time.time() - t0) < DURATION:
        script = [f"openspace.navigation.addGlobalRotation({f['orbit']}, 0.0)"]
        if switches and elapsed >= switches[0]:
            script.append("openspace.action.triggerAction('sgl.telescope.next_planet_size')")
            switches.pop(0)
        send(script, timeout=30)
        time.sleep(0.4)
    send(["openspace.time.setPause(true)",
          f"openspace.sessionRecording.stopRecording('{out}', 'Ascii', true)"], timeout=60)

    # A recording stores both the triggerAction call and the script the action ran, so
    # playback would step the planet size twice per press. Keep only the triggerAction lines.
    with open(out, encoding="utf-8") as fh:
        lines = fh.readlines()
    kept = [ln for ln in lines if not (ln.startswith("script") and "local uri =" in ln)]
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(kept)
    print("saved", out, f"({len(lines) - len(kept)} duplicate action lines removed)")


if __name__ == "__main__":
    main()
