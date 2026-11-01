"""Play back a session recording in a running OpenSpace and save every frame as a PNG.

Usage: python export_frames.py <recording file> [fps] [--line-scale k]
Hides the GUI, dashboard and on-screen log while it plays so the frames are clean, and
shows them again afterwards. Screen-space items such as the telescope eyepiece stay.
Frames land in OpenSpace's user/screenshots/<date-time>/ folder.

Line widths and label size clamps are in framebuffer pixels, so for frames rendered at k
times the clip's size (and scaled down when encoded), --line-scale k scales every line,
trail and label clamp for the run, and puts them back afterwards.
"""
import argparse
import time

from oslua import send

OVERLAYS = ("Modules.CefWebGui.Visible", "Dashboard.IsEnabled", "RenderEngine.ShowLog",
            "RenderEngine.ShowVersion")
LINES = ("Scene.*.Renderable.LineWidth", "Scene.*.Renderable.Appearance.LineWidth")
CLAMPS = ("Scene.*.Renderable.MinMaxSize",)


def scale_lines(k):
    return ([f'for _, uri in ipairs(openspace.property("{p}")) do '
             f'openspace.setPropertyValueSingle(uri, openspace.propertyValue(uri) * {k}) end' for p in LINES]
            + [f'for _, uri in ipairs(openspace.property("{p}")) do local v = openspace.propertyValue(uri); '
               f'openspace.setPropertyValueSingle(uri, {{ math.floor(v[1] * {k} + 0.5), math.floor(v[2] * {k} + 0.5) }}) end'
               for p in CLAMPS])


ap = argparse.ArgumentParser()
ap.add_argument("recording")
ap.add_argument("fps", type=int, nargs="?", default=30)
ap.add_argument("--line-scale", type=float, default=1.0)
args = ap.parse_args()
recording = args.recording.replace("\\", "/")

send([f"openspace.setPropertyValueSingle('{p}', false)" for p in OVERLAYS]
     + (scale_lines(args.line_scale) if args.line_scale != 1 else [])
     + [f"openspace.sessionRecording.startPlayback('{recording}', false, true, {args.fps})"],
     timeout=60)
start = time.time()
time.sleep(1.0)
while send(["return openspace.sessionRecording.isPlayingBack()"], timeout=60)[0].get("1"):
    time.sleep(2.0)
send([f"openspace.setPropertyValueSingle('{p}', true)" for p in OVERLAYS]
     + (scale_lines(1 / args.line_scale) if args.line_scale != 1 else []), timeout=60)
print(f"done in {time.time() - start:.0f}s")
