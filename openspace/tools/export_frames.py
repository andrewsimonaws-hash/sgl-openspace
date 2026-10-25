"""Play back a session recording in a running OpenSpace and save every frame as a PNG.

Usage: python export_frames.py <recording file> [fps]
Hides the GUI, dashboard and on-screen log while it plays so the frames are clean, and
shows them again afterwards. Screen-space items such as the telescope eyepiece stay.
Frames land in OpenSpace's user/screenshots/<date-time>/ folder.
"""
import sys
import time

from oslua import send

OVERLAYS = ("Modules.CefWebGui.Visible", "Dashboard.IsEnabled", "RenderEngine.ShowLog",
            "RenderEngine.ShowVersion")

recording = sys.argv[1].replace("\\", "/")
fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

send([f"openspace.setPropertyValueSingle('{p}', false)" for p in OVERLAYS]
     + [f"openspace.sessionRecording.startPlayback('{recording}', false, true, {fps})"],
     timeout=60)
start = time.time()
time.sleep(1.0)
while send(["return openspace.sessionRecording.isPlayingBack()"], timeout=60)[0].get("1"):
    time.sleep(2.0)
send([f"openspace.setPropertyValueSingle('{p}', true)" for p in OVERLAYS], timeout=60)
print(f"done in {time.time() - start:.0f}s")
