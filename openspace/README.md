# Layer B: the SGL scene in OpenSpace

Lua assets that place the solar gravitational lens mission in OpenSpace 0.22, built from the layer-A data. `build_asset.py` writes everything into `sgl/` and installs it into `C:\Users\andre\OpenSpace\v0.22.0\user\data\assets\sgl`. It also installs the `sgl` profile into `user\data\profiles`.

```bash
../.venv/Scripts/python build_asset.py
```

To launch, run this from `C:\Users\andre\OpenSpace\v0.22.0`:

```bash
./bin/OpenSpace.exe --profile sgl --bypassLauncher
```

## What's in the scene

| Node | What it shows |
|---|---|
| `SGL_Frame` | Local frame on the Sun. z runs down the focal line, away from the target (a hypothetical direction, galactic l = 200°, b = 25°). x and y span the image plane in metres. |
| `SGL_SunToFocal`, `SGL_FocalLine1`–`24` | A faint line from the Sun to 547.7 AU, then the focal line out to 1000 AU. It's drawn in 24 segments that run from 1 km long at the image plane up to 10¹³ m at the ends. |
| `SGL_SunDirection`, `SGL_SunMarker`, `SGL_SunLabel` | Marks the Sun as seen from the image plane. From 650 AU, nothing placed at the Sun itself draws in OpenSpace except lines and orbit trails, so the marker sits 10⁶ km sunward on the focal axis. From near the axis it falls within about 10⁻⁶ rad of the true Sun. |
| `SGL_ImagePlane` | The image plane at 650 AU, with the camera focus and bounding spheres sized for the 1,340 m image |
| `SGL_ImageGlow` | Image-plane brightness under the physical PSF, in color: EPIC's 680, 551 and 443 nm bands (on by default) |
| `SGL_ImageGlowBlue` | The same for the 443 nm band only, as in the thesis. The sample points are colored from this band. (Off by default.) |
| `SGL_ImageGlowThesis` | The thesis PSF's too-sharp version, for the before/after (off by default) |
| `SGL_ImageTruth`, `SGL_ImageRecovered` | The EPIC Earth in color, and its band-by-band recovery at SNR ~12 (off by default) |
| `SGL_Samples` | One point per formation stop inside the disk, colored by the recorded brightness (inferno colormap) |
| `SGL_SC00`–`SGL_SC11` and trails | The 12-craft formation on keyframed bang-coast-bang paths. The mission starts 2075-01-01 and lasts 13.4 days. |

The profile opens paused at the mission start. The camera sits just behind the image plane, slightly off the focal axis, looking back toward the Sun, so the image plane, the formation, the focal line and the Sun are all in view. At a delta time of 3600 (one hour per second), the whole sweep plays in about 5½ minutes.

## Things learned while building it (OpenSpace 0.22)

- **Profile paths:** assets in the user folder must be listed as `${USER_ASSETS}/sgl/sgl`. A bare `sgl/sgl` is looked up only in `data/assets`.
- **Rotation tables:** a 3×3 `StaticRotation` table is read column by column. The generator writes the columns: the local x, y and z axes in galactic coordinates.
- **Line nodes:** `RenderableNodeLine` computes its endpoints in world space, then also applies its own node's transform. Line nodes must therefore hang off `SolarSystemBarycenter`, which has no rotation or offset. Parented to the rotated SGL frame, the focal line came out pointing the wrong way.
- **Long lines:** the endpoints go to the GPU as 32-bit floats relative to the camera's focus, so one 452 AU line would be drawn up to ~1000 km off near the image plane. The axis is segmented for that reason.
- **Screenshots:** the left monitor is 4K at 250% scaling. Window captures miss most of the frame, so use `openspace.takeScreenshot()` (3840×2104).
- **Image layers:** `RenderablePlaneImageLocal` needs `BlendMode = "Additive"`. Otherwise the texture's black surround is opaque and hides the stars, samples and spacecraft. `Size` is the half-width.
- **Verified in the running app:**
  - The image plane sits 650.0000 AU from the Sun along the intended direction.
  - Looking down the focal line, the texture's orientation matches the sample coordinates.
  - Spacecraft 1 starts top-left, as in the CSV.

## Layer C: the telescope view (C++ module)

`module_sgl/` is an OpenSpace module, `Sgl`, with one screen-space renderable, `ScreenSpaceSglTelescope`. It shows what one spacecraft's telescope sees when it looks back at the Sun: the planet lensed into an Einstein ring around a coronagraph occulter, drawn in a circular eyepiece.

- **What it computes:** the shader `module_sgl/shaders/telescope_fs.glsl` is a line-for-line port of `layer_c_proto/ring_view.py`. Per pixel, it inverts the point-mass lens equation in units of the Einstein angle and reads the planet texture there. Each frame, the C++ side works out three things:
  - the Einstein angle and the Sun's size, from the image plane's distance to the Sun
  - the telescope's position in image-plane radii, from the spacecraft node
  - the magnification shown on screen
- **Properties:** `PlanetSize` (three presets), `Magnification` (read-only), `TelescopeNode` or `ManualOffset`, `OcculterRadius`, `CoronaStrength`, `Eyepiece`, `Resolution`, and `SaveFrame`, which writes the view to a PNG for testing.
- **Install and build:** run `python install_module.py`. Then configure with `-DOPENSPACE_MODULE_SGL=ON` and build. Only the source build can load `sgl_telescope.asset`, through the `sgl_telescope` profile.
- **Controls:** **M** steps the planet size through ×22,600, ×56,600 and ×135,700, and the label above the eyepiece follows.

**Verified against the Python reference** (u = (0.35, 0.2), κ = 0.05, 640 px):
- **Match:** r = 0.9992, mean difference 1.2/255 on the ring, and 0.17% of pixels off by more than 16/255 (8-bit texture versus float reference). Mirrored or rotated frames score r ≈ 0.66, so the orientation is right.
- **Geometry:** the magnifications match the prototype exactly, and the telescope offset read from spacecraft 6 matches its path.
- **Speed:** about 28 fps on the Iris Xe, with the scene and the view together.
- **Lessons from the port:**
  - In this ghoul version, textures bind through `TextureUnit::bind(texture)`.
  - Saved textures come out bottom-up, so flip them before comparing.
  - PNG planet textures already carry their display gain, so the module's gain defaults to 1.
  - A dashboard property readout never resolved the screen-space property, so the label is a `ScreenSpaceText` that the M action updates.

## Not done yet

- **Samples appear all at once.** They aren't revealed as the formation reaches them; a time-filtered point cloud could fix that.
- **The fisheye window and the default window open partly above the left monitor.** They need a nudge after launch.
- **Week 3:** the instrument-real view (decision B3). It's a second pass that draws the ring as a 1 m telescope at 443 nm would record it, behind a switch.
