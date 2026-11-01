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
| `SGL_SunToFocal`, `SGL_FocalLine1`–`24` | A faint line from the Sun to 547.7 AU, then the focal line out to 1000 AU. It's drawn in 24 segments that run from 1 km long at the image plane up to 4×10¹³ m at the far end. |
| `SGL_SunDirection`, `SGL_SunMarker`, `SGL_SunLabel` | Marks the Sun as seen from the image plane. From 650 AU, nothing placed at the Sun itself draws in OpenSpace except lines and orbit trails, so the marker sits 10⁶ km sunward on the focal axis. From near the axis it falls within about 10⁻⁶ rad of the true Sun. |
| `SGL_ImagePlane` | The image plane at 650 AU, with the camera focus and bounding spheres sized for the 1,340 m image |
| `SGL_ImageGlow` | Image-plane brightness under the physical PSF, in color: EPIC's 680, 551 and 443 nm bands (on by default) |
| `SGL_ImageGlowThesis` | The thesis PSF's too-sharp version, for the before/after (off by default) |
| `SGL_ImageTruth`, `SGL_ImageRecovered` | The EPIC Earth in color, and its band-by-band recovery at SNR ~12 (off by default) |
| `SGL_Samples` | One point per formation stop inside the disk, in the color recorded there |
| `SGL_BioNDVITruth`, `…Recorded`, `…Recovered`, `…Mission` | Vegetation (NDVI) over the planet's disk: the planet, the image plane as recorded, recovered at SNR 10, and recovered with 1 m telescopes (off by default) |
| `SGL_BioO2Truth`, `SGL_BioO2Recovered` | Oxygen A-band depth: the planet and its recovery at SNR 10 (off by default) |
| `SGL_SC00`–`SGL_SC11` and trails | The 12-craft formation on keyframed bang-coast-bang paths. The mission starts 2075-01-01 and lasts 13.4 days. |

Press **B** to step through the biosignature maps, with a caption for each, and then back to the normal view. In the `sgl_telescope` profile the eyepiece switches to the same false color. The numbers behind the maps are in [`../layer_a/README.md`](../layer_a/README.md#biosignatures-outbiosignatures).

**The opening** (`sgl_intro.asset`, loaded by the `sgl_telescope` profile, all hidden until its recording plays):

| Node | What it shows |
|---|---|
| `SGL_IntroPlanet` | The planet at 30 pc in the target direction: a billboard of the EPIC Earth at 1024 px, Earth-sized, white-balanced on its clouds, with a thin atmosphere glow past the limb |
| `SGL_Photon` | The light pulse. `tools/make_intro.py` moves it and sizes it every frame. |
| `SGL_RayLimb0`–`7`, `SGL_RayIn0`–`7`, `SGL_RayOut0`–`7` | Eight rays from the planet past the Sun at 1.09 solar radii, then on to the image plane. Ray 0 is the one followed. |
| `SGL_IntroPlanetDot`, `SGL_IntroSunDot` | Markers that keep the planet and the Sun visible from parsecs away. The planet's marker fades out as its image grows past a few pixels. |
| `SGL_IntroCorona` | A glow around the textured Sun for the close pass |
| `SGL_IntroCaption`, `SGL_IntroClock`, `SGL_IntroDiagram`, name labels | Screen-space captions, the date and distance, the not-to-scale bending diagram (joined by two lines to a box around the Sun, which the overview holds still on screen), and names placed over the planet and the Sun each frame |

`intro.py` holds the geometry and generates the asset, and `tools/make_intro.py` writes the recording. With `--dome` it lays the opening out for the fisheye master: the overview 1.6× larger, the clock above it, and each caption split over two lines (`SGL_IntroDomeCaption1` and `2`) below, all turned to face the camera. The recording ends on the flat reel's first keyframe and restores the reel's state (eyepiece, labels, Sun glare).

The profile opens paused at the mission start. The camera sits just behind the image plane, slightly off the focal axis, looking back toward the Sun, so the image plane, the formation, the focal line and the Sun are all in view. At a delta time of 3600 (one hour per second), the whole sweep plays in about 5½ minutes.

## Things learned while building it (OpenSpace 0.22)

- **Profile paths:** assets in the user folder must be listed as `${USER_ASSETS}/sgl/sgl`. A bare `sgl/sgl` is looked up only in `data/assets`.
- **Rotation tables:** a 3×3 `StaticRotation` table is read column by column. The generator writes the columns: the local x, y and z axes in galactic coordinates.
- **Line nodes:** `RenderableNodeLine` computes its endpoints in world space, then also applies its own node's transform. Line nodes must therefore hang off `SolarSystemBarycenter`, which has no rotation or offset. Parented to the rotated SGL frame, the focal line came out pointing the wrong way.
- **Long lines:** the endpoints go to the GPU as 32-bit floats relative to the camera's focus, so one 452 AU line would be drawn up to ~1000 km off near the image plane. The axis is segmented for that reason.
- **Screenshots:** on a high-DPI display, window captures can miss part of the frame, so use `openspace.takeScreenshot()`.
- **Image layers:** `RenderablePlaneImageLocal` needs `BlendMode = "Additive"`. Otherwise the texture's black surround is opaque and hides the stars, samples and spacecraft. `Size` is the half-width. The biosignature maps instead use `"Normal"` with a transparent surround, sit 1 m toward the camera so they draw over the glow, and hide the far half of the focal line as a solid plane should.
- **Per-point colors:** `RenderablePointCloud` colors points through a colormap, not per point. The generator quantizes the sample colors to a 256-entry palette (mean error 1.2/255), writes it as `data/sample_colors.cmap` and gives each point its index plus 0.5. The colormap is a 1D texture with nearest filtering, so with `ValueRange = { 0, 256 }` each point lands in the middle of its entry.
- **Actions:** an action's command runs in the global scope, so the B action has its views and texture paths written into the command string.
- **Dataset cache:** OpenSpace caches parsed point-cloud CSVs under `cache/` and didn't notice when the samples file gained a column. The color mapping quietly fell back to one color. After changing a CSV's columns, delete its folder in the cache.
- **Writing a recording by hand** (what `make_intro.py` does):
  - **Camera lines:** each is `camera <time> <sim time> <x y z> <w x y z> <scale> F <anchor>`. `F` puts position and rotation in the anchor's own frame, and the camera looks down its −z. `scale` is 50 divided by the distance to what the camera looks at, as the orbital navigator sets it; computed from the anchor distance instead, it can push nearby objects inside the near plane.
  - **Order and parsing:** entries must be in time order. Parsing stops at the first bad line, which silently ends the playback there; a NumPy `np.float64(...)` repr did that.
  - **Simulation time:** only the first entry sets it. After that it follows scripts, and playback queues a script on the frame it's reached but runs it on the next, so each frame's scripts go in a frame early.
- **Moving anchors:** the camera is placed from its anchor's position before the scene updates, so it trails an anchor that moves each frame by one frame. The opening anchors to fixed nodes (the planet, the limb point, the image plane) and moves the light pulse by script.
- **30 pc from the origin:** world positions in 32-bit floats are off by ~5×10¹⁰ m there. `RenderableLabel` and the point-cloud pulse didn't draw, so the names are screen-space text placed each frame and the pulse is a billboard plane. A scripted plane `Size` must be a table `{ w, h }`.
- **The Sun up close:** in these profiles the Sun is drawn as glare only, and its textured globe (`Scene.Sun` with the `sun_texture` layer) is off.
  - **Texture loading:** a globe switched on mid-shot draws grey, then fills in tile by tile. The opening switches it on from the first frame, far too small to see, so the texture has loaded by the close pass.
  - **Glare to globe:** the glare fades out between 10¹² and 2×10¹¹ m from the Sun, while it's still small. Faded closer in, it ballooned into a halo first.
  - **SunStar:** the stars asset also draws the Sun as a star (`SunStar`) for views from outside the solar system. It swells into a halo on approach and only fades between 0.1 and 0.0075 pc, a flash at the opening's speed, so the opening fades it out between 10¹⁷ and 10¹⁶ m.
  - **Corona:** a glow billboard at the Sun's center fades in. It faces the camera from inside the globe, so the globe hides the part over its disk and only the ring past the limb shows.
- **Sharper frames:**
  - **MSAA:** with `"msaa": 4` in the window config, OpenSpace's frame capture came out black. Instead the flat clip renders at 3840×2160 (`"res"` in the window config, independent of the window's size) and is scaled to 1080p when encoded.
  - **Line widths:** these are in framebuffer pixels, so `export_frames.py --line-scale 2` doubles every line and trail for the run, and the `MinMaxSize` pixel clamp on labels too.
  - **Text:** `ScreenSpaceText` draws at `FontSize` (15 by default) and then stretches to its `Scale`, so the captions set `FontSize = 60` to stay sharp.
  - **Text boxes:** OpenSpace 0.22 draws `ScreenSpaceText` into an RGB texture with no alpha, so each caption sits on an opaque black box. It's invisible against space, but it shows over the Earth or the Sun. `install_module.py` patches the texture to RGBA. The caption and the clock then get a 40% `BackgroundColor` band, so white text stays legible over clouds and the Sun.
  - **Config path:** `${CONFIG}` resolves to the source build's `config` folder.
- **Fisheye framing:** SGCT rotates the cube map by 90° − `tilt`, so with `"tilt": 90` the camera's forward direction is the center of the dome master and its up is the top. Angle from forward maps linearly to radius (90° at the rim). The dome reel looks across the focal axis from 2 km off it, so the image plane and the Sun land about 62° either side of center.
- **Trail widths:** a trail's width is `Renderable.Appearance.LineWidth`, and planet orbits default to 10 px. The profiles set them to 6 px at 0.8 opacity, and the spacecraft trails are 1.5 px at 0.8.
- **Verified in the running app:**
  - The image plane sits 650.0000 AU from the Sun along the intended direction.
  - Looking down the focal line, the texture's orientation matches the sample coordinates.
  - Spacecraft 1 starts top-left, as in the CSV.

## Layer C: the telescope view (C++ module)

`module_sgl/` is an OpenSpace module, `Sgl`, with one screen-space renderable, `ScreenSpaceSglTelescope`. It shows what one spacecraft's telescope sees when it looks back at the Sun: the planet lensed into an Einstein ring around a coronagraph occulter, drawn in a circular eyepiece.

- **What it computes:** the shader `module_sgl/shaders/telescope_fs.glsl` is a line-for-line port of `layer_c_proto/ring_view.py`. It ray-traces each pixel back through the point-mass lens equation, in units of the Einstein angle, and reads the planet texture there. Each frame, the C++ side works out three things:
  - the Einstein angle and the Sun's size, from the image plane's distance to the Sun
  - the telescope's position in image-plane radii, from the spacecraft node
  - the magnification shown on screen
- **Properties:** `PlanetSize` (three presets), `Magnification` (read-only), `TelescopeNode` or `ManualOffset`, `OcculterRadius`, `CoronaStrength`, `Eyepiece`, `Resolution`, and `SaveFrame`, which writes the view to a PNG for testing.
- **Install and build:** run `python install_module.py`. Then configure with `-DOPENSPACE_MODULE_SGL=ON` and build. Only the source build can load `sgl_telescope.asset`, through the `sgl_telescope` profile.
- **Controls:** **M** steps the planet size through ×22,600, ×56,600 and ×135,700, and the label below the eyepiece follows.

**Verified against the Python reference** (u = (0.35, 0.2), κ = 0.05, 640 px; rerun with `tools/compare_gpu_frame.py`, which compares the committed GPU frame `layer_c_proto/out/gpu_frame_c_occulter.png` or captures a fresh one):
- **Match:** r = 0.9992, mean difference 1.2/255 on the ring, and 0.17% of pixels off by more than 16/255 (8-bit texture versus float reference). Mirrored or rotated frames score r ≈ 0.66, so the orientation is right.
- **Geometry:** the magnifications match the prototype exactly, and the telescope offset read from spacecraft 6 matches its path.
- **Speed:** about 28 fps on the Iris Xe, with the scene and the view together.
- **Lessons from the port:**
  - In this ghoul version, textures bind through `TextureUnit::bind(texture)`.
  - Saved textures come out bottom-up, so flip them before comparing.
  - PNG planet textures already carry their display gain, so the module's gain defaults to 1.
  - A dashboard property readout never resolved the screen-space property, so the label is a `ScreenSpaceText` that the M action updates.

## Next steps

- **Reveal the samples over time.** They currently appear all at once; a time-filtered point cloud would reveal each one as the formation reaches it.
- **An instrument view.** A second pass that draws the ring as a 1 m telescope at 443 nm would record it, behind a switch.
