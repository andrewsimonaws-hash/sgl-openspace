"""Generate the OpenSpace SGL assets (layer B) from the layer-A data.

Writes openspace/sgl/:
  sgl.asset            focal line, image-plane disk (textures), sample point cloud
  sgl_formation.asset  12 spacecraft with keyframed paths and trails
  textures/, data/     copies of the layer-A textures and the sample CSV
and copies the folder into OpenSpace's user/data/assets/sgl and the profiles into
user/data/profiles, unless --no-install. Needs OPENSPACE_ROOT (see paths.py) either way,
because the overview diagram is drawn in OpenSpace's own font.

    python build_asset.py
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

import intro
import paths

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER_A = os.path.join(HERE, "..", "layer_a", "out")
OUT = os.path.join(HERE, "sgl")

AU = 1.495978707e11
EPOCH = datetime(2075, 1, 1)
# Hypothetical target direction, galactic longitude/latitude (degrees). The focal line
# points the opposite way, from the Sun away from the target.
TARGET_L, TARGET_B = 200.0, 25.0
FOCAL_END_AU = 1000.0
HOP_SUBSAMPLES = 6
TELESCOPE_SC = 5        # whose view the telescope inset shows (mid-formation)
PALETTE = [(1.0, 0.66, 0.25), (0.45, 0.68, 1.0), (0.55, 0.88, 0.6), (0.95, 0.5, 0.8)]
TEXTURES = ["image_plane_stretched_rgb.png", "image_plane_thesis_psf_rgb.png", "truth_rgb.png",
            "truth_rgba_rgb.png", "recovered_filtered_snr10_rgb.png"]
BIO_TEXTURES = ["bio_ndvi_truth.png", "bio_ndvi_recorded.png", "bio_ndvi_snr10.png", "bio_ndvi_tt_fig4.png",
                "bio_o2_truth.png", "bio_o2_snr10.png"]
# N steps through these: (layer, caption, eyepiece texture). With the telescope view loaded,
# the eyepiece shows the planet in the same false color.
BIO_VIEWS = [
    ("SGL_BioNDVITruth", "Vegetation (NDVI): green plants, tan desert, blue ocean, white cloud", "ndvi"),
    ("SGL_BioNDVIRecorded", "Vegetation, as recorded: the blur flattens it to about the Sahara's", "ndvi"),
    ("SGL_BioNDVIRecovered", "Vegetation, recovered at SNR 10 per band", "ndvi"),
    ("SGL_BioNDVIMission", "Vegetation, recovered with 1 m telescopes (SNR 4, 128 px)", "ndvi"),
    ("SGL_BioO2Truth", "Oxygen A-band depth: pale over high cloud, dark toward the limb", "o2"),
    ("SGL_BioO2Recovered", "Oxygen A-band depth, recovered at SNR 10 per band", "o2"),
]


def sample_palette(rgb):
    """Quantize sample colors to a 256-entry palette sorted by brightness.

    OpenSpace colors a point cloud through a colormap, not per point, so each point stores
    its palette index and the palette becomes the colormap. Returns (palette, index).
    """
    img = Image.fromarray((rgb[None] * 255).round().astype(np.uint8), mode="RGB")
    q = img.quantize(256, method=Image.Quantize.MEDIANCUT)
    idx = np.asarray(q)[0]
    used = int(idx.max()) + 1
    palette = np.array(q.getpalette()[:3 * used], dtype=float).reshape(used, 3) / 255
    order = np.argsort(palette @ [0.2126, 0.7152, 0.0722])
    rank = np.empty(used, int)
    rank[order] = np.arange(used)
    return palette[order], rank[idx]


def iso(seconds):
    return (EPOCH + timedelta(seconds=float(seconds))).strftime("%Y-%m-%dT%H:%M:%S")


def frame_matrix():
    """3x3 matrix mapping local (x, y, z) to galactic; z points down the focal line."""
    l, b = math.radians(TARGET_L), math.radians(TARGET_B)
    target = np.array([math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b)])
    z = -target
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def axis_offsets(focal_start_au):
    """Points along the focal axis, in metres from the image plane: 0, then +/-1 km,
    10 km, ... out to the focal line's start (toward the Sun) and its far end."""
    near = (focal_start_au - 650.0) * AU
    far = (FOCAL_END_AU - 650.0) * AU
    steps = [10.0**k for k in range(3, 14)]
    toward_sun = [-s for s in steps if s < -near] + [near]
    away = [s for s in steps if s < far] + [far]
    return sorted(toward_sun) + [0.0] + away


def lua_vec(v):
    return "{ " + ", ".join(f"{float(a):.10g}" for a in v) + " }"


def bcb_points(p0, p1, duration, thrust, n):
    """Positions at n interior times of a bang-coast-bang hop (same profile as layer A)."""
    d = p1 - p0
    length = float(np.linalg.norm(d))
    ta = (duration - math.sqrt(max(duration**2 - 4 * length / thrust, 0.0))) / 2
    out = []
    for k in range(1, n + 1):
        t = duration * k / (n + 1)
        if t < ta:
            s = 0.5 * thrust * t**2
        elif t <= duration - ta:
            s = 0.5 * thrust * ta**2 + thrust * ta * (t - ta)
        else:
            s = length - 0.5 * thrust * (duration - t) ** 2
        out.append((t, p0 + d * (s / length)))
    return out


def keyframes_for(stops, dwell, thrust):
    """Arrive/depart keyframes at each stop plus bang-coast-bang points along each hop."""
    keys = []
    for i, (t, p) in enumerate(stops):
        if i > 0:
            t_prev, p_prev = stops[i - 1]
            dep = t_prev + dwell
            for dt, q in bcb_points(p_prev, p, t - dep, thrust, HOP_SUBSAMPLES):
                keys.append((dep + dt, q))
        keys.append((t, p))
        keys.append((t + dwell, p))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-install", action="store_true")
    args = ap.parse_args()
    # The overview diagram is drawn in OpenSpace's own font, so check for it before writing
    if not os.path.isfile(paths.mono_font()):
        sys.exit(f"OpenSpace's Mono font isn't at {paths.mono_font()}; check OPENSPACE_ROOT")

    with open(os.path.join(LAYER_A, "textures", "manifest.json")) as f:
        tex = json.load(f)
    with open(os.path.join(LAYER_A, "trajectories", "manifest.json")) as f:
        traj = json.load(f)
    geo = tex["geometry"]
    image_d = geo["image_diameter_m"]
    ext_span = geo["extended_texture_span_m"]

    # Overwrite in place: the close-up planet below stays as committed when the EPIC file is missing
    os.makedirs(os.path.join(OUT, "textures"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    for folder, names in (("textures", TEXTURES), ("biosignatures", BIO_TEXTURES)):
        for name in names:
            shutil.copy(os.path.join(LAYER_A, folder, name), os.path.join(OUT, "textures", name))

    # Formation stops, grouped per spacecraft, from the layer-A samples CSV.
    stops = {}
    with open(os.path.join(LAYER_A, "trajectories", "formation12_samples.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        stops.setdefault(int(r["sc"]), []).append((float(r["t_s"]), np.array([float(r["x_m"]), float(r["y_m"])])))
    end_s = max(s[-1][0] for s in stops.values()) + traj["dwell_s"]
    inside = [r for r in rows if r["in_disk"] == "1"]
    palette, cidx = sample_palette(np.array([[float(r[c]) for c in "rgb"] for r in inside]))
    with open(os.path.join(OUT, "data", "formation12_samples.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "brightness", "cidx", "sc"])
        for r, k in zip(inside, cidx):
            # k + 0.5 lands in the middle of colormap entry k (nearest lookup over 0..len)
            w.writerow([r["x_m"], r["y_m"], "0", r["brightness"], f"{k + 0.5:.1f}", r["sc"]])
    with open(os.path.join(OUT, "data", "sample_colors.cmap"), "w", newline="\n") as f:
        f.write("# Formation sample colors (EPIC 680 / 551 / 443 nm), indexed by cidx\n")
        f.write(f"{len(palette)}\n")
        for c in palette:
            # +0.25/255 so OpenSpace's truncation to 8 bits gives back the intended level
            f.write(" ".join(f"{min(v + 0.25 / 255, 1.0):.6f}" for v in c) + " 1.0\n")

    # OpenSpace reads a 3x3 Rotation table column by column, so list the columns
    # (the local x, y, z axes in galactic coordinates) one after another.
    rot = frame_matrix().T
    main_asset = f'''-- Solar gravitational lens: focal line and image plane for an exoplanet at 30 pc.
-- Generated by openspace/build_asset.py from the layer-A data. Edit that script.
-- Local frame "SGL_Frame": z runs from the Sun down the focal line, away from the target
-- (galactic l = {TARGET_L}, b = {TARGET_B}); x and y span the image plane in metres.

local sun = asset.require("scene/solarsystem/sun/transforms")

local AU = {AU}
local ImageDiameter = {image_d:.3f}      -- metres, planet image at 650 AU

local Frame = {{
  Identifier = "SGL_Frame",
  Parent = sun.SunCenter.Identifier,
  Transform = {{
    Rotation = {{ Type = "StaticRotation", Rotation = {{
      {rot[0][0]:.12f}, {rot[0][1]:.12f}, {rot[0][2]:.12f},
      {rot[1][0]:.12f}, {rot[1][1]:.12f}, {rot[1][2]:.12f},
      {rot[2][0]:.12f}, {rot[2][1]:.12f}, {rot[2][2]:.12f}
    }} }}
  }},
  GUI = {{ Name = "SGL Frame", Path = "/SGL", Hidden = true, Focusable = false }}
}}

local FocalStart = {{
  Identifier = "SGL_FocalStart",
  Parent = Frame.Identifier,
  Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {{ 0, 0, {geo["focal_distance_au"]:.4f} * AU }} }} }},
  InteractionSphere = 1e9,
  GUI = {{ Name = "Focal line start (547.7 AU)", Path = "/SGL",
    Description = "Rays grazing the solar limb first meet the optical axis here." }}
}}

local FocalEnd = {{
  Identifier = "SGL_FocalEnd",
  Parent = Frame.Identifier,
  Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {{ 0, 0, {FOCAL_END_AU} * AU }} }} }},
  GUI = {{ Name = "Focal line ({FOCAL_END_AU:.0f} AU)", Path = "/SGL", Hidden = true }}
}}

local ImagePlane = {{
  Identifier = "SGL_ImagePlane",
  Parent = Frame.Identifier,
  Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {{ 0, 0, 650 * AU }} }} }},
  BoundingSphere = {ext_span / 2:.1f},
  InteractionSphere = {image_d / 2:.1f},
  GUI = {{ Name = "SGL image plane (650 AU)", Path = "/SGL",
    Description = "The exoplanet's image: {image_d:.0f} m across at 650 AU." }}
}}

-- RenderableNodeLine works out its endpoints in world space and then also applies its own
-- node's transform, so line nodes sit on the barycenter (no rotation, no offset), as in
-- OpenSpace's own line assets. Parented to SGL_Frame, the line came out rotated.
local SunToFocal = {{
  Identifier = "SGL_SunToFocal",
  Parent = sun.SolarSystemBarycenter.Identifier,
  Renderable = {{
    Type = "RenderableNodeLine",
    StartNode = sun.SunCenter.Identifier,
    EndNode = FocalStart.Identifier,
    Color = {{ 1.0, 0.66, 0.25 }},
    Opacity = 0.6,
    LineWidth = 3.0
  }},
  GUI = {{ Name = "Sun to focal line", Path = "/SGL" }}
}}

-- At 650 AU the Sun's disk is about 3 arcseconds across, and markers placed at the Sun itself
-- don't draw from out here (only lines and orbit trails do). So the marker stands in on the
-- line of sight: 1e6 km sunward along the focal axis. Seen from a camera within a few km of
-- the axis it lands within ~1e-6 rad of the true Sun, exactly where the focal line ends.
local SunDirection = {{
  Identifier = "SGL_SunDirection",
  Parent = ImagePlane.Identifier,
  Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {{ 0, 0, -1e9 }} }} }},
  GUI = {{ Name = "Sun direction (marker anchor)", Path = "/SGL", Hidden = true }}
}}

local SunMarker = {{
  Identifier = "SGL_SunMarker",
  Parent = SunDirection.Identifier,
  Renderable = {{
    Type = "RenderablePointCloud",
    File = asset.resource("data/origin.csv"),
    Unit = "m",
    Coloring = {{ FixedColor = {{ 1.0, 0.82, 0.5 }} }},
    SizeSettings = {{ ScaleExponent = 7.45 }}
  }},
  GUI = {{ Name = "Sun marker", Path = "/SGL",
    Description = "Where the Sun is, seen from the image plane. Not to scale." }}
}}

local SunLabel = {{
  Identifier = "SGL_SunLabel",
  Parent = SunDirection.Identifier,
  -- Nudged up and right of the marker (about 0.7 degrees each way from the default view)
  Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {{ 1.1e7, 1.3e7, 0 }} }} }},
  Renderable = {{
    Type = "RenderableLabel",
    Text = "Sun, 650 AU",
    FontSize = 70.0,
    Size = 6.6,
    MinMaxSize = {{ 14, 28 }},
    OrientationOption = "Camera View Direction",
    BlendMode = "Additive"
  }},
  GUI = {{ Name = "Sun label", Path = "/SGL" }}
}}

-- The focal line is drawn as segments that get longer away from the image plane. Line
-- endpoints reach the GPU as 32-bit floats relative to the camera's focus, so one 452 AU
-- line would be drawn up to ~1000 km off near the image plane. Short segments near the
-- plane keep it exact where the camera is.
local AxisOffsets = {{ {", ".join(f"{z:.6g}" for z in axis_offsets(geo["focal_distance_au"]))} }}
local axisNodes = {{}}
for i, z in ipairs(AxisOffsets) do
  axisNodes[i] = {{
    Identifier = "SGL_Axis" .. i,
    Parent = ImagePlane.Identifier,
    Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {{ 0, 0, z }} }} }},
    GUI = {{ Name = "Focal axis point " .. i, Path = "/SGL/Focal line", Hidden = true }}
  }}
end
local axisLines = {{}}
for i = 1, #AxisOffsets - 1 do
  axisLines[i] = {{
    Identifier = "SGL_FocalLine" .. i,
    Parent = sun.SolarSystemBarycenter.Identifier,
    Renderable = {{
      Type = "RenderableNodeLine",
      StartNode = axisNodes[i].Identifier,
      EndNode = axisNodes[i + 1].Identifier,
      Color = {{ 1.0, 0.66, 0.25 }},
      LineWidth = 4.5
    }},
    GUI = {{ Name = "Focal line segment " .. i, Path = "/SGL/Focal line" }}
  }}
end

local function imagePlaneLayer(id, name, texture, size, enabled, description)
  return {{
    Identifier = id,
    Parent = ImagePlane.Identifier,
    Renderable = {{
      Type = "RenderablePlaneImageLocal",
      Texture = asset.resource("textures/" .. texture),
      Size = size,
      MirrorBackside = true,
      -- Additive so the texture's black surround doesn't hide stars, samples or spacecraft.
      BlendMode = "Additive",
      Enabled = enabled
    }},
    GUI = {{ Name = name, Path = "/SGL/Image plane", Description = description }}
  }}
end

-- Size is the half-width. Colour layers combine EPIC's 680, 551 and 443 nm bands.
local Glow = imagePlaneLayer("SGL_ImageGlow", "Image-plane brightness (colour, stretched)",
  "image_plane_stretched_rgb.png", {ext_span:.3f} / 2, true,
  "What the telescopes record across the image plane with the physical SGL blur, per band.")
local GlowThesis = imagePlaneLayer("SGL_ImageGlowThesis", "Thesis PSF (too sharp)",
  "image_plane_thesis_psf_rgb.png", {ext_span:.3f} / 2, false,
  "The thesis PSF put 65% of the light in the directly imaged spot; the physical share is 0.4%.")
local Truth = imagePlaneLayer("SGL_ImageTruth", "Planet (ground truth)",
  "truth_rgb.png", ImageDiameter / 2, false,
  "DSCOVR EPIC, 13 June 2015, 680/551/443 nm. The thesis used the 443 nm band.")
local Recovered = imagePlaneLayer("SGL_ImageRecovered", "Recovered image (SNR 10)",
  "recovered_filtered_snr10_rgb.png", ImageDiameter / 2, false,
  "Deconvolved band by band from the image-plane measurements at recovered SNR ~12.")

local Samples = {{
  Identifier = "SGL_Samples",
  Parent = ImagePlane.Identifier,
  Renderable = {{
    Type = "RenderablePointCloud",
    File = asset.resource("data/formation12_samples.csv"),
    Unit = "m",
    -- cidx indexes a palette of the recorded colors (see build_asset.py, sample_palette)
    Coloring = {{
      ColorMapping = {{ File = asset.resource("data/sample_colors.cmap"), Parameter = "cidx",
        ValueRange = {{ 0, {len(palette)} }} }}
    }},
    SizeSettings = {{ ScaleExponent = 1.2 }}
  }},
  GUI = {{ Name = "Formation samples", Path = "/SGL",
    Description = "One point per stop inside the disk, in the color recorded there (680 / 551 / 443 nm)." }}
}}

-- Biosignatures over the planet's disk: vegetation (NDVI, 680 and 780 nm) and the depth of
-- the oxygen A-band (764 against 780 nm). Off by default; N steps through them.
local function bioLayer(id, name, texture, description)
  return {{
    Identifier = id,
    Parent = ImagePlane.Identifier,
    -- 1 m toward the camera so it draws over the image layers in the same plane
    Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {{ 0, 0, 1 }} }} }},
    Renderable = {{
      Type = "RenderablePlaneImageLocal",
      Texture = asset.resource("textures/" .. texture),
      Size = ImageDiameter / 2,
      MirrorBackside = true,
      BlendMode = "Normal",
      Enabled = false
    }},
    GUI = {{ Name = name, Path = "/SGL/Biosignatures", Description = description }}
  }}
end

local BioNDVITruth = bioLayer("SGL_BioNDVITruth", "Vegetation (NDVI): planet", "bio_ndvi_truth.png",
  "NDVI = (R780 - R680) / (R780 + R680) from EPIC's calibrated bands. Green: plants; tan: desert; blue: ocean; white: cloud.")
local BioNDVIRecorded = bioLayer("SGL_BioNDVIRecorded", "Vegetation (NDVI): as recorded", "bio_ndvi_recorded.png",
  "NDVI of the image-plane measurements before recovery. The blur averages it toward the whole disk's value, 0.08, the same as the Sahara's.")
local BioNDVIRecovered = bioLayer("SGL_BioNDVIRecovered", "Vegetation (NDVI): recovered, SNR 10", "bio_ndvi_snr10.png",
  "From the 680 and 780 nm bands, each recovered at SNR ~10 with the filter strength set from the noise level alone.")
local BioNDVIMission = bioLayer("SGL_BioNDVIMission", "Vegetation (NDVI): recovered, 1 m telescopes", "bio_ndvi_tt_fig4.png",
  "128 x 128 samples with 1 m telescopes at SNR 4.1 per band, the regime of Toth & Turyshev (2021) Fig. 4.")
local BioO2Truth = bioLayer("SGL_BioO2Truth", "Oxygen A-band depth: planet", "bio_o2_truth.png",
  "1 - R764 / R780. Oxygen is everywhere; the depth follows the light path: shallow over high cloud, deep toward the limb.")
local BioO2Recovered = bioLayer("SGL_BioO2Recovered", "Oxygen A-band depth: recovered, SNR 10", "bio_o2_snr10.png",
  "From the 764 and 780 nm bands, each recovered at SNR ~10.")

-- Caption for the biosignature views, hidden until N is pressed. ScreenSpaceText sizes its
-- box from the first text it shows, so it starts with the longest caption.
local BioCaption = {{
  Identifier = "SGL_BioCaption",
  Name = "Biosignature caption",
  Type = "ScreenSpaceText",
  Text = "{max((v[1] for v in BIO_VIEWS), key=len)} (N)",
  CartesianPosition = {{ 0.0, -0.56, -2.0 }},
  Scale = 1.15,
  FontSize = {intro.FONT_SIZE},
  FaceCamera = false,
  Enabled = false
}}

local BioTextures = {{
  rgb = asset.resource("textures/truth_rgb.png"),
  ndvi = asset.resource("textures/bio_ndvi_truth.png"),
  o2 = asset.resource("textures/bio_o2_truth.png")
}}

local NextBioView = {{
  Identifier = "sgl.biosignatures.next_view",
  Name = "Next biosignature view",
  -- Actions run in the global scope, so the views and texture paths are written into the command
  Command = "local tex = {{ rgb = [==[" .. BioTextures.rgb .. "]==], ndvi = [==[" .. BioTextures.ndvi
    .. "]==], o2 = [==[" .. BioTextures.o2 .. "]==] }}\\n" .. [[
    local views = {{
{chr(10).join(f'      {{ "{layer}", "{caption}", "{tex}" }},' for layer, caption, tex in BIO_VIEWS)}
    }}
    local current = 0
    for i, v in ipairs(views) do
      if openspace.propertyValue("Scene." .. v[1] .. ".Renderable.Enabled") then current = i end
    end
    local nextView = (current + 1) % (#views + 1)
    for i, v in ipairs(views) do
      openspace.setPropertyValueSingle("Scene." .. v[1] .. ".Renderable.Enabled", i == nextView)
    end
    local caption = "ScreenSpace.SGL_BioCaption."
    openspace.setPropertyValueSingle(caption .. "Enabled", nextView > 0)
    if nextView > 0 then
      local text = views[nextView][2] .. " (N)"
      openspace.setPropertyValueSingle(caption .. "Text", text)
    end
    local eyepiece = "ScreenSpace.SGL_Telescope.PlanetTexture"
    if openspace.hasProperty(eyepiece) then
      openspace.setPropertyValueSingle(eyepiece, tex[nextView > 0 and views[nextView][3] or "rgb"])
    end
  ]],
  Documentation = "Steps through the vegetation and oxygen maps: planet, as recorded, recovered, then off",
  GuiPath = "/SGL",
  IsLocal = false
}}

local nodes = {{ Frame, FocalStart, FocalEnd, ImagePlane, SunToFocal, SunDirection, SunMarker,
  SunLabel, Glow, GlowThesis, Truth, Recovered, Samples, BioNDVITruth, BioNDVIRecorded,
  BioNDVIRecovered, BioNDVIMission, BioO2Truth, BioO2Recovered }}
for _, n in ipairs(axisNodes) do table.insert(nodes, n) end
for _, n in ipairs(axisLines) do table.insert(nodes, n) end

asset.onInitialize(function()
  for _, n in ipairs(nodes) do openspace.addSceneGraphNode(n) end
  openspace.addScreenSpaceRenderable(BioCaption)
  openspace.action.registerAction(NextBioView)
  openspace.bindKey("N", NextBioView.Identifier)
end)

asset.onDeinitialize(function()
  openspace.clearKey("N")
  openspace.action.removeAction(NextBioView)
  openspace.removeScreenSpaceRenderable(BioCaption)
  for i = #nodes, 1, -1 do openspace.removeSceneGraphNode(nodes[i]) end
end)

for _, n in ipairs(nodes) do asset.export(n) end

asset.meta = {{
  Name = "Solar gravitational lens",
  Description = "Focal line, image plane, samples and biosignature maps for an SGL imaging mission at 650 AU.",
  Author = "Andrew Simon",
  License = "MIT"
}}
'''
    with open(os.path.join(OUT, "sgl.asset"), "w", encoding="utf-8") as f:
        f.write(main_asset)

    thrust, dwell = traj["thrust_m_s2"], traj["dwell_s"]
    # The formation is rigid: every spacecraft flies the first one's path, shifted by a fixed
    # offset. So the asset carries that one path and the offsets, and Lua builds the twelve.
    paths_by_sc = {sc: keyframes_for(sc_stops, dwell, thrust) for sc, sc_stops in sorted(stops.items())}
    # Lua names the spacecraft by their order, so the ids must run 0, 1, 2, ...
    assert list(paths_by_sc) == list(range(len(paths_by_sc))), sorted(paths_by_sc)
    lead = paths_by_sc[0]
    offsets = {}
    for sc, keys in paths_by_sc.items():
        offsets[sc] = keys[0][1] - lead[0][1]
        assert [t for t, _ in keys] == [t for t, _ in lead], f"spacecraft {sc} keeps different times"
        assert all(np.allclose(p, q + offsets[sc], rtol=0, atol=1e-6)
                   for (_, p), (_, q) in zip(keys, lead)), \
            f"spacecraft {sc} doesn't fly the lead path at a fixed offset"
    key_rows = ",\n".join(f'  {{ "{iso(t)}", {p[0]:.3f}, {p[1]:.3f} }}' for t, p in lead)
    offset_rows = ", ".join(f"{{ {o[0]:.3f}, {o[1]:.3f} }}" for o in offsets.values())
    formation = f'''-- Twelve spacecraft sweeping the SGL image plane in a rigid line, 120 m apart.
-- Generated by openspace/build_asset.py: bang-coast-bang reference paths. The formation is
-- rigid, so the first spacecraft's path is written once and each spacecraft adds its offset.

local sgl = asset.require("./sgl")

-- Spacecraft 1's path in the image plane (m): time, x, y
local LeadPath = {{
{key_rows}
}}

-- Each spacecraft's offset from spacecraft 1 (m), and the colors they cycle through
local Offsets = {{ {offset_rows} }}
local Colors = {{ {", ".join(lua_vec(c) for c in PALETTE)} }}

local function translation(offset)
  local keyframes = {{}}
  for _, k in ipairs(LeadPath) do
    keyframes[k[1]] = {{
      Type = "StaticTranslation",
      Position = {{ k[2] + offset[1], k[3] + offset[2], 0 }}
    }}
  end
  return {{ Type = "TimelineTranslation", Keyframes = keyframes }}
end

local nodes = {{}}
for i, offset in ipairs(Offsets) do
  local id = string.format("SGL_SC%02d", i - 1)
  local color = Colors[(i - 1) % #Colors + 1]
  local path = translation(offset)
  table.insert(nodes, {{
    Identifier = id,
    Parent = sgl.SGL_ImagePlane.Identifier,
    Transform = {{ Translation = path }},
    Renderable = {{
      Type = "RenderablePointCloud",
      File = asset.resource("data/origin.csv"),
      Unit = "m",
      Coloring = {{ FixedColor = color }},
      SizeSettings = {{ ScaleExponent = 1.4 }}
    }},
    BoundingSphere = 10,
    GUI = {{ Name = "Spacecraft " .. i, Path = "/SGL/Formation" }}
  }})
  table.insert(nodes, {{
    Identifier = id .. "_Trail",
    Parent = sgl.SGL_ImagePlane.Identifier,
    Renderable = {{
      Type = "RenderableTrailTrajectory",
      Translation = path,
      Color = color,
      StartTime = "{iso(0)}",
      EndTime = "{iso(end_s)}",
      SampleInterval = 600,
      EnableFade = false,
      LineWidth = 1.5,
      Opacity = 0.8
    }},
    GUI = {{ Name = "Spacecraft " .. i .. " trail", Path = "/SGL/Formation" }}
  }})
end

asset.onInitialize(function()
  for _, n in ipairs(nodes) do openspace.addSceneGraphNode(n) end
end)

asset.onDeinitialize(function()
  for i = #nodes, 1, -1 do openspace.removeSceneGraphNode(nodes[i]) end
end)

asset.meta = {{
  Name = "SGL formation",
  Description = "Twelve spacecraft 120 m apart sweeping the image plane on bang-coast-bang paths.",
  Author = "Andrew Simon",
  License = "MIT"
}}
'''
    with open(os.path.join(OUT, "sgl_formation.asset"), "w", encoding="utf-8") as f:
        f.write(formation)
    with open(os.path.join(OUT, "data", "origin.csv"), "w", newline="") as f:
        f.write("x,y,z\n0,0,0\n")

    profile = {
        "assets": ["base_blank", "base_keybindings", "events/toggle_sun",
                   "scene/solarsystem/sun/sun", "scene/solarsystem/sun/glare",
                   "scene/solarsystem/planets/planets", "scene/digitaluniverse/stars",
                   "${USER_ASSETS}/sgl/sgl", "${USER_ASSETS}/sgl/sgl_formation"],
        # Looking back toward the Sun from just behind the image plane, slightly off-axis:
        # the image plane, formation, focal line and the Sun at its end are all in view.
        "camera": {"type": "setNavigationState", "anchor": "SGL_ImagePlane", "aim": "",
                   "frame": "SGL_ImagePlane",
                   "position": {"x": 970.0, "y": 300.0, "z": 3000.0},
                   "up": {"x": 0.0, "y": 1.0, "z": 0.0}, "yaw": 0.0},
        # Planet orbits: thinner than OpenSpace's 10 px default, and a little transparent
        "properties": [
            {"type": "setPropertyValue", "name": "{planetTrail_solarSystem}.Renderable.Appearance.LineWidth", "value": "6.0"},
            {"type": "setPropertyValue", "name": "{planetTrail_solarSystem}.Renderable.Opacity", "value": "0.8"},
        ],
        "delta_times": [1.0, 60.0, 600.0, 3600.0, 21600.0, 86400.0],
        "time": {"type": "absolute", "value": iso(0), "is_paused": True},
        "meta": {"name": "SGL", "author": "Andrew Simon", "version": "1.0",
                 "description": "Solar gravitational lens imaging mission at 650 AU."},
        "version": {"major": 1, "minor": 5},
    }
    with open(os.path.join(OUT, "sgl.profile"), "w") as f:
        json.dump(profile, f, indent=2)

    # Layer C: the telescope view. Needs the Sgl module, so only the source build loads it.
    with open(os.path.join(OUT, "sgl_telescope.asset"), "w", encoding="utf-8") as f:
        f.write(f'''-- The view from one spacecraft's telescope, looking back at the Sun: the planet lensed
-- into an Einstein ring. Needs the Sgl module (OpenSpace built from source with
-- OPENSPACE_MODULE_SGL=ON). Generated by openspace/build_asset.py.
-- Press M to step the planet size through thin, medium and bold.

local sgl = asset.require("./sgl")
local formation = asset.require("./sgl_formation")

local Telescope = {{
  Identifier = "SGL_Telescope",
  Name = "Telescope view",
  Type = "ScreenSpaceSglTelescope",
  PlanetTexture = asset.resource("textures/truth_rgb.png"),
  TelescopeNode = "SGL_SC{TELESCOPE_SC:02d}",
  ImagePlaneNode = sgl.SGL_ImagePlane.Identifier,
  SunNode = "SunCenter",
  PlanetRadius = {tex["geometry"]["planet_diameter_m"] / 2:.1f},
  TargetDistance = {tex["geometry"]["target_distance_pc"]} * 3.0856775814913673e16,
  PlanetSize = 1,
  OcculterRadius = 1.02,
  CoronaStrength = 0.06,
  Eyepiece = true,
  Resolution = 2048,
  -- Upper left, clear of the image plane, focal line and Sun; not turned toward the camera,
  -- which would foreshorten it into an ellipse this far off-axis
  CartesianPosition = {{ -1.05, 0.42, -2.0 }},
  Scale = 0.38,
  FaceCamera = false
}}

-- Matches the module's Magnification property for the three planet sizes at 650 AU.
local Labels = {{ [0] = "22,600", [1] = "56,600", [2] = "135,700" }}
local function labelText(size)
  -- Padded to the longest label: the text box is sized from the first text it shows
  local text = "Telescope: planet x" .. Labels[size] .. " (M)"
  return text .. string.rep(" ", 30 - #text)
end

local Label = {{
  Identifier = "SGL_TelescopeLabel",
  Name = "Telescope view label",
  Type = "ScreenSpaceText",
  Text = labelText(1),
  CartesianPosition = {{ -1.05, -0.03, -2.0 }},
  Scale = 0.4,
  FontSize = {intro.FONT_SIZE},
  FaceCamera = false
}}

local NextPlanetSize = {{
  Identifier = "sgl.telescope.next_planet_size",
  Name = "Next planet size",
  Command = [[
    local uri = "ScreenSpace.SGL_Telescope.PlanetSize"
    local size = (openspace.propertyValue(uri) + 1) % 3
    openspace.setPropertyValueSingle(uri, size)
    local labels = {{ [0] = "22,600", [1] = "56,600", [2] = "135,700" }}
    local text = "Telescope: planet x" .. labels[size] .. " (M)"
    openspace.setPropertyValueSingle(
      "ScreenSpace.SGL_TelescopeLabel.Text", text .. string.rep(" ", 30 - #text)
    )
  ]],
  Documentation = "Steps the telescope view through the thin, medium and bold planet sizes",
  GuiPath = "/SGL",
  IsLocal = false
}}

asset.onInitialize(function()
  openspace.addScreenSpaceRenderable(Telescope)
  openspace.addScreenSpaceRenderable(Label)
  openspace.action.registerAction(NextPlanetSize)
  openspace.bindKey("M", NextPlanetSize.Identifier)
end)

asset.onDeinitialize(function()
  openspace.clearKey("M")
  openspace.action.removeAction(NextPlanetSize)
  openspace.removeScreenSpaceRenderable(Label)
  openspace.removeScreenSpaceRenderable(Telescope)
end)
''')
    # The reel's opening (planet, light rays, captions), hidden until the intro recording plays.
    b = geo["einstein_ring_in_solar_radii"] * intro.R_SUN
    with open(os.path.join(OUT, "sgl_intro.asset"), "w", encoding="utf-8") as f:
        f.write(intro.asset_lua(b))
    intro.draw_bend_diagram(os.path.join(OUT, "textures", "intro_bend_diagram.png"))
    intro.draw_photon_sprite(os.path.join(OUT, "textures", "intro_photon.png"))
    if intro.has_epic():
        intro.draw_planet_map(os.path.join(OUT, "textures", "intro_planet_map.png"))
    else:
        print("no EPIC granule (see layer_a/README.md): keeping the committed intro_planet_map.png")
    intro.draw_corona(os.path.join(OUT, "textures", "intro_corona.png"))

    telescope_profile = dict(profile)
    telescope_profile["assets"] = profile["assets"] + ["${USER_ASSETS}/sgl/sgl_telescope",
                                                       "${USER_ASSETS}/sgl/sgl_intro"]
    telescope_profile["meta"] = dict(profile["meta"], name="SGL telescope",
                                     description="SGL mission at 650 AU with the telescope view (source build).")
    with open(os.path.join(OUT, "sgl_telescope.profile"), "w") as f:
        json.dump(telescope_profile, f, indent=2)

    print(f"wrote {OUT}: {len(stops)} spacecraft, mission {end_s / 86400:.1f} days from {iso(0)}")
    if not args.no_install:
        dst = os.path.join(paths.assets(), "sgl")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(OUT, dst, ignore=shutil.ignore_patterns("*.profile"))
        prof_dir = paths.profiles()
        os.makedirs(prof_dir, exist_ok=True)
        for name in ("sgl.profile", "sgl_telescope.profile"):
            shutil.copy(os.path.join(OUT, name), os.path.join(prof_dir, name))
        print("installed to", dst)


if __name__ == "__main__":
    main()
