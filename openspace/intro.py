"""The opening of the reel: the planet at 30 pc, its light, the bend at the Sun, the image.

Shared by build_asset.py, which writes sgl_intro.asset and the bending diagram, and by
tools/make_intro.py, which writes the camera path. Positions are in the SGL frame (metres):
the Sun at the origin, z down the focal line, so the planet sits at z = -30 pc and the image
plane at z = +650 AU.

The light drawn is real geometry: rays leave the planet, pass the Sun at the Einstein
radius for 650 AU (1.09 solar radii) and meet on the axis at the image plane. The bend is
1.6 arcseconds, too small to see from the side, so the diagram shown during the overview
exaggerates it and says so.
"""
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

import paths

LAYER_A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "layer_a")

PC = 3.0856775814913673e16
AU = 1.495978707e11
LY = 9.4607304725808e15
C = 299792458.0
R_SUN = 6.957e8
R_EARTH = 6.3781e6

TARGET_PC = 30.0
IMAGE_AU = 650.0
N_RAYS = 8
HERO_ANGLE = math.pi / 2        # the followed ray passes over the Sun's +y limb
# The camera's rig on the light pulse (make_intro.py): behind it and lifted off the axis, in
# units of the camera distance, which is D_PLANET as the pulse leaves the planet
RIG_BACK, RIG_LIFT = 0.94, 0.35
PULSE_START = 1.05 * R_EARTH    # the pulse sets off just above the planet's surface (m)
D_PLANET = 3.5e7                # camera distance behind it then (m)
# The globe's day side faces the camera where the flight ends, turned this far toward the Sun,
# so the close-up is gibbous: a soft shadow edge reads as a sphere, a fully lit disk as flat
GLOBE_TILT = math.radians(30)

# (start time in s, text). Centered by padding, so the text box is sized for the longest.
CAPTIONS = {
    "overview": "The Sun's gravity bends light from a planet 30 parsecs away to a focus 550-1,000 AU beyond",
    "planet": "The planet: an Earth twin 98 light-years away (the real Earth, imaged by DSCOVR EPIC)",
    "travel": "Its light crosses 30 parsecs to the Sun, a 98-year trip",
    "bend": "It skims 0.09 solar radii above the Sun's surface and bends by 1.6 arcseconds",
    "converge": "Light bent around the whole rim of the Sun meets on the axis 650 AU out",
    "image": "There it forms the planet's image, 1.3 km wide, swept by twelve spacecraft",
}
CLOCK_WIDTH = 58                # characters; the clock text is padded to this
LABELS = {"planet": "Earth-like planet", "sun": "Sun", "distance": "30 parsecs = 98 light-years"}

# The flat clip is rendered at SSAA times its output size and scaled down when encoded, which
# smooths every edge, line and letter (with 4x MSAA instead, OpenSpace captured black frames).
OUTPUT = (1920, 1080)
SSAA = 2
# Screen-space text is drawn at FONT_SIZE and then scaled to its Scale; 60 keeps the captions
# and labels at least as sharp as the frames they land in.
FONT_SIZE = 60
CORONA_REACH = 4.0              # the Sun's glow texture spans this many solar radii from the center
TEXT_BAND = 0.4                # opacity of the dark band behind the caption and the clock

# The overview's layout in screen-space units (z = -2). The flat frames (hfov 80°, vfov 50.534°)
# span x within ±2 tan 40° and y within ±2 tan 25.267°.
WINDOW = (OUTPUT[0] * SSAA, OUTPUT[1] * SSAA)                     # rendered frames
SCREEN = (2 * math.tan(math.radians(40.0)), 2 * math.tan(math.radians(50.534015846724 / 2)))
PX = (WINDOW[0] / (2 * SCREEN[0]), WINDOW[1] / (2 * SCREEN[1]))     # pixels per unit
SUN_XY = (0.35, -0.28)          # where the overview holds the Sun, under the diagram
PANEL_X, PANEL_TOP = (-0.9, 0.9), 0.91                             # diagram panel edges
BOX = (0.15, 0.06)              # half-width and half-height of the box around the Sun


def planet_pos():
    return (0.0, 0.0, -TARGET_PC * PC)


def image_pos():
    return (0.0, 0.0, IMAGE_AU * AU)


def close_pass_direction(b):
    """Unit vector from the planet to the camera where the opening's flight ends, just
    behind the light pulse as it leaves the surface. The globe turns its day side this way."""
    import numpy as np
    planet, limb = np.array(planet_pos()), np.array(limb_points(b)[0])
    f1, n_out = (limb - planet) / np.linalg.norm(limb - planet), limb / np.linalg.norm(limb)
    rel = f1 * PULSE_START + D_PLANET * (-RIG_BACK * f1 + RIG_LIFT * n_out)
    return rel / np.linalg.norm(rel)


def limb_points(b):
    """Where each drawn ray passes the Sun, b metres from the axis (ray 0 is followed)."""
    return [(b * math.cos(HERO_ANGLE + 2 * math.pi * k / N_RAYS),
             b * math.sin(HERO_ANGLE + 2 * math.pi * k / N_RAYS), 0.0) for k in range(N_RAYS)]


def pad(text, width):
    """Center text in a fixed-width box (ScreenSpaceText sizes its box from its first text)."""
    left = (width - len(text)) // 2
    return " " * left + text + " " * (width - len(text) - left)


def caption_width():
    return max(len(t) for t in CAPTIONS.values())


def dome_caption_lines(text):
    """A caption split into two lines of about equal length, for the dome."""
    words = text.split()
    # never between a number and its unit ("1.3 km")
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))
              if not any(c.isdigit() for c in words[i - 1]) or any(c.isdigit() for c in words[i])]
    return min(splits, key=lambda ab: max(len(ab[0]), len(ab[1])))


def dome_caption_width():
    return max(len(line) for t in CAPTIONS.values() for line in dome_caption_lines(t))


def lua_vec(v):
    return "{ " + ", ".join(f"{float(a):.17g}" for a in v) + " }"


def panel_bottom():
    """Bottom edge of the diagram panel, which keeps the diagram's 960 x 380 shape on screen."""
    width_px = (PANEL_X[1] - PANEL_X[0]) * PX[0]
    return PANEL_TOP - width_px * 380 / 960 / PX[1]


def overlay_rect():
    """Edges (x0, x1, y0, y1) of the overview overlay: the diagram, its lines and the box."""
    m = 0.01                    # room for the strokes
    return (min(PANEL_X[0], SUN_XY[0] - BOX[0]) - m, max(PANEL_X[1], SUN_XY[0] + BOX[0]) + m,
            SUN_XY[1] - BOX[1] - m, PANEL_TOP + m)


def asset_lua(b):
    """sgl_intro.asset: everything starts hidden; the intro recording switches it on."""
    import numpy as np
    limbs = limb_points(b)
    # The globe's day side faces the camera at the close pass, turned GLOBE_TILT toward the Sun,
    # with north toward the camera's up (+y). The texture's day side is centered on local -x,
    # so the rotation's columns (local x, y, z in the SGL frame) are -day, day x north and north.
    cam, sun = close_pass_direction(b), -np.array(planet_pos()) / np.linalg.norm(planet_pos())
    toward_sun = sun - sun.dot(cam) * cam
    day = math.cos(GLOBE_TILT) * cam + math.sin(GLOBE_TILT) * toward_sun / np.linalg.norm(toward_sun)
    north = np.array([0.0, 1.0, 0.0]) - day[1] * day
    north /= np.linalg.norm(north)
    columns = np.concatenate([-day, np.cross(day, north), north])
    globe_rotation = ", ".join(f"{v:.12f}" for v in np.where(abs(columns) < 1e-12, 0.0, columns))
    x0, x1, y0, y1 = overlay_rect()
    limb_nodes = ",\n".join(
        f'''  {{ Identifier = "SGL_RayLimb{k}", Parent = sgl.SGL_Frame.Identifier,
    Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {lua_vec(p)} }} }},
    GUI = {{ Name = "Ray {k} at the limb", Path = "/SGL/Intro", Hidden = true }} }}'''
        for k, p in enumerate(limbs))
    width = caption_width()
    return f'''-- The opening of the reel: the planet at 30 pc, its light, the bend at the Sun, the image.
-- Generated by openspace/build_asset.py (see intro.py). Everything starts hidden;
-- tools/make_intro.py writes a recording that switches it on and flies the camera.

local sgl = asset.require("./sgl")
local sun = asset.require("scene/solarsystem/sun/transforms")
-- The Sun's textured globe, which the intro switches on for the close pass (glare off)
asset.require("scene/solarsystem/sun/layers/colorlayers/sun_texture")

local PC = {PC!r}

-- The camera anchors here; the globe is a child, so turning it doesn't turn the anchor frame
local Planet = {{
  Identifier = "SGL_IntroPlanet",
  Parent = sgl.SGL_Frame.Identifier,
  Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {lua_vec(planet_pos())} }} }},
  BoundingSphere = {R_EARTH},
  InteractionSphere = {R_EARTH},
  GUI = {{ Name = "Exoplanet (30 pc)", Path = "/SGL/Intro",
    Description = "An Earth twin at 30 pc, drawn with the DSCOVR EPIC Earth." }}
}}

-- The EPIC hemisphere faces the camera where the flight ends, turned 30 degrees toward the Sun,
-- as if the planet's own star were behind and above the camera; the far side is night.
local Globe = {{
  Identifier = "SGL_IntroGlobe",
  Parent = Planet.Identifier,
  Transform = {{
    Rotation = {{ Type = "StaticRotation", Rotation = {{ {globe_rotation} }} }}
  }},
  Renderable = {{
    Type = "RenderableSphereImageLocal",
    Texture = asset.resource("textures/intro_planet_map.png"),
    Size = {R_EARTH},
    Segments = 192,
    Orientation = "Outside",
    DisableFadeInOut = true,
    Enabled = false
  }},
  GUI = {{ Name = "Exoplanet globe", Path = "/SGL/Intro" }}
}}

local function marker(id, name, parent, position, color)
  return {{
    Identifier = id,
    Parent = parent,
    Transform = {{ Translation = {{ Type = "StaticTranslation", Position = position }} }},
    Renderable = {{
      Type = "RenderablePointCloud",
      File = asset.resource("data/origin.csv"),
      Unit = "m",
      Coloring = {{ FixedColor = color }},
      -- Big enough to see from parsecs away, capped so it stays a dot up close
      SizeSettings = {{ ScaleExponent = 16.0, EnableMaxSizeControl = true, MaxSize = 0.6 }},
      Enabled = false
    }},
    GUI = {{ Name = name, Path = "/SGL/Intro" }}
  }}
end

-- RenderableLabel goes through 32-bit world positions, which are ~5e10 m off at 30 pc,
-- so the names are screen-space text that make_intro.py places over the right spot.
local function label(id, text)
  return {{
    Identifier = id,
    Name = text,
    Type = "ScreenSpaceText",
    Text = text,
    CartesianPosition = {{ 0.0, 0.0, -2.0 }},
    Scale = 0.018 * #text,
    FontSize = {FONT_SIZE},
    FaceCamera = false,
    Enabled = false
  }}
end

local PlanetDot = marker("SGL_IntroPlanetDot", "Exoplanet marker", Planet.Identifier,
  {{ 0, 0, 0 }}, {{ 0.55, 0.78, 1.0 }})
local SunDot = marker("SGL_IntroSunDot", "Sun marker (far)", sgl.SGL_Frame.Identifier,
  {{ 0, 0, 0 }}, {{ 1.0, 0.84, 0.5 }})
local PlanetLabel = label("SGL_IntroPlanetLabel", "{LABELS["planet"]}")
local SunLabel = label("SGL_IntroSunLabel", "{LABELS["sun"]}")
local DistanceLabel = label("SGL_IntroDistanceLabel", "{LABELS["distance"]}")

-- The light pulse, a glowing billboard (point clouds don't draw reliably 30 pc out).
-- make_intro.py moves it and sizes it every frame.
local Photon = {{
  Identifier = "SGL_Photon",
  Parent = sgl.SGL_Frame.Identifier,
  Transform = {{ Translation = {{ Type = "StaticTranslation", Position = {lua_vec(planet_pos())} }} }},
  Renderable = {{
    Type = "RenderablePlaneImageLocal",
    Texture = asset.resource("textures/intro_photon.png"),
    Size = 1.0e6,
    Billboard = true,
    BlendMode = "Additive",
    Enabled = false
  }},
  BoundingSphere = 1.0,
  InteractionSphere = 1.0,
  GUI = {{ Name = "Light pulse", Path = "/SGL/Intro" }}
}}

-- A glow around the textured Sun for the close pass. It sits at the Sun's center facing the
-- camera, so the globe hides the part over its disk and only the ring past the limb shows.
local Corona = {{
  Identifier = "SGL_IntroCorona",
  Parent = sun.SunCenter.Identifier,
  Renderable = {{
    Type = "RenderablePlaneImageLocal",
    Texture = asset.resource("textures/intro_corona.png"),
    Size = {CORONA_REACH * R_SUN},
    Billboard = true,
    BlendMode = "Additive",
    Enabled = false
  }},
  GUI = {{ Name = "Sun glow (close pass)", Path = "/SGL/Intro" }}
}}

-- Rays: planet to the limb, limb to the image plane. Line nodes sit on the barycenter (see
-- sgl.asset); the camera's anchor stays near the camera, which keeps them precise.
local LimbNodes = {{
{limb_nodes}
}}
local Rays = {{}}
for k, limb in ipairs(LimbNodes) do
  local hero = (k == 1)
  for _, leg in ipairs({{ {{ "In", Planet.Identifier, limb.Identifier }},
                          {{ "Out", limb.Identifier, sgl.SGL_ImagePlane.Identifier }} }}) do
    table.insert(Rays, {{
      Identifier = "SGL_Ray" .. leg[1] .. (k - 1),
      Parent = sun.SolarSystemBarycenter.Identifier,
      Renderable = {{
        Type = "RenderableNodeLine",
        StartNode = leg[2],
        EndNode = leg[3],
        Color = hero and {{ 1.0, 0.95, 0.75 }} or {{ 0.95, 0.85, 0.55 }},
        Opacity = hero and 0.8 or 0.45,
        LineWidth = hero and 2.0 or 1.4,
        Enabled = false
      }},
      GUI = {{ Name = "Ray " .. (k - 1) .. " " .. leg[1], Path = "/SGL/Intro", Hidden = true }}
    }})
  end
end

local Caption = {{
  Identifier = "SGL_IntroCaption",
  Name = "Intro caption",
  Type = "ScreenSpaceText",
  Text = "{pad(max(CAPTIONS.values(), key=len), width)}",
  CartesianPosition = {{ 0.0, -0.8, -2.0 }},
  Scale = 1.45,
  FontSize = {FONT_SIZE},
  -- A translucent band keeps the text legible over the Earth and the Sun
  BackgroundColor = {{ 0.0, 0.0, 0.0, {TEXT_BAND} }},
  FaceCamera = false,
  Enabled = false
}}

local Clock = {{
  Identifier = "SGL_IntroClock",
  Name = "Intro clock",
  Type = "ScreenSpaceText",
  Text = "{" " * CLOCK_WIDTH}",
  CartesianPosition = {{ 0.0, 0.84, -2.0 }},
  Scale = 0.85,
  FontSize = {FONT_SIZE},
  BackgroundColor = {{ 0.0, 0.0, 0.0, {TEXT_BAND} }},
  FaceCamera = false,
  Enabled = false
}}

local Diagram = {{
  Identifier = "SGL_IntroDiagram",
  Name = "How the Sun bends light (not to scale)",
  Type = "ScreenSpaceImageLocal",
  TexturePath = asset.resource("textures/intro_bend_diagram.png"),
  -- The diagram with lines down to a box around the Sun, which the overview holds still
  CartesianPosition = {{ {(x0 + x1) / 2:.6f}, {(y0 + y1) / 2:.6f}, -2.0 }},
  Scale = {(x1 - x0) / 2:.6f},
  FaceCamera = false,
  Enabled = false
}}

local nodes = {{ Planet, Globe, PlanetDot, SunDot, Photon, Corona }}
for _, n in ipairs(LimbNodes) do table.insert(nodes, n) end
for _, n in ipairs(Rays) do table.insert(nodes, n) end
-- On the dome a full-width caption would be too small to read, so it's split over two
-- lines, each its own text sized to the longest line (make_intro.py --dome uses these).
local function domeCaption(id)
  return {{
    Identifier = id,
    Name = "Intro caption (dome)",
    Type = "ScreenSpaceText",
    Text = "{pad(max((line for t in CAPTIONS.values() for line in dome_caption_lines(t)), key=len), dome_caption_width())}",
    CartesianPosition = {{ 0.0, -2.2, -2.0 }},
    Scale = 1.0,
    FontSize = {FONT_SIZE},
    BackgroundColor = {{ 0.0, 0.0, 0.0, {TEXT_BAND} }},
    FaceCamera = true,
    Enabled = false
  }}
end
local DomeCaption1 = domeCaption("SGL_IntroDomeCaption1")
local DomeCaption2 = domeCaption("SGL_IntroDomeCaption2")

local screens = {{ Caption, Clock, Diagram, PlanetLabel, SunLabel, DistanceLabel, DomeCaption1, DomeCaption2 }}

asset.onInitialize(function()
  for _, n in ipairs(nodes) do openspace.addSceneGraphNode(n) end
  for _, s in ipairs(screens) do openspace.addScreenSpaceRenderable(s) end
end)

asset.onDeinitialize(function()
  for i = #screens, 1, -1 do openspace.removeScreenSpaceRenderable(screens[i]) end
  for i = #nodes, 1, -1 do openspace.removeSceneGraphNode(nodes[i]) end
end)
'''


def draw_photon_sprite(path, n=128):
    """Soft white-gold glow for the light pulse, on black for additive blending."""
    import numpy as np
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) / (n / 2)
    core = np.exp(-(r / 0.18) ** 2)
    halo = 0.55 * np.exp(-(r / 0.5) ** 2) * (r < 1)
    rgb = np.stack([core + halo, core + 0.92 * halo, core + 0.65 * halo], axis=-1)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path)


def has_epic():
    """Whether layer A's EPIC granule is on disk, which draw_planet_map needs."""
    sys.path.insert(0, LAYER_A)
    import make_textures as T
    return os.path.isfile(T.EPIC_H5)


def planet_disk(n):
    """The EPIC Earth as an n x n disk filling the square, with the reel's display stretch
    (its textures are 256 px) and white-balanced on its brightest clouds. Zero outside."""
    import numpy as np
    sys.path.insert(0, LAYER_A)
    import make_textures as T
    disk = np.dstack(T.load_bands(n, color=True)[0])
    # Match the reel's 256 px planet in overall brightness (finer pixels keep higher peaks,
    # which shifts the normalization a little).
    coarse = np.dstack(T.load_bands(256, color=True)[0])
    binned = disk.reshape(256, n // 256, 256, n // 256, 3).mean(axis=(1, 3))
    disk = disk * coarse.sum() / binned.sum()
    # White-balance on the brightest clouds: the bands' raw scales tint them mint
    p = np.percentile(disk[disk.sum(axis=2) > 0], 99.5, axis=0)
    return np.clip(disk * (p.max() / p) * T.RGB_GAIN, 0, 1)


def draw_planet_map(path, width=4096, n=2048):
    """The planet's globe texture: the EPIC disk projected back onto the hemisphere that faces
    its star (the day side), with the far side in night. Equirectangular, as
    RenderableSphereImageLocal maps it: longitude from local +x toward +y, latitude down from
    the north pole at local +z. The day side is centered on local -x, half a turn from the
    texture's seam."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    disk = planet_disk(n)
    h = width // 2
    theta, phi = np.meshgrid(np.pi * (np.arange(h) + 0.5) / h, 2 * np.pi * (np.arange(width) + 0.5) / width,
                             indexing="ij")
    x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    # Seen from -x with north up, the disk's right is -y and its top is +z
    rows, cols = (n - 1) / 2 - z * n / 2, (n - 1) / 2 - y * n / 2
    rgb = np.dstack([map_coordinates(disk[..., c], [rows, cols], order=1) for c in range(3)])
    # EPIC's limb is bright with haze, so the light fades out where its star is within 16
    # degrees of the horizon, or the terminator is a hard cut when the planet is seen from the
    # side. The fade reaches zero 5.7 degrees up, which also drops EPIC's last few pixels (they
    # fade into space and fringe red where the bands don't quite overlay) and the night side.
    soft = np.clip((-x - 0.1) / 0.18, 0, 1)
    rgb *= (soft * soft * (3 - 2 * soft))[..., None]
    Image.fromarray((rgb * 255).round().astype(np.uint8), mode="RGB").save(path)


def draw_corona(path, n=1024):
    """Glow past the Sun's limb, on black for additive blending: bright at the limb and fading
    over about a solar radius. Dark over the disk itself, which the globe covers anyway."""
    import numpy as np
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) / (n / 2) * CORONA_REACH   # solar radii
    out = 0.75 * np.exp(-np.abs(r - 1) / 0.12) + 0.3 * np.exp(-np.abs(r - 1) / 0.7)
    out *= np.clip((r - 0.94) / 0.06, 0, 1) * np.clip((CORONA_REACH - r) / 0.5, 0, 1)
    rgb = out[..., None] * np.array([1.0, 0.82, 0.55])
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8), mode="RGB").save(path)


def bend_diagram(s):
    """Side view of rays bending past the Sun, exaggerated and labelled as such: a 960 x 380
    layout drawn at s pixels per layout unit."""
    w, h = 960, 380
    img = Image.new("RGBA", (round(w * s), round(h * s)), (8, 10, 16, 190))
    d = ImageDraw.Draw(img)
    S = lambda *v: tuple(a * s for a in v)
    W = lambda px: max(1, round(px * s))
    # OpenSpace's own "Mono" font, so the labels match the captions and names around the panel
    font = ImageFont.truetype(paths.mono_font(), round(20 * s))
    small = ImageFont.truetype(paths.mono_font(), round(17 * s))
    axis_y, sun_x, sun_r = 196, 230, 46
    d.line(S(30, axis_y, w - 30, axis_y), fill=(110, 110, 120, 200), width=W(1))
    # Rays at impact parameter h (pixels) cross the axis at a distance growing as h^2.
    focus = lambda hh: sun_x + 330 * (hh / sun_r) ** 2
    for hh, hero in ((sun_r + 2, False), (sun_r * 1.09, True), (sun_r * 1.28, False), (sun_r * 1.5, False)):
        xf = focus(hh)
        color = (255, 244, 200, 255) if hero else (240, 215, 140, 150)
        width = W(3 if hero else 2)
        for sign in (-1, 1):
            y0 = axis_y + sign * hh
            d.line(S(30, y0, sun_x, y0), fill=color, width=width)
            end_x = w - 30
            slope = (axis_y - y0) / (xf - sun_x)
            d.line(S(sun_x, y0, end_x, y0 + slope * (end_x - sun_x)), fill=color, width=width)
    # Focal line from the limb-grazing focus outward, and the image plane.
    x548, x650 = focus(sun_r + 2), focus(sun_r * 1.09)
    d.line(S(x548, axis_y, w - 30, axis_y), fill=(255, 168, 64, 255), width=W(5))
    d.line(S(x650, axis_y - 70, x650, axis_y + 70), fill=(150, 230, 210, 255), width=W(3))
    d.ellipse(S(sun_x - sun_r, axis_y - sun_r, sun_x + sun_r, axis_y + sun_r), fill=(255, 205, 100, 255))
    d.text(S(sun_x, axis_y + sun_r + 32), "Sun", fill=(255, 225, 160), font=font, anchor="ma")
    d.text(S(40, 22), "light from the planet", fill=(240, 225, 170), font=font)
    # Below the lowest rays: the monospaced label is too wide to fit between them
    d.text(S(x548 - 10, axis_y + 70), "focal line starts, 548 AU", fill=(255, 190, 120), font=small, anchor="ra")
    d.text(S(x650 + 8, axis_y - 98), "image plane, 650 AU", fill=(170, 240, 220), font=small)
    d.text(S(w // 2, h - 30), "Not to scale: the real bend is 1.6 arcseconds (0.00045 degrees)",
           fill=(200, 200, 205), font=small, anchor="ma")
    return img


def draw_bend_diagram(path, ss=2):
    """The overview overlay: the diagram in a square-cornered panel, joined by two lines from
    its bottom corners to the top corners of a box around the Sun. Drawn at the rendered
    frames' own pixel size, supersampled ss times."""
    x0, x1, y0, y1 = overlay_rect()
    w_px, h_px = (x1 - x0) * PX[0], (y1 - y0) * PX[1]
    img = Image.new("RGBA", (round(w_px * ss), round(h_px * ss)), (0, 0, 0, 0))
    at = lambda x, y: ((x - x0) * PX[0] * ss, (y1 - y) * PX[1] * ss)
    left, top = at(PANEL_X[0], PANEL_TOP)
    right, bottom = at(PANEL_X[1], panel_bottom())
    img.alpha_composite(bend_diagram((right - left) / 960), (round(left), round(top)))
    box_left, box_top = at(SUN_XY[0] - BOX[0], SUN_XY[1] + BOX[1])
    box_right, box_bottom = at(SUN_XY[0] + BOX[0], SUN_XY[1] - BOX[1])
    d = ImageDraw.Draw(img)
    color, width = (165, 170, 182, 230), 2 * SSAA * ss          # 2 px in the encoded clip
    for l, t, r, b in ((left, top, right, bottom), (box_left, box_top, box_right, box_bottom)):
        # Strokes centered on the edges, so the lines meet the corners exactly
        d.rectangle((l - width / 2, t - width / 2, r + width / 2, b + width / 2), outline=color, width=width)
    d.line((left, bottom, box_left, box_top), fill=color, width=width)
    d.line((right, bottom, box_right, box_top), fill=color, width=width)
    # The texture's shape has to match its size in units; OpenSpace stretches it back to pixels.
    w = round(w_px)
    img.resize((w, round(w * (y1 - y0) / (x1 - x0))), Image.LANCZOS).save(path)
