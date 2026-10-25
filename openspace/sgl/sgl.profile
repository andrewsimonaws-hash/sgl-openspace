{
  "assets": [
    "base_blank",
    "base_keybindings",
    "events/toggle_sun",
    "scene/solarsystem/sun/sun",
    "scene/solarsystem/sun/glare",
    "scene/solarsystem/planets/planets",
    "scene/digitaluniverse/stars",
    "${USER_ASSETS}/sgl/sgl",
    "${USER_ASSETS}/sgl/sgl_formation"
  ],
  "camera": {
    "type": "setNavigationState",
    "anchor": "SGL_ImagePlane",
    "aim": "",
    "frame": "SGL_ImagePlane",
    "position": {
      "x": 970.0,
      "y": 300.0,
      "z": 3000.0
    },
    "up": {
      "x": 0.0,
      "y": 1.0,
      "z": 0.0
    },
    "yaw": 0.0
  },
  "delta_times": [
    1.0,
    60.0,
    600.0,
    3600.0,
    21600.0,
    86400.0
  ],
  "time": {
    "type": "absolute",
    "value": "2075-01-01T00:00:00",
    "is_paused": true
  },
  "meta": {
    "name": "SGL",
    "author": "Andrew Simon",
    "version": "1.0",
    "description": "Solar gravitational lens imaging mission at 650 AU."
  },
  "version": {
    "major": 1,
    "minor": 5
  }
}