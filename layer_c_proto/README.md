# Layer C prototype: the telescope view in NumPy

The math the eyepiece runs on the GPU, written first in NumPy so it could be checked against independent physics. [`ring_view.py`](ring_view.py) traces each pixel through the point-mass lens equation, and the shader [`telescope_fs.glsl`](../openspace/module_sgl/shaders/telescope_fs.glsl) is a port of it.

## Checks

Both run from committed files, without OpenSpace, and exit with an error if a result is out of tolerance.

| Command | What it checks | Result |
|---|---|---|
| `python validate.py` | Ring radius from a point source vs θ<sub>E</sub> at 650 AU | 1.6076″ vs 1.6076″ |
| | Integrated ring light vs the wave-optics image plane from layer A, at 81 telescope positions, with flipped copies as controls | r = 1.0000, ratio spread 0.2%; flipped copies fit 23–35× worse |
| | Ring light divided by the planet's exaggeration, at three exaggerations | varies by only 0.18–0.32% |
| `python ../openspace/tools/compare_gpu_frame.py` | The Sgl module's GPU frame (`out/gpu_frame_c_occulter.png`) vs this reference (`out/visuals/c_occulter.png`), with mirrored and rotated copies as controls | r = 0.9992, mean difference 1.2/255 on the ring; controls r ≈ 0.66 |

`validate.py` writes its numbers to [`out/validation.json`](out/validation.json). `compare_gpu_frame.py --capture out.png` takes a fresh frame from a running OpenSpace first.

## Renders

[`render_options.py`](render_options.py) writes `out/visuals/`: the reference images above, a sequence as a telescope crosses the image plane, and the design study behind the eyepiece's look (planet exaggeration, optics, Sun treatment and inset layouts).
