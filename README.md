# RenderedToSensor

Transforms rendered images so they look like frames captured by a real grayscale camera sensor to limit the photometric sim-to-real gap in simulations. Each image goes through an optics stage (Gaussian blur, resample to correct size, then Brown-Conrady lens distortion) and then a photometric sensor stage that applies per-pixel fixed-pattern response measured from the camera itself by collected dark and flat images, adds shot/read/row noise. Output is written as an 8-bit PNG.

The photometric response comes from **master dark and flat frames** built from real captures off the image sensor (dark = lens covered, flat = uniform illumination), rather than from synthetic noise parameters. The master flat supplies the per-pixel gain and the master dark supplies the per-pixel offset. Nothing here is specific to one sensor: this is standard dark/flat photometric calibration and works for any camera you can capture dark and flat frames from.

Requires Python 3 with `numpy` and `Pillow` (`pip install numpy pillow`).

## Files

```
sensor_profile.json              the camera: geometry, distortion, noise, scale
transform_rendered_to_sensor.py  sensor model and single image transformation
batch_transform.py               the same model over a batch of images
make_masters.py                  capture stacks, then get master dark + master flat
patch_dead_pixels.py             repairs persistent dead pixels, if necessary
examples/                        a worked run (see below)
```

Camera properties are in `sensor_profile.json`, so modelling a different sensor requires change in this file only. Both transform scripts and `make_masters.py` read it.

## Quick start

```bash
# 1. build the masters from calibration captures (once per camera)
python make_masters.py \
  --dark-folder examples/darks --flat-folder examples/flats \
  --dark-output examples/master_dark --flat-output examples/master_flat

# 2. transform a render
python transform_rendered_to_sensor.py \
  --input examples/rendered.png --output out.png \
  --master-flat examples/master_flat.npy --master-dark examples/master_dark.npy

# or a whole folder
python batch_transform.py \
  --input-dir renders/ --output-dir out/ \
  --master-flat examples/master_flat.npy --master-dark examples/master_dark.npy
```

Master frames are the expected input. If you do not have them, synthetic fixed-pattern noise is available instead with the following:

```bash
python transform_rendered_to_sensor.py --input r.png --output o.png \
  --prnu-std 0.25 --dsnu-std 0.79
```

## How does it work?

The **calibration chain** runs once per camera and turns captured dark and flat stacks into two small maps. The **transform chain** runs per image and consumes `master_dark.npy` and `master_flat.npy`.

### Calibration chain

**`patch_dead_pixels.py`** — *in:* one or more directories of 16-bit PNG captures. *out:* the same files, patched in place.

A pixel is only patched if it is an outlier in every directory passed, a defect appearing in only one is most likely not a sensor defect. Found defects are replaced by the mean of their  neighbours.

**`make_masters.py`** — *in:* the dark and flat directories, and the sensor profile. *out:* `master_dark.npy` and `master_flat.npy`.

 `master_dark.npy` is an **additive offset in DN** and represents the sensor's measured intensity when there is no scene content (no photons reaching the sensor). `master_flat.npy` is a **unitless multiplicative gain**, which peaks at 1.0, to mimic the sensor's sensitivity in each pixel. It's basically a sensitivity map of how pixels of the sensor capture a scene. 

### Transform chain

**`transform_rendered_to_sensor.py`** — *in:* a render image, a sensor profile, and the masters. *out:* one 8-bit PNG.

*Optics:* load the render as grey in 0-1, blur, scale to sensor's resolution, then apply  distortion.

*Sensor:* `convert_to_dn` scales to DN, then multiplies by flat gain, adds the dark's offset, applies Poisson shot noise, adds per pixel Gaussian read noise and row noise, then clips to scale.

`apply_master_flat` multiplies the pixel intensity in the render with the gain value in the corresponding location in the master flat.

**`batch_transform.py`** — *in:* a directory of renders. *out:* transformed PNGs with the same filenames. Does transformation on a directory of images.

## Examples

`examples/` contains the following:

- `darks/` — 25 dark frames, 16-bit grayscale.
- `flats/` — 25 flat frames, taken with a diffuser against a white screen.
- `master_dark.npy` / `master_flat.npy` — built from those 50 frames.
- `rendered.png` and `transformed.png` — one frame before and after of the rendered simulation scene.

More images should be considered for better averaging.
