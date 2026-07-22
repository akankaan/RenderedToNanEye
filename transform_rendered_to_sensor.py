"""Turn a rendered image into a sensor-style image

Optics stage (blur, resample, Brown-Conrady distortion) followed by a photometric
sensor stage that uses a sensitivity map and master dark made by make_masters.py

"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_PROFILE = "sensor_profile.json"

# Treats 99.5th percentile as the peak so small inconsistencies don't change the scale
PEAK_PERCENTILE = 99.5


@dataclass(frozen=True)
class SensorProfile:
    """Everything about the camera that is not a calibration frame"""

    name: str
    width: int
    height: int
    full_scale_dn: float
    source_white_level: float
    blur_radius: float
    seed: int
    read_noise_dn: float
    row_noise_std: float
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        intrinsics = raw.get("intrinsics", {})
        distortion = raw.get("distortion", {})
        try:
            return cls(
                name=raw.get("name", Path(path).stem),
                width=int(raw["width"]),
                height=int(raw["height"]),
                full_scale_dn=float(raw["full_scale_dn"]),
                source_white_level=float(raw["source_white_level"]),
                blur_radius=float(raw["blur_radius"]),
                seed=int(raw["seed"]),
                read_noise_dn=float(raw["read_noise_dn"]),
                row_noise_std=float(raw["row_noise_std"]),
                fx_px=float(intrinsics["fx_px"]),
                fy_px=float(intrinsics["fy_px"]),
                cx_px=float(intrinsics["cx_px"]),
                cy_px=float(intrinsics["cy_px"]),
                k1=float(distortion["k1"]),
                k2=float(distortion["k2"]),
                p1=float(distortion["p1"]),
                p2=float(distortion["p2"]),
                k3=float(distortion["k3"]),
            )
        except KeyError as exc:
            raise ValueError(f"Sensor profile {path} is missing required key {exc}.") from exc

    @property
    def shape(self):
        return (self.height, self.width)


def load_image_linear(path):
    # Load as grayscale 
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def bilinear_interpolate(image, x, y):
    h, w = image.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = x - x0
    dy = y - y0

    return (
        image[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + image[y0, x1] * dx * (1.0 - dy)
        + image[y1, x0] * (1.0 - dx) * dy
        + image[y1, x1] * dx * dy
    )


def inverse_radtan_map(x_dist, y_dist, k1, k2, p1, p2, k3=0.0, iterations=8):
    # Invert Brown-Conrady (OpenCV/Kalibr radtan) in normalized image coords
    x = x_dist.copy()
    y = y_dist.copy()

    for _ in range(iterations):
        r2 = x**2 + y**2
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        radial = np.where(np.abs(radial) < 1e-6, 1e-6, radial)

        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x**2)
        delta_y = p1 * (r2 + 2.0 * y**2) + 2.0 * p2 * x * y

        x = (x_dist - delta_x) / radial
        y = (y_dist - delta_y) / radial

    return x, y


def apply_optics(gray, profile):
    # Blur, resample to the sensor's resolution, then distort
    if profile.fx_px <= 0 or profile.fy_px <= 0:
        raise ValueError("Intrinsics fx_px/fy_px must be positive.")

    img = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=profile.blur_radius))
    img = img.resize((profile.width, profile.height), Image.Resampling.BICUBIC)

    gray = np.array(img, dtype=np.float32) / 255.0
    h, w = gray.shape

    y, x = np.indices((h, w), dtype=np.float32)
    x_dist = (x - profile.cx_px) / profile.fx_px
    y_dist = (y - profile.cy_px) / profile.fy_px

    # Inverse map so each output pixel samples the correct undistorted source
    x_undist, y_undist = inverse_radtan_map(
        x_dist, y_dist, profile.k1, profile.k2, profile.p1, profile.p2, profile.k3
    )

    return bilinear_interpolate(
        gray,
        x_undist * profile.fx_px + profile.cx_px,
        y_undist * profile.fy_px + profile.cy_px,
    )


def convert_to_dn(gray, full_scale_dn):
    # Convert normalized intensity to sensor DN
    return gray * full_scale_dn


def generate_fixed_pattern_maps(shape, prnu_std, dsnu_std, rng):
    # Fixed per-pixel maps to use across all frames
    prnu_map = rng.normal(0, prnu_std, shape).astype(np.float32) if prnu_std else None
    dsnu_map = rng.normal(0, dsnu_std, shape).astype(np.float32) if dsnu_std else None
    return prnu_map, dsnu_map


def apply_prnu(signal_dn, prnu_std, rng, prnu_map=None):
    if prnu_map is None:
        prnu_map = rng.normal(0, prnu_std, signal_dn.shape)
    return signal_dn * (1 + prnu_map)


def apply_shot_noise(signal_dn, rng):
    # Shot noise modeled by a Poisson distribution
    return rng.poisson(np.clip(signal_dn, 0, None)).astype(np.float32)


def apply_dark_current(signal_dn, dsnu_std, rng, dark_current_mean=0.0, dsnu_map=None):
    # Dark current fixed-pattern noise plus dark shot noise
    if dsnu_map is None:
        dsnu_map = rng.normal(0, dsnu_std, signal_dn.shape)
    dark_signal = np.clip(dark_current_mean + dsnu_map, 0, None)
    return signal_dn + rng.poisson(dark_signal).astype(np.float32)


def apply_readout_noise(signal_dn, read_noise_dn, row_noise_std, rng):
    # Signal independent
    noisy = signal_dn + rng.normal(0, read_noise_dn, signal_dn.shape)
    noisy += rng.normal(0, row_noise_std, (signal_dn.shape[0], 1))
    return noisy


def apply_sensitivity_map(signal_dn, sensitivity_map):
    # Scales to peak intensity so applying the gain attenuates and not increases intensity
    # in the centre
    gain = np.clip(sensitivity_map, 1e-6, None)
    peak = float(np.percentile(gain, PEAK_PERCENTILE))
    return signal_dn * (gain / max(peak, 1e-6))


def adc_quantize(signal_dn, full_scale_dn):
    # Clip to full scale and quantize to integer DN
    signal_dn = np.clip(signal_dn, 0, full_scale_dn)
    return np.round(signal_dn) / full_scale_dn


def save_image(out, output_path):
    Image.fromarray((out * 255).astype(np.uint8), mode="L").save(output_path)


def load_master_npy(path):
    return np.load(path).astype(np.float32)


def load_calibration(profile, sensitivity_path=None, dark_path=None):
    """Load and sanity-check calibration maps. They must come from make_masters.py."""
    sensitivity_map = load_master_npy(sensitivity_path) if sensitivity_path else None
    master_dark = load_master_npy(dark_path) if dark_path else None

    for label, arr in (("Sensitivity map", sensitivity_map), ("Master dark", master_dark)):
        if arr is not None and arr.shape != profile.shape:
            raise ValueError(
                f"{label} shape {arr.shape} does not match the profile's "
                f"sensor shape {profile.shape}."
            )

    if sensitivity_map is not None:
        peak = float(np.percentile(sensitivity_map, PEAK_PERCENTILE))
        if not 0.5 <= peak <= 2.0:
            raise ValueError(
                f"Sensitivity map does not look peak-normalized (p{PEAK_PERCENTILE} = {peak:.3g}, "
                "expected about 1.0). Rebuild it with make_masters.py."
            )

    if master_dark is not None and float(master_dark.max()) > profile.full_scale_dn * 1.5:
        raise ValueError(
            f"Master dark peaks at {master_dark.max():.0f}, well above the profile's "
            f"full_scale_dn of {profile.full_scale_dn:.0f}, so it is not in sensor DN. "
            "Rebuild it with make_masters.py."
        )

    return sensitivity_map, master_dark


def transform_render(
    input_path,
    profile,
    rng,
    sensitivity_map=None,
    master_dark=None,
    prnu_std=None,
    dsnu_std=None,
    prnu_map=None,
    dsnu_map=None,
):
    """Run one render through the optics and sensor stages. Returns 0..1 floats."""
    gray = apply_optics(load_image_linear(input_path), profile)
    signal_dn = convert_to_dn(gray, profile.full_scale_dn)

    if sensitivity_map is not None:
        signal_dn = apply_sensitivity_map(signal_dn, sensitivity_map)
    else:
        signal_dn = apply_prnu(signal_dn, prnu_std, rng, prnu_map=prnu_map)

    # Shot noise is photon noise, so it only applies to the signal
    # before the dark is added
    signal_dn = apply_shot_noise(signal_dn, rng)

    if master_dark is not None:
        signal_dn = signal_dn + master_dark
    else:
        signal_dn = apply_dark_current(signal_dn, dsnu_std, rng, dsnu_map=dsnu_map)

    signal_dn = apply_readout_noise(
        signal_dn, profile.read_noise_dn, profile.row_noise_std, rng
    )
    return adc_quantize(signal_dn, profile.full_scale_dn)


def add_common_arguments(parser):
    """Arguments shared with batch_transform.py."""
    parser.add_argument("--sensor-profile", type=Path, default=Path(DEFAULT_PROFILE),
                        help=f"Sensor profile JSON (default: {DEFAULT_PROFILE}).")
    parser.add_argument("--sensitivity-map", type=Path, help="Sensitivity map .npy from make_masters.py.")
    parser.add_argument("--master-dark", type=Path, help="Master dark .npy from make_masters.py.")
    parser.add_argument("--prnu-std", type=float,
                        help="Synthetic PRNU standard deviation, used instead of --sensitivity-map.")
    parser.add_argument("--dsnu-std", type=float,
                        help="Synthetic DSNU standard deviation in DN, used instead of --master-dark.")
    parser.add_argument("--seed", type=int, help="Random seed (default: the profile's).")
    return parser


def resolve_sources(args):
    """Calibration maps are the expected input; synthetic values must be explicit."""
    if (args.sensitivity_map is None) == (args.prnu_std is None):
        raise SystemExit(
            "error: give exactly one of --sensitivity-map or --prnu-std.\n"
            "  --sensitivity-map is the normal path; --prnu-std substitutes a synthetic gain map."
        )
    if (args.master_dark is None) == (args.dsnu_std is None):
        raise SystemExit(
            "error: give exactly one of --master-dark or --dsnu-std.\n"
            "  --master-dark is the normal path; --dsnu-std substitutes a synthetic offset."
        )
    for path in (args.sensitivity_map, args.master_dark, args.sensor_profile):
        if path is not None and not path.exists():
            raise SystemExit(f"error: {path} not found.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transform a rendered image into a sensor-style image."
    )
    parser.add_argument("--input", type=Path, required=True, help="Rendered input image.")
    parser.add_argument("--output", type=Path, required=True, help="Output image path.")
    return add_common_arguments(parser).parse_args()


def main():
    args = parse_arguments()
    resolve_sources(args)

    try:
        profile = SensorProfile.load(args.sensor_profile)
        sensitivity_map, master_dark = load_calibration(profile, args.sensitivity_map, args.master_dark)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    rng = np.random.default_rng(args.seed if args.seed is not None else profile.seed)

    out = transform_render(
        args.input, profile, rng,
        sensitivity_map=sensitivity_map, master_dark=master_dark,
        prnu_std=args.prnu_std, dsnu_std=args.dsnu_std,
    )
    save_image(out, args.output)
    print(f"Saved to: {args.output}  (profile: {profile.name})")

if __name__ == "__main__":
    main()
