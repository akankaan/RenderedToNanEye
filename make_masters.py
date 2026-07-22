"""Build a master dark and a sensitivity map from folders of calibration captures.

Two outputs: master_dark.npy (additive offset in DN) and sensitivity_map.npy
(the averaged flats with the master dark subtracted, then peak-normalized into a
unitless per-pixel gain). The plain averaged flat is an intermediate and is not saved.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from transform_rendered_to_sensor import DEFAULT_PROFILE, SensorProfile


def list_images(folder):
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}
    folder_path = Path(folder)
    return sorted(
        [p for p in folder_path.iterdir() if p.suffix.lower() in suffixes and p.is_file()]
    )


def load_image(path):
    img = Image.open(path)
    img = img.convert("I")
    return np.array(img, dtype=np.float32)


def compute_master_image(image_paths, method="mean"):
    if len(image_paths) == 0:
        raise ValueError("No input images found.")

    stack = np.stack([load_image(path) for path in image_paths], axis=0)

    if method == "median":
        return np.median(stack, axis=0).astype(np.float32)
    return np.mean(stack, axis=0).astype(np.float32)


PEAK_PERCENTILE = 99.5


def normalize_to_peak(image, peak_percentile=PEAK_PERCENTILE):
    # Normalize by the peak, not the mean because the gain map is applied by multiplying
    # a render, so the best-responding pixel should sit at 1.0 and the rest smaller than 1.
    # Dividing by the mean works when one divides by a flat to correct
    # an image. In this case, it amplifies the center and causes saturation.
    # Will use 99.5th percentile rather than the max, so small inconsistencies cannot change
    # the frame's scale.
    peak = float(np.percentile(image, peak_percentile))
    if peak <= 0:
        raise ValueError("Flat-field image has a non-positive peak and cannot be normalized.")
    return image / peak


def save_npy(image, output_path):
    np.save(output_path, image)


def save_png(image, output_path, display_white):
    # True-brightness 8-bit preview: map [0, display_white] to [0, 255] with no per-image
    # stretch. A per-image min-max stretch would blow a near-uniform dark's tiny spread
    # across the full range and make it look like dramatic speckle it does not have.
    disp = np.clip(np.asarray(image, dtype=np.float32) / display_white, 0.0, 1.0)
    Image.fromarray(np.round(disp * 255.0).astype(np.uint8)).save(output_path)


def make_master_dark(
    dark_folder,
    output_base,
    method,
    source_white_level,
    full_scale_dn,
):
    dark_paths = list_images(dark_folder)
    if not dark_paths:
        raise ValueError(f"No dark frames found in '{dark_folder}'.")

    master_dark = compute_master_image(dark_paths, method=method)
    # Convert master dark to DN scale
    master_dark = (master_dark * (full_scale_dn / source_white_level)).astype(np.float32)

    save_npy(master_dark, output_base.with_suffix(".npy"))
    save_png(master_dark, output_base.with_suffix(".png"), display_white=full_scale_dn)
    return master_dark


def make_sensitivity_map(
    flat_folder,
    output_base,
    method,
    normalize,
    source_white_level,
    full_scale_dn,
    master_dark=None,
):
    flat_paths = list_images(flat_folder)
    if not flat_paths:
        raise ValueError(f"No flat frames found in '{flat_folder}'.")

    sensitivity = compute_master_image(flat_paths, method=method)
    if master_dark is not None:
        # Convert averaged flat to DN so it matches master dark before subtraction
        sensitivity = sensitivity * (full_scale_dn / source_white_level)
        sensitivity = np.clip(sensitivity - master_dark, 1e-6, None)

    if normalize:
        sensitivity = normalize_to_peak(sensitivity)
        display_white = 1.0
    elif master_dark is not None:
        display_white = full_scale_dn
    else:
        display_white = source_white_level

    save_npy(sensitivity, output_base.with_suffix(".npy"))
    save_png(sensitivity, output_base.with_suffix(".png"), display_white=display_white)
    return sensitivity


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build a master dark and a sensitivity map from folders of dark and flat frames."
    )
    parser.add_argument("--dark-folder", type=str, help="Folder containing dark frame images.")
    parser.add_argument("--flat-folder", type=str, help="Folder containing flat frame images.")
    parser.add_argument("--dark-output", type=str, default="master_dark", help="Base output name for the master dark files (no extension).")
    parser.add_argument("--sensitivity-output", type=str, default="sensitivity_map", help="Base output name for the sensitivity map files (no extension).")
    parser.add_argument("--dark-method", choices=["mean", "median"], default="median", help="Combine dark frames using mean or median.")
    parser.add_argument("--flat-method", choices=["mean", "median"], default="mean", help="Combine flat frames using mean or median.")
    parser.add_argument("--no-normalize", action="store_true", help="Do not peak-normalize the sensitivity map (debugging).")
    parser.add_argument("--sensor-profile", type=Path, default=Path(DEFAULT_PROFILE),
                        help=f"Sensor profile JSON supplying full_scale_dn and source_white_level (default: {DEFAULT_PROFILE}).")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.sensor_profile.exists():
        raise SystemExit(f"error: {args.sensor_profile} not found.")
    profile = SensorProfile.load(args.sensor_profile)

    outputs = {}
    master_dark = None

    if args.dark_folder:
        dark_base = Path(args.dark_output)
        master_dark = make_master_dark(
            args.dark_folder,
            dark_base,
            method=args.dark_method,
            source_white_level=profile.source_white_level,
            full_scale_dn=profile.full_scale_dn,
        )
        outputs["master_dark"] = {
            "npy": str(dark_base.with_suffix(".npy")),
            "png": str(dark_base.with_suffix(".png")),
            "shape": master_dark.shape,
            "dtype": str(master_dark.dtype),
        }

    if args.flat_folder:
        if master_dark is None:
            print(
                "Warning: no --dark-folder given, so the dark pedestal stays in the "
                "sensitivity map and it will understate the sensor response."
            )

        sens_base = Path(args.sensitivity_output)
        sensitivity = make_sensitivity_map(
            args.flat_folder,
            sens_base,
            method=args.flat_method,
            normalize=not args.no_normalize,
            master_dark=master_dark,
            source_white_level=profile.source_white_level,
            full_scale_dn=profile.full_scale_dn,
        )
        outputs["sensitivity_map"] = {
            "npy": str(sens_base.with_suffix(".npy")),
            "png": str(sens_base.with_suffix(".png")),
            "shape": sensitivity.shape,
            "dtype": str(sensitivity.dtype),
            "normalized": not args.no_normalize,
            "dark_subtracted": master_dark is not None,
        }

    if outputs:
        print(f"Files created (profile: {profile.name}, "
              f"{profile.source_white_level:.0f} -> {profile.full_scale_dn:.0f} DN):")
        for name, info in outputs.items():
            print(f"  {name}:")
            for key, value in info.items():
                print(f"    {key}: {value}")
    else:
        print("No dark or flat folder provided. Nothing was created.")


if __name__ == "__main__":
    main()
