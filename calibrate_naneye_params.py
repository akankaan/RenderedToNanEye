import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path):
    # Load an image from disk as a floating-point array.
    img = Image.open(path)
    img = img.convert("I")
    arr = np.array(img, dtype=np.float32)
    return arr


def list_images(folder):
    # Gather all image files in a folder.
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}
    return sorted(
        [Path(folder) / p for p in os.listdir(folder) if Path(p).suffix.lower() in suffixes]
    )


def smooth_image(image, kernel_size=31):
    # Smooth a 2D image with a local box filter.
    if kernel_size <= 1:
        return image

    pad = kernel_size // 2
    padded = np.pad(image, pad, mode="reflect")
    ii = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    h, w = image.shape
    output = np.empty_like(image)

    for i in range(h):
        for j in range(w):
            y0 = i
            x0 = j
            y1 = y0 + kernel_size - 1
            x1 = x0 + kernel_size - 1
            total = ii[y1, x1]
            if y0 > 0:
                total -= ii[y0 - 1, x1]
            if x0 > 0:
                total -= ii[y1, x0 - 1]
            if y0 > 0 and x0 > 0:
                total += ii[y0 - 1, x0 - 1]
            output[i, j] = total / (kernel_size * kernel_size)

    return output


def compute_flat_field_metrics(image_paths):
    # Estimate PRNU from flat-field images.
    if len(image_paths) == 0:
        return None

    images = [load_image(str(p)) for p in image_paths]
    stack = np.stack(images, axis=0)

    mean_image = np.mean(stack, axis=0)
    norm = mean_image / np.mean(mean_image)
    prnu = np.std(norm)

    return {
        "flat_count": len(image_paths),
        "flat_mean_value": float(np.mean(mean_image)),
        "prnu_estimate": float(prnu),
        "flat_mean_std": float(np.std(mean_image)),
    }


def compute_dark_metrics(image_paths):
    # Estimate dark current and dark fixed-pattern noise.
    if len(image_paths) == 0:
        return None

    images = [load_image(str(p)) for p in image_paths]
    stack = np.stack(images, axis=0)

    mean_dark = np.mean(stack, axis=0)
    dark_pixel_std = np.std(stack, axis=0, ddof=1)

    row_means = np.mean(mean_dark, axis=1)
    row_noise = float(np.std(row_means, ddof=1))

    return {
        "dark_count": len(image_paths),
        "dark_mean_value": float(np.mean(mean_dark)),
        "dark_mean_std": float(np.std(mean_dark)),
        "dark_pixel_std_mean": float(np.mean(dark_pixel_std)),
        "dark_pixel_std_std": float(np.std(dark_pixel_std)),
        "row_noise_estimate": row_noise,
    }


def compute_general_noise_estimates(image_paths, sample_limit=200):
    # Estimate noise metrics from a general set of normal images.
    if len(image_paths) == 0:
        return None

    if len(image_paths) > sample_limit:
        step = max(1, len(image_paths) // sample_limit)
        image_paths = image_paths[::step]

    images = [load_image(str(p)) for p in image_paths]
    stack = np.stack(images, axis=0)

    mean_image = np.mean(stack, axis=0)
    pixel_means = np.mean(stack, axis=0)
    pixel_vars = np.var(stack, axis=0, ddof=1)

    smoothed = smooth_image(mean_image, kernel_size=31)
    normalized = mean_image / np.clip(smoothed, 1e-6, None)

    prnu_estimate = float(np.std(normalized))
    dsnu_estimate = float(np.std(mean_image - smoothed))

    row_means = np.mean(mean_image, axis=1)
    row_noise_estimate = float(np.std(row_means, ddof=1))

    means = pixel_means.ravel()
    vars_ = pixel_vars.ravel()
    mask = np.isfinite(means) & np.isfinite(vars_) & (means >= 0) & (vars_ >= 0)
    means = means[mask]
    vars_ = vars_[mask]

    if len(means) >= 20:
        slope, intercept = np.polyfit(means, vars_, 1)
        read_noise_estimate = float(np.sqrt(max(intercept, 0.0)))
    else:
        slope = 0.0
        intercept = 0.0
        read_noise_estimate = None

    return {
        "general_count": len(image_paths),
        "global_mean_value": float(np.mean(mean_image)),
        "global_std_value": float(np.std(mean_image)),
        "prnu_estimate": prnu_estimate,
        "dsnu_estimate": dsnu_estimate,
        "row_noise_estimate": row_noise_estimate,
        "variance_slope": float(slope),
        "variance_intercept": float(intercept),
        "read_noise_estimate": read_noise_estimate,
    }


def compute_image_statistics(image_paths):
    # Compute simple image statistics for a large image folder.
    if len(image_paths) == 0:
        return None

    sums = []
    mins = []
    maxs = []

    for path in image_paths:
        arr = load_image(str(path))
        sums.append(np.mean(arr))
        mins.append(np.min(arr))
        maxs.append(np.max(arr))

    return {
        "image_count": len(image_paths),
        "mean_of_means": float(np.mean(sums)),
        "std_of_means": float(np.std(sums, ddof=1)),
        "min_value": float(np.min(mins)),
        "max_value": float(np.max(maxs)),
    }


def save_calibration(params, output_path):
    # Write calibration parameters to a JSON file.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calibrate NanEye sensor parameters from real images.")
    parser.add_argument("--all-folder", type=str, help="Folder containing sensor images for general statistics.")
    parser.add_argument("--dark-folder", type=str, help="Folder containing dark frames for dark current/DSNU estimation.")
    parser.add_argument("--flat-folder", type=str, help="Folder containing flat-field frames for PRNU estimation.")
    parser.add_argument("--output", type=str, default="naneye_calibration.json", help="Output JSON file for estimated parameters.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    calibration = {}

    if args.all_folder:
        image_paths = list_images(args.all_folder)
        calibration["general"] = compute_image_statistics(image_paths)
        calibration["general_noise"] = compute_general_noise_estimates(image_paths)

    if args.dark_folder:
        dark_paths = list_images(args.dark_folder)
        calibration["dark"] = compute_dark_metrics(dark_paths)

    if args.flat_folder:
        flat_paths = list_images(args.flat_folder)
        calibration["flat"] = compute_flat_field_metrics(flat_paths)

    recommended = {
        "full_scale_dn": 860.0,
        "read_noise_dn": None,
        "prnu_std": None,
        "dsnu_std": None,
        "row_noise_std": None,
        "notes": "Estimates are approximate when only normal images are available. Flat/dark frames give better calibration."
    }

    if calibration.get("flat") is not None:
        recommended["prnu_std"] = calibration["flat"]["prnu_estimate"]

    if calibration.get("dark") is not None:
        recommended["dsnu_std"] = calibration["dark"]["dark_pixel_std_mean"]
        recommended["row_noise_std"] = calibration["dark"]["row_noise_estimate"]
        recommended["read_noise_dn"] = calibration["dark"]["dark_pixel_std_mean"]

    if calibration.get("general_noise") is not None:
        gen = calibration["general_noise"]
        if recommended["prnu_std"] is None:
            recommended["prnu_std"] = gen["prnu_estimate"]
        if recommended["dsnu_std"] is None:
            recommended["dsnu_std"] = gen["dsnu_estimate"]
        if recommended["row_noise_std"] is None:
            recommended["row_noise_std"] = gen["row_noise_estimate"]
        if recommended["read_noise_dn"] is None and gen["read_noise_estimate"] is not None and gen["read_noise_estimate"] > 0:
            recommended["read_noise_dn"] = gen["read_noise_estimate"]

    calibration["recommended"] = recommended
    save_calibration(calibration, args.output)

    print(f"Calibration complete. Results written to {args.output}")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
