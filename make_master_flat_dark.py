import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


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


def normalize_flat(flat_image):
    mean_value = np.mean(flat_image)
    if mean_value == 0:
        raise ValueError("Flat-field image has zero mean and cannot be normalized.")
    return flat_image / mean_value


def save_npy(image, output_path):
    np.save(output_path, image)


def save_png(image, output_path):
    image = np.asarray(image, dtype=np.float32)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    image = np.round(image * 65535.0).astype(np.uint16)
    Image.fromarray(image, mode="I;16").save(output_path)


def make_master_dark(dark_folder, output_base, method="median"):
    dark_paths = list_images(dark_folder)
    if not dark_paths:
        raise ValueError(f"No dark frames found in '{dark_folder}'.")

    master_dark = compute_master_image(dark_paths, method=method)
    save_npy(master_dark, output_base.with_suffix(".npy"))
    save_png(master_dark, output_base.with_suffix(".png"))
    return master_dark


def make_master_flat(flat_folder, output_base, method="mean", normalize=True):
    flat_paths = list_images(flat_folder)
    if not flat_paths:
        raise ValueError(f"No flat frames found in '{flat_folder}'.")

    master_flat = compute_master_image(flat_paths, method=method)
    if normalize:
        master_flat = normalize_flat(master_flat)

    save_npy(master_flat, output_base.with_suffix(".npy"))
    save_png(master_flat, output_base.with_suffix(".png"))
    return master_flat


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create master dark and master flat images from folders of dark and flat frames."
    )
    parser.add_argument("--dark-folder", type=str, help="Folder containing dark frame images.")
    parser.add_argument("--flat-folder", type=str, help="Folder containing flat frame images.")
    parser.add_argument("--dark-output", type=str, default="master_dark", help="Base output name for the master dark files (no extension).")
    parser.add_argument("--flat-output", type=str, default="master_flat", help="Base output name for the master flat files (no extension).")
    parser.add_argument("--dark-method", choices=["mean", "median"], default="median", help="Combine dark frames using mean or median.")
    parser.add_argument("--flat-method", choices=["mean", "median"], default="mean", help="Combine flat frames using mean or median.")
    parser.add_argument("--no-normalize-flat", action="store_true", help="Do not normalize the master flat by its mean.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    outputs = {}

    if args.dark_folder:
        dark_base = Path(args.dark_output)
        master_dark = make_master_dark(args.dark_folder, dark_base, method=args.dark_method)
        outputs["master_dark"] = {
            "npy": str(dark_base.with_suffix(".npy")),
            "png": str(dark_base.with_suffix(".png")),
            "shape": master_dark.shape,
            "dtype": str(master_dark.dtype),
        }

    if args.flat_folder:
        flat_base = Path(args.flat_output)
        master_flat = make_master_flat(args.flat_folder, flat_base, method=args.flat_method, normalize=not args.no_normalize_flat)
        outputs["master_flat"] = {
            "npy": str(flat_base.with_suffix(".npy")),
            "png": str(flat_base.with_suffix(".png")),
            "shape": master_flat.shape,
            "dtype": str(master_flat.dtype),
            "normalized": not args.no_normalize_flat,
        }

    if outputs:
        print("Master files created:")
        for name, info in outputs.items():
            print(f"  {name}:")
            for key, value in info.items():
                print(f"    {key}: {value}")
    else:
        print("No dark or flat folder provided. Nothing was created.")


if __name__ == "__main__":
    main()
