"""Run transform_rendered_to_sensor over a folder of renders"""
import argparse
from pathlib import Path

import numpy as np

import transform_rendered_to_sensor as sensor

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def list_input_images(input_dir):
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transform every render in a folder into sensor-style images."
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of rendered input images.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where transformed images are written.")
    return sensor.add_common_arguments(parser).parse_args()


def main():
    args = parse_arguments()
    sensor.resolve_sources(args)

    if not args.input_dir.exists():
        raise SystemExit(f"error: input directory not found: {args.input_dir}")
    image_paths = list_input_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"error: no supported image files in {args.input_dir}")

    try:
        profile = sensor.SensorProfile.load(args.sensor_profile)
        sensitivity_map, master_dark = sensor.load_calibration(profile, args.sensitivity_map, args.master_dark)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    rng = np.random.default_rng(args.seed if args.seed is not None else profile.seed)

    # Fixed-pattern noise is identical between frames
    # Shot and read noise are changing
    prnu_map, dsnu_map = sensor.generate_fixed_pattern_maps(
        profile.shape, args.prnu_std, args.dsnu_std, rng
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        out = sensor.transform_render(
            image_path, profile, rng,
            sensitivity_map=sensitivity_map, master_dark=master_dark,
            prnu_std=args.prnu_std, dsnu_std=args.dsnu_std,
            prnu_map=prnu_map, dsnu_map=dsnu_map,
        )
        output_path = args.output_dir / image_path.name
        sensor.save_image(out, output_path)
        print(f"[{index}/{total}] {image_path.name} -> {output_path}")

    print(f"Finished. Wrote {total} images to {args.output_dir}  (profile: {profile.name})")


if __name__ == "__main__":
    main()
