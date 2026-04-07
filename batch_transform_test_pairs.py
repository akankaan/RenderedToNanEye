import argparse
from pathlib import Path

import numpy as np
import transform_rendered_to_naneye as sim


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def list_input_images(input_dir: Path):
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def transform_one_image(input_path: Path, output_path: Path, rng):
    gray = sim.load_image_linear(str(input_path))
    gray = sim.apply_optics(
        gray,
        sim.width,
        sim.height,
        sim.blur_radius,
        sim.fx_px,
        sim.fy_px,
        sim.cx_px,
        sim.cy_px,
        sim.k1,
        sim.k2,
        sim.p1,
        sim.p2,
        sim.k3,
    )

    signal_dn = sim.convert_to_electrons(gray, sim.full_scale_dn)
    signal_dn = sim.apply_prnu(signal_dn, sim.prnu_std, rng)
    signal_dn = sim.apply_shot_noise(signal_dn, rng)
    signal_dn = sim.apply_dark_current(signal_dn, sim.dsnu_std, rng)
    signal_dn = sim.apply_readout_noise(signal_dn, sim.read_noise_dn, sim.row_noise_std, rng)

    out = sim.adc_quantize(signal_dn, sim.full_scale_dn)
    sim.save_image(out, str(output_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform all renders in Test_Pairs/Test_Renders and write outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Test_Pairs/Test_Renders"),
        help="Directory of rendered input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Test_Pairs/Test_Transformed_Output"),
        help="Directory where transformed images are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=sim.seed,
        help="Base random seed for deterministic batch noise generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_input_images(args.input_dir)
    if not image_paths:
        raise RuntimeError(f"No supported image files found in: {args.input_dir}")

    rng = np.random.default_rng(args.seed)
    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        output_path = args.output_dir / image_path.name
        transform_one_image(image_path, output_path, rng)
        print(f"[{idx}/{total}] {image_path.name} -> {output_path}")

    print(f"Finished. Wrote {total} transformed images to {args.output_dir}")


if __name__ == "__main__":
    main()
