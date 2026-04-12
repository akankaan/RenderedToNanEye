import argparse
from pathlib import Path

import numpy as np
import transform_rendered_to_naneye as sim


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def list_input_images(input_dir: Path):
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def transform_one_image(
    input_path: Path,
    output_path: Path,
    rng,
    prnu_map,
    dsnu_map,
    master_flat=None,
    master_dark=None,
    flat_is_normalized=False,
):
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

    has_master_flat = master_flat is not None
    has_master_dark = master_dark is not None

    if has_master_flat:
        if flat_is_normalized and has_master_dark:
            signal_dn = sim.apply_master_flat(signal_dn, master_flat, master_dark=None)
        elif has_master_dark:
            signal_dn = sim.apply_master_flat(signal_dn, master_flat, master_dark=master_dark)
        else:
            signal_dn = sim.apply_master_flat(signal_dn, master_flat, master_dark=None)
    else:
        signal_dn = sim.apply_prnu(signal_dn, sim.prnu_std, rng, prnu_map=prnu_map)

    if has_master_dark:
        signal_dn += master_dark
    else:
        signal_dn = sim.apply_dark_current(signal_dn, sim.dsnu_std, rng, dsnu_map=dsnu_map)

    signal_dn = sim.apply_shot_noise(signal_dn, rng)
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
    parser.add_argument(
        "--master-flat",
        type=Path,
        help="Optional master flat .npy file used for calibrated batch transforms.",
    )
    parser.add_argument(
        "--master-dark",
        type=Path,
        help="Optional master dark .npy file used for calibrated batch transforms.",
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

    if args.master_flat is not None and not args.master_flat.exists():
        raise FileNotFoundError(f"Master flat not found: {args.master_flat}")
    if args.master_dark is not None and not args.master_dark.exists():
        raise FileNotFoundError(f"Master dark not found: {args.master_dark}")

    master_flat = sim.load_master_npy(args.master_flat) if args.master_flat else None
    master_dark = sim.load_master_npy(args.master_dark) if args.master_dark else None
    expected_shape = (sim.height, sim.width)
    sim.validate_master_shape(master_flat, expected_shape, "Master flat")
    sim.validate_master_shape(master_dark, expected_shape, "Master dark")

    flat_is_normalized = master_flat is not None and sim.looks_like_normalized_flat(master_flat)
    if master_flat is not None or master_dark is not None:
        master_flat, master_dark, scale, source_white = sim.maybe_rescale_master_maps(
            master_flat,
            master_dark,
            full_scale_dn=sim.full_scale_dn,
            scale_flat=not flat_is_normalized,
        )
        if scale != 1.0:
            print(
                "Info: scaled master calibration maps by "
                f"{scale:.6f} to match full_scale_dn={sim.full_scale_dn:.1f} "
                f"(assumed source white level {source_white:.0f} DN)."
            )
        if flat_is_normalized and master_dark is not None:
            print(
                "Info: master flat looks normalized; using it directly as gain "
                "and skipping (master_flat - master_dark)."
            )

    if master_flat is None and master_dark is None:
        prnu_map, dsnu_map = sim.generate_fixed_pattern_maps(
            (sim.height, sim.width),
            sim.prnu_std,
            sim.dsnu_std,
            rng,
        )
    else:
        prnu_map, dsnu_map = None, None

    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        output_path = args.output_dir / image_path.name
        transform_one_image(
            image_path,
            output_path,
            rng,
            prnu_map,
            dsnu_map,
            master_flat=master_flat,
            master_dark=master_dark,
            flat_is_normalized=flat_is_normalized,
        )
        print(f"[{idx}/{total}] {image_path.name} -> {output_path}")

    print(f"Finished. Wrote {total} transformed images to {args.output_dir}")


if __name__ == "__main__":
    main()
