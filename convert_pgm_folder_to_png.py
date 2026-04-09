import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def list_pgm_files(input_dir: Path, recursive: bool):
    pattern = "**/*.pgm" if recursive else "*.pgm"
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def convert_pgm_to_png(input_path: Path, output_path: Path):
    with Image.open(input_path) as img:
        arr = np.array(img)

    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale PGM at {input_path}, got shape {arr.shape}")

    if arr.dtype.kind == "f":
        arr = np.nan_to_num(arr, nan=0.0, posinf=65535.0, neginf=0.0)

    # Save as 16-bit grayscale PNG to preserve sensor dynamic range.
    arr = np.clip(arr, 0, 65535).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(output_path, format="PNG")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a folder of PGM files to PNG files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .pgm files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where .png files are written.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also convert .pgm files in subfolders and mirror structure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    pgm_files = list_pgm_files(args.input_dir, recursive=args.recursive)
    if not pgm_files:
        raise RuntimeError(f"No .pgm files found in: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(pgm_files)

    for idx, input_path in enumerate(pgm_files, start=1):
        relative = input_path.relative_to(args.input_dir)
        output_path = args.output_dir / relative.with_suffix(".png")
        convert_pgm_to_png(input_path, output_path)
        print(f"[{idx}/{total}] {input_path} -> {output_path}")

    print(f"Finished. Wrote {total} PNG files to {args.output_dir}")


if __name__ == "__main__":
    main()
