import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


Coord = Tuple[int, int]


def list_png_files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.glob("*.png") if p.is_file())


def load_png_stack(folder: Path) -> Tuple[List[Path], np.ndarray]:
    files = list_png_files(folder)
    if not files:
        raise RuntimeError(f"No PNG files found in: {folder}")

    arrays = []
    for path in files:
        with Image.open(path) as img:
            arrays.append(np.array(img, dtype=np.float32))

    stack = np.stack(arrays, axis=0)
    return files, stack


def neighbor_mean_8(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    summed = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return summed / 8.0


def build_ignore_mask(
    shape: Tuple[int, int],
    ignore_left_cols: int,
    ignore_right_cols: int,
) -> np.ndarray:
    ignore = np.zeros(shape, dtype=bool)
    if ignore_left_cols > 0:
        ignore[:, :ignore_left_cols] = True
    if ignore_right_cols > 0:
        ignore[:, -ignore_right_cols:] = True
    return ignore


def select_peak_pixels_by_component(
    candidate_mask: np.ndarray,
    residual_map: np.ndarray,
    max_per_component: int = 2,
    secondary_ratio: float = 0.12,
) -> np.ndarray:
    """Pick top outliers per connected component to avoid oscillating paired defects."""
    h, w = candidate_mask.shape
    visited = np.zeros_like(candidate_mask, dtype=bool)
    selected = np.zeros_like(candidate_mask, dtype=bool)

    for y in range(h):
        for x in range(w):
            if not candidate_mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component = []

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))

                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if not candidate_mask[ny, nx] or visited[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            component.sort(key=lambda p: float(residual_map[p[0], p[1]]), reverse=True)
            best_y, best_x = component[0]
            best_value = float(residual_map[best_y, best_x])
            selected[best_y, best_x] = True

            if max_per_component > 1 and best_value > 0:
                kept = 1
                for cy, cx in component[1:]:
                    if kept >= max_per_component:
                        break
                    value = float(residual_map[cy, cx])
                    if value >= secondary_ratio * best_value:
                        selected[cy, cx] = True
                        kept += 1

    return selected


def detect_dead_pixel_mask(
    stack: np.ndarray,
    ignore_mask: np.ndarray,
    detection_mode: str,
    std_threshold: float,
    residual_threshold: Optional[float],
    residual_sigma_scale: float,
    residual_floor: float,
) -> Tuple[np.ndarray, float]:
    median = np.median(stack, axis=0)
    spatial_residual = np.abs(median - neighbor_mean_8(median))

    if residual_threshold is None:
        valid = spatial_residual[~ignore_mask]
        med = np.median(valid)
        mad = np.median(np.abs(valid - med))
        robust_sigma = 1.4826 * mad
        threshold = max(residual_floor, med + residual_sigma_scale * robust_sigma)
    else:
        threshold = residual_threshold

    candidate = (spatial_residual >= threshold) & (~ignore_mask)

    if detection_mode == "peak":
        dead = select_peak_pixels_by_component(candidate, spatial_residual)
    elif detection_mode == "stuck":
        temporal_std = np.std(stack, axis=0)
        dead = candidate & (temporal_std <= std_threshold)
    else:
        raise ValueError(f"Unsupported detection mode: {detection_mode}")

    return dead, float(threshold)


def sorted_coords(mask: np.ndarray) -> List[Coord]:
    ys, xs = np.where(mask)
    return sorted((int(x), int(y)) for y, x in zip(ys, xs))


def detect_persistent_dead_pixels(
    stacks: Sequence[np.ndarray],
    ignore_mask: np.ndarray,
    detection_mode: str,
    std_threshold: float,
    residual_threshold: Optional[float],
    residual_sigma_scale: float,
    residual_floor: float,
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    per_stack_masks = []
    per_stack_thresholds = []

    for stack in stacks:
        mask, threshold = detect_dead_pixel_mask(
            stack=stack,
            ignore_mask=ignore_mask,
            detection_mode=detection_mode,
            std_threshold=std_threshold,
            residual_threshold=residual_threshold,
            residual_sigma_scale=residual_sigma_scale,
            residual_floor=residual_floor,
        )
        per_stack_masks.append(mask)
        per_stack_thresholds.append(threshold)

    persistent = per_stack_masks[0].copy()
    for mask in per_stack_masks[1:]:
        persistent &= mask

    return persistent, per_stack_masks, per_stack_thresholds


def mean_valid_neighbors(
    image: np.ndarray,
    y: int,
    x: int,
    dead_mask: np.ndarray,
    ignore_mask: np.ndarray,
) -> float:
    h, w = image.shape
    values = []

    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if ny == y and nx == x:
                continue
            if dead_mask[ny, nx] or ignore_mask[ny, nx]:
                continue
            values.append(float(image[ny, nx]))

    if not values:
        for ny in range(max(0, y - 1), min(h, y + 2)):
            for nx in range(max(0, x - 1), min(w, x + 2)):
                if ny == y and nx == x:
                    continue
                if dead_mask[ny, nx]:
                    continue
                values.append(float(image[ny, nx]))

    if not values:
        return float(image[y, x])

    return float(np.mean(values))


def patch_image_array(
    image: np.ndarray,
    dead_coords: Sequence[Coord],
    dead_mask: np.ndarray,
    ignore_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    patched = image.copy()
    changes = 0

    if np.issubdtype(image.dtype, np.integer):
        dtype_info = np.iinfo(image.dtype)
        min_value = int(dtype_info.min)
        max_value = int(dtype_info.max)
    else:
        min_value = None
        max_value = None

    for x, y in dead_coords:
        replacement = mean_valid_neighbors(
            image=patched,
            y=y,
            x=x,
            dead_mask=dead_mask,
            ignore_mask=ignore_mask,
        )

        if min_value is not None and max_value is not None:
            replacement = int(round(replacement))
            replacement = max(min_value, min(max_value, replacement))

        old_value = patched[y, x]
        patched[y, x] = replacement
        if patched[y, x] != old_value:
            changes += 1

    return patched, changes


def write_png_safely(array: np.ndarray, output_path: Path):
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    Image.fromarray(array).save(tmp_path, format="PNG")
    tmp_path.replace(output_path)


def patch_folder(
    input_folder: Path,
    output_folder: Path,
    in_place: bool,
    dead_coords: Sequence[Coord],
    dead_mask: np.ndarray,
    ignore_mask: np.ndarray,
) -> Tuple[int, int]:
    files = list_png_files(input_folder)
    if not files:
        raise RuntimeError(f"No PNG files found in: {input_folder}")

    if not in_place:
        output_folder.mkdir(parents=True, exist_ok=True)

    file_count = 0
    pixel_updates = 0

    for path in files:
        with Image.open(path) as img:
            arr = np.array(img)

        patched, changes = patch_image_array(
            image=arr,
            dead_coords=dead_coords,
            dead_mask=dead_mask,
            ignore_mask=ignore_mask,
        )

        if in_place:
            out_path = path
        else:
            out_path = output_folder / path.name

        write_png_safely(patched, out_path)
        file_count += 1
        pixel_updates += changes

    return file_count, pixel_updates


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect persistent dead pixels in one or more PNG folders and patch "
            "them with neighbor averages."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="PNG folder to include (repeat for darks and flats).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source PNG files in-place.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_patched",
        help="Suffix for output folders when not using --in-place.",
    )
    parser.add_argument(
        "--ignore-left-cols",
        type=int,
        default=5,
        help="Ignore this many left-edge columns (start markers).",
    )
    parser.add_argument(
        "--ignore-right-cols",
        type=int,
        default=2,
        help="Ignore this many right-edge columns (end markers).",
    )
    parser.add_argument(
        "--detection-mode",
        type=str,
        choices=("peak", "stuck"),
        default="peak",
        help="Dead-pixel detection strategy: 'peak' catches persistent spatial outlier peaks.",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=5.0,
        help="Used only in 'stuck' mode: max temporal std for a stuck pixel.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=None,
        help="Override spatial residual threshold; default is robust auto-threshold.",
    )
    parser.add_argument(
        "--residual-sigma-scale",
        type=float,
        default=20.0,
        help="Auto-threshold scale applied to robust residual sigma when no override is set.",
    )
    parser.add_argument(
        "--residual-floor",
        type=float,
        default=2000.0,
        help="Minimum spatial residual threshold in DN for auto-threshold mode.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional text file path to save dead-pixel coordinates.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=3,
        help="Maximum iterative detect+patch passes to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dirs = args.input_dir
    for folder in input_dirs:
        if not folder.exists():
            raise FileNotFoundError(f"Input directory not found: {folder}")

    if args.max_passes < 1:
        raise ValueError("--max-passes must be at least 1.")

    all_patched_coords: List[Coord] = []
    seen_coords = set()

    for pass_idx in range(1, args.max_passes + 1):
        stacks = []
        folder_names = []
        base_shape = None

        print(f"\nPass {pass_idx}/{args.max_passes}")
        for folder in input_dirs:
            files, stack = load_png_stack(folder)
            folder_names.append(folder)
            stacks.append(stack)
            if base_shape is None:
                base_shape = stack.shape[1:]
            elif stack.shape[1:] != base_shape:
                raise ValueError(
                    f"All folders must share image size. {folder} has {stack.shape[1:]}, "
                    f"expected {base_shape}."
                )
            print(f"Loaded {len(files)} PNG files from {folder}")

        assert base_shape is not None
        ignore_mask = build_ignore_mask(
            shape=base_shape,
            ignore_left_cols=args.ignore_left_cols,
            ignore_right_cols=args.ignore_right_cols,
        )

        persistent_mask, per_stack_masks, per_stack_thresholds = detect_persistent_dead_pixels(
            stacks=stacks,
            ignore_mask=ignore_mask,
            detection_mode=args.detection_mode,
            std_threshold=args.std_threshold,
            residual_threshold=args.residual_threshold,
            residual_sigma_scale=args.residual_sigma_scale,
            residual_floor=args.residual_floor,
        )

        print(
            f"Detection mode: {args.detection_mode} "
            f"(ignore_left_cols={args.ignore_left_cols}, ignore_right_cols={args.ignore_right_cols})"
        )
        for folder, mask, threshold in zip(folder_names, per_stack_masks, per_stack_thresholds):
            print(
                f"{folder}: dead candidates={int(np.sum(mask))} "
                f"(residual_threshold={threshold:.3f})"
            )

        dead_coords = sorted_coords(persistent_mask)
        print(f"Persistent dead pixels across all folders: {len(dead_coords)}")
        for x, y in dead_coords:
            print(f"  (x={x}, y={y})")
            if (x, y) not in seen_coords:
                seen_coords.add((x, y))
                all_patched_coords.append((x, y))

        if not dead_coords:
            if pass_idx == 1:
                print("No persistent dead pixels detected. No files patched.")
            else:
                print("Converged: no additional persistent dead pixels remain.")
            break

        for folder in input_dirs:
            if args.in_place:
                output_folder = folder
            else:
                output_folder = folder.parent / f"{folder.name}{args.output_suffix}"

            file_count, pixel_updates = patch_folder(
                input_folder=folder,
                output_folder=output_folder,
                in_place=args.in_place,
                dead_coords=dead_coords,
                dead_mask=persistent_mask,
                ignore_mask=ignore_mask,
            )
            print(
                f"Patched {file_count} files in {folder} "
                f"(updated {pixel_updates} pixel values)."
            )
            if not args.in_place:
                print(f"Output folder: {output_folder}")

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as handle:
            for x, y in sorted(all_patched_coords):
                handle.write(f"{x},{y}\n")
        print(f"\nWrote coordinate report to {args.report_path}")


if __name__ == "__main__":
    main()
