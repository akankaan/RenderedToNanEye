import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path):
    # Load a raw reference image in integer mode.
    img = Image.open(path)
    img = img.convert("I")
    return np.array(img, dtype=np.float32)


def sobel_gradients(image):
    # Compute approximate gradients using Sobel kernels.
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = kx.T
    padded = np.pad(image, 1, mode="reflect")
    gx = np.empty_like(image)
    gy = np.empty_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            patch = padded[y : y + 3, x : x + 3]
            gx[y, x] = np.sum(patch * kx)
            gy[y, x] = np.sum(patch * ky)
    return gx, gy


def bilinear_sample(image, xs, ys):
    h, w = image.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    xa = xs - x0
    ya = ys - y0
    wa = (1 - xa) * (1 - ya)
    wb = xa * (1 - ya)
    wc = (1 - xa) * ya
    wd = xa * ya
    return (
        image[y0, x0] * wa
        + image[y0, x1] * wb
        + image[y1, x0] * wc
        + image[y1, x1] * wd
    )


def sample_edge_profile(image, center, normal, tangent, length=31, width=5):
    half = length // 2
    offsets = np.arange(-half, half + 1, dtype=np.float32)
    profile = np.zeros((width, length), dtype=np.float32)
    for i, t in enumerate(np.linspace(-(width // 2), width // 2, num=width, dtype=np.float32)):
        xs = center[1] + offsets * normal[1] + t * tangent[1]
        ys = center[0] + offsets * normal[0] + t * tangent[0]
        profile[i] = bilinear_sample(image, xs, ys)
    return np.mean(profile, axis=0)


def estimate_sigma_from_profile(profile):
    if profile.size < 5:
        return None
    smooth = np.convolve(profile, np.ones(5) / 5.0, mode="same")
    amin = float(np.min(smooth))
    amax = float(np.max(smooth))
    if amax <= amin:
        return None
    v10 = amin + 0.1 * (amax - amin)
    v90 = amin + 0.9 * (amax - amin)
    x = np.arange(len(smooth), dtype=np.float32)
    def interp_level(level):
        idx = np.where((smooth[:-1] <= level) & (smooth[1:] >= level))[0]
        if idx.size == 0:
            idx = np.where((smooth[:-1] >= level) & (smooth[1:] <= level))[0]
        if idx.size == 0:
            return None
        i = idx[0]
        frac = (level - smooth[i]) / (smooth[i + 1] - smooth[i]) if smooth[i + 1] != smooth[i] else 0.0
        return float(i + frac)
    x10 = interp_level(v10)
    x90 = interp_level(v90)
    if x10 is None or x90 is None or x90 <= x10:
        return None
    width = x90 - x10
    return width / 2.56


def estimate_blur(image, top_n=200, min_gradient=1e-2, sample_width=5):
    gx, gy = sobel_gradients(image)
    magnitude = np.hypot(gx, gy)
    h, w = image.shape
    indices = np.dstack(np.unravel_index(np.argsort(magnitude.ravel())[::-1], (h, w)))[0]
    sigmas = []
    count = 0
    for y, x in indices:
        if magnitude[y, x] < min_gradient:
            break
        if count >= top_n:
            break
        mag = magnitude[y, x]
        nx, ny = gx[y, x] / (mag + 1e-12), gy[y, x] / (mag + 1e-12)
        tangent = np.array([-ny, nx], dtype=np.float32)
        normal = np.array([nx, ny], dtype=np.float32)
        if np.any(np.isnan(normal)):
            continue
        if x < 20 or x >= w - 20 or y < 20 or y >= h - 20:
            continue
        profile = sample_edge_profile(image, (y, x), normal, tangent, length=41, width=sample_width)
        sigma = estimate_sigma_from_profile(profile)
        if sigma is not None and 0.1 < sigma < 10:
            sigmas.append(sigma)
            count += 1
    if len(sigmas) == 0:
        return None
    return float(np.median(sigmas)), float(np.mean(sigmas)), float(np.std(sigmas)), len(sigmas)


def list_images(folder):
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}
    folder_path = Path(folder)
    return sorted([p for p in folder_path.iterdir() if p.suffix.lower() in suffixes])


def main():
    parser = argparse.ArgumentParser(description="Estimate Gaussian blur radius from NanEye reference images.")
    parser.add_argument("folder", type=str, help="Folder containing reference images.")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum number of images to analyze.")
    parser.add_argument("--top-edges", type=int, default=120, help="Number of strong edges to sample per image.")
    parser.add_argument("--output", type=str, default="blur_estimate.json", help="JSON output file.")
    args = parser.parse_args()

    image_paths = list_images(args.folder)[: args.max_images]
    results = []
    for path in image_paths:
        image = load_image(path)
        estimate = estimate_blur(image, top_n=args.top_edges)
        if estimate is not None:
            median_sigma, mean_sigma, sigma_std, samples = estimate
            results.append(
                {
                    "image": str(path.name),
                    "median_sigma": median_sigma,
                    "mean_sigma": mean_sigma,
                    "sigma_std": sigma_std,
                    "samples": samples,
                }
            )
            print(f"{path.name}: median σ={median_sigma:.3f}, mean σ={mean_sigma:.3f}, samples={samples}")
        else:
            print(f"{path.name}: no valid edge estimates")

    if not results:
        print("No blur estimates could be computed.")
        return

    medians = [r["median_sigma"] for r in results]
    print("\nSummary:")
    print(f"Median of medians: {np.median(medians):.3f}")
    print(f"Mean of medians: {np.mean(medians):.3f}")
    print(f"Std of medians: {np.std(medians):.3f}")

    import json

    output = {
        "results": results,
        "summary": {
            "median_of_medians": float(np.median(medians)),
            "mean_of_medians": float(np.mean(medians)),
            "std_of_medians": float(np.std(medians)),
            "image_count": len(results),
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Written output to {args.output}")


if __name__ == "__main__":
    main()
