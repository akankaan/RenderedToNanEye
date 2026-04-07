import numpy as np
from PIL import Image, ImageFilter

# File names
input_path = "input.png"
output_path = "output.png"

# NanEye resolution
width = 320
height = 320

# Need to calibrate these
full_scale_dn = 860.0 # full scale digital num before value gets clipped
read_noise_dn = 1
prnu_std = 0.013
dsnu_std = 0.84
row_noise_std = 1

blur_radius = 0.3
seed = 42

# Distortion section, use negative k1 for barrel distortion
k1 = -0.08


def load_image_linear(path):
    # Load grayscale image. The function loads as an 8-bit depth image but
    # NanEye images are 10-bit depth in reality.
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def bilinear_interpolate(image, x, y):
    h, w = image.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return (
        image[y0, x0] * wa
        + image[y0, x1] * wb
        + image[y1, x0] * wc
        + image[y1, x1] * wd
    )


def apply_optics(gray, width, height, blur_radius, k1):
    # Simple optical stage: blur, then distortion, then resample to sensor.
    img = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img = img.resize((width, height), Image.Resampling.BICUBIC)

    gray = np.array(img, dtype=np.float32) / 255.0
    h, w = gray.shape

    y, x = np.indices((h, w), dtype=np.float32)
    cx = w / 2.0
    cy = h / 2.0

    # Normalize coordinates to [-1, 1]
    x_n = (x - cx) / (w / 2.0)
    y_n = (y - cy) / (h / 2.0)

    r2 = x_n**2 + y_n**2
    factor = 1 + k1 * r2

    x_dist = x_n * factor
    y_dist = y_n * factor

    x_src = x_dist * (w / 2.0) + cx
    y_src = y_dist * (h / 2.0) + cy

    x_src = np.clip(x_src, 0, w - 1)
    y_src = np.clip(y_src, 0, h - 1)

    return bilinear_interpolate(gray, x_src, y_src)


def convert_to_electrons(gray, full_scale_dn):
    # Convert normalized intensity to sensor units.
    return gray * full_scale_dn


def apply_prnu(signal_dn, prnu_std, rng):
    # Photo Response Non-Uniformity (PRNU) is a multiplicative fixed-pattern.
    prnu_map = rng.normal(0, prnu_std, signal_dn.shape)
    return signal_dn * (1 + prnu_map)


def apply_shot_noise(signal_dn, rng):
    # Photon shot noise is described by the Poisson process.
    signal_dn = np.clip(signal_dn, 0, None)
    return rng.poisson(signal_dn).astype(np.float32)


def apply_dark_current(signal_dn, dsnu_std, rng, dark_current_mean=0.0):
    # Add dark current fixed-pattern noise and dark shot noise.
    dsnu_map = rng.normal(0, dsnu_std, signal_dn.shape)
    dark_signal = np.clip(dark_current_mean + dsnu_map, 0, None)
    dark_noise = rng.poisson(dark_signal).astype(np.float32)
    return signal_dn + dark_noise


def apply_readout_noise(signal_dn, read_noise_dn, row_noise_std, rng):
    # Read noise and row noise are added after charge-to-voltage conversion.
    noisy = signal_dn + rng.normal(0, read_noise_dn, signal_dn.shape)
    noisy += rng.normal(0, row_noise_std, (signal_dn.shape[0], 1))
    return noisy


def adc_quantize(signal_dn, full_scale_dn):
    # Clip to sensor full-scale and quantize to integer DN.
    signal_dn = np.clip(signal_dn, 0, full_scale_dn)
    signal_dn = np.round(signal_dn)
    return signal_dn / full_scale_dn


def save_image(out, output_path):
    # Save the result scaled to 8-bit for visualization.
    out_img = Image.fromarray((out * 255).astype(np.uint8), mode="L")
    out_img = out_img.resize((640, 640), Image.Resampling.NEAREST)
    out_img.save(output_path)


def main():
    gray = load_image_linear(input_path)
    gray = apply_optics(gray, width, height, blur_radius, k1)
    signal_dn = convert_to_electrons(gray, full_scale_dn)

    rng = np.random.default_rng(seed)

    signal_dn = apply_prnu(signal_dn, prnu_std, rng)
    signal_dn = apply_shot_noise(signal_dn, rng)
    signal_dn = apply_dark_current(signal_dn, dsnu_std, rng)
    signal_dn = apply_readout_noise(signal_dn, read_noise_dn, row_noise_std, rng)

    out = adc_quantize(signal_dn, full_scale_dn)
    save_image(out, output_path)

    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
