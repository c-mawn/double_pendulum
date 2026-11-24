"""
Neighbor-difference ("roughness") maps: make islands of stability stand out.

Neighboring pixels start almost identically. Inside an island the motion is
regular, so neighbors stay close together and the image is smooth. In chaotic
regions neighbors quickly become unrelated, so the image is noisy. A
roughness map colors each pixel by how different it is from its neighbors:

    dark  = smooth  = regular motion (islands)
    light = rough   = chaos

Brightness is on a log scale, because differences range from tiny fractions
of a radian (islands) up to about a radian (chaos).

Quick preview from any image grid.py already saved (instant, no simulation,
uses just that one frame):
    python roughness.py grid_t30s.png           -> grid_t30s_roughness.png

The better version is computed during a simulation, averaged over the whole
run at full precision (see PendulumGrid.render(roughness=True)):
    grid = PendulumGrid(points=2000)
    grid.render(t_end=30, video_path=None, image_pattern=None, roughness=True)
"""

import argparse
import os

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Roughness of fully chaotic pixels: unrelated neighbors differ by about a
# radian or two on average, i.e. log10 ~ 0.3. Anything at or above this is white.
CHAOS_LOG10 = 0.3


def wrap_difference(d: np.ndarray) -> np.ndarray:
    """Wrap angle differences into [-pi, pi), so a full flip counts as no change."""
    return (d + np.pi) % (2 * np.pi) - np.pi


def neighbor_difference(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """
    For each pixel, the average angle distance to its up/down/left/right neighbors.

    Distance between two pendulums is sqrt(dθ₁² + dθ₂²), with each difference
    wrapped so that a full flip (2π) counts as zero. Edge pixels average over
    the neighbors they have.

    Args:
        theta1, theta2: current angles, shape (rows, cols)

    Returns:
        Average neighbor distance in radians, shape (rows, cols)
    """
    total = np.zeros(theta1.shape)
    count = np.zeros(theta1.shape)

    # Left-right neighbor pairs
    d = np.hypot(
        wrap_difference(theta1[:, 1:] - theta1[:, :-1]),
        wrap_difference(theta2[:, 1:] - theta2[:, :-1]),
    )
    total[:, 1:] += d
    total[:, :-1] += d
    count[:, 1:] += 1
    count[:, :-1] += 1

    # Up-down neighbor pairs
    d = np.hypot(
        wrap_difference(theta1[1:, :] - theta1[:-1, :]),
        wrap_difference(theta2[1:, :] - theta2[:-1, :]),
    )
    total[1:, :] += d
    total[:-1, :] += d
    count[1:, :] += 1
    count[:-1, :] += 1

    return total / count


def log_roughness(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """log10 of neighbor_difference, safe for exact zeros."""
    return np.log10(np.maximum(neighbor_difference(theta1, theta2), 1e-15))


def roughness_image(
    log10_roughness: np.ndarray, smooth_log10: float, invert: bool = False
) -> np.ndarray:
    """
    Turn log10 roughness into an 8-bit grayscale RGB image.

    Args:
        log10_roughness: shape (rows, cols); row 0 = lowest theta2
        smooth_log10: log10 roughness drawn as pure black. Tie it to the
            pixel spacing (neighbors in an island stay within a few times
            their starting spacing), so the scale adapts when you zoom in.
        invert: draw islands light and chaos dark instead

    Returns:
        uint8 array (rows, cols, 3), flipped so image row 0 is the highest theta2
    """
    value = (log10_roughness - smooth_log10) / (CHAOS_LOG10 - smooth_log10)
    value = np.clip(value, 0.0, 1.0)
    if invert:
        value = 1.0 - value
    gray = np.round(value * 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(gray[::-1, :, None], 3, axis=2))


def from_png(path: str, out_path: str, invert: bool = False):
    """Quick single-frame roughness map from an image saved by grid.py."""
    image = Image.open(path)
    pixels = np.asarray(image.convert("RGB"))
    # Undo the color mapping: green = theta1, blue = theta2, as bytes 0..255
    # spanning -pi..pi. Flip so row 0 is the lowest theta2, like the grid.
    theta1 = pixels[::-1, :, 1] / 255.0 * 2 * np.pi - np.pi
    theta2 = pixels[::-1, :, 2] / 255.0 * 2 * np.pi - np.pi

    # The image only stores angles to the nearest 1/255 of a turn, so anything
    # smoother than that step can't be distinguished: draw that as black.
    step = 2 * np.pi / 255
    out = roughness_image(log_roughness(theta1, theta2), np.log10(step / 2), invert)

    info = PngInfo()
    for key, value in image.info.items():  # keep the angle ranges, time, etc.
        if isinstance(key, str) and isinstance(value, str) and key != "kind":
            info.add_text(key, value)
    info.add_text("kind", "roughness-preview")
    Image.fromarray(out).save(out_path, pnginfo=info)


def main():
    parser = argparse.ArgumentParser(
        description="Quick roughness map from a grid image (one frame, no simulation)."
    )
    parser.add_argument("image", help="PNG saved by grid.py")
    parser.add_argument(
        "-o", "--output", help="output PNG (default: <image>_roughness.png)"
    )
    parser.add_argument(
        "--invert", action="store_true", help="islands light, chaos dark"
    )
    args = parser.parse_args()
    out_path = args.output or os.path.splitext(args.image)[0] + "_roughness.png"
    from_png(args.image, out_path, args.invert)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
