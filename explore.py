"""
Explore a grid image: zoom in, re-simulate at full detail, and click pixels
to find their starting angles.

Interactive:
    python explore.py grid_t30s.png

    - Hover: the toolbar readout shows x = initial theta1, y = initial theta2
      under the cursor (in radians).
    - Zoom: use the toolbar's magnifier (or pan) as usual. That only enlarges
      the existing pixels.
    - E: "enhance". Re-simulates exactly the current view at full detail
      (--zoom-points pendulums across), up to the same time as the image.
      On a roughness map (from roughness.py or render(roughness=True)) it
      makes a roughness map of the view, averaged over the run.
      Progress shows in the terminal; the window pauses until it's done.
    - U: undo, back to the previous view.
    - W: write the current view to a pixel-exact PNG (with its angle ranges
      stored inside, so it can be opened with explore.py later).
    - Click: prints that pixel's exact starting angles and a quick chaos
      check, marks the pixel, and opens one window with that pendulum's
      animation, θ₁ vs θ₂ curve, angles over time and energy check.
      (Clicks are ignored while the zoom or pan tool is active.)

Without a window, for a pixel position you read off an image viewer
(x = column from the left, y = row from the top, both starting at 0):
    python explore.py grid_t30s.png --pixel 812 1040

Images saved by grid.py record their angle ranges and time inside the PNG.
Older images don't, so -pi to pi is assumed for both axes; override with
--range1 / --range2 if your image covered something else.
"""

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from grid import PendulumGrid
from pendulum import Pendulum
from viz import PendulumViz


@dataclass
class View:
    """One image being shown: its pixels and the starting angles it covers."""

    pixels: np.ndarray  # (rows, cols, 3) uint8
    theta1_range: tuple[float, float]  # left column .. right column
    theta2_range: tuple[float, float]  # bottom row .. top row
    grid: PendulumGrid | None = None  # set for views made with E (used by W)

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape[0], self.pixels.shape[1]

    def extent(self) -> tuple[float, float, float, float]:
        """imshow bounds: each pixel is centered on its angles."""
        rows, cols = self.shape
        (lo1, hi1), (lo2, hi2) = self.theta1_range, self.theta2_range
        h1 = (hi1 - lo1) / (cols - 1) / 2
        h2 = (hi2 - lo2) / (rows - 1) / 2
        return (lo1 - h1, hi1 + h1, lo2 - h2, hi2 + h2)

    def pixel_at(self, theta1: float, theta2: float) -> tuple[int, int]:
        """Nearest pixel (x from left, y from top) to a point on the axes."""
        rows, cols = self.shape
        (lo1, hi1), (lo2, hi2) = self.theta1_range, self.theta2_range
        x = round((theta1 - lo1) / (hi1 - lo1) * (cols - 1))
        y = round((hi2 - theta2) / (hi2 - lo2) * (rows - 1))
        return int(np.clip(x, 0, cols - 1)), int(np.clip(y, 0, rows - 1))


def read_image_info(path: str, args) -> dict:
    """The image's pixels and grid settings (from the PNG, the command line, or defaults)."""
    image = Image.open(path)
    meta = image.info

    def parse_range(key, override):
        if override is not None:
            return tuple(override)
        if key in meta:
            lo, hi = meta[key].split(",")
            return float(lo), float(hi)
        return (-np.pi, np.pi)

    pendulum_kwargs = {}
    if "pendulum" in meta:
        for item in meta["pendulum"].split(","):
            name, value = item.split("=")
            pendulum_kwargs[name] = float(value)

    return {
        "pixels": np.asarray(image.convert("RGB")),
        "theta1_range": parse_range("theta1_range", args.range1),
        "theta2_range": parse_range("theta2_range", args.range2),
        "time": float(meta["time"]) if "time" in meta else None,
        "pendulum_kwargs": pendulum_kwargs,
        "has_metadata": "theta1_range" in meta,
        "kind": meta.get("kind", "angles"),
    }


def pixel_to_angles(
    x: int, y: int, shape, theta1_range, theta2_range
) -> tuple[float, float]:
    """
    Starting angles of the pendulum drawn at pixel (x, y).

    x is the column from the left and y is the row from the top, both from 0,
    as image viewers report them. Pixel centers sit exactly on the grid's
    angles: the left column is theta1_range[0], the top row is theta2_range[1].
    """
    rows, cols = shape[:2]
    # np.linspace, exactly as grid.py builds the grid, so the angles match the
    # simulated pendulum to the last digit. (That matters for chaotic pixels,
    # where even a 1e-16 rad difference changes the outcome completely.)
    theta1 = np.linspace(theta1_range[0], theta1_range[1], cols)[x]
    theta2 = np.linspace(theta2_range[0], theta2_range[1], rows)[
        rows - 1 - y
    ]  # row 0 is the top
    return float(theta1), float(theta2)


def chaos_check(pendulum_kwargs: dict, theta1: float, theta2: float, t_end: float):
    """
    Run the pendulum and a copy started 1e-8 rad away, and compare after t_end.

    Regular (island) motion: the gap stays tiny, growing roughly in step with
    time. Chaotic motion: the gap grows exponentially until the two runs are
    unrelated (a gap of order 1 or more).
    """
    eps = 1e-8
    p = Pendulum(**pendulum_kwargs)
    _, a = p.simulate(t_end, dt=t_end, state0=np.array([theta1, theta2, 0.0, 0.0]))
    _, b = p.simulate(
        t_end, dt=t_end, state0=np.array([theta1 + eps, theta2, 0.0, 0.0])
    )
    gap = float(np.linalg.norm(a[:, -1] - b[:, -1]))
    verdict = "looks REGULAR (island)" if gap < 1e-3 else "looks CHAOTIC"
    return gap, verdict


def describe(view: View, pendulum_kwargs: dict, x: int, y: int, t_check: float):
    """Print everything about one pixel and return its starting angles."""
    theta1, theta2 = pixel_to_angles(
        x, y, view.shape, view.theta1_range, view.theta2_range
    )
    gap, verdict = chaos_check(pendulum_kwargs, theta1, theta2, t_check)
    print(
        f"\npixel (x={x}, y={y})  color {tuple(int(c) for c in view.pixels[y, x])}\n"
        f"  theta1 = {theta1:.10f} rad ({np.degrees(theta1):.6f} deg)\n"
        f"  theta2 = {theta2:.10f} rad ({np.degrees(theta2):.6f} deg)\n"
        f"  chaos check over {t_check:g} s: runs 1e-8 rad apart end {gap:.1e} apart -> {verdict}\n"
        f"  Pendulum(theta1={theta1!r}, theta2={theta2!r})"
    )
    return theta1, theta2


class Explorer:
    """The interactive window: one image at a time, with a history for undo."""

    def __init__(self, first_view: View, info: dict, args, image_name: str):
        self.history = [first_view]
        self.pendulum_kwargs = info["pendulum_kwargs"]
        self.t_image = args.check_time or info["time"] or 30.0
        self.zoom_points = args.zoom_points
        self.animate_clicks = not args.no_animate
        self.image_name = image_name
        # Roughness maps (and quick previews) re-simulate as roughness maps
        self.roughness = str(info["kind"]).startswith("roughness")
        self.open_animations = []  # keep references so the animations keep running
        self.saved_count = 0

        self.fig, self.ax = plt.subplots(figsize=(8.5, 8.8))
        self.image = self.ax.imshow(
            first_view.pixels, extent=first_view.extent(), interpolation="nearest"
        )
        (self.marker,) = self.ax.plot([], [], "+", color="white", ms=16, mew=2)
        self.ax.set_xlabel("initial θ₁ (rad)")
        self.ax.set_ylabel("initial θ₂ (rad)")
        self._show(first_view)
        self.fig.tight_layout()

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    @property
    def view(self) -> View:
        return self.history[-1]

    # --- Display ---------------------------------------------------------

    def _show(self, view: View, status: str | None = None):
        """Display a view, fitting the axes to it."""
        self.image.set_data(view.pixels)
        left, right, bottom, top = view.extent()
        self.image.set_extent((left, right, bottom, top))
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)
        self.marker.set_data([], [])
        self._set_title(status)
        self.fig.canvas.draw_idle()

    def _set_title(self, status: str | None = None):
        rows, cols = self.view.shape
        (lo1, hi1), (lo2, hi2) = self.view.theta1_range, self.view.theta2_range
        level = (
            "original image"
            if len(self.history) == 1
            else f"zoom level {len(self.history) - 1}"
        )
        kind = (
            "roughness map (dark = island, light = chaos)"
            if self.roughness
            else "angle colors"
        )
        lines = [
            f"{self.image_name}, t = {self.t_image:g} s, {kind}",
            f"{level} ({cols} x {rows} px)",
            f"θ₁ {lo1:.6g} … {hi1:.6g},  θ₂ {lo2:.6g} … {hi2:.6g}",
            status
            or "zoom with the toolbar, then E = re-simulate view, U = undo, W = save, click = inspect",
        ]
        self.ax.set_title("\n".join(lines), fontsize=9)

    # --- Events ------------------------------------------------------------

    def on_key(self, event):
        if event.key == "e":
            self.enhance()
        elif event.key == "u":
            self.undo()
        elif event.key == "w":
            self.write_png()

    def on_click(self, event):
        if event.inaxes is not self.ax or event.button != 1 or event.xdata is None:
            return
        toolbar = self.fig.canvas.toolbar
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # zoom/pan tool is active
        x, y = self.view.pixel_at(event.xdata, event.ydata)
        theta1, theta2 = describe(self.view, self.pendulum_kwargs, x, y, self.t_image)
        self.marker.set_data([theta1], [theta2])
        self.fig.canvas.draw_idle()

        if self.animate_clicks:
            pendulum = Pendulum(**self.pendulum_kwargs, theta1=theta1, theta2=theta2)
            viz = PendulumViz(pendulum, t_end=self.t_image)
            self.open_animations.append(
                viz.animate(
                    title=f"θ₁ = {theta1:.8f}, θ₂ = {theta2:.8f} rad (starting at rest)"
                )
            )
            plt.gcf().show()

    # --- Actions ------------------------------------------------------------

    def enhance(self):
        """Re-simulate the visible region at full detail and show it."""
        lo1, hi1 = sorted(self.ax.get_xlim())
        lo2, hi2 = sorted(self.ax.get_ylim())

        # Square pixels in angle space: the longer side gets zoom_points
        width, height = hi1 - lo1, hi2 - lo2
        if width >= height:
            n1, n2 = self.zoom_points, max(2, round(self.zoom_points * height / width))
        else:
            n1, n2 = max(2, round(self.zoom_points * width / height)), self.zoom_points

        self._set_title(f"simulating {n1 * n2:,} pendulums… (progress in the terminal)")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        print(f"\nRe-simulating θ₁ {lo1:.8g}…{hi1:.8g}, θ₂ {lo2:.8g}…{hi2:.8g}")
        grid = PendulumGrid(
            points=(n1, n2),
            theta1_range=(lo1, hi1),
            theta2_range=(lo2, hi2),
            pendulum=Pendulum(**self.pendulum_kwargs),
        )
        pixels = grid.render(
            t_end=self.t_image,
            video_path=None,
            image_pattern=None,
            roughness=self.roughness,
            roughness_path=None,
        )
        if self.roughness and grid.roughness_image is not None:
            pixels = grid.roughness_image
        view = View(pixels, (lo1, hi1), (lo2, hi2), grid)
        self.history.append(view)
        self._show(view)

    def undo(self):
        if len(self.history) == 1:
            print("\nAlready at the original image.")
            return
        self.history.pop()
        self._show(self.view)

    def write_png(self):
        """Save the current view's pixels (not the window) with its angle ranges."""
        view = self.view
        if view.grid is None:
            print(
                "\nThis is the original image; press E first to make a new view to save."
            )
            return
        self.saved_count += 1
        stem = os.path.splitext(self.image_name)[0]
        path = f"{stem}_zoom{self.saved_count}.png"
        view.grid.save_png(
            path,
            view.pixels,
            self.t_image,
            kind="roughness" if self.roughness else "angles",
        )
        print(f"\nSaved {path} ({view.shape[1]} x {view.shape[0]} px)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", help="PNG saved by grid.py (save_image or render)")
    parser.add_argument(
        "--pixel",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="print one pixel's angles and exit (x from left, y from top)",
    )
    parser.add_argument(
        "--range1",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="theta1 range of the image, if not stored in it",
    )
    parser.add_argument(
        "--range2",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="theta2 range of the image, if not stored in it",
    )
    parser.add_argument(
        "--check-time",
        type=float,
        default=None,
        help="seconds for re-simulating, the chaos check and animations "
        "(default: the image's time, or 30)",
    )
    parser.add_argument(
        "--zoom-points",
        type=int,
        default=800,
        help="pendulums across the longer side when re-simulating a "
        "zoomed view with E (default 800)",
    )
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="on click, only print (don't open the animation)",
    )
    args = parser.parse_args()

    info = read_image_info(args.image, args)
    first_view = View(info["pixels"], info["theta1_range"], info["theta2_range"])
    rows, cols = first_view.shape
    if not info["has_metadata"]:
        (lo1, hi1), (lo2, hi2) = first_view.theta1_range, first_view.theta2_range
        print(
            f"No angle ranges stored in {args.image}: assuming theta1 {lo1:.4f}..{hi1:.4f}, "
            f"theta2 {lo2:.4f}..{hi2:.4f}"
        )

    if args.pixel is not None:
        x, y = args.pixel
        if not (0 <= x < cols and 0 <= y < rows):
            parser.error(
                f"pixel must be within 0..{cols - 1} (x) and 0..{rows - 1} (y)"
            )
        t_check = args.check_time or info["time"] or 30.0
        describe(first_view, info["pendulum_kwargs"], x, y, t_check)
        return

    # Our keys: stop matplotlib from also treating them as its own shortcuts
    for keymap in ("keymap.back", "keymap.forward", "keymap.home", "keymap.save"):
        plt.rcParams[keymap] = [
            k for k in plt.rcParams[keymap] if k not in ("e", "u", "w")
        ]

    explorer = Explorer(
        first_view, info, args, os.path.basename(args.image)
    )  # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
