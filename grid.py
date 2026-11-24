"""
A grid of double pendulums, one for each pair of starting angles (theta1, theta2).

Each pendulum stays at its *starting* position on the plot (x = initial theta1,
y = initial theta2), but is colored by its *current* angles:
    current theta1 -> green channel (more green as theta1 increases)
    current theta2 -> blue channel  (more blue as theta2 increases)

At t = 0 this is a smooth color gradient. As time runs, regions where the
motion is regular keep smooth colors, while chaotic regions scramble into noise.

Workflow:
    grid = PendulumGrid(points=2000)     # or resolution=0.05 (rad)
    grid.simulate(t_end=10)              # all the physics, with a progress line
    grid.save_image("t8.png", t=8)       # pixel-exact: one pendulum per pixel
    grid.save_video("grid.mp4")          # pixel-exact video
    grid.plot(t=8)                       # matplotlib figure with axes + key
    anim = grid.animate()                # matplotlib animation with axes + key

save_image / save_video keep every pendulum. plot / animate draw into a
matplotlib figure, which shrinks large grids to fit the window.

simulate() keeps every frame in memory (2 bytes per pendulum per frame), which
gets large for long runs. For those, use render() instead: it writes the video
and images as it goes and keeps only the current state in memory:
    grid = PendulumGrid(points=2000)
    grid.render(t_end=60, video_path="grid.mp4", image_every=10)

Run this file directly for a demo:
    python grid.py
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import grid_kernel
import roughness as roughness_map
from pendulum import Pendulum

RED_LEVEL = 0.25  # constant red channel, so no pendulum is drawn pure black


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class FixedPaletteGifWriter(PillowWriter):
    """
    GIF writer that uses ONE color palette for every frame.

    A GIF frame can only hold 256 colors. matplotlib's PillowWriter lets each
    frame pick its own 256, so as the live panel fills with new colors the
    palette shifts and unchanging parts (like the color key) flicker. Here the
    palette is built once from the first frame, whose live panel shows every
    color the grid can produce, and every frame is matched to that palette.
    """

    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        self.rgb_frames: list[Image.Image] = []

    def grab_frame(self, **savefig_kwargs):
        """Render the current figure and keep it as an RGB image."""
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
        self.rgb_frames.append(Image.fromarray(rgba[..., :3].copy()))

    def finish(self):
        frames = self.rgb_frames
        palette = frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        frames_p = [
            frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
            for frame in frames
        ]
        frames_p[0].save(
            self.outfile,
            save_all=True,
            append_images=frames_p[1:],
            duration=int(1000 / self.fps),
            loop=0,
        )


class Progress:
    """
    A one-line progress report that rewrites itself in place, e.g.

        130,000 / 395,641 pendulums (32.9%) | 0:00:41 elapsed | ~0:01:24 left
    """

    def __init__(self, total: int, unit: str):
        self.total = total
        self.unit = unit
        self.start = time.perf_counter()

    @staticmethod
    def _clock(seconds: float) -> str:
        seconds = int(round(seconds))
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"

    def update(self, done: int):
        elapsed = time.perf_counter() - self.start
        line = (
            f"\r  {done:,} / {self.total:,} {self.unit} ({100 * done / self.total:.1f}%)"
            f" | {self._clock(elapsed)} elapsed"
        )
        if 0 < done < self.total:
            line += f" | ~{self._clock(elapsed / done * (self.total - done))} left"
        sys.stdout.write(line.ljust(80))
        sys.stdout.flush()
        if done >= self.total:
            sys.stdout.write("\n")


def rk4_step(f, state: np.ndarray, dt: float) -> np.ndarray:
    """One fixed-size Runge-Kutta 4 step of d(state)/dt = f(t, state)."""
    k1 = f(0.0, state)
    k2 = f(0.0, state + 0.5 * dt * k1)
    k3 = f(0.0, state + 0.5 * dt * k2)
    k4 = f(0.0, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """
    Wrap angles into [-pi, pi], so a full flip returns to the same color.

    Angles already exactly at -pi or +pi are left alone (rather than +pi
    being sent to -pi), so the color key's edges stay a clean gradient.
    """
    return theta - 2 * np.pi * np.round(theta / (2 * np.pi))


def angle_to_byte(theta: np.ndarray) -> np.ndarray:
    """Wrap an angle and store it as 0..255 (-pi -> 0, +pi -> 255)."""
    fraction = (wrap_angle(theta) + np.pi) / (2 * np.pi)
    return np.round(fraction * 255).astype(np.uint8)


def byte_colors(byte1: np.ndarray, byte2: np.ndarray) -> np.ndarray:
    """
    RGB colors (floats in [0, 1]) from stored angle bytes, shape (..., 3).

    theta1 sets green, theta2 sets blue, red is constant.
    """
    green = byte1 / 255.0
    blue = byte2 / 255.0
    red = np.full_like(green, RED_LEVEL)
    return np.stack([red, green, blue], axis=-1)


# ----------------------------------------------------------------------
# The grid
# ----------------------------------------------------------------------


class PendulumGrid:
    """
    Many pendulums laid out on a grid of starting angles.
    """

    def __init__(
        self,
        resolution: float = 0.05,
        theta1_range: tuple[float, float] = (-np.pi, np.pi),
        theta2_range: tuple[float, float] = (-np.pi, np.pi),
        pendulum: Pendulum | None = None,
        points: int | tuple[int, int] | None = None,
    ):
        """
        Args:
            resolution: spacing between neighboring pendulums (rad).
                e.g. 0.1 -> one pendulum every 0.1 rad. Smaller is sharper
                but slower: the count grows like 1 / resolution^2.
            theta1_range: (min, max) starting theta1, the x axis (rad)
            theta2_range: (min, max) starting theta2, the y axis (rad)
            pendulum: supplies the masses, lengths and g. Its own starting
                angles are ignored. Defaults to Pendulum() (1 kg, 1 m).
                Every pendulum starts at rest.
            points: exact number of pendulums along each axis, instead of a
                resolution. An int gives a square grid (points=2000 ->
                2000 x 2000); a tuple is (theta1 count, theta2 count). This
                is the easy way to get one pendulum per pixel in save_image()
                and save_video().
        """
        self.pendulum = pendulum if pendulum is not None else Pendulum()

        if points is not None:
            n1, n2 = (points, points) if isinstance(points, int) else points
            self.theta1_values = np.linspace(*theta1_range, n1)
            self.theta2_values = np.linspace(*theta2_range, n2)
        else:
            # Pendulums sit exactly on both ends of each range. If the range
            # isn't a whole multiple of the resolution, the spacing is nudged
            # slightly (e.g. 0.05 -> 0.04987 for -pi to pi) so the ends line up.
            self.theta1_values = self._axis_values(theta1_range, resolution)
            self.theta2_values = self._axis_values(theta2_range, resolution)
        self.resolution = float(self.theta1_values[1] - self.theta1_values[0])

        # Filled in by simulate()
        self.times: np.ndarray | None = None
        self.frames: np.ndarray | None = None
        self._sim_settings: tuple[float, float, float] | None = None
        self.roughness_image: np.ndarray | None = None  # set by render(roughness=True)
        self.roughness_log10: np.ndarray | None = None  # the raw values behind it

    @staticmethod
    def _axis_values(value_range: tuple[float, float], resolution: float) -> np.ndarray:
        """Evenly spaced angles covering value_range, about `resolution` apart."""
        lo, hi = value_range
        n = max(1, int(round((hi - lo) / resolution))) + 1
        return np.linspace(lo, hi, n)

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns) = (number of theta2 values, number of theta1 values)."""
        return self.theta2_values.size, self.theta1_values.size

    @property
    def count(self) -> int:
        """Number of pendulums in the grid."""
        return self.theta1_values.size * self.theta2_values.size

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_end: float = 10.0,
        frame_dt: float = 0.05,
        dt: float = 0.01,
        batch_size: int | None = None,
        threads: int | None = None,
        use_numba: bool = True,
        verbose: bool = True,
    ):
        """
        Run every pendulum from t = 0 to t_end and store its angles at each frame.

        The grid is processed in batches: each batch of pendulums is run all
        the way to t_end before the next starts. That's what makes progress
        reports possible, and it also keeps memory use small during the
        physics. Results are stored as 2 bytes per pendulum per frame.

        Uses the compiled, multi-core Numba version when Numba is installed,
        and the NumPy version otherwise.

        Args:
            t_end: how long to run (s)
            frame_dt: simulated time between stored frames (s)
            dt: integration step (s). 0.01 s keeps energy errors small for
                typical starts; chaotic pendulums still diverge from the exact
                solution eventually, as any simulation of chaos does.
            batch_size: pendulums per batch; progress updates after each one.
                Defaults to 100,000 with Numba and 10,000 with NumPy.
            threads: CPU threads for Numba. Defaults to all but 2.
            use_numba: set False to force the NumPy version
            verbose: print the setup summary and progress line
        """
        n_frames = int(round(t_end / frame_dt)) + 1
        steps_per_frame = max(1, int(round(frame_dt / dt)))
        step = frame_dt / steps_per_frame  # exact, so frames land on frame_dt multiples

        fast = use_numba and grid_kernel.NUMBA_AVAILABLE
        if batch_size is None:
            batch_size = 100_000 if fast else 10_000

        frame_bytes = n_frames * self.count * 2
        if verbose:
            print(
                f"Simulating {self.count:,} pendulums for {t_end:g} s "
                f"({n_frames} frames, {(n_frames - 1) * steps_per_frame:,} steps each)\n"
                f"  stored frames will use about {frame_bytes / 1e9:.2f} GB of memory"
            )
            if fast:
                print(f"  using Numba on {grid_kernel.set_threads(threads)} threads")
            elif use_numba:
                print("  Numba not installed: using the slower NumPy version")
        elif fast:
            grid_kernel.set_threads(threads)

        # frames[k, row, col, 0 or 1] = theta1 or theta2 of that pendulum at frame k
        frames = np.empty((n_frames, *self.shape, 2), dtype=np.uint8)
        flat = frames.reshape(n_frames, self.count, 2)  # a view: writes go into frames

        th1_all, th2_all = (
            a.ravel() for a in np.meshgrid(self.theta1_values, self.theta2_values)
        )
        flat[0, :, 0] = angle_to_byte(th1_all)
        flat[0, :, 1] = angle_to_byte(th2_all)

        if fast and verbose:
            print("  compiling (first run only, a few seconds)...", end="", flush=True)
            t0 = time.perf_counter()
            self._simulate_batch_numba(th1_all[:2], th2_all[:2], flat[:2, :2], 1, step)
            print(f" done in {time.perf_counter() - t0:.1f} s")

        progress = Progress(self.count, "pendulums") if verbose else None
        for start in range(0, self.count, batch_size):
            stop = min(start + batch_size, self.count)
            if fast:
                self._simulate_batch_numba(
                    th1_all[start:stop],
                    th2_all[start:stop],
                    flat[:, start:stop],
                    steps_per_frame,
                    step,
                )
            else:
                self._simulate_batch_numpy(
                    th1_all[start:stop],
                    th2_all[start:stop],
                    flat[:, start:stop],
                    steps_per_frame,
                    step,
                )
            if progress is not None:
                progress.update(stop)

        self.times = np.arange(n_frames) * frame_dt
        self.frames = frames
        self._sim_settings = (t_end, frame_dt, dt)

    def _simulate_batch_numba(self, th1, th2, out, steps_per_frame, step):
        """Fill frames 1+ of `out` for one batch, using the compiled kernel."""
        p = self.pendulum
        grid_kernel.simulate_batch(
            np.ascontiguousarray(th1),
            np.ascontiguousarray(th2),
            out,
            steps_per_frame,
            step,
            p.m1,
            p.m2,
            p.l1,
            p.l2,
            p.g,
        )

    def _simulate_batch_numpy(self, th1, th2, out, steps_per_frame, step):
        """Fill frames 1+ of `out` for one batch, with NumPy (no Numba needed)."""
        f = self.pendulum.derivatives
        state = np.stack([th1, th2, np.zeros_like(th1), np.zeros_like(th1)])
        for k in range(1, out.shape[0]):
            for _ in range(steps_per_frame):
                state = rk4_step(f, state, step)
            out[k, :, 0] = angle_to_byte(state[0])
            out[k, :, 1] = angle_to_byte(state[1])

    def _ensure_simulated(self, t_end: float, frame_dt: float, dt: float):
        """Simulate only if there are no stored frames for these settings."""
        if self._sim_settings != (t_end, frame_dt, dt):
            self.simulate(t_end=t_end, frame_dt=frame_dt, dt=dt)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def colors(self, frame: int) -> np.ndarray:
        """Color of every pendulum at a stored frame, shape (rows, cols, 3)."""
        assert self.frames is not None, "call simulate() first"
        return byte_colors(self.frames[frame, ..., 0], self.frames[frame, ..., 1])

    # ------------------------------------------------------------------
    # Pixel-exact output: one pendulum = one pixel, no axes, no resizing
    # ------------------------------------------------------------------

    def pixel_image(self, frame: int) -> np.ndarray:
        """
        A stored frame as an 8-bit RGB image, one pixel per pendulum.

        Shape (rows, cols, 3). theta1 increases left to right and theta2
        increases bottom to top, matching the plots.
        """
        assert self.frames is not None, "call simulate() first"
        return self._bytes_to_image(
            self.frames[frame, ..., 0], self.frames[frame, ..., 1]
        )

    @staticmethod
    def _bytes_to_image(byte1: np.ndarray, byte2: np.ndarray) -> np.ndarray:
        """Angle bytes, shape (rows, cols), to an 8-bit RGB image with theta2 up."""
        rgb = np.empty((*byte1.shape, 3), dtype=np.uint8)
        rgb[..., 0] = round(RED_LEVEL * 255)
        rgb[..., 1] = byte1
        rgb[..., 2] = byte2
        return np.ascontiguousarray(
            rgb[::-1]
        )  # image row 0 is the top (highest theta2)

    @staticmethod
    def _open_video(path: str, size: tuple[int, int], fps: int, crf: int):
        """Start an ffmpeg process that accepts RGB frames of `size` (width, height)."""
        import imageio_ffmpeg  # only needed for video

        writer = imageio_ffmpeg.write_frames(
            path,
            size,
            fps=fps,
            codec="libx264",
            pix_fmt_out="yuv444p",
            quality=None,  # type: ignore[arg-type]  # None = use our -crf instead
            macro_block_size=1,
            output_params=["-crf", str(crf), "-preset", "veryfast"],
        )
        writer.send(None)  # start ffmpeg
        return writer

    def _frame_at(self, t: float) -> int:
        assert self.times is not None, "call simulate() first"
        return int(np.argmin(np.abs(self.times - t)))

    def save_image(self, path: str, t: float):
        """Save the stored frame nearest time t as a pixel-exact PNG."""
        assert self.times is not None, "call simulate() first"
        frame = self._frame_at(t)
        self.save_png(path, self.pixel_image(frame), float(self.times[frame]))

    def save_png(self, path: str, image: np.ndarray, t: float, kind: str = "angles"):
        """
        Save a pixel-exact PNG, recording the grid's angle ranges and time inside
        the file, so explore.py can map pixels back to starting angles.

        kind is "angles" for the usual colored image or "roughness" for a
        roughness map (explore.py uses it to re-simulate the same kind).
        """
        info = PngInfo()
        info.add_text("kind", kind)
        # float() so NumPy numbers are written as plain text like "3.14159..."
        t1, t2 = self.theta1_values, self.theta2_values
        info.add_text("theta1_range", f"{float(t1[0])!r},{float(t1[-1])!r}")
        info.add_text("theta2_range", f"{float(t2[0])!r},{float(t2[-1])!r}")
        info.add_text("time", f"{float(t)!r}")
        p = self.pendulum
        params = {"l1": p.l1, "l2": p.l2, "m1": p.m1, "m2": p.m2, "g": p.g}
        info.add_text(
            "pendulum", ",".join(f"{k}={float(v)!r}" for k, v in params.items())
        )
        Image.fromarray(image).save(path, pnginfo=info)

    def save_video(self, path: str = "grid.mp4", crf: int = 18):
        """
        Save every stored frame as a pixel-exact MP4 video, playing in real time.

        Uses full color resolution (yuv444p) so neighboring pixels keep their
        own colors. VLC and most desktop players handle this; some web
        browsers and QuickTime don't. Needs the imageio-ffmpeg package.

        Args:
            path: output file, ending in .mp4
            crf: quality, 0 (lossless, huge) to 51 (tiny, blurry). 18 looks
                essentially perfect; raise it to shrink the file.
        """
        assert self._sim_settings is not None, "call simulate() first"
        frame_dt = self._sim_settings[1]
        rows, cols = self.shape
        writer = self._open_video(path, (cols, rows), round(1 / frame_dt), crf)
        n_frames = len(self.times) if self.times is not None else 0
        print(f"Saving {path} ({cols} x {rows} px)")
        progress = Progress(n_frames, "frames")
        for k in range(n_frames):
            writer.send(self.pixel_image(k))
            progress.update(k + 1)
        writer.close()

    # ------------------------------------------------------------------
    # Long runs: stream frames straight to disk
    # ------------------------------------------------------------------

    def render(
        self,
        t_end: float = 60.0,
        frame_dt: float = 0.05,
        dt: float = 0.01,
        video_path: str | None = "grid.mp4",
        image_every: float | None = None,
        image_pattern: str | None = "grid_t{t:g}s.png",
        crf: int = 18,
        threads: int | None = None,
        use_numba: bool = True,
        roughness: bool = False,
        roughness_every: float = 0.5,
        roughness_path: str | None = "roughness_t{t:g}s.png",
        roughness_invert: bool = False,
        roughness_black: float = 3.0,
    ) -> np.ndarray:
        """
        Simulate and write the output as it goes, for long or large runs.

        Unlike simulate(), this keeps only the current state of each pendulum
        in memory (32 bytes each, ~130 MB for 2000 x 2000) and writes each
        frame to the video the moment it's computed. Memory use doesn't grow
        with t_end, so runs can be as long as you like. Nothing is kept
        afterward, so plot() / animate() / save_image() don't apply here;
        use image_every to get still images instead.

        Args:
            t_end: how long to run (s)
            frame_dt: simulated time between video frames (s); 0.05 -> 20 fps,
                so the video plays in real time
            dt: integration step (s)
            video_path: MP4 to write, or None for no video
            image_every: also save a pixel-exact PNG every this many simulated
                seconds (e.g. 10 -> t = 10, 20, ...). The final frame is
                always saved as a PNG when image_pattern is set.
            image_pattern: file name for the PNGs, where {t} is the time,
                or None for no PNGs
            crf: video quality, see save_video()
            threads: CPU threads for Numba. Defaults to all but 2.
            use_numba: set False to force the NumPy version
            roughness: also build a roughness map (see roughness.py): every
                roughness_every simulated seconds, measure how different each
                pendulum is from its neighbors, and average the log of that
                over the run. Islands come out dark, chaos light, and regions
                that turn chaotic later come out in between.
            roughness_every: seconds between roughness measurements
            roughness_path: where to save the roughness map ({t} = t_end), or
                None to not save it. Either way it's kept in
                self.roughness_image.
            roughness_invert: draw islands light and chaos dark instead
            roughness_black: contrast. Neighbors that end up (on average) this
                many times their starting spacing apart are drawn pure black.
                Inside islands neighbors still drift apart slowly, because
                nearby pendulums swing at slightly different rates; 3 makes
                that drift black. Raise it for more black, lower it to see
                structure inside islands.

        Returns:
            The final frame as an 8-bit RGB image (one pixel per pendulum),
            so it can be shown or saved without re-running.
        """
        n_frames = int(round(t_end / frame_dt)) + 1
        steps_per_frame = max(1, int(round(frame_dt / dt)))
        step = frame_dt / steps_per_frame
        fast = use_numba and grid_kernel.NUMBA_AVAILABLE
        rows, cols = self.shape

        print(
            f"Rendering {self.count:,} pendulums for {t_end:g} s "
            f"({n_frames:,} frames, {(n_frames - 1) * steps_per_frame:,} steps each)\n"
            f"  state uses about {self.count * 32 / 1e9:.2f} GB of memory"
        )
        if fast:
            print(f"  using Numba on {grid_kernel.set_threads(threads)} threads")
        elif use_numba:
            print("  Numba not installed: using the slower NumPy version")

        # Current state of every pendulum, flattened (row-major, like the image)
        th1_grid, th2_grid = np.meshgrid(self.theta1_values, self.theta2_values)
        th1 = np.ascontiguousarray(th1_grid.ravel(), dtype=np.float64)
        th2 = np.ascontiguousarray(th2_grid.ravel(), dtype=np.float64)
        w1 = np.zeros_like(th1)
        w2 = np.zeros_like(th1)

        if fast:
            print("  compiling (first run only, a few seconds)...", end="", flush=True)
            t0 = time.perf_counter()
            self._advance_numba(
                th1[:2].copy(), th2[:2].copy(), w1[:2].copy(), w2[:2].copy(), 1, step
            )
            print(f" done in {time.perf_counter() - t0:.1f} s")

        writer = None
        if video_path is not None:
            writer = self._open_video(
                video_path, (cols, rows), round(1 / frame_dt), crf
            )
            print(f"  writing {video_path} ({cols} x {rows} px)")
        image_stride = (
            max(1, int(round(image_every / frame_dt)))
            if image_every is not None
            else None
        )

        final_image = np.empty((rows, cols, 3), dtype=np.uint8)
        rough_sum = np.zeros((rows, cols)) if roughness else None
        rough_samples = 0
        rough_stride = max(1, int(round(roughness_every / frame_dt)))
        progress = Progress(n_frames, "frames")
        for k in range(n_frames):
            if k > 0:
                if fast:
                    self._advance_numba(th1, th2, w1, w2, steps_per_frame, step)
                else:
                    self._advance_numpy(th1, th2, w1, w2, steps_per_frame, step)

            save_png = image_pattern is not None and (
                k == n_frames - 1
                or (image_stride is not None and k > 0 and k % image_stride == 0)
            )
            if writer is not None or save_png or k == n_frames - 1:
                image = self._bytes_to_image(
                    angle_to_byte(th1).reshape(rows, cols),
                    angle_to_byte(th2).reshape(rows, cols),
                )
                final_image = image
                if writer is not None:
                    writer.send(image)
                if save_png and image_pattern is not None:
                    t_now = round(k * frame_dt, 6)
                    self.save_png(image_pattern.format(t=t_now), image, t_now)
            if (
                rough_sum is not None
                and k > 0
                and (k % rough_stride == 0 or k == n_frames - 1)
            ):
                rough_sum += roughness_map.log_roughness(
                    th1.reshape(rows, cols), th2.reshape(rows, cols)
                )
                rough_samples += 1
            progress.update(k + 1)

        if writer is not None:
            writer.close()

        if rough_sum is not None:
            spacing = min(
                abs(self.theta1_values[1] - self.theta1_values[0]),
                abs(self.theta2_values[1] - self.theta2_values[0]),
            )
            self.roughness_log10 = rough_sum / max(1, rough_samples)
            self.roughness_image = roughness_map.roughness_image(
                self.roughness_log10,
                np.log10(spacing * roughness_black),
                roughness_invert,
            )
            if roughness_path is not None:
                path = roughness_path.format(t=round(t_end, 6))
                self.save_png(path, self.roughness_image, t_end, kind="roughness")
                print(f"  saved roughness map {path}")
        return final_image

    def _advance_numba(self, th1, th2, w1, w2, n_steps, step):
        p = self.pendulum
        grid_kernel.advance(
            th1, th2, w1, w2, n_steps, step, p.m1, p.m2, p.l1, p.l2, p.g
        )

    def _advance_numpy(self, th1, th2, w1, w2, n_steps, step, batch_size=100_000):
        """NumPy fallback for advance(), in batches to limit memory."""
        f = self.pendulum.derivatives
        for start in range(0, th1.size, batch_size):
            sl = slice(start, start + batch_size)
            state = np.stack([th1[sl], th2[sl], w1[sl], w2[sl]])
            for _ in range(n_steps):
                state = rk4_step(f, state, step)
            th1[sl], th2[sl], w1[sl], w2[sl] = state

    def _extent(self) -> list[float]:
        """Image bounds so each pendulum's square is centered on its angles."""
        h1 = (self.theta1_values[1] - self.theta1_values[0]) / 2
        h2 = (self.theta2_values[1] - self.theta2_values[0]) / 2
        return [
            self.theta1_values[0] - h1,
            self.theta1_values[-1] + h1,
            self.theta2_values[0] - h2,
            self.theta2_values[-1] + h2,
        ]

    def _setup_figure(self, frame: int):
        fig, (ax_key, ax_now) = plt.subplots(1, 2, figsize=(12, 5.6))
        imshow_args = dict(
            origin="lower", extent=self._extent(), interpolation="nearest"
        )
        ax_key.imshow(self.colors(0), **imshow_args)
        ax_key.set_title("Color key (t = 0)")
        image = ax_now.imshow(self.colors(frame), **imshow_args)
        for ax in (ax_key, ax_now):
            ax.set_xlabel("initial θ₁ (rad)")
            ax.set_ylabel("initial θ₂ (rad)")
        fig.suptitle(
            f"{self.count:,} pendulums, one every {self.resolution:g} rad "
            "(green = current θ₁, blue = current θ₂)",
            fontsize=10,
        )
        fig.tight_layout()
        return fig, ax_now, image

    def plot(self, t: float, dt: float = 0.01):
        """
        Static figure: the color key, and the grid's colors at time t.

        Uses stored frames if simulate() already covered time t; otherwise
        simulates just the start and end (t = 0 and t).
        """
        if self.times is None or not np.any(np.isclose(self.times, t)):
            self.simulate(t_end=t, frame_dt=t, dt=dt)
        assert self.times is not None
        frame = int(np.argmin(np.abs(self.times - t)))
        fig, ax_now, _ = self._setup_figure(frame)
        ax_now.set_title(f"t = {self.times[frame]:.2f} s")
        return fig

    def animate(
        self,
        t_end: float = 10.0,
        frame_dt: float = 0.05,
        dt: float = 0.01,
        save_path: str | None = None,
    ):
        """
        Animate the grid's colors from t = 0 to t_end.

        Simulates first if simulate() hasn't been run with these settings.

        Args:
            t_end: how long to run (s)
            frame_dt: simulated time between frames (s)
            dt: integration step (s)
            save_path: if given, save to that file. ".gif" uses one fixed
                256-color palette; ".mp4" is full color but needs ffmpeg
                installed on your computer.

        Returns:
            The FuncAnimation. Keep a reference to it, or it will stop.
        """
        self._ensure_simulated(t_end, frame_dt, dt)
        assert self.times is not None
        times = self.times
        fig, ax_now, image = self._setup_figure(0)
        ax_now.set_title("t = 0.00 s")

        def update(frame):
            image.set_data(self.colors(frame))
            ax_now.set_title(f"t = {times[frame]:.2f} s")
            return (image,)

        anim = FuncAnimation(
            fig, update, frames=len(times), interval=frame_dt * 1000, blit=False
        )
        if save_path is not None:
            print(f"Saving {save_path}")
            progress = Progress(len(times), "frames")
            fps = round(1 / frame_dt)
            report = lambda i, n: progress.update(i + 1)  # noqa: E731
            if save_path.lower().endswith(".gif"):
                anim.save(
                    save_path,
                    writer=FixedPaletteGifWriter(fps=fps),
                    progress_callback=report,
                )
            else:
                # e.g. ".mp4": full color, no palette limit (needs ffmpeg)
                anim.save(save_path, fps=fps, progress_callback=report)
        return anim


if __name__ == "__main__":
    # 2000 x 2000 pendulums = one per pixel in a 2000 x 2000 px image/video
    grid = PendulumGrid(
        points=2000,
        theta1_range=(-np.pi, np.pi),  # x axis
        theta2_range=(-np.pi, np.pi),  # y axis
    )
    # Streams to disk as it goes, so memory stays small however long it runs.
    # Saves grid.mp4, a PNG every 10 simulated seconds, the final frame, and a
    # roughness map (islands dark, chaos light) averaged over the whole run.
    grid.render(t_end=60, video_path="grid.mp4", image_every=10, roughness=True)
