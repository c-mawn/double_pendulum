"""
Plots and animation for a single double pendulum run, all in one window.

    ┌──────────────────────┬──────────────────────┐
    │  pendulum animation  │   θ₁ vs θ₂ (phase /  │
    │  with fading trail   │   Lissajous curve)   │
    ├──────────────────────┴──────────────────────┤
    │  θ₁ and θ₂ over time                        │
    ├─────────────────────────────────────────────┤
    │  energy drift (correctness check)           │
    └─────────────────────────────────────────────┘

Every panel follows the animation: the θ₁ vs θ₂ panel draws the recent part of
the curve brightly with a dot at the current point, and the time plots show a
moving cursor.

Run this file directly for a demo:
    python viz.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba

from pendulum import Pendulum

# One color per mass / per angle, used consistently in every panel
COLOR_1 = "#2a78d6"  # blue: mass 1, theta1
COLOR_2 = "#eb6834"  # orange: mass 2, theta2
ROD_COLOR = "#3d3d3a"
GRID_COLOR = "#e4e3dd"
TEXT_MUTED = "#6b6a63"
PATH_COLOR = "#b4b2a9"  # full theta1-theta2 path, behind the bright recent part


def _style_axes(ax):
    """Recessive grid and axes so the data stands out."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED)


def _fading_trail(color: str, width: float = 1.5) -> LineCollection:
    """A line whose segments fade from transparent (old) to solid (new)."""
    r, g, b, _ = to_rgba(color)
    cmap = LinearSegmentedColormap.from_list("trail", [(r, g, b, 0.0), (r, g, b, 1.0)])
    return LineCollection([], linewidths=width, cmap=cmap, norm=Normalize(0.0, 1.0))


def _to_shared_palette(frames: list[np.ndarray], background_rgb: list[int]) -> list:
    """
    Convert RGB frames to palette images that all share ONE exact palette.

    The background color is always entry 0, exactly as given, so a GIF placed
    on a page of the same color shows no box around it. If the whole clip
    uses 256 colors or fewer, every color is kept exactly. Otherwise the most
    representative 255 are chosen and each pixel gets its nearest one.
    (Pillow's own palette matching isn't exact, so this does it with NumPy.)
    """
    from PIL import Image

    def codes(frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.int32)
        return (f[..., 0] << 16) | (f[..., 1] << 8) | f[..., 2]

    bg_code = (background_rgb[0] << 16) | (background_rgb[1] << 8) | background_rgb[2]
    unique = np.unique(
        np.concatenate([np.unique(codes(f)) for f in frames] + [np.array([bg_code])])
    )
    unique_rgb = np.stack(
        [(unique >> 16) & 255, (unique >> 8) & 255, unique & 255], axis=1
    )

    others = unique[unique != bg_code]
    if len(others) <= 255:
        palette_codes = np.concatenate([[bg_code], others])
    else:
        # Too many colors: let median cut pick 255 from a sample of the frames
        sample = frames[:: max(1, len(frames) // 12)]
        montage = Image.fromarray(np.concatenate(sample, axis=0))
        flat = (
            montage.quantize(colors=255, method=Image.Quantize.MEDIANCUT).getpalette()
            or []
        )
        picked = np.array(flat[: 255 * 3]).reshape(-1, 3)
        palette_codes = np.concatenate(
            [[bg_code], (picked[:, 0] << 16) | (picked[:, 1] << 8) | picked[:, 2]]
        )
    palette_rgb = np.stack(
        [(palette_codes >> 16) & 255, (palette_codes >> 8) & 255, palette_codes & 255],
        axis=1,
    )

    # Exact nearest palette entry for every color that appears in the clip
    dist = ((unique_rgb[:, None, :] - palette_rgb[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(dist, axis=1).astype(
        np.uint8
    )  # ties -> lowest index (background)

    flat_palette = palette_rgb.astype(np.uint8).ravel().tolist()
    flat_palette += [0] * (768 - len(flat_palette))
    images = []
    for frame in frames:
        index = nearest[np.searchsorted(unique, codes(frame))]
        image = Image.fromarray(index, mode="P")
        image.putpalette(flat_palette)
        images.append(image)
    return images


def _update_trail(trail: LineCollection, x: np.ndarray, y: np.ndarray):
    """Set a fading trail's points (oldest first)."""
    pts = np.column_stack([x, y])
    segments = list(np.stack([pts[:-1], pts[1:]], axis=1))
    trail.set_segments(segments)
    trail.set_array(np.linspace(0.0, 1.0, len(segments)))


class PendulumViz:
    """
    Visualizer for a double pendulum
    """

    def __init__(self, pendulum: Pendulum, t_end: float = 20.0, dt: float = 0.02):
        """
        Simulates the pendulum once and keeps the result for plotting.

        Args:
            pendulum: the pendulum to visualize (its initial conditions are used)
            t_end: how long to simulate (s)
            dt: time between frames / samples (s). 0.02 s gives 50 fps.
        """
        self.pendulum = pendulum
        self.dt = dt
        self.t, self.states = pendulum.simulate(t_end, dt)
        self.x1, self.y1, self.x2, self.y2 = pendulum.positions(self.states)

    def animate(
        self,
        trail_seconds: float = 2.0,
        save_path: str | None = None,
        title: str | None = None,
    ):
        """
        Open one window with the animation and all plots, playing in real time.

        Args:
            trail_seconds: how much recent motion the fading trails show, both
                behind mass 2 and on the θ₁ vs θ₂ curve
            save_path: if given (e.g. "pendulum.gif"), also save the whole
                window as an animation to that file
            title: window title; defaults to the starting angles

        Returns:
            The FuncAnimation. Keep a reference to it, or matplotlib will
            garbage-collect it and the animation will stop.
        """
        th1, th2 = self.states[0], self.states[1]
        trail_len = max(1, int(trail_seconds / self.dt))

        fig = plt.figure(figsize=(13, 10))
        gs = fig.add_gridspec(
            3, 2, height_ratios=[3.2, 1.3, 0.8], hspace=0.45, wspace=0.2
        )
        ax_anim = fig.add_subplot(gs[0, 0])
        ax_phase = fig.add_subplot(gs[0, 1])
        ax_angles = fig.add_subplot(gs[1, :])
        ax_energy = fig.add_subplot(gs[2, :], sharex=ax_angles)

        if title is None:
            title = (
                f"Double pendulum: θ₁ = {th1[0]:.4f} rad, θ₂ = {th2[0]:.4f} rad "
                f"(starting at rest)"
            )
        fig.suptitle(title, fontsize=12)

        # --- Pendulum animation -----------------------------------------
        self._set_square_limits(ax_anim)
        ax_anim.set_xticks([])
        ax_anim.set_yticks([])
        for spine in ax_anim.spines.values():
            spine.set_visible(False)
        swing_trail = _fading_trail(COLOR_2)
        ax_anim.add_collection(swing_trail)
        (rods,) = ax_anim.plot([], [], "-", color=ROD_COLOR, lw=2)
        (bob1,) = ax_anim.plot([], [], "o", color=COLOR_1, ms=12)
        (bob2,) = ax_anim.plot([], [], "o", color=COLOR_2, ms=12)
        ax_anim.plot(0, 0, "o", color=ROD_COLOR, ms=5)  # pivot
        time_text = ax_anim.text(
            0.03, 0.95, "", transform=ax_anim.transAxes, color=TEXT_MUTED
        )

        # --- θ₁ vs θ₂ (Lissajous) -----------------------------------------
        ax_phase.plot(th1, th2, color=PATH_COLOR, lw=0.7)  # whole run, faint
        phase_trail = _fading_trail(ROD_COLOR, width=2.0)
        ax_phase.add_collection(phase_trail)
        (phase_dot,) = ax_phase.plot([], [], "o", color=ROD_COLOR, ms=7)
        ax_phase.set_xlabel("θ₁ (rad)")
        ax_phase.set_ylabel("θ₂ (rad)")
        ax_phase.set_title("θ₁ vs θ₂", fontsize=11)
        ax_phase.set_aspect("equal", adjustable="datalim")
        _style_axes(ax_phase)

        # --- Angles over time ---------------------------------------------
        ax_angles.plot(self.t, th1, color=COLOR_1, lw=1.5, label="θ₁ (upper arm)")
        ax_angles.plot(self.t, th2, color=COLOR_2, lw=1.5, label="θ₂ (lower arm)")
        ax_angles.set_ylabel("angle (rad)")
        ax_angles.set_title("Angles over time", fontsize=11)
        ax_angles.legend(
            frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2
        )
        ax_angles.tick_params(labelbottom=False)
        _style_axes(ax_angles)

        # --- Energy drift: should stay near zero; any drift is numerical error
        energy = self.pendulum.energy(self.states)
        ax_energy.plot(self.t, energy - energy[0], color=ROD_COLOR, lw=1.5)
        ax_energy.set_ylabel("ΔE (J)")
        ax_energy.set_xlabel("time (s)")
        ax_energy.set_title("Energy drift (should stay near zero)", fontsize=10)
        ax_energy.set_xlim(self.t[0], self.t[-1])
        _style_axes(ax_energy)

        # Moving "now" cursors on the time plots
        cursors = [
            ax.axvline(0.0, color=TEXT_MUTED, lw=1, alpha=0.8)
            for ax in (ax_angles, ax_energy)
        ]

        def update(i):
            rods.set_data(
                np.array([0.0, self.x1[i], self.x2[i]]),
                np.array([0.0, self.y1[i], self.y2[i]]),
            )
            bob1.set_data(self.x1[i : i + 1], self.y1[i : i + 1])
            bob2.set_data(self.x2[i : i + 1], self.y2[i : i + 1])

            start = max(0, i - trail_len)
            _update_trail(swing_trail, self.x2[start : i + 1], self.y2[start : i + 1])
            _update_trail(phase_trail, th1[start : i + 1], th2[start : i + 1])
            phase_dot.set_data(th1[i : i + 1], th2[i : i + 1])

            t_now = self.t[i]
            for cursor in cursors:
                cursor.set_xdata([t_now, t_now])
            time_text.set_text(f"t = {t_now:.2f} s")
            return (
                swing_trail,
                rods,
                bob1,
                bob2,
                phase_trail,
                phase_dot,
                time_text,
                *cursors,
            )

        anim = FuncAnimation(
            fig, update, frames=len(self.t), interval=self.dt * 1000, blit=True
        )
        if save_path is not None:
            anim.save(save_path, writer=PillowWriter(fps=round(1 / self.dt)))
        return anim

    # ------------------------------------------------------------------
    # Clean clip for a web page
    # ------------------------------------------------------------------

    def save_clip(
        self,
        path: str = "pendulum.gif",
        seconds: float = 10.0,
        fps: int = 25,
        size_px: int = 600,
        trail_seconds: float = 1.5,
        background: str = "white",
        crop_to_motion: bool = True,
    ):
        """
        Save just the swinging pendulum (no axes, labels or plots) as a looping GIF.

        Made for embedding on a web page: plays at real speed, loops forever,
        and uses one fixed color palette for the whole clip, so nothing
        flickers. Re-simulates at exactly the frame rate, so it doesn't depend
        on this PendulumViz's own dt or t_end.

        Args:
            path: output file (.gif)
            seconds: length of the clip (s of simulated, and real, time)
            fps: frames per second. GIFs store frame times in hundredths of a
                second, so 25 (4/100 s) and 50 (2/100 s) play at exactly
                real speed; 50 is smoother but makes a file about twice as big.
            size_px: width of the GIF in pixels (also the height when
                crop_to_motion is False)
            trail_seconds: length of the fading trail behind the lower mass
                (0 for none)
            background: background color (any matplotlib color). It's kept
                exact in the GIF, so it blends seamlessly into a page of the
                same color.
            crop_to_motion: frame only the area the pendulum actually swings
                through during the clip (plus a margin), instead of the full
                square it could reach. A pendulum that never goes over the
                top then fills the GIF instead of leaving the top half empty.
        """
        from PIL import Image  # only needed here
        from matplotlib.colors import to_rgb

        t, states = self.pendulum.simulate(seconds, dt=1.0 / fps)
        x1, y1, x2, y2 = self.pendulum.positions(states)
        trail_len = int(round(trail_seconds * fps))

        # Area to show: everything the pivot and masses touch, plus a margin
        margin = 0.12 * (self.pendulum.l1 + self.pendulum.l2)
        if crop_to_motion:
            xs = np.concatenate([[0.0], x1, x2])
            ys = np.concatenate([[0.0], y1, y2])
            x_lo, x_hi = xs.min() - margin, xs.max() + margin
            y_lo, y_hi = ys.min() - margin, ys.max() + margin
        else:
            reach = self.pendulum.l1 + self.pendulum.l2 + margin
            x_lo, x_hi, y_lo, y_hi = -reach, reach, -reach, reach
        width_px = size_px
        height_px = int(round(size_px * (y_hi - y_lo) / (x_hi - x_lo) / 2)) * 2  # even

        dpi = 100
        fig = plt.figure(
            figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=background
        )
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor=background)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("auto")  # limits already match the pixel aspect ratio
        ax.axis("off")

        # Keep line widths and dot sizes proportional to the pendulum's size on screen
        scale = (
            size_px
            / 600
            * 4.4
            * (self.pendulum.l1 + self.pendulum.l2)
            / (x_hi - x_lo)
            / 2
        )
        trail = _fading_trail(COLOR_2, width=2.0 * scale)
        ax.add_collection(trail)
        (rods,) = ax.plot(
            [], [], "-", color=ROD_COLOR, lw=3 * scale, solid_capstyle="round"
        )
        (bob1,) = ax.plot([], [], "o", color=COLOR_1, ms=16 * scale)
        (bob2,) = ax.plot([], [], "o", color=COLOR_2, ms=16 * scale)
        ax.plot(0, 0, "o", color=ROD_COLOR, ms=7 * scale)  # pivot

        frames: list[np.ndarray] = []
        print(f"Rendering {len(t)} frames for {path}")
        for i in range(len(t)):
            rods.set_data(np.array([0.0, x1[i], x2[i]]), np.array([0.0, y1[i], y2[i]]))
            bob1.set_data(x1[i : i + 1], y1[i : i + 1])
            bob2.set_data(x2[i : i + 1], y2[i : i + 1])
            if trail_len > 0 and i > 0:
                start = max(0, i - trail_len)
                _update_trail(trail, x2[start : i + 1], y2[start : i + 1])
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
            frames.append(rgba[..., :3].copy())
        plt.close(fig)

        frames_p = _to_shared_palette(
            frames, [round(c * 255) for c in to_rgb(background)]
        )
        frames_p[0].save(
            path,
            save_all=True,
            append_images=frames_p[1:],
            duration=round(1000 / fps),
            loop=0,  # 0 = repeat forever
        )
        import os

        print(
            f"Saved {path} ({width_px} x {height_px} px, {len(t)} frames, "
            f"{os.path.getsize(path) / 1e6:.1f} MB)"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_square_limits(self, ax):
        """Fixed, equal-aspect limits that fit the fully extended pendulum."""
        reach = (self.pendulum.l1 + self.pendulum.l2) * 1.1
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")


if __name__ == "__main__":
    # A moderate start inside a stable island, where θ₁ vs θ₂ traces a
    # Lissajous-like curve. Try theta1=2.0, theta2=2.5 for chaos.
    pendulum = Pendulum(theta1=0.2, theta2=0.9)
    viz = PendulumViz(pendulum, t_end=30)
    anim = viz.animate()
    plt.show()
