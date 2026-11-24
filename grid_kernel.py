"""
Fast, compiled simulation of many pendulums at once, using Numba.

The physics here is the same as Pendulum.accelerations in pendulum.py, rewritten
for one pendulum at a time with plain floats. Numba compiles that into machine
code and runs different pendulums on different CPU cores at the same time.

If Numba isn't installed, NUMBA_AVAILABLE is False and grid.py falls back to
its (slower) NumPy version. The functions below still work without Numba, as
ordinary Python, but far too slowly for real use. That's only for testing.
"""

import math

try:
    from numba import njit, prange, set_num_threads
    from numba.core import config as numba_config

    NUMBA_AVAILABLE = True
except ImportError:  # run as plain Python instead
    numba_config = None
    set_num_threads = None
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        def decorate(func):
            return func

        return decorate


TWO_PI = 2.0 * math.pi


def set_threads(threads: int | None) -> int:
    """
    Choose how many CPU threads Numba uses, and return the number chosen.

    None means "all logical CPUs except 2", leaving room for the rest of the
    computer (your editor, browser, etc.) to stay responsive.
    """
    if numba_config is None or set_num_threads is None:
        return 1
    # Numba creates its config values at runtime, so editors can't see them;
    # getattr reads the same documented setting without a false warning.
    available = getattr(numba_config, "NUMBA_NUM_THREADS")
    if threads is None:
        threads = max(1, available - 2)
    threads = max(1, min(threads, available))
    set_num_threads(threads)
    return threads


@njit(cache=True, fastmath=True)
def _accelerations(th1, th2, w1, w2, m1, m2, l1, l2, g):
    """Angular accelerations of one pendulum (same formulas as pendulum.py)."""
    delta = th1 - th2
    sin_d = math.sin(delta)
    cos_d = math.cos(delta)
    sin1 = math.sin(th1)
    sin2 = math.sin(th2)
    denom = m1 + m2 * sin_d * sin_d

    alpha1 = (
        -m2 * sin_d * (l2 * w2 * w2 + l1 * w1 * w1 * cos_d)
        - (m1 + m2) * g * sin1
        + m2 * g * sin2 * cos_d
    ) / (l1 * denom)

    alpha2 = (
        sin_d * ((m1 + m2) * l1 * w1 * w1 + m2 * l2 * w2 * w2 * cos_d)
        + (m1 + m2) * g * (sin1 * cos_d - sin2)
    ) / (l2 * denom)

    return alpha1, alpha2


@njit(cache=True, fastmath=True)
def _angle_to_byte(theta):
    """Wrap an angle to one turn and store it as 0..255 (-pi -> 0, +pi -> 255)."""
    fraction = (theta + math.pi) / TWO_PI
    fraction -= math.floor(fraction)
    return int(fraction * 255.0 + 0.5)


@njit(cache=True, fastmath=True)
def _rk4_step(a, b, w1, w2, h, m1, m2, l1, l2, g):
    """One classic RK4 step of size h for one pendulum's state (a, b, w1, w2)."""
    k1_al1, k1_al2 = _accelerations(a, b, w1, w2, m1, m2, l1, l2, g)

    a2 = a + 0.5 * h * w1
    b2 = b + 0.5 * h * w2
    w1_2 = w1 + 0.5 * h * k1_al1
    w2_2 = w2 + 0.5 * h * k1_al2
    k2_al1, k2_al2 = _accelerations(a2, b2, w1_2, w2_2, m1, m2, l1, l2, g)

    a3 = a + 0.5 * h * w1_2
    b3 = b + 0.5 * h * w2_2
    w1_3 = w1 + 0.5 * h * k2_al1
    w2_3 = w2 + 0.5 * h * k2_al2
    k3_al1, k3_al2 = _accelerations(a3, b3, w1_3, w2_3, m1, m2, l1, l2, g)

    a4 = a + h * w1_3
    b4 = b + h * w2_3
    w1_4 = w1 + h * k3_al1
    w2_4 = w2 + h * k3_al2
    k4_al1, k4_al2 = _accelerations(a4, b4, w1_4, w2_4, m1, m2, l1, l2, g)

    return (
        a + (h / 6.0) * (w1 + 2.0 * w1_2 + 2.0 * w1_3 + w1_4),
        b + (h / 6.0) * (w2 + 2.0 * w2_2 + 2.0 * w2_3 + w2_4),
        w1 + (h / 6.0) * (k1_al1 + 2.0 * k2_al1 + 2.0 * k3_al1 + k4_al1),
        w2 + (h / 6.0) * (k1_al2 + 2.0 * k2_al2 + 2.0 * k3_al2 + k4_al2),
    )


@njit(parallel=True, cache=True, fastmath=True)
def simulate_batch(th1_start, th2_start, out, steps_per_frame, h, m1, m2, l1, l2, g):
    """
    Run a batch of pendulums (all starting at rest) and record their angles.

    Used by PendulumGrid.simulate(), which keeps every frame in memory.

    Args:
        th1_start, th2_start: starting angles, 1D float arrays of length n
        out: uint8 array of shape (n_frames, n, 2). Frames 1 onward are filled
            in: out[k, i, 0] and out[k, i, 1] are pendulum i's theta1 and
            theta2 at frame k, as bytes. Frame 0 is left for the caller.
        steps_per_frame: RK4 steps between recorded frames
        h: RK4 step size (s)
        m1, m2, l1, l2, g: pendulum parameters
    """
    n_frames = out.shape[0]
    n = th1_start.shape[0]
    for i in prange(n):  # each pendulum is independent, so these run in parallel
        a = th1_start[i]
        b = th2_start[i]
        w1 = 0.0
        w2 = 0.0
        for k in range(1, n_frames):
            for _ in range(steps_per_frame):
                a, b, w1, w2 = _rk4_step(a, b, w1, w2, h, m1, m2, l1, l2, g)
            out[k, i, 0] = _angle_to_byte(a)
            out[k, i, 1] = _angle_to_byte(b)


@njit(parallel=True, cache=True, fastmath=True)
def advance(th1, th2, w1, w2, n_steps, h, m1, m2, l1, l2, g):
    """
    Move every pendulum forward by n_steps RK4 steps, updating the arrays in place.

    Used by PendulumGrid.render(), which keeps only the current state in
    memory and writes each frame out as soon as it's computed.

    Args:
        th1, th2, w1, w2: current angles and angular velocities, 1D float
            arrays of the same length (one entry per pendulum)
        n_steps: number of RK4 steps to take
        h: RK4 step size (s)
        m1, m2, l1, l2, g: pendulum parameters
    """
    for i in prange(th1.shape[0]):
        a = th1[i]
        b = th2[i]
        x = w1[i]
        y = w2[i]
        for _ in range(n_steps):
            a, b, x, y = _rk4_step(a, b, x, y, h, m1, m2, l1, l2, g)
        th1[i] = a
        th2[i] = b
        w1[i] = x
        w2[i] = y
