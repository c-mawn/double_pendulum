"""
simple double pendulum physics

Conventions:
    - Angles are measured from straight down, counterclockwise positive
    - The pivot is at the origin, with y pointing up
    - A state is the array [theta1, theta2, omega1, omega2], where
      omega = d(theta)/dt

every method that takes a state also accepts arrays of states with shape
(4, ...), so many pendulums can be evaluated at once with NumPy
"""

import numpy as np
from scipy.integrate import solve_ivp


class Pendulum:

    def __init__(
        self,
        l1: float = 1.0,
        l2: float = 1.0,
        m1: float = 1.0,
        m2: float = 1.0,
        theta1: float = 0.0,
        omega1: float = 0.0,
        theta2: float = 0.0,
        omega2: float = 0.0,
        g: float = 9.80665,
    ):
        """
        Creates a double pendulum to simulate over time

        Args:
            l1, l2: rod lengths (m)
            m1, m2: point masses (kg)
            theta1, theta2: initial angles from vertical (rad)
            omega1, omega2: initial angular velocities (rad/s)
            g: gravitational acceleration (m/s^2)
        """
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.theta1 = theta1
        self.theta2 = theta2
        self.omega1 = omega1
        self.omega2 = omega2
        self.g = g

    @property
    def initial_state(self) -> np.ndarray:
        """The initial conditions as a state array [theta1, theta2, omega1, omega2]."""
        return np.array([self.theta1, self.theta2, self.omega1, self.omega2])

    # Equations of motion

    def accelerations(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Angular accelerations (theta1_ddot, theta2_ddot) for a given state.

        Args:
            state: array [theta1, theta2, theta1_dot, theta2_dot]
        """
        th1, th2, w1, w2 = state
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g

        delta = th1 - th2
        sin_d = np.sin(delta)
        cos_d = np.cos(delta)
        denom = m1 + m2 * sin_d**2

        alpha1 = (
            -m2 * sin_d * (l2 * w2**2 + l1 * w1**2 * cos_d)
            - (m1 + m2) * g * np.sin(th1)
            + m2 * g * np.sin(th2) * cos_d
        ) / (l1 * denom)

        alpha2 = (
            sin_d * ((m1 + m2) * l1 * w1**2 + m2 * l2 * w2**2 * cos_d)
            + (m1 + m2) * g * (np.sin(th1) * cos_d - np.sin(th2))
        ) / (l2 * denom)

        return alpha1, alpha2

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Time derivative of the state

        d/dt [theta1, theta2, omega1, omega2] = [omega1, omega2, alpha1, alpha2]
        """
        _, _, w1, w2 = state
        alpha1, alpha2 = self.accelerations(state)
        return np.array([w1, w2, alpha1, alpha2])

    # Simulation

    def simulate(
        self,
        t_end: float,
        dt: float = 0.01,
        state0: np.ndarray | None = None,
        rtol: float = 1e-10,
        atol: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate the equations of motion from t = 0 to t = t_end.

        Args:
            t_end: how long to simulate (s)
            dt: spacing of the output samples (s). The solver picks its own
                internal step size; this only controls what is returned.
            state0: starting state; defaults to this pendulum's initial_state
            rtol, atol: solver error tolerances

        Returns:
            t: array of times, shape (N,)
            states: array of states, shape (4, N), rows are
                    theta1, theta2, omega1, omega2
        """
        if state0 is None:
            state0 = self.initial_state

        t_eval = np.arange(0.0, t_end + dt / 2, dt)
        sol = solve_ivp(
            self.derivatives,
            (0.0, t_eval[-1]),
            state0,
            method="DOP853",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        return sol.t, sol.y

    # Derived quantities

    def positions(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        positions of both masses, (x1, y1, x2, y2).

        y is measured upward, so hanging masses have y < 0.
        """
        th1, th2 = state[0], state[1]
        x1 = self.l1 * np.sin(th1)
        y1 = -self.l1 * np.cos(th1)
        x2 = x1 + self.l2 * np.sin(th2)
        y2 = y1 - self.l2 * np.cos(th2)
        return x1, y1, x2, y2

    def kinetic_energy(self, state: np.ndarray) -> np.ndarray:
        """T = 1/2 (m1+m2) l1^2 w1^2 + 1/2 m2 l2^2 w2^2 + m2 l1 l2 w1 w2 cos(delta theta)"""
        th1, th2, w1, w2 = state
        m1, m2, l1, l2 = self.m1, self.m2, self.l1, self.l2
        return (
            0.5 * (m1 + m2) * l1**2 * w1**2
            + 0.5 * m2 * l2**2 * w2**2
            + m2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
        )

    def potential_energy(self, state: np.ndarray) -> np.ndarray:
        """V = -(m1+m2) g l1 cos(theta1) - m2 g l2 cos(theta2)"""
        th1, th2 = state[0], state[1]
        return -(self.m1 + self.m2) * self.g * self.l1 * np.cos(
            th1
        ) - self.m2 * self.g * self.l2 * np.cos(th2)

    def energy(self, state: np.ndarray) -> np.ndarray:
        """Total mechanical energy T + V: Should be constant"""
        return self.kinetic_energy(state) + self.potential_energy(state)
