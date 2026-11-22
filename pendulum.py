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
    ):
        """
        Creates a double pendulum to simulate over time
        """
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.theta1 = theta1
        self.theta2 = theta2
        self.omega1 = omega1
        self.omega2 = omega2
