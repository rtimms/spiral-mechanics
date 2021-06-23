import numpy as np
from numpy import pi, exp


class OuterSolution:
    def __init__(self, r0, delta, E, nu, alpha):

        # physical (dimensionless) parameters
        self.alpha = alpha
        self.delta = delta
        self.E = E
        self.nu = nu
        self.lam = E * nu / (1 + nu) / (1 - 2 * nu)
        self.mu = E / 2 / (1 + nu)
        self.r0 = r0

        # constants
        self.omega = np.sqrt(self.mu / (self.lam + 2 * self.mu))
        self.A = (
            self.alpha
            * (3 * self.lam + 2 * self.mu)
            / (self.lam + 2 * self.mu)
            * exp(2 * pi * self.omega)
        )
        self.B = 0
        self.C = self.A / (1 - exp(2 * pi * self.omega))
        self.D = 0

    def f1(self, theta):
        alpha = self.alpha
        lam = self.lam
        mu = self.mu
        omega = self.omega
        A = self.A
        return -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + A * exp(
            -omega * (theta + 2 * pi)
        )

    def f2(self, theta):
        return self.B + self.C * exp(-self.omega * theta)

    def g1(self, theta):
        return (
            (self.lam + 2 * self.mu)
            / self.mu
            * self.omega
            * self.A
            * exp(-self.omega * (theta + 2 * pi))
        )

    def g2(self, theta):
        return self.D + self.C / self.omega * exp(-self.omega * theta)

    def u(self, r, theta):
        """
        radial displacement at fixed R-theta/2/pi=0.5
        """
        alpha = self.alpha
        delta = self.delta
        r0 = self.r0
        lam = self.lam
        mu = self.mu
        R = (r - r0) / delta
        u = delta * (
            alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * (R - theta / 2 / pi)
            + self.f1(theta) * (R - theta / 2 / pi)
            + self.f2(theta)
        )
        return u

    def v(self, r, theta):
        """
        azimuthal displacement at fixed R-theta/2/pi=0.5
        """
        delta = self.delta
        r0 = self.r0
        R = (r - r0) / delta
        return delta * (self.g1(theta) * (R - theta / 2 / pi) + self.g2(theta))

    def e_rr(self, theta):
        """ radial strain """
        alpha = self.alpha
        lam = self.lam
        mu = self.mu
        return alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + self.f1(theta)

    def e_tt(self, theta):
        """ azimuthal strain """
        delta = self.delta
        r0 = self.r0
        r = r0 + delta / 2 + delta * theta / 2 / pi
        return (delta / r0) * (np.gradient(self.v(r, theta), theta) + self.u(r, theta))

    def e_rt(self, theta):
        """ shear strain """
        return self.g1(theta) / 2

    def s_rr(self, theta):
        """ radial stress """
        return (self.lam + 2 * self.mu) * self.f1(theta)

    def s_tt(self, theta):
        """ azimuthal stress """
        alpha = self.alpha
        lam = self.lam
        mu = self.mu
        return -2 * mu * alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + lam * self.f1(
            theta
        )

    def s_rt(self, theta):
        """ shear stress """
        return self.mu * self.g1(theta)

    def T(self, theta):
        """
        tension
        note: the tension is different for the first wind
        """
        lam = self.lam
        mu = self.mu
        r0 = self.r0
        theta1 = theta[theta < 2 * pi]
        theta2 = theta[theta >= 2 * pi]
        T1 = r0 * (lam + 2 * mu) * self.f1(theta1)
        T2 = r0 * (lam + 2 * mu) * (self.f1(theta2) - self.f1(theta2 - 2 * pi))
        return np.concatenate((T1, T2))
