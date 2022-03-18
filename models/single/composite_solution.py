import numpy as np
from numpy import pi, exp


class OuterSolution:
    def __init__(self, r0, delta, mu, lam, alpha, alpha_cc):
        """
        Computes the composite of the bulk solution and the 'outer solution' near r=r0.
        The paramaters and solutions are stored as attributes of the class.
        """

        # physical (dimensionless) parameters
        self.r0 = r0
        self.delta = delta
        self.mu = mu
        self.lam = lam
        self.alpha = alpha
        self.alpha_cc = alpha_cc

        # constants
        # S_1 = alpha_hat
        # S_2 = 1 / M_hat
        # S_3 = 1 / mu_hat
        self.S_1 = (
            self.alpha * (3 * self.lam + 2 * self.mu) - 2 * self.lam * self.alpha_cc
        ) / (self.lam + 2 * self.mu)
        self.S_2 = 1 / (self.lam + 2 * self.mu)
        self.S_3 = 1 / self.mu

        self.omega = np.sqrt(self.S_2 / self.S_3)

        self.A = self.S_1 * exp(2 * pi * self.omega)
        self.B = 0
        self.C = self.A / (1 - exp(2 * pi * self.omega))
        self.D = 0

    def f1(self, theta):
        return (-self.S_1 + self.A * exp(-self.omega * (theta + 2 * pi))) / self.S_2

    def f2(self, theta):
        return self.B + self.C * exp(-self.omega * theta)

    def g1(self, theta):
        return (self.omega * self.A / self.S_2) * exp(-self.omega * (theta + 2 * pi))

    def dg1dt(self, theta):
        return (
            -self.omega
            * (self.omega * self.A / self.S_2)
            * exp(-self.omega * (theta + 2 * pi))
        )

    def g2(self, theta):
        return self.D + self.C / self.omega * exp(-self.omega * theta)

    def dg2dt(self, theta):
        return -self.C * exp(-self.omega * theta)

    def u(self, r, theta):
        """Radial displacement"""
        alpha = self.alpha
        alpha_cc = self.alpha_cc
        delta = self.delta
        r0 = self.r0
        lam = self.lam
        mu = self.mu
        R = (r - r0) / delta
        u = alpha_cc * r + delta * (
            (alpha * (3 * lam + 2 * mu) - 2 * lam * alpha_cc)
            / (lam + 2 * mu)
            * (R - theta / 2 / pi)
            + self.f1(theta) / (lam + 2 * mu) * (R - theta / 2 / pi)
            + self.f2(theta)
        )
        # returns bulk + outer - common part
        return u

    def v(self, r, theta):
        """Azimuthal displacement"""
        delta = self.delta
        r0 = self.r0
        mu = self.mu
        R = (r - r0) / delta
        # returns bulk + outer - common part
        return delta * (self.g1(theta) / mu * (R - theta / 2 / pi) + self.g2(theta))

    def e_rr(self, theta):
        """Radial strain"""
        # alpha = self.alpha
        alpha_cc = self.alpha_cc
        lam = self.lam
        mu = self.mu
        # returns bulk + outer - common part
        # note f1 -> -S1/S2 as theta -> inf
        return (
            alpha_cc
            + self.f1(theta) / (lam + 2 * mu)
            + self.S_1 / self.S_2 / (lam + 2 * mu)
        )

    def e_tt(self, r, theta):
        """Azimuthal strain"""
        delta = self.delta
        r0 = self.r0
        mu = self.mu
        R = (r - r0) / delta
        dvdt = delta * (
            self.dg1dt(theta) / mu * (R - theta / 2 / pi)
            - self.g1(theta) / mu / 2 / pi
            + self.dg2dt(theta)
        )
        # returns bulk + outer - common part
        return (1 / r) * (dvdt + self.u(r, theta))

    def e_rt(self, theta):
        """Shear strain"""
        mu = self.mu
        # returns bulk + outer - common part
        return self.g1(theta) / mu / 2

    def s_rr(self, theta):
        """Radial stress"""
        alpha = self.alpha
        alpha_cc = self.alpha_cc
        lam = self.lam
        mu = self.mu
        # returns bulk + outer - common part
        # note f1 -> -S1/S2 as theta -> inf
        return (
            (alpha_cc - alpha) * (3 * lam + 2 * mu)
            + self.f1(theta)
            + self.S_1 / self.S_2
        )

    def s_tt(self, theta):
        """Azimuthal stress"""
        alpha = self.alpha
        alpha_cc = self.alpha_cc
        lam = self.lam
        mu = self.mu
        # returns bulk + outer - common part
        return lam / (lam + 2 * mu) * self.s_rr(theta) + 2 * mu * (alpha_cc - alpha) * (
            3 * lam + 2 * mu
        ) / (lam + 2 * mu)

    def s_rt(self, theta):
        """Shear stress"""
        # returns bulk + outer - common part
        return self.g1(theta)

    def T(self, theta):
        """
        Tension
        Note: the tension is different for the first wind
        """
        r0 = self.r0
        theta1 = theta[theta < 2 * pi]
        theta2 = theta[theta >= 2 * pi]
        T1 = r0 * self.f1(theta1)
        T2 = r0 * (self.f1(theta2) - self.f1(theta2 - 2 * pi))
        return np.concatenate((T1, T2))
