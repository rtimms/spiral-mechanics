import numpy as np
from numpy import pi, exp


class OuterSolution:
    def __init__(self, param):
        """
        Computes the 'outer solution' near r=r0. The solutions are stored
        as attributes of the class.
        """

        # constants
        param.S_1 = (
            (
                param.l_n
                * param.alpha_n
                * (3 * param.lam_n + 2 * param.mu_n)
                / (param.lam_n + 2 * param.mu_n)
            )
            + (
                param.l_s
                * param.alpha_s
                * (3 * param.lam_s + 2 * param.mu_s)
                / (param.lam_s + 2 * param.mu_s)
            )
            + (
                param.l_p
                * param.alpha_p
                * (3 * param.lam_p + 2 * param.mu_p)
                / (param.lam_p + 2 * param.mu_p)
            )
        )
        param.S_2 = (
            param.l_n / (param.lam_n + 2 * param.mu_n)
            + param.l_s / (param.lam_s + 2 * param.mu_s)
            + param.l_p / (param.lam_p + 2 * param.mu_p)
        )

        param.omega_p = np.sqrt(2 * param.mu_p * param.S_2)

        param.A = 0
        param.B = param.S_1 * exp(2 * pi * param.omega_p)
        param.C = -param.B * exp(-2 * pi * param.omega_p)
        param.D = param.B - param.C
        param.E = 0
        param.F = param.D / (1 - exp(2 * pi * param.omega_p))
        param.G = 0
        param.H = 0

        # add parameters as an attribute
        self.param = param

    def f1(self, theta):
        param = self.param
        return (
            -param.S_1 + param.B * exp(-param.omega_p * (theta + 2 * pi))
        ) / param.S_2

    def f2(self, theta):
        param = self.param
        return param.E + (param.C + param.F) * exp(-param.omega_p * theta)

    def f3(self, theta):
        return self.f1(theta)

    def f4(self, theta):
        param = self.param
        return param.E + param.F * exp(-param.omega_p * theta)

    def g1(self, theta):
        param = self.param
        return (
            -(param.omega_p * param.C)
            / (param.mu_p * param.S_2)
            * exp(-param.omega_p * theta)
        )

    def g2(self, theta):
        param = self.param
        return param.G + (
            (param.C + param.F) / param.omega_p * exp(-param.omega_p * theta)
        )

    def g3(self, theta):
        return self.g1(theta)

    def g4(self, theta):
        param = self.param
        return param.H + (param.F / param.omega_p) * exp(-param.omega_p * theta)

    def Tp(self, theta):
        """
        Tension in the positive current collector
        Note: the tension is different for the first wind
        """
        r0 = self.param.r0
        theta1 = theta[theta < 2 * pi]
        theta2 = theta[theta >= 2 * pi]
        T1 = r0 * self.f1(theta1)
        T2 = r0 * (self.f1(theta2) - self.f1(theta2 - 2 * pi))
        return np.concatenate((T1, T2))

    def Tn(self, theta):
        """
        Tension in the negative current collector (zero to leading order)
        """
        return np.zeros_like(theta)
