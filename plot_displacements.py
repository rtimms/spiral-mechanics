#
# Plot the results in the boundary layer near r=r0 in the jelly roll mechanics
# problem
#
import numpy as np
from numpy import pi, exp
import pandas as pd
import matplotlib.pyplot as plt

# Parameters ------------------------------------------------------------------
alpha = 0.10  # expansion coefficient
delta = 0.1
E = 1  # active material Young's modulus
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
r1 = r0 + delta * N
omega = np.sqrt(mu / (lam + 2 * mu))
N_plot = 4  # number of winds to plot
path = "data/d01h0001/"  # path to data

# Compute the boundary layer solution -----------------------------------------

# constants
A = -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * exp(2 * pi * omega)
B = 0
C = A / (1 - exp(2 * pi * omega))
D = 0


# functions of theta
def f1(theta):
    return -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) - A * exp(
        -omega * (theta + 2 * pi)
    )


def f2(theta):
    return B + C * exp(-omega * theta)


def g1(theta):
    return -(lam + 2 * mu) * omega / (2 * pi * mu) * A * exp(-omega * (theta + 2 * pi))


def g2(theta):
    return D - C / omega * exp(-omega * theta)


# radial displacement
def u(r, theta):
    R = (r - r0) / delta
    return (
        alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * (R - theta / 2 / pi)
        + f1(theta) * (R - theta / 2 / pi)
        + f2(theta)
    )


# azimuthal displacement
def v(r, theta):
    R = (r - r0) / delta
    return g1(theta) * (R - theta / 2 / pi) + g2(theta)


# Plot solution(s) ------------------------------------------------------------
theta = pi
r = np.linspace(
    r0 + delta * theta / 2 / pi, r0 + delta * (N_plot - 1) + delta * theta / 2 / pi, 100
)

# radial displacement
fig, ax = plt.subplots()
plt.plot(r, u(r, theta), "-", label="Asymptotic")
comsol = pd.read_csv(path + f"u_pi.csv", comment="#", header=None).to_numpy()
plt.plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
plt.xlim([r0, r0 + delta * N_plot])
plt.xlabel(r"$r$")
plt.ylabel(r"$u$")
plt.title(r"$u(r; \theta=\pi)$")
plt.legend()

# azimuthal displacement
fig, ax = plt.subplots()
plt.plot(r, v(r, theta), "-", label="Asymptotic")
comsol = pd.read_csv(path + f"u_pi.csv", comment="#", header=None).to_numpy()
plt.plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
plt.xlim([r0, r0 + delta * N_plot])
plt.xlabel(r"$r$")
plt.ylabel(r"$v$")
plt.title(r"$v(r; \theta=\pi)$")
plt.legend()

plt.show()
