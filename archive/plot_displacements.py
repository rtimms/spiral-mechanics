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
hh = 0.01  # current collector thickness
E = 1  # active material Young's modulus
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
r1 = r0 + delta * N
omega = np.sqrt(mu / (lam + 2 * mu))
N_plot = 9  # number of winds to plot
path = "data/fixed/h001/"  # path to data

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
    return D + C / omega * exp(-omega * theta)


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

fig, ax = plt.subplots(2, 2)

# at theta = 0
r = []
theta = [2 * pi * n for n in list(range(N_plot))]
for tt in theta:
    r.append(r0 + delta * tt / 2 / pi)
r = np.array(r)
theta = np.array(theta)
winds_m = [r0 - hh / 2 + delta * n for n in list(range(N_plot))]
winds = [r0 + delta * n for n in list(range(N_plot))]
winds_p = [r0 + hh / 2 + delta * n for n in list(range(N_plot))]

# radial displacement
ax[0, 0].plot(r, u(r, theta), "-", label="Asymptotic")
comsol = pd.read_csv(path + f"u_0.csv", comment="#", header=None).to_numpy()
ax[0, 0].plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
ax[0, 0].set_ylabel(r"$u$")
ax[0, 0].set_title(r"$\theta=0$")
ax[0, 0].legend()
for w_m, w_p in zip(winds_m, winds_p):
    ax[0, 0].axvline(x=w_m, linestyle=":", color="lightgrey")
    ax[0, 0].axvline(x=w_p, linestyle=":", color="lightgrey")
ax[0, 0].set_xlim([r[0], r[-1]])
ax[0, 0].set_xlabel(r"$r$")

# azimuthal displacement
ax[0, 1].plot(r, v(r, theta), "-", label="Asymptotic")
comsol = pd.read_csv(path + f"v_0.csv", comment="#", header=None).to_numpy()
ax[0, 1].plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
ax[0, 1].set_xlim([r[0], r[-1]])
ax[0, 1].set_ylabel(r"$v$")
ax[0, 1].set_title(r"$\theta=0$")
for w_m, w_p in zip(winds_m, winds_p):
    ax[0, 1].axvline(x=w_m, linestyle=":", color="lightgrey")
    ax[0, 1].axvline(x=w_p, linestyle=":", color="lightgrey")
ax[0, 1].set_xlim([r[0], r[-1]])
ax[0, 1].set_xlabel(r"$r$")

# at theta = pi
r = []
theta = [2 * pi * n + pi for n in list(range(N_plot))]
for tt in theta:
    r.append(r0 + delta * tt / 2 / pi)
r = np.array(r)
theta = np.array(theta)
winds_m = [
    r0 - hh / 2 + delta * (2 * pi * n + pi) / 2 / pi for n in list(range(N_plot))
]
winds = [r0 + delta * (2 * pi * n + pi) / 2 / pi for n in list(range(N_plot))]
winds_p = [
    r0 + hh / 2 + delta * (2 * pi * n + pi) / 2 / pi for n in list(range(N_plot))
]

# radial displacement
ax[1, 0].plot(r, u(r, theta), "-", label="Asymptotic")
comsol = pd.read_csv(path + f"u_pi.csv", comment="#", header=None).to_numpy()
ax[1, 0].plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
ax[1, 0].set_ylabel(r"$u$")
ax[1, 0].set_title(r"$\theta=\pi$")
for w_m, w_p in zip(winds_m, winds_p):
    ax[1, 0].axvline(x=w_m, linestyle=":", color="lightgrey")
    ax[1, 0].axvline(x=w_p, linestyle=":", color="lightgrey")
ax[1, 0].set_xlim([r[0], r[-1]])
ax[1, 0].set_xlabel(r"$r$")
# azimuthal displacement
ax[1, 1].plot(r, v(r, theta), "-", label="Asymptotic")
comsol = pd.read_csv(path + f"v_pi.csv", comment="#", header=None).to_numpy()
ax[1, 1].plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
ax[1, 1].set_xlim([r[0], r[-1]])
ax[1, 1].set_ylabel(r"$v$")
ax[1, 1].set_title(r"$\theta=\pi$")
for w_m, w_p in zip(winds_m, winds_p):
    ax[1, 1].axvline(x=w_m, linestyle=":", color="lightgrey")
    ax[1, 1].axvline(x=w_p, linestyle=":", color="lightgrey")
ax[1, 1].set_xlim([r[0], r[-1]])
ax[1, 1].set_xlabel(r"$r$")
plt.tight_layout()
plt.savefig("figs/displacements.pdf", dpi=300)

plt.show()
