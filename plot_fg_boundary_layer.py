import numpy as np
from numpy import pi, exp
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib import colorbar, cm, colors
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os

# Parameters ------------------------------------------------------------------
alpha = 0.10  # expansion coefficient
delta = 0.1
hh = 0.005  # current collector thickness
E = 1  # active material Young's modulus
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 5
r0 = 0.5
r1 = r0 + delta * N
omega = np.sqrt(mu / (lam + 2 * mu))
c = alpha * (2 * lam + mu) * omega
N_plot = 9  # number of winds to plot
path = "data/boundary_layer/"  # path to data
## make directory for figures if it doesn't exist
# try:
#    os.mkdir("figs" + path[4:])
# except FileExistsError:
#    pass

# Compute the boundary layer displacements ------------------------------------

# constants
A = alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * exp(2 * pi * omega)
B = 0
C = A / (1 - exp(2 * pi * omega))
D = 0


# functions of theta
def f1(theta):
    return -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + A * exp(
        -omega * (theta + 2 * pi)
    )


def f2(theta):
    return B + C * exp(-omega * theta)


def g1(theta):
    return (lam + 2 * mu) / mu * omega * A * exp(-omega * (theta + 2 * pi))


def g2(theta):
    return D + C / omega * exp(-omega * theta)


# Load COMSOL data ------------------------------------------------------------
# Note: currently exported on a 101x101 grid
comsol = pd.read_csv(path + "u_grid.csv", comment="#", header=None).to_numpy()
u_t_data = comsol[:, 0].reshape(101, 101)
u_r_data = comsol[:, 1].reshape(101, 101)
u_data = comsol[:, 2].reshape(101, 101)
comsol = pd.read_csv(path + "v_grid.csv", comment="#", header=None).to_numpy()
v_t_data = comsol[:, 0].reshape(101, 101)
v_r_data = comsol[:, 1].reshape(101, 101)
v_data = comsol[:, 2].reshape(101, 101)

# load f2 and g2
path = "data/E1e4h005/"
theta = np.linspace(0, 2 * N * pi, 60 * N)
# f2 = u(R=theta/2/pi)/delta
comsol = pd.read_csv(path + "u1.csv", comment="#", header=None).to_numpy()
f2_r_data = comsol[:, 0]
f2_data = comsol[:, 1] / delta
f2_interp = interp.interp1d(f2_r_data, f2_data, bounds_error=False)
# In COMSOL we evaluate f_2 at r = r0+hh/2+delta*theta/2/pi
r = r0 + hh / 2 + delta * theta / 2 / pi
f2_comsol = f2_interp(r)
# g2 = v(R=theta/2/pi)/delta
comsol = pd.read_csv(path + "v1.csv", comment="#", header=None).to_numpy()
g2_r_data = comsol[:, 0]
g2_data = comsol[:, 1] / delta
g2_interp = interp.interp1d(g2_r_data, g2_data, bounds_error=False)
# In COMSOL we evaluate g_2 at r = r0+hh/2+delta*theta/2/pi
r = r0 + hh / 2 + delta * theta / 2 / pi
g2_comsol = g2_interp(r)

# Plots -----------------------------------------------------------------------
# f2 and g2
fig, ax = plt.subplots(1, 2)
# plot COMSOL solutions
ax[0].plot(theta, f2_comsol, linestyle="-", color="tab:orange", label="COMSOL")
ax[1].plot(theta, g2_comsol, linestyle="-", color="tab:orange", label="COMSOL")
# plot outer solutions
ax[0].plot(theta, f2(theta), linestyle=":", color="black", label="Outer")
ax[1].plot(theta, g2(theta), linestyle=":", color="black", label="Outer")
# plot inner and composite solutions
for n in range(N):
    idx = int(n * 100 / N)
    if n == 0:
        Theta = u_t_data[0, 50:]
        u_tilde = u_data[idx, 50:]
        v_tilde = v_data[idx, 50:]
    else:
        Theta = u_t_data[0, :]
        u_tilde = u_data[idx, :]
        v_tilde = v_data[idx, :]
    theta = delta * Theta / r0 + 2 * n * pi
    ax[0].plot(
        theta,
        c * u_tilde + f2(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[0].plot(
        theta,
        c * u_tilde + f2(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    ax[1].plot(
        theta,
        c * v_tilde + g2(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[1].plot(
        theta,
        c * v_tilde + g2(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
ax[0].set_ylabel(r"$f_2$")
ax[1].set_ylabel(r"$g_2$")
ax[0].legend()
for ax in ax.reshape(-1):
    # plot dashed line every 2*pi
    winds = [2 * pi * n for n in list(range(N))]
    for w in winds:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    # add labels etc.
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: "{}$\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N * 2 * pi])
    ax.set_xlabel(r"$\theta$")
plt.tight_layout()
plt.show()
