import numpy as np
from numpy import pi, exp
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib import colorbar, cm, colors
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


# Outer solutions as a function of R ------------------------------------------
def u_outer(R):
    n = np.floor(R)
    return (
        alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * (R - n)
        + f1(2 * n * pi) * (R - n)
        + f2(2 * n * pi)
    )


def v_outer(R):
    n = np.floor(R)
    return g1(2 * n * pi) * (R - n) + g2(2 * n * pi)


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


# Plot displacements ----------------------------------------------------------
u_min = min([np.nanmin(c * u_data), f2(0), np.nanmin(c * u_data + u_outer(u_r_data))])
u_max = max([np.nanmax(c * u_data), 0, np.nanmax(c * u_data + u_outer(u_r_data))])
v_min = min([np.nanmin(c * v_data), g2(0), np.nanmin(c * v_data + v_outer(v_r_data))])
v_max = max([np.nanmax(c * v_data), 0, np.nanmax(c * v_data + v_outer(v_r_data))])

fig, ax = plt.subplots(3, 1)
cu_tilde_plot = ax[0].pcolormesh(
    u_t_data,
    u_r_data,
    c * u_data,
    vmin=u_min,
    vmax=u_max,
    cmap="viridis",
    shading="gouraud",
)
fig.colorbar(cu_tilde_plot, ax=ax[0])
ax[0].set_title(r"$c\tilde{u}$")
u_outer_plot = ax[1].pcolormesh(
    u_t_data,
    u_r_data,
    u_outer(u_r_data),
    vmin=u_min,
    vmax=u_max,
    cmap="viridis",
    shading="gouraud",
)
fig.colorbar(u_outer_plot, ax=ax[1])
ax[1].set_title(r"$u_{outer}$")
u_plot = ax[2].pcolormesh(
    u_t_data,
    u_r_data,
    c * u_data + u_outer(u_r_data),
    vmin=u_min,
    vmax=u_max,
    cmap="viridis",
    shading="gouraud",
)
fig.colorbar(u_plot, ax=ax[2])
ax[2].set_title(r"$u=c\tilde{u}+u_{outer}$")
for ax in ax.reshape(-1):
    ax.set_xlabel(r"$\Theta$")
    ax.set_ylabel(r"$R$")
plt.tight_layout()

fig, ax = plt.subplots(3, 1)
cv_tilde_plot = ax[0].pcolormesh(
    v_t_data,
    v_r_data,
    v_data,
    vmin=v_min,
    vmax=v_max,
    cmap="cividis",
    shading="gouraud",
)
fig.colorbar(cv_tilde_plot, ax=ax[0])
ax[0].set_title(r"$c\tilde{v}$")
v_outer_plot = ax[1].pcolormesh(
    v_t_data,
    v_r_data,
    v_outer(v_r_data),
    vmin=v_min,
    vmax=v_max,
    cmap="cividis",
    shading="gouraud",
)
fig.colorbar(v_outer_plot, ax=ax[1])
ax[1].set_title(r"$v_{outer}$")
v_plot = ax[2].pcolormesh(
    v_t_data,
    v_r_data,
    c * v_data + v_outer(v_r_data),
    vmin=v_min,
    vmax=v_max,
    cmap="cividis",
    shading="gouraud",
)
fig.colorbar(v_plot, ax=ax[2])
ax[2].set_title(r"$u=c\tilde{u}+u_{outer}$")
for ax in ax.reshape(-1):
    ax.set_xlabel(r"$\Theta$")
    ax.set_ylabel(r"$R$")
plt.tight_layout()

plt.show()
