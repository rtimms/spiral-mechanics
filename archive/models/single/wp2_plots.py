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
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the outer displacements ---------------------------------------------

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
comsol = pd.read_csv(path + "srr_grid.csv", comment="#", header=None).to_numpy()
srr_t_data = comsol[:, 0].reshape(101, 101)
srr_r_data = comsol[:, 1].reshape(101, 101)
srr_data = comsol[:, 2].reshape(101, 101)
comsol = pd.read_csv(path + "srt_grid.csv", comment="#", header=None).to_numpy()
srt_t_data = comsol[:, 0].reshape(101, 101)
srt_r_data = comsol[:, 1].reshape(101, 101)
srt_data = comsol[:, 2].reshape(101, 101)

# load f_i and g_i from full simulation
full_path = "data/single/E1e4h005/"
theta = np.linspace(0, 2 * N * pi, 60 * N)
# f1 = sigma_rr / (lambda+2*mu)
comsol = pd.read_csv(full_path + "srr3.csv", comment="#", header=None).to_numpy()
f1_r_data = comsol[:, 0]
f1_data = comsol[:, 1] / (lam + 2 * mu)
f1_interp = interp.interp1d(f1_r_data, f1_data, bounds_error=False)
# In COMSOL we evaluate f_1 at r = r0+delta/2+delta*theta/2/pi
r = r0 + delta / 2 + delta * theta / 2 / pi
f1_comsol = f1_interp(r)
# f2 = u(R=theta/2/pi)/delta
comsol = pd.read_csv(full_path + "u1.csv", comment="#", header=None).to_numpy()
f2_r_data = comsol[:, 0]
f2_data = comsol[:, 1] / delta
f2_interp = interp.interp1d(f2_r_data, f2_data, bounds_error=False)
# In COMSOL we evaluate f_2 at r = r0+hh/2+delta*theta/2/pi
r = r0 + hh / 2 + delta * theta / 2 / pi
f2_comsol = f2_interp(r)
# g1 = sigma_rt/mu
comsol = pd.read_csv(full_path + "srt3.csv", comment="#", header=None).to_numpy()
g1_r_data = comsol[:, 0]
g1_data = comsol[:, 1] / mu
g1_interp = interp.interp1d(g1_r_data, g1_data, bounds_error=False)
# In COMSOL we evaluate g_1 at r = r0+delta/2+delta*theta/2/pi
r = r0 + delta / 2 + delta * theta / 2 / pi
g1_comsol = g1_interp(r)
# g2 = v(R=theta/2/pi)/delta
comsol = pd.read_csv(full_path + "v1.csv", comment="#", header=None).to_numpy()
g2_r_data = comsol[:, 0]
g2_data = comsol[:, 1] / delta
g2_interp = interp.interp1d(g2_r_data, g2_data, bounds_error=False)
# In COMSOL we evaluate g_2 at r = r0+hh/2+delta*theta/2/pi
r = r0 + hh / 2 + delta * theta / 2 / pi
g2_comsol = g2_interp(r)

# Plots -----------------------------------------------------------------------

# s_rr and s_rt
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(
    theta,
    f1_comsol * (lam + 2 * mu),
    linestyle="--",
    color="tab:orange",
    label="COMSOL",
)
ax[1].plot(theta, g1_comsol * mu, linestyle="--", color="tab:orange", label="COMSOL")
# plot inner and composite solutions
for n in range(N):
    idx1 = int(n * 100 / N + 10)
    idx2 = int(n * 100 / N)
    if n == 0:
        Theta = u_t_data[0, 50:]
        srr_tilde = srr_data[idx1, 50:]
        u_tilde = u_data[idx2, 50:]
        v_tilde = v_data[idx2, 50:]
        srt_tilde = srt_data[idx1, 50:]
    else:
        Theta = u_t_data[0, :]
        srr_tilde = srr_data[idx1, :]
        u_tilde = u_data[idx2, :]
        v_tilde = v_data[idx2, :]
        srt_tilde = srt_data[idx1, :]

    theta = delta * Theta / r0 + 2 * n * pi
    ax[0].plot(
        theta,
        (
            c * srr_tilde
            + (alpha * (3 * lam + 2 * mu) + (lam + 2 * mu) * f1(theta))
            - alpha * (3 * lam + 2 * mu)
        ),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    ax[1].plot(
        theta,
        c * srt_tilde + g1(theta) * mu,
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )

ax[0].set_ylabel(r"$\sigma_{rr}$")
ax[1].set_ylabel(r"$\sigma_{rt}$")
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
plt.savefig("figs" + path[4:] + "srr_srt.pdf", dpi=300)
plt.show()
