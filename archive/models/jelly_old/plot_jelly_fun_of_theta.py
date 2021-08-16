import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from comsol_jelly_solution import ComsolSolution

# set style for paper
# import matplotlib
# matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Parameters ------------------------------------------------------------------

#  geometry
r0 = 0.25
r1 = 1
N = 10
delta = (r1 - r0) / N
hh = 0.01 * delta
l_p = 0.4 / 2
l_s = 0.2 / 2
l_n = 0.4 / 2

# material properties
alpha_p = 1  # expansion coefficient
mu_p = 1  # shear modulus
nu_p = 1 / 3  # Poisson ratio
lam_p = 2 * mu_p * nu_p / (1 - 2 * nu_p)  # 1st Lame parameter
alpha_s = 0  # expansion coefficient
mu_s = 1e-3  # shear modulus
nu_s = 1 / 3  # Poisson ratio
lam_s = 2 * mu_s * nu_s / (1 - 2 * nu_s)  # 1st Lame parameter
alpha_n = 1  # expansion coefficient
mu_n = 1e-1  # shear modulus
nu_n = 1 / 3  # Poisson ratio
lam_n = 2 * mu_n * nu_n / (1 - 2 * nu_n)  # 1st Lame parameter

N_plot = N - 1  # number of winds to plot
path = "data/jelly_old/04/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------

# Load COMSOL solution --------------------------------------------------------
alpha_scale = 0.1
comsol = ComsolSolution(
    r0,
    delta,
    l_n,
    l_s,
    l_p,
    hh,
    N,
    lam_n,
    lam_s,
    lam_p,
    mu_n,
    mu_s,
    mu_p,
    alpha_scale,
    path,
)
theta = comsol["theta"]

# Plot solution(s) ------------------------------------------------------------
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi

# f_i, g_i
fig, ax = plt.subplots(6, 4, figsize=(12, 12))
for i in [1, 2, 3, 4, 5, 6]:
    ax[i - 1, 0].plot(theta, comsol[f"f_{i}1"], "-", label="COMSOL")
    ax[i - 1, 1].plot(theta, comsol[f"f_{i}2"], "-", label="COMSOL")
    ax[i - 1, 2].plot(theta, comsol[f"g_{i}1"], "-", label="COMSOL")
    ax[i - 1, 3].plot(theta, comsol[f"g_{i}2"], "-", label="COMSOL")
    ax[i - 1, 0].set_ylabel(f"$f_{{{i}1}}$")
    ax[i - 1, 1].set_ylabel(f"$f_{{{i}2}}$")
    ax[i - 1, 2].set_ylabel(f"$g_{{{i}1}}$")
    ax[i - 1, 3].set_ylabel(f"$g_{{{i}2}}$")
for ax in ax.reshape(-1):
    for w in winds:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
    ax.set_xlabel(r"$\theta$")
plt.tight_layout()
plt.savefig("figs" + path[4:] + "fg_of_theta.pdf", dpi=300)

# f_i, g_i overlayed
fig, ax = plt.subplots(2, 2, figsize=(6.4, 4))
for i in [1, 2, 3, 4, 5, 6]:
    ax[0, 0].plot(theta, comsol[f"f_{i}1"], "-", label=f"$f_{{{i}1}}$")
    ax[0, 1].plot(theta, comsol[f"f_{i}2"], "-", label=f"$f_{{{i}2}}$")
    ax[1, 0].plot(theta, comsol[f"g_{i}1"], "-", label=f"$g_{{{i}1}}$")
    ax[1, 1].plot(theta, comsol[f"g_{i}2"], "-", label=f"$g_{{{i}2}}$")
for ax in ax.reshape(-1):
    ax.legend()
    for w in winds:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
    ax.set_xlabel(r"$\theta$")
plt.tight_layout()
plt.savefig("figs" + path[4:] + "fg_of_theta_overlayed.pdf", dpi=300)

# tension
fig, ax = plt.subplots(3, 1, figsize=(6.4, 6))
ax[0].plot(theta, comsol["Tn"], "-", label="COMSOL")
ax[1].plot(theta, comsol["Tp"], "-", label="COMSOL")
ax[2].plot(theta, comsol["Tn"] + comsol["Tp"], "-", label="COMSOL")
# ax[0].legend(loc="lower right")
# add shared labels etc.
ax[0].set_ylabel(r"$T_n$")
ax[1].set_ylabel(r"$T_p$")
ax[2].set_ylabel(r"$T_n+T_p$")
for ax in ax.reshape(-1):
    for w in winds:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
    ax.set_xlabel(r"$\theta$")
plt.tight_layout()
plt.savefig("figs" + path[4:] + "T_of_theta.pdf", dpi=300)

plt.show()
