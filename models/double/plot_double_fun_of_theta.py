import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from comsol_double_solution import ComsolSolution

# set style for paper
# import matplotlib
# matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Parameters ------------------------------------------------------------------
r0 = 0.1
r1 = 1
N = 10
delta = (r1 - r0) / N
hh = 0.05 * delta
l_n = 0.5 / 2
l_p = 0.5 / 2

alpha_n = -0.1
alpha_p = 0.02
E_n = 1
E_p = 10
nu_n = 1 / 3
nu_p = 1 / 3
lam_n = E_n * nu_n / (1 + nu_n) / (1 - 2 * nu_n)
mu_n = E_n / 2 / (1 + nu_n)
lam_p = E_p * nu_p / (1 + nu_p) / (1 - 2 * nu_p)
mu_p = E_p / 2 / (1 + nu_p)
print("lam_n: ", lam_n, " lam_p: ", lam_p)
print("mu_n: ", mu_n, " mu_p: ", mu_p)
print("lam_n+2*mu_n: ", lam_n + 2 * mu_n, " lam_p+2*mu_p: ", lam_p + 2 * mu_p)

N_plot = 10  # number of winds to plot
path = "data/double/En1Ep10/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------

# Load COMSOL solution --------------------------------------------------------
comsol = ComsolSolution(r0, delta, l_n, l_p, hh, N, E_n, E_p, nu_n, nu_p, path)
theta = comsol.theta

# Plot solution(s) ------------------------------------------------------------
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi

# f_i, g_i
fig, ax = plt.subplots(4, 4, figsize=(12, 12))
ax[0, 0].plot(theta, comsol.f11, "-", label="COMSOL")
ax[0, 1].plot(theta, comsol.f12, "-", label="COMSOL")
ax[0, 2].plot(theta, comsol.g11, "-", label="COMSOL")
ax[0, 3].plot(theta, comsol.g12, "-", label="COMSOL")
ax[1, 0].plot(theta, comsol.f21, "-", label="COMSOL")
ax[1, 1].plot(theta, comsol.f22, "-", label="COMSOL")
ax[1, 2].plot(theta, comsol.g21, "-", label="COMSOL")
ax[1, 3].plot(theta, comsol.g22, "-", label="COMSOL")
ax[2, 0].plot(theta, comsol.f31, "-", label="COMSOL")
ax[2, 1].plot(theta, comsol.f32, "-", label="COMSOL")
ax[2, 2].plot(theta, comsol.g31, "-", label="COMSOL")
ax[2, 3].plot(theta, comsol.g32, "-", label="COMSOL")
ax[3, 0].plot(theta, comsol.f41, "-", label="COMSOL")
ax[3, 1].plot(theta, comsol.f42, "-", label="COMSOL")
ax[3, 2].plot(theta, comsol.g41, "-", label="COMSOL")
ax[3, 3].plot(theta, comsol.g42, "-", label="COMSOL")
ax[0, 0].set_ylabel(r"$f_{11}$")
ax[0, 1].set_ylabel(r"$f_{12}$")
ax[0, 2].set_ylabel(r"$g_{11}$")
ax[0, 3].set_ylabel(r"$g_{12}$")
ax[1, 0].set_ylabel(r"$f_{21}$")
ax[1, 1].set_ylabel(r"$f_{22}$")
ax[1, 2].set_ylabel(r"$g_{21}$")
ax[1, 3].set_ylabel(r"$g_{22}$")
ax[2, 0].set_ylabel(r"$f_{31}$")
ax[2, 1].set_ylabel(r"$f_{32}$")
ax[2, 2].set_ylabel(r"$g_{31}$")
ax[2, 3].set_ylabel(r"$g_{32}$")
ax[3, 0].set_ylabel(r"$f_{41}$")
ax[3, 1].set_ylabel(r"$f_{42}$")
ax[3, 2].set_ylabel(r"$g_{41}$")
ax[3, 3].set_ylabel(r"$g_{42}$")
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
ax[0, 0].plot(theta, comsol.f11, "-", label=r"$f_{11}$")
ax[0, 0].plot(theta, comsol.f21, "-", label=r"$f_{21}$")
ax[0, 0].plot(theta, comsol.f31, "-", label=r"$f_{31}$")
ax[0, 0].plot(theta, comsol.f41, "-", label=r"$f_{41}$")
ax[0, 1].plot(theta, comsol.f12, "-", label=r"$f_{12}$")
ax[0, 1].plot(theta, comsol.f22, "-", label=r"$f_{22}$")
ax[0, 1].plot(theta, comsol.f32, "-", label=r"$f_{32}$")
ax[0, 1].plot(theta, comsol.f42, "-", label=r"$f_{42}$")
ax[1, 0].plot(theta, comsol.g11, "-", label=r"$g_{11}$")
ax[1, 0].plot(theta, comsol.g21, "-", label=r"$g_{21}$")
ax[1, 0].plot(theta, comsol.g31, "-", label=r"$g_{31}$")
ax[1, 0].plot(theta, comsol.g41, "-", label=r"$g_{41}$")
ax[1, 1].plot(theta, comsol.g12, "-", label=r"$g_{12}$")
ax[1, 1].plot(theta, comsol.g22, "-", label=r"$g_{22}$")
ax[1, 1].plot(theta, comsol.g32, "-", label=r"$g_{32}$")
ax[1, 1].plot(theta, comsol.g42, "-", label=r"$g_{42}$")
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
ax[0].plot(theta, comsol.Tn, "-", label="COMSOL")
ax[1].plot(theta, comsol.Tp, "-", label="COMSOL")
ax[2].plot(theta, comsol.Tn + comsol.Tp, "-", label="COMSOL")
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
