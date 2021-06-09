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
r0 = 0.1
r1 = 1
N = 10
delta = (r1 - r0) / N
hh = 0.05 * delta
l_n = 0.4 / 2
l_s = 0.2 / 2
l_p = 0.4 / 2

alpha_n = -0.1
alpha_p = 0.02
E_n = 1
E_s = 1e-2
E_p = 10
nu_n = 1 / 3
nu_s = 1 / 3
nu_p = 1 / 3
lam_n = E_n * nu_n / (1 + nu_n) / (1 - 2 * nu_n)
mu_n = E_n / 2 / (1 + nu_n)
lam_s = E_s * nu_s / (1 + nu_s) / (1 - 2 * nu_s)
mu_s = E_s / 2 / (1 + nu_s)
lam_p = E_p * nu_p / (1 + nu_p) / (1 - 2 * nu_p)
mu_p = E_p / 2 / (1 + nu_p)

print("lam_n: ", lam_n, " lam_s: ", lam_s, " lam_p: ", lam_p)
print("mu_n: ", mu_n, " mu_s: ", mu_s, " mu_p: ", mu_p)
print(
    "lam_n+2*mu_n: ",
    lam_n + 2 * mu_n,
    " lam_s+2*mu_s: ",
    lam_s + 2 * mu_s,
    " lam_p+2*mu_p: ",
    lam_p + 2 * mu_p,
)

N_plot = N  # number of winds to plot
path = "data/jelly/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------

# Load COMSOL solution --------------------------------------------------------
comsol = ComsolSolution(
    r0, delta, l_n, l_s, l_p, hh, N, E_n, E_s, E_p, nu_n, nu_s, nu_p, path
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
