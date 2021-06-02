import numpy as np
from numpy import pi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from comsol_double_solution import ComsolSolution

# set style for paper
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
E_p = 1
nu_n = 1 / 3
nu_p = 1 / 3
lam_n = E_n * nu_n / (1 + nu_n) / (1 - 2 * nu_n)
mu_n = E_n / 2 / (1 + nu_n)
lam_p = E_p * nu_p / (1 + nu_p) / (1 - 2 * nu_p)
mu_p = E_p / 2 / (1 + nu_p)

N_plot = 10  # number of winds to plot
path = "data/double/"  # path to data
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
