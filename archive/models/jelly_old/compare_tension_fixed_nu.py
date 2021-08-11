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

# electrode material properties
alpha_p = 1  # expansion coefficient
mu_p = 1  # shear modulus
nu_p = 1 / 3  # Poisson ratio
lam_p = 2 * mu_p * nu_p / (1 - 2 * nu_p)  # 1st Lame parameter
alpha_n = 1  # expansion coefficient
mu_n = 1e-1  # shear modulus
nu_n = 1 / 3  # Poisson ratio
lam_n = 2 * mu_n * nu_n / (1 - 2 * nu_n)  # 1st Lame parameter

N_plot = N - 1  # number of winds to plot

# Loop over separator properties and plot tension -----------------------------

fig, ax = plt.subplots(2, 1, figsize=(6.4, 6))

alpha_s = 0  # expansion coefficient
nu_s = 1 / 3  # Poisson ratio
mu_s_list = [1e-2, 1e-1, 1]  # shear moduli
paths = [
    "data/jelly/mus1e-2/",
    "data/jelly_old/mus1e-1/",
    "data/jelly/mus1/",
]  # paths to data
alpha_scale = 0.1

for mu_s, path in zip(mu_s_list, paths):
    lam_s = 2 * mu_s * nu_s / (1 - 2 * nu_s)  # 1st Lame parameter
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
    ax[0].plot(comsol["theta"], comsol["Tn"], "-", label=r"$\mu_s/\mu_a=$" + f"{mu_s}")
    ax[1].plot(comsol["theta"], comsol["Tp"], "-", label=r"$\mu_s/\mu_a=$" + f"{mu_s}")
# add shared labels etc.
ax[1].legend(loc="upper right")
ax[0].set_ylabel(r"$T_n$")
ax[1].set_ylabel(r"$T_p$")
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi
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
plt.savefig("figs/jelly_old/compare_mu_s/T_of_theta.pdf", dpi=300)

plt.show()
