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

alpha_scale = 0.1
alpha_s = 0  # expansion coefficient

K_s_list = [1, 1, 1, 1e-1, 1e-1, 1e-2]  # bulk moduli
mu_s_list = [1, 1e-1, 1e-2, 1e-1, 1e-2, 1e-2]  # shear moduli
paths = [
    "Ks1_mus1/",
    "Ks1_mus1e-1/",
    "Ks1_mus1e-2/",
    "Ks1e-1_mus1e-1/",
    "Ks1e-1_mus1e-2/",
    "Ks1e-2_mus1e-2/",
]  # paths to data
linestyles = ["r-", "r--", "r:", "b--", "b:", "g:"]
for K_s, mu_s, path, linestyle in zip(K_s_list, mu_s_list, paths, linestyles):
    nu_s = (3 * K_s - 2 * mu_s) / 2 / (3 * K_s + mu_s)  # Poisson ratio
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
        "data/jelly/" + path,
    )
    ax[0].plot(comsol["theta"], comsol["Tn"], linestyle)
    ax[1].plot(
        comsol["theta"],
        comsol["Tp"],
        linestyle,
        label=r"$K_s/K_a=$"
        + f"{K_s}, "
        + r"$\mu_s/\mu_a=$"
        + f"{mu_s}, "
        + r"$\nu_s/\nu_a=$"
        + f"{nu_s:.2f}",
    )
# add shared labels etc.
ax[1].legend(loc="best")
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
