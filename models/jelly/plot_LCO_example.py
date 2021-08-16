import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution
from comsol_jelly_solution import ComsolSolution

# set style for paper
import matplotlib

matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)


# Parameters ------------------------------------------------------------------
class Parameters:
    "Empty class which will contain the parameters as attributes"
    pass


params = Parameters()

#  geometry
params.r0 = 0.25
params.r1 = 1
params.N = 20
params.delta = (params.r1 - params.r0) / params.N
params.hh = 0.005 * params.delta
params.l_p = 0.55 / 2
params.l_s = 0.05 / 2
params.l_n = 0.4 / 2

# positive electrode material properties
params.alpha_p = -0.2  # expansion coefficient
params.mu_p = 1  # shear modulus
params.nu_p = 0.49  # Poisson ratio
params.lam_p = (
    2 * params.mu_p * params.nu_p / (1 - 2 * params.nu_p)
)  # 1st Lame parameter

# separator electrode material properties
params.alpha_s = 0  # expansion coefficient
params.mu_s = 0.1  # shear modulus
params.nu_s = 0.49  # Poisson ratio
params.lam_s = (
    2 * params.mu_s * params.nu_s / (1 - 2 * params.nu_s)
)  # 1st Lame parameter

# negative electrode material properties
params.alpha_n = 1  # expansion coefficient
params.mu_n = 1  # shear modulus
params.nu_n = 0.3  # Poisson ratio
params.lam_n = (
    2 * params.mu_n * params.nu_n / (1 - 2 * params.nu_n)
)  # 1st Lame parameter

N_plot = params.N - 1  # number of winds to plot
path = "data/jelly/LCO_example/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------
outer = OuterSolution(params)

# Load COMSOL solution --------------------------------------------------------
alpha_scale = 0.1
comsol = ComsolSolution(
    params,
    alpha_scale,
    path,
)
theta = comsol.theta

# Plot solution(s) ------------------------------------------------------------
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi

# f_i, g_i
fig, ax = plt.subplots(2, 3, figsize=(6.4, 4))
ax[0, 0].plot(theta, outer.f1(theta), "-", label="Asymptotic")
ax[0, 0].plot(theta, comsol.f1, "--", label="COMSOL")
ax[0, 0].set_ylabel(r"$f_{{1,3}}$")
ax[0, 1].plot(theta, outer.f2(theta), "-", label="Asymptotic")
ax[0, 1].plot(theta, comsol.f2, "--", label="COMSOL")
ax[0, 1].set_ylabel(r"$f_2$")
ax[0, 2].plot(theta, outer.f4(theta), "-", label="Asymptotic")
ax[0, 2].plot(theta, comsol.f4, "--", label="COMSOL")
ax[0, 2].set_ylabel(r"$f_4$")
ax[1, 0].plot(theta, outer.g1(theta), "-", label="Asymptotic")
ax[1, 0].plot(theta, comsol.g1, "--", label="COMSOL")
ax[1, 0].set_ylabel(r"$g_{{1,3}}$")
ax[1, 1].plot(theta, outer.g2(theta), "-", label="Asymptotic")
ax[1, 1].plot(theta, comsol.g2, "--", label="COMSOL")
ax[1, 1].set_ylabel(r"$g_2$")
ax[1, 2].plot(theta, outer.g4(theta), "-", label="Asymptotic")
ax[1, 2].plot(theta, comsol.g4, "--", label="COMSOL")
ax[1, 2].set_ylabel(r"$g_4$")
# add shared labels etc.
fig.subplots_adjust(left=0.1, bottom=0.25, right=0.98, top=0.98, wspace=0.4, hspace=0.4)
ax[1, 1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.4),
    borderaxespad=0.0,
    ncol=2,
)
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
plt.savefig("figs" + path[4:] + "fg_of_theta_jelly_test.pdf", dpi=300)

# tension
fig, ax = plt.subplots(2, 1, figsize=(6.4, 4))
ax[0].plot(theta, outer.Tp(theta), "-", label="Asymptotic")
ax[0].plot(theta, comsol.Tp, "--", label="COMSOL")
ax[0].set_ylabel(r"$T_+$")
ax[1].plot(theta, outer.Tn(theta), "-", label="Asymptotic")
ax[1].plot(theta, comsol.Tn, "--", label="COMSOL")
ax[1].set_ylabel(r"$T_-$")
ax[1].legend(loc="lower right")
# ax[0].set_ylim([-2, 0.05])
# ax[1].set_ylim([-2, 0.05])
# add shared labels etc.
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
plt.savefig("figs" + path[4:] + "T_of_theta_jelly_test.pdf", dpi=300)

plt.show()
