import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution
from comsol_solution import ComsolSolution

# set style for paper
import matplotlib

matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Parameters (dimensionless) --------------------------------------------------
alpha = 1  # expansion coefficient
mu = 1  # shear modulus
nu = 1 / 3  # Poisson ratio
lam = 2 * mu * nu / (1 - 2 * nu)  # 1st Lame parameter
N = 10  # number of winds
r0 = 0.25  # inner radius
r1 = 1  # outer radius
delta = (r1 - r0) / N
hh = 0.01 * delta  # current collector thickness
N_plot = 9  # number of winds to plot
path = "data/single/mu1lam2/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------
outer = OuterSolution(r0, delta, mu, lam, alpha)

# Load COMSOL solution --------------------------------------------------------
alpha_scale = 0.1
comsol = ComsolSolution(r0, delta, hh, N, mu, lam, alpha_scale, path)
theta = comsol.theta

# Plot solution(s) ------------------------------------------------------------
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi

# f_i, g_i
fig, ax = plt.subplots(2, 2, figsize=(6.4, 4))
ax[0, 0].plot(theta, outer.f1(theta), "-", label="Asymptotic ")
ax[0, 0].plot(theta, comsol.f1, "--", label="COMSOL")
ax[0, 0].set_ylabel(r"$f_1$")
ax[0, 1].plot(theta, outer.f2(theta), "-", label="Asymptotic ")
ax[0, 1].plot(theta, comsol.f2, "--", label="COMSOL")
ax[0, 1].set_ylabel(r"$f_2$")
ax[1, 0].plot(theta, outer.g1(theta), "-", label="Asymptotic ")
ax[1, 0].plot(theta, comsol.g1, "--", label="COMSOL")
ax[1, 0].set_ylabel(r"$g_1$")
ax[1, 1].plot(theta, outer.g2(theta), "-", label="Asymptotic ")
ax[1, 1].plot(theta, comsol.g2, "--", label="COMSOL")
ax[1, 1].set_ylabel(r"$g_2$")
# add shared labels etc.
fig.subplots_adjust(
    left=0.1, bottom=0.25, right=0.98, top=0.98, wspace=0.33, hspace=0.4
)
ax[1, 0].legend(
    loc="upper center",
    bbox_to_anchor=(1.1, -0.4),
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
plt.savefig("figs" + path[4:] + "fg_of_theta.pdf", dpi=300)

# displacements at r = r0 + delta / 2 + delta * theta / 2 / pi
r = r0 + delta / 2 + delta * theta / 2 / pi
fig, ax = plt.subplots(1, 2, figsize=(6.4, 4))
ax[0].plot(theta, outer.u(r, theta), "-", label="Asymptotic")
ax[0].plot(theta, comsol.u, "--", label="COMSOL")
ax[0].set_ylabel(r"$u$")
ax[1].plot(theta, outer.v(r, theta), "-", label="Asymptotic")
ax[1].plot(theta, comsol.v, "--", label="COMSOL")
ax[1].set_ylabel(r"$v$")
# add shared labels etc.
ax[0].legend(loc="upper right")
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
plt.savefig("figs" + path[4:] + "uv_of_theta.pdf", dpi=300)

# stresses and strains at r = r0 + delta / 2 + delta * theta / 2 / pi
r = r0 + delta / 2 + delta * theta / 2 / pi
fig, ax = plt.subplots(2, 3, figsize=(6.4, 4))
ax[0, 0].plot(theta, outer.e_rr(theta), "-", label="Asymptotic")
ax[0, 0].plot(theta, comsol.err, "--", label="COMSOL")
ax[0, 0].set_ylabel(r"$\varepsilon_{rr}$")
ax[0, 1].plot(theta, outer.e_tt(r, theta), "-", label="Asymptotic")
ax[0, 1].plot(theta, comsol.ett, "--", label="COMSOL")
ax[0, 1].set_ylabel(r"$\varepsilon_{\theta\theta}$")
ax[0, 2].plot(theta, outer.e_rt(theta), "-", label="Asymptotic")
ax[0, 2].plot(theta, comsol.ert, "--", label="COMSOL")
ax[0, 2].set_ylabel(r"$\varepsilon_{r\theta}$")
ax[1, 0].plot(theta, outer.s_rr(theta), "-", label="Asymptotic")
ax[1, 0].plot(theta, comsol.srr, "--", label="COMSOL")
ax[1, 0].set_ylabel(r"$\sigma_{rr}$")
ax[1, 1].plot(theta, outer.s_tt(theta), "-", label="Asymptotic")
ax[1, 1].plot(theta, comsol.stt, "--", label="COMSOL")
ax[1, 1].set_ylabel(r"$\sigma_{\theta\theta}$")
ax[1, 2].plot(theta, outer.s_rt(theta), "-", label="Asymptotic")
ax[1, 2].plot(theta, comsol.srt, "--", label="COMSOL")
ax[1, 2].set_ylabel(r"$\sigma_{r\theta}$")
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
plt.savefig("figs" + path[4:] + "stress_strain_of_theta.pdf", dpi=300)

# tension
fig, ax = plt.subplots(figsize=(6.4, 2))
ax.plot(theta, outer.T(theta), "-", label="Asymptotic")
ax.plot(theta, comsol.T, "--", label="COMSOL")
ax.set_ylabel(r"$T$")
ax.legend(loc="lower right")
# add shared labels etc.
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
