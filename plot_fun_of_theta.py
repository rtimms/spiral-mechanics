import numpy as np
from numpy import pi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution
from comsol_solution import ComsolSolution

# set style for paper
# matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Parameters ------------------------------------------------------------------
alpha = 0.10  # expansion coefficient
delta = 0.1
hh = 0.005  # current collector thickness
E = 1  # active material Young's modulus
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
r1 = r0 + delta * N
omega = np.sqrt(mu / (lam + 2 * mu))
N_plot = 9  # number of winds to plot
path = "data/E1e4h005/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------
outer = OuterSolution(r0, delta, E, nu, alpha)

# Load COMSOL solution --------------------------------------------------------
comsol = ComsolSolution(r0, delta, hh, N, E, nu, path)
theta = comsol.theta

# Plot solution(s) ------------------------------------------------------------
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi

# f_i, g_i
fig, ax = plt.subplots(2, 2, figsize=(6.4, 4))
ax[0, 0].plot(theta, outer.f1(theta), "-", label="Asymptotic ")
ax[0, 0].plot(theta, comsol.f1, "-", label="COMSOL")
ax[0, 0].set_ylabel(r"$f_1$")
ax[0, 0].legend()
ax[0, 1].plot(theta, outer.f2(theta), "-", label="Asymptotic ")
ax[0, 1].plot(theta, comsol.f2, "-", label="COMSOL")
ax[0, 1].set_ylabel(r"$f_2$")
ax[1, 0].plot(theta, outer.g1(theta), "-", label="Asymptotic ")
ax[1, 0].plot(theta, comsol.g1, "-", label="COMSOL")
ax[1, 0].set_ylabel(r"$g_1$")
ax[1, 1].plot(theta, outer.g2(theta), "-", label="Asymptotic ")
ax[1, 1].plot(theta, comsol.g2, "-", label="COMSOL")
ax[1, 1].set_ylabel(r"$g_2$")
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
plt.savefig("figs" + path[4:] + "fg_of_theta.pdf", dpi=300)

# displacements at r = r0 + delta / 2 + delta * theta / 2 / pi
r = r0 + delta / 2 + delta * theta / 2 / pi
fig, ax = plt.subplots(1, 2, figsize=(6.4, 4))
ax[0].plot(theta, outer.u(r, theta), "-", label="Asymptotic")
ax[0].plot(theta, comsol.u, "-", label="COMSOL")
ax[0].set_ylabel(r"$u$")
ax[0].legend()
ax[1].plot(theta, outer.v(r, theta), "-", label="Asymptotic")
ax[1].plot(theta, comsol.v, "-", label="COMSOL")
ax[1].set_ylabel(r"$v$")
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
plt.savefig("figs" + path[4:] + "uv_of_theta.pdf", dpi=300)

# stresses and strains
fig, ax = plt.subplots(2, 3, figsize=(6.4, 4))
ax[0, 0].plot(theta, outer.e_rr(theta), "-", label="Asymptotic")
ax[0, 0].plot(theta, comsol.err, "-", label="COMSOL")
ax[0, 0].set_ylabel(r"$\varepsilon_{rr}$")
ax[0, 0].legend()
ax[0, 1].plot(theta, outer.e_tt(theta), "-", label="Asymptotic")
ax[0, 1].plot(theta, comsol.ett, "-", label="COMSOL")
ax[0, 1].set_ylabel(r"$\varepsilon_{\theta\theta}$")
ax[0, 2].plot(theta, outer.e_rt(theta), "-", label="Asymptotic")
ax[0, 2].plot(theta, comsol.ert, "-", label="COMSOL")
ax[0, 2].set_ylabel(r"$\varepsilon_{r\theta}$")
ax[1, 0].plot(theta, outer.s_rr(theta), "-", label="Asymptotic")
ax[1, 0].plot(theta, comsol.srr, "-", label="COMSOL")
ax[1, 0].set_ylabel(r"$\sigma_{rr}$")
ax[1, 1].plot(theta, outer.s_tt(theta), "-", label="Asymptotic")
ax[1, 1].plot(theta, comsol.stt, "-", label="COMSOL")
ax[1, 1].set_ylabel(r"$\sigma_{\theta\theta}$")
ax[1, 2].plot(theta, outer.s_rt(theta), "-", label="Asymptotic")
ax[1, 2].plot(theta, comsol.srt, "-", label="COMSOL")
ax[1, 2].set_ylabel(r"$\sigma_{r\theta}$")
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
plt.savefig("figs" + path[4:] + "stress_strain_of_theta.pdf", dpi=300)

# tension
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(theta, outer.T(theta), "-", label="Asymptotic")
ax.plot(theta, comsol.T, "-", label="COMSOL")
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
