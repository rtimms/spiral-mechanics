import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from comsol_solution import ComsolSolution

# set style for paper
# import matplotlib

# matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Parameters (dimensionless) --------------------------------------------------
alpha = 1  # expansion coefficient
alpha_cc = 0.1  # expansion coefficient
mu = 1  # shear modulus
nu = 1 / 3  # Poisson ratio
lam = 2 * mu * nu / (1 - 2 * nu)  # 1st Lame parameter
N = 10  # number of winds
r0 = 0.25  # inner radius
r1 = 1  # outer radius
delta = (r1 - r0) / N
hh = 0.01 * delta  # current collector thickness
N_plot = 4  # number of winds to plot

# Load COMSOL solution --------------------------------------------------------
alpha_scale = 0.1
comsol = ComsolSolution(
    r0, delta, hh, N, mu, lam, alpha, alpha_cc, alpha_scale, "data/a1al01/"
)
comsol_sf = ComsolSolution(
    r0, delta, hh, N, mu, lam, alpha, alpha_cc, alpha_scale, "data/a1al01_sf/"
)
theta = comsol.theta

# Plot solution(s) ------------------------------------------------------------
# plot dashed line every 2*pi
winds = [2 * pi * n for n in list(range(N_plot))]

comsols = [comsol, comsol_sf]
labels = ["fixed u", "stress free"]

# f_i, g_i
fig, ax = plt.subplots(2, 2, figsize=(6.4, 4))
for comsol, label in zip(comsols, labels):
    ax[0, 0].plot(theta, comsol.f1, "--", label=label)
    ax[0, 1].plot(theta, comsol.f2, "--", label=label)
    ax[1, 0].plot(theta, comsol.g1, "--", label=label)
    ax[1, 1].plot(theta, comsol.g2, "--", label=label)
ax[0, 0].set_ylabel(r"$f_1$")
ax[0, 1].set_ylabel(r"$f_2$")
ax[1, 0].set_ylabel(r"$g_1$")
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

# displacements at r = r0 + delta / 2 + delta * theta / 2 / pi
r = r0 + delta / 2 + delta * theta / 2 / pi
fig, ax = plt.subplots(1, 2, figsize=(6.4, 4))
for comsol, label in zip(comsols, labels):
    ax[0].plot(theta, comsol.u, "--", label=label)
    ax[1].plot(theta, comsol.v, "--", label=label)
ax[0].set_ylabel(r"$u$")
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

# stresses and strains at r = r0 + delta / 2 + delta * theta / 2 / pi
r = r0 + delta / 2 + delta * theta / 2 / pi
fig, ax = plt.subplots(2, 3, figsize=(6.4, 4))
for comsol, label in zip(comsols, labels):
    ax[0, 0].plot(theta, comsol.err, "--", label=label)
    ax[0, 1].plot(theta, comsol.ett, "--", label=label)
    ax[0, 2].plot(theta, comsol.ert, "--", label=label)
    ax[1, 0].plot(theta, comsol.srr, "--", label=label)
    ax[1, 1].plot(theta, comsol.stt, "--", label=label)
    ax[1, 2].plot(theta, comsol.srt, "--", label=label)
ax[0, 0].set_ylabel(r"$\varepsilon_{rr}$")
ax[0, 1].set_ylabel(r"$\varepsilon_{\theta\theta}$")
ax[0, 2].set_ylabel(r"$\varepsilon_{r\theta}$")
ax[1, 0].set_ylabel(r"$\sigma_{rr}$")
ax[1, 1].set_ylabel(r"$\sigma_{\theta\theta}$")
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

# tension
fig, ax = plt.subplots(figsize=(6.4, 2))
for comsol, label in zip(comsols, labels):
    ax.plot(theta, comsol.T, "--", label=label)
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

plt.show()
