import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution
from comsol_solution import ComsolSolution

# set style for paper
# import matplotlib
#
# matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Parameters (dimensionless) --------------------------------------------------
alpha = 1  # expansion coefficient
mu = 1  # shear modulus
nu = 1 / 3  # Poisson ratio
lam = 2 * mu * nu / (1 - 2 * nu)  # 1st Lame parameter
N = 10  # number of winds
r0 = 0.25  # inner radius
r1 = 1  # outer radius
delta = (r1 - r0) / N
hs = 0.01
hh = hs * delta  # current collector thickness
N_plot = 9  # number of winds to plot
alpha_scale = 0.1  # scale for COMSOL

# Load and plot solutions -----------------------------------------------------
theta = np.linspace(0, 2 * pi * N, 60 * (N - 1))
outer = OuterSolution(r0, delta, mu, lam, alpha)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# strain at r = r0 + delta / 2 + delta * theta / 2 / pi
r = r0 + delta / 2 + delta * theta / 2 / pi
ax[0].plot(theta, outer.e_tt(r, theta), "-", label="Asymptotic")
ax[1].plot(theta, outer.T(theta), "-", label="Asymptotic")
# mu_ccs = [1e3, 5e3, 1e4, 2e4, 1e5]
mu_ccs = [1e3, 1e4, 1e5]
# paths = ["1e3/", "5e3/", "mu1lam2/", "2e4/", "1e5/"]
paths = ["1e3/", "1e4/", "1e5/"]
paths = ["nu45/" + path for path in paths]
for mu_cc, path in zip(mu_ccs, paths):
    if hs == 0.01:
        pass
    elif hs == 0.025:
        path = "hh025/" + path
    elif hs == 0.05:
        path = "hh05/" + path
    comsol = ComsolSolution(
        r0, delta, hh, N, mu, lam, alpha_scale, "data/single/" + path
    )
    theta = comsol.theta
    ax[0].plot(theta, comsol.ett, "--")
    ax[1].plot(theta, comsol.T, "--", label=r"COMSOL ($\mu$=" + f"{int(mu_cc)})")
ax[0].set_ylim([-0.5, 1])
ax[1].set_ylim([-2.5, 0.5])
ax[0].set_ylabel(r"$\varepsilon_{\theta\theta}$")
ax[1].set_ylabel(r"$T$")

# add shared labels etc.
ax[0].set_title(r"$h=$" + f"{hs}" + r"$\delta$")
ax[1].legend(loc="lower right")
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
plt.savefig(f"figs/single/compare_mu_cc/ett_T_{int(hs*1000)}.pdf", dpi=300)
plt.show()