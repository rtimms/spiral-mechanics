import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution

# set style for paper
import matplotlib

matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Geometric parameters (dimensionless) ----------------------------------------
N = 10  # number of winds
r0 = 0.25  # inner radius
r1 = 1  # outer radius
delta = (r1 - r0) / N
N_plot = 5  # number of winds to plot
path = "data/spiral/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Plot tension ----------------------------------------------------------------
theta = np.linspace(0, 2 * pi * N, 120 * (N - 1))
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi
fig, ax = plt.subplots(2, 1, figsize=(6.4, 4))
linestyles = [":", "-.", "--", "-"]

# vary \hat{\alpha}\hat{M}, fix \hat{mu}/\hat{M}
mu = 1
nu = 1 / 3
lam = 2 * mu * nu / (1 - 2 * nu)
mu_over_M_hat_ref = 1 / (2 + 2 * 1)  # mu=1, lam=2
alpha_M_hats = [0.1, 1, 10]
for i, alpha_M_hat in enumerate(alpha_M_hats):
    alpha = alpha_M_hat / (3 * lam + 2 * mu)
    outer = OuterSolution(r0, delta, mu, lam, alpha)
    ax[0].plot(
        theta,
        outer.T(theta),
        linestyle=linestyles[i],
        label=r"$\hat{\alpha}\hat{M} = $" + f"{alpha_M_hat}",
    )
ax[0].set_title(r"$\hat{\mu}/\hat{M} = $" + f"{mu_over_M_hat_ref}")
ax[0].set_ylabel(r"$T$")
ax[0].legend(loc="lower right")

# vary \hat{\alpha}\hat{M}, fix \hat{mu}/\hat{M}
mu = 1
nu = 1 / 3
alpha_M_hat_ref = 1 * (3 * 2 + 2 * 1)  # alpha=1, mu=1, lam=2
mu_over_M_hats = [0.1, 1, 10]
for i, mu_over_M_hat in enumerate(mu_over_M_hats):
    lam = 1 / mu_over_M_hat - 2
    alpha = alpha_M_hat_ref / (3 * lam + 2 * mu)
    outer = OuterSolution(r0, delta, mu, lam, alpha)
    ax[1].plot(
        theta,
        outer.T(theta),
        linestyle=linestyles[i],
        label=r"$\hat{\mu}/\hat{M} = $" + f"{mu_over_M_hat}",
    )
ax[1].set_title(r"$\hat{\alpha}\hat{M} = $" + f"{alpha_M_hat_ref}")
ax[1].set_ylabel(r"$T$")
ax[1].legend(loc="lower right")

# add shared labels etc.
for ax in ax.reshape(-1):
    for w in winds:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: "{}$\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
    ax.set_xlabel(r"$\theta$")
plt.tight_layout()
plt.savefig("figs" + path[4:] + "T_vary_params.pdf", dpi=300)

plt.show()
