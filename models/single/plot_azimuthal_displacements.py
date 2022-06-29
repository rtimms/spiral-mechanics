import numpy as np
from numpy import pi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from composite_solution import OuterSolution
from comsol_solution import ComsolSolution, ComsolInnerSolution

# set style for paper
matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Parameters (dimensionless) --------------------------------------------------
alpha = 1  # expansion coefficient
mu = 1  # shear modulus
nu = 1 / 3  # Poisson ratio
lam = 2 * mu * nu / (1 - 2 * nu)  # 1st Lame parameter
omega = np.sqrt(mu / (lam + 2 * mu))
c = alpha * (2 * lam + mu) * omega
N = 10  # number of winds
r0 = 0.25  # inner radius
r1 = 1  # outer radius
delta = (r1 - r0) / N
hh = 0.01 * delta  # current collector thickness
N_BL = 5  # number of slabs in inner solution
N_plot = N_BL - 1  # number of winds to plot
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs/comparisons/")
except FileExistsError:
    pass

# displacements at r = r0 + delta*theta/2/pi, 0 < theta < 8*pi
fig, ax = plt.subplots(2, 1, figsize=(6.4, 4), sharex=True)

# Loop over alpha_cc and compare results --------------------------------------
alpha_ccs = [0, 0.5]
pathnames = ["a1al0", "a1al05"]

i = 0
for alpha_cc, pathname in zip(alpha_ccs, pathnames):
    # Compute the outer solution
    outer = OuterSolution(r0, delta, mu, lam, alpha, alpha_cc)
    v = outer.v

    # Load COMSOL data
    # inner (slab solution)
    inner = ComsolInnerSolution(f"data/inner_{pathname}/")
    t_data = inner.v_t_data
    v_data = inner.v_data

    # outer (full simulation)
    comsol = ComsolSolution(
        r0, delta, hh, N, mu, lam, alpha, alpha_cc, f"data/{pathname}/"
    )

    # Plots
    # plot leading (outer) solutions
    theta = comsol.theta
    r = r0 + delta * theta / 2 / pi
    ax[i].plot(
        theta,
        v(r, theta),
        linestyle=":",
        color="black",
        label="Surface solution",
    )
    # plot composite solution solutions
    for n in range(N_BL):
        idx1 = int(n * 100 / N_BL + 10)  # midpoint
        idx2 = int(n * 100 / N_BL)  # inner edge
        if n == 0:
            # 0 < theta < inf
            Theta = t_data[0, 50:]
            v_tilde = v_data[idx2, 50:]
        else:
            # -inf < theta < inf
            Theta = t_data[0, :]
            v_tilde = v_data[idx2, :]
        theta = delta * Theta / r0 + 2 * n * pi
        r = r0 + delta * theta / 2 / pi
        # u(R=theta/2/pi)
        ax[i].plot(
            theta,
            v(r, theta) + delta * c * v_tilde,
            linestyle="-",
            color="tab:blue",
            label="Surface-end composite solution" if n == 0 else "",
        )
    # plot COMSOL solutions
    theta = comsol.theta
    # v = delta*g2
    ax[i].plot(
        theta, delta * comsol.g2, linestyle="--", color="tab:orange", label="COMSOL"
    )

    # increment counter
    i += 1

ax[0].set_ylabel(r"$v$")
ax[1].set_ylabel(r"$v$")
ax[1].set_xlabel(r"$\theta$")
fig.subplots_adjust(left=0.1, bottom=0.3, right=0.98, top=0.98, wspace=0.33, hspace=0.4)
ax[1].legend(
    loc="upper center",
    bbox_to_anchor=(0.45, -0.5),
    borderaxespad=0.0,
    ncol=3,
)
ax[0].text(
    0.85,
    0.05,
    r"$\alpha_{\small\textrm{s}}=0$",
    verticalalignment="bottom",
    horizontalalignment="left",
    transform=ax[0].transAxes,
    fontsize=14,
)
ax[1].text(
    0.85,
    0.05,
    r"$\alpha_{\small\textrm{s}}=0.5$",
    verticalalignment="bottom",
    horizontalalignment="left",
    transform=ax[1].transAxes,
    fontsize=14,
)
for ax in ax.reshape(-1):
    # plot dashed line every 2*pi
    winds = [2 * pi * n for n in list(range(N_plot))]
    for w in winds:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    # add labels etc.
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
    ax.set_ylim([-0.3, 0.05])

plt.savefig("figs/comparisons/compare_v.pdf", dpi=300)

plt.show()
