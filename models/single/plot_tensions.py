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

# tension, 0 < theta < 8*pi
fig, ax = plt.subplots(1, 1, figsize=(6.4, 3))

# Loop over alpha_cc and compare results --------------------------------------
alpha_ccs = [0, 0.5]
pathnames = ["a1al0", "a1al05"]
labels = [r"($\alpha_{\small\textrm{s}}=0)$", r"($\alpha_{\small\textrm{s}}=0.5)$"]

i = 0
for alpha_cc, pathname in zip(alpha_ccs, pathnames):
    # Compute the outer solution
    outer = OuterSolution(r0, delta, mu, lam, alpha, alpha_cc)

    # outer (full simulation)
    comsol = ComsolSolution(
        r0, delta, hh, N, mu, lam, alpha, alpha_cc, f"data/{pathname}/"
    )
    theta = comsol.theta

    # Plots
    ax.plot(theta, outer.T(theta), "-", label="Surface solution " + labels[i])
    ax.plot(theta, comsol.T, "--", label="COMSOL " + labels[i])

    # increment counter
    i += 1

ax.annotate(
    r"$\alpha_{\small\textrm{s}}=0.5$",
    xy=(5.5, -0.76),
    xytext=(2, -0.2),
    arrowprops=dict(facecolor="black", width=0.5, headwidth=6, headlength=10),
)
ax.annotate(
    r"$\alpha_{\small\textrm{s}}=0$",
    xy=(7, -1.6),
    xytext=(9, -1.4),
    arrowprops=dict(facecolor="black", width=0.5, headwidth=6, headlength=10),
)

ax.set_ylabel(r"$T$")
fig.subplots_adjust(left=0.1, bottom=0.5, right=0.98, top=0.98, wspace=0.33, hspace=0.4)
handles, labels = ax.get_legend_handles_labels()
order = [0, 2, 1, 3]
ax.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.45),
    borderaxespad=0.0,
    ncol=2,
)
# add shared labels etc.
winds = [2 * pi * n for n in list(range(N_plot))]
for w in winds:
    ax.axvline(x=w, linestyle=":", color="lightgrey")
ax.xaxis.set_major_formatter(
    FuncFormatter(
        lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
    )
)
ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
ax.set_xlim([0, N_plot * 2 * pi])
ax.set_ylim([-2, 0.2])
ax.set_xlabel(r"$\theta$")
plt.savefig("figs/comparisons/compare_T.pdf", dpi=300)

plt.show()
