import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution
from comsol_solution import ComsolSolution

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
path = f"data/E1e4h{str(hh-int(hh))[2:]}/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------
outer = OuterSolution(r0, delta, E, nu, alpha)

# Load COMSOL solutions ----------------------------------------------
comsol = ComsolSolution(r0, delta, hh, N, E, nu, path)
theta = comsol.theta

# Plot solution(s) ------------------------------------------------------------
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi

# tension
fig, ax = plt.subplots()
ax.plot(theta, outer.T(theta), "-", label="Asymptotic")
# ax.plot(theta, comsol.T_a, ":", label="COMSOL (a)")
ax.plot(theta, comsol.T_b, ":", label="COMSOL (midpoint)")
# ax.plot(theta, comsol.T_c, ":", label="COMSOL (c)")
ax.plot(theta, comsol.T, "--", label="COMSOL (Simpson's)")
ax.set_ylabel(r"$T$")
ax.legend()
# add shared labels etc.
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
# plt.savefig("figs" + path[4:] + "T_of_theta.pdf", dpi=300)

plt.show()
