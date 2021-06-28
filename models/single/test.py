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
omega = np.sqrt(mu / (lam + 2 * mu))
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

fig, ax = plt.subplots(1, 2)
ax[0].plot(theta, outer.dg1dt(theta), "-")
ax[0].plot(theta, np.gradient(outer.g1(theta), theta), "o")
ax[1].plot(theta, outer.dg2dt(theta), "-")
ax[1].plot(theta, np.gradient(outer.g2(theta), theta), "o")
plt.show()