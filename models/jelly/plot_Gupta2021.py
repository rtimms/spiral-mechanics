import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution
from comsol_jelly_solution import ComsolSolution
from shared_plots import plot_fg, plot_tension

# set style for paper
# import matplotlib
# matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)


# Dimensional parameters ------------------------------------------------------
# Gupta and Gudmundson (2021)
h_n = 66 * 1e-6
h_s = 20 * 1e-6
h_p = 69 * 1e-6
h = 2 * (h_n + h_s + h_p)
L = 9 * 1e-3
N = 20
L_0 = L - h * N
E_n = 2.5 * 1e9
E_s = 0.4 * 1e9
E_p = 2 * 1e9
nu_n = 0.3
nu_s = 0.01
nu_p = 0.3

# compute shear moduli
mu_n = E_n / 2 / (1 + nu_n)
mu_s = E_s / 2 / (1 + nu_s)
mu_p = E_p / 2 / (1 + nu_p)

# estimates of expansion due to lithiation during charge from Willenberg (2020)
alpha_n = 0.1
alpha_p = -0.02


# Dimensionless parameters ----------------------------------------------------
class Parameters:
    "Empty class which will contain the parameters as attributes"
    pass


params = Parameters()

# reference values
mu_ref = mu_n
alpha_ref = alpha_n

#  geometry
params.r0 = L_0 / L
params.r1 = 1
params.N = N
params.delta = (params.r1 - params.r0) / params.N
params.hh = 0.005 * params.delta
params.l_p = h_p / h
params.l_s = h_s / h
params.l_n = h_n / h

# positive electrode material properties
params.alpha_p = alpha_p / alpha_ref  # expansion coefficient
params.mu_p = mu_p / mu_ref  # shear modulus
params.nu_p = nu_p  # Poisson ratio
params.lam_p = (
    2 * params.mu_p * params.nu_p / (1 - 2 * params.nu_p)
)  # 1st Lame parameter

# separator electrode material properties
params.alpha_s = 0  # expansion coefficient
params.mu_s = mu_s / mu_ref  # shear modulus
params.nu_s = nu_s  # Poisson ratio
params.lam_s = (
    2 * params.mu_s * params.nu_s / (1 - 2 * params.nu_s)
)  # 1st Lame parameter

# negative electrode material properties
params.alpha_n = alpha_n / alpha_ref  # expansion coefficient
params.mu_n = mu_n / mu_ref  # shear modulus
params.nu_n = nu_n  # Poisson ratio
params.lam_n = (
    2 * params.mu_n * params.nu_n / (1 - 2 * params.nu_n)
)  # 1st Lame parameter

N_plot = 6  # number of winds to plot
path = "data/jelly/Gupta2021/"  # path to data
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

# Plot solution(s) ------------------------------------------------------------

# f_i, g_i
plot_fg(outer, comsol, N_plot, path)

# tension
plot_tension(outer, comsol, N_plot, path)

# Compute and plot strain in positive current collector -----------------------
h_cp = 10 * 1e-6  # positive cc thickness [m]
E_cp = 70 * 1e9  # positive cc Young's modulus [Pa]

theta = np.linspace(0, 2 * pi * N, 60 * (N - 1))
T_p_dim = outer.Tp(theta) * mu_ref * alpha_ref * L
sigma_p_dim = T_p_dim / h_cp
eps_p_dim = sigma_p_dim / E_cp

fig, ax = plt.subplots(2, 1, figsize=(6.4, 4))
ax[0].plot(theta, outer.Tp(theta), "-")
ax[0].set_ylabel(r"$T_+$")
ax[1].plot(theta, eps_p_dim, "-")
ax[1].set_ylabel(r"$\varepsilon_+$")
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
plt.savefig("figs" + path[4:] + "eps_of_theta_jelly.pdf", dpi=300)

plt.show()
