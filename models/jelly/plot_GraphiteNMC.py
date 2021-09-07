import matplotlib.pyplot as plt
import os
from outer_solution import OuterSolution
from comsol_jelly_solution import ComsolSolution
from shared_plots import plot_fg, plot_tension

# set style for paper
# import matplotlib
# matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Dimensional material properties ---------------------------------------------
E_n = 6.9 * 1e9
E_s = 0.1 * 1e9
E_p = 40 * 1e9
nu_n = 0.3
nu_s = 0.3
nu_p = 0.2
mu_n = E_n / 2 / (1 + nu_n)
mu_s = E_s / 2 / (1 + nu_s)
mu_p = E_p / 2 / (1 + nu_p)

# estimates of expansion due to lithiation during charge from Willenberg (2020)
alpha_n = 0.1
alpha_p = -0.02

# reference values
mu_ref = mu_n
alpha_ref = alpha_n


# Dimensionless parameters ----------------------------------------------------
class Parameters:
    "Empty class which will contain the parameters as attributes"
    pass


params = Parameters()

#  geometry
params.r0 = 0.25
params.r1 = 1
params.N = 10
params.delta = (params.r1 - params.r0) / params.N
params.hh = 0.005 * params.delta
params.l_p = 0.4 / 2
params.l_s = 0.2 / 2
params.l_n = 0.4 / 2

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
path = "data/jelly/GraphiteNMC/"  # path to data
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

plt.show()