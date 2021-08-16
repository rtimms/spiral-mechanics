import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution

# set style for paper
# import matplotlib
# matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)


# Dimensional parameters ------------------------------------------------------
ref = "Valentin2017"

if ref == "Valentin2017":
    # dimensional parameters Valentin et al. (2017)
    h_n = 0.09 * 1e-3
    h_s = 0.01 * 1e-3
    h_p = 0.13 * 1e-3
    h = 2 * (h_n + h_s + h_p)
    L = 8.88 * 1e-3
    N = 16
    L_0 = L - h * N
    E_n = 12 * 1e9
    E_s = 2.5 * 1e9
    E_p = 10 * 1e9
    nu_n = 0.3
    nu_s = 0.49
    nu_p = 0.49
elif ref == "Nadimpalli2015":
    # dimensional parameters Nadimpalli et al. (2015)
    h_n = 35 * 1e-6
    h_s = 20 * 1e-6
    h_p = 35 * 1e-6
    h = 2 * (h_n + h_s + h_p)
    L = 9.3 * 1e-3
    N = 38
    L_0 = L - h * N
    E_n = 6.9 * 1e9
    E_s = 0.1 * 1e9
    E_p = 40 * 1e9
    nu_n = 0.3
    nu_s = 0.3
    nu_p = 0.2

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
E_ref = E_n
mu_ref = mu_n
alpha_ref = alpha_n

#  geometry
params.r0 = L_0 / L
params.r1 = 1
params.N = N
params.delta = (params.r1 - params.r0) / params.N
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
path = "data/jelly/asymptotic/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------
outer = OuterSolution(params)
print(f"S_1 = {outer.param.S_1}")
print(f"S_2 = {outer.param.S_2}")
print(f"S_3 = {outer.param.S_3}")
print(f"omega = {outer.param.omega}")

# Plot solution(s) ------------------------------------------------------------
theta = np.linspace(0, 2 * pi * params.N, 60 * (params.N - 1))
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi

# f_i, g_i
fig, ax = plt.subplots(2, 3, figsize=(6.4, 4))
ax[0, 0].plot(theta, outer.f1(theta), "-", label="Asymptotic")
ax[0, 0].set_ylabel(r"$f_{{1,3}}$")
ax[0, 1].plot(theta, outer.f2(theta), "-", label="Asymptotic")
ax[0, 1].set_ylabel(r"$f_2$")
ax[0, 2].plot(theta, outer.f4(theta), "-", label="Asymptotic")
ax[0, 2].set_ylabel(r"$f_4$")
ax[1, 0].plot(theta, outer.g1(theta), "-", label="Asymptotic")
ax[1, 0].set_ylabel(r"$g_{{1,3}}$")
ax[1, 1].plot(theta, outer.g2(theta), "-", label="Asymptotic")
ax[1, 1].set_ylabel(r"$g_2$")
ax[1, 2].plot(theta, outer.g4(theta), "-", label="Asymptotic")
ax[1, 2].set_ylabel(r"$g_4$")
# add shared labels etc.
fig.subplots_adjust(left=0.1, bottom=0.25, right=0.98, top=0.98, wspace=0.4, hspace=0.4)
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

# tension
fig, ax = plt.subplots(1, 2, figsize=(6.4, 4))
ax[0].plot(theta, outer.Tp(theta), "-", label="Asymptotic")
ax[0].set_ylabel(r"$T_p$")
ax[1].plot(theta, outer.Tn(theta), "-", label="Asymptotic")
ax[1].set_ylabel(r"$T_n$")
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
plt.savefig("figs" + path[4:] + "T_of_theta.pdf", dpi=300)

plt.show()