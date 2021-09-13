import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.pyplot import cm
from outer_solution import OuterSolution
from comsol_jelly_solution import ComsolSolution


# Parameters ------------------------------------------------------------------
class Parameters:
    "Empty class which will contain the parameters as attributes"
    pass


params = Parameters()

#  geometry
params.r1 = 1
params.delta = 0.05
params.hh = 0.005 * params.delta
params.l_p = 0.4 / 2
params.l_s = 0.2 / 2
params.l_n = 0.4 / 2

# positive electrode material properties
params.alpha_p = 1  # expansion coefficient
params.mu_p = 1  # shear modulus
params.nu_p = 1 / 3  # Poisson ratio
params.lam_p = (
    2 * params.mu_p * params.nu_p / (1 - 2 * params.nu_p)
)  # 1st Lame parameter

# separator electrode material properties
params.alpha_s = 1  # expansion coefficient
params.mu_s = 1  # shear modulus
params.nu_s = 1 / 3  # Poisson ratio
params.lam_s = (
    2 * params.mu_s * params.nu_s / (1 - 2 * params.nu_s)
)  # 1st Lame parameter

# negative electrode material properties
params.alpha_n = 1  # expansion coefficient
params.mu_n = 1  # shear modulus
params.nu_n = 1 / 3  # Poisson ratio
params.lam_n = (
    2 * params.mu_n * params.nu_n / (1 - 2 * params.nu_n)
)  # 1st Lame parameter

N_plot = 4  # number of winds to plot
alpha_scale = 0.1

# Plot tension for different N ------------------------------------------------

Ns = [15, 17, 19]

fig, ax = plt.subplots(3, 1, figsize=(6.4, 4))
color = iter(cm.Paired(np.linspace(0, 1, len(Ns))))
for N in Ns:
    c = next(color)

    # update params
    params.N = N
    params.r0 = 1 - params.delta * N

    # load solution
    outer = OuterSolution(params)
    comsol = ComsolSolution(
        params,
        alpha_scale,
        f"data/jelly/N{int(N)}/",  # path to data
    )
    theta = comsol.theta

    # plot tension
    ax[0].plot(theta, outer.Tp(theta), "-", c=c, label=f"r0={round(params.r0,2)}")
    ax[0].plot(theta, comsol.Tp, "--", c=c)
    ax[1].plot(
        theta, outer.Tn(theta), "-", c=c, label="Asymptotic" if N == Ns[0] else ""
    )
    ax[1].plot(theta, comsol.Tn, "--", c=c, label="COMSOL" if N == Ns[0] else "")
    ax[2].plot(theta, outer.Tp(theta) + outer.Tn(theta), "-", c=c)
    ax[2].plot(theta, comsol.Tn + comsol.Tp, "--", c=c)

# add shared labels etc.
ax[0].set_title(r"$\delta=$" + f"{params.delta}")
ax[0].set_ylabel(r"$T_+$")
ax[1].set_ylabel(r"$T_-$")
ax[2].set_ylabel(r"$T_-+T_+$")
ax[0].legend(loc="lower right")
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

plt.show()
