import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution
import pandas as pd

# set style for paper
import matplotlib

matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)


# Dimensional parameters ------------------------------------------------------
# Gupta and Gudmundson (2021) with 26 winds
h_n = 66 * 1e-6
h_s = 20 * 1e-6
h_p = 69 * 1e-6
h = 2 * (h_n + h_s + h_p)
L = 9 * 1e-3
N = 26
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
mu_ref = (h_n + h_s + h_p) / (h_n / mu_n + h_s / mu_s + h_p / mu_p)
alpha_ref = (h_n * alpha_n + h_p * alpha_p) / (h_n + h_s + h_p)
# mu_ref = mu_n
# alpha_ref = alpha_n

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

N_plot = 4  # number of winds to plot
path = "data/jelly/Gupta18650/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------
outer = OuterSolution(params)

# Plot slice through theta=theta_f --------------------------------------------

theta_f = pi
r0 = params.r0
delta = params.delta
l_p, l_s, l_n = params.l_p, params.l_s, params.l_n
mu_p, mu_s, mu_n = params.mu_p, params.mu_s, params.mu_n
lam_p, lam_s, lam_n = params.lam_p, params.lam_s, params.lam_n
alpha_p, alpha_s, alpha_n = params.alpha_p, params.alpha_s, params.alpha_n

fig, ax = plt.subplots(1, 3, figsize=(6.4, 2))

winds_p = [r0 + delta * (2 * pi * n + theta_f) / 2 / pi for n in list(range(N_plot))]
winds_n = [
    r0 + delta / 2 + delta * (2 * pi * n + theta_f) / 2 / pi
    for n in list(range(N_plot))
]

nr = 3
for N in list(range(N_plot)):
    ri = r0 + delta * (2 * pi * N + theta_f) / 2 / pi
    theta = 2 * pi * N + theta_f

    sigma_rr = outer.f1(theta)
    sigma_rt = outer.g1(theta)
    sigma_tt_n = -2 * mu_n * alpha_n * (3 * lam_n + 2 * mu_n) / (
        lam_n + 2 * mu_n
    ) + lam_n / (lam_n + 2 * mu_n) * outer.f1(theta)
    sigma_tt_s = -2 * mu_s * alpha_s * (3 * lam_s + 2 * mu_s) / (
        lam_s + 2 * mu_s
    ) + lam_s / (lam_s + 2 * mu_s) * outer.f1(theta)
    sigma_tt_p = -2 * mu_p * alpha_p * (3 * lam_p + 2 * mu_p) / (
        lam_p + 2 * mu_p
    ) + lam_p / (lam_p + 2 * mu_p) * outer.f1(theta)

    # positive electrode
    r_p1 = np.linspace(ri, ri + l_p * delta, nr)
    ax[0].plot(
        r_p1,
        sigma_rr * np.ones(nr),
        "-",
        color="tab:blue",
        label="Asymptotic" if N == 0 else "",
    )
    ax[1].plot(
        r_p1,
        sigma_tt_p * np.ones(nr),
        "-",
        color="tab:blue",
        label="Asymptotic" if N == 0 else "",
    )
    ax[2].plot(
        r_p1,
        sigma_rt * np.ones(nr),
        "-",
        color="tab:blue",
        label="Asymptotic" if N == 0 else "",
    )
    # separator
    r_s1 = np.linspace(ri + l_p * delta, ri + (l_p + l_s) * delta, nr)
    ax[0].plot(r_s1, sigma_rr * np.ones(nr), "-", color="tab:blue")
    ax[1].plot(r_s1, sigma_tt_s * np.ones(nr), "-", color="tab:blue")
    ax[2].plot(r_s1, sigma_rt * np.ones(nr), "-", color="tab:blue")
    # negative electrode
    r_n1 = np.linspace(ri + (l_p + l_s) * delta, ri + (l_p + l_s + l_n) * delta, nr)
    ax[0].plot(r_n1, sigma_rr * np.ones(nr), "-", color="tab:blue")
    ax[1].plot(r_n1, sigma_tt_n * np.ones(nr), "-", color="tab:blue")
    ax[2].plot(r_n1, sigma_rt * np.ones(nr), "-", color="tab:blue")
    # negative electrode
    r_n2 = np.linspace(
        ri + (l_p + l_s + l_n) * delta, ri + (l_p + l_s + 2 * l_n) * delta, nr
    )
    ax[0].plot(r_n2, sigma_rr * np.ones(nr), "-", color="tab:blue")
    ax[1].plot(r_n2, sigma_tt_n * np.ones(nr), "-", color="tab:blue")
    ax[2].plot(r_n2, sigma_rt * np.ones(nr), "-", color="tab:blue")
    # separator
    r_s2 = np.linspace(
        ri + (l_p + l_s + 2 * l_n) * delta, ri + (l_p + 2 * l_s + 2 * l_n) * delta, nr
    )
    ax[0].plot(r_s2, sigma_rr * np.ones(nr), "-", color="tab:blue")
    ax[1].plot(r_s2, sigma_tt_s * np.ones(nr), "-", color="tab:blue")
    ax[2].plot(r_s2, sigma_rt * np.ones(nr), "-", color="tab:blue")
    # positive electrode
    r_p2 = np.linspace(
        ri + (l_p + 2 * l_s + 2 * l_n) * delta,
        ri + (2 * l_p + 2 * l_s + 2 * l_n) * delta,
        nr,
    )
    ax[0].plot(r_p2, sigma_rr * np.ones(nr), "-", color="tab:blue")
    ax[1].plot(r_p2, sigma_tt_p * np.ones(nr), "-", color="tab:blue")
    ax[2].plot(r_p2, sigma_rt * np.ones(nr), "-", color="tab:blue")


def remove_cc(r, s):
    idx = []
    for winds in [winds_p, winds_n]:
        for w in winds:
            i = np.where(np.abs(r - w) < params.hh * 1.05)[0]
            idx.extend(list(i))

    return np.delete(r, idx), np.delete(s, idx)


comsol = pd.read_csv(path + f"srr_tpi.csv", comment="#", header=None).to_numpy()
r, s_rr = remove_cc(comsol[:, 0], comsol[:, 1] / alpha_ref)
ax[0].plot(r, s_rr, "--", color="tab:orange", label="COMSOL")

comsol = pd.read_csv(path + f"stt_tpi.csv", comment="#", header=None).to_numpy()
r, s_tt = remove_cc(comsol[:, 0], comsol[:, 1] / alpha_ref)
ax[1].plot(r, s_tt, "--", color="tab:orange", label="COMSOL")

comsol = pd.read_csv(path + f"srt_tpi.csv", comment="#", header=None).to_numpy()
r, s_rt = remove_cc(comsol[:, 0], comsol[:, 1] / alpha_ref)
ax[2].plot(r, s_rt, "--", color="tab:orange", label="COMSOL")

ax[0].set_ylim([-6, -3])
ax[1].set_ylim([-25, 5])
ax[2].set_ylim([-0.75, 0.75])
ax[0].set_ylabel(r"$\sigma_{rr}$")
ax[1].set_ylabel(r"$\sigma_{\theta\theta}$")
ax[2].set_ylabel(r"$\sigma_{r\theta}$")
# add shared labels etc.
fig.subplots_adjust(left=0.1, bottom=0.3, right=0.98, top=0.98, wspace=0.4, hspace=0.4)
ax[1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    borderaxespad=0.0,
    ncol=2,
)
for ax in ax.reshape(-1):
    for w in winds_p:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim(
        [
            r0 + delta * theta_f / 2 / pi,
            r0 + delta * (2 * pi * N_plot + theta_f) / 2 / pi,
        ]
    )
    ax.set_xlabel(r"$r$")
plt.savefig("figs" + path[4:] + "stress_of_r.pdf", dpi=300)
plt.show()