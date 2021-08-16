import numpy as np
from numpy import pi
import pandas as pd
import scipy.interpolate as interp
import matplotlib
import matplotlib.pyplot as plt
import os
from outer_solution import OuterSolution

# set style for paper
matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

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
hh = 0.01 * delta  # current collector thickness
N_plot = 9  # number of winds to plot
path = "data/single/mu1lam2/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the outer solution --------------------------------------------------
outer = OuterSolution(r0, delta, mu, lam, alpha)
# unpack
f1, f2, g1, g2 = outer.f1, outer.f2, outer.g1, outer.g2
u, v = outer.u, outer.v

# Load COMSOL solutions for f_i, g_i ------------------------------------------
# Note that we rescale the displacements, strains and stresses by alpha_scale
# since COMSOL didn't like having alpha=1.
alpha_scale = 0.1

# f1 = sigma_rr
comsol = pd.read_csv(path + "srr3.csv", comment="#", header=None).to_numpy()
f1_r_data = comsol[:, 0]
f1_data = comsol[:, 1] / alpha_scale
f1_interp = interp.interp1d(f1_r_data, f1_data, bounds_error=False)


def f1_comsol(theta):
    # In COMSOL we evaluate at r = r0+delta/2+delta*theta/2/pi
    r = r0 + delta / 2 + delta * theta / 2 / pi
    return f1_interp(r)


# f2 = u(R=theta/2/pi)/delta
comsol = pd.read_csv(path + "u1.csv", comment="#", header=None).to_numpy()
f2_r_data = comsol[:, 0]
f2_data = comsol[:, 1] / alpha_scale / delta
f2_interp = interp.interp1d(f2_r_data, f2_data, bounds_error=False)


def f2_comsol(theta):
    # In COMSOL we evaluate at r = r0+hh/2+delta*theta/2/pi
    r = r0 + hh / 2 + delta * theta / 2 / pi
    return f2_interp(r)


# g1 = sigma_rt
comsol = pd.read_csv(path + "srt3.csv", comment="#", header=None).to_numpy()
g1_r_data = comsol[:, 0]
g1_data = comsol[:, 1] / alpha_scale
g1_interp = interp.interp1d(g1_r_data, g1_data, bounds_error=False)


def g1_comsol(theta):
    # In COMSOL we evaluate at r = r0+delta/2+delta*theta/2/pi
    r = r0 + delta / 2 + delta * theta / 2 / pi
    return g1_interp(r)


# g2 = v(R=theta/2/pi)/delta
comsol = pd.read_csv(path + "v1.csv", comment="#", header=None).to_numpy()
g2_r_data = comsol[:, 0]
g2_data = comsol[:, 1] / alpha_scale / delta
g2_interp = interp.interp1d(g2_r_data, g2_data, bounds_error=False)


def g2_comsol(theta):
    # In COMSOL we evaluate at r = r0+hh/2+delta*theta/2/pi
    r = r0 + hh / 2 + delta * theta / 2 / pi
    return g2_interp(r)


# Plot at theta = 0 -----------------------------------------------------------

theta_f = 0
fig, ax = plt.subplots(2, 2)

# get the locations of the current collector boundaries
winds_m = [
    r0 - hh / 2 + delta * (2 * pi * n + theta_f) / 2 / pi for n in list(range(N_plot))
]
winds = [
    r0 + delta * (2 * pi * n + theta_f) / 2 / pi for n in list(range(N_plot))
]  # centre
winds_p = [
    r0 + hh / 2 + delta * (2 * pi * n + theta_f) / 2 / pi for n in list(range(N_plot))
]

# get r, theta values
nr = 5
r = np.zeros(nr * N_plot)
theta = np.zeros(nr * N_plot)
for N in list(range(N_plot)):
    rr = r0 + delta * (2 * pi * N + theta_f) / 2 / pi
    r[N * nr : N * nr + nr] = np.linspace(rr, rr + delta, nr)
    theta[N * nr : N * nr + nr] = np.ones(nr) * (2 * pi * N + theta_f)
# radial displacement
ax[0, 0].plot(r, u(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"u_0.csv", comment="#", header=None).to_numpy()
ax[0, 0].plot(
    comsol[:, 0], comsol[:, 1] / alpha_scale, "--", color="tab:orange", label="COMSOL"
)
ax[0, 0].set_ylabel(r"$u$")
# azimuthal displacement
ax[0, 1].plot(r, v(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"v_0.csv", comment="#", header=None).to_numpy()
ax[0, 1].plot(
    comsol[:, 0], comsol[:, 1] / alpha_scale, "--", color="tab:orange", label="COMSOL"
)
ax[0, 1].set_ylabel(r"$v$")
# normal stress
ax[1, 0].plot(r, f1(theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srr_0.csv", comment="#", header=None).to_numpy()
ax[1, 0].plot(
    comsol[:, 0],
    comsol[:, 1] / alpha_scale,
    "--",
    color="tab:orange",
    label="COMSOL",
)
ax[1, 0].set_ylabel(r"$\sigma_{rr}$")
# shear stress
ax[1, 1].plot(r, g1(theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srt_0.csv", comment="#", header=None).to_numpy()
ax[1, 1].plot(
    comsol[:, 0], comsol[:, 1] / alpha_scale, "--", color="tab:orange", label="COMSOL"
)
ax[1, 1].set_ylabel(r"$\sigma_{r\theta}$")
# add shared labels etc
ax[0, 0].legend(loc="lower right")
for ax in ax.reshape(-1):
    for w_m, w_p in zip(winds_m, winds_p):
        ax.axvline(x=w_m, linestyle=":", color="lightgrey")
        ax.axvline(x=w_p, linestyle=":", color="lightgrey")
    ax.set_xlim([r[0], r[-1]])
    ax.set_xlabel(r"$r$")
fig.suptitle(r"$\theta=0$")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("figs" + path[4:] + "slice_0.pdf", dpi=300)

# plot f_i(r), g_i(r)
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(r, f1(theta), "-", color="tab:blue", label="Asymptotic")
ax[0, 0].set_ylabel(r"$f_1$")
ax[0, 1].plot(r, f2(theta), "-", color="tab:blue", label="Asymptotic")
ax[0, 1].set_ylabel(r"$f_2$")
ax[1, 0].plot(r, g1(theta), "-", color="tab:blue", label="Asymptotic")
ax[1, 0].set_ylabel(r"$g_1$")
ax[1, 1].plot(r, g2(theta), "-", color="tab:blue", label="Asymptotic")
ax[1, 1].set_ylabel(r"$g_2$")
# add shared labels etc/
for ax in ax.reshape(-1):
    for w_m, w_p in zip(winds_m, winds_p):
        ax.axvline(x=w_m, linestyle=":", color="lightgrey")
        ax.axvline(x=w_p, linestyle=":", color="lightgrey")
    ax.set_xlim([r[0], r[-1]])
    ax.set_xlabel(r"$r$")
fig.suptitle(r"$\theta=0$")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("figs" + path[4:] + "fg_0.pdf", dpi=300)

# Plot at theta = pi ----------------------------------------------------------

theta_f = pi
fig, ax = plt.subplots(2, 2)

# get the locations of the current collector boundaries
winds_m = [
    r0 - hh / 2 + delta * (2 * pi * n + theta_f) / 2 / pi for n in list(range(N_plot))
]
winds = [
    r0 + delta * (2 * pi * n + theta_f) / 2 / pi for n in list(range(N_plot))
]  # centre
winds_p = [
    r0 + hh / 2 + delta * (2 * pi * n + theta_f) / 2 / pi for n in list(range(N_plot))
]

# get r, theta values
nr = 5
r = np.zeros(nr * N_plot)
theta = np.zeros(nr * N_plot)
for N in list(range(N_plot)):
    rr = r0 + delta * (2 * pi * N + theta_f) / 2 / pi
    r[N * nr : N * nr + nr] = np.linspace(rr, rr + delta, nr)
    theta[N * nr : N * nr + nr] = np.ones(nr) * (2 * pi * N + theta_f)

# radial displacement
ax[0, 0].plot(r, u(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"u_pi.csv", comment="#", header=None).to_numpy()
ax[0, 0].plot(
    comsol[:, 0], comsol[:, 1] / alpha_scale, "--", color="tab:orange", label="COMSOL"
)
ax[0, 0].set_ylabel(r"$u$")
# azimuthal displacement
ax[0, 1].plot(r, v(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"v_pi.csv", comment="#", header=None).to_numpy()
ax[0, 1].plot(
    comsol[:, 0], comsol[:, 1] / alpha_scale, "--", color="tab:orange", label="COMSOL"
)
ax[0, 1].set_ylabel(r"$v$")
# normal stress
ax[1, 0].plot(r, f1(theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srr_pi.csv", comment="#", header=None).to_numpy()
ax[1, 0].plot(
    comsol[:, 0],
    comsol[:, 1] / alpha_scale,
    "--",
    color="tab:orange",
    label="COMSOL",
)
ax[1, 0].set_ylabel(r"$\sigma_{rr}$")
# shear stress
ax[1, 1].plot(r, g1(theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srt_pi.csv", comment="#", header=None).to_numpy()
ax[1, 1].plot(
    comsol[:, 0], comsol[:, 1] / alpha_scale, "--", color="tab:orange", label="COMSOL"
)
ax[1, 1].set_ylabel(r"$\sigma_{r\theta}$")
# add shared labels etc
ax[0, 0].legend(loc="lower right")
for ax in ax.reshape(-1):
    for w_m, w_p in zip(winds_m, winds_p):
        ax.axvline(x=w_m, linestyle=":", color="lightgrey")
        ax.axvline(x=w_p, linestyle=":", color="lightgrey")
    ax.set_xlim([r[0], r[-1]])
    ax.set_xlabel(r"$r$")
fig.suptitle(r"$\theta=\pi$")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("figs" + path[4:] + "slice_pi.pdf", dpi=300)

# plot f_i(r), g_i(r)
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(r, f1(theta), "-", color="tab:blue", label="Asymptotic")
ax[0, 0].set_ylabel(r"$f_1$")
ax[0, 1].plot(r, f2(theta), "-", color="tab:blue", label="Asymptotic")
ax[0, 1].set_ylabel(r"$f_2$")
ax[1, 0].plot(r, g1(theta), "-", color="tab:blue", label="Asymptotic")
ax[1, 0].set_ylabel(r"$g_1$")
ax[1, 1].plot(r, g2(theta), "-", color="tab:blue", label="Asymptotic")
ax[1, 1].set_ylabel(r"$g_2$")
# add shared labels etc/
for ax in ax.reshape(-1):
    for w_m, w_p in zip(winds_m, winds_p):
        ax.axvline(x=w_m, linestyle=":", color="lightgrey")
        ax.axvline(x=w_p, linestyle=":", color="lightgrey")
    ax.set_xlim([r[0], r[-1]])
    ax.set_xlabel(r"$r$")
fig.suptitle(r"$\theta=0$")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("figs" + path[4:] + "fg_pi.pdf", dpi=300)

plt.show()
