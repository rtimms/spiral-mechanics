import numpy as np
from numpy import pi, exp
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

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
path = "data/h005/"  # path to data


# Compute the boundary layer displacements and stresses -----------------------

# constants
A = alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * exp(2 * pi * omega)
B = 0
C = A / (1 - exp(2 * pi * omega))
D = 0


# functions of theta
def f1(theta):
    return -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + A * exp(
        -omega * (theta + 2 * pi)
    )


def f2(theta):
    return B + C * exp(-omega * theta)


def g1(theta):
    return (lam + 2 * mu) / mu * omega * A * exp(-omega * (theta + 2 * pi))


def g2(theta):
    return D + C / omega * exp(-omega * theta)


# radial displacement
def u(r, theta):
    R = (r - r0) / delta
    return (
        alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * (R - theta / 2 / pi)
        + f1(theta) * (R - theta / 2 / pi)
        + f2(theta)
    )


# azimuthal displacement
def v(r, theta):
    R = (r - r0) / delta
    return g1(theta) * (R - theta / 2 / pi) + g2(theta)


# Load COMSOL solutions for f_i, g_i ------------------------------------------

# f1 = sigma_rr / (lambda+2*mu)
comsol = pd.read_csv(path + "srr3.csv", comment="#", header=None).to_numpy()
f1_r_data = comsol[:, 0]
f1_data = comsol[:, 1] / (lam + 2 * mu)
f1_interp = interp.interp1d(f1_r_data, f1_data, bounds_error=False)


def f1_comsol(theta):
    # In COMSOL we evaluate at r = r0+delta/2+delta*theta/2/pi
    r = r0 + delta / 2 + delta * theta / 2 / pi
    return f1_interp(r)


# f2 = u(R=theta/2/pi)
comsol = pd.read_csv(path + "u1.csv", comment="#", header=None).to_numpy()
f2_r_data = comsol[:, 0]
f2_data = comsol[:, 1]
f2_interp = interp.interp1d(f2_r_data, f2_data, bounds_error=False)


def f2_comsol(theta):
    # In COMSOL we evaluate at r = r0+hh/2+delta*theta/2/pi
    r = r0 + hh / 2 + delta * theta / 2 / pi
    return f2_interp(r)


# g1 = sigma_rt/mu
comsol = pd.read_csv(path + "srt3.csv", comment="#", header=None).to_numpy()
g1_r_data = comsol[:, 0]
g1_data = comsol[:, 1] / mu
g1_interp = interp.interp1d(g1_r_data, g1_data, bounds_error=False)


def g1_comsol(theta):
    # In COMSOL we evaluate at r = r0+delta/2+delta*theta/2/pi
    r = r0 + delta / 2 + delta * theta / 2 / pi
    return g1_interp(r)


# g2 = v(R=theta/2/pi)
comsol = pd.read_csv(path + "v1.csv", comment="#", header=None).to_numpy()
g2_r_data = comsol[:, 0]
g2_data = comsol[:, 1]
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
ax[0, 0].plot(comsol[:, 0], comsol[:, 1], "--", color="tab:orange", label="COMSOL")
ax[0, 0].set_ylabel(r"$u$")
ax[0, 0].legend()
# azimuthal displacement
ax[0, 1].plot(r, v(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"v_0.csv", comment="#", header=None).to_numpy()
ax[0, 1].plot(comsol[:, 0], comsol[:, 1], "--", color="tab:orange", label="COMSOL")
ax[0, 1].set_ylabel(r"$v$")
# normal stress
ax[1, 0].plot(r, f1(theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srr_0.csv", comment="#", header=None).to_numpy()
ax[1, 0].plot(
    comsol[:, 0],
    comsol[:, 1] / (lam + 2 * mu),
    "--",
    color="tab:orange",
    label="COMSOL",
)
ax[1, 0].set_ylabel(r"$f_1=\sigma_{rr}/(\lambda+2\mu)$")
# shear stress
ax[1, 1].plot(r, g1(theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srt_0.csv", comment="#", header=None).to_numpy()
ax[1, 1].plot(comsol[:, 0], comsol[:, 1] / mu, "--", color="tab:orange", label="COMSOL")
ax[1, 1].set_ylabel(r"$g_1 = \sigma_{r\theta}/\mu$")

# add shared lables etc/
for ax in ax.reshape(-1):
    for w_m, w_p in zip(winds_m, winds_p):
        ax.axvline(x=w_m, linestyle=":", color="lightgrey")
        ax.axvline(x=w_p, linestyle=":", color="lightgrey")
    ax.set_xlim([r[0], r[-1]])
    ax.set_xlabel(r"$r$")
fig.suptitle(r"$\theta=0$")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("figs/slice_0.pdf", dpi=300)

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
# add shared lables etc/
for ax in ax.reshape(-1):
    for w_m, w_p in zip(winds_m, winds_p):
        ax.axvline(x=w_m, linestyle=":", color="lightgrey")
        ax.axvline(x=w_p, linestyle=":", color="lightgrey")
    ax.set_xlim([r[0], r[-1]])
    ax.set_xlabel(r"$r$")
fig.suptitle(r"$\theta=0$")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("figs/fg_0.pdf", dpi=300)

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
ax[0, 0].plot(comsol[:, 0], comsol[:, 1], "--", color="tab:orange", label="COMSOL")
ax[0, 0].set_ylabel(r"$u$")
ax[0, 0].legend()
# azimuthal displacement
ax[0, 1].plot(r, v(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"v_pi.csv", comment="#", header=None).to_numpy()
ax[0, 1].plot(comsol[:, 0], comsol[:, 1], "--", color="tab:orange", label="COMSOL")
ax[0, 1].set_ylabel(r"$v$")
# normal stress
ax[1, 0].plot(r, f1(theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srr_pi.csv", comment="#", header=None).to_numpy()
ax[1, 0].plot(
    comsol[:, 0],
    comsol[:, 1] / (lam + 2 * mu),
    "--",
    color="tab:orange",
    label="COMSOL",
)
ax[1, 0].set_ylabel(r"$f_1=\sigma_{rr}/(\lambda+2\mu)$")
# shear stress
ax[1, 1].plot(r, g1(theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srt_pi.csv", comment="#", header=None).to_numpy()
ax[1, 1].plot(comsol[:, 0], comsol[:, 1] / mu, "--", color="tab:orange", label="COMSOL")
ax[1, 1].set_ylabel(r"$g_1 = \sigma_{r\theta}/\mu$")

# add shared lables etc/
for ax in ax.reshape(-1):
    for w_m, w_p in zip(winds_m, winds_p):
        ax.axvline(x=w_m, linestyle=":", color="lightgrey")
        ax.axvline(x=w_p, linestyle=":", color="lightgrey")
    ax.set_xlim([r[0], r[-1]])
    ax.set_xlabel(r"$r$")
fig.suptitle(r"$\theta=\pi$")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("figs/slice_pi.pdf", dpi=300)

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
# add shared lables etc/
for ax in ax.reshape(-1):
    for w_m, w_p in zip(winds_m, winds_p):
        ax.axvline(x=w_m, linestyle=":", color="lightgrey")
        ax.axvline(x=w_p, linestyle=":", color="lightgrey")
    ax.set_xlim([r[0], r[-1]])
    ax.set_xlabel(r"$r$")
fig.suptitle(r"$\theta=0$")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("figs/fg_pi.pdf", dpi=300)


plt.show()
