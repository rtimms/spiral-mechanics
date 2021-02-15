import numpy as np
from numpy import pi, exp
import pandas as pd
import scipy.interpolate as interp
import itertools
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
path = "data/"  # path to data


# Compute the boundary layer solution -----------------------------------------

# constants
A = -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * exp(2 * pi * omega)
B = 0
C = A / (1 - exp(2 * pi * omega))
D = 0


# functions of theta
def f1(theta):
    return -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) - A * exp(
        -omega * (theta + 2 * pi)
    )


def f2(theta):
    return B + C * exp(-omega * theta)


def g1(theta):
    return -(lam + 2 * mu) / mu * omega * A * exp(-omega * (theta + 2 * pi))


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


# Plot f_i, g_i ---------------------------------------------------------------
fig, ax = plt.subplots(2, 2)

# 0 < theta < 2*N_plot*pi
theta = np.linspace(0, 2 * pi * N_plot, 60 * N_plot)

# f1
ax[0, 0].plot(theta, f1(theta), "-", label="Asymptotic ")
ax[0, 0].plot(theta, f1_comsol(theta), "-", label="COMSOL")
ax[0, 0].set_ylabel(r"$f_1$")
ax[0, 0].legend()

# f2
ax[0, 1].plot(theta, f2(theta), "-", label="Asymptotic ")
ax[0, 1].plot(theta, f2_comsol(theta), "-", label="COMSOL")
ax[0, 1].set_ylabel(r"$f_2$")

# g1
ax[1, 0].plot(theta, g1(theta), "-", label="Asymptotic ")
ax[1, 0].plot(theta, g1_comsol(theta), "-", label="COMSOL")
ax[1, 0].set_ylabel(r"$g_1$")

# g2
ax[1, 1].plot(theta, g2(theta), "-", label="Asymptotic ")
ax[1, 1].plot(theta, g2_comsol(theta), "-", label="COMSOL")
ax[1, 1].set_ylabel(r"$g_2$")


# add shared labels etc.
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi
for ax in ax.reshape(-1):
    for w in winds:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: "{}$\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=2 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
    ax.set_xlabel(r"$\theta$")

plt.tight_layout()


# Plot at theta = 0 -----------------------------------------------------------

fig, ax = plt.subplots(2, 2)

# get r, theta values
r = []
theta = [2 * pi * n for n in list(range(N_plot))]
for tt in theta:
    r.append(r0 + delta * tt / 2 / pi)
r = np.array(r)
theta = np.array(theta)
winds_m = [r0 - hh / 2 + delta * n for n in list(range(N_plot))]
winds = [r0 + delta * n for n in list(range(N_plot))]
winds_p = [r0 + hh / 2 + delta * n for n in list(range(N_plot))]
windswinds = np.array(list(itertools.chain(*zip(winds[:-1], winds[1:]))))
thetatheta = np.array(list(itertools.chain(*zip(theta[:-1], theta[:-1]))))

# radial displacement
ax[0, 0].plot(r, u(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"u_0.csv", comment="#", header=None).to_numpy()
ax[0, 0].plot(comsol[:, 0], comsol[:, 1], "--", color="tab:orange", label="COMSOL")
ax[0, 0].plot(np.array(winds), f2(theta), "o", color="tab:blue", label=r"$f_2$")
ax[0, 0].plot(
    np.array(winds_p),
    f2_comsol(theta),
    "x",
    color="tab:orange",
    label=r"$f_2$ (COMSOL)",
)
# ax[0, 0].plot(
#    np.array(winds_m),
#    -(alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + f1_comsol(theta - 2 * pi))
#    + f2_comsol(theta - 2 * pi),
#    "o",
#    label=r"$-\alpha(3\lambda + 2\mu) / (\lambda + 2\mu)- f_1(\theta-2\pi) + f_2(\theta-2\pi)$",
# )
ax[0, 0].set_ylabel(r"$u$")
# ax[0, 0].legend()

# azimuthal displacement
ax[0, 1].plot(r, v(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"v_0.csv", comment="#", header=None).to_numpy()
ax[0, 1].plot(comsol[:, 0], comsol[:, 1], "--", color="tab:orange", label="COMSOL")
ax[0, 1].plot(np.array(winds), g2(theta), "o", color="tab:blue", label=r"$g_2$")
ax[0, 1].plot(
    np.array(winds_p),
    g2_comsol(theta),
    "x",
    color="tab:orange",
    label=r"$g_2$ (COMSOL)",
)
# ax[0, 1].plot(
#    np.array(winds_m),
#    -g1_comsol(theta - 2 * pi) * 0 + g2_comsol(theta - 2 * pi),
#    "o",
#    label=r"$-g_1(\theta-2\pi) + g_2(\theta-2\pi)$",
# )
ax[0, 1].set_ylabel(r"$v$")

# normal stress
ax[1, 0].plot(windswinds, f1(thetatheta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srr_0.csv", comment="#", header=None).to_numpy()
ax[1, 0].plot(
    comsol[:, 0],
    comsol[:, 1] / (lam + 2 * mu),
    "--",
    color="tab:orange",
    label="COMSOL",
)
ax[1, 0].plot(np.array(winds), f1(theta), "o", color="tab:blue", label=r"$f_1$")
ax[1, 0].plot(
    np.array(winds_p), f1_comsol(theta), "x", color="tab:orange", label=r"$f_1$ COMSOL"
)
ax[1, 0].set_ylabel(r"$f_1=\sigma_{rr}/(\lambda+2\mu)$")

# shear stress
ax[1, 1].plot(windswinds, g1(thetatheta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srt_0.csv", comment="#", header=None).to_numpy()
ax[1, 1].plot(comsol[:, 0], comsol[:, 1] / mu, "--", color="tab:orange", label="COMSOL")
ax[1, 1].plot(np.array(winds), g1(theta), "o", color="tab:blue", label=r"$g_1$")
ax[1, 1].plot(
    np.array(winds_p),
    g1_comsol(theta),
    "x",
    color="tab:orange",
    label=r"$g_1$ (COMSOL)",
)
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


# Plot at theta = pi ----------------------------------------------------------

fig, ax = plt.subplots(2, 2)

# get r, theta values
r = []
theta = [2 * pi * n + pi for n in list(range(N_plot))]
for tt in theta:
    r.append(r0 + delta * tt / 2 / pi)
r = np.array(r)
theta = np.array(theta)
winds_m = [
    r0 - hh / 2 + delta * (2 * pi * n + pi) / 2 / pi for n in list(range(N_plot))
]
winds = [r0 + delta * (2 * pi * n + pi) / 2 / pi for n in list(range(N_plot))]
winds_p = [
    r0 + hh / 2 + delta * (2 * pi * n + pi) / 2 / pi for n in list(range(N_plot))
]
windswinds = np.array(list(itertools.chain(*zip(winds[:-1], winds[1:]))))
thetatheta = np.array(list(itertools.chain(*zip(theta[:-1], theta[:-1]))))

# radial displacement
ax[0, 0].plot(r, u(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"u_pi.csv", comment="#", header=None).to_numpy()
ax[0, 0].plot(comsol[:, 0], comsol[:, 1], "--", color="tab:orange", label="COMSOL")
ax[0, 0].plot(np.array(winds), f2(theta), "o", color="tab:blue", label=r"$f_2$")
ax[0, 0].plot(
    np.array(winds_p),
    f2_comsol(theta),
    "x",
    color="tab:orange",
    label=r"$f_2$ (COMSOL)",
)
ax[0, 0].set_ylabel(r"$u$")

# azimuthal displacement
ax[0, 1].plot(r, v(r, theta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"v_pi.csv", comment="#", header=None).to_numpy()
ax[0, 1].plot(comsol[:, 0], comsol[:, 1], "--", color="tab:orange", label="COMSOL")
ax[0, 1].plot(np.array(winds), g2(theta), "o", color="tab:blue", label=r"$g_2$")
ax[0, 1].plot(
    np.array(winds_p),
    g2_comsol(theta),
    "x",
    color="tab:orange",
    label=r"$g_2$ (COMSOL)",
)
ax[0, 1].set_ylabel(r"$v$")

# normal stress
ax[1, 0].plot(windswinds, f1(thetatheta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srr_pi.csv", comment="#", header=None).to_numpy()
ax[1, 0].plot(
    comsol[:, 0],
    comsol[:, 1] / (lam + 2 * mu),
    "--",
    color="tab:orange",
    label="COMSOL",
)
ax[1, 0].plot(np.array(winds), f1(theta), "o", color="tab:blue", label=r"$f_1$")
ax[1, 0].plot(
    np.array(winds_p), f1_comsol(theta), "x", color="tab:orange", label=r"$f_1$ COMSOL"
)
ax[1, 0].set_ylabel(r"$f_1=\sigma_{rr}/(\lambda+2\mu)$")

# shear stress
ax[1, 1].plot(windswinds, g1(thetatheta), "-", color="tab:blue", label="Asymptotic")
comsol = pd.read_csv(path + f"srt_pi.csv", comment="#", header=None).to_numpy()
ax[1, 1].plot(comsol[:, 0], comsol[:, 1] / mu, "--", color="tab:orange", label="COMSOL")
ax[1, 1].plot(np.array(winds), g1(theta), "o", color="tab:blue", label=r"$g_1$")
ax[1, 1].plot(
    np.array(winds_p),
    g1_comsol(theta),
    "x",
    color="tab:orange",
    label=r"$g_1$ (COMSOL)",
)
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


plt.show()
