#
# Plot the results in the boundary layer near r=r0 in the jelly roll mechanics
# problem
#
import numpy as np
from numpy import pi, exp
from scipy import integrate
import scipy.interpolate as interp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


# Parameters ------------------------------------------------------------------
alpha = 0.10  # expansion coefficient
delta = 0.1
hh = 0.01  # current collector thickness
E = 1  # active material Young's modulus
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
r1 = r0 + delta * N
omega = np.sqrt(mu / (lam + 2 * mu))
N_plot = 9  # number of winds to plot
path = "data/fixed/h001/"  # path to data

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


# Load COMSOL solution --------------------------------------------------------

# 0 < theta < 2*N*pi
theta = np.linspace(0, 2 * pi * N, 360 * N)


# COMSOL returns f(s), so we plot everything as a function of arc length
def arc_length(theta, a=None, b=None):
    # default spiral is r = r0 +delta*theta/2/pi
    a = a or r0
    b = b or delta / 2 / pi
    integrand = np.sqrt((a + b * theta) ** 2 + b ** 2)
    return integrate.cumtrapz(integrand, theta, initial=0)


# f1 = sigma_rr / (lambda+2*mu)
# In COMSOL we evaluate at r = r0+delta/2+delta*theta/2/pi
s1 = interp.interp1d(theta, arc_length(theta, a=r0 + delta / 2))
comsol = pd.read_csv(path + "srr3.csv", comment="#", header=None).to_numpy()
s_data = comsol[:, 0]
f1_data = comsol[:, 1] / (lam + 2 * mu)
f1_interp = interp.interp1d(s_data, f1_data, fill_value="extrapolate")


def f1_comsol(theta):
    return f1_interp(s1(theta))


# f2 = u(R=theta/2/pi)
# In COMSOL we evaluate at r = r0+hh/2+delta*theta/2/pi
s2 = interp.interp1d(theta, arc_length(theta, a=r0 + hh / 2))
comsol = pd.read_csv(path + "u1.csv", comment="#", header=None).to_numpy()
s_data = comsol[:, 0]
f2_data = comsol[:, 1]
f2_interp = interp.interp1d(s_data, f2_data, fill_value="extrapolate")


def f2_comsol(theta):
    return f2_interp(s2(theta))


# g1 = sigma_rt/mu
# In COMSOL we evaluate at r = r0+delta/2+delta*theta/2/pi
comsol = pd.read_csv(path + "srt3.csv", comment="#", header=None).to_numpy()
s_data = comsol[:, 0]
g1_data = comsol[:, 1] / mu
g1_interp = interp.interp1d(s_data, g1_data, fill_value="extrapolate")


def g1_comsol(theta):
    return g1_interp(s1(theta))


# g2 = v(R=theta/2/pi)
# In COMSOL we evaluate at r = r0+hh/2+delta*theta/2/pi
comsol = pd.read_csv(path + "v1.csv", comment="#", header=None).to_numpy()
s_data = comsol[:, 0]
g2_data = comsol[:, 1]
g2_interp = interp.interp1d(s_data, g2_data, fill_value="extrapolate")


def g2_comsol(theta):
    return g2_interp(s2(theta))


# Plot solutions --------------------------------------------------------------

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
plt.savefig("figs/fg.pdf", dpi=300)


# Plot residuals ------------------------------------------------------------

fig, ax = plt.subplots(2, 2)

# 0 < theta < 2*N_plot*pi
theta = np.linspace(0, 2 * pi * N_plot, 60 * N_plot)

# inextensibility
ax[0, 0].plot(
    theta, np.gradient(g2(theta), theta) + f2(theta), "-", label="Asymptotic "
)
ax[0, 0].plot(
    theta, np.gradient(g2_comsol(theta), theta) + f2_comsol(theta), "-", label="COMSOL"
)
ax[0, 0].set_title("Inextensibility")
ax[0, 0].set_ylabel(r"$g_2' + f_2$")
ax[0, 0].legend()

# tension
ax[0, 1].plot(
    theta,
    (lam + 2 * mu) * np.gradient(f1(theta), theta) + mu * g1(theta),
    "-",
    label="Asymptotic ",
)
ax[0, 1].plot(
    theta,
    (lam + 2 * mu) * np.gradient(f1_comsol(theta), theta) + mu * g1_comsol(theta),
    "-",
    label="COMSOL",
)
ax[0, 1].set_title("Tension")
ax[0, 1].set_ylabel(r"$(\lambda+2\mu)f_1' + \mu g_1$")


# 2*pi < theta < 2*N_plot*pi
theta = np.linspace(2 * pi, 2 * pi * N_plot, 60 * (N_plot - 1))

# continuity of radial displacement (for theta > 2*pi)
ax[1, 0].plot(
    theta,
    f2(theta)
    + alpha * (3 * lam + 2 * mu) / (lam + 2 * mu)
    + f1(theta - 2 * pi)
    - f2(theta - 2 * pi),
    "-",
    label="Asymptotic ",
)
ax[1, 0].plot(
    theta,
    f2_comsol(theta)
    + alpha * (3 * lam + 2 * mu) / (lam + 2 * mu)
    + f1_comsol(theta - 2 * pi)
    - f2_comsol(theta - 2 * pi),
    "-",
    label="COMSOL",
)
ax[1, 0].set_title("Continuity of u")
ax[1, 0].set_ylabel(
    r"$f_2(\theta)+ \alpha(3\lambda + 2\mu) / (\lambda + 2\mu)$"
    + "\n"
    + r"$+ f_1(\theta-2\pi) - f_2(\theta-2\pi)$"
)

# continuity of azimuthal displacement (for theta > 2*pi)
ax[1, 1].plot(
    theta, g2(theta) + g1(theta - 2 * pi) - g2(theta - 2 * pi), "-", label="Asymptotic "
)
ax[1, 1].plot(
    theta,
    g2_comsol(theta) + g1_comsol(theta - 2 * pi) - g2_comsol(theta - 2 * pi),
    "-",
    label="COMSOL",
)
ax[1, 1].set_title("Continuity of v")
ax[1, 1].set_ylabel(r"$g_2(\theta) + g_1(\theta-2\pi) - g_2(\theta-2\pi)$")

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
plt.savefig("figs/residuals.pdf", dpi=300)


plt.show()
