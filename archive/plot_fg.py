#
# Plot the results in the boundary layer near r=r0 in the jelly roll mechanics
# problem
#
import numpy as np
from numpy import pi, exp
from scipy import integrate
import pandas as pd
import matplotlib.pyplot as plt

# Parameters ------------------------------------------------------------------
alpha = 0.10  # expansion coefficient
delta = 0.1
E = 1  # active material Young's modulus
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
r1 = r0 + delta * N
omega = np.sqrt(mu / (lam + 2 * mu))
N_plot = 9  # number of winds to plot
path = "data/fixed/h0005/"  # path to data

# Compute the boundary layer solution -----------------------------------------
theta = np.linspace(0, 2 * pi * N, 60 * N)

# constants
A = -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * exp(2 * pi * omega)
B = 0
C = A / (1 - exp(2 * pi * omega))
D = 0

# functions of theta
f1 = -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) - A * exp(-omega * (theta + 2 * pi))
f2 = B + C * exp(-omega * theta)
g1 = -(lam + 2 * mu) / mu * omega * A * exp(-omega * (theta + 2 * pi))
g2 = D - C / omega * exp(-omega * theta)


# Plot solution(s) ------------------------------------------------------------

# COMSOL returns f(s), so we plot everything as a function of arc length
def arc_length(theta):
    integrand = np.sqrt((r0 + delta * theta / 2 / pi) ** 2 + (delta / 2 / pi) ** 2)
    return integrate.cumtrapz(integrand, theta, initial=0)


# get s coords of the ends of each wind
winds = []
for n in list(range(N_plot)):
    w = arc_length((np.linspace(0, n * 2 * pi, 60)))[-1]
    winds.append(w)

fig, ax = plt.subplots(2, 2)

# f1
comsol = pd.read_csv(path + "srr3.csv", comment="#", header=None).to_numpy()
f1_s = comsol[:, 0]
f1_comsol = comsol[:, 1] / (lam + 2 * mu)  # f1 = sigma_rr / (lambda+2*mu)
ax[0, 0].plot(arc_length((theta)), f1, "-", label="Asymptotic ")
ax[0, 0].plot(f1_s, f1_comsol, "-", label="COMSOL")
for w in winds:
    ax[0, 0].axvline(x=w, linestyle=":", color="lightgrey")
ax[0, 0].set_xlim(arc_length(np.array([0, N_plot * 2 * pi])))
ax[0, 0].set_xlabel(r"$s$")
ax[0, 0].set_ylabel(r"$f_1$")
ax[0, 0].legend()

# f2
comsol = pd.read_csv(path + "u1.csv", comment="#", header=None).to_numpy()
f2_s = comsol[:, 0]
f2_comsol = comsol[:, 1]  # f2 = u(R=theta/2/pi)
ax[0, 1].plot(arc_length((theta)), f2, "-", label="Asymptotic ")
ax[0, 1].plot(f2_s, f2_comsol, "-", label="COMSOL")
for w in winds:
    ax[0, 1].axvline(x=w, linestyle=":", color="lightgrey")
ax[0, 1].set_xlim(arc_length(np.array([0, N_plot * 2 * pi])))
ax[0, 1].set_xlabel(r"$s$")
ax[0, 1].set_ylabel(r"$f_2$")
# ax[0, 1].legend()

# g1
comsol = pd.read_csv(path + "srt3.csv", comment="#", header=None).to_numpy()
g1_s = comsol[:, 0]
g1_comsol = comsol[:, 1] / mu  # g1 = sigma_rt/mu
ax[1, 0].plot(arc_length((theta)), g1, "-", label="Asymptotic ")
ax[1, 0].plot(g1_s, g1_comsol, "-", label="COMSOL")
for w in winds:
    ax[1, 0].axvline(x=w, linestyle=":", color="lightgrey")
ax[1, 0].set_xlim(arc_length(np.array([0, N_plot * 2 * pi])))
ax[1, 0].set_xlabel(r"$s$")
ax[1, 0].set_ylabel(r"$g_1$")
# ax[1, 0].legend()

# g2
comsol = pd.read_csv(path + "v1.csv", comment="#", header=None).to_numpy()
g2_s = comsol[:, 0]
g2_comsol = comsol[:, 1]  # g2 = v(R=theta/2/pi)
ax[1, 1].plot(arc_length((theta)), g2, "-", label="Asymptotic ")
ax[1, 1].plot(g2_s, g2_comsol, "-", label="COMSOL")
for w in winds:
    ax[1, 1].axvline(x=w, linestyle=":", color="lightgrey")
ax[1, 1].set_xlim(arc_length(np.array([0, N_plot * 2 * pi])))
ax[1, 1].set_xlabel(r"$s$")
ax[1, 1].set_ylabel(r"$g_2$")
# ax[1, 1].legend()

plt.tight_layout()
plt.savefig("figs/fg.pdf", dpi=300)

plt.show()
