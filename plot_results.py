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
N_plot = 7  # number of winds to plot
# path = "data/d01h0001/"  # path to data (free outer)
path = "data/d01h0001_u0r1/"  # path to data (fixed outer)

# Compute the boundary layer solution -----------------------------------------
# Note: the tension is different for the first wind
theta1 = np.linspace(0, 2 * pi, 60)
theta2 = np.linspace(2 * pi, 2 * pi * N, 60 * (N - 1))
theta = np.concatenate((theta1, theta2))

# plot displacements at fixed R
r = r0 + delta / 2 + delta * theta / 2 / pi
R = (r - r0) / delta

# constants
A = -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * exp(2 * pi * omega)
B = 0
C = A / (1 - exp(2 * pi * omega))
D = 0
# functions of theta
f1 = -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) - A * exp(-omega * (theta + 2 * pi))
f2 = B + C * exp(-omega * theta)
g1 = -(lam + 2 * mu) * omega / (2 * pi * mu) * A * exp(-omega * (theta + 2 * pi))
g2 = D - C / omega * exp(-omega * theta)

# radial displacement
u = (
    alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * (R - theta / 2 / pi)
    + f1 * (R - theta / 2 / pi)
    + f2
)
# azimuthal displacement
v = g1 * (R - theta / 2 / pi) + g2

# radial strain
e_rr = alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + f1
# azimuthal strain
dg1 = np.gradient(g1, theta)
dg2 = np.gradient(g2, theta)
e_tt = (delta / r0) * (dg1 * (R - theta / 2 / pi) + dg2 + u)
# shear strain
e_rt = g1 / 2

# tension
T1 = -r0 * alpha * (3 * lam + 2 * mu) * (1 - exp(-omega * theta1))
T2 = -r0 * alpha * (3 * lam + 2 * mu) * exp(-omega * theta2) * (exp(2 * pi * omega) - 1)
T = np.concatenate((T1, T2))


# Plot solution(s) ------------------------------------------------------------
alpha100 = int(alpha * 100)


# COMSOL returns T(s), so we plot everything as a function of arc length
def arc_length(theta):
    integrand = np.sqrt((r0 + delta * theta / 2 / pi) ** 2 + (delta / 2 / pi) ** 2)
    return integrate.cumtrapz(integrand, theta, initial=0)


# get s coords of the ends of each wind
winds = []
for n in list(range(N_plot)):
    w = arc_length((np.linspace(0, n * 2 * pi, 60)))[-1]
    winds.append(w)

# functions of theta
fig, ax = plt.subplots()
plt.plot(theta, f1, "-", label=r"$f_1$")
plt.plot(theta, f2, "-", label=r"$f_2$")
plt.plot(theta, g1, "-", label=r"$g_1$")
plt.plot(theta, g2, "-", label=r"$g_2$")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$\theta$")
plt.title(f"Solutions in the first {N_plot} winds \n" r"$\alpha$ = " + f"{alpha}")
plt.legend()

# radial displacement
fig, ax = plt.subplots()
plt.plot(arc_length(theta), u, "-", label="Asymptotic")
comsol = pd.read_csv(
    path + f"u3_alpha{alpha100}.csv", comment="#", header=None
).to_numpy()
plt.plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$u$")
plt.title(f"$u(s; R=0.5)$ in the first {N_plot} winds \n" r"$\alpha$ = " + f"{alpha}")
plt.legend()

# azimuthal displacement
fig, ax = plt.subplots()
plt.plot(arc_length(theta), v, "-", label="Asymptotic")
comsol = pd.read_csv(
    path + f"v3_alpha{alpha100}.csv", comment="#", header=None
).to_numpy()
plt.plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$v$")
plt.title(f"$v(s; R=0.5)$ in the first {N_plot} winds \n" r"$\alpha$ = " + f"{alpha}")
plt.legend()

# radial strain
fig, ax = plt.subplots()
plt.plot(arc_length(theta), e_rr, "-", label="Asymptotic")
comsol = pd.read_csv(
    path + f"err3_alpha{alpha100}.csv", comment="#", header=None
).to_numpy()
plt.plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$\epsilon_{rr}$")
plt.title(
    r"$\epsilon_{rr}(s; R=0.5)$" + f"in the first {N_plot} winds \n"
    r"$\alpha$ = " + f"{alpha}"
)
plt.legend()

# azimuthal strain
fig, ax = plt.subplots()
plt.plot(arc_length(theta), e_tt, "-", label="Asymptotic")
comsol = pd.read_csv(
    path + f"ett3_alpha{alpha100}.csv", comment="#", header=None
).to_numpy()
plt.plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$\epsilon_{\theta\theta}$")
plt.title(
    r"$\epsilon_{\theta\theta}(s; R=0.5)$" + f"in the first {N_plot} winds \n"
    r"$\alpha$ = " + f"{alpha}"
)
plt.legend()

# shear strain
fig, ax = plt.subplots()
plt.plot(arc_length(theta), e_rt, "-", label="Asymptotic")
comsol = pd.read_csv(
    path + f"ert3_alpha{alpha100}.csv", comment="#", header=None
).to_numpy()
plt.plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$\epsilon_{r\theta}$")
plt.title(
    r"$\epsilon_{r\theta}(s; R=0.5)$" + f"in the first {N_plot} winds \n"
    r"$\alpha$ = " + f"{alpha}"
)
plt.legend()

# tension
fig, ax = plt.subplots()
plt.plot(arc_length(theta), T, "-", label="Asymptotic")
comsol = pd.read_csv(
    path + f"T3_alpha{alpha100}.csv", comment="#", header=None
).to_numpy()
plt.plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$T$")
plt.title(f"$T(s)$ in the first {N_plot} winds \n" r"$\alpha$ = " + f"{alpha}")
plt.legend()

plt.show()
