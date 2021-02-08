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
g1 = -(lam + 2 * mu) / mu * omega * A * exp(-omega * (theta + 2 * pi))
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
e_tt = (delta / r0) * (np.gradient(v, theta) + u)
# shear strain
e_rt = g1 / 2

# radial stress
s_rr = (lam + 2 * mu) * f1
# azimuthal stress
s_tt = -2 * mu * alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + lam * f1
# shear stress
s_rt = mu * g1

# tension
T1 = -r0 * alpha * (3 * lam + 2 * mu) * (1 - exp(-omega * theta1))
T2 = -r0 * alpha * (3 * lam + 2 * mu) * exp(-omega * theta2) * (exp(2 * pi * omega) - 1)
T = np.concatenate((T1, T2))


# Plot solution(s) ------------------------------------------------------------

# COMSOL returns T(s), so we plot everything as a function of arc length
def arc_length(theta):
    integrand = np.sqrt((r0 + delta * theta / 2 / pi) ** 2 + (delta / 2 / pi) ** 2)
    return integrate.cumtrapz(integrand, theta, initial=0)


# get s coords of the ends of each wind
winds = []
for n in list(range(N_plot)):
    w = arc_length((np.linspace(0, n * 2 * pi, 60)))[-1]
    winds.append(w)


# names = ["free", "fixed"]
# paths = ["data/free/h001/", "data/fixed/h001/"]
names = ["fixed"]
paths = ["data/fixed/h001/"]

# radial displacement
fig, ax = plt.subplots()
plt.plot(arc_length(theta), u, "-", label=r"Asymptotic ($r=r_0+\delta R$)")
for name, path in zip(names, paths):
    comsol = pd.read_csv(path + "u3.csv", comment="#", header=None).to_numpy()
    plt.plot(comsol[:, 0], comsol[:, 1], "--", label=f"COMSOL ({name} outer)")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$u$")
plt.title(f"$u(s; R=0.5)$ in the first {N_plot} winds")
plt.legend()

# azimuthal displacement
fig, ax = plt.subplots()
plt.plot(arc_length(theta), v, "-", label=r"Asymptotic ($r=r_0+\delta R$)")
for name, path in zip(names, paths):
    comsol = pd.read_csv(path + "v3.csv", comment="#", header=None).to_numpy()
    plt.plot(comsol[:, 0], comsol[:, 1], "--", label=f"COMSOL ({name} outer)")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$v$")
plt.title(f"$v(s; R=0.5)$ in the first {N_plot} winds")
plt.legend()

# radial strain
fig, ax = plt.subplots()
plt.plot(arc_length(theta), e_rr, "-", label=r"Asymptotic ($r=r_0+\delta R$)")
for name, path in zip(names, paths):
    comsol = pd.read_csv(path + "err3.csv", comment="#", header=None).to_numpy()
    plt.plot(comsol[:, 0], comsol[:, 1], "--", label=f"COMSOL ({name} outer)")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$\epsilon_{rr}$")
plt.title(r"$\epsilon_{rr}(s; R=0.5)$" + f"in the first {N_plot} winds")
plt.legend()

# azimuthal strain
fig, ax = plt.subplots()
plt.plot(arc_length(theta), e_tt, "-", label=r"Asymptotic ($r=r_0+\delta R$)")
for name, path in zip(names, paths):
    comsol = pd.read_csv(path + "ett3.csv", comment="#", header=None).to_numpy()
    plt.plot(comsol[:, 0], comsol[:, 1], "--", label=f"COMSOL ({name} outer)")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$\epsilon_{\theta\theta}$")
plt.title(r"$\epsilon_{\theta\theta}(s; R=0.5)$" + f"in the first {N_plot} winds")
plt.legend()

# shear strain
fig, ax = plt.subplots()
plt.plot(arc_length(theta), e_rt, "-", label=r"Asymptotic ($r=r_0+\delta R$)")
for name, path in zip(names, paths):
    comsol = pd.read_csv(path + "ert3.csv", comment="#", header=None).to_numpy()
    plt.plot(comsol[:, 0], comsol[:, 1], "--", label=f"COMSOL ({name} outer)")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$\epsilon_{r\theta}$")
plt.title(r"$\epsilon_{r\theta}(s; R=0.5)$" + f"in the first {N_plot} winds")
plt.legend()

# radial stress
fig, ax = plt.subplots()
plt.plot(arc_length(theta), s_rr, "-", label=r"Asymptotic ($r=r_0+\delta R$)")
for name, path in zip(names, paths):
    comsol = pd.read_csv(path + "srr3.csv", comment="#", header=None).to_numpy()
    plt.plot(comsol[:, 0], comsol[:, 1], "--", label=f"COMSOL ({name} outer)")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$\sigma_{rr}$")
plt.title(r"$\sigma_{rr}(s)$" + f"in the first {N_plot} winds")
plt.legend()

# azimuthal stress
fig, ax = plt.subplots()
plt.plot(arc_length(theta), s_tt, "-", label=r"Asymptotic ($r=r_0+\delta R$)")
for name, path in zip(names, paths):
    comsol = pd.read_csv(path + "stt3.csv", comment="#", header=None).to_numpy()
    plt.plot(comsol[:, 0], comsol[:, 1], "--", label=f"COMSOL ({name} outer)")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$\sigma_{\theta\theta}$")
plt.title(r"$\sigma_{\theta\theta}(s)$" + f"in the first {N_plot} winds")
plt.legend()

# shear stress
fig, ax = plt.subplots()
plt.plot(arc_length(theta), s_rt, "-", label=r"Asymptotic ($r=r_0+\delta R$)")
for name, path in zip(names, paths):
    comsol = pd.read_csv(path + "srt3.csv", comment="#", header=None).to_numpy()
    plt.plot(comsol[:, 0], comsol[:, 1], "--", label=f"COMSOL ({name} outer)")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$\sigma_{r\theta}$")
plt.title(r"$\sigma_{r\theta}(s)$" + f"in the first {N_plot} winds")
plt.legend()

# tension
fig, ax = plt.subplots()
plt.plot(arc_length(theta), T, "-", label=r"Asymptotic ($r=r_0+\delta R$)")
for name, path in zip(names, paths):
    comsol = pd.read_csv(path + "T3.csv", comment="#", header=None).to_numpy()
    plt.plot(comsol[:, 0], comsol[:, 1], "--", label=f"COMSOL ({name} outer)")
for w in winds:
    plt.axvline(x=w, linestyle=":", color="lightgrey")
plt.xlim(arc_length(np.array([0, N_plot * 2 * pi])))
plt.xlabel(r"$s$")
plt.ylabel(r"$T$")
plt.title(f"$T(s)$ in the first {N_plot} winds")
plt.legend()

plt.show()
