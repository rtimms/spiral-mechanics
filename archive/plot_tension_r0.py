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
N_plot = 4  # number of winds to plot
path = "data/d01h0005/"  # path to data

# Compute the boundary layer solution -----------------------------------------
# Note: the tension is different for the first wind
theta1 = np.linspace(0, 2 * pi, 60)
theta2 = np.linspace(2 * pi, 2 * pi * (N - 1), 60 * (N - 2))
theta = np.concatenate((theta1, theta2))

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
plt.ylim([-r0 * alpha * (3 * lam + 2 * mu) * 1.2, 0.01])
plt.xlabel(r"$s$")
plt.ylabel(r"$T$")
plt.title(f"$T(s)$ in the first {N_plot} winds \n" r"$\alpha$ = " + f"{alpha}")
plt.legend()

plt.show()
