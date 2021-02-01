#
# Plot the tension in the boundary layers near r=r0 and r=r1 in the jelly roll
# mechanics problem
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
N_plot = 10  # number of winds to plot
path = "data/d01h001/"  # path to data


# Compute the tension from the boundary layer solution ------------------------
# Note: the solution is different for the first and last winds
theta1 = np.linspace(0, 2 * pi, 60)
theta2 = np.linspace(2 * pi, 2 * pi * (N - 1), 60 * (N - 2))
theta3 = np.linspace(2 * pi * (N - 1), 2 * pi * N, 60)
theta = np.concatenate((theta1, theta2, theta3))

# solution near r0 (decays away from r0)
T_r01 = -r0 * alpha * (3 * lam + 2 * mu) * (1 - exp(-omega * theta1))
T_r02 = (
    -r0 * alpha * (3 * lam + 2 * mu) * exp(-omega * theta2) * (exp(2 * pi * omega) - 1)
)
T_r03 = (
    -r0 * alpha * (3 * lam + 2 * mu) * exp(-omega * theta3) * (exp(2 * pi * omega) - 1)
)
T_r0 = np.concatenate((T_r01, T_r02, T_r03))

# solution near r1 (decays away from r1)
T_r11 = (
    r1
    * alpha
    * (3 * lam + 2 * mu)
    * exp(omega * (theta1 - 2 * N * pi))
    * (exp(2 * pi * omega) - 1)
)
T_r12 = (
    r1
    * alpha
    * (3 * lam + 2 * mu)
    * exp(omega * (theta2 - 2 * N * pi))
    * (exp(2 * pi * omega) - 1)
)
T_r13 = r1 * alpha * (3 * lam + 2 * mu) * (1 - exp(omega * (theta3 - 2 * N * pi)))
T_r1 = np.concatenate((T_r11, T_r12, T_r13))

# Plot solution(s) ------------------------------------------------------------
alpha100 = int(alpha * 100)


# COMSOL returns T(s), so we plot everything as a function of arc length
def arc_length(theta):
    integrand = np.sqrt((r0 + delta * theta / 2 / pi) ** 2 + (delta / 2 / pi) ** 2)
    return integrate.cumtrapz(integrand, theta, initial=0)


# Function to load COMSOL solution
def load_comsol(E_cc, alpha):
    path = f"data/E_cc_{int(E_cc)}/"
    alpha100 = int(alpha * 100)
    comsol = pd.read_csv(
        path + f"T3_alpha{alpha100}.csv", comment="#", header=None
    ).to_numpy()
    s = comsol[:, 0]  # arc length (at midpoint)
    T = comsol[:, 1]  # T at midpoint of current collector
    return s, T


# tension
fig, ax = plt.subplots()
plt.plot(arc_length(theta), T_r0, "-", label=r"Asymptotic ($r=r_0 + \delta R$)")
plt.plot(arc_length(theta), T_r1, "-", label=r"Asymptotic ($r=r_1 + \delta R$)")
comsol = pd.read_csv(
    path + f"T3_alpha{alpha100}.csv", comment="#", header=None
).to_numpy()
plt.plot(comsol[:, 0], comsol[:, 1], "--", label="COMSOL")
plt.xlabel(r"$s$")
plt.ylabel(r"$T$")
plt.title(f"$T(s)$ \n" r"$\alpha$ = " + f"{alpha}")
plt.legend()

plt.show()
