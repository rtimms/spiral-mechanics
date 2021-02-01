#
# Plot the tension in the boundary layers near r=r0 and r=r1 in the jelly roll
# mechanics problem
#
import numpy as np
from scipy import integrate
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
alpha = 0.1  # expansion coefficient
delta = 0.1
E = 1  # active material Young's modulus
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
r1 = r0 + delta * N
omega = np.sqrt(mu / (lam + 2 * mu))


# Compute the tension from the boundary layer solution
# Note: the solution is different for the first and last winds
theta1 = np.linspace(0, 2 * np.pi, 60)
theta2 = np.linspace(2 * np.pi, 2 * np.pi * (N - 1), 60 * (N - 2))
theta3 = np.linspace(2 * np.pi * (N - 1), 2 * np.pi * N, 60)
theta = np.concatenate((theta1, theta2, theta3))


def tension_r0(alpha):
    T1 = -r0 * alpha * (3 * lam + 2 * mu) * (1 - np.exp(-omega * theta1))
    T2 = (
        -r0
        * alpha
        * (3 * lam + 2 * mu)
        * np.exp(-omega * theta2)
        * (np.exp(2 * np.pi * omega) - 1)
    )
    T3 = (
        -r0
        * alpha
        * (3 * lam + 2 * mu)
        * np.exp(-omega * theta3)
        * (np.exp(2 * np.pi * omega) - 1)
    )
    return np.concatenate((T1, T2, T3))


def tension_r1(alpha):
    T1 = (
        r1
        * alpha
        * (3 * lam + 2 * mu)
        * np.exp(omega * (theta1 - 2 * N * np.pi))
        * (np.exp(2 * np.pi * omega) - 1)
    )
    T2 = (
        r1
        * alpha
        * (3 * lam + 2 * mu)
        * np.exp(omega * (theta2 - 2 * N * np.pi))
        * (np.exp(2 * np.pi * omega) - 1)
    )
    T3 = (
        r1 * alpha * (3 * lam + 2 * mu) * (1 - np.exp(omega * (theta3 - 2 * N * np.pi)))
    )
    return np.concatenate((T1, T2, T3))


# COMSOL returns T(s), so we plot everything as a function of arc length
def arc_length(theta):
    integrand = np.sqrt(
        (r0 + delta * theta / 2 / np.pi) ** 2 + (delta / 2 / np.pi) ** 2
    )
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


# Plot solution(s)
N_plot = N
path = "data/"
alpha100 = int(alpha * 100)
fig, ax = plt.subplots()
plt.plot(
    arc_length(theta), tension_r0(alpha), "-", label=r"Asymptotic ($r=r_0 + \delta R$)"
)
plt.plot(
    arc_length(theta), tension_r1(alpha), "-", label=r"Asymptotic ($r=r_1 + \delta R$)"
)
comsol = pd.read_csv(
    path + f"T3_alpha{alpha100}.csv", comment="#", header=None
).to_numpy()
s = comsol[:, 0]  # arc length
T = comsol[:, 1]  # tension
plt.plot(s, T, "--", label="COMSOL")
plt.xlim(arc_length(np.array([0, N_plot * 2 * np.pi])))
# plt.ylim([-0.3, 0.05])
plt.xlabel(r"$s$")
plt.ylabel(r"$T(s)$")
plt.title("$T(s)$, " r"$\alpha$ = " + f"{alpha}")
plt.legend()
plt.show()
