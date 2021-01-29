#
# Plot the tension in the boundary layer near r=r0 in the jelly roll mechanics
# problem
#
import numpy as np
from scipy import integrate
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
alphas = [0.05, 0.1, 0.15]  # expansion coefficient (list)
delta = 0.1
E = 1  # active material Young's modulus
E_ccs = [1e3, 1e4, 1e5]  # current collector Young's modulus (list)
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
omega = np.sqrt(2 * np.pi * mu / (lam + 2 * mu))
# omega = np.sqrt(mu / (lam + 2 * mu))

# Compute the tension from the boundary layer solution
# Note: the solution is different for the first wind
theta1 = np.linspace(0, 2 * np.pi, 60)
theta2 = np.linspace(2 * np.pi, 2 * np.pi * N, 60 * (N - 1))
theta = np.concatenate((theta1, theta2))


def tension(alpha):
    T1 = -r0 * alpha * (3 * lam + 2 * mu) * (1 - np.exp(-omega * theta1))
    T2 = (
        -r0
        * alpha
        * (3 * lam + 2 * mu)
        * np.exp(-omega * theta2)
        * (np.exp(2 * np.pi * omega) - 1)
    )
    return np.concatenate((T1, T2))


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
N_plot = 5
for E_cc in E_ccs:
    fig, ax = plt.subplots()
    for alpha in alphas:
        color = next(ax._get_lines.prop_cycler)["color"]
        s, T = load_comsol(E_cc, alpha)
        plt.plot(
            arc_length(theta),
            tension(alpha),
            "-",
            color=color,
            label=r"Asymptotic ($\alpha$=" + f"{alpha})",
        )
        plt.plot(s, T, "--", color=color, label=r"COMSOL ($\alpha$=" + f"{alpha})")
    plt.xlim(arc_length(np.array([0, N_plot * 2 * np.pi])))
    plt.ylim([-0.3, 0.05])
    plt.xlabel(r"$s$")
    plt.ylabel(r"$T(s)$")
    plt.title(
        f"$T(s)$ in the first {N_plot} winds \n" r"$E_{cc}/E_a$ = " + f"{int(E_cc)}"
    )
    plt.legend()
plt.show()
