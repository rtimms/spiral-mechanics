#
# Plot the tension in the boundary layer near r=r0 in the jelly roll mechanics
# problem
#
import numpy as np
from scipy import integrate
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
alpha = 0.1
delta = 0.1
E = 1
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
omega = np.sqrt(mu / (lam + 2 * mu))

# Compute the tension from the boundary layer solution
# Note: the solution is different for the first wind
theta1 = np.linspace(0, 2 * np.pi, 50)
theta2 = np.linspace(2 * np.pi, 2 * np.pi * N, 50 * (N - 1))
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


# Function to load COMSOL solution (hacky averaging for now)
def load_comsol(alpha):
    alpha100 = int(alpha * 100)
    comsol1 = pd.read_csv(
        f"data/E_cc_100000/T1_alpha{alpha100}.csv", comment="#", header=None
    ).to_numpy()
    # comsol2 = pd.read_csv(
    #    f"data/E_cc_100000/T2_alpha{alpha100}.csv", comment="#", header=None
    # ).to_numpy()
    comsol3 = pd.read_csv(
        f"data/E_cc_100000/T3_alpha{alpha100}.csv", comment="#", header=None
    ).to_numpy()
    # comsol4 = pd.read_csv(
    #    f"data/E_cc_100000/T4_alpha{alpha100}.csv", comment="#", header=None
    # ).to_numpy()
    comsol5 = pd.read_csv(
        f"data/E_cc_100000/T5_alpha{alpha100}.csv", comment="#", header=None
    ).to_numpy()
    s = comsol3[:, 0]  # arc length (at midpoint)
    T_m = comsol3[:, 1]  # T at midpoint of current collector
    T_av = (
        comsol1[:, 1] + comsol3[:, 1] + comsol5[:, 1]
    ) / 3  # "average" over 5 equispaced curves
    return s, T_m, T_av


# Plot solution(s)
alphas = [0.05, 0.1, 0.15]
N_plot = 4
fig, ax = plt.subplots()
for alpha in alphas:
    color = next(ax._get_lines.prop_cycler)["color"]
    s, T_m, T_av = load_comsol(alpha)
    plt.plot(
        arc_length(theta),
        tension(alpha),
        "-",
        color=color,
        label=f"Asymptotic (alpha={alpha})",
    )
    plt.plot(s, T_m, "--", color=color, label=f"COMSOL (T_m)")
    plt.plot(s, T_av, ":", color=color, label=f"COMSOL (T_av)")
plt.xlim(arc_length(np.array([0, N_plot * 2 * np.pi])))
plt.ylim([-0.5, 0.05])
plt.xlabel("s")
plt.ylabel("T")
plt.title(f"T(s) in the first {N_plot} winds")
plt.legend()
plt.show()