import numpy as np
from numpy import pi
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution

# Parameters ------------------------------------------------------------------
alpha = 0.10  # expansion coefficient
delta = 0.1
hh = 0.001  # current collector thickness
E = 1  # active material Young's modulus
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
r1 = r0 + delta * N
omega = np.sqrt(mu / (lam + 2 * mu))
N_plot = 9  # number of winds to plot
path = f"data/E5e4h{str(hh-int(hh))[2:]}/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------
theta = np.linspace(0, 2 * pi * N, 60 * (N - 1))
outer = OuterSolution(alpha, delta, E, nu, r0)
T = outer.T(theta)

# Load COMSOL solutions ----------------------------------------------
# Note: COMSOL data is (r, f) so we create interpolants to get (theta, f) data

# In COMSOL we evaluate the tension at three points:
# ra = r0-hh/2+delta*theta/2/pi
# rb = r0+delta*theta/2/pi
# rc = r0+hh/2+delta*theta/2/pi
ra = r0 - hh / 2 + delta * theta / 2 / pi
rb = r0 + delta * theta / 2 / pi
rc = r0 + hh / 2 + delta * theta / 2 / pi

# tension
comsola = pd.read_csv(path + "T1.csv", comment="#", header=None).to_numpy()
T_r_dataa = comsola[:, 0]
T_dataa = comsola[:, 1]
T_interpa = interp.interp1d(T_r_dataa, T_dataa, bounds_error=False)
T_comsola = T_interpa(ra)

comsolb = pd.read_csv(path + "T3.csv", comment="#", header=None).to_numpy()
T_r_datab = comsolb[:, 0]
T_datab = comsolb[:, 1]
T_interpb = interp.interp1d(T_r_datab, T_datab, bounds_error=False)
T_comsolb = T_interpb(rb)

comsolc = pd.read_csv(path + "T5.csv", comment="#", header=None).to_numpy()
T_r_datac = comsolc[:, 0]
T_datac = comsolc[:, 1]
T_interpc = interp.interp1d(T_r_datac, T_datac, bounds_error=False)
T_comsolc = T_interpb(rc)


# compute using simpsons rule
T_comsol = (T_comsola + 4 * T_comsolb + T_comsolc) / 6

# Plot solution(s) ------------------------------------------------------------
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi

# tension
fig, ax = plt.subplots()
ax.plot(theta, T, "-", label="Asymptotic")
ax.plot(theta, T_comsola, ":", label="COMSOL (a)")
ax.plot(theta, T_comsolb, ":", label="COMSOL (b)")
ax.plot(theta, T_comsolc, ":", label="COMSOL (c)")
ax.plot(theta, T_comsol, "--", label="COMSOL (Simpson's)")
ax.set_ylabel(r"$T$")
ax.legend()
# add shared labels etc.
for w in winds:
    ax.axvline(x=w, linestyle=":", color="lightgrey")
ax.xaxis.set_major_formatter(
    FuncFormatter(
        lambda val, pos: "{}$\pi$".format(int(val / np.pi)) if val != 0 else "0"
    )
)
ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
ax.set_xlim([0, N_plot * 2 * pi])
ax.set_xlabel(r"$\theta$")
plt.tight_layout()
# plt.savefig("figs" + path[4:] + "T_of_theta.pdf", dpi=300)

plt.show()
