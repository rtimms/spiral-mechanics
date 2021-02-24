import numpy as np
from numpy import pi, exp
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os

# Parameters ------------------------------------------------------------------
alpha = 0.10  # expansion coefficient
delta = 0.1
hh = 0.005  # current collector thickness
E = 1  # active material Young's modulus
nu = 1 / 3
lam = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
N = 10
r0 = 0.5
r1 = r0 + delta * N
omega = np.sqrt(mu / (lam + 2 * mu))
N_plot = 9  # number of winds to plot
path = "data/E1e4h005/"  # path to data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the boundary layer solution -----------------------------------------
# Note: the tension is different for the first wind
theta1 = np.linspace(0, 2 * pi, 60)
theta2 = np.linspace(2 * pi, 2 * pi * (N - 1), 60 * (N - 2))
theta = np.concatenate((theta1, theta2))

# plot displacements at fixed R-theta/2/pi=0.5
r = r0 + delta / 2 + delta * theta / 2 / pi
R = (r - r0) / delta

# constants
A = alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * exp(2 * pi * omega)
B = 0
C = A / (1 - exp(2 * pi * omega))
D = 0
# functions of theta
f1 = -alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + A * exp(-omega * (theta + 2 * pi))
f2 = B + C * exp(-omega * theta)
g1 = (lam + 2 * mu) / mu * omega * A * exp(-omega * (theta + 2 * pi))
g2 = D + C / omega * exp(-omega * theta)

# radial displacement
u = delta * (
    alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) * (R - theta / 2 / pi)
    + f1 * (R - theta / 2 / pi)
    + f2
)
# azimuthal displacement
v = delta * (g1 * (R - theta / 2 / pi) + g2)

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

# Load COMSOL solutions ----------------------------------------------
# Note: COMSOL data is (r, f) so we create interpolants to get (theta, f) data

# f1 = sigma_rr / (lambda+2*mu)
comsol = pd.read_csv(path + "srr3.csv", comment="#", header=None).to_numpy()
f1_r_data = comsol[:, 0]
f1_data = comsol[:, 1] / (lam + 2 * mu)
f1_interp = interp.interp1d(f1_r_data, f1_data, bounds_error=False)
# In COMSOL we evaluate f_1 at r = r0+delta/2+delta*theta/2/pi
r = r0 + delta / 2 + delta * theta / 2 / pi
f1_comsol = f1_interp(r)

# f2 = u(R=theta/2/pi)/delta
comsol = pd.read_csv(path + "u1.csv", comment="#", header=None).to_numpy()
f2_r_data = comsol[:, 0]
f2_data = comsol[:, 1] / delta
f2_interp = interp.interp1d(f2_r_data, f2_data, bounds_error=False)
# In COMSOL we evaluate f_2 at r = r0+hh/2+delta*theta/2/pi
r = r0 + hh / 2 + delta * theta / 2 / pi
f2_comsol = f2_interp(r)

# g1 = sigma_rt/mu
comsol = pd.read_csv(path + "srt3.csv", comment="#", header=None).to_numpy()
g1_r_data = comsol[:, 0]
g1_data = comsol[:, 1] / mu
g1_interp = interp.interp1d(g1_r_data, g1_data, bounds_error=False)
# In COMSOL we evaluate g_1 at r = r0+delta/2+delta*theta/2/pi
r = r0 + delta / 2 + delta * theta / 2 / pi
g1_comsol = g1_interp(r)


# g2 = v(R=theta/2/pi)/delta
comsol = pd.read_csv(path + "v1.csv", comment="#", header=None).to_numpy()
g2_r_data = comsol[:, 0]
g2_data = comsol[:, 1] / delta
g2_interp = interp.interp1d(g2_r_data, g2_data, bounds_error=False)
# In COMSOL we evaluate g_2 at r = r0+hh/2+delta*theta/2/pi
r = r0 + hh / 2 + delta * theta / 2 / pi
g2_comsol = g2_interp(r)


# In COMSOL we evaluate the displacements, stresses and strains
# at r = r0+delta/2+delta*theta/2/pi
r = r0 + delta / 2 + delta * theta / 2 / pi

# radial displacement
comsol = pd.read_csv(path + "u3.csv", comment="#", header=None).to_numpy()
u_r_data = comsol[:, 0]
u_data = comsol[:, 1]
u_interp = interp.interp1d(u_r_data, u_data, bounds_error=False)
u_comsol = u_interp(r)

# azimuthal displacement
comsol = pd.read_csv(path + "v3.csv", comment="#", header=None).to_numpy()
v_r_data = comsol[:, 0]
v_data = comsol[:, 1]
v_interp = interp.interp1d(v_r_data, v_data, bounds_error=False)
v_comsol = v_interp(r)

# radial strain
comsol = pd.read_csv(path + "err3.csv", comment="#", header=None).to_numpy()
err_r_data = comsol[:, 0]
err_data = comsol[:, 1]
err_interp = interp.interp1d(err_r_data, err_data, bounds_error=False)
err_comsol = err_interp(r)

# azimuthal strain
comsol = pd.read_csv(path + "ett3.csv", comment="#", header=None).to_numpy()
ett_r_data = comsol[:, 0]
ett_data = comsol[:, 1]
ett_interp = interp.interp1d(ett_r_data, ett_data, bounds_error=False)
ett_comsol = ett_interp(r)

# shear strain
comsol = pd.read_csv(path + "ert3.csv", comment="#", header=None).to_numpy()
ert_r_data = comsol[:, 0]
ert_data = comsol[:, 1]
ert_interp = interp.interp1d(ert_r_data, ert_data, bounds_error=False)
ert_comsol = ert_interp(r)

# radial stress
comsol = pd.read_csv(path + "srr3.csv", comment="#", header=None).to_numpy()
srr_r_data = comsol[:, 0]
srr_data = comsol[:, 1]
srr_interp = interp.interp1d(srr_r_data, srr_data, bounds_error=False)
srr_comsol = srr_interp(r)

# azimuthal stress
comsol = pd.read_csv(path + "stt3.csv", comment="#", header=None).to_numpy()
stt_r_data = comsol[:, 0]
stt_data = comsol[:, 1]
stt_interp = interp.interp1d(stt_r_data, stt_data, bounds_error=False)
stt_comsol = stt_interp(r)

# shear stress
comsol = pd.read_csv(path + "srt3.csv", comment="#", header=None).to_numpy()
srt_r_data = comsol[:, 0]
srt_data = comsol[:, 1]
srt_interp = interp.interp1d(srt_r_data, srt_data, bounds_error=False)
srt_comsol = srt_interp(r)

# In COMSOL we evaluate the tension at r = r0+delta*theta/2/pi
r = r0 + delta * theta / 2 / pi

# tension
comsol = pd.read_csv(path + "T3.csv", comment="#", header=None).to_numpy()
T_r_data = comsol[:, 0]
T_data = comsol[:, 1]
T_interp = interp.interp1d(T_r_data, T_data, bounds_error=False)
T_comsol = T_interp(r)

# Plot solution(s) ------------------------------------------------------------
winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi


# f_i, g_i
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(theta, f1, "-", label="Asymptotic ")
ax[0, 0].plot(theta, f1_comsol, "-", label="COMSOL")
ax[0, 0].set_ylabel(r"$f_1$")
ax[0, 0].legend()
ax[0, 1].plot(theta, f2, "-", label="Asymptotic ")
ax[0, 1].plot(theta, f2_comsol, "-", label="COMSOL")
ax[0, 1].set_ylabel(r"$f_2$")
ax[1, 0].plot(theta, g1, "-", label="Asymptotic ")
ax[1, 0].plot(theta, g1_comsol, "-", label="COMSOL")
ax[1, 0].set_ylabel(r"$g_1$")
ax[1, 1].plot(theta, g2, "-", label="Asymptotic ")
ax[1, 1].plot(theta, g2_comsol, "-", label="COMSOL")
ax[1, 1].set_ylabel(r"$g_2$")
# add shared labels etc.
for ax in ax.reshape(-1):
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
plt.savefig("figs" + path[4:] + "fg_of_theta.pdf", dpi=300)

# displacements
fig, ax = plt.subplots(1, 2)
ax[0].plot(theta, u, "-", label="Asymptotic")
ax[0].plot(theta, u_comsol, "-", label="COMSOL")
ax[0].set_ylabel(r"$u$")
ax[0].legend()
ax[1].plot(theta, v, "-", label="Asymptotic")
ax[1].plot(theta, v_comsol, "-", label="COMSOL")
ax[1].set_ylabel(r"$v$")
# add shared labels etc.
for ax in ax.reshape(-1):
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
plt.savefig("figs" + path[4:] + "uv_of_theta.pdf", dpi=300)

# stresses and strains
fig, ax = plt.subplots(2, 3)
ax[0, 0].plot(theta, e_rr, "-", label="Asymptotic")
ax[0, 0].plot(theta, err_comsol, "-", label="COMSOL")
ax[0, 0].set_ylabel(r"$\epsilon_{rr}$")
ax[0, 0].legend()
ax[0, 1].plot(theta, e_tt, "-", label="Asymptotic")
ax[0, 1].plot(theta, ett_comsol, "-", label="COMSOL")
ax[0, 1].set_ylabel(r"$\epsilon_{\theta\theta}$")
ax[0, 2].plot(theta, e_rt, "-", label="Asymptotic")
ax[0, 2].plot(theta, ert_comsol, "-", label="COMSOL")
ax[0, 2].set_ylabel(r"$\epsilon_{r\theta}$")
ax[1, 0].plot(theta, s_rr, "-", label="Asymptotic")
ax[1, 0].plot(theta, srr_comsol, "-", label="COMSOL")
ax[1, 0].set_ylabel(r"$\sigma_{rr}$")
ax[1, 1].plot(theta, s_tt, "-", label="Asymptotic")
ax[1, 1].plot(theta, stt_comsol, "-", label="COMSOL")
ax[1, 1].set_ylabel(r"$\sigma_{\theta\theta}$")
ax[1, 2].plot(theta, s_rt, "-", label="Asymptotic")
ax[1, 2].plot(theta, srt_comsol, "-", label="COMSOL")
ax[1, 2].set_ylabel(r"$\sigma_{r\theta}$")
# add shared labels etc.
for ax in ax.reshape(-1):
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
plt.savefig("figs" + path[4:] + "stress_strain_of_theta.pdf", dpi=300)

# tension
fig, ax = plt.subplots()
ax.plot(theta, T, "-", label="Asymptotic")
ax.plot(theta, T_comsol, "-", label="COMSOL")
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
plt.savefig("figs" + path[4:] + "T_of_theta.pdf", dpi=300)

plt.show()

# %%
