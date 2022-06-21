import numpy as np
from numpy import pi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from composite_solution import OuterSolution
from comsol_solution import ComsolSolution, ComsolInnerSolution

# Parameters (dimensionless) --------------------------------------------------
alpha = 1  # expansion coefficient
alpha_cc = 0  # expansion coefficient
mu = 1  # shear modulus
nu = 1 / 3  # Poisson ratio
lam = 2 * mu * nu / (1 - 2 * nu)  # 1st Lame parameter
omega = np.sqrt(mu / (lam + 2 * mu))
c = alpha * (2 * lam + mu) * omega
N = 10  # number of winds
r0 = 0.25  # inner radius
r1 = 1  # outer radius
delta = (r1 - r0) / N
hh = 0.01 * delta  # current collector thickness
N_BL = 5  # number of slabs in inner solution
N_plot = N_BL - 1  # number of winds to plot
path = "data/inner_a1al0/"  # path to inner simulation data
full_path = "data/a1al0/"  # path to full simulation data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the outer solution --------------------------------------------------
outer = OuterSolution(r0, delta, mu, lam, alpha, alpha_cc)
# unpack
f1, f2, g1, g2 = outer.f1, outer.f2, outer.g1, outer.g2
u, v = outer.u, outer.v
e_rr, e_tt, e_rt = outer.e_rr, outer.e_tt, outer.e_rt
s_rr, s_tt, s_rt = outer.s_rr, outer.s_tt, outer.s_rt


# Load COMSOL data ------------------------------------------------------------
inner = ComsolInnerSolution(path)
# unpack
t_data = inner.u_t_data
u_data, v_data = inner.u_data, inner.v_data
err_data, ett_data, ert_data = inner.err_data, inner.ett_data, inner.ert_data
srr_data, stt_data, srt_data = inner.srr_data, inner.stt_data, inner.srt_data

# outer (full simulation)
comsol = ComsolSolution(r0, delta, hh, N, mu, lam, alpha, alpha_cc, full_path)

# Plots -----------------------------------------------------------------------

# displacements at r = r0 + delta*theta/2/pi, 0 < theta < 8*pi
fig, ax = plt.subplots(2, 1, figsize=(6.4, 4), sharex=False)
# plot leading (outer) solutions
theta = comsol.theta
r = r0 + delta * theta / 2 / pi
ax[0].plot(
    theta,
    u(r, theta),
    linestyle="-",
    label="Surface layer solution",
)
ax[1].plot(
    theta,
    v(r, theta),
    linestyle="-",
    label="Surface layer solution",
)
# plot COMSOL solutions
theta = comsol.theta
# u = alpha_cc*r0 + delta*f2
ax[0].plot(
    theta,
    comsol.f2 * delta + alpha_cc * r0,
    linestyle="--",
    color="tab:orange",
    label="COMSOL",
)
# v = delta*g2
ax[1].plot(theta, delta * comsol.g2, linestyle="--", color="tab:orange", label="COMSOL")
ax[0].set_ylabel(r"$u$")
ax[1].set_ylabel(r"$v$")
ax[0].set_xlabel(r"$\theta$")
ax[1].set_xlabel(r"$\theta$")
ax[1].set_ylim([-4.5 * delta, 0.5 * delta])
fig.subplots_adjust(left=0.1, bottom=0.3, right=0.98, top=0.98, wspace=0.33, hspace=0.4)
ax[1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.5),
    borderaxespad=0.0,
    ncol=3,
)
for ax in ax.reshape(-1):
    # plot dashed line every 2*pi
    winds = [2 * pi * n for n in list(range(N_plot))]
    for w in winds:
        ax.axvline(x=w, linestyle="-", color="lightgrey")
    # add labels etc.
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
plt.savefig("figs" + path[4:] + "uv_composite.png", dpi=300)

# stresses and strains at r = r0 + delta / 2 + delta * theta / 2 / pi
fig, ax = plt.subplots(3, 2, figsize=(6.4, 6))

# plot leading (outer) solutions
theta = comsol.theta
r = r0 + delta / 2 + delta * theta / 2 / pi
ax[0, 0].plot(theta, e_rr(theta), linestyle="-", label="Surface layer solution")
ax[0, 1].plot(theta, s_rr(theta), linestyle="-", label="Surface layer solution")
ax[1, 0].plot(
    theta,
    e_tt(r, theta),
    linestyle="-",
    label="Surface layer solution",
)
ax[1, 1].plot(theta, s_tt(theta), linestyle="-", label="Surface layer solution")
ax[2, 0].plot(theta, e_rt(theta), linestyle="-", label="Surface layer solution")
ax[2, 1].plot(theta, s_rt(theta), linestyle="-", label="Surface layer solution")


# plot COMSOL solutions
theta = comsol.theta
r = r0 + delta / 2 + delta * theta / 2 / pi
ax[0, 0].plot(theta, comsol.err, linestyle="--", color="tab:orange", label="COMSOL")
ax[1, 0].plot(theta, comsol.ett, linestyle="--", color="tab:orange", label="COMSOL")
ax[2, 0].plot(theta, comsol.ert, linestyle="--", color="tab:orange", label="COMSOL")
ax[0, 1].plot(theta, comsol.srr, linestyle="--", color="tab:orange", label="COMSOL")
ax[1, 1].plot(theta, comsol.stt, linestyle="--", color="tab:orange", label="COMSOL")
ax[2, 1].plot(theta, comsol.srt, linestyle="--", color="tab:orange", label="COMSOL")
# add shared labels etc.
ax[0, 0].set_ylabel(r"$\varepsilon_{rr}$")
ax[1, 0].set_ylabel(r"$\varepsilon_{\theta\theta}$")
ax[2, 0].set_ylabel(r"$\varepsilon_{r\theta}$")
ax[0, 1].set_ylabel(r"$\sigma_{rr}$")
ax[1, 1].set_ylabel(r"$\sigma_{\theta\theta}$")
ax[2, 1].set_ylabel(r"$\sigma_{r\theta}$")
fig.subplots_adjust(
    left=0.1, bottom=0.22, right=0.98, top=0.98, wspace=0.25, hspace=0.4
)
ax[2, 0].legend(
    loc="upper center",
    bbox_to_anchor=(1.1, -0.5),
    borderaxespad=0.0,
    ncol=3,
)
for ax in ax.reshape(-1):
    for w in winds:
        ax.axvline(x=w, linestyle="-", color="lightgrey")
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
    ax.set_xlabel(r"$\theta$")
plt.savefig("figs" + path[4:] + "eps_sigma_composite.png", dpi=300)


# tension
fig, ax = plt.subplots(figsize=(6.4, 2))
ax.plot(theta, outer.T(theta), "-", label="Surface layer solution")
ax.plot(theta, comsol.T, "--", label="COMSOL")
ax.set_ylabel(r"$T$")
ax.legend(loc="lower right")
# add shared labels etc.
for w in winds:
    ax.axvline(x=w, linestyle="-", color="lightgrey")
ax.xaxis.set_major_formatter(
    FuncFormatter(
        lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
    )
)
ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
ax.set_xlim([0, N_plot * 2 * pi])
ax.set_ylim([-2, 0.2])
ax.set_xlabel(r"$\theta$")
plt.tight_layout()
plt.savefig("figs" + path[4:] + "T_of_theta.png", dpi=300)

plt.show()
