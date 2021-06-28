import numpy as np
from numpy import pi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
from outer_solution import OuterSolution
from comsol_solution import ComsolSolution, ComsolInnerSolution

# set style for paper
matplotlib.rc_file("_matplotlibrc_tex", use_default_template=True)

# Parameters (dimensionless) --------------------------------------------------
alpha = 1  # expansion coefficient
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
path = "data/boundary_layer/"  # path to inner simulation data
full_path = "data/single/mu1lam2/"  # path to full simulation data
# make directory for figures if it doesn't exist
try:
    os.mkdir("figs" + path[4:])
except FileExistsError:
    pass

# Compute the outer solution --------------------------------------------------
outer = OuterSolution(r0, delta, mu, lam, alpha)
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

# load f_i and g_i from full simulation
alpha_scale = 0.1
comsol = ComsolSolution(r0, delta, hh, N, mu, lam, alpha_scale, full_path)


# Plots -----------------------------------------------------------------------

# f_i and g_i
fig, ax = plt.subplots(2, 2)
# plot COMSOL solutions
theta = comsol.theta
ax[0, 0].plot(theta, comsol.f1, linestyle="--", color="tab:orange", label="COMSOL")
ax[0, 1].plot(theta, comsol.f2, linestyle="--", color="tab:orange", label="COMSOL")
ax[1, 0].plot(theta, comsol.g1, linestyle="--", color="tab:orange", label="COMSOL")
ax[1, 1].plot(theta, comsol.g2, linestyle="--", color="tab:orange", label="COMSOL")
# plot outer solutions
ax[0, 0].plot(theta, f1(theta), linestyle=":", color="black", label="Outer")
ax[0, 1].plot(theta, f2(theta), linestyle=":", color="black", label="Outer")
ax[1, 0].plot(theta, g1(theta), linestyle=":", color="black", label="Outer")
ax[1, 1].plot(theta, g2(theta), linestyle=":", color="black", label="Outer")
# plot inner and composite solutions
for n in range(N_BL):
    idx1 = int(n * 100 / N_BL + 10)  # midpoint
    idx2 = int(n * 100 / N_BL)  # inner edge
    if n == 0:
        # 0 < theta < inf
        Theta = t_data[0, 50:]
        srr_tilde = srr_data[idx1, 50:]
        u_tilde = u_data[idx2, 50:]
        v_tilde = v_data[idx2, 50:]
        srt_tilde = srt_data[idx1, 50:]
    else:
        # -inf < theta < inf
        Theta = t_data[0, :]
        srr_tilde = srr_data[idx1, :]
        u_tilde = u_data[idx2, :]
        v_tilde = v_data[idx2, :]
        srt_tilde = srt_data[idx1, :]

    theta = delta * Theta / r0 + 2 * n * pi
    # f1 = sigma_rr / (lambda+2*mu)
    ax[0, 0].plot(
        theta,
        c * srr_tilde / (lam + 2 * mu) + f1(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[0, 0].plot(
        theta,
        c * srr_tilde / (lam + 2 * mu) + f1(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    # f2 = u(R=theta/2/pi)/delta
    ax[0, 1].plot(
        theta,
        c * u_tilde + f2(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[0, 1].plot(
        theta,
        c * u_tilde + f2(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    # g1 = sigma_rt/mu
    ax[1, 0].plot(
        theta,
        c * srt_tilde / mu + g1(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[1, 0].plot(
        theta,
        c * srt_tilde / mu + g1(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    # g2 = v(R=theta/2/pi)/delta
    ax[1, 1].plot(
        theta,
        c * v_tilde + g2(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[1, 1].plot(
        theta,
        c * v_tilde + g2(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
ax[0, 0].set_ylabel(r"$f_1$")
ax[0, 1].set_ylabel(r"$f_2$")
ax[1, 0].set_ylabel(r"$g_1$")
ax[1, 1].set_ylabel(r"$g_2$")
ax[0, 0].legend(loc="upper right")
for ax in ax.reshape(-1):
    # plot dashed line every 2*pi
    winds = [2 * pi * n for n in list(range(N_plot))]
    for w in winds:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    # add labels etc.
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
    ax.set_xlabel(r"$\theta$")
plt.tight_layout()
plt.savefig("figs" + path[4:] + "fg_composite.pdf", dpi=300)

# stresses and strains at r = r0 + delta / 2 + delta * theta / 2 / pi
fig, ax = plt.subplots(2, 3)
# plot COMSOL solutions
theta = comsol.theta
r = r0 + delta / 2 + delta * theta / 2 / pi
ax[0, 0].plot(theta, comsol.err, linestyle="--", color="tab:orange", label="COMSOL")
ax[0, 1].plot(theta, comsol.ett, linestyle="--", color="tab:orange", label="COMSOL")
ax[0, 2].plot(theta, comsol.ert, linestyle="--", color="tab:orange", label="COMSOL")
ax[1, 0].plot(theta, comsol.srr, linestyle="--", color="tab:orange", label="COMSOL")
ax[1, 1].plot(theta, comsol.stt, linestyle="--", color="tab:orange", label="COMSOL")
ax[1, 2].plot(theta, comsol.srt, linestyle="--", color="tab:orange", label="COMSOL")
# plot outer solutions
ax[0, 0].plot(theta, e_rr(theta), linestyle=":", color="black", label="Outer")
ax[0, 1].plot(theta, e_tt(r, theta), linestyle=":", color="black", label="Outer")
ax[0, 2].plot(theta, e_rt(theta), linestyle=":", color="black", label="Outer")
ax[1, 0].plot(theta, s_rr(theta), linestyle=":", color="black", label="Outer")
ax[1, 1].plot(theta, s_tt(theta), linestyle=":", color="black", label="Outer")
ax[1, 2].plot(theta, s_rt(theta), linestyle=":", color="black", label="Outer")
# plot composite solutions
for n in range(N_BL):
    idx1 = int(n * 100 / N_BL + 10)  # midpoint
    idx2 = int(n * 100 / N_BL)  # inner edge
    if n == 0:
        # 0 < theta < inf
        Theta = t_data[0, 50:]
        u_tilde = u_data[idx2, 50:]
        v_tilde = v_data[idx2, 50:]
        err_tilde = err_data[idx1, 50:]
        ett_tilde = ett_data[idx1, 50:]
        ert_tilde = ert_data[idx1, 50:]
        srr_tilde = srr_data[idx1, 50:]
        stt_tilde = stt_data[idx1, 50:]
        srt_tilde = srt_data[idx1, 50:]
    else:
        # -inf < theta < inf
        Theta = t_data[0, :]
        u_tilde = u_data[idx2, :]
        v_tilde = v_data[idx2, :]
        err_tilde = err_data[idx1, :]
        ett_tilde = ett_data[idx1, :]
        ert_tilde = ert_data[idx1, :]
        srr_tilde = srr_data[idx1, :]
        stt_tilde = stt_data[idx1, :]
        srt_tilde = srt_data[idx1, :]

    theta = delta * Theta / r0 + 2 * n * pi
    # e_rr
    ax[0, 0].plot(
        theta,
        c * err_tilde + alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + f1(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[0, 0].plot(
        theta,
        c * err_tilde + alpha * (3 * lam + 2 * mu) / (lam + 2 * mu) + f1(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    # e_tt
    ax[0, 1].plot(
        theta,
        c * ett_tilde,
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[0, 1].plot(
        theta,
        c * ett_tilde + e_tt(r0 + delta / 2 + delta * theta / 2 / pi, theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    # e_rt
    ax[0, 2].plot(
        theta,
        c * ert_tilde + g1(2 * n * pi) / 2,
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[0, 2].plot(
        theta,
        c * ert_tilde + g1(theta) / 2,
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    # s_rr
    ax[1, 0].plot(
        theta,
        c * srr_tilde + (lam + 2 * mu) * f1(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[1, 0].plot(
        theta,
        c * srr_tilde + (lam + 2 * mu) * f1(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    # s_tt
    ax[1, 1].plot(
        theta,
        c * stt_tilde
        - 2 * mu * alpha * (3 * lam + 2 * mu) / (lam + 2 * mu)
        + lam * f1(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[1, 1].plot(
        theta,
        c * stt_tilde
        - 2 * mu * alpha * (3 * lam + 2 * mu) / (lam + 2 * mu)
        + lam * f1(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
    # s_rt
    ax[1, 2].plot(
        theta,
        c * srt_tilde + mu * g1(2 * n * pi),
        linestyle="-.",
        color="tab:green",
        label="Inner" if n == 0 else "",
    )
    ax[1, 2].plot(
        theta,
        c * srt_tilde + mu * g1(theta),
        linestyle="-",
        color="tab:blue",
        label="Composite" if n == 0 else "",
    )
# add shared labels etc.
ax[0, 0].set_ylabel(r"$\varepsilon_{rr}$")
ax[0, 1].set_ylabel(r"$\varepsilon_{\theta\theta}$")
ax[0, 2].set_ylabel(r"$\varepsilon_{r\theta}$")
ax[1, 0].set_ylabel(r"$\sigma_{rr}$")
ax[1, 1].set_ylabel(r"$\sigma_{\theta\theta}$")
ax[1, 2].set_ylabel(r"$\sigma_{r\theta}$")
ax[0, 0].legend(loc="upper right")
for ax in ax.reshape(-1):
    for w in winds:
        ax.axvline(x=w, linestyle=":", color="lightgrey")
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: r"${}\pi$".format(int(val / np.pi)) if val != 0 else "0"
        )
    )
    ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
    ax.set_xlim([0, N_plot * 2 * pi])
    ax.set_xlabel(r"$\theta$")
plt.tight_layout()
plt.savefig("figs" + path[4:] + "stress_strain_ocomposite.pdf", dpi=300)

plt.show()
