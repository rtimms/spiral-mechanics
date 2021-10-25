import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


def plot_fg(outer, comsol, N_plot, path):
    theta = comsol.theta

    fig, ax = plt.subplots(2, 3, figsize=(6.4, 4))
    ax[0, 0].plot(theta, outer.f1(theta), "-", label="Asymptotic")
    ax[0, 0].plot(theta, comsol.f1, "--", label="COMSOL")
    ax[0, 0].set_ylabel(r"$f_{{1,3}}$")
    ax[0, 1].plot(theta, outer.f2(theta), "-", label="Asymptotic")
    ax[0, 1].plot(theta, comsol.f2, "--", label="COMSOL")
    ax[0, 1].set_ylabel(r"$f_2$")
    ax[0, 2].plot(theta, outer.f4(theta), "-", label="Asymptotic")
    ax[0, 2].plot(theta, comsol.f4, "--", label="COMSOL")
    ax[0, 2].set_ylabel(r"$f_4$")
    ax[1, 0].plot(theta, outer.g1(theta), "-", label="Asymptotic")
    ax[1, 0].plot(theta, comsol.g1, "--", label="COMSOL")
    ax[1, 0].set_ylabel(r"$g_{{1,3}}$")
    ax[1, 1].plot(theta, outer.g2(theta), "-", label="Asymptotic")
    ax[1, 1].plot(theta, comsol.g2, "--", label="COMSOL")
    ax[1, 1].set_ylabel(r"$g_2$")
    ax[1, 2].plot(theta, outer.g4(theta), "-", label="Asymptotic")
    ax[1, 2].plot(theta, comsol.g4, "--", label="COMSOL")
    ax[1, 2].set_ylabel(r"$g_4$")
    # add shared labels etc.
    fig.subplots_adjust(
        left=0.1, bottom=0.25, right=0.98, top=0.98, wspace=0.45, hspace=0.45
    )
    ax[1, 1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        borderaxespad=0.0,
        ncol=2,
    )
    winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi
    for ax in ax.reshape(-1):
        for w in winds:
            ax.axvline(x=w, linestyle=":", color="lightgrey")
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda val, pos: r"${}\pi$".format(int(val / np.pi))
                if val != 0
                else "0"
            )
        )
        ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
        ax.set_xlim([0, N_plot * 2 * pi])
        ax.set_xlabel(r"$\theta$")
    plt.savefig("figs" + path[4:] + "fg_of_theta_jelly.pdf", dpi=300)


def plot_tension(outer, comsol, N_plot, path):
    theta = comsol.theta

    fig, ax = plt.subplots(3, 1, figsize=(6.4, 4))
    ax[0].plot(theta, outer.Tp(theta), "-", label="Asymptotic")
    ax[0].plot(theta, comsol.Tp, "--", label="COMSOL")
    ax[0].set_ylabel(r"$T_+$")
    ax[1].plot(theta, outer.Tn(theta), "-", label="Asymptotic")
    ax[1].plot(theta, comsol.Tn, "--", label="COMSOL")
    ax[1].set_ylabel(r"$T_-$")
    ax[1].legend(loc="lower right")
    ax[2].plot(theta, outer.Tn(theta) + outer.Tp(theta), "-", label="Asymptotic")
    ax[2].plot(theta, comsol.Tn + comsol.Tp, "--", label="COMSOL")
    ax[2].set_ylabel(r"$T_-+T_+$")
    # add shared labels etc.
    winds = [2 * pi * n for n in list(range(N_plot))]  # plot dashed line every 2*pi
    for ax in ax.reshape(-1):
        for w in winds:
            ax.axvline(x=w, linestyle=":", color="lightgrey")
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda val, pos: r"${}\pi$".format(int(val / np.pi))
                if val != 0
                else "0"
            )
        )
        ax.xaxis.set_major_locator(MultipleLocator(base=4 * pi))
        ax.set_xlim([0, N_plot * 2 * pi])
        ax.set_xlabel(r"$\theta$")
    plt.tight_layout()
    plt.savefig("figs" + path[4:] + "T_of_theta_jelly.pdf", dpi=300)
