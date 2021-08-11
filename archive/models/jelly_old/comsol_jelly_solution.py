import numpy as np
from numpy import pi
import pandas as pd
import scipy.interpolate as interp


class ComsolSolution:
    def __init__(
        self,
        r0,
        delta,
        l_n,
        l_s,
        l_p,
        hh,
        N,
        lam_n,
        lam_s,
        lam_p,
        mu_n,
        mu_s,
        mu_p,
        alpha_scale,
        path,
    ):
        """
        Loads the COMSOL solution. The variables are stored as a dict.
        Note that we rescale the displacements, strains and stresses
        by alpha_scale since COMSOL didn't like having alpha=1.
        """
        # dict to store variables
        self._variables = {}

        # compute numerical thicknesses
        h_cn = hh
        h_cp = hh
        h_n = l_n * delta - h_cn / 2
        h_p = l_p * delta - h_cp / 2
        h_s = l_s * delta

        # Note: COMSOL data is (r, f) so we create interpolants to get
        # (theta, f) data
        theta = np.linspace(0, 2 * pi * N, 60 * (N - 1))
        self._variables["theta"] = theta

        # Load f_i and g_i ----------------------------------------------------

        # curves (labelled i) on which the f_i and g_i are evaluated
        r_evals = {
            "11": r0 + h_cp / 2 + h_p / 2 + delta * theta / 2 / pi,
            "12": r0 + h_cp / 2 + delta * theta / 2 / pi,
            "21": r0 + h_cp / 2 + h_p + h_s / 2 + delta * theta / 2 / pi,
            "22": r0 + h_cp / 2 + h_p + delta * theta / 2 / pi,
            "31": r0 + delta / 2 - h_cn / 2 - h_n / 2 + delta * theta / 2 / pi,
            "32": r0 + delta / 2 - h_cn / 2 + delta * theta / 2 / pi,
            "41": r0 + delta / 2 + h_cn / 2 + h_n / 2 + delta * theta / 2 / pi,
            "42": r0 + delta / 2 + h_cn / 2 + delta * theta / 2 / pi,
            "51": r0 + delta - h_cp / 2 - h_p - h_s / 2 + delta * theta / 2 / pi,
            "52": r0 + delta - h_cp / 2 - h_p + delta * theta / 2 / pi,
            "61": r0 + delta - h_cp / 2 - h_p / 2 + delta * theta / 2 / pi,
            "62": r0 + delta - h_cp / 2 + delta * theta / 2 / pi,
        }

        # loop over curves we evaluate on
        for num, r_eval in r_evals.items():
            # set parameters based on region
            if num[0] in ["1", "6"]:
                lam, mu = lam_p, mu_p
            elif num[0] in ["2", "5"]:
                lam, mu = lam_s, mu_s
            elif num[0] in ["3", "4"]:
                lam, mu = lam_n, mu_n

            # evulate the f_i1 and g_i1
            if num[1] == "1":
                # f1 = sigma_rr / (lambda+2*mu)
                comsol = pd.read_csv(
                    path + f"srr_{num[0]}.csv", comment="#", header=None
                ).to_numpy()
                f1_r_data = comsol[:, 0]
                f1_data = comsol[:, 1] / alpha_scale / (lam + 2 * mu)
                f1_interp = interp.interp1d(f1_r_data, f1_data, bounds_error=False)
                self._variables["f_" + num] = f1_interp(r_eval)

                # g1 = sigma_rt/mu
                comsol = pd.read_csv(
                    path + f"srt_{num[0]}.csv", comment="#", header=None
                ).to_numpy()
                g1_r_data = comsol[:, 0]
                g1_data = comsol[:, 1] / alpha_scale / mu
                g1_interp = interp.interp1d(g1_r_data, g1_data, bounds_error=False)
                self._variables["g_" + num] = g1_interp(r_eval)

            # evulate the f_i2 and g_i2
            elif num[1] == "2":
                # f2 = u(R=theta/2/pi)/delta
                comsol = pd.read_csv(
                    path + f"u_{num[0]}.csv", comment="#", header=None
                ).to_numpy()
                f2_r_data = comsol[:, 0]
                f2_data = comsol[:, 1] / alpha_scale / delta
                f2_interp = interp.interp1d(f2_r_data, f2_data, bounds_error=False)
                self._variables["f_" + num] = f2_interp(r_eval)

                # g2 = v(R=theta/2/pi)/delta
                comsol = pd.read_csv(
                    path + f"v_{num[0]}.csv", comment="#", header=None
                ).to_numpy()
                g2_r_data = comsol[:, 0]
                g2_data = comsol[:, 1] / alpha_scale / delta
                g2_interp = interp.interp1d(g2_r_data, g2_data, bounds_error=False)
                self._variables["g_" + num] = g2_interp(r_eval)

        # Load tension --------------------------------------------------------

        # In COMSOL we evaluate the tension in the positive current collector
        # at three points:
        # ra = r0-h_cp/2+delta*theta/2/pi
        # rb = r0+delta*theta/2/pi
        # rc = r0+h_cp/2+delta*theta/2/pi
        # and the tension in the negative current collector
        # at three points:
        # ra = r0+delta/2-h_cn/2+delta*theta/2/pi
        # rb = r0+delta/2+delta*theta/2/pi
        # rc = r0+delta/2+h_cn/2+delta*theta/2/pi
        r_evals = {
            "Tp": {
                "a": r0 - h_cp / 2 + delta * theta / 2 / pi,
                "b": r0 + delta * theta / 2 / pi,
                "c": r0 + h_cp / 2 + delta * theta / 2 / pi,
            },
            "Tn": {
                "a": r0 + delta / 2 - h_cn / 2 + delta * theta / 2 / pi,
                "b": r0 + delta / 2 + delta * theta / 2 / pi,
                "c": r0 + delta / 2 + h_cn / 2 + delta * theta / 2 / pi,
            },
        }

        # loop over tensions
        for T, rs in r_evals.items():
            # loop over r values
            for label, r in rs.items():
                comsol = pd.read_csv(
                    path + f"{T}_{label}.csv", comment="#", header=None
                ).to_numpy()
                T_r_data = comsol[:, 0]
                T_data = comsol[:, 1] / alpha_scale
                T_interp = interp.interp1d(T_r_data, T_data, bounds_error=False)
                self._variables[f"{T} {label}"] = T_interp(r)

            # compute tension using Simpson's rule
            self._variables[f"{T}"] = (
                self._variables[f"{T} a"]
                + 4 * self._variables[f"{T} b"]
                + self._variables[f"{T} c"]
            ) / 6

    def __getitem__(self, T):
        return self._variables[T]