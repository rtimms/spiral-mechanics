import numpy as np
from numpy import pi
import pandas as pd
import scipy.interpolate as interp


class ComsolSolution:
    def __init__(self, params, alpha_scale, path):
        """
        Loads the COMSOL solution. The variables are stored as attributes of
        the class. Note that we rescale the displacements, strains and stresses
        by alpha_scale since COMSOL didn't like having alpha=1.
        """
        # Unpack parameters
        r0, delta, hh, N = params.r0, params.delta, params.hh, params.N
        l_n, l_s, l_p = params.l_n, params.l_s, params.l_p
        # lam_n, lam_s, lam_p = params.lam_n, params.lam_s, params.lam_p
        # mu_n, mu_s, mu_p = params.mu_n, params.mu_s, params.mu_p
        mu_p = params.mu_p

        # Compute numerical thicknesses
        h_cn = hh
        h_cp = hh
        h_n = l_n * delta - h_cn / 2
        h_p = l_p * delta - h_cp / 2
        h_s = l_s * delta

        # Note: COMSOL data is (r, f) so we create interpolants to get
        # (theta, f) data
        theta = np.linspace(0, 2 * pi * N, 60 * (N - 1))
        self.theta = theta

        # Dict of evaluation position (stresses are evaluated in the middle of
        # each layer and displacements at the inside edge)
        # keys use a+, as, a- to refer to the positive electrode, separator,
        # negative electrode in active region a = 1,2, and "mid" or "in" to
        # refer to the middle or inside edge of the layer
        r_evals = {
            "1+ in": r0 + h_cp / 2 + delta * theta / 2 / pi,
            "1+ mid": r0 + h_cp / 2 + h_p / 2 + delta * theta / 2 / pi,
            "1s in": r0 + h_cp / 2 + h_p + delta * theta / 2 / pi,
            "1s mid": r0 + h_cp / 2 + h_p + h_s / 2 + delta * theta / 2 / pi,
            "1- in": r0 + delta / 2 - h_cn / 2 + delta * theta / 2 / pi,
            "1- mid": r0 + delta / 2 - h_cn / 2 - h_n / 2 + delta * theta / 2 / pi,
            "2- in": r0 + delta / 2 + h_cn / 2 + delta * theta / 2 / pi,
            "2- mid": r0 + delta / 2 + h_cn / 2 + h_n / 2 + delta * theta / 2 / pi,
            "2s in": r0 + delta - h_cp / 2 - h_p + delta * theta / 2 / pi,
            "2s mid": r0 + delta - h_cp / 2 - h_p - h_s / 2 + delta * theta / 2 / pi,
            "2+ in": r0 + delta - h_cp / 2 + delta * theta / 2 / pi,
            "2+ mid": r0 + delta - h_cp / 2 - h_p / 2 + delta * theta / 2 / pi,
        }

        # Load f_i and g_i ----------------------------------------------------

        # f_1 = sigma_rr
        # we choose to evaluate sigma_rr in the positive electrode in active
        # region 1
        comsol = pd.read_csv(path + "srr_1.csv", comment="#", header=None).to_numpy()
        f1_r_data = comsol[:, 0]
        f1_data = comsol[:, 1] / alpha_scale
        f1_interp = interp.interp1d(f1_r_data, f1_data, bounds_error=False)
        self.f1 = f1_interp(r_evals["1+ mid"])
        self.f3 = self.f1  # f_1 = f_3

        # f_2 = u(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "u_1.csv", comment="#", header=None).to_numpy()
        f2_r_data = comsol[:, 0]
        f2_data = comsol[:, 1] / delta / alpha_scale
        f2_interp = interp.interp1d(f2_r_data, f2_data, bounds_error=False)
        self.f2 = f2_interp(r_evals["1+ in"])

        # f_4 = u(R=1/2+theta/2/pi)/delta
        comsol = pd.read_csv(path + "u_4.csv", comment="#", header=None).to_numpy()
        f4_r_data = comsol[:, 0]
        f4_data = comsol[:, 1] / delta / alpha_scale
        f4_interp = interp.interp1d(f4_r_data, f4_data, bounds_error=False)
        self.f4 = f4_interp(r_evals["2- in"])

        # g_1 = sigma_rt/mu
        # we choose to evaluate sigma_rt in the positive electrode in active
        # region 1
        comsol = pd.read_csv(path + "srt_1.csv", comment="#", header=None).to_numpy()
        g1_r_data = comsol[:, 0]
        g1_data = comsol[:, 1] / mu_p / alpha_scale
        g1_interp = interp.interp1d(g1_r_data, g1_data, bounds_error=False)
        self.g1 = g1_interp(r_evals["1+ mid"])
        self.g3 = self.g1  # g_1 = g_3

        # g_2 = v(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "v_1.csv", comment="#", header=None).to_numpy()
        g2_r_data = comsol[:, 0]
        g2_data = comsol[:, 1] / delta / alpha_scale
        g2_interp = interp.interp1d(g2_r_data, g2_data, bounds_error=False)
        self.g2 = g2_interp(r_evals["1+ in"])

        # g_4 = v(R=1/2+theta/2/pi)/delta
        comsol = pd.read_csv(path + "v_4.csv", comment="#", header=None).to_numpy()
        g4_r_data = comsol[:, 0]
        g4_data = comsol[:, 1] / delta / alpha_scale
        g4_interp = interp.interp1d(g4_r_data, g4_data, bounds_error=False)
        self.g4 = g4_interp(r_evals["2- in"])

        # Load tension --------------------------------------------------------

        # In COMSOL we evaluate the tension in the positive current collector
        # at three points:
        # ra = r0-h_cp/2+delta*theta/2/pi
        # rb = r0+delta*theta/2/pi
        # rc = r0+h_cp/2+delta*theta/2/pi
        ra = r0 - h_cp / 2 + delta * theta / 2 / pi
        rb = r0 + delta * theta / 2 / pi
        rc = r0 + h_cp / 2 + delta * theta / 2 / pi

        comsola = pd.read_csv(path + "Tp_a.csv", comment="#", header=None).to_numpy()
        Tp_r_dataa = comsola[:, 0]
        Tp_dataa = comsola[:, 1] / alpha_scale
        Tp_interpa = interp.interp1d(Tp_r_dataa, Tp_dataa, bounds_error=False)
        self.Tp_a = Tp_interpa(ra)

        comsolb = pd.read_csv(path + "Tp_b.csv", comment="#", header=None).to_numpy()
        Tp_r_datab = comsolb[:, 0]
        Tp_datab = comsolb[:, 1] / alpha_scale
        Tp_interpb = interp.interp1d(Tp_r_datab, Tp_datab, bounds_error=False)
        self.Tp_b = Tp_interpb(rb)

        comsolc = pd.read_csv(path + "Tp_c.csv", comment="#", header=None).to_numpy()
        Tp_r_datac = comsolc[:, 0]
        Tp_datac = comsolc[:, 1] / alpha_scale
        Tp_interpc = interp.interp1d(Tp_r_datac, Tp_datac, bounds_error=False)
        self.Tp_c = Tp_interpc(rc)

        # compute tension using simpsons rule
        self.Tp = (self.Tp_a + 4 * self.Tp_b + self.Tp_c) / 6

        # and the tension in the negative current collector
        # at three points:
        # ra = r0+delta/2-h_cn/2+delta*theta/2/pi
        # rb = r0+delta/2+delta*theta/2/pi
        # rc = r0+delta/2+h_cn/2+delta*theta/2/pi
        ra = r0 + delta / 2 - h_cn / 2 + delta * theta / 2 / pi
        rb = r0 + delta / 2 + delta * theta / 2 / pi
        rc = r0 + delta / 2 + h_cn / 2 + delta * theta / 2 / pi

        comsola = pd.read_csv(path + "Tn_a.csv", comment="#", header=None).to_numpy()
        Tn_r_dataa = comsola[:, 0]
        Tn_dataa = comsola[:, 1] / alpha_scale
        Tn_interpa = interp.interp1d(Tn_r_dataa, Tn_dataa, bounds_error=False)
        self.Tn_a = Tn_interpa(ra)

        comsolb = pd.read_csv(path + "Tn_b.csv", comment="#", header=None).to_numpy()
        Tn_r_datab = comsolb[:, 0]
        Tn_datab = comsolb[:, 1] / alpha_scale
        Tn_interpb = interp.interp1d(Tn_r_datab, Tn_datab, bounds_error=False)
        self.Tn_b = Tn_interpb(rb)

        comsolc = pd.read_csv(path + "Tn_c.csv", comment="#", header=None).to_numpy()
        Tn_r_datac = comsolc[:, 0]
        Tn_datac = comsolc[:, 1] / alpha_scale
        Tn_interpc = interp.interp1d(Tn_r_datac, Tn_datac, bounds_error=False)
        self.Tn_c = Tn_interpc(rc)

        # compute tension using simpsons rule
        self.Tn = (self.Tn_a + 4 * self.Tn_b + self.Tn_c) / 6
