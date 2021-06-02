import numpy as np
from numpy import pi
import pandas as pd
import scipy.interpolate as interp


class ComsolSolution:
    def __init__(self, r0, delta, l_n, l_p, hh, N, E_n, E_p, nu_n, nu_p, path):
        # compute numerical thicknesses
        h_cn = hh
        h_cp = hh
        h_n = l_n * delta - h_cn / 2
        h_p = l_p * delta - h_cp / 2

        # compute lame parameters
        lam_n = E_n * nu_n / (1 + nu_n) / (1 - 2 * nu_n)
        mu_n = E_n / 2 / (1 + nu_n)
        lam_p = E_p * nu_p / (1 + nu_p) / (1 - 2 * nu_p)
        mu_p = E_p / 2 / (1 + nu_p)

        # Note: COMSOL data is (r, f) so we create interpolants to get
        # (theta, f) data
        theta = np.linspace(0, 2 * pi * N, 60 * (N - 1))
        self.theta = theta

        # f1 = sigma_rr / (lambda+2*mu)
        comsol = pd.read_csv(path + "srr_1.csv", comment="#", header=None).to_numpy()
        f11_r_data = comsol[:, 0]
        f11_data = comsol[:, 1] / (lam_p + 2 * mu_p)
        f11_interp = interp.interp1d(f11_r_data, f11_data, bounds_error=False)
        # In COMSOL we evaluate f_11 at r = r0+h_cp/2+h_p/2+delta*theta/2/pi
        r = r0 + h_cp / 2 + h_p / 2 + delta * theta / 2 / pi
        self.f11 = f11_interp(r)

        # f2 = u(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "u_1.csv", comment="#", header=None).to_numpy()
        f12_r_data = comsol[:, 0]
        f12_data = comsol[:, 1] / delta
        f12_interp = interp.interp1d(f12_r_data, f12_data, bounds_error=False)
        # In COMSOL we evaluate f_12 at r = r0+h_cp/2+delta*theta/2/pi
        r = r0 + h_cp / 2 + delta * theta / 2 / pi
        self.f12 = f12_interp(r)

        # g1 = sigma_rt/mu
        comsol = pd.read_csv(path + "srt_1.csv", comment="#", header=None).to_numpy()
        g11_r_data = comsol[:, 0]
        g11_data = comsol[:, 1] / mu_p
        g11_interp = interp.interp1d(g11_r_data, g11_data, bounds_error=False)
        # In COMSOL we evaluate g_11 at r = r0+h_cp/2+h_p/2+delta*theta/2/pi
        r = r0 + h_cp / 2 + h_p / 2 + delta * theta / 2 / pi
        self.g11 = g11_interp(r)

        # g2 = v(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "v_1.csv", comment="#", header=None).to_numpy()
        g12_r_data = comsol[:, 0]
        g12_data = comsol[:, 1] / delta
        g12_interp = interp.interp1d(g12_r_data, g12_data, bounds_error=False)
        # In COMSOL we evaluate g_12 at r = r0+h_cp/2+delta*theta/2/pi
        r = r0 + h_cp / 2 + delta * theta / 2 / pi
        self.g12 = g12_interp(r)

        # f1 = sigma_rr / (lambda+2*mu)
        comsol = pd.read_csv(path + "srr_2.csv", comment="#", header=None).to_numpy()
        f21_r_data = comsol[:, 0]
        f21_data = comsol[:, 1] / (lam_n + 2 * mu_n)
        f21_interp = interp.interp1d(f21_r_data, f21_data, bounds_error=False)
        # In COMSOL we evaluate f_21 at r = r0+delta/2-h_cn/2-h_n/2+delta*theta/2/pi
        r = r0 + delta / 2 - h_cn / 2 - h_n / 2 + delta * theta / 2 / pi
        self.f21 = f21_interp(r)

        # f2 = u(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "u_2.csv", comment="#", header=None).to_numpy()
        f22_r_data = comsol[:, 0]
        f22_data = comsol[:, 1] / delta
        f22_interp = interp.interp1d(f22_r_data, f22_data, bounds_error=False)
        # In COMSOL we evaluate f_22 at r = r0+delta/2-h_cn/2+delta*theta/2/pi
        r = r0 + delta / 2 - h_cn / 2 + delta * theta / 2 / pi
        self.f22 = f22_interp(r)

        # g1 = sigma_rt/mu
        comsol = pd.read_csv(path + "srt_2.csv", comment="#", header=None).to_numpy()
        g21_r_data = comsol[:, 0]
        g21_data = comsol[:, 1] / mu_n
        g21_interp = interp.interp1d(g21_r_data, g21_data, bounds_error=False)
        # In COMSOL we evaluate g_21 at r = r0+delta/2-h_cn/2-h_n/2+delta*theta/2/pi
        r = r0 + delta / 2 - h_cn / 2 - h_n / 2 + delta * theta / 2 / pi
        self.g21 = g21_interp(r)

        # g2 = v(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "v_2.csv", comment="#", header=None).to_numpy()
        g22_r_data = comsol[:, 0]
        g22_data = comsol[:, 1] / delta
        g22_interp = interp.interp1d(g22_r_data, g22_data, bounds_error=False)
        # In COMSOL we evaluate g_22 at r = r0+delta/2-h_cn/2+delta*theta/2/pi
        r = r0 + delta / 2 - h_cn / 2 + delta * theta / 2 / pi
        self.g22 = g22_interp(r)

        # f1 = sigma_rr / (lambda+2*mu)
        comsol = pd.read_csv(path + "srr_3.csv", comment="#", header=None).to_numpy()
        f31_r_data = comsol[:, 0]
        f31_data = comsol[:, 1] / (lam_n + 2 * mu_n)
        f31_interp = interp.interp1d(f31_r_data, f31_data, bounds_error=False)
        # In COMSOL we evaluate f_31 at r = r0+delta/2+h_cn/2+h_n/2+delta*theta/2/pi
        r = r0 + delta / 2 + h_cn / 2 + h_n / 2 + delta * theta / 2 / pi
        self.f31 = f31_interp(r)

        # f2 = u(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "u_3.csv", comment="#", header=None).to_numpy()
        f32_r_data = comsol[:, 0]
        f32_data = comsol[:, 1] / delta
        f32_interp = interp.interp1d(f32_r_data, f32_data, bounds_error=False)
        # In COMSOL we evaluate f_32 at r = r0+delta/2+h_cn/2+delta*theta/2/pi
        r = r0 + delta / 2 + h_cn / 2 + delta * theta / 2 / pi
        self.f32 = f32_interp(r)

        # g1 = sigma_rt/mu
        comsol = pd.read_csv(path + "srt_3.csv", comment="#", header=None).to_numpy()
        g31_r_data = comsol[:, 0]
        g31_data = comsol[:, 1] / mu_n
        g31_interp = interp.interp1d(g31_r_data, g31_data, bounds_error=False)
        # In COMSOL we evaluate g_31 at r = r0+delta/2+h_cn/2+h_n/2+delta*theta/2/pi
        r = r0 + delta / 2 + h_cn / 2 + h_n / 2 + delta * theta / 2 / pi
        self.g31 = g31_interp(r)

        # g2 = v(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "v_3.csv", comment="#", header=None).to_numpy()
        g32_r_data = comsol[:, 0]
        g32_data = comsol[:, 1] / delta
        g32_interp = interp.interp1d(g32_r_data, g32_data, bounds_error=False)
        # In COMSOL we evaluate g_32 at r = r0+delta/2+h_cn/2+delta*theta/2/pi
        r = r0 + delta / 2 + h_cn / 2 + delta * theta / 2 / pi
        self.g32 = g32_interp(r)

        # f1 = sigma_rr / (lambda+2*mu)
        comsol = pd.read_csv(path + "srr_4.csv", comment="#", header=None).to_numpy()
        f41_r_data = comsol[:, 0]
        f41_data = comsol[:, 1] / (lam_p + 2 * mu_p)
        f41_interp = interp.interp1d(f41_r_data, f41_data, bounds_error=False)
        # In COMSOL we evaluate f_41 at r = r0+delta-h_cp/2-h_p/2+delta*theta/2/pi
        r = r0 + delta - h_cp / 2 - h_p / 2 + delta * theta / 2 / pi
        self.f41 = f41_interp(r)

        # f2 = u(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "u_4.csv", comment="#", header=None).to_numpy()
        f42_r_data = comsol[:, 0]
        f42_data = comsol[:, 1] / delta
        f42_interp = interp.interp1d(f42_r_data, f42_data, bounds_error=False)
        # In COMSOL we evaluate f_42 at r = r0+delta-h_cp/2+delta*theta/2/pi
        r = r0 + delta - h_cp / 2 + delta * theta / 2 / pi
        self.f42 = f42_interp(r)

        # g1 = sigma_rt/mu
        comsol = pd.read_csv(path + "srt_4.csv", comment="#", header=None).to_numpy()
        g41_r_data = comsol[:, 0]
        g41_data = comsol[:, 1] / mu_p
        g41_interp = interp.interp1d(g41_r_data, g41_data, bounds_error=False)
        # In COMSOL we evaluate g_41 at r = r0+delta-h_cp/2-h_p/2+delta*theta/2/pi
        r = r0 + delta - h_cp / 2 - h_p / 2 + delta * theta / 2 / pi
        self.g41 = g41_interp(r)

        # g2 = v(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "v_4.csv", comment="#", header=None).to_numpy()
        g42_r_data = comsol[:, 0]
        g42_data = comsol[:, 1] / delta
        g42_interp = interp.interp1d(g42_r_data, g42_data, bounds_error=False)
        # In COMSOL we evaluate g_42 at r = r0+delta-h_cp/2+delta*theta/2/pi
        r = r0 + delta - h_cp / 2 + delta * theta / 2 / pi
        self.g42 = g42_interp(r)

        # In COMSOL we evaluate the tension in the positive current collector
        # at three points:
        # ra = r0-h_cp/2+delta*theta/2/pi
        # rb = r0+delta*theta/2/pi
        # rc = r0+h_cp/2+delta*theta/2/pi
        ra = r0 - h_cp / 2 + delta * theta / 2 / pi
        rb = r0 + delta * theta / 2 / pi
        rc = r0 + h_cp / 2 + delta * theta / 2 / pi

        comsola = pd.read_csv(path + "Tp_a.csv", comment="#", header=None).to_numpy()
        T_r_dataa = comsola[:, 0]
        T_dataa = comsola[:, 1]
        T_interpa = interp.interp1d(T_r_dataa, T_dataa, bounds_error=False)
        self.Tp_a = T_interpa(ra)

        comsolb = pd.read_csv(path + "Tp_b.csv", comment="#", header=None).to_numpy()
        T_r_datab = comsolb[:, 0]
        T_datab = comsolb[:, 1]
        T_interpb = interp.interp1d(T_r_datab, T_datab, bounds_error=False)
        self.Tp_b = T_interpb(rb)

        comsolc = pd.read_csv(path + "Tp_c.csv", comment="#", header=None).to_numpy()
        T_r_datac = comsolc[:, 0]
        T_datac = comsolc[:, 1]
        T_interpc = interp.interp1d(T_r_datac, T_datac, bounds_error=False)
        self.Tp_c = T_interpc(rc)

        # compute tension using simpsons rule
        self.Tp = (self.Tp_a + 4 * self.Tp_b + self.Tp_c) / 6

        # In COMSOL we evaluate the tension in the negative current collector
        # at three points:
        # ra = r0+delta/2-h_cn/2+delta*theta/2/pi
        # rb = r0+delta/2+delta*theta/2/pi
        # rc = r0+delta/2+h_cn/2+delta*theta/2/pi
        ra = r0 + delta / 2 - h_cn / 2 + delta * theta / 2 / pi
        rb = r0 + delta / 2 + delta * theta / 2 / pi
        rc = r0 + delta / 2 + h_cn / 2 + delta * theta / 2 / pi

        comsola = pd.read_csv(path + "Tn_a.csv", comment="#", header=None).to_numpy()
        T_r_dataa = comsola[:, 0]
        T_dataa = comsola[:, 1]
        T_interpa = interp.interp1d(T_r_dataa, T_dataa, bounds_error=False)
        self.Tn_a = T_interpa(ra)

        comsolb = pd.read_csv(path + "Tn_b.csv", comment="#", header=None).to_numpy()
        T_r_datab = comsolb[:, 0]
        T_datab = comsolb[:, 1]
        T_interpb = interp.interp1d(T_r_datab, T_datab, bounds_error=False)
        self.Tn_b = T_interpb(rb)

        comsolc = pd.read_csv(path + "Tn_c.csv", comment="#", header=None).to_numpy()
        T_r_datac = comsolc[:, 0]
        T_datac = comsolc[:, 1]
        T_interpc = interp.interp1d(T_r_datac, T_datac, bounds_error=False)
        self.Tn_c = T_interpc(rc)

        # compute tension using simpsons rule
        self.Tn = (self.Tn_a + 4 * self.Tn_b + self.Tn_c) / 6
