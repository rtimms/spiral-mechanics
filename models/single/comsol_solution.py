import numpy as np
from numpy import pi
import pandas as pd
import scipy.interpolate as interp


class ComsolSolution:
    def __init__(self, r0, delta, hh, N, mu, lam, alpha, alpha_cc, path):
        """
        Loads the COMSOL solution. The parameters and variables are stored as
        attributes of the class. Note that we rescale the displacements, strains
        and stresses by alpha_scale since COMSOL didn't like having alpha=1.
        """

        # Note: COMSOL data is (r, f) so we create interpolants to get
        # (theta, f) data
        theta = np.linspace(0, 2 * pi * N, 30 * (N - 1))
        self.theta = theta

        # f1 = sigma_rr
        comsol = pd.read_csv(path + "srr3.csv", comment="#", header=None).to_numpy()
        f1_r_data = comsol[:, 0]
        f1_data = comsol[:, 1]
        f1_interp = interp.interp1d(
            f1_r_data, f1_data, bounds_error=False, kind="cubic"
        )
        # In COMSOL we evaluate f_1 at r = r0+delta/2+delta*theta/2/pi
        r = r0 + delta / 2 + delta * theta / 2 / pi
        self.f1 = f1_interp(r)

        # f2 = (u(R=theta/2/pi) - alpha_cc*r0)/delta
        comsol = pd.read_csv(path + "u1.csv", comment="#", header=None).to_numpy()
        f2_r_data = comsol[:, 0]
        f2_data = (comsol[:, 1] - alpha_cc * r0) / delta
        f2_interp = interp.interp1d(
            f2_r_data, f2_data, bounds_error=False, kind="cubic"
        )
        # In COMSOL we evaluate f_2 at r = r0+hh/2+delta*theta/2/pi
        r = r0 + hh / 2 + delta * theta / 2 / pi
        self.f2 = f2_interp(r)

        # g1 = sigma_rt
        comsol = pd.read_csv(path + "srt3.csv", comment="#", header=None).to_numpy()
        g1_r_data = comsol[:, 0]
        g1_data = comsol[:, 1]
        g1_interp = interp.interp1d(
            g1_r_data, g1_data, bounds_error=False, kind="cubic"
        )
        # In COMSOL we evaluate g_1 at r = r0+delta/2+delta*theta/2/pi
        r = r0 + delta / 2 + delta * theta / 2 / pi
        self.g1 = g1_interp(r)

        # g2 = v(R=theta/2/pi)/delta
        comsol = pd.read_csv(path + "v1.csv", comment="#", header=None).to_numpy()
        g2_r_data = comsol[:, 0]
        g2_data = comsol[:, 1] / delta
        g2_interp = interp.interp1d(
            g2_r_data, g2_data, bounds_error=False, kind="cubic"
        )
        # In COMSOL we evaluate g_2 at r = r0+hh/2+delta*theta/2/pi
        r = r0 + hh / 2 + delta * theta / 2 / pi
        self.g2 = g2_interp(r)

        # In COMSOL we evaluate the displacements, stresses and strains
        # at r = r0+delta/2+delta*theta/2/pi
        r = r0 + delta / 2 + delta * theta / 2 / pi

        # radial displacement
        comsol = pd.read_csv(path + "u3.csv", comment="#", header=None).to_numpy()
        u_r_data = comsol[:, 0]
        u_data = comsol[:, 1]
        u_interp = interp.interp1d(u_r_data, u_data, bounds_error=False, kind="cubic")
        self.u = u_interp(r)

        # azimuthal displacement
        comsol = pd.read_csv(path + "v3.csv", comment="#", header=None).to_numpy()
        v_r_data = comsol[:, 0]
        v_data = comsol[:, 1]
        v_interp = interp.interp1d(v_r_data, v_data, bounds_error=False, kind="cubic")
        self.v = v_interp(r)

        # radial strain
        comsol = pd.read_csv(path + "err3.csv", comment="#", header=None).to_numpy()
        err_r_data = comsol[:, 0]
        err_data = comsol[:, 1]
        err_interp = interp.interp1d(
            err_r_data, err_data, bounds_error=False, kind="cubic"
        )
        self.err = err_interp(r)

        # azimuthal strain
        comsol = pd.read_csv(path + "ett3.csv", comment="#", header=None).to_numpy()
        ett_r_data = comsol[:, 0]
        ett_data = comsol[:, 1]
        ett_interp = interp.interp1d(
            ett_r_data, ett_data, bounds_error=False, kind="cubic"
        )
        self.ett = ett_interp(r)

        # shear strain
        comsol = pd.read_csv(path + "ert3.csv", comment="#", header=None).to_numpy()
        ert_r_data = comsol[:, 0]
        ert_data = comsol[:, 1]
        ert_interp = interp.interp1d(
            ert_r_data, ert_data, bounds_error=False, kind="cubic"
        )
        self.ert = ert_interp(r)

        # radial stress
        comsol = pd.read_csv(path + "srr3.csv", comment="#", header=None).to_numpy()
        srr_r_data = comsol[:, 0]
        srr_data = comsol[:, 1]
        srr_interp = interp.interp1d(
            srr_r_data, srr_data, bounds_error=False, kind="cubic"
        )
        self.srr = srr_interp(r)

        # azimuthal stress
        comsol = pd.read_csv(path + "stt3.csv", comment="#", header=None).to_numpy()
        stt_r_data = comsol[:, 0]
        stt_data = comsol[:, 1]
        stt_interp = interp.interp1d(
            stt_r_data, stt_data, bounds_error=False, kind="cubic"
        )
        self.stt = stt_interp(r)

        # shear stress
        comsol = pd.read_csv(path + "srt3.csv", comment="#", header=None).to_numpy()
        srt_r_data = comsol[:, 0]
        srt_data = comsol[:, 1]
        srt_interp = interp.interp1d(
            srt_r_data, srt_data, bounds_error=False, kind="cubic"
        )
        self.srt = srt_interp(r)

        # In COMSOL we evaluate the tension at three points:
        # ra = r0-hh/2+delta*theta/2/pi
        # rb = r0+delta*theta/2/pi
        # rc = r0+hh/2+delta*theta/2/pi
        ra = r0 - hh / 2 + delta * theta / 2 / pi
        rb = r0 + delta * theta / 2 / pi
        rc = r0 + hh / 2 + delta * theta / 2 / pi

        comsola = pd.read_csv(path + "T1.csv", comment="#", header=None).to_numpy()
        T_r_dataa = comsola[:, 0]
        T_dataa = comsola[:, 1]
        T_interpa = interp.interp1d(
            T_r_dataa, T_dataa, bounds_error=False, kind="cubic"
        )
        self.T_a = T_interpa(ra)

        comsolb = pd.read_csv(path + "T3.csv", comment="#", header=None).to_numpy()
        T_r_datab = comsolb[:, 0]
        T_datab = comsolb[:, 1]
        T_interpb = interp.interp1d(
            T_r_datab, T_datab, bounds_error=False, kind="cubic"
        )
        self.T_b = T_interpb(rb)

        comsolc = pd.read_csv(path + "T5.csv", comment="#", header=None).to_numpy()
        T_r_datac = comsolc[:, 0]
        T_datac = comsolc[:, 1]
        T_interpc = interp.interp1d(
            T_r_datac, T_datac, bounds_error=False, kind="cubic"
        )
        self.T_c = T_interpc(rc)

        # compute tension using simpsons rule
        self.T = (self.T_a + 4 * self.T_b + self.T_c) / 6


class ComsolInnerSolution:
    def __init__(self, path):
        """
        Loads the COMSOL solution for the inner problem. The variables are
        stored as attributes of the class.
        """
        # Note: currently exported on a 101x101 grid
        comsol = pd.read_csv(path + "u_grid.csv", comment="#", header=None).to_numpy()
        # displacements
        self.u_t_data = comsol[:, 0].reshape(101, 101)
        self.u_r_data = comsol[:, 1].reshape(101, 101)
        self.u_data = comsol[:, 2].reshape(101, 101)
        comsol = pd.read_csv(path + "v_grid.csv", comment="#", header=None).to_numpy()
        self.v_t_data = comsol[:, 0].reshape(101, 101)
        self.v_r_data = comsol[:, 1].reshape(101, 101)
        self.v_data = comsol[:, 2].reshape(101, 101)
        # strains
        comsol = pd.read_csv(path + "err_grid.csv", comment="#", header=None).to_numpy()
        self.err_t_data = comsol[:, 0].reshape(101, 101)
        self.err_r_data = comsol[:, 1].reshape(101, 101)
        self.err_data = comsol[:, 2].reshape(101, 101)
        comsol = pd.read_csv(path + "ett_grid.csv", comment="#", header=None).to_numpy()
        self.ett_t_data = comsol[:, 0].reshape(101, 101)
        self.ett_r_data = comsol[:, 1].reshape(101, 101)
        self.ett_data = comsol[:, 2].reshape(101, 101)
        comsol = pd.read_csv(path + "ert_grid.csv", comment="#", header=None).to_numpy()
        self.ert_t_data = comsol[:, 0].reshape(101, 101)
        self.ert_r_data = comsol[:, 1].reshape(101, 101)
        self.ert_data = comsol[:, 2].reshape(101, 101)
        # stresses
        comsol = pd.read_csv(path + "srr_grid.csv", comment="#", header=None).to_numpy()
        self.srr_t_data = comsol[:, 0].reshape(101, 101)
        self.srr_r_data = comsol[:, 1].reshape(101, 101)
        self.srr_data = comsol[:, 2].reshape(101, 101)
        comsol = pd.read_csv(path + "stt_grid.csv", comment="#", header=None).to_numpy()
        self.stt_t_data = comsol[:, 0].reshape(101, 101)
        self.stt_r_data = comsol[:, 1].reshape(101, 101)
        self.stt_data = comsol[:, 2].reshape(101, 101)
        comsol = pd.read_csv(path + "srt_grid.csv", comment="#", header=None).to_numpy()
        self.srt_t_data = comsol[:, 0].reshape(101, 101)
        self.srt_r_data = comsol[:, 1].reshape(101, 101)
        self.srt_data = comsol[:, 2].reshape(101, 101)
