#!/usr/bin/env python

import os
from netCDF4 import Dataset

import scipy.ndimage
import scipy.interpolate

import veros
import veros.tools
import ruamel.yaml as yaml

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets("global_2deg", os.path.join(BASE_PATH, "assets_2deg.yml"))


class GlobalTwoDegree(veros.Veros):
    """Global 2 degree model with 45 vertical levels.

    `Adapted from pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/2x2%20global%20model%2045%20levels>`_.
    """

    @veros.veros_method
    def set_parameter(self):
        """
        set main parameters
        """
        self.identifier = "2deg"
        self.nx = 128
        self.ny = 64
        self.nz = 45
        self.dt_mom = 3600.0
        self.dt_tracer = 3600.0 * 3
        self.dt_bio = self.dt_tracer // 4
        # self.runlen = 365 * 86400. * 50
        self.runlen = 200 * 360 * 86400

        self.coord_degree = True
        self.enable_cyclic_x = True

        self.congr_epsilon = 1e-8
        self.congr_max_iterations = 20000

        self.enable_hor_friction = True
        self.A_h = (2.8 * self.degtom)**3 * 2e-11
        self.enable_hor_friction_cos_scaling = True
        self.hor_friction_cosPower = 1
        self.enable_tempsalt_sources = True
        self.enable_implicit_vert_friction = True

        self.enable_npzd = True
        self.enable_carbon = True
        self.enable_nitrogen = False
        self.enable_calcifiers = False
        self.enable_iron = False
        self.enable_oxygen = False

        with open( os.path.join(BASE_PATH, "npzd.yaml")) as yaml_file:
            cfg = yaml.safe_load(yaml_file)["npzd"]
            self.npzd_selected_rules = cfg["selected_rules"]

        # isoneutral
        self.enable_neutral_diffusion = True
        self.K_iso_0 = 1000.0
        self.K_iso_steep = 500.0
        self.iso_dslope=0.001
        self.iso_slopec=0.001
        self.enable_skew_diffusion = True

        # tke
        self.enable_tke = True
        self.c_k = 0.1
        self.c_eps = 0.7
        self.alpha_tke = 30.0
        self.mxl_min = 1e-8
        self.tke_mxl_choice = 2
        self.kappaM_min = 2e-4
        self.kappaH_min = 2e-5
        self.enable_Prandtl_tke = False
        self.enable_tke_superbee_advection = True

        # eke
        self.K_gm_0 = 1000.0
        self.enable_eke = False
        self.eke_k_max = 1e4
        self.eke_c_k = 0.4
        self.eke_c_eps = 0.5
        self.eke_cross = 2.
        self.eke_crhin = 1.0
        self.eke_lmin = 100.0
        self.enable_eke_superbee_advection = False
        self.enable_eke_isopycnal_diffusion = False

        # idemix
        self.enable_idemix = False
        self.enable_idemix_hor_diffusion = False
        self.enable_eke_diss_surfbot = False
        self.eke_diss_surfbot_frac = 0.2  #  fraction which goes into bottom
        self.enable_idemix_superbee_advection = False

        self.eq_of_state_type = 5

    @veros.veros_method
    def _read_forcing(self, var):
        with Dataset(DATA_FILES["forcing"], "r") as infile:
            return np.array(infile.variables[var][...]).T

    @veros.veros_method
    def set_grid(self):
        dz_data = self._read_forcing("dz")
        self.dzt[...] = dz_data[::-1]
        self.dxt[...] = 2.8125
        self.dyt[...] = 2.8125
        self.y_origin = -87.1875
        self.x_origin = 2.8125

    @veros.veros_method
    def set_coriolis(self):
        self.coriolis_t[...] = 2 * self.omega * np.sin(self.yt[np.newaxis, :] / 180. * self.pi)

    @veros.veros_method
    def set_topography(self):
        bathymetry_data = self._read_forcing("bathymetry")
        salt_data = np.squeeze(self._read_forcing("salinity"))
        salt_data[salt_data <= -1e33] = 0.

        mask_salt = salt_data == 0.
        self.kbot[2:-2, 2:-2] = 1 + np.sum(mask_salt.astype(np.int), axis=2)
        mask_bathy = bathymetry_data >= 0
        self.kbot[2:-2, 2:-2][mask_bathy] = 0
        self.kbot[self.kbot >= self.nz] = 0

    @veros.veros_method
    def set_initial_conditions(self):
        self.t_star = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self.s_star = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self.qnec = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self.qnet = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self.qsol = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self.divpen_shortwave = np.zeros(self.nz, dtype=self.default_float_type)
        self.taux = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)
        self.tauy = np.zeros((self.nx + 4, self.ny + 4, 12), dtype=self.default_float_type)

        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        # initial conditions
        temp_data = np.squeeze(self._read_forcing("temperature"))
        temp_data[temp_data <= -1e33] = 0.
        self.temp[2:-2, 2:-2, :, 0] = temp_data[...] * self.maskT[2:-2, 2:-2, :]
        self.temp[2:-2, 2:-2, :, 1] = temp_data[...] * self.maskT[2:-2, 2:-2, :]

        salt_data = np.squeeze(self._read_forcing("salinity"))
        salt_data[salt_data <= -1e33] = 0.
        self.salt[2:-2, 2:-2, :, 0] = salt_data[...] * self.maskT[2:-2, 2:-2, :]
        self.salt[2:-2, 2:-2, :, 1] = salt_data[...] * self.maskT[2:-2, 2:-2, :]

        # wind stress on MIT grid
        taux_data = self._read_forcing("tau_x")
        self.taux[2:-2, 2:-2, :] = taux_data * self.maskU[2:-2, 2:-2, -1, np.newaxis] / self.rho_0

        tauy_data = self._read_forcing("tau_y")
        self.tauy[2:-2, 2:-2, :] = tauy_data * self.maskV[2:-2, 2:-2, -1, np.newaxis] / self.rho_0


        # Qnet and dQ/dT and Qsol
        qnet_data = self._read_forcing("q_net")
        self.qnet[2:-2, 2:-2, :] = -qnet_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        qnec_data = self._read_forcing("dqdt")
        self.qnec[2:-2, 2:-2, :] = qnec_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        qsol_data = self._read_forcing("swf")
        self.qsol[2:-2, 2:-2, :] = -qsol_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        # SST and SSS
        sst_data = self._read_forcing("sst")
        self.t_star[2:-2, 2:-2, :] = sst_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        sss_data = self._read_forcing("sss")
        self.s_star[2:-2, 2:-2, :] = sss_data * self.maskT[2:-2, 2:-2, -1, np.newaxis]

        if self.enable_idemix:
            tidal_energy_data = self._read_forcing("tidal_energy")
            mask = np.maximum(0, self.kbot[2:-2, 2:-2] - 1)[:, :, np.newaxis] == np.arange(self.nz)[np.newaxis, np.newaxis, :]
            tidal_energy_data[:, :] *= self.maskW[2:-2, 2:-2, :][mask].reshape(self.nx, self.ny) / self.rho_0
            self.forc_iw_bottom[2:-2, 2:-2] = tidal_energy_data

            wind_energy_data = self._read_forcing("wind_energy")
            wind_energy_data[:, :] *= self.maskW[2:-2, 2:-2, -1] / self.rho_0 * 0.2
            self.forc_iw_surface[2:-2, 2:-2] = wind_energy_data

        """
        Initialize penetration profile for solar radiation and store divergence in divpen
        note that pen is set to 0.0 at the surface instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = self.zw / efold1_shortwave
        swarg2 = self.zw / efold2_shortwave
        pen = rpart_shortwave * np.exp(swarg1) + (1.0 - rpart_shortwave) * np.exp(swarg2)
        pen[-1] = 0.
        self.divpen_shortwave = np.zeros(self.nz, dtype=self.default_float_type)
        self.divpen_shortwave[1:] = (pen[1:] - pen[:-1]) / self.dzt[1:]
        self.divpen_shortwave[0] = pen[0] / self.dzt[0]

        if self.enable_npzd:
            with Dataset(DATA_FILES["corev2"], "r") as infile:
                self.swr_initial = infile.variables["SWDN_MOD"][...].T

            phytoplankton = 0.14 * np.exp(self.zw) * self.maskT
            zooplankton = 0.014 * np.exp(self.zw) * self.maskT

            self.phytoplankton[:, :, :, :] = phytoplankton[..., np.newaxis]
            self.zooplankton[:, :, :, :] = zooplankton[..., np.newaxis]
            self.detritus[:, :, :, :] = 1e-4 * self.maskT[..., np.newaxis]

            self.po4[:, :, :, :] = 2.5
            # self.po4[:, :, -1, -1:] = 0.05
            self.po4[...] *= self.maskT[..., np.newaxis]




        if self.enable_nitrogen:
            self.diazotroph[...] = self.phytoplankton / 10
            self.no3[...] = self.po4[...] * 10
            self.don[...] = 20 * self.maskT[..., np.newaxis]
            self.dop[...] = 20 * self.maskT[..., np.newaxis]

        if self.enable_carbon:
            self.dic[...] = 2200
            self.atmospheric_co2[...] = 280
            self.alkalinity[...] = 2400

            self.hSWS[...] = 5e-7


        if self.enable_calcifiers:
            self.coccolitophore[...] = self.phytoplankton / 10
            self.caco3[...] = 0.01 * self.maskT[..., np.newaxis]

        if self.enable_iron:
            self.fe[2:-2, 2:-2, -2:] = 8
            self.particulate_fe[2:-2, 2:-2, -2:] = 9

        if self.enable_oxygen:
            self.o2[2:-2, 2:-2, -2:] = 20

    @veros.veros_method
    def set_forcing(self):
        t_rest = 30. * 86400.
        cp_0 = 3991.86795711963  # J/kg /K

        year_in_seconds = veros.time.convert_time(self, 1., "years", "seconds")
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(self.time, year_in_seconds,
                                                         year_in_seconds / 12., 12)

        # linearly interpolate wind stress and shift from MITgcm U/V grid to this grid
        self.surface_taux[:-1, :] = f1 * self.taux[1:, :, n1] + f2 * self.taux[1:, :, n2]
        self.surface_tauy[:, :-1] = f1 * self.tauy[:, 1:, n1] + f2 * self.tauy[:, 1:, n2]

        if self.enable_tke:
            self.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (self.surface_taux[1:-1, 1:-1] \
                                                              + self.surface_taux[:-2, 1:-1])) ** 2 \
                                                      + (0.5 * (self.surface_tauy[1:-1, 1:-1] \
                                                              + self.surface_tauy[1:-1, :-2])) ** 2) ** (3. / 2.)

        # W/m^2 K kg/J m^3/kg = K m/s
        t_star_cur = f1 * self.t_star[..., n1] + f2 * self.t_star[..., n2]
        self.qqnec = f1 * self.qnec[..., n1] + f2 * self.qnec[..., n2]
        self.qqnet = f1 * self.qnet[..., n1] + f2 * self.qnet[..., n2]
        self.forc_temp_surface[...] = (self.qqnet + self.qqnec * (t_star_cur - self.temp[..., -1, self.tau])) \
            * self.maskT[..., -1] / cp_0 / self.rho_0
        s_star_cur = f1 * self.s_star[..., n1] + f2 * self.s_star[..., n2]
        self.forc_salt_surface[...] = 1. / t_rest * \
            (s_star_cur - self.salt[..., -1, self.tau]) * self.maskT[..., -1] * self.dzt[-1]

        # apply simple ice mask
        mask1 = self.temp[:, :, -1, self.tau] * self.maskT[:, :, -1] <= -1.8
        mask2 = self.forc_temp_surface <= 0
        ice = ~(mask1 & mask2)
        self.forc_temp_surface *= ice
        self.forc_salt_surface *= ice

        # solar radiation
        if self.enable_tempsalt_sources:
            self.temp_source[..., :] = (f1 * self.qsol[..., n1, None] + f2 * self.qsol[..., n2, None]) \
                * self.divpen_shortwave[None, None, :] * ice[..., None] \
                * self.maskT[..., :] / cp_0 / self.rho_0


        if self.enable_npzd:
            # incomming shortwave radiation for plankton production
            self.swr[2:-2, 2:-2] = (f1 * self.swr_initial[:, :, n1] + f2 * self.swr_initial[:, :, n2])

    @veros.veros_method
    def set_diagnostics(self):
        average_vars = ["surface_taux", "surface_tauy", "forc_temp_surface", "forc_salt_surface",
                        "psi", "temp", "salt", "u", "v", "w", "Nsqr", "Hd", "rho",
                        "K_diss_v", "P_diss_v", "P_diss_nonlin", "P_diss_iso", "kappaH"]
        # if self.enable_skew_diffusion:
        #     average_vars += ["B1_gm", "B2_gm"]
        # if self.enable_TEM_friction:
        #     average_vars += ["kappa_gm", "K_diss_gm"]
        # if self.enable_tke:
        #     average_vars += ["tke", "Prandtlnumber", "mxl", "tke_diss",
        #                      "forc_tke_surface", "tke_surf_corr"]
        # if self.enable_idemix:
        #     average_vars += ["E_iw", "forc_iw_surface", "forc_iw_bottom", "iw_diss",
        #                      "c0", "v0"]
        # if self.enable_eke:
        #     average_vars += ["eke", "K_gm", "L_rossby", "L_rhines"]
        if self.enable_npzd:
            average_vars += ["phytoplankton", "zooplankton", "detritus", "po4"]
        if self.enable_carbon:
            average_vars += ["dic", "alkalinity", "cflux", "pCO2"]

        self.diagnostics["averages"].output_variables = average_vars
        # self.diagnostics["cfl_monitor"].output_frequency = 86400.0
        self.diagnostics["snapshot"].output_frequency = 365 * 86400 #/ 24.
        # self.diagnostics["overturning"].output_frequency = 365 * 86400
        # self.diagnostics["overturning"].sampling_frequency = 365 * 86400 / 24.
        # self.diagnostics["energy"].output_frequency = 365 * 86400
        # self.diagnostics["energy"].sampling_frequency = 365 * 86400 / 24.
        self.diagnostics["averages"].output_frequency = 365 * 86400
        self.diagnostics["averages"].sampling_frequency = 365 * 86400 / 24.

    def after_timestep(self):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = GlobalTwoDegree(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()
