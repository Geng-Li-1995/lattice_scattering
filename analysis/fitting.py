# fitting.py

import numpy as np
import lsqfit as lsf
import gvar as gv

from input.config import Config
from input.selector import SelectorType
from input.types import FileDict, FitResultList
from analysis.models import MathModels


class RunFitting:

    def __init__(self, config: Config):
        self.config = config
        self.lattice_Nt = config.lattice_Nt
        self.results: FitResultList = []

    def effective_mass(self, data_dict: FileDict) -> FitResultList:
        """
        Fitting effective mass
        """

        data = SelectorType(self.config, data_dict).get_data()
        fit_function, fit_prior = SelectorType(self.config, data_dict).get_model()
        chan_momt_list = self.config.chan_momt_list

        for ch_idx, mom_list in enumerate(chan_momt_list):
            for mom_idx, mom in enumerate(mom_list):

                # fit x, y, prior
                tmin = self.config.t_min[ch_idx][mom]
                tmax = self.config.t_max[ch_idx][mom]
                t_range = np.arange(tmin, tmax)

                normalization = data[ch_idx][mom][self.lattice_Nt // 2].mean()
                y_sample = data[ch_idx][mom][tmin:tmax] / normalization
                y_gv = gv.dataset.avg_data(y_sample.T)

                prior = fit_prior(
                    self.config.meff_prior[ch_idx][mom],
                    self.config.weff_prior[ch_idx][mom],
                )

                # run fit
                fit_kwargs = dict(
                    data=(t_range, y_gv),
                    fcn=lambda t_range, parameter: fit_function(
                        t_range, parameter, lattice_Nt=self.lattice_Nt
                    ),
                    prior=prior,
                    p0=None,
                )

                fit = lsf.nonlinear_fit(**fit_kwargs)

                for key in fit.p:
                    if key.startswith("weff_"):
                        fit.p[key] *= normalization  # Fix normalization

                if not getattr(self.config, "run_resample", False):
                    print("channel, momentum =", ch_idx, mom, "\n", fit)

                # store
                self.results.append(
                    {
                        "channel": ch_idx,
                        "mom": mom,
                        "fit": fit,
                    }
                )

        return self.results

    def dispersion(self, En_result: FitResultList) -> FitResultList:
        """
        Fitting dispersion relation from meson Ens
        """
        at_invs = self.config.at_invs
        Ns = self.config.ensemble_key[0]
        chan_momt_list = self.config.chan_momt_list
        Ens_sq = {}

        for ch_idx, mom_list in enumerate(chan_momt_list):
            for mom_idx, mom in enumerate(mom_list):
                En_fit = next(
                    (
                        r["fit"]
                        for r in En_result
                        if r["channel"] == ch_idx and r["mom"] == mom
                    ),
                    None,
                )
                if En_fit is None:
                    raise ValueError(f"Missing fit: channel={ch_idx}, mom={mom}")
                # Save effective masses
                En = En_fit.p["meff_0"] * at_invs
                Ens_sq.setdefault(ch_idx, []).append(En**2)

            fit_linear = lsf.nonlinear_fit(
                data=(np.array(mom_list), Ens_sq[ch_idx]),
                fcn=MathModels.linear,
                prior=MathModels.prior_linear(),
                p0=None,
            )

            ksi = 2 * np.pi * at_invs / np.sqrt(fit_linear.p["a"]) / Ns
            if not getattr(self.config, "run_resample", False):
                print("###fit_ksi is:", ch_idx, ksi, fit_linear)

            self.results.append(
                {
                    "channel": ch_idx,
                    "fit": fit_linear,
                }
            )

        return self.results
