import numpy as np
import lsqfit as lsf
import gvar as gv

from analysis.models import MathModels
from analysis.utils import en_fit_lookup, ksi_from_disp_fit
from data.correlators import AnalysisCorrelators
from input.config import Config
from input.selector import SelectorType
from input.types import FitResultList


class RunFitting:

    def __init__(self, config: Config):
        self.config = config
        self.lattice_Nt = config.lattice_Nt
        self.results: FitResultList = []

    def effective_mass(self, corr: AnalysisCorrelators) -> FitResultList:
        selector = SelectorType(self.config, corr)
        data = selector.get_data()
        model_fn, prior_fn = selector.get_model()

        en_fit_list: FitResultList = []
        for ch_idx, mom_list in enumerate(self.config.chan_momt_list):
            for mom in mom_list:
                t_min = self.config.t_min[ch_idx][mom]
                t_max = self.config.t_max[ch_idx][mom]
                t_fit = np.arange(t_min, t_max)

                norm = data.at(ch_idx, mom)[self.lattice_Nt // 2].mean()
                y_gv = gv.dataset.avg_data(data.at(ch_idx, mom)[t_min:t_max].T / norm)

                mass_fit = lsf.nonlinear_fit(
                    data=(t_fit, y_gv),
                    fcn=lambda t, p: model_fn(t, p, lattice_Nt=self.lattice_Nt),
                    prior=prior_fn(
                        self.config.meff_prior[ch_idx][mom],
                        self.config.weff_prior[ch_idx][mom],
                    ),
                    p0=None,
                )

                for key in mass_fit.p:
                    if key.startswith("weff_"):
                        mass_fit.p[key] *= norm

                if not self.config.run_resample:
                    print("ch_idx, mom =", ch_idx, mom, "\n", mass_fit)

                en_fit_list.append({"ch_idx": ch_idx, "mom": mom, "fit": mass_fit})

        self.results = en_fit_list
        return en_fit_list

    def dispersion(self, en_fit_list: FitResultList) -> FitResultList:
        at_invs = self.config.at_invs
        ns = self.config.ensemble_key[0]
        lookup = en_fit_lookup(en_fit_list)

        disp_fit_list: FitResultList = []
        for ch_idx, mom_list in enumerate(self.config.chan_momt_list):
            en_sq_list = [
                (lookup[(ch_idx, mom)].p["meff_0"] * at_invs) ** 2 for mom in mom_list
            ]

            disp_fit = lsf.nonlinear_fit(
                data=(np.array(mom_list), en_sq_list),
                fcn=MathModels.linear,
                prior=MathModels.prior_linear(),
                p0=None,
            )

            if not self.config.run_resample:
                print("ch_idx, ksi =", ch_idx, ksi_from_disp_fit(disp_fit, at_invs, ns), disp_fit)

            disp_fit_list.append({"ch_idx": ch_idx, "fit": disp_fit})

        self.results = disp_fit_list
        return disp_fit_list
