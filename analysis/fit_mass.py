from typing import Any

import gvar as gv
import lsqfit as lsf
import numpy as np

from analysis.models import MathModels
from data.correlators import AnalysisCorrelators
from input.config import Config, FitResultList, SelectorType


def en_fit_lookup(en_fit_list: FitResultList) -> dict[tuple[int, int], Any]:
    return {
        (entry["chan_idx"], entry["mom"]): entry["fit"]
        for entry in en_fit_list
        if "mom" in entry
    }


def disp_fit_lookup(disp_fit_list: FitResultList) -> dict[int, Any]:
    return {entry["chan_idx"]: entry["fit"] for entry in disp_fit_list}


def ksi_from_disp_fit(disp_fit, at_invs: float, Ns: int) -> float:
    return 2 * np.pi * at_invs / np.sqrt(disp_fit.p["a"]) / Ns


def fit_dispersion(mom_list: list[int] | np.ndarray, en_sq_list) -> Any:
    return lsf.nonlinear_fit(
        data=(np.asarray(mom_list), en_sq_list),
        fcn=MathModels.linear,
        prior=MathModels.prior_linear(),
        p0=None,
    )


class RunFitting:

    def __init__(self, config: Config):
        self.config = config
        self.lattice_Nt = config.lattice_Nt

    def effective_mass(self, corr: AnalysisCorrelators) -> FitResultList:
        selector = SelectorType(self.config, corr)
        data = selector.get_data()
        model_fn, prior_fn = selector.get_model()

        en_fit_list: FitResultList = []
        for chan_idx, mom_list in enumerate(self.config.chan_momentum_list):
            for mom in mom_list:
                t_min = self.config.t_min[chan_idx][mom]
                t_max = self.config.t_max[chan_idx][mom]
                t_fit = np.arange(t_min, t_max)

                norm = data.at(chan_idx, mom)[self.lattice_Nt // 2].mean()
                y_gv = gv.dataset.avg_data(data.at(chan_idx, mom)[t_min:t_max].T / norm)

                mass_fit = lsf.nonlinear_fit(
                    data=(t_fit, y_gv),
                    fcn=lambda t, p: model_fn(t, p, lattice_Nt=self.lattice_Nt),
                    prior=prior_fn(
                        self.config.meff_prior[chan_idx][mom],
                        self.config.weff_prior[chan_idx][mom],
                        self.config.meff_prior_error,
                        self.config.weff_prior_error,
                    ),
                    p0=None,
                )

                for key in mass_fit.p:
                    if key.startswith("weff_"):
                        mass_fit.p[key] *= norm

                if not self.config.run_resample_analysis:
                    print("chan_idx, mom =", chan_idx, mom, "\n", mass_fit)

                en_fit_list.append({"chan_idx": chan_idx, "mom": mom, "fit": mass_fit})

        return en_fit_list

    def dispersion(self, en_fit_list: FitResultList) -> FitResultList:
        at_invs = self.config.at_invs
        Ns = self.config.ensemble_key[0]
        lookup = en_fit_lookup(en_fit_list)

        disp_fit_list: FitResultList = []
        for chan_idx, mom_list in enumerate(self.config.chan_momentum_list):
            en_sq_list = [
                (lookup[(chan_idx, mom)].p["meff_0"] * at_invs) ** 2 for mom in mom_list
            ]

            disp_fit = fit_dispersion(mom_list, en_sq_list)

            if not self.config.run_resample_analysis:
                print("chan_idx, ksi =", chan_idx, ksi_from_disp_fit(disp_fit, at_invs, Ns), disp_fit)

            disp_fit_list.append({"chan_idx": chan_idx, "fit": disp_fit})

        return disp_fit_list
