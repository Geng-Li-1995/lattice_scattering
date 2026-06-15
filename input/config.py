# config.py

from dataclasses import dataclass
from typing import List, Tuple
import importlib
import numpy as np


# ===============================
# Output Config used in analysis
# ===============================
@dataclass
class Config:
    input_name: str
    chan_momt_list: List[List[int]]
    chan_name_list: List[str]

    ensemble_key: Tuple[int, int, int, int]
    scattering_list: List[int]
    tag_name: str
    lattice_Nt: int
    at_invs: float

    t_GEVP: Tuple[int, int, int]
    t_min: List[List[int]]
    t_max: List[List[int]]

    n_state: int

    meff_prior: List[List[List[float]]]
    weff_prior: List[List[List[float]]]

    is_meson_analysis: bool
    is_tetraquark_analysis: bool
    is_gevp: bool
    is_svd: bool
    is_ratio: bool

    run_tmin: bool
    run_resample: bool
    run_scattering: bool

    plot_meff: bool
    plot_dispersion: bool

    resample_type: str
    sample_axis: int
    n_boot: int


# ===============================
# BuildConfig
# ===============================
@dataclass
class BuildConfig:
    input_name: str

    def __post_init__(self):
        module_name = f"input.{self.input_name}_input"
        self.input_module = importlib.import_module(module_name)

        for name in ["InputControl", "ENSEMBLE_DB"]:
            if not hasattr(self.input_module, name):
                raise ValueError(f"{module_name} missing {name}")

        self.input_control = self.input_module.InputControl()
        self.ensemble_db = self.input_module.ENSEMBLE_DB
        self.ensemble_key = self.input_control.ensemble_key()

    # --------------------------------------------------
    # Convert [state][chan][mom] -> [chan][mom][state]
    # --------------------------------------------------
    @staticmethod
    def _reorder_prior(raw_prior):
        arr = np.array(raw_prior, dtype=float)
        return np.transpose(arr, (1, 2, 0)).tolist()

    # --------------------------------------------------
    # Build Config
    # --------------------------------------------------
    def build_config_from_control(self) -> Config:
        ctrl = self.input_control
        key = self.ensemble_key

        lattice_Ns, lattice_Nt, pion_mass, num_eigenvectors = key
        tag_name = f"L{lattice_Ns}M{pion_mass}_EV{num_eigenvectors}"

        analysis_type = ctrl.analysis_type()
        if key not in self.ensemble_db:
            raise ValueError(f"Unknown ensemble key: {key}")

        db = self.ensemble_db[key][analysis_type]

        chan_momt_list = db["chan_momt_list"]
        chan_name_list = db["chan_name_list"]

        t_min = db.get("tmin_arry", [[20] * len(m) for m in chan_momt_list])
        t_max = [[lattice_Nt - t for t in row] for row in t_min]

        # extract priors
        meff_prior_raw = [v for k, v in sorted(db.items()) if "prir_meff" in k]
        weff_prior_raw = [v for k, v in sorted(db.items()) if "prir_weff" in k]

        meff_prior = self._reorder_prior(meff_prior_raw)
        weff_prior = self._reorder_prior(weff_prior_raw)
        n_state = len(meff_prior_raw)

        return Config(
            input_name=self.input_name,
            chan_momt_list=chan_momt_list,
            chan_name_list=chan_name_list,
            ensemble_key=key,
            scattering_list=ctrl.scattering_list,
            tag_name=tag_name,
            lattice_Nt=lattice_Nt,
            at_invs=self.ensemble_db[key]["at_invs"],
            t_GEVP=self.ensemble_db[key]["GEVP"],
            t_min=t_min,
            t_max=t_max,
            n_state=n_state,
            meff_prior=meff_prior,
            weff_prior=weff_prior,
            is_meson_analysis=ctrl.is_meson_analysis,
            is_tetraquark_analysis=ctrl.is_tetraquark_analysis,
            is_gevp=ctrl.is_gevp,
            is_svd=ctrl.is_svd,
            is_ratio=ctrl.is_ratio,
            run_tmin=ctrl.run_tmin,
            run_resample=ctrl.run_resample,
            run_scattering=ctrl.run_scattering,
            plot_meff=ctrl.plot_meff,
            plot_dispersion=ctrl.plot_dispersion,
            resample_type=ctrl.resample_type,
            sample_axis=ctrl.sample_axis,
            n_boot=ctrl.n_boot,
        )
