from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias
import importlib

import numpy as np

from analysis.models import MODEL_REGISTRY
from data.correlators import AnalysisCorrelators, Correlator4D

EnsembleKey: TypeAlias = Tuple[int, int, int, int]  # (Ns, Nt, pion_mass, num_eigenvectors)
ScatteringList: TypeAlias = List[EnsembleKey]
ResampleDataDict: TypeAlias = Dict[str, Dict[EnsembleKey, np.ndarray]]

ModelFn: TypeAlias = Callable[..., Any]
PriorFn: TypeAlias = Callable[..., Dict[str, Any]]
FitResultList: TypeAlias = List[Dict[str, Any]]
EnsembleEntry: TypeAlias = Dict[str, Any]
RatioScanPoint: TypeAlias = Tuple[int, ...]  # (tetra_ch, tetra_mom) or (tetra_ch, tetra_mom, meson_ch, meson_mom)


def ensemble_tag(ensemble_key: EnsembleKey) -> str:
    ns, _, pion_mass, num_ev = ensemble_key
    return f"L{ns}M{pion_mass}_EV{num_ev}"


def moving_frame_d_tag(d_vec: tuple[int, int, int]) -> str:
    return "d" + "".join(str(int(component)) for component in d_vec)


class InputControlMixin:
    """Common post-init and helpers; subclasses supply field defaults and ``get_lattice_params``."""

    lattice_Ns: int
    lattice_Nt: int
    pion_mass: int
    num_eigenvectors: int
    plot_format: str
    Ns_list: list[int]
    scattering_list: ScatteringList
    is_meson_analysis: bool
    is_tetraquark_analysis: bool

    def __post_init__(self) -> None:
        if self.plot_format not in ("png", "pdf"):
            raise ValueError(
                f'plot_format must be "png" or "pdf", got {self.plot_format!r}'
            )

        self.lattice_Ns, self.lattice_Nt, self.pion_mass, self.num_eigenvectors = (
            self.get_lattice_params(self.lattice_Ns)
        )
        self.scattering_list = [self.get_lattice_params(ns) for ns in self.Ns_list]

    @staticmethod
    def get_lattice_params(lattice_Ns: int) -> EnsembleKey:
        raise NotImplementedError

    def ensemble_key(self) -> EnsembleKey:
        return (self.lattice_Ns, self.lattice_Nt, self.pion_mass, self.num_eigenvectors)

    def analysis_type(self) -> str:
        if self.is_tetraquark_analysis:
            return "tetraquark"
        if self.is_meson_analysis:
            return "meson"
        raise ValueError("Neither meson nor tetraquark analysis is selected.")


@dataclass
class Config:
    input_name: str
    chan_momt_list: List[List[int]]
    chan_name_list: List[str]

    ensemble_key: EnsembleKey
    scattering_list: ScatteringList
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
    is_ratio: bool

    run_tmin: bool
    run_resample: bool
    run_scattering: bool
    is_moving_frame: bool

    plot_meff: bool
    plot_dispersion: bool
    plot_format: str

    resample_type: str
    sample_axis: int
    n_boot: int

    ch_meson_a: int
    ch_meson_b: int
    ch_tetra: int
    ch_tetra_MF: int
    fit_mom_by_ns: Dict[int, List[int]]
    fit_mom_by_ns_MF: Dict[int, List[int]]
    scattering_fit_mode: str
    rest_zeta_lamda: int
    rest_zeta_n_q: int
    regen_rest_zeta: bool
    moving_frame_zeta_lamda: int
    moving_frame_zeta_n_q: int
    regen_moving_frame_zeta: bool
    regen_moving_frame_scattering: bool
    k_sq_plot_range: Tuple[float, float]
    s_plot_range: Tuple[float, float]
    moving_frame_d_vec: Tuple[int, int, int]

    t_run_start: int
    t_run_stop: int | None
    t_run_step: int
    t_run_end_offset: int
    ratio_ta: int
    ratio_scan_points: List[RatioScanPoint]
    plot_tmin: bool


@dataclass
class BuildConfig:
    input_name: str

    def __post_init__(self):
        module_name = f"input.input_{self.input_name}"
        self.input_module = importlib.import_module(module_name)

        for name in ["InputControl", "ENSEMBLE_DB"]:
            if not hasattr(self.input_module, name):
                raise ValueError(f"{module_name} missing {name}")

        self.input_control = self.input_module.InputControl()
        self.ensemble_db = self.input_module.ENSEMBLE_DB
        self.ensemble_key = self.input_control.ensemble_key()

    @staticmethod
    def _reorder_prior(raw_prior):
        arr = np.array(raw_prior, dtype=float)
        return np.transpose(arr, (1, 2, 0)).tolist()

    def build_config_from_control(self, analysis_type: str | None = None) -> Config:
        ctrl = self.input_control
        key = self.ensemble_key

        lattice_Ns, lattice_Nt, pion_mass, num_eigenvectors = key
        tag_name = ensemble_tag(key)

        analysis_type = analysis_type or ctrl.analysis_type()
        if analysis_type not in ("meson", "tetraquark"):
            raise ValueError(f"Unknown analysis_type: {analysis_type}")
        if key not in self.ensemble_db:
            raise ValueError(f"Unknown ensemble key: {key}")

        db = self.ensemble_db[key][analysis_type]

        chan_momt_list = db["chan_momt_list"]
        chan_name_list = db["chan_name_list"]

        t_min = db.get("tmin_arry", [[20] * len(m) for m in chan_momt_list])
        t_max = [[lattice_Nt - t for t in row] for row in t_min]

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
            is_meson_analysis=analysis_type == "meson",
            is_tetraquark_analysis=analysis_type == "tetraquark",
            is_gevp=ctrl.is_gevp,
            is_ratio=ctrl.is_ratio,
            run_tmin=ctrl.run_tmin,
            run_resample=ctrl.run_resample,
            run_scattering=ctrl.run_scattering,
            is_moving_frame=ctrl.is_moving_frame,
            plot_meff=ctrl.plot_meff,
            plot_dispersion=ctrl.plot_dispersion,
            plot_format=ctrl.plot_format,
            resample_type=ctrl.resample_type,
            sample_axis=ctrl.sample_axis,
            n_boot=ctrl.n_boot,
            ch_meson_a=ctrl.ch_meson_a,
            ch_meson_b=ctrl.ch_meson_b,
            ch_tetra=ctrl.ch_tetra,
            ch_tetra_MF=ctrl.ch_tetra_MF,
            fit_mom_by_ns=ctrl.fit_mom_by_ns,
            fit_mom_by_ns_MF=ctrl.fit_mom_by_ns_MF,
            scattering_fit_mode=ctrl.scattering_fit_mode,
            rest_zeta_lamda=ctrl.rest_zeta_lamda,
            rest_zeta_n_q=ctrl.rest_zeta_n_q,
            regen_rest_zeta=ctrl.regen_rest_zeta,
            moving_frame_zeta_lamda=ctrl.moving_frame_zeta_lamda,
            moving_frame_zeta_n_q=ctrl.moving_frame_zeta_n_q,
            regen_moving_frame_zeta=ctrl.regen_moving_frame_zeta,
            regen_moving_frame_scattering=ctrl.regen_moving_frame_scattering,
            k_sq_plot_range=ctrl.k_sq_plot_range,
            s_plot_range=ctrl.s_plot_range,
            moving_frame_d_vec=ctrl.moving_frame_d_vec,
            t_run_start=ctrl.t_run_start,
            t_run_stop=ctrl.t_run_stop,
            t_run_step=ctrl.t_run_step,
            t_run_end_offset=ctrl.t_run_end_offset,
            ratio_ta=ctrl.ratio_ta,
            ratio_scan_points=list(ctrl.ratio_scan_points),
            plot_tmin=ctrl.plot_tmin,
        )


class SelectorType:
    """Return the active Correlator4D and cosh fit model for the current analysis mode."""

    def __init__(self, config: Config, corr: AnalysisCorrelators):
        self.config = config
        self.corr = corr

    def get_data(self) -> Correlator4D:
        return self.corr.active(
            self.config.is_meson_analysis,
            self.config.is_tetraquark_analysis,
        )

    def get_model(self) -> tuple[ModelFn, PriorFn]:
        key = {2: "two_states", 3: "three_states"}.get(self.config.n_state)
        if key is None:
            raise ValueError(f"Unsupported n_state={self.config.n_state}")
        return MODEL_REGISTRY[key]["fn"], MODEL_REGISTRY[key]["prior"]
