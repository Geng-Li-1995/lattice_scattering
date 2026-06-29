"""Lattice input: user switches, ensemble database, and channel-name resolution.

Each system defines ``input/input_<Name>.py`` with:

- ``InputControl``: ``InputControlBase`` subclass (effective mass / scattering / plot switches).
- ``ENSEMBLE_DB``: per-ensemble meson/tetraquark channels, ``t_min``, fit priors.
- ``get_lattice_params``: map ``lattice_Ns`` → full ``EnsembleKey``.

``BuildConfig`` loads a system module and builds runtime ``Config`` objects.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import numpy as np

from analysis.models import MODEL_REGISTRY
from data.correlators import AnalysisCorrelators, Correlator4D

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
EnsembleKey: TypeAlias = Tuple[int, int, int, int]  # (Ns, Nt, pion_mass, num_eigenvectors)
PlotLimits: TypeAlias = Tuple[float, float, float, float]  # x_lo, x_hi, y_lo, y_hi
ScatteringList: TypeAlias = List[EnsembleKey]
ResampleDataDict: TypeAlias = Dict[str, Dict[EnsembleKey, np.ndarray]]
RatioScanPoint: TypeAlias = Tuple[int, ...]
ModelFn: TypeAlias = Callable[..., Any]
PriorFn: TypeAlias = Callable[..., Dict[str, Any]]
FitResultList: TypeAlias = List[Dict[str, Any]]
EnsembleEntry: TypeAlias = Dict[str, Any]

_ANALYSIS_TYPES = frozenset({"meson", "tetraquark"})
_PLOT_FORMATS = frozenset({"png", "pdf"})
_N_STATE_MODEL = {2: "two_states", 3: "three_states"}


# ---------------------------------------------------------------------------
# Ensemble tags
# ---------------------------------------------------------------------------
def ensemble_tag(ensemble_key: EnsembleKey) -> str:
    Ns, _, pion_mass, num_ev = ensemble_key
    return f"L{Ns}M{pion_mass}_EV{num_ev}"


def MF_d_tag(d_vec: tuple[int, int, int]) -> str:
    return "d" + "".join(str(int(component)) for component in d_vec)


def scattering_Ns_values(
    scattering_Ns_mom: Dict[int, List[int]],
    scattering_Ns_mom_MF: Dict[int, List[int]] | None = None,
) -> list[int]:
    """Spatial sizes (L) for scattering: keys of rest (+ moving-frame) mom dicts."""
    mf = scattering_Ns_mom_MF or {}
    return sorted(set(scattering_Ns_mom) | set(mf))


# ---------------------------------------------------------------------------
# User input (input/input_*.py)
# ---------------------------------------------------------------------------
class InputControlMixin:
    """Common post-init; subclasses supply defaults and ``get_lattice_params``."""

    lattice_Ns: int
    lattice_Nt: int
    pion_mass: int
    num_eigenvectors: int
    plot_format: str
    is_plot_title: bool
    is_plot_show: bool
    scattering_Ns_mom: Dict[int, List[int]]
    scattering_Ns_mom_MF: Dict[int, List[int]]
    scattering_list: ScatteringList
    run_meson_analysis: bool
    run_dispersion_analysis: bool
    run_weight_analysis: bool
    run_tetraquark_analysis: bool
    run_GEVP_analysis: bool
    run_ratio_analysis: bool
    is_ratio_shift: bool
    run_tmin_analysis: bool
    run_resample_analysis: bool
    run_scattering_analysis: bool

    def __post_init__(self) -> None:
        if self.plot_format not in _PLOT_FORMATS:
            raise ValueError(
                f'plot_format must be "png" or "pdf", got {self.plot_format!r}'
            )
        self.lattice_Ns, self.lattice_Nt, self.pion_mass, self.num_eigenvectors = (
            self.get_lattice_params(self.lattice_Ns)
        )
        ns_list = scattering_Ns_values(self.scattering_Ns_mom, self.scattering_Ns_mom_MF)
        self.scattering_list = [self.get_lattice_params(ns) for ns in ns_list]

    @staticmethod
    def get_lattice_params(lattice_Ns: int) -> EnsembleKey:
        raise NotImplementedError

    def ensemble_key(self) -> EnsembleKey:
        return (self.lattice_Ns, self.lattice_Nt, self.pion_mass, self.num_eigenvectors)

    def analysis_type(self) -> str:
        if self.run_tetraquark_analysis:
            return "tetraquark"
        if self.run_meson_analysis:
            return "meson"
        raise ValueError("Neither meson nor tetraquark analysis is selected.")


@dataclass
class InputControlBase(InputControlMixin):
    """Editable switches; override per system in ``input/input_*.py``."""

    tetraquark_name: str = ""

    lattice_Nt: int = 0
    num_eigenvectors: int = 0
    pion_mass: int = 0  # filled by get_lattice_params(lattice_Ns)

    # --- Effective mass ---
    lattice_Ns: int = 16
    t_run_start: int = 10
    t_run_stop: int | None = None  # default: lattice_Nt//2 - t_run_end_offset
    t_run_step: int = 1
    t_run_end_offset: int = 1
    ratio_at: int = 1  # lattice offset a_t in R_n(t+a_t); step = 2*ratio_at
    resample_type: str = "jackknife"  # "jackknife" | "bootstrap"
    sample_axis: int = -1
    n_boot: int = 1000

    # --- Scattering ---
    scattering_channel: str = ""
    scattering_Ns_mom: Dict[int, List[int]] = field(default_factory=dict)
    scattering_fit_mode: str = "Ks_linear"  # "Ks_linear" | "kcot_quadratic"
    rest_zeta_lamda: int = 50
    rest_zeta_n_q: int = 100_000
    regen_rest_zeta: bool = False
    run_MF_analysis: bool = False
    scattering_channel_MF: str | None = None
    scattering_Ns_mom_MF: Dict[int, List[int]] = field(default_factory=dict)
    MF_zeta_lamda: int = 50
    MF_zeta_n_q: int = 1000
    regen_MF_zeta: bool = False
    regen_MF_scattering: bool = False
    MF_d_vec: tuple[int, int, int] = (0, 0, 1)

    # --- Plot options ---
    plot_format: str = "png"
    is_plot_title: bool = True
    is_plot_show: bool = True
    k_sq_plot_range: PlotLimits = (-0.1, 0.5, -2.0, 2.0)
    s_plot_range: PlotLimits = (14, 18, -0.5, 2.5)

    scattering_list: ScatteringList = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime config (InputControl + ENSEMBLE_DB)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Frozen configuration for one analysis branch at one ensemble."""

    input_name: str
    ensemble_key: EnsembleKey
    scattering_list: ScatteringList
    tag_name: str
    lattice_Nt: int
    at_invs: float
    chan_momentum_list: List[List[int]]
    chan_name_list: List[str]
    meson_chan_momentum_list: List[List[int]]
    meson_chan_name_list: List[str]
    tetra_chan_name_list: List[str]
    t_GEVP: Tuple[int, int, int]
    t_min: List[List[int]]
    t_max: List[List[int]]
    n_state: int
    meff_prior: List[List[List[float]]]
    weff_prior: List[List[List[float]]]
    meff_prior_error: List[float]
    weff_prior_error: List[float]
    run_meson_analysis: bool
    run_dispersion_analysis: bool
    run_weight_analysis: bool
    run_tetraquark_analysis: bool
    run_GEVP_analysis: bool
    run_ratio_analysis: bool
    is_ratio_shift: bool
    run_tmin_analysis: bool
    run_resample_analysis: bool
    t_run_start: int
    t_run_stop: int | None
    t_run_step: int
    t_run_end_offset: int
    ratio_at: int
    resample_type: str
    sample_axis: int
    n_boot: int
    plot_format: str
    is_plot_title: bool
    is_plot_show: bool
    run_scattering_analysis: bool
    scattering_chan: str
    scattering_chan_MF: str | None
    scattering_Ns_mom: Dict[int, List[int]]
    scattering_Ns_mom_MF: Dict[int, List[int]]
    scattering_fit_mode: str
    run_MF_analysis: bool
    rest_zeta_lamda: int
    rest_zeta_n_q: int
    regen_rest_zeta: bool
    MF_zeta_lamda: int
    MF_zeta_n_q: int
    regen_MF_zeta: bool
    regen_MF_scattering: bool
    k_sq_plot_range: PlotLimits
    s_plot_range: PlotLimits
    MF_d_vec: Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Channel labels → indices (field-based matching)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChanMom:
    """Channel index + momentum n²."""

    chan: int
    mom: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.chan, self.mom)


@dataclass(frozen=True)
class ScatteringChanMatch:
    """``scattering_channel`` label resolved to chan indices."""

    meson_a: int
    meson_b: int
    tetra: int


@dataclass(frozen=True)
class ScatteringChanIndices:
    """Rest + moving-frame tetra indices for one ensemble."""

    meson_a: int
    meson_b: int
    tetra: int
    tetra_mf: int


def split_tetra_chan_name(name: str) -> list[str]:
    """Split a tetraquark channel label into meson LaTeX tokens."""
    if r"\," in name:
        parts = name.split(r"\,")
    elif "," in name:
        parts = name.split(",")
    else:
        parts = [name]
    return [part.strip() for part in parts if part.strip()]


def normalize_chan_label(label: str) -> str:
    return label.replace(r"\,", "").replace(",", "").replace(" ", "")


def meson_chan_index(name: str, meson_names: list[str]) -> int:
    try:
        return meson_names.index(name)
    except ValueError as exc:
        raise ValueError(
            f"Meson {name!r} not found in meson chan_name_list {meson_names}"
        ) from exc


def tetra_chan_index(name: str, tetra_names: list[str]) -> int:
    try:
        return tetra_names.index(name)
    except ValueError as exc:
        raise ValueError(
            f"Tetraquark {name!r} not found in tetraquark chan_name_list {tetra_names}"
        ) from exc


def find_tetra_chan_by_label(label: str, tetra_names: list[str]) -> int:
    key = normalize_chan_label(label)
    for idx, name in enumerate(tetra_names):
        if normalize_chan_label(name) == key:
            return idx
    raise ValueError(
        f"scattering_chan {label!r} does not match any tetraquark label "
        f"(normalized keys: {[normalize_chan_label(n) for n in tetra_names]})"
    )


def chan_mom_by_label(
    label: str,
    mom: int,
    chan_names: list[str],
    *,
    normalize: bool = False,
) -> ChanMom:
    """Resolve ``(chan, mom)`` from a channel label and n²."""
    if normalize:
        chan = find_tetra_chan_by_label(label, chan_names)
    else:
        chan = meson_chan_index(label, chan_names)
    return ChanMom(chan, mom)


def resolve_scattering_chan(
    label: str,
    meson_names: list[str],
    tetra_names: list[str],
) -> ScatteringChanMatch:
    """``scattering_channel`` label → meson a/b and tetra chan indices."""
    tetra_chan = find_tetra_chan_by_label(label, tetra_names)
    parts = split_tetra_chan_name(tetra_names[tetra_chan])
    if len(parts) != 2:
        raise ValueError(
            f"Tetraquark {tetra_names[tetra_chan]!r} has {len(parts)} meson parts; "
            "expected 2"
        )
    return ScatteringChanMatch(
        meson_a=meson_chan_index(parts[0], meson_names),
        meson_b=meson_chan_index(parts[1], meson_names),
        tetra=tetra_chan,
    )


def scattering_chan_indices_for_ensemble(
    ensemble_db: dict,
    ensemble_key: EnsembleKey,
    scattering_chan: str,
    scattering_chan_MF: str | None = None,
) -> ScatteringChanIndices:
    entry = ensemble_db[ensemble_key]
    meson_names = entry["meson"]["channel_name_list"]
    tetra_names = entry["tetraquark"]["channel_name_list"]
    rest = resolve_scattering_chan(scattering_chan, meson_names, tetra_names)
    mf_label = scattering_chan_MF or scattering_chan
    mf = resolve_scattering_chan(mf_label, meson_names, tetra_names)
    return ScatteringChanIndices(
        meson_a=rest.meson_a,
        meson_b=rest.meson_b,
        tetra=rest.tetra,
        tetra_mf=mf.tetra,
    )


def scattering_momenta_from_db(
    ensemble_db: dict,
    ensemble_key: EnsembleKey,
    scattering_chan: str,
) -> list[int]:
    entry = ensemble_db[ensemble_key]
    meson_names = entry["meson"]["channel_name_list"]
    tetra_names = entry["tetraquark"]["channel_name_list"]
    match = resolve_scattering_chan(scattering_chan, meson_names, tetra_names)
    return list(entry["tetraquark"]["channel_momentum_list"][match.tetra])


def resolve_mesons_from_tetra_chan(
    tetra_chan: int, tetra_mom: int, config: Config
) -> tuple[int, int, int, int]:
    """Meson (chan, mom) for a/b from tetraquark label and matching n^2."""
    meson_names = config.meson_chan_name_list
    meson_moms = config.meson_chan_momentum_list
    if not meson_names or not meson_moms:
        raise ValueError("Config missing meson_chan_name_list / meson_chan_momentum_list")

    tetra_name = config.chan_name_list[tetra_chan]
    parts = split_tetra_chan_name(tetra_name)
    if len(parts) == 1:
        ma_chan = meson_chan_index(parts[0], meson_names)
        mb_chan = ma_chan
    elif len(parts) == 2:
        ma_chan = meson_chan_index(parts[0], meson_names)
        mb_chan = meson_chan_index(parts[1], meson_names)
    else:
        raise ValueError(
            f"Tetraquark {tetra_name!r} has {len(parts)} meson tokens; expected 1 or 2"
        )

    ma_mom = _validate_meson_momentum(ma_chan, tetra_mom, meson_names, meson_moms)
    mb_mom = _validate_meson_momentum(mb_chan, tetra_mom, meson_names, meson_moms)
    return ma_chan, ma_mom, mb_chan, mb_mom


def _validate_meson_momentum(
    meson_chan: int, mom: int, meson_names: list[str], meson_momt_list: list[list[int]]
) -> int:
    available = meson_momt_list[meson_chan]
    if mom not in available:
        raise ValueError(
            f"n^2={mom} not in meson chan {meson_chan} ({meson_names[meson_chan]}); "
            f"available {available}"
        )
    return mom


# ---------------------------------------------------------------------------
# Ratio t_min scan
# ---------------------------------------------------------------------------
def ratio_chan_labels(config: Config) -> list[str]:
    return list(config.tetra_chan_name_list)


def ratio_scan_points_for_config(config: Config) -> List[RatioScanPoint]:
    if not config.run_ratio_analysis:
        return []
    if not config.tetra_chan_name_list:
        raise ValueError("run_ratio_analysis=True requires tetraquark chan_name_list")
    return [point.key for point in iter_chan_mom_points(config)]


def iter_chan_mom_points(config: Config) -> list[ChanMom]:
    """All (chan, mom) scan points on the active branch."""
    return [
        ChanMom(chan, mom)
        for chan, mom_list in enumerate(config.chan_momentum_list)
        for mom in mom_list
    ]


def ratio_point_by_label(config: Config, tetra_label: str, mom: int) -> ChanMom:
    """Ratio / t_min lookup key from tetraquark label and n²."""
    return chan_mom_by_label(
        tetra_label, mom, config.tetra_chan_name_list, normalize=True
    )


# ---------------------------------------------------------------------------
# BuildConfig: InputControl + ENSEMBLE_DB → Config
# ---------------------------------------------------------------------------
def _prior_mean_keys(db: EnsembleEntry, stem: str) -> list[str]:
    prefix = f"{stem}_"
    keys = [k for k in db if k.startswith(prefix) and k[len(prefix) :].isdigit()]
    return sorted(keys, key=lambda k: int(k[len(prefix) :]))


def _load_prior_errors(db: EnsembleEntry, stem: str, n_state: int) -> list[float]:
    errors: list[float] = []
    for i in range(n_state):
        key = f"{stem}_{i}_error"
        if key not in db:
            raise ValueError(f"ENSEMBLE_DB missing {key!r}")
        errors.append(float(db[key]))
    return errors


def _reorder_prior(raw_prior: list) -> list:
    return np.transpose(np.asarray(raw_prior, dtype=float), (1, 2, 0)).tolist()


def _fit_windows(db: EnsembleEntry, lattice_Nt: int) -> tuple[list, list]:
    chan_momentum_list = db["channel_momentum_list"]
    t_min = db.get("tmin_selected", [[20] * len(moms) for moms in chan_momentum_list])
    t_max = [[lattice_Nt - t for t in row] for row in t_min]
    return t_min, t_max


def _load_fit_priors(db: EnsembleEntry) -> dict[str, Any]:
    meff_keys = _prior_mean_keys(db, "prior_meff")
    weff_keys = _prior_mean_keys(db, "prior_weff")
    if len(meff_keys) != len(weff_keys):
        raise ValueError(
            f"prior_meff / prior_weff state count mismatch: "
            f"{len(meff_keys)} vs {len(weff_keys)}"
        )
    n_state = len(meff_keys)
    return {
        "n_state": n_state,
        "meff_prior": _reorder_prior([db[k] for k in meff_keys]),
        "weff_prior": _reorder_prior([db[k] for k in weff_keys]),
        "meff_prior_error": _load_prior_errors(db, "prior_meff", n_state),
        "weff_prior_error": _load_prior_errors(db, "prior_weff", n_state),
    }


def _branch_switches(ctrl: InputControlBase, analysis_type: str) -> dict[str, bool]:
    is_meson = analysis_type == "meson"
    is_tetra = analysis_type == "tetraquark"
    return {
        "run_meson_analysis": is_meson and ctrl.run_meson_analysis,
        "run_dispersion_analysis": is_meson and ctrl.run_dispersion_analysis,
        "run_weight_analysis": is_meson and ctrl.run_weight_analysis,
        "run_tetraquark_analysis": is_tetra and ctrl.run_tetraquark_analysis,
        "run_GEVP_analysis": is_tetra and ctrl.run_GEVP_analysis,
        "run_ratio_analysis": is_tetra and ctrl.run_ratio_analysis,
        "is_ratio_shift": ctrl.is_ratio_shift,
        "run_tmin_analysis": ctrl.run_tmin_analysis
        and ((is_meson and ctrl.run_meson_analysis) or (is_tetra and ctrl.run_tetraquark_analysis)),
        "run_resample_analysis": ctrl.run_resample_analysis,
        "run_scattering_analysis": ctrl.run_scattering_analysis,
        "run_MF_analysis": ctrl.run_MF_analysis,
    }


@dataclass
class BuildConfig:
    """Load ``input.input_<input_name>`` and build ``Config`` instances."""

    input_name: str

    def __post_init__(self) -> None:
        module_name = f"input.input_{self.input_name}"
        self.input_module = importlib.import_module(module_name)
        for name in ("InputControl", "ENSEMBLE_DB"):
            if not hasattr(self.input_module, name):
                raise ValueError(f"{module_name} missing {name}")
        self.input_control = self.input_module.InputControl()
        self.ensemble_db = self.input_module.ENSEMBLE_DB
        self.ensemble_key = self.input_control.ensemble_key()

    def build_config_from_control(self, analysis_type: str | None = None) -> Config:
        ctrl = self.input_control
        analysis_type = analysis_type or ctrl.analysis_type()
        if analysis_type not in _ANALYSIS_TYPES:
            raise ValueError(f"Unknown analysis_type: {analysis_type!r}")

        key = self.ensemble_key
        if key not in self.ensemble_db:
            raise ValueError(f"Unknown ensemble key: {key}")

        entry = self.ensemble_db[key]
        db = entry[analysis_type]
        meson_db = entry["meson"]
        _, lattice_Nt, _, _ = key
        t_min, t_max = _fit_windows(db, lattice_Nt)

        return Config(
            input_name=self.input_name,
            ensemble_key=key,
            scattering_list=ctrl.scattering_list,
            tag_name=ensemble_tag(key),
            lattice_Nt=lattice_Nt,
            at_invs=entry["at_invs"],
            chan_momentum_list=db["channel_momentum_list"],
            chan_name_list=db["channel_name_list"],
            meson_chan_momentum_list=meson_db["channel_momentum_list"],
            meson_chan_name_list=meson_db["channel_name_list"],
            tetra_chan_name_list=entry["tetraquark"]["channel_name_list"],
            t_GEVP=entry["GEVP"],
            t_min=t_min,
            t_max=t_max,
            **_load_fit_priors(db),
            **_branch_switches(ctrl, analysis_type),
            t_run_start=ctrl.t_run_start,
            t_run_stop=ctrl.t_run_stop,
            t_run_step=ctrl.t_run_step,
            t_run_end_offset=ctrl.t_run_end_offset,
            ratio_at=ctrl.ratio_at,
            resample_type=ctrl.resample_type,
            sample_axis=ctrl.sample_axis,
            n_boot=ctrl.n_boot,
            plot_format=ctrl.plot_format,
            is_plot_title=ctrl.is_plot_title,
            is_plot_show=ctrl.is_plot_show,
            scattering_chan=ctrl.scattering_channel,
            scattering_chan_MF=ctrl.scattering_channel_MF,
            scattering_Ns_mom=ctrl.scattering_Ns_mom,
            scattering_Ns_mom_MF=ctrl.scattering_Ns_mom_MF,
            scattering_fit_mode=ctrl.scattering_fit_mode,
            rest_zeta_lamda=ctrl.rest_zeta_lamda,
            rest_zeta_n_q=ctrl.rest_zeta_n_q,
            regen_rest_zeta=ctrl.regen_rest_zeta,
            MF_zeta_lamda=ctrl.MF_zeta_lamda,
            MF_zeta_n_q=ctrl.MF_zeta_n_q,
            regen_MF_zeta=ctrl.regen_MF_zeta,
            regen_MF_scattering=ctrl.regen_MF_scattering,
            k_sq_plot_range=ctrl.k_sq_plot_range,
            s_plot_range=ctrl.s_plot_range,
            MF_d_vec=ctrl.MF_d_vec,
        )


# ---------------------------------------------------------------------------
# Active correlator + fit model for current analysis mode
# ---------------------------------------------------------------------------
class SelectorType:
    def __init__(self, config: Config, corr: AnalysisCorrelators):
        self.config = config
        self.corr = corr

    def get_data(self) -> Correlator4D:
        return self.corr.active(
            self.config.run_meson_analysis,
            self.config.run_tetraquark_analysis,
        )

    def get_model(self) -> tuple[ModelFn, PriorFn]:
        key = _N_STATE_MODEL.get(self.config.n_state)
        if key is None:
            raise ValueError(f"Unsupported n_state={self.config.n_state}")
        return MODEL_REGISTRY[key]["fn"], MODEL_REGISTRY[key]["prior"]
