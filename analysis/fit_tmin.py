"""Scan fit windows in t_min and 4Q/2Q ratio fits."""

from __future__ import annotations

from dataclasses import dataclass

import gvar as gv
import lsqfit as lsf
import numpy as np

from analysis.models import MathModels, MODEL_REGISTRY
from input.config import Config, RatioScanPoint


@dataclass(frozen=True)
class RatioScanTarget:
    """One (channel, momentum) entry for combined 3-state + ratio t_min scan."""

    tetra_ch: int
    tetra_mom: int
    meson_ch: int
    meson_mom: int
    state_idx: int
    delta_prior: float


@dataclass
class TminScanResult:
    t_min_list: np.ndarray
    meff: list
    chi2_dof: np.ndarray
    ref_t_min: int
    ref_index: int


@dataclass
class RatioScanResult:
    t_min_list: np.ndarray
    delta_m: list
    chi2_dof: np.ndarray
    cosh_scan: TminScanResult


def t_run_list(config: Config) -> np.ndarray:
    stop = config.t_run_stop
    if stop is None:
        stop = config.lattice_Nt // 2 - config.t_run_end_offset
    return np.arange(config.t_run_start, stop, config.t_run_step)


def _chi2_dof(fit, dof_adjust: int) -> float:
    return float(fit.chi2 / max(fit.dof - dof_adjust, 1))


def _cosh_dof_adjust(n_state: int) -> int:
    return {2: 3, 3: 5}.get(n_state, n_state + 2)


def _primary_meff_key(n_state: int) -> str:
    return "meff_0"


def _cosh_model(config: Config):
    key = {2: "two_states", 3: "three_states"}.get(config.n_state)
    if key is None:
        raise ValueError(f"Unsupported n_state={config.n_state}")
    return MODEL_REGISTRY[key]["fn"], MODEL_REGISTRY[key]["prior"]


def fit_cosh_at_window(
    config: Config,
    data_1d: np.ndarray,
    t_min: int,
    t_max: int,
    ch_idx: int,
    mom: int,
) -> lsf.nonlinear_fit:
    """Fit one (time, sample) correlator slice; ``data_1d`` is ``Correlator4D.at(ch, mom)``."""
    model_fn, prior_fn = _cosh_model(config)
    t_fit = np.arange(t_min, t_max + 1)
    norm = data_1d[config.lattice_Nt // 2].mean()
    y_gv = gv.dataset.avg_data(data_1d[t_min : t_max + 1].T / norm)
    fit = lsf.nonlinear_fit(
        data=(t_fit, y_gv),
        fcn=lambda t, p: model_fn(t, p, lattice_Nt=config.lattice_Nt),
        prior=prior_fn(
            config.meff_prior[ch_idx][mom],
            config.weff_prior[ch_idx][mom],
        ),
        p0=None,
    )
    for key in fit.p:
        if key.startswith("weff_"):
            fit.p[key] *= norm
    return fit


def _ref_index(t_mins: np.ndarray, ref_t_min: int) -> int:
    return int(np.argmin(np.abs(t_mins - ref_t_min)))


def scan_cosh_tmin(
    config: Config,
    data_1d: np.ndarray,
    ch_idx: int,
    mom: int,
    t_mins: np.ndarray | None = None,
) -> TminScanResult:
    t_mins = np.asarray(t_run_list(config) if t_mins is None else t_mins)
    meff_key = _primary_meff_key(config.n_state)
    dof_adj = _cosh_dof_adjust(config.n_state)
    meff_list = []
    chi2_list = []
    for t_min in t_mins:
        t_max = config.lattice_Nt - int(t_min)
        fit = fit_cosh_at_window(config, data_1d, int(t_min), t_max, ch_idx, mom)
        meff_list.append(fit.p[meff_key])
        chi2_list.append(_chi2_dof(fit, dof_adj))
    ref_t_min = config.t_min[ch_idx][mom]
    ref_index = _ref_index(t_mins, ref_t_min)
    return TminScanResult(t_mins, meff_list, np.asarray(chi2_list), ref_t_min, ref_index)


def build_ratio_series(
    tetra_1d: np.ndarray,
    meson_1d: np.ndarray,
    ta: int,
    half_Nt: int,
) -> np.ndarray:
    """R(t) = (C4(t)-C4(t+ta)) / (C2(t)^2 - C2(t+ta)^2), shifted by ta/2.

    ``tetra_1d`` and ``meson_1d`` have shape (time, sample).
    """
    ratio = (tetra_1d - np.roll(tetra_1d, -ta, axis=0)) / (
        meson_1d**2 - np.roll(meson_1d, -ta, axis=0) ** 2
    )
    return np.roll(ratio, int(ta / 2), axis=0)[:half_Nt]


def fit_meson_mass_for_ratio(
    meson_config: Config,
    meson_1d: np.ndarray,
    meson_ch: int,
    meson_mom: int,
) -> float:
    """Single-meson mass for ratio denominator (old ``fit_m1`` from 2-state cosh)."""
    t_min = meson_config.t_min[meson_ch][meson_mom]
    t_max = meson_config.t_max[meson_ch][meson_mom]
    fit = fit_cosh_at_window(meson_config, meson_1d, t_min, t_max, meson_ch, meson_mom)
    return float(gv.mean(fit.p[_primary_meff_key(meson_config.n_state)]))


def fit_ratio_at_window(
    config: Config,
    ratio_data: np.ndarray,
    t_min: int,
    half_Nt: int,
    meson_m: float,
    delta_prior: float,
) -> lsf.nonlinear_fit:
    t_fit = np.arange(t_min, half_Nt)
    y_gv = gv.dataset.avg_data(ratio_data[t_min:half_Nt].T)
    prior = MathModels.prior_ratio_delta(delta_prior)
    return lsf.nonlinear_fit(
        data=(t_fit, y_gv),
        fcn=lambda t, p: MathModels.ratio(t, p, config.lattice_Nt, meson_m),
        prior=prior,
        p0=None,
    )


def scan_ratio_tmin(
    config: Config,
    tetra_1d: np.ndarray,
    meson_1d: np.ndarray,
    tetra_ch: int,
    tetra_mom: int,
    meson_config: Config,
    meson_ch: int,
    meson_mom: int,
    delta_prior: float,
    t_mins: np.ndarray | None = None,
) -> RatioScanResult:
    t_mins = np.asarray(t_run_list(config) if t_mins is None else t_mins)
    half_Nt = config.lattice_Nt // 2
    ratio_data = build_ratio_series(tetra_1d, meson_1d, config.ratio_ta, half_Nt)
    meson_m = fit_meson_mass_for_ratio(meson_config, meson_1d, meson_ch, meson_mom)
    delta_list = []
    chi2_list = []
    for t_min in t_mins:
        fit = fit_ratio_at_window(
            config, ratio_data, int(t_min), half_Nt, meson_m, delta_prior
        )
        delta_list.append(fit.p["delta_m"])
        chi2_list.append(_chi2_dof(fit, 2))
    cosh_scan = scan_cosh_tmin(config, tetra_1d, tetra_ch, tetra_mom, t_mins)
    return RatioScanResult(t_mins, delta_list, np.asarray(chi2_list), cosh_scan)


def ratio_delta_prior_value(config: Config, tetra_ch: int, tetra_mom: int) -> float:
    """Tetraquark energy prior for ratio fits (from meff_prior)."""
    return float(config.meff_prior[tetra_ch][tetra_mom][0])


def resolve_ratio_pair(point: RatioScanPoint) -> tuple[int, int, int, int]:
    """(tetra_ch, n^2) → (tetra_ch, tetra_mom, meson_ch, meson_mom).

    Two-tuple: meson indices match tetra. Four-tuple: explicit meson override.
    """
    if len(point) == 2:
        ch, mom = int(point[0]), int(point[1])
        return ch, mom, ch, mom
    if len(point) == 4:
        return int(point[0]), int(point[1]), int(point[2]), int(point[3])
    raise ValueError(
        f"ratio_scan_points entries must be (ch, mom) or "
        f"(tetra_ch, tetra_mom, meson_ch, meson_mom); got {point!r}"
    )


def iter_scan_points(config: Config):
    for point in config.ratio_scan_points:
        yield resolve_ratio_pair(point)


def ratio_scan_lookup(config: Config) -> dict[tuple[int, int], RatioScanTarget]:
    """Map (tetra_ch, tetra_mom) → ratio scan parameters."""
    lookup: dict[tuple[int, int], RatioScanTarget] = {}
    for state_idx, (t_ch, t_mom, m_ch, m_mom) in enumerate(iter_scan_points(config)):
        lookup[(t_ch, t_mom)] = RatioScanTarget(
            tetra_ch=t_ch,
            tetra_mom=t_mom,
            meson_ch=m_ch,
            meson_mom=m_mom,
            state_idx=state_idx,
            delta_prior=ratio_delta_prior_value(config, t_ch, t_mom),
        )
    return lookup
