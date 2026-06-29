"""Scan fit windows in t_min and ratio fits (shifted ΔC₄/ΔC₂² and direct C₄/CₐCᵦ)."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import gvar as gv
import lsqfit as lsf
import numpy as np

from analysis.models import MathModels, MODEL_REGISTRY
from data.correlators import AnalysisCorrelators
from input.config import Config, RatioScanPoint, ratio_scan_points_for_config, resolve_mesons_from_tetra_chan
from statistics.resample import get_resampler, jackknife_gvar


@dataclass(frozen=True)
class RatioPair:
    """Resolved ratio scan point: tetra + meson a/b indices."""

    tetra_chan: int
    tetra_mom: int
    meson_a_chan: int
    meson_a_mom: int
    meson_b_chan: int
    meson_b_mom: int
    is_ratio_shift: bool


@dataclass(frozen=True)
class RatioScanTarget:
    """One (chan, momentum) entry for combined 3-state + ratio t_min scan."""

    tetra_chan: int
    tetra_mom: int
    meson_chan: int
    meson_mom: int
    meson_b_chan: int
    meson_b_mom: int
    state_idx: int
    delta_prior: float
    is_ratio_shift: bool


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


@contextmanager
def _ignore_gvar_overflow():
    """t_min scans probe many windows; some yield ill-conditioned gvar covariances."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="gvar")
        yield


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
    chan_idx: int,
    mom: int,
) -> lsf.nonlinear_fit:
    """Fit one (time, sample) correlator slice; ``data_1d`` is ``Correlator4D.at(chan, mom)``."""
    model_fn, prior_fn = _cosh_model(config)
    t_fit = np.arange(t_min, t_max + 1)
    norm = data_1d[config.lattice_Nt // 2].mean()
    y_gv = gv.dataset.avg_data(data_1d[t_min : t_max + 1].T / norm)
    fit = lsf.nonlinear_fit(
        data=(t_fit, y_gv),
        fcn=lambda t, p: model_fn(t, p, lattice_Nt=config.lattice_Nt),
        prior=prior_fn(
            config.meff_prior[chan_idx][mom],
            config.weff_prior[chan_idx][mom],
            config.meff_prior_error,
            config.weff_prior_error,
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
    chan_idx: int,
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
        with _ignore_gvar_overflow():
            fit = fit_cosh_at_window(config, data_1d, int(t_min), t_max, chan_idx, mom)
        meff_list.append(fit.p[meff_key])
        chi2_list.append(_chi2_dof(fit, dof_adj))
    ref_t_min = config.t_min[chan_idx][mom]
    ref_index = _ref_index(t_mins, ref_t_min)
    return TminScanResult(t_mins, meff_list, np.asarray(chi2_list), ref_t_min, ref_index)


def build_ratio_series(
    tetra_1d: np.ndarray,
    meson_1d: np.ndarray,
    ratio_at: int,
    half_Nt: int,
) -> np.ndarray:
    """Build R_n with shift ``ratio_at`` (a_t in R_n(t+a_t)).

    Uses correlator differences at ``2 * ratio_at`` lattice times, then rolls
    by ``ratio_at``. Points with zero denominator are set to NaN.
    """
    step = 2 * ratio_at
    denom = meson_1d**2 - np.roll(meson_1d, -step, axis=0) ** 2
    numer = tetra_1d - np.roll(tetra_1d, -step, axis=0)
    ratio = np.full(np.shape(numer), np.nan, dtype=np.float64)
    np.divide(numer, denom, out=ratio, where=denom != 0)
    return np.roll(ratio, ratio_at, axis=0)[:half_Nt]


def build_ratio_series_distinguishable(
    tetra_1d: np.ndarray,
    meson_a_1d: np.ndarray,
    meson_b_1d: np.ndarray,
    half_Nt: int,
) -> np.ndarray:
    """R(t) = C4(t) / (C_a(t) C_b(t)) for distinguishable mesons."""
    denom = meson_a_1d * meson_b_1d
    ratio = np.full(np.shape(tetra_1d), np.nan, dtype=np.float64)
    np.divide(tetra_1d, denom, out=ratio, where=denom != 0)
    return ratio[:half_Nt]


def build_ratio_data(
    config: Config,
    tetra_1d: np.ndarray,
    meson_a_1d: np.ndarray,
    meson_b_1d: np.ndarray,
    *,
    is_ratio_shift: bool,
) -> np.ndarray:
    """Ratio time series with shape (half_Nt, n_sample)."""
    half_Nt = config.lattice_Nt // 2
    if is_ratio_shift:
        return build_ratio_series(tetra_1d, meson_a_1d, config.ratio_at, half_Nt)
    return build_ratio_series_distinguishable(
        tetra_1d, meson_a_1d, meson_b_1d, half_Nt
    )


def meson_slices(
    corr: AnalysisCorrelators, target: RatioScanTarget
) -> tuple[np.ndarray, np.ndarray]:
    meson_a = corr.meson.at(target.meson_chan, target.meson_mom)
    if target.is_ratio_shift:
        return meson_a, meson_a
    return meson_a, corr.meson.at(target.meson_b_chan, target.meson_b_mom)


def _ratio_on_jackknife(
    config: Config,
    tetra_jk: np.ndarray,
    meson_a_jk: np.ndarray,
    meson_b_jk: np.ndarray,
    *,
    is_ratio_shift: bool,
) -> np.ndarray:
    n_jack = tetra_jk.shape[-1]
    half_Nt = config.lattice_Nt // 2
    ratio_jk = np.empty((half_Nt, n_jack))
    for j in range(n_jack):
        sl = slice(j, j + 1)
        ratio_jk[:, j] = build_ratio_data(
            config,
            tetra_jk[..., sl],
            meson_a_jk[..., sl],
            meson_b_jk[..., sl],
            is_ratio_shift=is_ratio_shift,
        )[:, 0]
    return ratio_jk


def ratio_series_mean_err(
    config: Config,
    tetra_1d: np.ndarray,
    meson_a_1d: np.ndarray,
    meson_b_1d: np.ndarray,
    *,
    is_ratio_shift: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Jackknife mean/error of R(t); same workflow as ``MassPlotter._cosh_with_error``."""
    ratio_jk = _ratio_on_jackknife(
        config,
        get_resampler(config, tetra_1d).resample(),
        get_resampler(config, meson_a_1d).resample(),
        get_resampler(config, meson_b_1d).resample(),
        is_ratio_shift=is_ratio_shift,
    )
    return jackknife_gvar(config, ratio_jk)


@dataclass(frozen=True)
class RatioReferenceFit:
    target: RatioScanTarget
    fit: lsf.nonlinear_fit
    ratio_data: np.ndarray
    meson_m: float | None
    total_energy: gv.GVar


def fit_ratio_reference(
    config: Config,
    tetra_1d: np.ndarray,
    meson_a_1d: np.ndarray,
    meson_b_1d: np.ndarray,
    target: RatioScanTarget,
    meson_config: Config,
) -> RatioReferenceFit:
    """Single ratio fit at the configured reference ``t_min`` (not a t_min scan)."""
    half_Nt = config.lattice_Nt // 2
    t_min = config.t_min[target.tetra_chan][target.tetra_mom]
    ratio_data = build_ratio_data(
        config, tetra_1d, meson_a_1d, meson_b_1d, is_ratio_shift=target.is_ratio_shift
    )
    if target.is_ratio_shift:
        meson_m = float(
            gv.mean(
                fit_meson_mass_for_ratio(
                    meson_config, meson_a_1d, target.meson_chan, target.meson_mom
                )
            )
        )
        fit = fit_ratio_at_window(
            config, ratio_data, t_min, half_Nt, meson_m, target.delta_prior
        )
        return RatioReferenceFit(
            target=target,
            fit=fit,
            ratio_data=ratio_data,
            meson_m=meson_m,
            total_energy=fit.p["delta_m"],
        )

    meson_m_a = fit_meson_mass_for_ratio(
        meson_config, meson_a_1d, target.meson_chan, target.meson_mom
    )
    meson_m_b = fit_meson_mass_for_ratio(
        meson_config, meson_b_1d, target.meson_b_chan, target.meson_b_mom
    )
    fit = fit_ratio_exponential_at_window(
        config,
        ratio_data,
        t_min,
        half_Nt,
        ratio_delta_e_prior_value(
            config,
            target.tetra_chan,
            target.tetra_mom,
            float(gv.mean(meson_m_a)),
            float(gv.mean(meson_m_b)),
        ),
    )
    return RatioReferenceFit(
        target=target,
        fit=fit,
        ratio_data=ratio_data,
        meson_m=None,
        total_energy=meson_m_a + meson_m_b + fit.p["delta_E"],
    )


def fit_meson_mass_for_ratio(
    meson_config: Config,
    meson_1d: np.ndarray,
    meson_chan: int,
    meson_mom: int,
):
    """Single-meson mass (gvar) for ratio denominator or energy sum."""
    t_min = meson_config.t_min[meson_chan][meson_mom]
    t_max = meson_config.t_max[meson_chan][meson_mom]
    fit = fit_cosh_at_window(meson_config, meson_1d, t_min, t_max, meson_chan, meson_mom)
    return fit.p[_primary_meff_key(meson_config.n_state)]


def _ratio_fit_data_gv(
    config: Config, ratio_data: np.ndarray, t_min: int, half_Nt: int
) -> tuple[np.ndarray, np.ndarray]:
    t_fit = np.arange(t_min, half_Nt)
    y_gv = gv.dataset.avg_data(ratio_data[t_min:half_Nt].T)
    return t_fit, y_gv


def fit_ratio_at_window(
    config: Config,
    ratio_data: np.ndarray,
    t_min: int,
    half_Nt: int,
    meson_m: float,
    delta_prior: float,
) -> lsf.nonlinear_fit:
    t_fit, y_gv = _ratio_fit_data_gv(config, ratio_data, t_min, half_Nt)
    prior = MathModels.prior_ratio_delta(delta_prior)
    return lsf.nonlinear_fit(
        data=(t_fit, y_gv),
        fcn=lambda t, p: MathModels.ratio(t, p, config.lattice_Nt, meson_m),
        prior=prior,
        p0=None,
    )


def fit_ratio_exponential_at_window(
    config: Config,
    ratio_data: np.ndarray,
    t_min: int,
    half_Nt: int,
    delta_e_prior: float,
) -> lsf.nonlinear_fit:
    """Fit R(t) ≈ A exp(-ΔE t); caller adds meson masses for total energy."""
    t_fit, y_gv = _ratio_fit_data_gv(config, ratio_data, t_min, half_Nt)
    prior = MathModels.prior_ratio_exponential(delta_e_prior)
    return lsf.nonlinear_fit(
        data=(t_fit, y_gv),
        fcn=lambda t, p: MathModels.ratio_exponential(t, p, config.lattice_Nt),
        prior=prior,
        p0=None,
    )


def ratio_delta_prior_value(config: Config, tetra_chan: int, tetra_mom: int) -> float:
    """Tetraquark energy prior for identical-meson ratio fits (from meff_prior)."""
    return float(config.meff_prior[tetra_chan][tetra_mom][0])


def ratio_delta_e_prior_value(
    config: Config,
    tetra_chan: int,
    tetra_mom: int,
    meson_m_a: float,
    meson_m_b: float,
) -> float:
    """Binding-energy prior ΔE = E_tetra - m_a - m_b for distinguishable ratio."""
    total = float(config.meff_prior[tetra_chan][tetra_mom][0])
    return total - meson_m_a - meson_m_b


def mesons_are_identical(
    meson_a_chan: int,
    meson_a_mom: int,
    meson_b_chan: int,
    meson_b_mom: int,
) -> bool:
    return meson_a_chan == meson_b_chan and meson_a_mom == meson_b_mom


def resolve_ratio_pair(point: RatioScanPoint, config: Config) -> RatioPair:
    """Resolve scan point → meson indices; method from ``config.is_ratio_shift``."""
    shift = config.is_ratio_shift
    if len(point) == 2:
        t_chan, t_mom = int(point[0]), int(point[1])
        ma_chan, ma_mom, mb_chan, mb_mom = resolve_mesons_from_tetra_chan(
            t_chan, t_mom, config
        )
        return RatioPair(t_chan, t_mom, ma_chan, ma_mom, mb_chan, mb_mom, shift)
    if len(point) == 4:
        t_chan, t_mom, m_chan, m_mom = (int(point[i]) for i in range(4))
        return RatioPair(t_chan, t_mom, m_chan, m_mom, m_chan, m_mom, shift)
    if len(point) == 6:
        t_chan, t_mom, ma_chan, ma_mom, mb_chan, mb_mom = (int(point[i]) for i in range(6))
        return RatioPair(t_chan, t_mom, ma_chan, ma_mom, mb_chan, mb_mom, shift)
    raise ValueError(
        "ratio scan point must be length 2, 4, or 6; got "
        f"{point!r}"
    )


def iter_scan_points(config: Config):
    for point in ratio_scan_points_for_config(config):
        yield resolve_ratio_pair(point, config)


def ratio_scan_lookup(config: Config) -> dict[tuple[int, int], RatioScanTarget]:
    """Map (tetra_chan, tetra_mom) → ratio scan parameters."""
    lookup: dict[tuple[int, int], RatioScanTarget] = {}
    for state_idx, pair in enumerate(iter_scan_points(config)):
        lookup[(pair.tetra_chan, pair.tetra_mom)] = RatioScanTarget(
            tetra_chan=pair.tetra_chan,
            tetra_mom=pair.tetra_mom,
            meson_chan=pair.meson_a_chan,
            meson_mom=pair.meson_a_mom,
            meson_b_chan=pair.meson_b_chan,
            meson_b_mom=pair.meson_b_mom,
            state_idx=state_idx,
            delta_prior=ratio_delta_prior_value(config, pair.tetra_chan, pair.tetra_mom),
            is_ratio_shift=pair.is_ratio_shift,
        )
    return lookup


def _ratio_energies_over_tmins(
    config: Config,
    ratio_data: np.ndarray,
    meson_config: Config,
    meson_a_1d: np.ndarray,
    meson_b_1d: np.ndarray,
    target: RatioScanTarget,
    t_mins: np.ndarray,
    half_Nt: int,
) -> tuple[list, list]:
    energy_list: list = []
    chi2_list: list = []
    if target.is_ratio_shift:
        meson_m = float(
            gv.mean(
                fit_meson_mass_for_ratio(
                    meson_config, meson_a_1d, target.meson_chan, target.meson_mom
                )
            )
        )
        for t_min in t_mins:
            with _ignore_gvar_overflow():
                fit = fit_ratio_at_window(
                    config, ratio_data, int(t_min), half_Nt, meson_m, target.delta_prior
                )
            energy_list.append(fit.p["delta_m"])
            chi2_list.append(_chi2_dof(fit, 2))
        return energy_list, chi2_list

    meson_m_a = fit_meson_mass_for_ratio(
        meson_config, meson_a_1d, target.meson_chan, target.meson_mom
    )
    meson_m_b = fit_meson_mass_for_ratio(
        meson_config, meson_b_1d, target.meson_b_chan, target.meson_b_mom
    )
    delta_e_prior = ratio_delta_e_prior_value(
        config,
        target.tetra_chan,
        target.tetra_mom,
        float(gv.mean(meson_m_a)),
        float(gv.mean(meson_m_b)),
    )
    for t_min in t_mins:
        with _ignore_gvar_overflow():
            fit = fit_ratio_exponential_at_window(
                config, ratio_data, int(t_min), half_Nt, delta_e_prior
            )
        energy_list.append(meson_m_a + meson_m_b + fit.p["delta_E"])
        chi2_list.append(_chi2_dof(fit, 2))
    return energy_list, chi2_list


def scan_ratio_tmin(
    config: Config,
    tetra_1d: np.ndarray,
    meson_a_1d: np.ndarray,
    meson_b_1d: np.ndarray,
    meson_config: Config,
    target: RatioScanTarget,
    t_mins: np.ndarray | None = None,
) -> RatioScanResult:
    t_mins = np.asarray(t_run_list(config) if t_mins is None else t_mins)
    half_Nt = config.lattice_Nt // 2
    ratio_data = build_ratio_data(
        config, tetra_1d, meson_a_1d, meson_b_1d, is_ratio_shift=target.is_ratio_shift
    )
    energy_list, chi2_list = _ratio_energies_over_tmins(
        config,
        ratio_data,
        meson_config,
        meson_a_1d,
        meson_b_1d,
        target,
        t_mins,
        half_Nt,
    )
    cosh_scan = scan_cosh_tmin(config, tetra_1d, target.tetra_chan, target.tetra_mom, t_mins)
    return RatioScanResult(t_mins, energy_list, np.asarray(chi2_list), cosh_scan)
