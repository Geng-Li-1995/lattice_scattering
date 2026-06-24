"""Scattering analysis: zeta functions, phase extraction, and K(s)/kcot fits."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import gvar as gv
import lsqfit as lsf
import numpy as np
from joblib import Parallel, delayed

from analysis.models import MathModels
from data.io import (
    SCATTER_STAT_NAMES,
    load_mf_scatter_all,
    load_mf_scatter_ref,
    mf_scatter_path,
    mf_scatter_ref_path,
    save_mf_scatter_all,
    save_mf_scatter_ref,
)
from input.config import Config, ResampleDataDict, ensemble_tag, moving_frame_d_tag
from statistics.resample import get_resampler

_TWO_PI = 2.0 * np.pi
_KCOT_PREFACTOR = 2.0 / np.sqrt(np.pi)

Q_SQ_MIN = -5.0
Q_SQ_MAX = 10.0
Y_00 = 1.0 / np.sqrt(4.0 * np.pi)

SCATTERING_FIT_MODES = ("Ks_linear", "kcot_quadratic")


# ---------------------------------------------------------------------------
# Zeta helpers
# ---------------------------------------------------------------------------
def q_sq_linspace(n_q: int) -> np.ndarray:
    return np.linspace(Q_SQ_MIN, Q_SQ_MAX, n_q)


def kallen(a, b, c) -> float:
    """Källén function."""
    return a**2 + b**2 + c**2 - 2 * (a * b + a * c + b * c)


def build_n_sq_array(lamda: int) -> np.ndarray:
    n_range = np.arange(-lamda - 1, lamda + 2)
    nx, ny, nz = np.meshgrid(n_range, n_range, n_range, indexing="ij")
    n_sq = nx**2 + ny**2 + nz**2
    return n_sq[n_sq <= lamda**2].astype(float)


def build_r_sq_from_n_vec(
    n_vec_array: np.ndarray,
    d_vec: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    d_hat = d_vec / np.linalg.norm(d_vec)
    x = n_vec_array - 0.5 * alpha * d_vec[None, :]
    x_para = np.dot(x, d_hat)[:, None] * d_hat[None, :]
    x_perp = x - x_para
    r_vec = x_perp + x_para / gamma
    return np.sum(r_vec * r_vec, axis=-1)


def build_moving_frame_n_vec_array(
    lamda: int,
    d_vec: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    shift = 0.5 * abs(alpha) * np.linalg.norm(d_vec)
    max_n = int(np.ceil(max(lamda, gamma * lamda + shift))) + 2
    n_range = np.arange(-max_n, max_n + 1)
    return np.array(np.meshgrid(n_range, n_range, n_range, indexing="ij")).T.reshape(-1, 3)


def gen_zeta_00_rest_from_q_sq_array(
    q_sq_grid: np.ndarray, lamda: int, regen_flag: bool = False
) -> np.ndarray:
    save_path = Path(f"data/zeta/zeta_00_rest_lam{lamda}_nq{len(q_sq_grid)}.npy")
    if (not regen_flag) and save_path.exists():
        print(f"Loading {save_path}")
        return np.load(save_path)

    print(f"Generating {save_path} ...")
    n_sq_array = build_n_sq_array(lamda)
    zeta_array = np.asarray(
        Parallel(n_jobs=-1)(
            delayed(_gen_zeta_00_rest_from_q_sq)(n_sq_array, q_sq, lamda)
            for q_sq in q_sq_grid
        ),
        dtype=float,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, zeta_array)
    print(f"Saved to {save_path}")
    return zeta_array


def gen_zeta_00_moving_from_q_sq_array(
    q_sq_grid: np.ndarray,
    config: Config,
    ensemble_key,
    d_vec: np.ndarray,
    lamda: int,
    gamma: float,
    alpha: float,
) -> np.ndarray:
    d_tag = moving_frame_d_tag(tuple(int(component) for component in d_vec))
    save_path = Path(
        f"data/zeta/zeta_00_moving_{config.input_name}_"
        f"{ensemble_tag(ensemble_key)}_"
        f"{d_tag}_lam{lamda}_nq{len(q_sq_grid)}.npy"
    )
    if (not config.regen_moving_frame_zeta) and save_path.exists():
        print(f"Loading {save_path}")
        return np.load(save_path)

    n_vec_array = build_moving_frame_n_vec_array(lamda, d_vec, alpha, gamma)
    r_sq_array = build_r_sq_from_n_vec(n_vec_array, d_vec, alpha, gamma)
    r_sq_array = r_sq_array[r_sq_array <= lamda**2]

    print(
        f"Generating {save_path} with alpha={alpha:.6g}, gamma={gamma:.6g}, "
        f"n_vec={len(n_vec_array)}, r_vec={len(r_sq_array)}"
    )
    zeta_array = np.asarray(
        [gen_zeta_00_moving_from_q_sq(r_sq_array, q_sq, lamda) for q_sq in q_sq_grid],
        dtype=float,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, zeta_array)
    print(f"Saved to {save_path}")
    return zeta_array


def gen_zeta_00_moving_from_q_sq(r_sq_array: np.ndarray, q_sq: float, lamda: int) -> float:
    mask = ~np.isclose(r_sq_array, q_sq)
    sum_part = np.sum(1.0 / (r_sq_array[mask] - q_sq))

    q = np.sqrt(q_sq) if q_sq > 1e-12 else 0.0
    if q > 0:
        integral_part = 4.0 * np.pi * lamda + 2.0 * np.pi * q * np.log(
            abs((lamda - q) / (lamda + q))
        )
    else:
        integral_part = 4.0 * np.pi * lamda

    return Y_00 * (sum_part - integral_part)


def _gen_zeta_00_rest_from_q_sq(n_sq_array: np.ndarray, q_sq: float, lamda: int) -> float:
    mask = ~np.isclose(n_sq_array, q_sq, atol=1e-12)
    return np.sum(1.0 / (n_sq_array[mask] - q_sq)) * Y_00 - 4.0 * np.pi * lamda * Y_00


# ---------------------------------------------------------------------------
# Scattering-phase fits
# ---------------------------------------------------------------------------
def fit_mom_indices(config: Config, scattering_dict: dict) -> dict[int, list[int]]:
    """Return momentum-level indices used in the scattering fit."""
    if not config.is_moving_frame:
        return config.fit_mom_by_ns

    fit_mom_by_ns: dict[int, list[int]] = {}
    for ensemble_key in config.scattering_list:
        ns = ensemble_key[0]
        rest_point_count = scattering_dict["rest_point_count"][ns]
        fit_mom_by_ns[ns] = list(config.fit_mom_by_ns.get(ns, [])) + [
            rest_point_count + mom for mom in config.fit_mom_by_ns_MF.get(ns, [])
        ]
    return fit_mom_by_ns


def run_scattering_fit(
    config: Config, scattering_dict: dict, fit_mom_by_ns: dict[int, list[int]]
) -> None:
    """Run the configured fit and store curves in ``scattering_dict``."""
    if config.scattering_fit_mode not in SCATTERING_FIT_MODES:
        raise ValueError(
            f"Unknown scattering_fit_mode={config.scattering_fit_mode!r}; "
            f"expected one of {SCATTERING_FIT_MODES}"
        )

    ref_ns = next(iter(scattering_dict["s_array"]))
    if config.scattering_fit_mode == "Ks_linear":
        _fit_Ks_linear(scattering_dict, fit_mom_by_ns, ref_ns)
    else:
        _fit_kcot_quadratic(scattering_dict, fit_mom_by_ns, ref_ns)


def _concat_fit_points(
    scattering_dict: dict, fit_mom_by_ns: dict[int, list[int]], key: str
) -> np.ndarray:
    return np.concatenate(
        [scattering_dict[key][ns][moms] for ns, moms in fit_mom_by_ns.items()]
    )


def _fit_gvar_model(
    scattering_dict: dict,
    fit_mom_by_ns: dict[int, list[int]],
    ref_ns: int,
    x_key: str,
    y_key: str,
    model,
    prior,
    curve_x_key: str,
    curve_y_key: str,
    label: str,
) -> None:
    x_all = _concat_fit_points(scattering_dict, fit_mom_by_ns, x_key)
    y_all = _concat_fit_points(scattering_dict, fit_mom_by_ns, y_key)
    fit_result = lsf.nonlinear_fit(
        data=(x_all[:, 0], gv.gvar(y_all[:, 0], y_all[:, 1])),
        fcn=model,
        prior=prior,
    )
    scattering_dict["fit_result"] = fit_result
    scattering_dict[curve_y_key] = model(scattering_dict[curve_x_key][ref_ns], fit_result.p)
    print(label + ":", x_all[:, 0], gv.gvar(y_all[:, 0], y_all[:, 1]), fit_result)


def _fit_Ks_linear(
    scattering_dict: dict,
    fit_mom_by_ns: dict[int, list[int]],
    ref_ns: int,
) -> None:
    _fit_gvar_model(
        scattering_dict,
        fit_mom_by_ns,
        ref_ns,
        x_key="s",
        y_key="Ks",
        model=MathModels.linear,
        prior=MathModels.prior_linear(),
        curve_x_key="s_array",
        curve_y_key="fit_Ks_curve",
        label="Ks fit",
    )
    scattering_dict["fit_kcot_curve"] = (
        scattering_dict["sqrt_s_array"][ref_ns] / scattering_dict["fit_Ks_curve"]
    )


def _fit_kcot_quadratic(
    scattering_dict: dict,
    fit_mom_by_ns: dict[int, list[int]],
    ref_ns: int,
) -> None:
    _fit_gvar_model(
        scattering_dict,
        fit_mom_by_ns,
        ref_ns,
        x_key="k_sq",
        y_key="kcot",
        model=MathModels.quadratic,
        prior=MathModels.prior_quadratic(),
        curve_x_key="k_sq_array",
        curve_y_key="fit_kcot_curve",
        label="kcot quadratic fit",
    )


# ---------------------------------------------------------------------------
# Scattering analysis pipeline
# ---------------------------------------------------------------------------
def run_scattering_analysis(
    config: Config, resampled_dict: ResampleDataDict
) -> dict | None:
    if not (config.run_scattering and config.is_tetraquark_analysis):
        return None

    q_sq_grid = q_sq_linspace(config.rest_zeta_n_q)
    zeta_00_rest = gen_zeta_00_rest_from_q_sq_array(
        q_sq_grid, lamda=config.rest_zeta_lamda, regen_flag=config.regen_rest_zeta
    )
    scattering_dict = defaultdict(dict)

    for ensemble_key in config.scattering_list:
        _analyze_ensemble(
            config,
            resampled_dict,
            ensemble_key,
            q_sq_grid,
            zeta_00_rest,
            scattering_dict,
        )

    fit_mom_by_ns = fit_mom_indices(config, scattering_dict)
    scattering_dict["scattering_fit_mode"] = config.scattering_fit_mode
    run_scattering_fit(config, scattering_dict, fit_mom_by_ns)
    return scattering_dict


def _analyze_ensemble(
    config: Config,
    resampled_dict: ResampleDataDict,
    ensemble_key,
    q_sq_grid: np.ndarray,
    zeta_00_rest: np.ndarray,
    scattering_dict: dict,
) -> None:
    ch_meson_a = config.ch_meson_a
    ch_meson_b = config.ch_meson_b
    ch_tetra = config.ch_tetra

    ksi_a = resampled_dict["ksi"][ensemble_key][ch_meson_a]
    ksi_b = resampled_dict["ksi"][ensemble_key][ch_meson_b]
    e0_meson_a = resampled_dict["meson"][ensemble_key][ch_meson_a, 0]
    e0_meson_b = resampled_dict["meson"][ensemble_key][ch_meson_b, 0]
    en_tetra = resampled_dict["tetraquark"][ensemble_key][ch_tetra]

    ns = ensemble_key[0]
    lattice_size = ns * (ksi_a + ksi_b) / 2 / config.at_invs
    _store_rest_reference_arrays(
        scattering_dict, ns, q_sq_grid, zeta_00_rest, e0_meson_a, e0_meson_b, lattice_size
    )

    if config.is_moving_frame:
        mom_results = _analyze_rest_momenta(
            en_tetra, range(en_tetra.shape[0]), e0_meson_a, e0_meson_b, lattice_size, q_sq_grid, zeta_00_rest
        )
        scattering_dict["rest_point_count"][ns] = en_tetra.shape[0]
        en_tetra_mf = resampled_dict["tetraquark_MF"][ensemble_key][config.ch_tetra_MF]
        mf_results, k_sq_mf_ref, kcot_mf_ref = _analyze_moving_frame_momenta(
            config, ensemble_key, en_tetra_mf, e0_meson_a, e0_meson_b, lattice_size
        )
        scattering_dict["k_sq_array_MF"][ns] = k_sq_mf_ref
        scattering_dict["kcot_array_MF"][ns] = kcot_mf_ref
        mom_results.extend(mf_results)
    else:
        mom_list = config.chan_momt_list[ch_tetra]
        mom_results = _analyze_rest_momenta(
            en_tetra, mom_list, e0_meson_a, e0_meson_b, lattice_size, q_sq_grid, zeta_00_rest
        )

    for name in SCATTER_STAT_NAMES:
        scattering_dict[name][ns] = np.array(
            [get_resampler(config, result[name]).gvar() for result in mom_results]
        )


def _store_rest_reference_arrays(
    scattering_dict: dict,
    ns: int,
    q_sq_grid: np.ndarray,
    zeta_00_rest: np.ndarray,
    e0_meson_a: np.ndarray,
    e0_meson_b: np.ndarray,
    lattice_size: np.ndarray,
) -> None:
    k_sq_grid = q_sq_grid * (_TWO_PI / lattice_size.mean()) ** 2
    sqrt_s_grid = np.sqrt(e0_meson_a.mean() ** 2 + k_sq_grid) + np.sqrt(
        e0_meson_b.mean() ** 2 + k_sq_grid
    )
    scattering_dict["sqrt_s_array"][ns] = sqrt_s_grid
    scattering_dict["s_array"][ns] = sqrt_s_grid**2
    scattering_dict["k_sq_array"][ns] = k_sq_grid
    scattering_dict["kcot_rest_array"][ns] = (
        zeta_00_rest * _KCOT_PREFACTOR / lattice_size.mean()
    )


def _analyze_rest_momenta(
    en_tetra: np.ndarray,
    mom_indices,
    e0_meson_a: np.ndarray,
    e0_meson_b: np.ndarray,
    lattice_size: np.ndarray,
    q_sq_grid: np.ndarray,
    zeta_00_rest: np.ndarray,
) -> list[dict]:
    return [
        _analyze_rest_momentum(
            en_tetra[mom], e0_meson_a, e0_meson_b, lattice_size, q_sq_grid, zeta_00_rest
        )
        for mom in mom_indices
    ]


def _analyze_rest_momentum(
    en_tetra: np.ndarray,
    e0_meson_a: np.ndarray,
    e0_meson_b: np.ndarray,
    lattice_size: np.ndarray,
    q_sq_grid: np.ndarray,
    zeta_00_rest: np.ndarray,
) -> dict:
    k_sq = kallen(en_tetra**2, e0_meson_a**2, e0_meson_b**2) / (4 * en_tetra**2)
    sqrt_s = np.sqrt(e0_meson_a**2 + k_sq) + np.sqrt(e0_meson_b**2 + k_sq)
    q_sq = k_sq * (lattice_size / _TWO_PI) ** 2
    zeta = np.interp(q_sq, q_sq_grid, zeta_00_rest)
    kcot = zeta * _KCOT_PREFACTOR / lattice_size
    return {"Ks": sqrt_s / kcot, "s": sqrt_s**2, "sqrt_s": sqrt_s, "k_sq": k_sq, "kcot": kcot}


def _analyze_moving_frame_momenta(
    config: Config,
    ensemble_key,
    en_tetra_mf: np.ndarray,
    e0_meson_a: np.ndarray,
    e0_meson_b: np.ndarray,
    lattice_size: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    d_vec = np.array(config.moving_frame_d_vec, dtype=float)
    lamda = config.moving_frame_zeta_lamda
    k_sq_ref, kcot_ref = _load_or_build_moving_frame_reference(
        config, ensemble_key, en_tetra_mf[0], e0_meson_a, e0_meson_b, lattice_size, d_vec, lamda
    )

    scatter_path = mf_scatter_path(config, ensemble_key)
    if not config.regen_moving_frame_scattering:
        cached = load_mf_scatter_all(scatter_path)
        if cached is not None:
            print(f"Loading {scatter_path}")
            return cached, k_sq_ref, kcot_ref

    results = [
        _analyze_moving_frame_momentum(
            en_tetra, e0_meson_a, e0_meson_b, lattice_size, d_vec, lamda
        )
        for en_tetra in en_tetra_mf
    ]
    save_mf_scatter_all(scatter_path, results)
    print(f"Saved {scatter_path}")
    return results, k_sq_ref, kcot_ref


def _load_or_build_moving_frame_reference(
    config: Config,
    ensemble_key,
    en_tetra_gs: np.ndarray,
    e0_meson_a: np.ndarray,
    e0_meson_b: np.ndarray,
    lattice_size: np.ndarray,
    d_vec: np.ndarray,
    lamda: int,
) -> tuple[np.ndarray, np.ndarray]:
    ref_path = mf_scatter_ref_path(config, ensemble_key)
    if not config.regen_moving_frame_scattering:
        cached_ref = load_mf_scatter_ref(ref_path)
        if cached_ref is not None:
            print(f"Loading {ref_path}")
            return cached_ref

    q_sq_grid = q_sq_linspace(config.moving_frame_zeta_n_q)
    lattice_size_mean, gamma_mean, alpha_mean = _moving_frame_mean_params_ground_state(
        en_tetra_gs, e0_meson_a, e0_meson_b, lattice_size, d_vec
    )
    zeta_00_moving = gen_zeta_00_moving_from_q_sq_array(
        q_sq_grid, config, ensemble_key, d_vec, lamda, gamma_mean, alpha_mean
    )
    k_sq_ref = q_sq_grid * (_TWO_PI / lattice_size_mean) ** 2
    kcot_ref = zeta_00_moving * _KCOT_PREFACTOR / lattice_size_mean / gamma_mean
    save_mf_scatter_ref(ref_path, k_sq_ref, kcot_ref)
    print(f"Saved {ref_path}")
    return k_sq_ref, kcot_ref


def _analyze_moving_frame_momentum(
    en_tetra: np.ndarray,
    e0_meson_a: np.ndarray,
    e0_meson_b: np.ndarray,
    lattice_size: np.ndarray,
    d_vec: np.ndarray,
    lamda: int,
) -> dict:
    k_sq = np.zeros_like(en_tetra)
    kcot = np.zeros_like(en_tetra)
    sqrt_s = np.zeros_like(en_tetra)

    for sample_idx in range(en_tetra.shape[-1]):
        total_momentum_sq = np.sum((_TWO_PI * d_vec / lattice_size[sample_idx]) ** 2)
        e_cm_sq = max(en_tetra[sample_idx] ** 2 - total_momentum_sq, 0.0)
        e_cm = np.sqrt(e_cm_sq)
        gamma = en_tetra[sample_idx] / e_cm
        alpha = 1.0 + (
            (e0_meson_a[sample_idx] ** 2 - e0_meson_b[sample_idx] ** 2) / e_cm_sq
        )

        n_vec_array = build_moving_frame_n_vec_array(lamda, d_vec, alpha, gamma)
        r_sq_array = build_r_sq_from_n_vec(n_vec_array, d_vec, alpha, gamma)
        r_sq_array = r_sq_array[r_sq_array <= lamda**2]

        k_sq[sample_idx] = (
            kallen(e_cm_sq, e0_meson_a[sample_idx] ** 2, e0_meson_b[sample_idx] ** 2)
            / (4.0 * e_cm_sq)
        )
        q_sq = k_sq[sample_idx] * (lattice_size[sample_idx] / _TWO_PI) ** 2
        zeta_00 = gen_zeta_00_moving_from_q_sq(r_sq_array, q_sq, lamda)
        kcot[sample_idx] = _KCOT_PREFACTOR * zeta_00 / lattice_size[sample_idx] / gamma
        sqrt_s[sample_idx] = np.sqrt(e0_meson_a[sample_idx] ** 2 + k_sq[sample_idx]) + np.sqrt(
            e0_meson_b[sample_idx] ** 2 + k_sq[sample_idx]
        )

    return {"Ks": sqrt_s / kcot, "s": sqrt_s**2, "sqrt_s": sqrt_s, "k_sq": k_sq, "kcot": kcot}


def _moving_frame_mean_params_ground_state(
    en_tetra_gs: np.ndarray,
    e0_meson_a: np.ndarray,
    e0_meson_b: np.ndarray,
    lattice_size: np.ndarray,
    d_vec: np.ndarray,
) -> tuple[float, float, float]:
    lattice_size_mean = float(np.mean(lattice_size))
    en_mean = float(np.mean(en_tetra_gs))
    m_a = float(np.mean(e0_meson_a))
    m_b = float(np.mean(e0_meson_b))
    total_momentum_sq = float(np.sum((_TWO_PI * d_vec / lattice_size_mean) ** 2))
    e_cm_sq = en_mean**2 - total_momentum_sq
    if e_cm_sq <= 0:
        raise ValueError(f"Non-positive E_cm^2 for moving-frame ground state: {e_cm_sq}")

    gamma = en_mean / np.sqrt(e_cm_sq)
    alpha = 1.0 + (m_a**2 - m_b**2) / e_cm_sq
    return lattice_size_mean, gamma, alpha
