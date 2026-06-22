from collections import defaultdict

import numpy as np

from analysis.fit_scattering import fit_mom_indices, run_scattering_fit
from analysis.zeta import (
    build_moving_frame_n_vec_array,
    build_r_sq_from_n_vec,
    gen_zeta_00_moving_from_q_sq,
    gen_zeta_00_moving_from_q_sq_array,
    gen_zeta_00_rest_from_q_sq_array,
    kallen,
    q_sq_linspace,
)
from data.scattering_io import (
    SCATTER_STAT_NAMES,
    load_mf_scatter_all,
    load_mf_scatter_ref,
    mf_scatter_path,
    mf_scatter_ref_path,
    save_mf_scatter_all,
    save_mf_scatter_ref,
)
from input.config import Config
from input.types import ResampleDataDict
from statistics.resample import get_resampler

_TWO_PI = 2.0 * np.pi
_KCOT_PREFACTOR = 2.0 / np.sqrt(np.pi)


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
