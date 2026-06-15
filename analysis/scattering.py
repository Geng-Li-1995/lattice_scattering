import os
from collections import defaultdict

import gvar as gv
import lsqfit as lsf
import numpy as np
from joblib import Parallel, delayed

from analysis.models import MathModels
from input.config import Config
from input.types import ResampleDataDict
from statistics.resample import get_resampler

STAT_NAMES = ("Ks", "s", "k_sq", "kcot")


def run_scattering_analysis(
    config: Config, resampled_dict: ResampleDataDict
) -> dict | None:
    if not (config.run_scattering and config.is_tetraquark_analysis):
        return None

    ch_meson_a = config.ch_meson_a
    ch_meson_b = config.ch_meson_b
    ch_tetra = config.ch_tetra
    fit_mom_by_ns = config.fit_mom_by_ns

    mom_list = config.chan_momt_list[ch_tetra]
    at_invs = config.at_invs
    q_sq_grid = np.linspace(-5, 10, 10**5)
    zeta_00_rest = gen_zeta_00_rest_from_q_sq_array(q_sq_grid, lamda=50, regen_flag=False)
    scattering_dict = defaultdict(dict)

    for ensemble_key in config.scattering_list:
        ksi_a = resampled_dict["ksi"][ensemble_key][ch_meson_a]
        ksi_b = resampled_dict["ksi"][ensemble_key][ch_meson_b]
        e0_meson_a = resampled_dict["meson"][ensemble_key][ch_meson_a, 0]
        e0_meson_b = resampled_dict["meson"][ensemble_key][ch_meson_b, 0]
        en_tetra = resampled_dict["tetraquark"][ensemble_key][ch_tetra]

        ns = ensemble_key[0]
        lattice_size = ns * (ksi_a + ksi_b) / 2 / at_invs

        k_sq_grid = q_sq_grid * (2 * np.pi / lattice_size.mean()) ** 2
        sqrt_s_grid = np.sqrt(e0_meson_a.mean() ** 2 + k_sq_grid) + np.sqrt(
            e0_meson_b.mean() ** 2 + k_sq_grid
        )
        kcot_rest_grid = zeta_00_rest * 2 / np.sqrt(np.pi) / lattice_size.mean()

        scattering_dict["sqrt_s_array"][ns] = sqrt_s_grid
        scattering_dict["s_array"][ns] = sqrt_s_grid**2
        scattering_dict["k_sq_array"][ns] = k_sq_grid
        scattering_dict["kcot_rest_array"][ns] = kcot_rest_grid

        mom_results = {
            mom: _analyze_momentum(
                en_tetra[mom], e0_meson_a, e0_meson_b, lattice_size, q_sq_grid, zeta_00_rest
            )
            for mom in mom_list
        }
        for name in STAT_NAMES:
            scattering_dict[name][ns] = np.array(
                [get_resampler(config, mom_results[mom][name]).gvar() for mom in mom_list]
            )

    s_all = np.concatenate([scattering_dict["s"][ns][moms] for ns, moms in fit_mom_by_ns.items()])
    ks_all = np.concatenate([scattering_dict["Ks"][ns][moms] for ns, moms in fit_mom_by_ns.items()])

    ks_linear_fit = lsf.nonlinear_fit(
        data=(s_all[:, 0], gv.gvar(ks_all[:, 0], ks_all[:, 1])),
        fcn=MathModels.linear,
        prior=MathModels.prior_linear(),
    )

    ref_ns = next(iter(scattering_dict["s_array"]))
    scattering_dict["fit_Ks_curve"] = MathModels.linear(
        scattering_dict["s_array"][ref_ns], ks_linear_fit.p
    )
    scattering_dict["fit_kcot_curve"] = (
        scattering_dict["sqrt_s_array"][ref_ns] / scattering_dict["fit_Ks_curve"]
    )

    print("Ks fit:", s_all[:, 0], gv.gvar(ks_all[:, 0], ks_all[:, 1]), ks_linear_fit)
    return scattering_dict


def _analyze_momentum(
    en_tetra, e0_meson_a, e0_meson_b, lattice_size, q_sq_grid, zeta_00_rest
) -> dict:
    k_sq = kallen(en_tetra**2, e0_meson_a**2, e0_meson_b**2) / (4 * en_tetra**2)
    sqrt_s = np.sqrt(e0_meson_a**2 + k_sq) + np.sqrt(e0_meson_b**2 + k_sq)
    q_sq = k_sq * (lattice_size / (2 * np.pi)) ** 2
    zeta = np.interp(q_sq, q_sq_grid, zeta_00_rest)
    kcot = zeta * 2 / np.sqrt(np.pi) / lattice_size
    return {"Ks": sqrt_s / kcot, "s": sqrt_s**2, "sqrt_s": sqrt_s, "k_sq": k_sq, "kcot": kcot}


def kallen(a, b, c) -> float:
    """Källén function."""
    return a**2 + b**2 + c**2 - 2 * (a * b + a * c + b * c)


def build_n_sq_array(lamda: int) -> np.ndarray:
    n_range = np.arange(-lamda - 1, lamda + 2)
    nx, ny, nz = np.meshgrid(n_range, n_range, n_range, indexing="ij")
    n_sq = nx**2 + ny**2 + nz**2
    return n_sq[n_sq <= lamda**2].astype(float)


def gen_zeta_00_rest_from_q_sq_array(
    q_sq_grid: np.ndarray, lamda: int, regen_flag: bool = False
) -> np.ndarray:
    save_path = "data/zeta/zeta_00_rest_array.npy"
    if (not regen_flag) and os.path.exists(save_path):
        print(f"Loading {save_path}")
        return np.load(save_path)

    print("Generating zeta_00_rest_array.npy ...")
    n_sq_array = build_n_sq_array(lamda)
    zeta_array = np.asarray(
        Parallel(n_jobs=-1)(
            delayed(gen_zeta_00_rest_from_q_sq)(n_sq_array, q_sq, lamda)
            for q_sq in q_sq_grid
        ),
        dtype=float,
    )
    np.save(save_path, zeta_array)
    print(f"Saved to {save_path}")
    return zeta_array


def gen_zeta_00_rest_from_q_sq(n_sq_array: np.ndarray, q_sq: float, lamda: int) -> float:
    y_00 = 1.0 / np.sqrt(4.0 * np.pi)
    mask = ~np.isclose(n_sq_array, q_sq, atol=1e-12)
    return np.sum(1.0 / (n_sq_array[mask] - q_sq)) * y_00 - 4.0 * np.pi * lamda * y_00
