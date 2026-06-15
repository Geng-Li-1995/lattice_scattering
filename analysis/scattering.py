# scattering.py

import os
import numpy as np
import gvar as gv
import lsqfit as lsf

from collections import defaultdict
from joblib import Parallel, delayed
from statistics.resample import get_resampler
from analysis.models import MathModels


# ==================================================
# Rest-frame single-channel analysis
# ==================================================
def run_scattering_analysis(config, resample_data_dict):
    if not (
        getattr(config, "run_scattering", False)
        and getattr(config, "is_tetraquark_analysis", False)
    ):
        return

    # choose channel and momentum
    ch_meson_a = ch_meson_b = ch_tetraquark = 1
    momt_tetraquark = {12: [0, 1, 2], 16: [0, 1]}

    momt_list = config.chan_momt_list[ch_tetraquark]
    at_invs = config.at_invs
    q_sq_array = np.linspace(-5, 10, 10**5)

    lamda_rest = 50
    zeta_00_rest = gen_zeta_00_rest_from_q_sq_array(
        q_sq_array, lamda_rest, regen_flag=False
    )

    results_dict = defaultdict(dict)

    def analyze_momentum(momt, En_tetraquark, E0_meson_a, E0_meson_b, lattice_size):
        k_sq = lambda_function(En_tetraquark**2, E0_meson_a**2, E0_meson_b**2) / (
            4 * En_tetraquark**2
        )

        sqrt_s = np.sqrt(E0_meson_a**2 + k_sq) + np.sqrt(E0_meson_b**2 + k_sq)

        q_sq = k_sq * (lattice_size / (2 * np.pi)) ** 2
        zeta = np.interp(q_sq, q_sq_array, zeta_00_rest)

        kcot = zeta * 2 / np.sqrt(np.pi) / lattice_size
        Ks = sqrt_s / kcot

        return {"Ks": Ks, "s": sqrt_s**2, "sqrt_s": sqrt_s, "k_sq": k_sq, "kcot": kcot}

    for idx, scattering in enumerate(config.scattering_list):

        ksi_a = resample_data_dict["ksi"][scattering][ch_meson_a]
        ksi_b = resample_data_dict["ksi"][scattering][ch_meson_b]

        E0_meson_a = resample_data_dict["meson"][scattering][ch_meson_a, 0]
        E0_meson_b = resample_data_dict["meson"][scattering][ch_meson_b, 0]
        En_tetraquark = resample_data_dict["tetraquark"][scattering][ch_tetraquark]

        Ns = scattering[0]
        lattice_size = Ns * (ksi_a + ksi_b) / 2 / at_invs

        k_sq_array = q_sq_array * (2 * np.pi / lattice_size.mean()) ** 2
        sqrt_s_array = np.sqrt(E0_meson_a.mean() ** 2 + k_sq_array) + np.sqrt(
            E0_meson_b.mean() ** 2 + k_sq_array
        )
        kcot_rest_array = zeta_00_rest * 2 / np.sqrt(np.pi) / lattice_size.mean()

        results_dict["sqrt_s_array"][Ns] = sqrt_s_array
        results_dict["s_array"][Ns] = sqrt_s_array**2
        results_dict["k_sq_array"][Ns] = k_sq_array
        results_dict["kcot_rest_array"][Ns] = kcot_rest_array

        per_mom = {
            momt: analyze_momentum(
                momt, En_tetraquark[momt], E0_meson_a, E0_meson_b, lattice_size
            )
            for momt in momt_list
        }

        stats = {
            name: np.array(
                [
                    get_resampler(config, per_mom[momt][name]).gvar()
                    for momt in momt_list
                ]
            )
            for name in ["Ks", "s", "k_sq", "kcot"]
        }

        results_dict["s"][Ns] = stats["s"]
        results_dict["Ks"][Ns] = stats["Ks"]
        results_dict["k_sq"][Ns] = stats["k_sq"]
        results_dict["kcot"][Ns] = stats["kcot"]

    # fit
    s_list = []
    Ks_list = []
    for Ns, idx_list in momt_tetraquark.items():
        s_list.append(results_dict["s"][Ns][idx_list])
        Ks_list.append(results_dict["Ks"][Ns][idx_list])
    s_all = np.concatenate(s_list, axis=0)
    Ks_all = np.concatenate(Ks_list, axis=0)

    fit = lsf.nonlinear_fit(
        data=(s_all[:, 0], gv.gvar(Ks_all[:, 0], Ks_all[:, 1])),
        fcn=MathModels.linear,
        prior=MathModels.prior_linear(),
    )
    results_dict["fit_Ks_curve"] = MathModels.linear(
        results_dict["s_array"][next(iter(results_dict["s_array"]))], fit.p
    )

    results_dict["fit_kcot_curve"] = (
        results_dict["sqrt_s_array"][next(iter(results_dict["sqrt_s_array"]))]
        / results_dict["fit_Ks_curve"]
    )

    print("fit parameter is:", s_all[:, 0], gv.gvar(Ks_all[:, 0], Ks_all[:, 1]), fit)

    return results_dict


# ==================================================
# Helper functions
# ==================================================
def lambda_function(a, b, c):
    """Källén function."""
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * a * c - 2 * b * c


def build_n_sq_array(lamda):
    """Generate n^2 lattice grid with cutoff."""
    n_range = np.arange(-lamda - 1, lamda + 2)
    nx, ny, nz = np.meshgrid(n_range, n_range, n_range, indexing="ij")
    n_sq = nx**2 + ny**2 + nz**2
    return n_sq[n_sq <= lamda**2].astype(float)


# ==================================================
# Rest-frame zeta
# ==================================================
def gen_zeta_00_rest_from_q_sq_array(q_sq_array, lamda, regen_flag=False):
    """Compute rest-frame zeta function on q^2 grid."""

    save_path = "data/zeta/zeta_00_rest_array.npy"

    if (not regen_flag) and os.path.exists(save_path):
        print(f"Loading {save_path}")
        return np.load(save_path)

    print("Generating zeta_00_rest_array.npy ...")

    n_sq_array = build_n_sq_array(lamda)

    results = Parallel(n_jobs=-1)(
        delayed(gen_zeta_00_rest_from_q_sq)(n_sq_array, q_sq, lamda)
        for q_sq in q_sq_array
    )

    zeta_00_array = np.asarray(results, dtype=float)

    np.save(save_path, zeta_00_array)
    print(f"Saved to {save_path}")

    return zeta_00_array


def gen_zeta_00_rest_from_q_sq(n_sq_array, q_sq, lamda):
    """Rest-frame zeta function for a single q^2."""
    Y_00 = 1.0 / np.sqrt(4.0 * np.pi)

    mask = ~np.isclose(n_sq_array, q_sq, atol=1e-12)
    sum_part = np.sum(1.0 / (n_sq_array[mask] - q_sq))

    return sum_part * Y_00 - 4.0 * np.pi * lamda * Y_00
