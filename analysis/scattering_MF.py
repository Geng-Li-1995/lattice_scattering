import os
import numpy as np
import gvar as gv
from joblib import Parallel, delayed
from utils.plotter import Plotter
from analysis.resample import get_resampler


# ==================================================
# Main analysis
# ==================================================
def run_scattering_analysis(config, resample_data_dict):
    if not getattr(config, "run_scattering", False):
        return

    lattice_Ns, lattice_Nt, pion_mass, num_eigenvectors = config.ensemble_key
    at_invs = config.at_invs
    input_name = config.input_name

    ksi = 5.006
    lattice_size = lattice_Ns * ksi / at_invs

    q_sq_array = np.linspace(-5, 10, 10**5)
    k_sq_array = q_sq_array * (2 * np.pi / lattice_size) ** 2

    # ==================================================
    # Rest frame
    # ==================================================
    lamda_rest = 50

    zeta_00_rest_array = gen_zeta_00_rest_from_q_sq_array(
        q_sq_array, lamda_rest, regen_flag=False
    )

    kcot_rest_array = zeta_00_rest_array * 2 / np.sqrt(np.pi) / lattice_size

    for Ns in config.scattering_list:
        resample_E0_meson_a = resample_data_dict["meson"][Ns][4, 0]
        resample_E0_meson_b = resample_data_dict["meson"][Ns][5, 0]
        resample_En_tetraquark = resample_data_dict["tetraquark"][Ns]

        resample_k_sq = np.zeros_like(resample_En_tetraquark)
        resample_kcot = np.zeros_like(resample_En_tetraquark)

        # ==================================================
        # Rest frame (per channel)
        # ==================================================
        for chan in [0, 1, 2]:
            resample_k_sq[chan] = lambda_function(
                resample_En_tetraquark[chan] ** 2,
                resample_E0_meson_a**2,
                resample_E0_meson_b**2,
            ) / (4.0 * resample_En_tetraquark[chan] ** 2)

            resample_q_sq = resample_k_sq[chan] * (lattice_size / (2.0 * np.pi)) ** 2

            resample_zeta_00 = np.interp(resample_q_sq, q_sq_array, zeta_00_rest_array)

            resample_kcot[chan] = resample_zeta_00 * 2 / np.sqrt(np.pi) / lattice_size

        # ==================================================
        # Moving frame
        # ==================================================
        print(f"Running moving frame")

        d_vec = np.array([0, 0, 1])
        lamda = 100
        n_range = np.arange(-lamda, lamda + 1)

        # Generate lattice vectors once
        n_vec_array = np.array(
            np.meshgrid(n_range, n_range, n_range, indexing="ij")
        ).T.reshape(-1, 3)

        resample_gamma, resample_E_cm = gen_Lorentz_gamma_from_En(
            d_vec, lattice_size, resample_En_tetraquark
        )

        resample_alpha = (
            1.0 + (resample_E0_meson_a**2 - resample_E0_meson_b**2) / resample_E_cm**2
        )

        for chan in [3, 4, 5]:
            for sam in range(resample_En_tetraquark.shape[-1]):

                # --------------------------------------------------
                # Build r^2 in moving frame
                # --------------------------------------------------
                r_sq_array = build_r_sq_from_n_vec(
                    n_vec_array,
                    d_vec,
                    resample_alpha[chan, sam],
                    resample_gamma[chan, sam],
                )

                # --------------------------------------------------
                # CUT performed per configuration
                # --------------------------------------------------
                r_sq_array = r_sq_array[r_sq_array <= lamda**2]

                # --------------------------------------------------
                # k^2 and q^2
                # --------------------------------------------------
                resample_k_sq[chan, sam] = lambda_function(
                    resample_E_cm[chan, sam] ** 2,
                    resample_E0_meson_a[sam] ** 2,
                    resample_E0_meson_b[sam] ** 2,
                ) / (4.0 * resample_E_cm[chan, sam] ** 2)

                q_sq = resample_k_sq[chan, sam] * (lattice_size / (2.0 * np.pi)) ** 2

                # --------------------------------------------------
                # Moving-frame zeta function
                # --------------------------------------------------
                zeta_00 = gen_zeta_00_moving_from_q_sq(r_sq_array, q_sq, lamda)

                resample_kcot[chan, sam] = (
                    2.0
                    * zeta_00
                    / np.sqrt(np.pi)
                    / lattice_size
                    / resample_gamma[chan, sam]
                )

    # ==================================================
    # Plot results
    # ==================================================
    Plotter(config).plot_kcot(k_sq_array, kcot_rest_array, resample_k_sq, resample_kcot)

    print("Job finished!")


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

    save_path = "data/zeta_00_rest_array.npy"

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


# ==================================================
# Lorentz boost
# ==================================================
def gen_Lorentz_gamma_from_En(d_vec, lattice_size, resample_En_tetraquark):
    """Compute Lorentz gamma and CM energy."""

    d_vec = np.asarray(d_vec, dtype=float)
    resample_En_tetraquark = np.asarray(resample_En_tetraquark, dtype=float)

    P_vec = (2.0 * np.pi / lattice_size) * d_vec
    P_sq = np.sum(P_vec**2)

    resample_E_cm = np.sqrt(np.maximum(resample_En_tetraquark**2 - P_sq, 0.0))
    resample_gamma = resample_En_tetraquark / resample_E_cm

    return resample_gamma, resample_E_cm


# ==================================================
# Moving-frame zeta
# ==================================================
def build_r_sq_from_n_vec(n_vec_array, d_vec, alpha, gamma):
    """Compute boosted r^2 in moving frame."""

    d_hat = d_vec / np.linalg.norm(d_vec)

    x = n_vec_array - 0.5 * alpha * d_vec[None, :]

    x_para = np.dot(x, d_hat)[:, None] * d_hat[None, :]
    x_perp = x - x_para

    r = x_perp + x_para / gamma
    r_sq = np.sum(r * r, axis=-1)

    return r_sq


def gen_zeta_00_moving_from_q_sq(r_sq_array, q_sq, lamda):
    """Moving-frame zeta function for a single q^2."""

    Y_00 = 1.0 / np.sqrt(4.0 * np.pi)

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
