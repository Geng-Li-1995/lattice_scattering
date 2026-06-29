"""Tcccc6600: doubly-charmed tetraquark (J/psi J/psi, eta_c eta_c channels).

Default workflow: meson + tetraquark GEVP, t_min scan, ratio, rest-frame scattering.
No moving-frame data in this project — enable via run_MF_analysis if added later.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from input.config import EnsembleEntry, EnsembleKey, InputControlBase, PlotLimits


@dataclass
class InputControl(InputControlBase):

    tetraquark_name: str = "Tcccc6600"

    # --- Effective mass (GEVP, fits, ratio, t_min, resample) ---
    lattice_Ns: int = 12  # primary ensemble for GEVP / mass / ratio

    run_meson_analysis: bool = True
    run_dispersion_analysis: bool = True  # meson branch only
    run_weight_analysis: bool = True  # meson branch only

    run_tetraquark_analysis: bool = True
    run_GEVP_analysis: bool = True  # tetraquark branch only
    run_ratio_analysis: bool = True  # tetraquark branch only
    is_ratio_shift: bool = True  # True: R_n shift (ratio_at); False: R = C4/(C_a C_b)

    run_tmin_analysis: bool = True  # meson: 2-state E; tetraquark: 3-state + (ratio)
    run_resample_analysis: bool = False

    # --- Scattering (K(s), kcot from resampled En) ---
    scattering_channel: str = r"J/\psi\,J/\psi"
    scattering_Ns_mom: Dict[int, List[int]] = field(
        default_factory=lambda: {
            12: [0, 1, 2],
            16: [0, 1],
        }  # L -> tetra momentum indices
    )

    run_scattering_analysis: bool = True
    run_MF_analysis: bool = False  # moving frame (off for this system)

    # --- Plot options ---
    is_plot_title: bool = True  # figure title with ensemble tag (L*M*_EV*)
    is_plot_show: bool = True  # plt.show() after each save
    k_sq_plot_range: PlotLimits = (-0.25, 1.75, -15.0, 15.0)
    s_plot_range: PlotLimits = (37, 45, -9.0, 6.0)

    @staticmethod
    def get_lattice_params(lattice_Ns: int) -> EnsembleKey:
        """Map spatial size to (Ns, Nt, pion_mass, num_eigenvectors)."""
        match lattice_Ns:
            case 12:
                return 12, 96, 420, 170
            case 16:
                return 16, 128, 420, 120
            case _:
                raise ValueError(f"Unknown lattice_Ns={lattice_Ns}")


# See EnsembleEntry in input/config.py for block layout.
ENSEMBLE_DB: Dict[EnsembleKey, EnsembleEntry] = {
    (12, 96, 420, 170): {
        "at_invs": 7.219,
        "GEVP": (15, 25, 20),
        "meson": {
            "channel_name_list": [
                r"\eta_c",
                r"J/\psi",
            ],
            "channel_momentum_list": [
                [0, 1, 2, 3, 4, 5, 6, 8, 9],  # no n^2=7 on lattice
                [0, 1, 2, 3, 4, 5, 6, 8, 9],
            ],
            "tmin_selected": [
                [32, 27, 26, 20, 20, 20, 20, 50, 20, 20],
                [29, 23, 20, 20, 20, 20, 20, 50, 20, 20],
            ],
            "prior_weff_0_error": 0.1,
            "prior_weff_0": [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            "prior_weff_1_error": 0.01,
            "prior_weff_1": [
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prior_meff_0_error": 0.01,
            "prior_meff_0": [
                [0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.49, 1.0, 0.51, 0.52],
                [0.43, 0.44, 0.45, 0.46, 0.48, 0.49, 0.50, 1.0, 0.52, 0.53],
            ],
            "prior_meff_1_error": 0.1,
            "prior_meff_1": [
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            ],
        },
        "tetraquark": {
            "channel_name_list": [
                r"\eta_c\,\eta_c",
                r"J/\psi\,J/\psi",
            ],
            "channel_momentum_list": [
                [1, 2, 4],
                [0, 1, 2, 3, 4],
            ],
            "tmin_selected": [
                [50, 29, 28, 50, 28],
                [27, 28, 23, 26, 22],
            ],
            "prior_weff_0_error": 0.1,
            "prior_weff_0": [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            "prior_weff_1_error": 0.1,
            "prior_weff_1": [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            "prior_weff_2_error": 0.01,
            "prior_weff_2": [
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prior_meff_0_error": 0.01,
            "prior_meff_0": [
                [1.00, 0.86, 0.88, 1.00, 0.93],
                [0.86, 0.88, 0.90, 0.92, 0.94],
            ],
            "prior_meff_1_error": 0.1,
            "prior_meff_1": [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "prior_meff_2_error": 0.1,
            "prior_meff_2": [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        },
    },
    (16, 128, 420, 120): {
        "at_invs": 7.219,
        "GEVP": (30, 40, 35),
        "meson": {
            "channel_name_list": [
                r"\eta_c",
                r"J/\psi",
            ],
            "channel_momentum_list": [
                [0, 1, 2, 3, 4, 5, 6, 8, 9],  # no n^2=7 on lattice
                [0, 1, 2, 3, 4, 5, 6, 8, 9],
            ],
            "tmin_selected": [
                [26, 20, 21, 21, 21, 21, 21, 50, 21, 21],
                [26, 19, 18, 20, 20, 23, 20, 50, 20, 20],
            ],
            "prior_weff_0_error": 0.1,
            "prior_weff_0": [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            "prior_weff_1_error": 0.1,
            "prior_weff_1": [
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prior_meff_0_error": 0.01,
            "prior_meff_0": [
                [0.41, 0.42, 0.43, 0.435, 0.44, 0.45, 0.46, 1.0, 0.47, 0.48],
                [0.42, 0.43, 0.44, 0.45, 0.455, 0.46, 0.47, 1.0, 0.48, 0.49],
            ],
            "prior_meff_1_error": 0.1,
            "prior_meff_1": [
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            ],
        },
        "tetraquark": {
            "channel_name_list": [
                r"\eta_c\,\eta_c",
                r"J/\psi\,J/\psi",
            ],
            "channel_momentum_list": [
                [1, 2, 4],
                [0, 1, 2, 3, 4],
            ],
            "tmin_selected": [
                [50, 29, 29, 50, 22, 50],
                [24, 28, 29, 29, 33, 50],
            ],
            "prior_weff_0_error": 0.1,
            "prior_weff_0": [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            "prior_weff_1_error": 0.1,
            "prior_weff_1": [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            "prior_weff_2_error": 0.01,
            "prior_weff_2": [
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prior_meff_0_error": 0.01,
            "prior_meff_0": [
                [1.00, 0.84, 0.86, 1.00, 0.89],
                [0.86, 0.87, 0.88, 0.90, 0.91],
            ],
            "prior_meff_1_error": 0.1,
            "prior_meff_1": [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "prior_meff_2_error": 0.1,
            "prior_meff_2": [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        },
    },
}
