"""Zc3900: rho eta_c scattering (rest frame); pi J/psi reserved for moving frame.

Scattering-only workflow on L=16. Moving-frame switch left off; moving-frame channel preset
if run_MF_analysis is enabled later.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from input.config import EnsembleEntry, EnsembleKey, InputControlBase, PlotLimits


@dataclass
class InputControl(InputControlBase):

    tetraquark_name: str = "Zc3900"

    # --- Effective mass (GEVP, fits, ratio, t_min, resample) ---
    lattice_Ns: int = 16  # primary ensemble; raw analysis off (use resampled En)

    run_meson_analysis: bool = False
    run_dispersion_analysis: bool = False  # meson branch only
    run_weight_analysis: bool = False  # meson branch only

    run_tetraquark_analysis: bool = False
    run_GEVP_analysis: bool = True  # tetraquark branch only
    run_ratio_analysis: bool = False  # tetraquark branch only
    is_ratio_shift: bool = False  # True: R_n shift (ratio_at); False: R = C4/(C_a C_b)

    run_tmin_analysis: bool = False  # meson: 2-state E; tetraquark: 3-state + (ratio)
    run_resample_analysis: bool = False

    # --- Scattering (K(s), kcot from resampled En) ---
    scattering_channel: str = r"\rho\,\eta_c"
    scattering_Ns_mom: Dict[int, List[int]] = field(
        default_factory=lambda: {16: [0, 1]}  # L -> tetra momentum indices
    )

    run_scattering_analysis: bool = True
    run_MF_analysis: bool = False  # moving frame (off; channel preset below)
    scattering_channel_MF: str = r"\pi\,J/\psi"  # moving frame tetraquark channel
    scattering_Ns_mom_MF: Dict[int, List[int]] = field(
        default_factory=lambda: {16: [0, 1]}  # moving frame: momentum indices for fit
    )

    # --- Plot options ---
    is_plot_title: bool = True  # figure title with ensemble tag (L*M*_EV*)
    is_plot_show: bool = True  # plt.show() after each save
    k_sq_plot_range: PlotLimits = (-0.1, 0.5, -2.0, 2.0)
    s_plot_range: PlotLimits = (14, 18, -0.5, 2.5)

    @staticmethod
    def get_lattice_params(lattice_Ns: int) -> EnsembleKey:
        """Map spatial size to (Ns, Nt, pion_mass, num_eigenvectors)."""
        match lattice_Ns:
            case 12:
                return 12, 96, 420, 70
            case 16:
                return 16, 128, 420, 70
            case _:
                raise ValueError(f"Unknown lattice_Ns={lattice_Ns}")


# See EnsembleEntry in input/config.py for block layout.
ENSEMBLE_DB: Dict[EnsembleKey, EnsembleEntry] = {
    (16, 128, 420, 70): {
        "at_invs": 7.219,
        "GEVP": (20, 30, 25),
        "meson": {
            "channel_name_list": [
                r"\pi",
                r"J/\psi",
                r"\rho",
                r"\eta_c",
                r"D",
                r"D^*",
            ],
            "channel_momentum_list": [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ],
            "tmin_selected": [
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
            ],
            "prior_weff_0_error": 0.1,
            "prior_weff_0": [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            "prior_weff_1_error": 0.01,
            "prior_weff_1": [
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prior_meff_0_error": 0.01,
            "prior_meff_0": [
                [0.06, 0.095, 0.12, 0.145, 0.16],
                [0.425, 0.435, 0.44, 0.45, 0.455],
                [0.12, 0.14, 0.16, 0.175, 0.19],
                [0.415, 0.42, 0.43, 0.435, 0.44],
                [0.26, 0.275, 0.285, 0.295, 0.305],
                [0.28, 0.29, 0.3, 0.31, 0.32],
            ],
            "prior_meff_1_error": 0.1,
            "prior_meff_1": [
                [0.3, 0.3, 0.3, 0.3, 0.3],
                [0.6, 0.6, 0.6, 0.6, 0.6],
                [0.3, 0.3, 0.3, 0.3, 0.3],
                [0.6, 0.6, 0.6, 0.6, 0.6],
                [0.4, 0.4, 0.4, 0.4, 0.4],
                [0.4, 0.4, 0.4, 0.4, 0.4],
            ],
        },
        "tetraquark": {
            "channel_name_list": [
                r"\pi\,J/\psi",
                r"\rho\,\eta_c",
                r"D\,D^*",
                r"D^*\,D^*",
            ],
            "channel_momentum_list": [
                [0, 1],
                [0, 1],
                [0],
                [],
            ],
            "tmin_selected": [
                [25, 25, 25],
                [25, 20, 25],
                [25, 20, 20],
                [25, 25, 25],
            ],
            "prior_weff_0_error": 0.1,
            "prior_weff_0": [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            "prior_weff_1_error": 0.1,
            "prior_weff_1": [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, 0.5],
            ],
            "prior_weff_2_error": 0.01,
            "prior_weff_2": [
                [0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01],
                [0.01, 0.2, 0.2],
                [0.01, 0.2, 0.1],
            ],
            "prior_meff_0_error": 0.01,
            "prior_meff_0": [
                [0.49, 0.53, 0.565],
                [0.53, 0.57, 0.59],
                [0.54, 0.56, 0.59],
                [0.56, 0.50, 0.60],
            ],
            "prior_meff_1_error": 0.1,
            "prior_meff_1": [
                [0.37, 0.34, 0.317],
                [0.30, 0.30, 0.27],
                [0.06, 0.4, 0.2],
                [0.00, 0.00, 0.00],
            ],
            "prior_meff_2_error": 0.1,
            "prior_meff_2": [
                [0.6, 0.6, 0.7],
                [0.7, 0.7, 0.7],
                [0.6, 0.6, 0.6],
                [0.6, 0.6, 0.6],
            ],
        },
    },
}
