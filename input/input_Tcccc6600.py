# input_Tcccc6600.py

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

from input.types import EnsembleKey, EnsembleEntry, ScatteringList


@dataclass
class InputControl:
    """Central configuration controlling analysis, plotting, and lattice setup.

    Meson and tetraquark switches are independent. If both are False, the main
    analysis steps are skipped while scattering can still use resampled inputs.
    """

    tetraquark_name: str = "Tcccc6600"

    # Lattice setup
    lattice_Ns: int = 12  # supported: 12, 16
    lattice_Nt: int = 0
    num_eigenvectors: int = 0
    pion_mass: int = 0

    # Analysis options
    is_meson_analysis: bool = False
    is_tetraquark_analysis: bool = True
    is_gevp: bool = True
    is_svd: bool = False
    is_ratio: bool = False

    run_tmin: bool = False
    run_resample: bool = False  # run resampling from main.py
    run_scattering: bool = True

    plot_meff: bool = True  # En / Zn plots (fit always runs)
    plot_dispersion: bool = True  # dispersion fit + plot (meson mode only)
    plot_format: str = "png"  # figure output: "png" or "pdf"

    resample_type: str = "jackknife"
    sample_axis: int = -1
    n_boot: int = 1000

    Ns_list: ClassVar[list[int]] = [12, 16]  # lattice sizes for scattering

    # Scattering analysis: channel indices and fit momentum subsets per Ns
    ch_meson_a: int = 1
    ch_meson_b: int = 1
    ch_tetra: int = 1
    fit_mom_by_ns: Dict[int, List[int]] = field(
        default_factory=lambda: {12: [0, 1, 2], 16: [0, 1]}
    )

    scattering_list: ScatteringList = field(default_factory=list)

    def __post_init__(self):
        """Initialize lattice parameters based on lattice_Ns."""
        if self.plot_format not in ("png", "pdf"):
            raise ValueError(
                f'plot_format must be "png" or "pdf", got {self.plot_format!r}'
            )

        self.lattice_Ns, self.lattice_Nt, self.pion_mass, self.num_eigenvectors = (
            self.get_lattice_params(self.lattice_Ns)
        )
        # Generate list of lattice params for scattering_list
        self.scattering_list = [self.get_lattice_params(ns) for ns in self.Ns_list]

    @staticmethod
    def get_lattice_params(lattice_Ns: int) -> EnsembleKey:
        """Return (lattice_Ns, lattice_Nt, pion_mass, num_eigenvectors) for a given lattice_Ns."""
        match lattice_Ns:
            case 12:
                return 12, 96, 420, 170
            case 16:
                return 16, 128, 420, 120
            case _:
                raise ValueError(f"Unknown lattice_Ns={lattice_Ns}")

    def ensemble_key(self) -> EnsembleKey:
        """Map configuration → ENSEMBLE_DB key."""
        return (self.lattice_Ns, self.lattice_Nt, self.pion_mass, self.num_eigenvectors)

    def analysis_type(self) -> str:
        """Decide analysis branch automatically (tetraquark overrides meson)."""
        if self.is_tetraquark_analysis:
            return "tetraquark"
        if self.is_meson_analysis:
            return "meson"
        raise ValueError("Neither meson nor tetraquark analysis is selected.")


# ======================================================
# 1. ENSEMBLE_DB (UNCHANGED - FULL DATA PRESERVED)
# ======================================================
ENSEMBLE_DB: Dict[EnsembleKey, EnsembleEntry] = {
    # ======================================================
    # Ensemble: (12, 96, 420, 170)
    # ======================================================
    (12, 96, 420, 170): {
        "at_invs": 7.219,
        "GEVP": (15, 25, 20),
        "meson": {
            "chan_name_list": [
                r"\eta_c",
                r"J/\psi",
            ],
            "chan_momt_list": [
                [0, 1, 2, 3, 4, 5, 6, 8, 9],  # no n^2=7 on lattice
                [0, 1, 2, 3, 4, 5, 6, 8, 9],
            ],
            "tmin_arry": [
                [32, 27, 26, 20, 20, 20, 20, 50, 20, 20],
                [29, 23, 20, 20, 20, 20, 20, 50, 20, 20],
            ],
            "prir_weff_0": [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            "prir_weff_1": [
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prir_meff_0": [
                [0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.49, 1.0, 0.51, 0.52],
                [0.43, 0.44, 0.45, 0.46, 0.48, 0.49, 0.50, 1.0, 0.52, 0.53],
            ],
            "prir_meff_1": [
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            ],
        },
        "tetraquark": {
            "chan_name_list": [
                r"\eta_c\,\eta_c",
                r"J/\psi\,J/\psi",
            ],
            "chan_momt_list": [
                [1, 2, 4],
                [0, 1, 2, 3, 4],
            ],
            "tmin_arry": [
                [50, 29, 28, 50, 28],
                [27, 28, 23, 26, 22],
            ],
            "prir_weff_0": [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            "prir_weff_1": [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            "prir_weff_2": [
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prir_meff_0": [
                [1.00, 0.86, 0.88, 1.00, 0.93],
                [0.86, 0.88, 0.90, 0.92, 0.94],
            ],
            "prir_meff_1": [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "prir_meff_2": [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        },
    },
    # ======================================================
    # Ensemble: (16, 128, 420, 120)
    # ======================================================
    (16, 128, 420, 120): {
        "at_invs": 7.219,
        "GEVP": (30, 40, 35),
        "meson": {
            "chan_name_list": [
                r"\eta_c",
                r"J/\psi",
            ],
            "chan_momt_list": [
                [0, 1, 2, 3, 4, 5, 6, 8, 9],  # no n^2=7 on lattice
                [0, 1, 2, 3, 4, 5, 6, 8, 9],
            ],
            "tmin_arry": [
                [26, 20, 21, 21, 21, 21, 21, 50, 21, 21],
                [26, 19, 18, 20, 20, 23, 20, 50, 20, 20],
            ],
            "prir_weff_0": [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            "prir_weff_1": [
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prir_meff_0": [
                [0.41, 0.42, 0.43, 0.435, 0.44, 0.45, 0.46, 1.0, 0.47, 0.48],
                [0.42, 0.43, 0.44, 0.45, 0.455, 0.46, 0.47, 1.0, 0.48, 0.49],
            ],
            "prir_meff_1": [
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            ],
        },
        "tetraquark": {
            "chan_name_list": [
                r"\eta_c\,\eta_c",
                r"J/\psi\,J/\psi",
            ],
            "chan_momt_list": [
                [1, 2, 4],
                [0, 1, 2, 3, 4],
            ],
            "tmin_arry": [
                [50, 29, 29, 50, 22, 50],
                [24, 28, 29, 29, 33, 50],
            ],
            "prir_weff_0": [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            "prir_weff_1": [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            "prir_weff_2": [
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prir_meff_0": [
                [1.00, 0.84, 0.86, 1.00, 0.89],
                [0.86, 0.87, 0.88, 0.90, 0.91],
            ],
            "prir_meff_1": [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "prir_meff_2": [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        },
    },
}
