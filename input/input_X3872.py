# input_X3872.py

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

from input.types import EnsembleKey, EnsembleEntry, ScatteringList


@dataclass
class InputControl:
    """Central configuration controlling analysis, plotting, and lattice setup.

    Meson and tetraquark switches are independent. If both are False, the main
    analysis steps are skipped while scattering can still use resampled inputs.
    """

    tetraquark_name: str = "X3872"

    # Lattice setup
    lattice_Ns: int = 16  # supported: 12, 16
    lattice_Nt: int = 0
    num_eigenvectors: int = 0
    pion_mass: int = 0

    # Analysis options
    is_meson_analysis: bool = True
    is_tetraquark_analysis: bool = False
    is_gevp: bool = True
    is_svd: bool = False
    is_ratio: bool = False

    run_tmin: bool = False
    run_resample: bool = True  
    run_scattering: bool = False

    plot_meff: bool = True  # En / Zn plots (fit always runs)
    plot_dispersion: bool = True  # dispersion fit + plot (meson mode only)
    plot_format: str = "png"  # figure output: "png" or "pdf"

    resample_type: str = "jackknife"
    sample_axis: int = -1
    n_boot: int = 1000

    Ns_list: ClassVar[list[int]] = [16]  # lattice sizes for scattering

    # Scattering analysis: channel indices and fit momentum subsets per Ns
    ch_meson_a: int = 5
    ch_meson_b: int = 6
    ch_tetra: int = 0
    fit_mom_by_ns: Dict[int, List[int]] = field(default_factory=lambda: {16: [0, 1]})

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
                return 12, 96, 420, 70
            case 16:
                return 16, 128, 420, 70
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
    # Ensemble: (16, 128,420, 70)
    # ======================================================
    (16, 128, 420, 70): {
        "at_invs": 7.219,
        "GEVP": (20, 30, 25),
        "meson": {
            "chan_name_list": [
                r"\pi",
                r"J/\psi",
                r"\rho",
                r"\eta_c",
                r"D",
                r"D^*",
            ],
            "chan_momt_list": [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ],
            "tmin_arry": [
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
                [20, 20, 20, 20, 20],
            ],
            "prir_weff_0": [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            "prir_weff_1": [
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            "prir_meff_0": [
                [0.06, 0.095, 0.12, 0.145, 0.16],
                [0.425, 0.435, 0.44, 0.45, 0.455],
                [0.12, 0.14, 0.16, 0.175, 0.19],
                [0.415, 0.42, 0.43, 0.435, 0.44],
                [0.26, 0.275, 0.285, 0.295, 0.305],
                [0.28, 0.29, 0.3, 0.31, 0.32],
            ],
            "prir_meff_1": [
                [0.3, 0.3, 0.3, 0.3, 0.3],
                [0.6, 0.6, 0.6, 0.6, 0.6],
                [0.3, 0.3, 0.3, 0.3, 0.3],
                [0.6, 0.6, 0.6, 0.6, 0.6],
                [0.4, 0.4, 0.4, 0.4, 0.4],
                [0.4, 0.4, 0.4, 0.4, 0.4],
            ],
        },
        "tetraquark": {
            "chan_name_list": [
                r"\pi\,J/\psi",
                r"\rho\,\eta_c",
                r"D\,D^*",
                r"D^*\,D^*",
            ],
            "chan_momt_list": [
                [0, 1],
                [0, 1],
                [0],
                [],
            ],
            "tmin_arry": [
                [25, 25, 25],
                [25, 20, 25],
                [25, 20, 20],
                [25, 25, 25],
            ],
            "prir_weff_0": [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            "prir_weff_1": [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, 0.5],
            ],
            "prir_weff_2": [
                [0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01],
                [0.01, 0.2, 0.2],
                [0.01, 0.2, 0.1],
            ],
            "prir_meff_0": [
                [0.49, 0.53, 0.565],
                [0.53, 0.57, 0.59],
                [0.54, 0.56, 0.59],
                [0.56, 0.50, 0.60],
            ],
            "prir_meff_1": [
                [0.37, 0.34, 0.317],
                [0.30, 0.30, 0.27],
                [0.06, 0.4, 0.2],
                [0.00, 0.00, 0.00],
            ],
            "prir_meff_2": [
                [0.6, 0.6, 0.7],
                [0.7, 0.7, 0.7],
                [0.6, 0.6, 0.6],
                [0.6, 0.6, 0.6],
            ],
        },
    },
}
