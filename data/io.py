import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any


def read_file(config) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[Any, np.ndarray]]]:
    """
    Load lattice data according to the given lattice configuration.

    Meson shape:      [channel, momentum, time, sample]
    Tetraquark shape: [channel_src, momentum_src, channel_snk, momentum_snk, time, sample]
    """

    input_name = config.input_name

    raw_dir = Path(f"data/{input_name}/raw")
    resampled_dir = Path(f"data/{input_name}/resampled")

    lattice_Ns, lattice_Nt, pion_mass, num_eigenvectors = config.ensemble_key

    file_dict: Dict[str, np.ndarray] = {}
    resample_data_dict: Dict[str, Dict[Any, np.ndarray]] = {}

    # -----------------------------
    # Determine which channels to load
    # -----------------------------
    type_list = []
    if getattr(config, "is_meson_analysis", False):
        type_list.append("meson")
    if getattr(config, "is_tetraquark_analysis", False):
        type_list.extend(["meson", "tetraquark"])

    if not type_list:
        raise ValueError("No channel requested in config.")

    # -----------------------------
    # Helper
    # -----------------------------
    def load_npy(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        return np.asarray(np.load(path), dtype=np.float64)

    def build_raw_path(*parts: str) -> Path:
        return raw_dir.joinpath(*parts)

    def build_resampled_path(*parts: str) -> Path:
        return resampled_dir.joinpath(*parts)

    # -----------------------------
    # Load data
    # -----------------------------
    for type_name in type_list:
        corr_file = build_raw_path(
            f"correlation_{type_name}_L{lattice_Ns}M{pion_mass}_EV{num_eigenvectors}.npy"
        )
        file_dict[type_name] = load_npy(corr_file)
        print(f"{type_name} data.shape: {file_dict[type_name].shape}")

        # -----------------------------
        # Scattering resamples
        # -----------------------------
        if getattr(config, "run_scattering", False):
            resample_data_dict.setdefault(type_name, {})
            resample_data_dict.setdefault("ksi", {})

            for scattering in getattr(config, "scattering_list", []):
                Ns, _, M, EV = scattering

                resample_En_file = build_resampled_path(
                    f"resample_En_{type_name}_L{Ns}M{M}_EV{EV}.npy"
                )
                resample_data_dict[type_name][scattering] = load_npy(resample_En_file)
                print(
                    f"Resampled {type_name} Ns={Ns}, shape={resample_data_dict[type_name][scattering].shape}"
                )

                resample_ksi_file = build_resampled_path(
                    f"resample_ksi_meson_L{Ns}M{M}_EV{EV}.npy"
                )
                resample_data_dict["ksi"][scattering] = load_npy(resample_ksi_file)
                print(
                    f"Resampled ksi meson shape: {resample_data_dict['ksi'][scattering].shape}"
                )

    return file_dict, resample_data_dict
