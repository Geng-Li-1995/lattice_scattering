import numpy as np
from pathlib import Path
from typing import List, Tuple

from input.config import Config
from input.types import FileDict, ResampleDataDict


def _analysis_type_list(config: Config) -> List[str]:
    type_list = []
    if config.is_meson_analysis:
        type_list.append("meson")
    if config.is_tetraquark_analysis:
        type_list.extend(["meson", "tetraquark"])

    if not type_list:
        raise ValueError("No channel requested in config.")
    return type_list


def read_raw_files(config: Config) -> FileDict:
    """
    Load raw correlation functions only (no resampled data).

    Meson shape:      [channel, momentum, time, sample]
    Tetraquark shape: [channel_src, momentum_src, channel_snk, momentum_snk, time, sample]
    """
    input_name = config.input_name
    raw_dir = Path(f"data/{input_name}/raw")
    lattice_Ns, lattice_Nt, pion_mass, num_eigenvectors = config.ensemble_key

    file_dict: FileDict = {}

    for type_name in _analysis_type_list(config):
        corr_file = raw_dir / (
            f"correlation_{type_name}_L{lattice_Ns}M{pion_mass}_EV{num_eigenvectors}.npy"
        )
        if not corr_file.exists():
            raise FileNotFoundError(f"Missing file: {corr_file}")
        file_dict[type_name] = np.asarray(np.load(corr_file), dtype=np.float64)
        print(f"{type_name} data.shape: {file_dict[type_name].shape}")

    return file_dict


def read_file(config: Config) -> Tuple[FileDict, ResampleDataDict]:
    """
    Load lattice data according to the given lattice configuration.

    Meson shape:      [channel, momentum, time, sample]
    Tetraquark shape: [channel_src, momentum_src, channel_snk, momentum_snk, time, sample]
    """
    input_name = config.input_name
    resampled_dir = Path(f"data/{input_name}/resampled")

    file_dict = read_raw_files(config)
    resample_data_dict: ResampleDataDict = {}

    if not config.run_scattering:
        return file_dict, resample_data_dict

    type_list = _analysis_type_list(config)

    for type_name in type_list:
        resample_data_dict.setdefault(type_name, {})

        for scattering in config.scattering_list:
            Ns, _, M, EV = scattering

            resample_En_file = resampled_dir / (
                f"resample_En_{type_name}_L{Ns}M{M}_EV{EV}.npy"
            )
            if not resample_En_file.exists():
                raise FileNotFoundError(f"Missing file: {resample_En_file}")
            resample_data_dict[type_name][scattering] = np.asarray(
                np.load(resample_En_file), dtype=np.float64
            )
            print(
                f"Resampled {type_name} Ns={Ns}, "
                f"shape={resample_data_dict[type_name][scattering].shape}"
            )

    resample_data_dict.setdefault("ksi", {})
    for scattering in config.scattering_list:
        Ns, _, M, EV = scattering

        resample_ksi_file = resampled_dir / (
            f"resample_ksi_meson_L{Ns}M{M}_EV{EV}.npy"
        )
        if not resample_ksi_file.exists():
            raise FileNotFoundError(f"Missing file: {resample_ksi_file}")
        resample_data_dict["ksi"][scattering] = np.asarray(
            np.load(resample_ksi_file), dtype=np.float64
        )
        print(
            f"Resampled ksi meson shape: "
            f"{resample_data_dict['ksi'][scattering].shape}"
        )

    return file_dict, resample_data_dict
