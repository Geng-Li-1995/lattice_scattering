import numpy as np
from pathlib import Path

from input.config import Config
from input.types import EnsembleKey, FileDict, ResampleDataDict


def _corr_types(config: Config) -> list[str]:
    if config.is_tetraquark_analysis:
        return ["meson", "tetraquark"]
    if config.is_meson_analysis:
        return ["meson"]
    raise ValueError("No channel requested in config.")


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.asarray(np.load(path), dtype=np.float64)


def _ensemble_tag(ensemble_key: EnsembleKey) -> str:
    ns, _, pion_mass, num_ev = ensemble_key
    return f"L{ns}M{pion_mass}_EV{num_ev}"


def read_raw_files(config: Config) -> FileDict:
    """
    Load raw correlation functions only (no resampled data).

    Meson shape:      [channel, momentum, time, sample]
    Tetraquark shape: [channel_src, momentum_src, channel_snk, momentum_snk, time, sample]
    """
    raw_dir = Path(f"data/{config.input_name}/raw")
    tag = _ensemble_tag(config.ensemble_key)
    raw_dict: FileDict = {}

    for corr_type in _corr_types(config):
        path = raw_dir / f"correlation_{corr_type}_{tag}.npy"
        raw_dict[corr_type] = _load_npy(path)
        print(f"{corr_type} data.shape: {raw_dict[corr_type].shape}")

    return raw_dict


def read_file(config: Config) -> tuple[FileDict, ResampleDataDict]:
    """Load raw correlators and, when scattering is enabled, resampled energies."""
    raw_dict = read_raw_files(config)
    resampled_dict: ResampleDataDict = {}

    if not config.run_scattering:
        return raw_dict, resampled_dict

    resampled_dir = Path(f"data/{config.input_name}/resampled")

    for corr_type in _corr_types(config):
        resampled_dict[corr_type] = {}
        for ensemble_key in config.scattering_list:
            path = resampled_dir / f"resample_En_{corr_type}_{_ensemble_tag(ensemble_key)}.npy"
            arr = _load_npy(path)
            resampled_dict[corr_type][ensemble_key] = arr
            print(f"Resampled {corr_type} Ns={ensemble_key[0]}, shape={arr.shape}")

    resampled_dict["ksi"] = {}
    for ensemble_key in config.scattering_list:
        path = resampled_dir / f"resample_ksi_meson_{_ensemble_tag(ensemble_key)}.npy"
        arr = _load_npy(path)
        resampled_dict["ksi"][ensemble_key] = arr
        print(f"Resampled ksi meson shape: {arr.shape}")

    return raw_dict, resampled_dict
