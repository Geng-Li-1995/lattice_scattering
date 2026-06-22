import numpy as np
from pathlib import Path

from data.correlators import Correlator4D, RawCorrelators, TetraquarkCorrelator
from input.config import Config
from input.types import EnsembleKey, ResampleDataDict


def _corr_types(config: Config) -> list[str]:
    corr_types = []
    if config.is_meson_analysis:
        corr_types.append("meson")
    if config.is_tetraquark_analysis:
        corr_types.append("tetraquark")
    return corr_types


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.asarray(np.load(path), dtype=np.float64)


def _ensemble_tag(ensemble_key: EnsembleKey) -> str:
    ns, _, pion_mass, num_ev = ensemble_key
    return f"L{ns}M{pion_mass}_EV{num_ev}"


def read_raw_files(config: Config) -> RawCorrelators:
    raw_dir = Path(f"data/{config.input_name}/raw")
    tag = _ensemble_tag(config.ensemble_key)
    meson = None
    tetraquark = None

    for corr_type in _corr_types(config):
        path = raw_dir / f"correlation_{corr_type}_{tag}.npy"
        arr = _load_npy(path)
        print(f"{corr_type} data.shape: {arr.shape}")
        if corr_type == "meson":
            meson = Correlator4D(arr)
        else:
            tetraquark = TetraquarkCorrelator(arr)

    return RawCorrelators(meson=meson, tetraquark=tetraquark)


def read_resampled_files(config: Config) -> ResampleDataDict:
    resampled: ResampleDataDict = {}

    if not config.run_scattering:
        return resampled

    resampled_dir = Path(f"data/{config.input_name}/resampled")

    for corr_type in ("meson", "tetraquark"):
        resampled[corr_type] = {}
        for ensemble_key in config.scattering_list:
            path = resampled_dir / f"resample_En_{corr_type}_{_ensemble_tag(ensemble_key)}.npy"
            arr = _load_npy(path)
            resampled[corr_type][ensemble_key] = arr
            print(f"Resampled {corr_type} Ns={ensemble_key[0]}, shape={arr.shape}")

    resampled["ksi"] = {}
    for ensemble_key in config.scattering_list:
        path = resampled_dir / f"resample_ksi_meson_{_ensemble_tag(ensemble_key)}.npy"
        arr = _load_npy(path)
        resampled["ksi"][ensemble_key] = arr
        print(f"Resampled ksi meson shape: {arr.shape}")

    return resampled


def read_file(config: Config) -> tuple[RawCorrelators, ResampleDataDict]:
    raw = read_raw_files(config)
    return raw, read_resampled_files(config)
