from pathlib import Path

import numpy as np

from data.correlators import Correlator4D, RawCorrelators, TetraquarkCorrelator
from input.config import (
    Config,
    EnsembleKey,
    ResampleDataDict,
    ensemble_tag,
    MF_d_tag,
)

SCATTER_STAT_NAMES = ("Ks", "s", "k_sq", "kcot")


def data_dir(config: Config) -> Path:
    return Path(f"data/{config.input_name}")


def raw_dir(config: Config) -> Path:
    return data_dir(config) / "raw"


def resampled_dir(config: Config) -> Path:
    return data_dir(config) / "resampled"


def raw_correlator_path(config: Config, corr_type: str) -> Path:
    tag = ensemble_tag(config.ensemble_key)
    return raw_dir(config) / f"correlation_{corr_type}_{tag}.npy"


def resample_en_path(config: Config, corr_type: str, ensemble_key: EnsembleKey) -> Path:
    tag = ensemble_tag(ensemble_key)
    return resampled_dir(config) / f"resample_En_{corr_type}_{tag}.npy"


def resample_en_mf_tetraquark_path(config: Config, ensemble_key: EnsembleKey) -> Path:
    d_tag = MF_d_tag(config.MF_d_vec)
    tag = ensemble_tag(ensemble_key)
    return resampled_dir(config) / f"resample_En_{d_tag}_tetraquark_{tag}.npy"


def resample_ksi_meson_path(config: Config, ensemble_key: EnsembleKey) -> Path:
    tag = ensemble_tag(ensemble_key)
    return resampled_dir(config) / f"resample_ksi_meson_{tag}.npy"


def _corr_types(config: Config) -> list[str]:
    corr_types = []
    if config.run_meson_analysis or (config.run_tetraquark_analysis and config.run_ratio_analysis):
        corr_types.append("meson")
    if config.run_tetraquark_analysis:
        corr_types.append("tetraquark")
    return corr_types


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.asarray(np.load(path), dtype=np.float64)


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_resample_en(
    config: Config, corr_type: str, ensemble_key: EnsembleKey, array: np.ndarray
) -> Path:
    path = resample_en_path(config, corr_type, ensemble_key)
    save_npy(path, array)
    return path


def save_resample_ksi_meson(
    config: Config, ensemble_key: EnsembleKey, array: np.ndarray
) -> Path:
    path = resample_ksi_meson_path(config, ensemble_key)
    save_npy(path, array)
    return path


# ---------------------------------------------------------------------------
# Raw and resampled energy I/O
# ---------------------------------------------------------------------------
def read_raw_files(config: Config) -> RawCorrelators:
    meson = None
    tetraquark = None

    for corr_type in _corr_types(config):
        path = raw_correlator_path(config, corr_type)
        arr = _load_npy(path)
        print(f"{corr_type} data.shape: {arr.shape}")
        if corr_type == "meson":
            meson = Correlator4D(arr)
        else:
            tetraquark = TetraquarkCorrelator(arr)

    return RawCorrelators(meson=meson, tetraquark=tetraquark)


def read_resampled_files(config: Config) -> ResampleDataDict:
    resampled: ResampleDataDict = {}

    if not config.run_scattering_analysis:
        return resampled

    for corr_type in ("meson", "tetraquark"):
        resampled[corr_type] = {}
        for ensemble_key in config.scattering_list:
            path = resample_en_path(config, corr_type, ensemble_key)
            arr = _load_npy(path)
            resampled[corr_type][ensemble_key] = arr
            print(f"Resampled {corr_type} Ns={ensemble_key[0]}, shape={arr.shape}")

    if config.run_MF_analysis:
        resampled["tetraquark_MF"] = {}
        d_tag = MF_d_tag(config.MF_d_vec)
        for ensemble_key in config.scattering_list:
            path = resample_en_mf_tetraquark_path(config, ensemble_key)
            arr = _load_npy(path)
            resampled["tetraquark_MF"][ensemble_key] = arr
            print(f"Resampled MF tetraquark ({d_tag}) Ns={ensemble_key[0]}, shape={arr.shape}")

    resampled["ksi"] = {}
    for ensemble_key in config.scattering_list:
        path = resample_ksi_meson_path(config, ensemble_key)
        arr = _load_npy(path)
        resampled["ksi"][ensemble_key] = arr
        print(f"Resampled ksi meson shape: {arr.shape}")

    return resampled


# ---------------------------------------------------------------------------
# Moving-frame scatter cache I/O
# ---------------------------------------------------------------------------
def mf_scatter_path(config: Config, ensemble_key: EnsembleKey) -> Path:
    d_tag = MF_d_tag(config.MF_d_vec)
    tag = ensemble_tag(ensemble_key)
    lamda = config.MF_zeta_lamda
    return resampled_dir(config) / f"resample_scatter_MF_{d_tag}_lam{lamda}_{tag}.npy"


def mf_scatter_ref_path(config: Config, ensemble_key: EnsembleKey) -> Path:
    d_tag = MF_d_tag(config.MF_d_vec)
    tag = ensemble_tag(ensemble_key)
    lamda = config.MF_zeta_lamda
    n_q = config.MF_zeta_n_q
    return (
        resampled_dir(config)
        / f"resample_scatter_MF_ref_{d_tag}_lam{lamda}_nq{n_q}_{tag}.npy"
    )


def mf_scatter_results_to_array(results: list[dict]) -> np.ndarray:
    """Shape (n_level, n_sample, n_stat)."""
    return np.stack(
        [
            np.stack([result[name] for name in SCATTER_STAT_NAMES], axis=-1)
            for result in results
        ],
        axis=0,
    )


def mf_scatter_array_to_results(array: np.ndarray) -> list[dict[str, np.ndarray]]:
    return [
        {
            name: array[level_idx, :, stat_idx]
            for stat_idx, name in enumerate(SCATTER_STAT_NAMES)
        }
        for level_idx in range(array.shape[0])
    ]


def load_mf_scatter_all(path: Path) -> list[dict[str, np.ndarray]] | None:
    if not path.exists():
        return None
    array = np.asarray(np.load(path), dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != len(SCATTER_STAT_NAMES):
        raise ValueError(f"Unexpected MF scatter cache shape in {path}: {array.shape}")
    return mf_scatter_array_to_results(array)


def save_mf_scatter_all(path: Path, results: list[dict]) -> None:
    save_npy(path, mf_scatter_results_to_array(results))


def load_mf_scatter_ref(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    array = np.asarray(np.load(path), dtype=np.float64)
    if array.shape[0] != 2:
        raise ValueError(f"Unexpected MF scatter ref cache shape in {path}: {array.shape}")
    return array[0], array[1]


def save_mf_scatter_ref(path: Path, k_sq_ref: np.ndarray, kcot_ref: np.ndarray) -> None:
    save_npy(path, np.stack([k_sq_ref, kcot_ref], axis=0))
