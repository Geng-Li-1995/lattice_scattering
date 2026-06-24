from pathlib import Path

import numpy as np

from input.config import Config, EnsembleKey, ensemble_tag, moving_frame_d_tag

SCATTER_STAT_NAMES = ("Ks", "s", "k_sq", "kcot")


def _mf_scatter_dir(config: Config) -> Path:
    return Path(f"data/{config.input_name}/resampled")


def mf_scatter_path(config: Config, ensemble_key: EnsembleKey) -> Path:
    d_tag = moving_frame_d_tag(config.moving_frame_d_vec)
    tag = ensemble_tag(ensemble_key)
    lamda = config.moving_frame_zeta_lamda
    return _mf_scatter_dir(config) / f"resample_scatter_MF_{d_tag}_lam{lamda}_{tag}.npy"


def mf_scatter_ref_path(config: Config, ensemble_key: EnsembleKey) -> Path:
    d_tag = moving_frame_d_tag(config.moving_frame_d_vec)
    tag = ensemble_tag(ensemble_key)
    lamda = config.moving_frame_zeta_lamda
    n_q = config.moving_frame_zeta_n_q
    return (
        _mf_scatter_dir(config)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mf_scatter_results_to_array(results))


def load_mf_scatter_ref(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    array = np.asarray(np.load(path), dtype=np.float64)
    if array.shape[0] != 2:
        raise ValueError(f"Unexpected MF scatter ref cache shape in {path}: {array.shape}")
    return array[0], array[1]


def save_mf_scatter_ref(path: Path, k_sq_ref: np.ndarray, kcot_ref: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.stack([k_sq_ref, kcot_ref], axis=0))
