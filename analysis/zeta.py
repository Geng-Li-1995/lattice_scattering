import os

import numpy as np
from joblib import Parallel, delayed

from data.io import moving_frame_d_tag
from input.config import Config

Q_SQ_MIN = -5.0
Q_SQ_MAX = 10.0
Y_00 = 1.0 / np.sqrt(4.0 * np.pi)


def q_sq_linspace(n_q: int) -> np.ndarray:
    return np.linspace(Q_SQ_MIN, Q_SQ_MAX, n_q)


def kallen(a, b, c) -> float:
    """Källén function."""
    return a**2 + b**2 + c**2 - 2 * (a * b + a * c + b * c)


def build_n_sq_array(lamda: int) -> np.ndarray:
    n_range = np.arange(-lamda - 1, lamda + 2)
    nx, ny, nz = np.meshgrid(n_range, n_range, n_range, indexing="ij")
    n_sq = nx**2 + ny**2 + nz**2
    return n_sq[n_sq <= lamda**2].astype(float)


def build_r_sq_from_n_vec(
    n_vec_array: np.ndarray,
    d_vec: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    d_hat = d_vec / np.linalg.norm(d_vec)
    x = n_vec_array - 0.5 * alpha * d_vec[None, :]
    x_para = np.dot(x, d_hat)[:, None] * d_hat[None, :]
    x_perp = x - x_para
    r_vec = x_perp + x_para / gamma
    return np.sum(r_vec * r_vec, axis=-1)


def build_moving_frame_n_vec_array(
    lamda: int,
    d_vec: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    shift = 0.5 * abs(alpha) * np.linalg.norm(d_vec)
    max_n = int(np.ceil(max(lamda, gamma * lamda + shift))) + 2
    n_range = np.arange(-max_n, max_n + 1)
    return np.array(np.meshgrid(n_range, n_range, n_range, indexing="ij")).T.reshape(-1, 3)


def gen_zeta_00_rest_from_q_sq_array(
    q_sq_grid: np.ndarray, lamda: int, regen_flag: bool = False
) -> np.ndarray:
    save_path = f"data/zeta/zeta_00_rest_lam{lamda}_nq{len(q_sq_grid)}.npy"
    if (not regen_flag) and os.path.exists(save_path):
        print(f"Loading {save_path}")
        return np.load(save_path)

    print(f"Generating {save_path} ...")
    n_sq_array = build_n_sq_array(lamda)
    zeta_array = np.asarray(
        Parallel(n_jobs=-1)(
            delayed(_gen_zeta_00_rest_from_q_sq)(n_sq_array, q_sq, lamda)
            for q_sq in q_sq_grid
        ),
        dtype=float,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, zeta_array)
    print(f"Saved to {save_path}")
    return zeta_array


def gen_zeta_00_moving_from_q_sq_array(
    q_sq_grid: np.ndarray,
    config: Config,
    ensemble_key,
    d_vec: np.ndarray,
    lamda: int,
    gamma: float,
    alpha: float,
) -> np.ndarray:
    ns, _, pion_mass, num_ev = ensemble_key
    d_tag = moving_frame_d_tag(tuple(int(component) for component in d_vec))
    save_path = (
        f"data/zeta/zeta_00_moving_{config.input_name}_"
        f"L{ns}M{pion_mass}_EV{num_ev}_"
        f"{d_tag}_lam{lamda}_nq{len(q_sq_grid)}.npy"
    )
    if (not config.regen_moving_frame_zeta) and os.path.exists(save_path):
        print(f"Loading {save_path}")
        return np.load(save_path)

    n_vec_array = build_moving_frame_n_vec_array(lamda, d_vec, alpha, gamma)
    r_sq_array = build_r_sq_from_n_vec(n_vec_array, d_vec, alpha, gamma)
    r_sq_array = r_sq_array[r_sq_array <= lamda**2]

    print(
        f"Generating {save_path} with alpha={alpha:.6g}, gamma={gamma:.6g}, "
        f"n_vec={len(n_vec_array)}, r_vec={len(r_sq_array)}"
    )
    zeta_array = np.asarray(
        [_gen_zeta_00_moving_from_q_sq(r_sq_array, q_sq, lamda) for q_sq in q_sq_grid],
        dtype=float,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, zeta_array)
    print(f"Saved to {save_path}")
    return zeta_array


def gen_zeta_00_moving_from_q_sq(r_sq_array: np.ndarray, q_sq: float, lamda: int) -> float:
    return _gen_zeta_00_moving_from_q_sq(r_sq_array, q_sq, lamda)


def _gen_zeta_00_rest_from_q_sq(n_sq_array: np.ndarray, q_sq: float, lamda: int) -> float:
    mask = ~np.isclose(n_sq_array, q_sq, atol=1e-12)
    return np.sum(1.0 / (n_sq_array[mask] - q_sq)) * Y_00 - 4.0 * np.pi * lamda * Y_00


def _gen_zeta_00_moving_from_q_sq(r_sq_array: np.ndarray, q_sq: float, lamda: int) -> float:
    mask = ~np.isclose(r_sq_array, q_sq)
    sum_part = np.sum(1.0 / (r_sq_array[mask] - q_sq))

    q = np.sqrt(q_sq) if q_sq > 1e-12 else 0.0
    if q > 0:
        integral_part = 4.0 * np.pi * lamda + 2.0 * np.pi * q * np.log(
            abs((lamda - q) / (lamda + q))
        )
    else:
        integral_part = 4.0 * np.pi * lamda

    return Y_00 * (sum_part - integral_part)
