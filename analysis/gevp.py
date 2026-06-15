from dataclasses import dataclass

import numpy as np
from scipy import linalg

from analysis.utils import fve_offsets
from input.config import Config
from input.types import FileDict


@dataclass
class GEVPPlotData:
    matrix_before: np.ndarray
    matrix_after: np.ndarray
    eigenvectors: np.ndarray


def process_GEVP(
    config: Config, raw_dict: FileDict
) -> tuple[FileDict, GEVPPlotData | None]:
    """
    Build FVE matrix from tetraquark correlators, optionally solve GEVP,
    and return correlators as [channel, momentum, time, sample].
    """
    corr_dict = dict(raw_dict)

    if not config.is_tetraquark_analysis:
        return corr_dict, None

    tetra_raw = raw_dict["tetraquark"]
    chan_momt_list = config.chan_momt_list
    n_mom_per_ch = [len(moms) for moms in chan_momt_list]
    n_fve = sum(n_mom_per_ch)
    offsets = fve_offsets(n_mom_per_ch)

    matrix_before_gevp = np.zeros(
        (n_fve, n_fve, tetra_raw.shape[-2], tetra_raw.shape[-1])
    )
    for ch_src, mom_list_src in enumerate(chan_momt_list):
        for mom_idx_src, mom_src in enumerate(mom_list_src):
            for ch_snk, mom_list_snk in enumerate(chan_momt_list):
                for mom_idx_snk, mom_snk in enumerate(mom_list_snk):
                    fve_src = offsets[ch_src] + mom_idx_src
                    fve_snk = offsets[ch_snk] + mom_idx_snk
                    matrix_before_gevp[fve_src, fve_snk] = tetra_raw[
                        ch_src, mom_src, ch_snk, mom_snk
                    ]

    print("matrix_before_gevp.shape:", matrix_before_gevp.shape)

    gevp_plot_data = None
    if config.is_gevp:
        matrix_after_gevp, _, eigenvectors = solve_GEVP(config, matrix_before_gevp)
        gevp_plot_data = GEVPPlotData(matrix_before_gevp, matrix_after_gevp, eigenvectors)
    else:
        matrix_after_gevp = matrix_before_gevp

    diag = np.diagonal(matrix_after_gevp, axis1=0, axis2=1)
    corr_after_gevp = np.zeros(
        (len(chan_momt_list), max(n_mom_per_ch), diag.shape[0], diag.shape[1])
    )

    fve_idx = 0
    for ch_idx, mom_list in enumerate(chan_momt_list):
        for mom in mom_list:
            corr_after_gevp[ch_idx, mom] = diag[:, :, fve_idx]
            fve_idx += 1

    corr_dict["tetraquark"] = corr_after_gevp
    print("corr_after_gevp.shape:", corr_after_gevp.shape)
    return corr_dict, gevp_plot_data


def solve_GEVP(
    config: Config, matrix_before_gevp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve C(t1) v = lambda C(t0) v and rotate the correlator matrix."""
    sample_axis = config.sample_axis
    t0, t1, _ = config.t_GEVP

    eigenvalues, eigenvectors = linalg.eig(
        a=matrix_before_gevp[:, :, t1, :].mean(sample_axis),
        b=matrix_before_gevp[:, :, t0, :].mean(sample_axis),
        right=True,
    )
    eigenvectors, _ = greedy_sort_eigenvectors(eigenvectors)

    matrix_after_gevp = np.einsum(
        "ai,abxy,bj->ijxy",
        eigenvectors.conj(),
        matrix_before_gevp,
        eigenvectors,
    )
    return matrix_after_gevp, eigenvalues, eigenvectors


def greedy_sort_eigenvectors(
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy assignment of eigenvectors to dominant matrix components."""
    n = eigenvectors.shape[1]
    candidates = sorted(
        (
            (abs(eigenvectors[row, col]), row, col)
            for col in range(n)
            for row in range(eigenvectors.shape[0])
        ),
        reverse=True,
    )

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    assignment: dict[int, int] = {}

    for _, row, col in candidates:
        if col in used_cols or row in used_rows:
            continue
        assignment[col] = row
        used_cols.add(col)
        used_rows.add(row)
        if len(assignment) == n:
            break

    for col, row in zip(
        (c for c in range(n) if c not in assignment),
        (r for r in range(n) if r not in used_rows),
    ):
        assignment[col] = row

    perm = np.array([col for col, _ in sorted(assignment.items(), key=lambda x: x[1])])
    return eigenvectors[:, perm], perm
