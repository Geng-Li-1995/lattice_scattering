from dataclasses import dataclass

import numpy as np
from scipy import linalg

from data.correlators import AnalysisCorrelators, Correlator4D, RawCorrelators
from input.config import Config


@dataclass
class GEVPPlotData:
    matrix_before: np.ndarray
    matrix_after: np.ndarray
    eigenvectors: np.ndarray


def process_GEVP(
    config: Config, raw: RawCorrelators
) -> tuple[AnalysisCorrelators, GEVPPlotData | None]:
    corr = AnalysisCorrelators.from_raw(raw)

    if not config.run_tetraquark_analysis or raw.tetraquark is None:
        return corr, None

    tetra = raw.tetraquark
    chan_momentum_list = config.chan_momentum_list
    n_mom_per_chan = [len(moms) for moms in chan_momentum_list]
    n_fve = sum(n_mom_per_chan)
    offsets = np.cumsum([0, *n_mom_per_chan[:-1]])

    matrix_before_gevp = np.zeros((n_fve, n_fve, tetra.n_time, tetra.n_sample))
    for chan_src, mom_list_src in enumerate(chan_momentum_list):
        for mom_idx_src, mom_src in enumerate(mom_list_src):
            for chan_snk, mom_list_snk in enumerate(chan_momentum_list):
                for mom_idx_snk, mom_snk in enumerate(mom_list_snk):
                    fve_src = offsets[chan_src] + mom_idx_src
                    fve_snk = offsets[chan_snk] + mom_idx_snk
                    matrix_before_gevp[fve_src, fve_snk] = tetra.at(
                        chan_src, mom_src, chan_snk, mom_snk
                    )

    print("matrix_before_gevp.shape:", matrix_before_gevp.shape)

    gevp_plot_data = None
    if config.run_GEVP_analysis:
        matrix_after_gevp, _, eigenvectors = solve_GEVP(config, matrix_before_gevp)
        gevp_plot_data = GEVPPlotData(matrix_before_gevp, matrix_after_gevp, eigenvectors)
    else:
        matrix_after_gevp = matrix_before_gevp

    diag = np.diagonal(matrix_after_gevp, axis1=0, axis2=1)
    corr_after_gevp = np.zeros(
        (len(chan_momentum_list), max(n_mom_per_chan), diag.shape[0], diag.shape[1])
    )

    fve_idx = 0
    for chan_idx, mom_list in enumerate(chan_momentum_list):
        for mom in mom_list:
            corr_after_gevp[chan_idx, mom] = diag[:, :, fve_idx]
            fve_idx += 1

    print("corr_after_gevp.shape:", corr_after_gevp.shape)
    return corr.with_tetraquark(corr_after_gevp), gevp_plot_data


def solve_GEVP(
    config: Config, matrix_before_gevp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
