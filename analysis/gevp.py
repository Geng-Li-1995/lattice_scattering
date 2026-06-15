# gevp.py

import numpy as np
from scipy import linalg
from plotting.plot_gevp import GEVPPlotter


def process_GEVP(config, file_dict):
    """
    Reshape tetraquark data from
    [channel_src, momentum_src, channel_snk, momentum_snk, time, sample]
    to
    [nop_src, nop_snk, time, sample]
    by selecting source and sink operators in chan_momt_list.

    Also used for visualizing correlation matrices before/after GEVP.

    Return:
        data_after_GEVP [channel, momentum, time, sample],
        where source and sink operators are identified (src = snk).
    """
    # -----------------------------
    # 0. Meson data unchanged
    # -----------------------------
    data_dict = {key: file_dict[key] for key in file_dict}

    if not getattr(config, "is_tetraquark_analysis", False):
        return data_dict

    # -----------------------------
    # 1. Process tetraquark data
    # -----------------------------
    # Select source and sink channels/momenta
    data = file_dict["tetraquark"]
    chan_momt_list = config.chan_momt_list
    n_chan = len(chan_momt_list)
    n_mom_perchannel = [len(x) for x in chan_momt_list]
    n_fve = sum(n_mom_perchannel)
    matrix_before_GEVP = np.zeros((n_fve, n_fve, data.shape[-2], data.shape[-1]))
    for ch_idx_src, mom_list_src in enumerate(chan_momt_list):
        for mom_idx_src, mom_src in enumerate(mom_list_src):
            for ch_idx_snk, mom_list_snk in enumerate(chan_momt_list):
                for mom_idx_snk, mom_snk in enumerate(mom_list_snk):
                    # fill available GEVP diagonal element
                    matrix_before_GEVP[
                        sum(n_mom_perchannel[:ch_idx_src]) + mom_idx_src,
                        sum(n_mom_perchannel[:ch_idx_snk]) + mom_idx_snk,
                    ] = data[ch_idx_src, mom_src, ch_idx_snk, mom_snk]

    print("matrix_before_GEVP.shape is:", matrix_before_GEVP.shape)

    # Decide whether to perform GEVP and plot the matrix.
    # matrix_after_GEVP has shape: (nop_src, nop_snk, time, sample)
    if getattr(config, "is_gevp", False):
        matrix_after_GEVP, eigenvalues, sorted_eigenvectors = solve_GEVP(
            config, matrix_before_GEVP
        )
    else:
        matrix_after_GEVP = matrix_before_GEVP

    if not getattr(config, "run_resample", False):
        GEVPPlotter(config).plot_GEVP(
            matrix_before_GEVP, matrix_after_GEVP, sorted_eigenvectors
        )

    # Extract diagonal shape -> (time, sample, n_fves)
    diag_data = np.diagonal(matrix_after_GEVP, axis1=0, axis2=1)

    # Prepare output shape -> (channel, momentum, time, sample)
    data_after_GEVP = np.zeros(
        (n_chan, np.max(n_mom_perchannel), diag_data.shape[0], diag_data.shape[1])
    )

    fve_idx = 0
    for ch_idx, mom_list in enumerate(chan_momt_list):
        for mom_idx, mom in enumerate(mom_list):
            # fill available GEVP diagonal element
            data_after_GEVP[ch_idx, mom] = diag_data[:, :, fve_idx]
            fve_idx += 1

    data_dict["tetraquark"] = data_after_GEVP

    print("data_after_GEVP.shape is:", data_after_GEVP.shape)

    return data_dict


def solve_GEVP(config, matrix):
    """
    Solve GEVP:
        C(t) v_n = lambda_n C(t0) v_n

    Returns
    -------
    ndarray
        Rotated correlator matrix.
    """
    sample_axis = config.sample_axis
    t0, t1, tx = config.t_GEVP

    eigenvalues, eigenvectors = linalg.eig(
        a=matrix[:, :, t1, :].mean(sample_axis),
        b=matrix[:, :, t0, :].mean(sample_axis),
        right=True,
    )

    # Sort columns by their maximum absolute value (largest first)

    # sorted_eigenvectors = eigenvectors[
    #     :, np.argsort(np.argmax(np.abs(eigenvectors), axis=0))
    # ]

    sorted_eigenvectors, perm = greedy_sort_eigenvectors(eigenvectors)

    data_after_GEVP = np.einsum(
        "ai,abxy,bj->ijxy",
        sorted_eigenvectors.conj(),
        matrix,
        sorted_eigenvectors,
    )

    return data_after_GEVP, eigenvalues, sorted_eigenvectors


def greedy_sort_eigenvectors(eigenvectors):
    """
    Greedy assignment of eigenvectors to dominant components.

    Parameters
    ----------
    eigenvectors : (N, N) array
        Columns are eigenvectors.

    Returns
    -------
    sorted_eigenvectors : (N, N)
        Columns reordered uniquely.
    perm : (N,)
        permutation indices.
    """

    n = eigenvectors.shape[1]

    # (score, row, col)
    # score = |eigenvector entry|
    candidates = []
    for col in range(n):
        for row in range(eigenvectors.shape[0]):
            candidates.append((abs(eigenvectors[row, col]), row, col))

    # sort by strongest components first
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_rows = set()
    used_cols = set()
    assignment = {}

    # greedy matching
    for val, row, col in candidates:
        if col in used_cols:
            continue
        if row in used_rows:
            continue

        assignment[col] = row
        used_cols.add(col)
        used_rows.add(row)

        if len(assignment) == n:
            break

    # fallback: ensure completeness (in case degeneracy)
    remaining_cols = [c for c in range(n) if c not in assignment]
    remaining_rows = [r for r in range(n) if r not in used_rows]

    for c, r in zip(remaining_cols, remaining_rows):
        assignment[c] = r

    # build permutation
    perm = np.array([col for col, _ in sorted(assignment.items(), key=lambda x: x[1])])

    sorted_eigenvectors = eigenvectors[:, perm]

    return sorted_eigenvectors, perm
