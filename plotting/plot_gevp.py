import numpy as np
import matplotlib.pyplot as plt

from input.config import Config
from plotting.plot_set import (
    BasePlotter,
    FIG_MATRIX,
    FS_COLORBAR,
    FS_TICK,
    FS_TICK_DENSE,
    label_axes,
    new_figure,
    set_figure_title,
)


class GEVPPlotter(BasePlotter):

    def __init__(self, config: Config):
        super().__init__(config)
        self.tx = config.t_GEVP[2]

    def _save_gevp(self, name: str) -> None:
        self._save(f"result/{self.input_name}/{name}_{self.tag_name}")

    def _fve_labels(self) -> list[str]:
        return [
            rf"${self.chan_name_list[chan_idx]}({mom})$"
            for chan_idx, mom_list in enumerate(self.chan_momentum_list)
            for mom in mom_list
        ]

    @staticmethod
    def _normalize_at_t(matrix: np.ndarray, t: int) -> np.ndarray:
        tx_mean = matrix[:, :, t, :].mean(-1)
        return tx_mean / np.sqrt(np.outer(np.diag(tx_mean), np.diag(tx_mean)))

    def _plot_matrix(
        self, matrix, labels, title, filename, vmin=None, vmax=None
    ) -> None:
        new_figure(FIG_MATRIX)
        plt.imshow(matrix, cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax)
        for i in range(matrix.shape[0]):
            plt.gca().add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="black"))
        plt.colorbar().ax.tick_params(labelsize=FS_COLORBAR)
        plt.xticks(np.arange(len(labels)), labels, fontsize=FS_TICK_DENSE, rotation=45)
        plt.yticks(np.arange(len(labels)), labels, fontsize=FS_TICK_DENSE)
        set_figure_title(self._figure_title(prefix=title, sep=" on "))
        label_axes("snk", "src")
        self._save_gevp(filename)

    def plot_GEVP(
        self,
        matrix_before_gevp: np.ndarray,
        matrix_after_gevp: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> None:
        labels = self._fve_labels()

        new_figure(FIG_MATRIX)
        plt.imshow(eigenvectors.real, cmap="jet", interpolation="nearest", vmin=-1, vmax=1)
        plt.colorbar().ax.tick_params(labelsize=FS_COLORBAR)
        n = len(labels)
        plt.xticks(np.arange(n), np.arange(n) + 1, fontsize=FS_TICK)
        plt.yticks(np.arange(n), np.arange(n) + 1, fontsize=FS_TICK)
        label_axes(r"$\beta$", r"$n$")
        set_figure_title(
            self._figure_title(
                prefix=rf"GEVP eigenvector $v_\beta^{{(n)}}$",
                sep=" on ",
            )
        )
        self._save_gevp("GEVP_eigenvector")

        for matrix, title, filename, vmin, vmax in (
            (matrix_before_gevp, "Before", "GEVP_before", -0.2, 0.2),
            (matrix_after_gevp, "After", "GEVP_after", -0.02, 0.02),
        ):
            self._plot_matrix(
                self._normalize_at_t(matrix, self.tx),
                labels,
                title=rf"{title} GEVP ($t/a_t$={self.tx})",
                filename=filename,
                vmin=vmin,
                vmax=vmax,
            )
