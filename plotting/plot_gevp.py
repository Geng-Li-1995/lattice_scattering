import numpy as np
import matplotlib.pyplot as plt

from input.config import Config
from plotting.plot_set import (
    FIG_MATRIX,
    FS_COLORBAR,
    FS_TICK,
    FS_TICK_DENSE,
    apply_plot_style,
    label_axes,
    new_figure,
    save_figure,
)


class GEVPPlotter:

    def __init__(self, config: Config):
        apply_plot_style()
        self.config = config
        self.input_name = config.input_name
        self.tag_name = config.tag_name
        self.chan_momt_list = config.chan_momt_list
        self.chan_name_list = config.chan_name_list
        self.t0, self.t1, self.tx = config.t_GEVP

    def _save(self, name: str) -> None:
        save_figure(f"result/{self.input_name}/{name}_{self.tag_name}.pdf")

    def _fve_labels(self) -> list[str]:
        return [
            rf"${self.chan_name_list[ch_idx]}({mom})$"
            for ch_idx, mom_list in enumerate(self.chan_momt_list)
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
        plt.title(title)
        label_axes("snk", "src")
        self._save(filename)

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
        plt.title(rf"GEVP eigenvector $v_\beta^{{(n)}}$ on {self.tag_name}")
        self._save("GEVP_eigenvector")

        for matrix, title, filename, vmin, vmax in (
            (matrix_before_gevp, "Before", "GEVP_before", -0.2, 0.2),
            (matrix_after_gevp, "After", "GEVP_after", -0.02, 0.02),
        ):
            self._plot_matrix(
                self._normalize_at_t(matrix, self.tx),
                labels,
                title=rf"{title} GEVP on {self.tag_name} ($t/a_t$={self.tx})",
                filename=filename,
                vmin=vmin,
                vmax=vmax,
            )
