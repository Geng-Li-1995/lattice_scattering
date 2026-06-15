import numpy as np
import matplotlib.pyplot as plt

from input.config import Config


class GEVPPlotter:

    def __init__(self, config: Config):

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
            }
        )

        self.config = config
        self.input_name = config.input_name
        self.tag_name = config.tag_name
        self.chan_momt_list = config.chan_momt_list
        self.chan_name_list = config.chan_name_list

        self.t0, self.t1, self.tx = config.t_GEVP

    # ==================================================
    # colors / styles (optional consistency)
    # ==================================================
    COLORS = [
        "blue",
        "red",
        "green",
        "violet",
        "orange",
        "black",
        "cyan",
        "navy",
        "yellow",
        "brown",
    ]

    # ==================================================
    # save helper
    # ==================================================
    def _save(self, name):
        plt.tight_layout()
        plt.savefig(f"result/{self.input_name}/{name}_{self.tag_name}.pdf")
        plt.show()

    # ==================================================
    # matrix helper (unchanged logic)
    # ==================================================
    def _plot_matrix(self, matrix, labels, title, filename, vmin=None, vmax=None):

        plt.figure(figsize=(7, 6))
        plt.imshow(matrix, cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax)

        for i in range(matrix.shape[0]):
            plt.gca().add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="black"))

        plt.colorbar().ax.tick_params(labelsize=12)
        plt.xticks(np.arange(len(labels)), labels, fontsize=10, rotation=45)
        plt.yticks(np.arange(len(labels)), labels, fontsize=10)

        plt.title(title, fontsize=15)
        plt.xlabel("snk", fontsize=15)
        plt.ylabel("src", fontsize=15)

        self._save(filename)

    # ==================================================
    # main API
    # ==================================================
    def plot_GEVP(
        self,
        matrix_before_GEVP: np.ndarray,
        matrix_after_GEVP: np.ndarray,
        sorted_eigenvectors: np.ndarray,
    ) -> None:

        # -------------------------
        # build labels
        # -------------------------
        label_list = [
            rf"${self.chan_name_list[ch_idx]}({mom})$"
            for ch_idx, mom_list in enumerate(self.chan_momt_list)
            for mom in mom_list
        ]

        # ==================================================
        # 1. eigenvectors
        # ==================================================
        plt.figure(figsize=(7, 6))

        plt.imshow(
            sorted_eigenvectors.real,
            cmap="jet",
            interpolation="nearest",
            vmin=-1,
            vmax=1,
        )

        plt.colorbar().ax.tick_params(labelsize=15)
        plt.xticks(
            np.arange(len(label_list)), np.arange(len(label_list)) + 1, fontsize=15
        )
        plt.yticks(
            np.arange(len(label_list)), np.arange(len(label_list)) + 1, fontsize=15
        )

        plt.xlabel(r"$\beta$", fontsize=15)
        plt.ylabel(r"$n$", fontsize=15)
        plt.title(
            rf"GEVP eigenvector $v_\beta^{{(n)}}$ on {self.tag_name}", fontsize=15
        )

        self._save("GEVP_eigenvector")

        # ==================================================
        # 2. before GEVP
        # ==================================================
        matrix_before_tx_mean = matrix_before_GEVP[:, :, self.tx, :].mean(-1)

        matrix_before_tx_norm = matrix_before_tx_mean / np.sqrt(
            np.outer(
                np.diag(matrix_before_tx_mean),
                np.diag(matrix_before_tx_mean),
            )
        )

        self._plot_matrix(
            matrix_before_tx_norm,
            label_list,
            title=rf"Before GEVP on {self.tag_name} ($t/a_t$={self.tx})",
            filename="GEVP_before",
            vmin=-0.2,
            vmax=0.2,
        )

        # ==================================================
        # 3. after GEVP
        # ==================================================
        matrix_after_tx_mean = matrix_after_GEVP[:, :, self.tx, :].mean(-1)

        matrix_after_tx_norm = matrix_after_tx_mean / np.sqrt(
            np.outer(
                np.diag(matrix_after_tx_mean),
                np.diag(matrix_after_tx_mean),
            )
        )

        self._plot_matrix(
            matrix_after_tx_norm,
            label_list,
            title=rf"After GEVP on {self.tag_name} ($t/a_t$={self.tx})",
            filename="GEVP_after",
            vmin=-0.02,
            vmax=0.02,
        )
