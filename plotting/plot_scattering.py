import matplotlib.pyplot as plt

from input.config import Config
from plotting.plot_set import (
    COLORS,
    ERRORBAR_KW,
    FIG_STANDARD,
    ZORDER_AUX_LINE,
    ZORDER_FIT_CURVE,
    add_legend,
    apply_plot_style,
    label_axes,
    new_figure,
    plot_gvar_band,
    save_figure,
)


class ScatteringPlotter:

    def __init__(self, config: Config):
        apply_plot_style()
        self.config = config
        self.input_name = config.input_name

    def _save(self, name: str) -> None:
        save_figure(
            f"result/{self.input_name}/{name}_scattering",
            plot_format=self.config.plot_format,
        )

    def plot_Ks(self, scattering_dict: dict) -> None:
        if not self.config.is_tetraquark_analysis:
            return

        new_figure(FIG_STANDARD)
        ref_ns = self.config.scattering_list[0][0]
        plot_gvar_band(scattering_dict["s_array"][ref_ns], scattering_dict["fit_Ks_curve"])

        for ensemble_idx, ensemble_key in enumerate(self.config.scattering_list):
            ns = ensemble_key[0]
            color = COLORS[ensemble_idx]
            s_pts, ks_pts = scattering_dict["s"][ns], scattering_dict["Ks"][ns]
            plt.errorbar(
                s_pts[:, 0], ks_pts[:, 0],
                xerr=s_pts[:, 1], yerr=ks_pts[:, 1],
                fmt="x", color=color, label=f"L{ns}", **ERRORBAR_KW,
            )

        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.xlim(37, 45)
        plt.ylim(-9, 6)
        label_axes(r"$s\,(\rm{GeV}^2)$", r"$K(s)=\sqrt s / (k\cot\delta_0)$")
        add_legend("upper left")
        self._save("K_s")

    def plot_kcot(self, scattering_dict: dict) -> None:
        if not self.config.is_tetraquark_analysis:
            return

        new_figure(FIG_STANDARD)
        ref_ns = self.config.scattering_list[0][0]
        plot_gvar_band(scattering_dict["k_sq_array"][ref_ns], scattering_dict["fit_kcot_curve"])

        for ensemble_idx, ensemble_key in enumerate(self.config.scattering_list):
            ns = ensemble_key[0]
            color = COLORS[ensemble_idx]
            k_sq_grid = scattering_dict["k_sq_array"][ns]
            kcot_rest_grid = scattering_dict["kcot_rest_array"][ns]
            k_sq_mean = scattering_dict["k_sq"][ns][:, 0]
            k_sq_err = scattering_dict["k_sq"][ns][:, 1]

            plt.plot(
                k_sq_grid, kcot_rest_grid,
                color=color, linestyle="--", alpha=0.3, zorder=ZORDER_AUX_LINE,
            )
            for m, e in zip(k_sq_mean, k_sq_err):
                mask = (k_sq_grid >= m - e) & (k_sq_grid <= m + e)
                plt.plot(
                    k_sq_grid[mask], kcot_rest_grid[mask],
                    color=color, linestyle="-", zorder=ZORDER_FIT_CURVE,
                )

            plt.errorbar(
                k_sq_mean, scattering_dict["kcot"][ns][:, 0], xerr=k_sq_err,
                fmt="x", color=color, label=f"L{ns}", **ERRORBAR_KW,
            )

        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.xlim(-0.25, 1.75)
        plt.ylim(-15, 15)
        label_axes(r"$k^2\,(\rm{GeV}^2)$", r"$k\cot\delta_0\,$(GeV)")
        add_legend("upper left")
        self._save("kcot")
