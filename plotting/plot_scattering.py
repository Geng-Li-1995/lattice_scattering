import matplotlib.pyplot as plt
import numpy as np

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

    def _frame_slices(self, scattering_dict: dict, ns: int, n_points: int, ensemble_idx: int):
        if not self.config.is_moving_frame:
            return [(slice(0, n_points), f"L{ns}", COLORS[ensemble_idx % len(COLORS)])]

        rest_count = scattering_dict["rest_point_count"][ns]
        color_idx = 2 * ensemble_idx
        return [
            (slice(0, rest_count), f"L{ns} rest", COLORS[color_idx % len(COLORS)]),
            (slice(rest_count, n_points), f"L{ns} moving", COLORS[(color_idx + 1) % len(COLORS)]),
        ]

    @staticmethod
    def _axis_limits(mean, err, padding_fraction: float = 0.15) -> tuple[float, float]:
        mean = np.asarray(mean, dtype=float)
        err = np.asarray(err, dtype=float)
        lo = np.min(mean - err)
        hi = np.max(mean + err)
        span = hi - lo
        if span <= 0:
            span = max(abs(hi), 1.0)
        padding = padding_fraction * span
        return lo - padding, hi + padding

    @staticmethod
    def _combine_axis_limits(*limits: tuple[float, float]) -> tuple[float, float]:
        return min(lo for lo, _ in limits), max(hi for _, hi in limits)

    def plot_Ks(self, scattering_dict: dict) -> None:
        if not self.config.is_tetraquark_analysis:
            return

        new_figure(FIG_STANDARD)
        ref_ns = self.config.scattering_list[0][0]
        plot_gvar_band(scattering_dict["s_array"][ref_ns], scattering_dict["fit_Ks_curve"])
        x_limits = []
        y_limits = []

        for ensemble_idx, ensemble_key in enumerate(self.config.scattering_list):
            ns = ensemble_key[0]
            s_pts, ks_pts = scattering_dict["s"][ns], scattering_dict["Ks"][ns]
            x_limits.append(self._axis_limits(s_pts[:, 0], s_pts[:, 1]))
            y_limits.append(self._axis_limits(ks_pts[:, 0], ks_pts[:, 1]))
            for point_slice, label, color in self._frame_slices(
                scattering_dict, ns, len(s_pts), ensemble_idx
            ):
                if len(s_pts[point_slice]) == 0:
                    continue
                plt.errorbar(
                    s_pts[point_slice, 0],
                    ks_pts[point_slice, 0],
                    xerr=s_pts[point_slice, 1],
                    yerr=ks_pts[point_slice, 1],
                    fmt="x",
                    color=color,
                    label=label,
                    **ERRORBAR_KW,
                )

        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.xlim(*self._combine_axis_limits(*x_limits))
        plt.ylim(*self._combine_axis_limits(*y_limits))
        label_axes(r"$s\,(\rm{GeV}^2)$", r"$K(s)=\sqrt s / (k\cot\delta_0)$")
        add_legend("upper left")
        self._save("K_s")

    def plot_kcot(self, scattering_dict: dict) -> None:
        if not self.config.is_tetraquark_analysis:
            return

        new_figure(FIG_STANDARD)
        ref_ns = self.config.scattering_list[0][0]
        plot_gvar_band(scattering_dict["k_sq_array"][ref_ns], scattering_dict["fit_kcot_curve"])
        x_limits = []
        y_limits = []

        for ensemble_idx, ensemble_key in enumerate(self.config.scattering_list):
            ns = ensemble_key[0]
            k_sq_grid = scattering_dict["k_sq_array"][ns]
            kcot_rest_grid = scattering_dict["kcot_rest_array"][ns]
            k_sq_mean = scattering_dict["k_sq"][ns][:, 0]
            k_sq_err = scattering_dict["k_sq"][ns][:, 1]
            kcot = scattering_dict["kcot"][ns]
            x_limits.append(self._axis_limits(k_sq_mean, k_sq_err))
            y_limits.append(self._axis_limits(kcot[:, 0], kcot[:, 1]))
            frame_slices = self._frame_slices(scattering_dict, ns, len(k_sq_mean), ensemble_idx)
            rest_slice, _, rest_color = frame_slices[0]

            plt.plot(
                k_sq_grid, kcot_rest_grid,
                color=rest_color, linestyle="--", alpha=0.3, zorder=ZORDER_AUX_LINE,
            )
            for m, e in zip(k_sq_mean[rest_slice], k_sq_err[rest_slice]):
                mask = (k_sq_grid >= m - e) & (k_sq_grid <= m + e)
                plt.plot(
                    k_sq_grid[mask], kcot_rest_grid[mask],
                    color=rest_color, linestyle="-", zorder=ZORDER_FIT_CURVE,
                )

            for point_slice, label, color in frame_slices:
                if len(k_sq_mean[point_slice]) == 0:
                    continue
                plt.errorbar(
                    k_sq_mean[point_slice],
                    kcot[point_slice, 0],
                    xerr=k_sq_err[point_slice],
                    fmt="x",
                    color=color,
                    label=label,
                    **ERRORBAR_KW,
                )

        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.xlim(*self._combine_axis_limits(*x_limits))
        plt.ylim(*self._combine_axis_limits(*y_limits))
        label_axes(r"$k^2\,(\rm{GeV}^2)$", r"$k\cot\delta_0\,$(GeV)")
        add_legend("upper left")
        self._save("kcot")
