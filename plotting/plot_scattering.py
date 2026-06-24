import matplotlib.pyplot as plt

from input.config import Config
from plotting.plot_set import (
    BasePlotter,
    COLORS,
    FIT_CURVE_COLOR,
    FIG_STANDARD,
    add_legend,
    axis_limits_from_values,
    combine_axis_limits,
    draw_origin_axes,
    ik_parabola_axis_limits,
    label_axes,
    new_figure,
    palette_color_at,
    plot_errorbars,
    plot_gvar_band,
    plot_ik_parabola,
    plot_kcot_zeta_reference,
)


class ScatteringPlotter(BasePlotter):

    def _save_scattering(self, name: str) -> None:
        self._save(f"result/{self.input_name}/{name}_scattering")

    def _frame_slices(self, scattering_dict: dict, ns: int, n_points: int, ensemble_idx: int):
        if not self.config.is_moving_frame:
            return [(slice(0, n_points), f"L{ns}", COLORS[ensemble_idx % len(COLORS)])]

        rest_count = scattering_dict["rest_point_count"][ns]
        color_idx = 2 * ensemble_idx
        return [
            (slice(0, rest_count), f"L{ns} rest", COLORS[color_idx % len(COLORS)]),
            (slice(rest_count, n_points), f"L{ns} moving", COLORS[(color_idx + 1) % len(COLORS)]),
        ]

    def _n_frame_colors(self) -> int:
        n_ensembles = len(self.config.scattering_list)
        return 2 * n_ensembles if self.config.is_moving_frame else n_ensembles

    def _plot_frame_errorbars(
        self,
        x_mean,
        x_err,
        y_mean,
        y_err,
        frame_slices,
    ) -> None:
        for point_slice, label, color in frame_slices:
            if len(x_mean[point_slice]) == 0:
                continue
            plot_errorbars(
                x_mean[point_slice],
                y_mean[point_slice],
                xerr=x_err[point_slice],
                yerr=y_err[point_slice],
                color=color,
                label=label,
            )

    def _plot_kcot_references(
        self,
        scattering_dict: dict,
        ns: int,
        frame_slices,
        k_sq_mean,
        k_sq_err,
    ) -> None:
        rest_slice, _, rest_color = frame_slices[0]
        plot_kcot_zeta_reference(
            scattering_dict["k_sq_array"][ns],
            scattering_dict["kcot_rest_array"][ns],
            rest_color,
            k_sq_mean[rest_slice],
            k_sq_err[rest_slice],
        )
        if not self.config.is_moving_frame:
            return
        moving_slice, _, moving_color = frame_slices[1]
        plot_kcot_zeta_reference(
            scattering_dict["k_sq_array_MF"][ns],
            scattering_dict["kcot_array_MF"][ns],
            moving_color,
            k_sq_mean[moving_slice],
            k_sq_err[moving_slice],
        )

    def plot_Ks(self, scattering_dict: dict) -> None:
        if not self.config.is_tetraquark_analysis:
            return

        new_figure(FIG_STANDARD)
        ref_ns = self.config.scattering_list[0][0]
        if self.config.scattering_fit_mode == "Ks_linear":
            plot_gvar_band(scattering_dict["s_array"][ref_ns], scattering_dict["fit_Ks_curve"])

        y_limits = []
        for ensemble_idx, ensemble_key in enumerate(self.config.scattering_list):
            ns = ensemble_key[0]
            s_pts, ks_pts = scattering_dict["s"][ns], scattering_dict["Ks"][ns]
            y_limits.append(axis_limits_from_values(ks_pts[:, 0]))
            self._plot_frame_errorbars(
                s_pts[:, 0],
                s_pts[:, 1],
                ks_pts[:, 0],
                ks_pts[:, 1],
                self._frame_slices(scattering_dict, ns, len(s_pts), ensemble_idx),
            )

        draw_origin_axes()
        plt.xlim(*self.config.s_plot_range)
        plt.ylim(*combine_axis_limits(*y_limits))
        label_axes(r"$s\,(\rm{GeV}^2)$", r"$K(s)=\sqrt s / (k\cot\delta_0)$")
        add_legend("upper left")
        self._save_scattering("K_s")

    def plot_kcot(self, scattering_dict: dict) -> None:
        if not self.config.is_tetraquark_analysis:
            return

        new_figure(FIG_STANDARD)
        ref_ns = self.config.scattering_list[0][0]
        plot_gvar_band(scattering_dict["k_sq_array"][ref_ns], scattering_dict["fit_kcot_curve"])

        y_limits = []
        for ensemble_idx, ensemble_key in enumerate(self.config.scattering_list):
            ns = ensemble_key[0]
            k_sq_mean = scattering_dict["k_sq"][ns][:, 0]
            k_sq_err = scattering_dict["k_sq"][ns][:, 1]
            kcot = scattering_dict["kcot"][ns]
            frame_slices = self._frame_slices(scattering_dict, ns, len(k_sq_mean), ensemble_idx)

            y_limits.append(axis_limits_from_values(kcot[:, 0]))
            self._plot_kcot_references(scattering_dict, ns, frame_slices, k_sq_mean, k_sq_err)
            self._plot_frame_errorbars(
                k_sq_mean, k_sq_err, kcot[:, 0], kcot[:, 1], frame_slices
            )

        draw_origin_axes()
        ik_limits = ik_parabola_axis_limits(self.config.k_sq_plot_range[0])
        if ik_limits is not None:
            y_limits.append(ik_limits)
        plt.xlim(*self.config.k_sq_plot_range)
        plt.ylim(*combine_axis_limits(*y_limits))
        plot_ik_parabola(
            self.config.k_sq_plot_range,
            palette_color_at(self._n_frame_colors(), skip=(FIT_CURVE_COLOR,)),
        )
        label_axes(r"$k^2\,(\rm{GeV}^2)$", r"$k\cot\delta_0\,$(GeV)")
        add_legend("upper left")
        self._save_scattering("kcot")
