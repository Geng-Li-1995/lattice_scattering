"""t_min scan plots: 3-state cosh, 4Q/2Q ratio, and combine reference."""

from __future__ import annotations

import gvar as gv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from analysis.fit_tmin import (
    RatioScanResult,
    RatioScanTarget,
    TminScanResult,
    ratio_scan_lookup,
    scan_cosh_tmin,
    scan_ratio_tmin,
    t_run_list,
)
from data.correlators import AnalysisCorrelators
from input.config import Config
from plotting.plot_set import (
    BasePlotter,
    COLORS,
    ERRORBAR_KW,
    FIG_STANDARD,
    FS_LABEL,
    FS_LEGEND,
    ZORDER_AUX_LINE,
    ZORDER_DATA,
    ZORDER_ERROR_BAND,
)

TMIN_X_LO_MIN = 9
TMIN_YLIM_ERR_SCALE = 10.0
CHI2_YLIM = (0.0, 2.0)
TMIN_PANEL_RATIOS = (3, 1)


class TminPlotter(BasePlotter):
    """t_min scan plots: cosh-only or 3-state + ratio."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.tag_name = config.tag_name
        self.at_invs = config.at_invs
        self.corr_type = "tetraquark" if config.is_tetraquark_analysis else "meson"

    def run(
        self,
        corr: AnalysisCorrelators,
        meson_config: Config | None = None,
    ) -> None:
        if not self.config.plot_tmin or not self.config.run_tmin:
            return

        self._print_scan_header()
        ratio_lookup = self._ratio_lookup(corr, meson_config)

        data = corr.active(
            self.config.is_meson_analysis,
            self.config.is_tetraquark_analysis,
        )
        for ch_idx, mom_list in enumerate(self.config.chan_momt_list):
            for mom in mom_list:
                slice_1d = data.at(ch_idx, mom)
                target = ratio_lookup.get((ch_idx, mom))
                if target is not None:
                    self._scan_and_plot_ratio(
                        corr, meson_config, ch_idx, mom, slice_1d, target
                    )
                else:
                    self._scan_and_plot_cosh(ch_idx, mom, slice_1d)

    def _ratio_lookup(
        self,
        corr: AnalysisCorrelators,
        meson_config: Config | None,
    ) -> dict[tuple[int, int], RatioScanTarget]:
        if not self._ratio_enabled(corr, meson_config):
            if self.config.is_ratio:
                print(
                    "WARNING: is_ratio=True but ratio tmin skipped — "
                    "need meson correlator loaded and ratio_scan_points set."
                )
            return {}
        return ratio_scan_lookup(self.config)

    def _ratio_enabled(self, corr: AnalysisCorrelators, meson_config: Config | None) -> bool:
        return (
            self.config.is_ratio
            and meson_config is not None
            and corr.meson is not None
            and corr.tetraquark is not None
            and bool(self.config.ratio_scan_points)
        )

    def _print_scan_header(self) -> None:
        t_mins = t_run_list(self.config)
        n_channels = sum(len(moms) for moms in self.config.chan_momt_list)
        print(
            f"Running t_min scan on {self.tag_name}: "
            f"{n_channels} channel(s), "
            f"t_min = {int(t_mins[0])}..{int(t_mins[-1])} "
            f"(step {self.config.t_run_step}, {len(t_mins)} points) ..."
        )

    def _scan_and_plot_cosh(
        self, ch_idx: int, mom: int, slice_1d: np.ndarray
    ) -> None:
        print(f"  Scanning E{ch_idx + 1} mom={mom}: 3-state cosh ...")
        scan = scan_cosh_tmin(self.config, slice_1d, ch_idx, mom)
        self._plot_cosh_scan(ch_idx, mom, scan)

    def _scan_and_plot_ratio(
        self,
        corr: AnalysisCorrelators,
        meson_config: Config,
        ch_idx: int,
        mom: int,
        slice_1d: np.ndarray,
        target: RatioScanTarget,
    ) -> None:
        print(
            f"  Scanning E{ch_idx + 1} mom={mom}: "
            f"3-state + ratio (meson ch={target.meson_ch} mom={target.meson_mom}) ..."
        )
        meson_1d = corr.meson.at(target.meson_ch, target.meson_mom)
        scan = scan_ratio_tmin(
            self.config,
            slice_1d,
            meson_1d,
            ch_idx,
            mom,
            meson_config,
            target.meson_ch,
            target.meson_mom,
            target.delta_prior,
        )
        self._plot_ratio_scan(ch_idx, mom, target.state_idx, scan, target)

    def _tmin_x_limits(self, t_list: np.ndarray) -> tuple[int, int]:
        return max(int(t_list[0]), TMIN_X_LO_MIN), int(self.config.lattice_Nt // 2 - 1)

    def _make_tmin_figure(
        self, t_list: np.ndarray
    ) -> tuple[plt.Figure, plt.Axes, plt.Axes, int, int]:
        x_lo, x_hi = self._tmin_x_limits(t_list)
        fig = plt.figure(figsize=FIG_STANDARD)
        gs = gridspec.GridSpec(2, 1, height_ratios=list(TMIN_PANEL_RATIOS))
        ax_meff, ax_chi = (fig.add_subplot(gs[i]) for i in range(2))
        for ax in (ax_meff, ax_chi):
            ax.set_xlim(x_lo, x_hi)
        ax_chi.set_ylim(*CHI2_YLIM)
        return fig, ax_meff, ax_chi, x_lo, x_hi

    def _setup_meff_axis(
        self,
        ax: plt.Axes,
        ylabel: str,
        ref_val: float,
        ref_err: float,
    ) -> None:
        ax.set_ylim(*self._y_limits(ref_val, ref_err))
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_xticks([])

    def _plot_cosh_scan(self, ch_idx: int, mom: int, scan: TminScanResult) -> None:
        t_list = scan.t_min_list
        ref_val, ref_err = self._ref_energy_gev(scan)
        _, ax_meff, ax_chi, _, _ = self._make_tmin_figure(t_list)

        self._setup_meff_axis(ax_meff, rf"$E$ (GeV) on {self.tag_name}", ref_val, ref_err)
        ax_meff.errorbar(
            t_list,
            self._energy_gev_mean(scan.meff),
            self._energy_gev_sdev(scan.meff),
            fmt="o",
            color=COLORS[1],
            label="Jackknife error",
            **ERRORBAR_KW,
        )

        self._plot_chi2_panel(
            ax_chi,
            t_list,
            [(scan.chi2_dof, COLORS[1], "D")],
            xlabel=rf"Fit $E_{{{ch_idx + 1}}}$ with $t_{{min}}/a_t$",
        )
        plt.tight_layout()
        self._save_tmin(f"E{ch_idx + 1}_mom{mom}_tmin")

    def _plot_ratio_scan(
        self,
        ch_idx: int,
        mom: int,
        state_idx: int,
        scan: RatioScanResult,
        target: RatioScanTarget,
    ) -> None:
        cosh = scan.cosh_scan
        t_list = cosh.t_min_list
        ref_val, ref_err = self._ref_energy_gev(cosh)
        _, ax_meff, ax_chi, _, _ = self._make_tmin_figure(t_list)

        self._setup_meff_axis(
            ax_meff,
            rf"$E_{{{state_idx + 1}}}$ (GeV) on {self.tag_name}",
            ref_val,
            ref_err,
        )
        ax_meff.errorbar(
            t_list,
            self._energy_gev_mean(cosh.meff),
            self._energy_gev_sdev(cosh.meff),
            fmt="o",
            color=COLORS[0],
            label="3-state",
            **ERRORBAR_KW,
        )
        ax_meff.errorbar(
            t_list + 0.1,
            self._energy_gev_mean(scan.delta_m),
            self._energy_gev_sdev(scan.delta_m),
            fmt="x",
            color=COLORS[1],
            label="Ratio",
            **ERRORBAR_KW,
        )
        ax_meff.plot(
            t_list,
            np.full(len(t_list), ref_val),
            color="green",
            label="Combine",
            zorder=ZORDER_AUX_LINE,
        )
        ax_meff.fill_between(
            t_list,
            ref_val - ref_err,
            ref_val + ref_err,
            color="lightgreen",
            zorder=ZORDER_ERROR_BAND,
        )
        ax_meff.legend(
            loc="lower center",
            ncol=3,
            fontsize=FS_LEGEND,
            framealpha=0.95,
        )
        print(
            f"  Done E{ch_idx + 1} mom={mom} "
            f"(meson ch={target.meson_ch} mom={target.meson_mom}): "
            f"3-state={ref_val:.5f} GeV, "
            f"ratio={self._energy_gev_scalar(scan.delta_m[cosh.ref_index]):.5f} GeV "
            f"(t_min={cosh.ref_t_min})"
        )

        self._plot_chi2_panel(
            ax_chi,
            t_list,
            [
                (cosh.chi2_dof, COLORS[0], "o"),
                (scan.chi2_dof, COLORS[1], "x"),
            ],
            xlabel=r"Fit $t_{min}/a_t$",
        )
        plt.tight_layout()
        self._save_tmin(f"E{ch_idx + 1}_mom{mom}_tmin_ratio")

    def _plot_chi2_panel(
        self,
        ax: plt.Axes,
        t_list: np.ndarray,
        series: list[tuple[np.ndarray, str, str]],
        *,
        xlabel: str,
    ) -> None:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
        ax.set_ylabel(r"$\chi^2/$d.o.f.", fontsize=FS_LABEL)
        for chi2, color, marker in series:
            ax.plot(t_list, chi2, **self._chi2_marker_kw(color, marker))

    def _chi2_marker_kw(self, color: str, marker: str) -> dict:
        return {
            "linestyle": "none",
            "marker": marker,
            "color": color,
            "markersize": ERRORBAR_KW["markersize"],
            "markeredgecolor": color,
            "markerfacecolor": ERRORBAR_KW["markerfacecolor"],
            "zorder": ZORDER_DATA,
        }

    def _ref_energy_gev(self, scan: TminScanResult) -> tuple[float, float]:
        ref = scan.meff[scan.ref_index]
        return self._energy_gev_scalar(ref), self._energy_gev_sdev_scalar(ref)

    def _y_limits(
        self, center: float, err: float, *, scale: float = TMIN_YLIM_ERR_SCALE
    ) -> tuple[float, float]:
        half = max(abs(err) * scale, 1e-12)
        return center - half, center + half

    def _energy_gev_mean(self, values) -> np.ndarray:
        return np.asarray(gv.mean(values), dtype=float) * self.at_invs

    def _energy_gev_sdev(self, values) -> np.ndarray:
        return np.asarray(gv.sdev(values), dtype=float) * self.at_invs

    def _energy_gev_scalar(self, value) -> float:
        return float(gv.mean(value)) * self.at_invs

    def _energy_gev_sdev_scalar(self, value) -> float:
        return float(gv.sdev(value)) * self.at_invs

    def _save_tmin(self, name: str) -> None:
        self._save(f"result/{self.input_name}/{name}_{self.tag_name}")
