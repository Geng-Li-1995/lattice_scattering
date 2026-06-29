"""t_min scan plots: n-state cosh + combine; optional ratio overlay for tetraquark."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from analysis.fit_tmin import (
    RatioScanResult,
    RatioScanTarget,
    TminScanResult,
    meson_slices,
    ratio_scan_lookup,
    scan_cosh_tmin,
    scan_ratio_tmin,
    t_run_list,
)
from data.correlators import AnalysisCorrelators
from input.config import Config, ratio_chan_labels
from plotting.plot_set import (
    BasePlotter,
    COLORS,
    ERRORBAR_KW,
    FIG_STANDARD,
    FS_LABEL,
    FS_LEGEND,
    chi2_marker_kw,
    log_channel_tag,
    meson_energy_subscript,
    plot_combine_reference,
    set_figure_title,
    y_limits_from_error,
)

TMIN_X_LO_MIN = 9
TMIN_YLIM_ERR_SCALE = 10.0
CHI2_YLIM = (0.0, 2.0)
TMIN_PANEL_RATIOS = (3, 1)
TMIN_XLABEL = r"Fit $t_{min}/a_t$"


def _cosh_state_label(n_state: int) -> str:
    return f"{n_state}-state"


@dataclass(frozen=True)
class _TminScanBundle:
    cosh: TminScanResult
    ratio: RatioScanResult | None = None
    target: RatioScanTarget | None = None


class TminPlotter(BasePlotter):
    """t_min scan: 2-state (meson) or 3-state (tetraquark) cosh + combine; ratio optional."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.corr_type = "tetraquark" if config.run_tetraquark_analysis else "meson"

    def run(
        self,
        corr: AnalysisCorrelators,
        meson_config: Config | None = None,
    ) -> None:
        if not self.config.run_tmin_analysis:
            return

        self._print_scan_header()
        ratio_lookup = self._ratio_lookup(corr, meson_config)
        data = corr.active(
            self.config.run_meson_analysis,
            self.config.run_tetraquark_analysis,
        )
        for chan_idx, mom_list in enumerate(self.chan_momentum_list):
            for mom in mom_list:
                self._scan_and_plot(
                    corr,
                    meson_config,
                    chan_idx,
                    mom,
                    data.at(chan_idx, mom),
                    ratio_lookup.get((chan_idx, mom)),
                )

    def _ratio_lookup(
        self,
        corr: AnalysisCorrelators,
        meson_config: Config | None,
    ) -> dict[tuple[int, int], RatioScanTarget]:
        if not self._ratio_enabled(corr, meson_config):
            if self.config.run_ratio_analysis and self.config.run_tetraquark_analysis:
                print(
                    "WARNING: run_ratio_analysis=True but ratio tmin skipped — "
                    "need meson correlator loaded and tetraquark chan_name_list."
                )
            return {}
        return ratio_scan_lookup(self.config)

    def _ratio_enabled(self, corr: AnalysisCorrelators, meson_config: Config | None) -> bool:
        return (
            self.config.run_tetraquark_analysis
            and self.config.run_ratio_analysis
            and meson_config is not None
            and corr.meson is not None
            and corr.tetraquark is not None
            and bool(ratio_chan_labels(self.config))
        )

    def _print_scan_header(self) -> None:
        t_mins = t_run_list(self.config)
        n_chans = sum(len(moms) for moms in self.chan_momentum_list)
        print(
            f"Running t_min scan on {self.tag_name}: "
            f"{n_chans} chan(s), "
            f"t_min = {int(t_mins[0])}..{int(t_mins[-1])} "
            f"(step {self.config.t_run_step}, {len(t_mins)} points) ..."
        )

    def _scan_and_plot(
        self,
        corr: AnalysisCorrelators,
        meson_config: Config | None,
        chan_idx: int,
        mom: int,
        slice_1d: np.ndarray,
        target: RatioScanTarget | None,
    ) -> None:
        energy_tag = log_channel_tag(chan_idx, mom)
        state_label = _cosh_state_label(self.config.n_state)

        if target is not None:
            assert meson_config is not None
            print(
                f"  Scanning {energy_tag}: {state_label} + ratio "
                f"(meson a chan={target.meson_chan} mom={target.meson_mom}"
                f"{'' if target.is_ratio_shift else f', b chan={target.meson_b_chan} mom={target.meson_b_mom}'}) ..."
            )
            meson_a_1d, meson_b_1d = meson_slices(corr, target)
            ratio_scan = scan_ratio_tmin(
                self.config,
                slice_1d,
                meson_a_1d,
                meson_b_1d,
                meson_config,
                target,
            )
            bundle = _TminScanBundle(ratio_scan.cosh_scan, ratio_scan, target)
        else:
            print(f"  Scanning {energy_tag}: {state_label} cosh ...")
            bundle = _TminScanBundle(scan_cosh_tmin(self.config, slice_1d, chan_idx, mom))

        self._plot_scan(chan_idx, mom, bundle)

    def _meff_ylabel(self, chan_idx: int, mom: int, target: RatioScanTarget | None) -> str:
        if self.corr_type == "meson":
            e_sub = meson_energy_subscript(self.chan_name_list[chan_idx], mom)
            return rf"${e_sub}$ (GeV)"
        if target is not None:
            return rf"$E_{{{target.state_idx + 1}}}$ (GeV)"
        return r"$E$ (GeV)"

    def _plot_scan(self, chan_idx: int, mom: int, bundle: _TminScanBundle) -> None:
        cosh = bundle.cosh
        ratio = bundle.ratio
        target = bundle.target
        has_ratio = ratio is not None

        t_list = cosh.t_min_list
        ref_val, ref_err = self._ref_energy_gev(cosh)
        _, ax_meff, ax_chi, _, _ = self._make_tmin_figure(t_list)

        self._setup_meff_axis(ax_meff, self._meff_ylabel(chan_idx, mom, target), ref_val, ref_err)

        cosh_color = COLORS[0] if has_ratio else COLORS[1]
        cosh_label = "3-state" if has_ratio else _cosh_state_label(self.config.n_state)
        ax_meff.errorbar(
            t_list,
            self._energy_gev_mean(cosh.meff),
            self._energy_gev_sdev(cosh.meff),
            fmt="o",
            color=cosh_color,
            label=cosh_label,
            **ERRORBAR_KW,
        )
        if has_ratio:
            ax_meff.errorbar(
                t_list + 0.1,
                self._energy_gev_mean(ratio.delta_m),
                self._energy_gev_sdev(ratio.delta_m),
                fmt="x",
                color=COLORS[1],
                label="Ratio",
                **ERRORBAR_KW,
            )

        plot_combine_reference(ax_meff, t_list, ref_val, ref_err)
        ax_meff.legend(
            loc="lower center",
            ncol=3 if has_ratio else 2,
            fontsize=FS_LEGEND,
            framealpha=0.95,
        )
        self._print_done(chan_idx, mom, bundle, ref_val, ref_err)

        chi_marker = "o" if has_ratio else "D"
        chi_series: list[tuple[np.ndarray, str, str]] = [
            (cosh.chi2_dof, cosh_color, chi_marker),
        ]
        if has_ratio:
            chi_series.append((ratio.chi2_dof, COLORS[1], "x"))

        self._plot_chi2_panel(ax_chi, t_list, chi_series, xlabel=TMIN_XLABEL)
        set_figure_title(self._figure_title(), ax=ax_meff)
        self._save_tmin(chan_idx, mom)

    def _print_done(
        self,
        chan_idx: int,
        mom: int,
        bundle: _TminScanBundle,
        ref_val: float,
        ref_err: float,
    ) -> None:
        energy_tag = log_channel_tag(chan_idx, mom)
        cosh = bundle.cosh
        if bundle.ratio is None:
            print(
                f"  Done {energy_tag}: "
                f"{self._format_energy_gev(ref_val, ref_err)} (t_min={cosh.ref_t_min})"
            )
            return

        target = bundle.target
        assert target is not None
        print(
            f"  Done {energy_tag} "
            f"(meson a chan={target.meson_chan} mom={target.meson_mom}"
            f"{'' if target.is_ratio_shift else f', b chan={target.meson_b_chan} mom={target.meson_b_mom}'}): "
            f"3-state={ref_val:.5f} GeV, "
            f"ratio={self._energy_gev_scalar(bundle.ratio.delta_m[cosh.ref_index]):.5f} GeV "
            f"(t_min={cosh.ref_t_min})"
        )

    def _tmin_x_limits(self, t_list: np.ndarray) -> tuple[int, int]:
        return max(int(t_list[0]), TMIN_X_LO_MIN), int(self.lattice_Nt // 2 - 1)

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
        ax.set_ylim(*y_limits_from_error(ref_val, ref_err, scale=TMIN_YLIM_ERR_SCALE))
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_xticks([])

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
            ax.plot(t_list, chi2, **chi2_marker_kw(color, marker))

    def _ref_energy_gev(self, scan: TminScanResult) -> tuple[float, float]:
        ref = scan.meff[scan.ref_index]
        return self._energy_gev_scalar(ref), self._energy_gev_sdev_scalar(ref)

    def _save_tmin(self, chan_idx: int, mom: int) -> None:
        branch = "meson" if self.corr_type == "meson" else "tetraquark"
        stem = f"En_tmin_{branch}{chan_idx}_mom{mom}"
        self._save(f"result/{self.input_name}/{stem}_{self.tag_name}")
