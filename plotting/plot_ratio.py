"""Ratio correlator fits: R(t) data points and model curve at reference t_min."""

from __future__ import annotations

from dataclasses import dataclass

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np

from analysis.fit_tmin import (
    RatioReferenceFit,
    fit_ratio_reference,
    meson_slices,
    ratio_scan_lookup,
    ratio_series_mean_err,
)
from analysis.models import MathModels
from data.correlators import AnalysisCorrelators
from input.config import Config, ratio_chan_labels
from plotting.plot_set import (
    BasePlotter,
    COLORS,
    ERRORBAR_KW,
    FIG_WIDE,
    LINESTYLES,
    MARKERS,
    ZORDER_FIT_CURVE,
    add_legend,
    axis_limits_from_values,
    chan_mom_latex,
    combine_axis_limits,
    fill_error_band,
    label_axes,
    new_figure,
)


@dataclass
class _RatioChanPlot:
    chan_idx: int
    mom: int
    t_min: int
    t_fit: np.ndarray
    t_all: np.ndarray
    ratio_mean: np.ndarray
    ratio_err: np.ndarray
    finite: np.ndarray
    y_fit_m: np.ndarray
    y_fit_s: np.ndarray
    ref_fit: RatioReferenceFit
    is_ratio_shift: bool


class RatioPlotter(BasePlotter):
    """Plot all ratio chans on one figure (same layout style as ``plot_En``)."""

    def run(self, corr: AnalysisCorrelators, meson_config: Config) -> None:
        if not self._enabled(corr):
            return

        half_Nt = self.lattice_Nt // 2
        chans = [
            chan
            for (chan_idx, mom), target in ratio_scan_lookup(self.config).items()
            if (chan := self._build_chan(corr, meson_config, chan_idx, mom, target, half_Nt))
        ]
        if not chans:
            print("WARNING: no ratio chans with finite R(t) — skip ratio plot.")
            return

        ylim = None
        new_figure(FIG_WIDE)
        for chan in chans:
            color, ls = COLORS[chan.mom], LINESTYLES[chan.chan_idx]
            fill_error_band(chan.t_fit, chan.y_fit_m - chan.y_fit_s, chan.y_fit_m + chan.y_fit_s, color)
            plt.plot(chan.t_fit, chan.y_fit_m, linestyle=ls, color=color, zorder=ZORDER_FIT_CURVE)
            chan_lim = axis_limits_from_values(
                self._chan_y_values(chan, half_Nt), padding_fraction=0.08
            )
            ylim = chan_lim if ylim is None else combine_axis_limits(ylim, chan_lim)

        for chan in chans:
            plt.errorbar(
                chan.t_all[chan.finite],
                chan.ratio_mean[chan.finite],
                chan.ratio_err[chan.finite],
                fmt=MARKERS[chan.chan_idx],
                color=COLORS[chan.mom],
                label=chan_mom_latex(self.chan_name_list[chan.chan_idx], chan.mom),
                **ERRORBAR_KW,
            )

        label_axes(r"$t/a_t$", self._ylabel(chans), title=self._figure_title())
        plt.xlim(1, half_Nt - 1)
        if ylim is not None:
            plt.ylim(*ylim)
        add_legend("upper left", ncol=2)
        if not self.config.run_resample_analysis:
            for chan in chans:
                print("chan_idx, mom =", chan.chan_idx, chan.mom, "\n", chan.ref_fit.fit)
        self._save(f"result/{self.input_name}/Ratio_{self.tag_name}")

    def _enabled(self, corr: AnalysisCorrelators) -> bool:
        return (
            self.config.run_ratio_analysis
            and corr.meson is not None
            and corr.tetraquark is not None
            and bool(ratio_chan_labels(self.config))
        )

    def _build_chan(
        self,
        corr: AnalysisCorrelators,
        meson_config: Config,
        chan_idx: int,
        mom: int,
        target,
        half_Nt: int,
    ) -> _RatioChanPlot | None:
        tetra_1d = corr.tetraquark.at(chan_idx, mom)
        meson_a_1d, meson_b_1d = meson_slices(corr, target)
        ref_fit = fit_ratio_reference(
            self.config, tetra_1d, meson_a_1d, meson_b_1d, target, meson_config
        )
        t_min = self.config.t_min[chan_idx][mom]
        t_all = np.arange(half_Nt)
        t_fit = np.arange(t_min, half_Nt)
        ratio_mean, ratio_err = ratio_series_mean_err(
            self.config,
            tetra_1d,
            meson_a_1d,
            meson_b_1d,
            is_ratio_shift=target.is_ratio_shift,
        )
        finite = np.isfinite(ratio_mean) & np.isfinite(ratio_err)
        if not np.any(finite):
            print(
                f"WARNING: skip ratio plot "
                f"{self.chan_name_list[chan_idx]} n^2={mom} — no finite R(t)."
            )
            return None

        model_y = self._model_curve(ref_fit, t_fit)
        return _RatioChanPlot(
            chan_idx=chan_idx,
            mom=mom,
            t_min=t_min,
            t_fit=t_fit,
            t_all=t_all,
            ratio_mean=ratio_mean,
            ratio_err=ratio_err,
            finite=finite,
            y_fit_m=np.asarray(gv.mean(model_y), dtype=float),
            y_fit_s=np.asarray(gv.sdev(model_y), dtype=float),
            ref_fit=ref_fit,
            is_ratio_shift=target.is_ratio_shift,
        )

    def _model_curve(self, ref_fit: RatioReferenceFit, t_fit: np.ndarray):
        if ref_fit.target.is_ratio_shift:
            return MathModels.ratio(
                t_fit, ref_fit.fit.p, self.lattice_Nt, ref_fit.meson_m
            )
        return MathModels.ratio_exponential(t_fit, ref_fit.fit.p, self.lattice_Nt)

    @staticmethod
    def _chan_y_values(chan: _RatioChanPlot, half_Nt: int) -> np.ndarray:
        """Values for auto y-limits: fit window only (exclude early-time spikes)."""
        t = chan.t_all[chan.finite]
        m = chan.ratio_mean[chan.finite]
        e = chan.ratio_err[chan.finite]
        in_window = (t >= chan.t_min) & (t < half_Nt)
        chunks: list[np.ndarray] = []
        if np.any(in_window):
            chunks.extend([m[in_window] - e[in_window], m[in_window] + e[in_window]])
        fit_ok = np.isfinite(chan.y_fit_m) & np.isfinite(chan.y_fit_s)
        if np.any(fit_ok):
            chunks.extend(
                [
                    chan.y_fit_m[fit_ok] - chan.y_fit_s[fit_ok],
                    chan.y_fit_m[fit_ok] + chan.y_fit_s[fit_ok],
                ]
            )
        if chunks:
            return np.concatenate(chunks)
        # t_min ≥ half_Nt: no in-window data — use fit curve if present
        if np.any(fit_ok):
            return np.concatenate(
                [
                    chan.y_fit_m[fit_ok] - chan.y_fit_s[fit_ok],
                    chan.y_fit_m[fit_ok] + chan.y_fit_s[fit_ok],
                ]
            )
        return np.array([], dtype=float)

    @staticmethod
    def _ylabel(chans: list[_RatioChanPlot]) -> str:
        if all(chan.is_ratio_shift for chan in chans):
            return r"$R_n(t/a_t)$"
        if all(not chan.is_ratio_shift for chan in chans):
            return r"$R(t/a_t)$"
        return r"$R(t/a_t)$"
