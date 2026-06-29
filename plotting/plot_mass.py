from typing import Any

import gvar as gv
import lsqfit as lsf
import matplotlib.pyplot as plt
import numpy as np

from analysis.models import MathModels
from analysis.fit_mass import disp_fit_lookup, en_fit_lookup
from data.correlators import AnalysisCorrelators
from input.config import Config, SelectorType, FitResultList
from plotting.plot_ratio import RatioPlotter
from plotting.plot_tmin import TminPlotter
from plotting.plot_set import (
    BasePlotter,
    COLORS,
    ERRORBAR_KW,
    FIG_STANDARD,
    FIG_WIDE,
    LINESTYLES,
    MARKERS,
    ZORDER_FIT_CURVE,
    add_legend,
    axis_limits_from_values,
    chan_mom_label,
    chan_mom_latex,
    chan_mom_latex_n2,
    fill_error_band,
    label_axes,
    new_figure,
)
from statistics.resample import jackknife_map_gvar


class MassPlotter(BasePlotter):

    def __init__(self, config: Config):
        super().__init__(config)
        self.corr_type = "tetraquark" if config.run_tetraquark_analysis else "meson"

    def _save_mass(self, name: str) -> None:
        self._save(f"result/{self.input_name}/{name}_{self.corr_type}_{self.tag_name}")

    def _cosh_with_error(self, corr_1d: np.ndarray):
        return jackknife_map_gvar(
            self.config,
            corr_1d,
            lambda jack: MathModels.generate_cosh_from_data(jack, time_axis=0),
        )

    def _format_meff_gev(self, meff) -> str:
        return f"{meff * self.at_invs} GeV"

    def _print_en_fit_summary(self, en_lookup: dict[tuple[int, int], Any]) -> None:
        print(f"En fits on {self.tag_name} ({self.config.n_state}-state):")
        for chan_idx, mom_list in enumerate(self.chan_momentum_list):
            for mom in mom_list:
                mass_fit = en_lookup[(chan_idx, mom)]
                meff_keys = sorted(
                    (k for k in mass_fit.p if k.startswith("meff_")),
                    key=lambda k: int(k.split("_", 1)[1]),
                )
                levels = ", ".join(
                    f"E{i}={self._format_meff_gev(mass_fit.p[k])}"
                    for i, k in enumerate(meff_keys)
                )
                print(f"  {chan_mom_label(self.chan_name_list[chan_idx], mom)}: {levels}")

    def plot_En(self, corr: AnalysisCorrelators, en_fit_list: FitResultList) -> None:
        selector = SelectorType(self.config, corr)
        data = selector.get_data()
        model_fn, _ = selector.get_model()
        en_lookup = en_fit_lookup(en_fit_list)
        self._print_en_fit_summary(en_lookup)

        ylim_min = 1.0 if self.corr_type == "tetraquark" else 0.1
        en_by_chan: dict[int, list] = {}
        new_figure(FIG_WIDE)

        for chan_idx, mom_list in enumerate(self.chan_momentum_list):
            en_by_chan[chan_idx] = []
            for mom in mom_list:
                mass_fit = en_lookup[(chan_idx, mom)]
                t_min = self.config.t_min[chan_idx][mom]
                t_max = self.config.t_max[chan_idx][mom]
                t_fit = np.arange(t_min, t_max)

                cosh_fit = MathModels.generate_cosh_from_data(
                    model_fn(t_fit, mass_fit.p, self.lattice_Nt), time_axis=0
                )
                m = gv.mean(cosh_fit) * self.at_invs
                s = gv.sdev(cosh_fit) * self.at_invs
                fill_error_band(t_fit, m - s, m + s, COLORS[mom])
                plt.plot(
                    t_fit, m,
                    linestyle=LINESTYLES[chan_idx], color=COLORS[mom], zorder=ZORDER_FIT_CURVE,
                )
                en_by_chan[chan_idx].append(mass_fit.p["meff_0"] * self.at_invs)

        for chan_idx, mom_list in enumerate(self.chan_momentum_list):
            for mom in mom_list:
                mean, err = self._cosh_with_error(data.at(chan_idx, mom))
                plt.errorbar(
                    np.arange(self.lattice_Nt),
                    mean * self.at_invs,
                    err * self.at_invs,
                    fmt=MARKERS[chan_idx],
                    color=COLORS[mom],
                    label=chan_mom_latex(self.chan_name_list[chan_idx], mom),
                    **ERRORBAR_KW,
                )

        all_en = np.concatenate([np.array(vals) for vals in en_by_chan.values()])
        label_axes(r"$t/a_t$", r"$E_n$ (GeV)", title=self._figure_title())
        plt.xlim(1, self.lattice_Nt - 1)
        plt.ylim(gv.mean(np.min(all_en)) - ylim_min, gv.mean(np.max(all_en)) + 0.2)
        add_legend("lower right", ncol=2)
        self._save_mass("En")

    def plot_Zn(self, en_fit_list: FitResultList) -> None:
        if not self.config.run_weight_analysis:
            return

        en_lookup = en_fit_lookup(en_fit_list)
        mom_max = max(max(moms) for moms in self.chan_momentum_list if moms)
        zn_values = []
        new_figure(FIG_STANDARD)

        for chan_idx, mom_list in enumerate(self.chan_momentum_list):
            ref_weff = en_lookup[(chan_idx, mom_list[0])].p["weff_0"]
            zn_list = [
                en_lookup[(chan_idx, mom)].p["weff_0"] / ref_weff for mom in mom_list
            ]

            zn_fit = lsf.nonlinear_fit(
                data=(np.array(mom_list), np.array(zn_list)),
                fcn=MathModels.exponential,
                prior=MathModels.prior_exponential(),
            )
            x = np.linspace(0, mom_max, 100)
            y_fit = MathModels.exponential(x, zn_fit.p)
            m = gv.mean(y_fit)
            s = gv.sdev(y_fit)
            color = COLORS[chan_idx]
            fill_error_band(x, m - s, m + s, color)
            plt.plot(x, m, linestyle=LINESTYLES[chan_idx], color=color, zorder=ZORDER_FIT_CURVE)

        for chan_idx, mom_list in enumerate(self.chan_momentum_list):
            ref_weff = en_lookup[(chan_idx, mom_list[0])].p["weff_0"]
            for mom in mom_list:
                zn = en_lookup[(chan_idx, mom)].p["weff_0"] / ref_weff
                zn_values.append(gv.mean(zn))
                plt.errorbar(
                    mom, gv.mean(zn), gv.sdev(zn),
                    fmt=MARKERS[chan_idx], color=COLORS[chan_idx],
                    label=chan_mom_latex_n2(self.chan_name_list[chan_idx]) if mom == mom_list[0] else None,
                    **ERRORBAR_KW,
                )

        label_axes(r"$n^2$", r"$Z_n/Z_0$", title=self._figure_title())
        plt.xlim(0, mom_max)
        plt.ylim(*axis_limits_from_values(zn_values))
        add_legend("upper right")
        self._save_mass("Zn")

    def plot_dispersion(
        self, en_fit_list: FitResultList, disp_fit_list: FitResultList
    ) -> None:
        if self.corr_type != "meson":
            print("Plot dispersion relation only from meson En.")
            return

        en_lookup = en_fit_lookup(en_fit_list)
        disp_lookup = disp_fit_lookup(disp_fit_list)
        mom_max = max(max(moms) for moms in self.chan_momentum_list if moms)
        en_sq_by_chan: dict[int, list] = {}

        new_figure(FIG_STANDARD)
        for chan_idx, mom_list in enumerate(self.chan_momentum_list):
            en_sq_by_chan[chan_idx] = [
                (en_lookup[(chan_idx, mom)].p["meff_0"] * self.at_invs) ** 2
                for mom in mom_list
            ]
            x = np.linspace(0, mom_max, 100)
            y_fit = MathModels.linear(x, disp_lookup[chan_idx].p)
            m = gv.mean(y_fit)
            s = gv.sdev(y_fit)
            color = COLORS[chan_idx]
            fill_error_band(x, m - s, m + s, color)
            plt.plot(x, m, linestyle=LINESTYLES[chan_idx], color=color, zorder=ZORDER_FIT_CURVE)

        for chan_idx, mom_list in enumerate(self.chan_momentum_list):
            y, yerr = gv.mean(en_sq_by_chan[chan_idx]), gv.sdev(en_sq_by_chan[chan_idx])
            for idx, mom in enumerate(mom_list):
                plt.errorbar(
                    mom, y[idx], yerr[idx],
                    fmt=MARKERS[chan_idx], color=COLORS[chan_idx],
                    label=chan_mom_latex_n2(self.chan_name_list[chan_idx]) if mom == mom_list[0] else None,
                    **ERRORBAR_KW,
                )

        label_axes(r"$n^2$", r"$E_n^2$ (GeV$^2$)", title=self._figure_title())
        plt.xlim(0, mom_max)
        all_vals = np.concatenate([gv.mean(v) for v in en_sq_by_chan.values()])
        plt.ylim(*axis_limits_from_values(all_vals))
        add_legend("upper left")
        self._save_mass("Dispersion")

    def plot_ratio_workflow(
        self,
        corr: AnalysisCorrelators,
        meson_config: Config | None,
    ) -> None:
        if meson_config is None:
            return
        RatioPlotter(self.config).run(corr, meson_config)

    def plot_tmin_workflow(
        self,
        corr: AnalysisCorrelators,
        meson_config: Config | None = None,
    ) -> None:
        TminPlotter(self.config).run(corr, meson_config)
