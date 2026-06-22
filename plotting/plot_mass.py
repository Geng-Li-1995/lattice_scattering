import numpy as np
import gvar as gv
import lsqfit as lsf
import matplotlib.pyplot as plt

from analysis.models import MathModels
from analysis.utils import disp_fit_lookup, en_fit_lookup
from data.correlators import AnalysisCorrelators
from input.config import Config
from input.selector import SelectorType
from input.types import FitResultList
from plotting.plot_set import (
    COLORS,
    ERRORBAR_KW,
    FIG_STANDARD,
    FIG_WIDE,
    LINESTYLES,
    MARKERS,
    ZORDER_FIT_CURVE,
    add_legend,
    apply_plot_style,
    fill_error_band,
    label_axes,
    new_figure,
    save_figure,
)
from statistics.resample import get_resampler


class MassPlotter:

    def __init__(self, config: Config):
        apply_plot_style()
        self.config = config
        self.input_name = config.input_name
        self.tag_name = config.tag_name
        self.at_invs = config.at_invs
        self.lattice_Nt = config.lattice_Nt
        self.chan_momt_list = config.chan_momt_list
        self.chan_name_list = config.chan_name_list
        self.corr_type = "tetraquark" if config.is_tetraquark_analysis else "meson"

    def _cosh_with_error(self, corr_1d: np.ndarray):
        jack_samples = get_resampler(self.config, corr_1d).resample()
        cosh_samples = MathModels.generate_cosh_from_data(jack_samples, time_axis=0)
        return get_resampler(self.config, cosh_samples).gvar()

    def _save(self, name: str) -> None:
        save_figure(
            f"result/{self.input_name}/{name}_{self.corr_type}_{self.tag_name}",
            plot_format=self.config.plot_format,
        )

    def plot_En(self, corr: AnalysisCorrelators, en_fit_list: FitResultList) -> None:
        selector = SelectorType(self.config, corr)
        data = selector.get_data()
        model_fn, _ = selector.get_model()
        en_lookup = en_fit_lookup(en_fit_list)

        ylim_min = 1.0 if self.corr_type == "tetraquark" else 0.1
        en_by_ch: dict[int, list] = {}
        new_figure(FIG_WIDE)

        for ch_idx, mom_list in enumerate(self.chan_momt_list):
            en_by_ch[ch_idx] = []
            for mom in mom_list:
                mass_fit = en_lookup[(ch_idx, mom)]
                t_min = self.config.t_min[ch_idx][mom]
                t_max = self.config.t_max[ch_idx][mom]
                t_fit = np.arange(t_min, t_max)

                cosh_fit = MathModels.generate_cosh_from_data(
                    model_fn(t_fit, mass_fit.p, self.lattice_Nt), time_axis=0
                )
                m = gv.mean(cosh_fit) * self.at_invs
                s = gv.sdev(cosh_fit) * self.at_invs
                fill_error_band(t_fit, m - s, m + s, COLORS[mom])
                plt.plot(
                    t_fit, m,
                    linestyle=LINESTYLES[ch_idx], color=COLORS[mom], zorder=ZORDER_FIT_CURVE,
                )
                en_by_ch[ch_idx].append(mass_fit.p["meff_0"] * self.at_invs)

        for ch_idx, mom_list in enumerate(self.chan_momt_list):
            for mom in mom_list:
                mean, err = self._cosh_with_error(data.at(ch_idx, mom))
                plt.errorbar(
                    np.arange(self.lattice_Nt),
                    mean * self.at_invs,
                    err * self.at_invs,
                    fmt=MARKERS[ch_idx],
                    color=COLORS[mom],
                    label=rf"${self.chan_name_list[ch_idx]}(n^2={mom})$",
                    **ERRORBAR_KW,
                )

        all_en = np.concatenate([np.array(vals) for vals in en_by_ch.values()])
        label_axes(r"$t/a_t$", rf"$E_n$ (GeV) on {self.tag_name}")
        plt.xlim(1, self.lattice_Nt - 1)
        plt.ylim(gv.mean(np.min(all_en)) - ylim_min, gv.mean(np.max(all_en)) + 0.2)
        add_legend("lower right", ncol=2)
        self._save("En")

    def plot_Zn(self, en_fit_list: FitResultList) -> None:
        if not self.config.is_meson_analysis:
            return

        en_lookup = en_fit_lookup(en_fit_list)
        mom_max = max(max(moms) for moms in self.chan_momt_list if moms)
        new_figure(FIG_STANDARD)

        for ch_idx, mom_list in enumerate(self.chan_momt_list):
            ref_weff = en_lookup[(ch_idx, mom_list[0])].p["weff_0"]
            zn_list = [
                en_lookup[(ch_idx, mom)].p["weff_0"] / ref_weff for mom in mom_list
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
            color = COLORS[ch_idx]
            fill_error_band(x, m - s, m + s, color)
            plt.plot(x, m, linestyle=LINESTYLES[ch_idx], color=color, zorder=ZORDER_FIT_CURVE)

        for ch_idx, mom_list in enumerate(self.chan_momt_list):
            ref_weff = en_lookup[(ch_idx, mom_list[0])].p["weff_0"]
            for mom in mom_list:
                zn = en_lookup[(ch_idx, mom)].p["weff_0"] / ref_weff
                plt.errorbar(
                    mom, gv.mean(zn), gv.sdev(zn),
                    fmt=MARKERS[ch_idx], color=COLORS[ch_idx],
                    label=rf"${self.chan_name_list[ch_idx]}(n^2)$" if mom == mom_list[0] else None,
                    **ERRORBAR_KW,
                )

        label_axes(r"$n^2$", rf"$Z_n/Z_0$ on {self.tag_name}")
        plt.xlim(0, mom_max)
        plt.ylim(0, 1)
        add_legend("upper right")
        self._save("Zn")

    def plot_dispersion(
        self, en_fit_list: FitResultList, disp_fit_list: FitResultList
    ) -> None:
        if self.corr_type != "meson":
            print("Plot dispersion relation only from meson En.")
            return

        en_lookup = en_fit_lookup(en_fit_list)
        disp_lookup = disp_fit_lookup(disp_fit_list)
        mom_max = max(max(moms) for moms in self.chan_momt_list if moms)
        en_sq_by_ch: dict[int, list] = {}

        new_figure(FIG_STANDARD)
        for ch_idx, mom_list in enumerate(self.chan_momt_list):
            en_sq_by_ch[ch_idx] = [
                (en_lookup[(ch_idx, mom)].p["meff_0"] * self.at_invs) ** 2
                for mom in mom_list
            ]
            x = np.linspace(0, mom_max, 100)
            y_fit = MathModels.linear(x, disp_lookup[ch_idx].p)
            m = gv.mean(y_fit)
            s = gv.sdev(y_fit)
            color = COLORS[ch_idx]
            fill_error_band(x, m - s, m + s, color)
            plt.plot(x, m, linestyle=LINESTYLES[ch_idx], color=color, zorder=ZORDER_FIT_CURVE)

        for ch_idx, mom_list in enumerate(self.chan_momt_list):
            y, yerr = gv.mean(en_sq_by_ch[ch_idx]), gv.sdev(en_sq_by_ch[ch_idx])
            for idx, mom in enumerate(mom_list):
                plt.errorbar(
                    mom, y[idx], yerr[idx],
                    fmt=MARKERS[ch_idx], color=COLORS[ch_idx],
                    label=rf"${self.chan_name_list[ch_idx]}(n^2)$" if mom == mom_list[0] else None,
                    **ERRORBAR_KW,
                )

        label_axes(r"$n^2$", rf"$E_n^2$ (GeV$^2$) on {self.tag_name}")
        plt.xlim(0, mom_max)
        all_vals = np.concatenate([gv.mean(v) for v in en_sq_by_ch.values()])
        plt.ylim(np.min(all_vals), np.max(all_vals))
        add_legend("upper left")
        self._save("Dispersion")
