"""Shared plot aesthetics: styles, helpers, and ``BasePlotter``."""

from __future__ import annotations

from pathlib import Path

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np

from input.config import Config

# ---------------------------------------------------------------------------
# Figure sizes (width, height) in inches
# ---------------------------------------------------------------------------
FIG_STANDARD = (8, 6)
FIG_WIDE = (13, 8)
FIG_MATRIX = (7, 6)

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
FS_LABEL = 20
FS_TITLE = 18
FS_TICK = 16
FS_LEGEND = 18
FS_COLORBAR = 14
FS_TICK_DENSE = 11

# ---------------------------------------------------------------------------
# Palettes and draw order
# ---------------------------------------------------------------------------
COLORS = [
    "blue", "red", "green", "purple", "orange",
    "black", "cyan", "navy", "yellow", "brown",
]
MARKERS = ["o", "x", "s", "^", "v", "<", ">", "*", "D", "p"]
LINESTYLES = [
    "-", "--", ":", "-.",
    (0, (1, 1)), (0, (5, 2)), (0, (3, 1, 1, 1)),
    (0, (5, 5)), (0, (2, 2)), (0, (1, 2)),
]

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
}

FIT_CURVE_COLOR = "green"
FIT_CURVE_ALPHA = 0.3
COMBINE_LINE_COLOR = "green"
COMBINE_BAND_COLOR = "lightgreen"
MAX_BAND_POINTS = 2000
PLOT_FORMATS = ("png", "pdf")

ZORDER_ERROR_BAND = 1
ZORDER_FIT_CURVE = 2
ZORDER_AUX_LINE = 3
ZORDER_DATA = 5
ZORDER_LEGEND = 10

ERRORBAR_KW = dict(
    markersize=4,
    capsize=3,
    linewidth=1.5,
    capthick=1,
    markeredgecolor="black",
    markerfacecolor="white",
    zorder=ZORDER_DATA,
)


# ---------------------------------------------------------------------------
# Channel labels (terminal + LaTeX)
# ---------------------------------------------------------------------------
def chan_mom_label(chan_name: str, mom: int) -> str:
    return f"{chan_name}(n^2={mom})"


def chan_mom_latex(chan_name: str, mom: int) -> str:
    return rf"${chan_name}(n^2={mom})$"


def chan_mom_latex_n2(chan_name: str) -> str:
    return rf"${chan_name}(n^2)$"


def log_channel_tag(chan_idx: int, mom: int) -> str:
    return f"channel {chan_idx} n^2={mom}"


def meson_energy_subscript(chan_name: str, mom: int) -> str:
    return rf"E_{{{chan_name}(n^2={mom})}}"


# ---------------------------------------------------------------------------
# Array / axis helpers
# ---------------------------------------------------------------------------
def _as_float_array(y) -> np.ndarray:
    if isinstance(y, (int, float, np.floating)):
        return np.array([float(y)], dtype=float)
    try:
        return np.asarray(gv.mean(y), dtype=float)
    except (TypeError, ValueError, AttributeError):
        return np.asarray(y, dtype=float)


def axis_limits_from_values(values, padding_fraction: float = 0.10) -> tuple[float, float]:
    arr = _as_float_array(values)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = np.min(arr)
    hi = np.max(arr)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    padding = padding_fraction * span
    return lo - padding, hi + padding


def combine_axis_limits(*limits: tuple[float, float]) -> tuple[float, float]:
    return min(lo for lo, _ in limits), max(hi for _, hi in limits)


def y_limits_from_error(
    center: float, err: float, *, scale: float = 10.0
) -> tuple[float, float]:
    half = max(abs(err) * scale, 1e-12)
    return center - half, center + half


def energy_gev_mean(at_invs: float, values) -> np.ndarray:
    return np.asarray(gv.mean(values), dtype=float) * at_invs


def energy_gev_sdev(at_invs: float, values) -> np.ndarray:
    return np.asarray(gv.sdev(values), dtype=float) * at_invs


def energy_gev_scalar(at_invs: float, value) -> float:
    return float(gv.mean(value)) * at_invs


def energy_gev_sdev_scalar(at_invs: float, value) -> float:
    return float(gv.sdev(value)) * at_invs


def format_energy_gev(val: float, err: float) -> str:
    return f"{gv.gvar(val, err)} GeV"


# ---------------------------------------------------------------------------
# Figure primitives
# ---------------------------------------------------------------------------
def apply_plot_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def new_figure(figsize: tuple[float, float] = FIG_STANDARD):
    return plt.figure(figsize=figsize)


def label_axes(xlabel: str, ylabel: str, *, title: str | None = None) -> None:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    set_figure_title(title)


def ensemble_title(
    tag_name: str, *, prefix: str | None = None, sep: str = ", "
) -> str:
    if prefix:
        return f"{prefix}{sep}{tag_name}"
    return tag_name


def figure_title(
    config: Config,
    tag_name: str,
    *,
    prefix: str | None = None,
    sep: str = ", ",
) -> str | None:
    if not config.is_plot_title:
        return None
    return ensemble_title(tag_name, prefix=prefix, sep=sep)


def set_figure_title(
    title: str | None,
    *,
    ax: plt.Axes | None = None,
    fontsize: float = FS_TITLE,
    pad: float = 6,
) -> None:
    if title is None:
        return
    if ax is None:
        plt.title(title, fontsize=fontsize)
    else:
        ax.set_title(title, fontsize=fontsize, pad=pad)


def fill_error_band(
    x,
    ylo,
    yhi,
    color,
    alpha: float = FIT_CURVE_ALPHA,
    zorder: int = ZORDER_ERROR_BAND,
) -> None:
    ax = plt.gca()
    x = _as_float_array(x)
    ylo = _as_float_array(ylo)
    yhi = _as_float_array(yhi)
    if len(x) > MAX_BAND_POINTS:
        idx = np.linspace(0, len(x) - 1, MAX_BAND_POINTS, dtype=int)
        x, ylo, yhi = x[idx], ylo[idx], yhi[idx]
    ax.fill_between(
        x, ylo, yhi,
        color=color, alpha=alpha,
        edgecolor="none", linewidth=0,
        zorder=zorder,
    )


def plot_combine_reference(
    ax: plt.Axes,
    x,
    ref_val: float,
    ref_err: float,
) -> None:
    x_arr = _as_float_array(x)
    ax.plot(
        x_arr, np.full(len(x_arr), ref_val),
        color=COMBINE_LINE_COLOR, label="Combine", zorder=ZORDER_AUX_LINE,
    )
    ax.fill_between(
        x_arr, ref_val - ref_err, ref_val + ref_err,
        color=COMBINE_BAND_COLOR, zorder=ZORDER_ERROR_BAND,
    )


def chi2_marker_kw(color: str, marker: str) -> dict:
    return {
        "linestyle": "none",
        "marker": marker,
        "color": color,
        "markersize": ERRORBAR_KW["markersize"],
        "markeredgecolor": color,
        "markerfacecolor": ERRORBAR_KW["markerfacecolor"],
        "zorder": ZORDER_DATA,
    }


def draw_origin_axes() -> None:
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")


def palette_color_at(index: int, *, skip: tuple[str, ...] = ()) -> str:
    while COLORS[index % len(COLORS)] in skip:
        index += 1
    return COLORS[index % len(COLORS)]


def plot_errorbars(
    x, y, *, xerr=None, yerr=None, color: str, label: str, fmt: str = "x",
) -> None:
    plt.errorbar(
        x, y, xerr=xerr, yerr=yerr,
        fmt=fmt, color=color, label=label,
        **ERRORBAR_KW,
    )


def plot_gvar_band(
    x, curve, *, color: str = FIT_CURVE_COLOR,
    alpha: float = FIT_CURVE_ALPHA, zorder: int = ZORDER_ERROR_BAND,
) -> None:
    mean = _as_float_array(curve)
    err = _as_float_array(gv.sdev(curve))
    x = _as_float_array(x)
    fill_error_band(x, mean - err, mean + err, color, alpha=alpha, zorder=zorder)
    plt.gca().plot(x, mean, color=color, linestyle="-", zorder=ZORDER_FIT_CURVE)


def plot_kcot_zeta_reference(
    k_sq_grid: np.ndarray,
    kcot_grid: np.ndarray,
    color: str,
    k_sq_mean: np.ndarray,
    k_sq_err: np.ndarray,
) -> None:
    plt.plot(k_sq_grid, kcot_grid, color=color, linestyle="--", alpha=0.3, zorder=ZORDER_AUX_LINE)
    for mean, err in zip(k_sq_mean, k_sq_err):
        mask = (k_sq_grid >= mean - err) & (k_sq_grid <= mean + err)
        plt.plot(k_sq_grid[mask], kcot_grid[mask], color=color, linestyle="-", zorder=ZORDER_FIT_CURVE)


def plot_ik_parabola(k_sq_lim: tuple[float, float], color: str) -> None:
    k_lo = min(k_sq_lim[0], 0.0)
    if k_lo >= 0:
        return
    k_sq = np.linspace(k_lo, 0.0, 300)
    ik = np.sqrt(-k_sq)
    plt.plot(k_sq, ik, color=color, linestyle=":", linewidth=1.5, label=r"$ik$", zorder=ZORDER_AUX_LINE)
    plt.plot(k_sq, -ik, color=color, linestyle=":", linewidth=1.5, zorder=ZORDER_AUX_LINE)


def ik_parabola_axis_limits(k_sq_lo: float) -> tuple[float, float] | None:
    if k_sq_lo >= 0:
        return None
    k_sq_neg = np.linspace(k_sq_lo, 0.0, 50)
    ik_vals = np.concatenate([np.sqrt(-k_sq_neg), -np.sqrt(-k_sq_neg)])
    return axis_limits_from_values(ik_vals)


def add_legend(loc: str, ncol: int = 1) -> None:
    leg = plt.legend(loc=loc, ncol=ncol, framealpha=0.95)
    if leg is not None:
        leg.set_zorder(ZORDER_LEGEND)


def save_figure(stem: str, *, plot_format: str = "png", show: bool = True) -> None:
    if plot_format not in PLOT_FORMATS:
        raise ValueError(f'plot_format must be one of {PLOT_FORMATS}, got {plot_format!r}')
    path = Path(f"{stem}.{plot_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    if plot_format == "pdf":
        plt.savefig(path, format="pdf", dpi=300)
    else:
        plt.savefig(path, format="png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Base plotter
# ---------------------------------------------------------------------------
class BasePlotter:
    """Style setup, config shortcuts, titles, and save."""

    def __init__(self, config: Config) -> None:
        apply_plot_style()
        self.config = config
        self.input_name = config.input_name
        self.tag_name = config.tag_name
        self.at_invs = config.at_invs
        self.chan_name_list = config.chan_name_list
        self.chan_momentum_list = config.chan_momentum_list
        self.lattice_Nt = config.lattice_Nt

    def _figure_title(
        self, *, prefix: str | None = None, sep: str = ", "
    ) -> str | None:
        return figure_title(self.config, self.tag_name, prefix=prefix, sep=sep)

    def _energy_gev_mean(self, values) -> np.ndarray:
        return energy_gev_mean(self.at_invs, values)

    def _energy_gev_sdev(self, values) -> np.ndarray:
        return energy_gev_sdev(self.at_invs, values)

    def _energy_gev_scalar(self, value) -> float:
        return energy_gev_scalar(self.at_invs, value)

    def _energy_gev_sdev_scalar(self, value) -> float:
        return energy_gev_sdev_scalar(self.at_invs, value)

    def _format_energy_gev(self, val: float, err: float) -> str:
        return format_energy_gev(val, err)

    def _save(self, stem: str) -> None:
        save_figure(
            stem,
            plot_format=self.config.plot_format,
            show=self.config.is_plot_show,
        )
