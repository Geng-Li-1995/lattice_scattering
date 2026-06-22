"""Shared plot aesthetics: figure sizes, fonts, colors, and helpers."""

import gvar as gv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Figure sizes (width, height) in inches
# ---------------------------------------------------------------------------
FIG_STANDARD = (8, 6)   # scattering, Zn, dispersion
FIG_WIDE = (13, 8)      # En (many channels / momenta)
FIG_MATRIX = (7, 6)     # GEVP heatmaps

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
FS_LABEL = 20
FS_TITLE = 18
FS_TICK = 16
FS_LEGEND = 18
FS_COLORBAR = 14
FS_TICK_DENSE = 11      # crowded GEVP matrix axis labels

# ---------------------------------------------------------------------------
# Color / marker / linestyle palettes
# ---------------------------------------------------------------------------
COLORS = [
    "blue", "red", "green", "violet", "orange",
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
MAX_BAND_POINTS = 2000
PLOT_FORMATS = ("png", "pdf")

# Draw order (bottom → top): error band → fit curve → data points
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


def _as_float_array(y) -> np.ndarray:
    if isinstance(y, (int, float, np.floating)):
        return np.array([float(y)], dtype=float)
    try:
        return np.asarray(gv.mean(y), dtype=float)
    except (TypeError, ValueError, AttributeError):
        return np.asarray(y, dtype=float)


def fill_error_band(
    x,
    ylo,
    yhi,
    color,
    alpha: float = FIT_CURVE_ALPHA,
    zorder: int = ZORDER_ERROR_BAND,
) -> None:
    """Semi-transparent shaded error band, drawn under fit curves and data points."""
    ax = plt.gca()
    x = _as_float_array(x)
    ylo = _as_float_array(ylo)
    yhi = _as_float_array(yhi)
    if len(x) > MAX_BAND_POINTS:
        idx = np.linspace(0, len(x) - 1, MAX_BAND_POINTS, dtype=int)
        x, ylo, yhi = x[idx], ylo[idx], yhi[idx]
    ax.fill_between(
        x,
        ylo,
        yhi,
        color=color,
        alpha=alpha,
        edgecolor="none",
        linewidth=0,
        zorder=zorder,
    )


def apply_plot_style() -> None:
    """Apply global matplotlib style (TeX, serif font, unified font sizes)."""
    plt.rcParams.update(RC_PARAMS)


def new_figure(figsize: tuple[float, float] = FIG_STANDARD):
    """Create a figure with a standard size."""
    return plt.figure(figsize=figsize)


def label_axes(xlabel: str, ylabel: str) -> None:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def add_legend(loc: str, ncol: int = 1) -> None:
    leg = plt.legend(loc=loc, ncol=ncol, framealpha=0.95)
    if leg is not None:
        leg.set_zorder(ZORDER_LEGEND)


def save_figure(stem: str, *, plot_format: str = "png") -> None:
    """Save the current figure; *stem* is the path without extension."""
    if plot_format not in PLOT_FORMATS:
        raise ValueError(f'plot_format must be one of {PLOT_FORMATS}, got {plot_format!r}')
    path = Path(f"{stem}.{plot_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    if plot_format == "pdf":
        plt.savefig(path, format="pdf", dpi=300)
    else:
        plt.savefig(path, format="png", dpi=150)
    plt.show()


def plot_gvar_band(
    x,
    curve,
    *,
    color: str = FIT_CURVE_COLOR,
    alpha: float = FIT_CURVE_ALPHA,
    zorder: int = ZORDER_ERROR_BAND,
) -> None:
    mean = _as_float_array(curve)
    err = _as_float_array(gv.sdev(curve))
    x = _as_float_array(x)
    ax = plt.gca()
    fill_error_band(x, mean - err, mean + err, color, alpha=alpha, zorder=zorder)
    ax.plot(x, mean, color=color, linestyle="-", zorder=ZORDER_FIT_CURVE)
