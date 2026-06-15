"""Shared plot aesthetics: figure sizes, fonts, colors, and helpers."""

import gvar as gv
import matplotlib.pyplot as plt

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

ERRORBAR_KW = dict(
    markersize=4,
    capsize=3,
    linewidth=1.5,
    capthick=1,
    markeredgecolor="black",
    markerfacecolor="white",
)

FIT_CURVE_COLOR = "green"
FIT_CURVE_ALPHA = 0.2


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
    plt.legend(loc=loc, ncol=ncol)


def save_figure(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_gvar_band(
    x,
    curve,
    *,
    color: str = FIT_CURVE_COLOR,
    alpha: float = FIT_CURVE_ALPHA,
) -> None:
    mean = gv.mean(curve)
    err = gv.sdev(curve)
    plt.plot(x, mean, color=color, linestyle="-")
    plt.fill_between(x, mean - err, mean + err, color=color, alpha=alpha)
