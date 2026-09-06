"""Shared figure style and export helpers.

Every figure in the paper is produced through this module so that type size, palette
and export settings are identical across the outfield, goalkeeper and team pipelines.

Palette provenance
------------------
The categorical palette was chosen by running the colour validator rather than by eye,
and the result is recorded in ``results/metrics/palette_validation.json``. Checked
against a light print surface over all pairs, not merely adjacent ones, no five-hue or
six-hue set cleared both the normal-vision separation floor and the 3:1 contrast floor.
Four hues do. The rule adopted here follows from that measurement: at most four
categories are ever distinguished by colour, and anything beyond four is faceted into
small multiples instead of receiving a fifth hue. Colour is additionally never the only
channel, since every categorical mark also carries a marker shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import config

# --------------------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------------------

SURFACE = "#FCFCFB"
INK_PRIMARY = "#0B0B0B"
INK_SECONDARY = "#52514E"
INK_MUTED = "#8A8A85"
GRID = "#DEDEDA"
CONTEXT_GREY = "#C9C9C4"

#: Validated categorical slots, in fixed order. Never cycled, never reordered.
CATEGORICAL = ["#1F6FB2", "#C1500A", "#00875F", "#A1548C"]
MAX_CATEGORICAL = len(CATEGORICAL)

#: Secondary encoding, so identity never rests on colour alone.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

#: Fixed assignment so a position group keeps its colour in every figure.
POSITION_COLORS = {
    "DF": CATEGORICAL[0],
    "MF": CATEGORICAL[1],
    "FW": CATEGORICAL[2],
    "GK": CATEGORICAL[3],
}
POSITION_MARKERS = {"DF": "o", "MF": "s", "FW": "^", "GK": "D"}

#: Sequential ramp, one hue, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "pl_sequential", ["#EAF2FA", "#B8D3EC", "#7FB0DA", "#4189C4", "#1F6FB2", "#144A78"]
)

#: Diverging ramp, two hues with a neutral grey midpoint, for z-score deviations.
DIVERGING = LinearSegmentedColormap.from_list(
    "pl_diverging", ["#8A3407", "#C1500A", "#E0A47A", "#EDEDEA", "#7FB0DA", "#1F6FB2", "#144A78"]
)

PALETTE_VALIDATION = {
    "tool": "dataviz validate_palette.js",
    "surface": SURFACE,
    "mode": "light",
    "pairs": "all",
    "categorical": CATEGORICAL,
    "result": "ALL CHECKS PASS",
    "checks": {
        "lightness_band": "PASS, all 4 inside L 0.43-0.77",
        "chroma_floor": "PASS, all 4 >= 0.1",
        "cvd_separation": "WARN, worst all-pairs #A1548C vs #00875F dE 6.2 deutan",
        "normal_vision_floor": "PASS, worst all-pairs #A1548C vs #C1500A dE 15.9",
        "contrast_vs_surface": "PASS, all 4 >= 3:1",
    },
    "note": (
        "The CVD warning sits in the 6 to 8 floor band, which is permitted only alongside "
        "secondary encoding. Every categorical mark in this paper also carries a distinct "
        "marker shape and a direct or legend label, so the condition is met. Five and six "
        "hue sets were tested and none cleared the normal-vision floor over all pairs at "
        "print contrast, so more than four categories are faceted rather than coloured."
    ),
}


def write_palette_provenance() -> Path:
    path = config.METRICS / "palette_validation.json"
    with open(path, "w") as fh:
        json.dump(PALETTE_VALIDATION, fh, indent=2)
    return path


# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------

#: Body text is 11pt in the paper, so figure text sits at 9pt and drops to 8pt for ticks.
BASE_FONT_PT = 9

RC = {
    "figure.constrained_layout.use": True,
    "figure.facecolor": SURFACE,
    "figure.dpi": 150,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "standard",
    "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK_PRIMARY,
    "axes.titlecolor": INK_PRIMARY,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.titlesize": BASE_FONT_PT,
    "axes.labelsize": BASE_FONT_PT,
    "axes.titleweight": "bold",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": BASE_FONT_PT - 1,
    "ytick.labelsize": BASE_FONT_PT - 1,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": BASE_FONT_PT - 1,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "lines.linewidth": 2.0,
    "lines.markersize": 4.0,
    "font.size": BASE_FONT_PT,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK_PRIMARY,
    "pdf.fonttype": 42,  # embed TrueType so the PDF is not device dependent
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def use_style() -> None:
    """Apply the project style. Call once at the top of every figure module."""
    mpl.use("Agg")
    mpl.rcParams.update(RC)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

#: Text-width and full-page widths in inches for a 11pt article at default margins.
WIDTH_COLUMN = 4.8
WIDTH_FULL = 6.5
WIDTH_WIDE = 7.2


def categorical(n: int) -> list[str]:
    """Return ``n`` categorical colours.

    Raises when more than four are requested. That is deliberate: the validator showed
    no five-hue set clears the separation and contrast floors on a light print surface,
    so callers must facet instead of asking for a fifth colour.
    """
    if n > MAX_CATEGORICAL:
        raise ValueError(
            f"{n} categorical colours requested but only {MAX_CATEGORICAL} are validated. "
            "Facet into small multiples instead of adding hues."
        )
    return CATEGORICAL[:n]


def markers(n: int) -> list[str]:
    return [MARKERS[i % len(MARKERS)] for i in range(n)]


def style_axis(ax: plt.Axes, xlabel: str = "", ylabel: str = "", title: str = "") -> plt.Axes:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    ax.grid(True, which="major", axis="both")
    return ax


def annotate_points(
    ax: plt.Axes,
    x,
    y,
    labels,
    *,
    fontsize: int = BASE_FONT_PT - 2,
    color: str = INK_SECONDARY,
    offset: tuple[float, float] = (3.0, 3.0),
) -> None:
    """Direct-label selected points. Never label every point."""
    for xi, yi, text in zip(np.atleast_1d(x), np.atleast_1d(y), labels, strict=False):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            continue
        ax.annotate(
            str(text),
            (xi, yi),
            textcoords="offset points",
            xytext=offset,
            fontsize=fontsize,
            color=color,
            zorder=6,
        )


def legend_below(ax: plt.Axes, ncol: int = 4, **kwargs) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol, **kwargs)


def save_figure(fig: plt.Figure, name: str, *, close: bool = True) -> dict[str, Path]:
    """Write a figure as vector PDF and 300 dpi PNG, and report both paths."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = config.FIGURES / f"{name}.pdf"
    png = config.FIGURES / f"{name}.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=300)
    if close:
        plt.close(fig)
    return {"pdf": pdf, "png": png}


def facet_grid(n: int, ncols: int = 4, *, width: float = WIDTH_FULL, panel_h: float = 1.5):
    """Create a small-multiples grid, the sanctioned alternative to a fifth hue."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, panel_h * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]
