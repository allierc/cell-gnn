"""Centralized plot styling for the cell-gnn project.

Design principles:
  - consistent figure sizes across all plots
  - consistent font sizes and DPI
  - single source of truth for visual parameters
  - two pre-built styles: default (white) and dark (black background)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter


@dataclass
class FigureStyle:
    """Centralized figure styling.

    Usage::

        from cell_gnn.figure_style import default_style as style

        style.apply_globally()          # once at program start
        fig, ax = style.figure()        # create pre-styled figure
        ax.plot(x, y)
        style.savefig(fig, "out.png")
    """

    # --- typography --------------------------------------------------------
    font_size: float = 14.0
    tick_font_size: float = 12.0
    label_font_size: float = 14.0
    annotation_font_size: float = 10.0
    use_latex: bool = False

    # --- large frame fonts (cell/field visualization) ------------------
    frame_title_font_size: float = 48.0
    frame_tick_font_size: float = 32.0
    frame_label_font_size: float = 32.0

    # --- geometry ----------------------------------------------------------
    figure_height: float = 8.0          # inches — cell-gnn uses large square figs
    default_aspect: float = 1.0         # square plots are the norm
    dpi: int = 170

    # --- colors ------------------------------------------------------------
    foreground: str = "black"
    background: str = "white"

    # --- decoration --------------------------------------------------------
    show_spines: bool = True
    show_grid: bool = False

    # --- line / marker defaults --------------------------------------------
    line_width: float = 2.0
    marker_size: float = 10.0

    # --- montage panel settings (epoch summary Fig_N.tif) -----------------
    # All panels saved as standalone images for the 2x2 montage must use
    # these settings so fonts / sizes are consistent when loaded as rasters.
    montage_height: float = 12.0
    montage_tick_font_size: float = 24.0
    montage_label_font_size: float = 24.0
    montage_annotation_font_size: float = 16.0
    montage_legend_font_size: float = 16.0

    # --- cell-gnn specific ---------------------------------------------
    cell_scatter_size: float = 10.0
    embedding_scatter_size: float = 5.0

    # --- dark mode flag (set by singletons) --------------------------------
    _is_dark: bool = False

    # ---------------------------------------------------------------------- #
    #  Public API
    # ---------------------------------------------------------------------- #

    def apply_globally(self) -> None:
        """Push style into matplotlib rcParams. Call once at program start."""
        if self._is_dark:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        plt.rcParams.update({
            "font.size": self.font_size,
            "axes.titlesize": self.font_size,
            "axes.labelsize": self.label_font_size,
            "xtick.labelsize": self.tick_font_size,
            "ytick.labelsize": self.tick_font_size,
            "legend.fontsize": self.tick_font_size,
            "text.usetex": self.use_latex,
            "figure.facecolor": self.background,
            "axes.facecolor": self.background,
            "savefig.dpi": self.dpi,
            "savefig.pad_inches": 0.05,
            "text.color": self.foreground,
            "axes.labelcolor": self.foreground,
            "xtick.color": self.foreground,
            "ytick.color": self.foreground,
        })

    def clean_ax(self, ax: Axes) -> Axes:
        """Apply consistent styling to a single axes."""
        for spine in ax.spines.values():
            spine.set_visible(self.show_spines)
        if not self.show_grid:
            ax.grid(False)
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=self.tick_font_size,
            colors=self.foreground,
        )
        return ax

    def figure(
        self,
        ncols: int = 1,
        nrows: int = 1,
        aspect: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        formatx: str = '%.2f',
        formaty: str = '%.2f',
        **subplot_kw,
    ) -> Tuple[Figure, Union[Axes, np.ndarray]]:
        """Create a pre-styled figure + axes.

        Height defaults to ``self.figure_height`` per row.
        Width is computed from aspect ratio unless overridden.

        Returns ``(fig, axes)`` where *axes* is a single ``Axes`` when
        ``nrows == ncols == 1``, otherwise a numpy array.
        """
        h = height or self.figure_height
        a = aspect or self.default_aspect
        w = width or (h * a * ncols / max(nrows, 1))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(w, h * nrows),
            facecolor=self.background,
            **subplot_kw,
        )

        if isinstance(axes, np.ndarray):
            for ax in axes.flat:
                self.clean_ax(ax)
                ax.xaxis.set_major_formatter(FormatStrFormatter(formatx))
                ax.yaxis.set_major_formatter(FormatStrFormatter(formaty))
        else:
            self.clean_ax(axes)
            axes.xaxis.set_major_locator(plt.MaxNLocator(3))
            axes.yaxis.set_major_locator(plt.MaxNLocator(3))
            axes.xaxis.set_major_formatter(FormatStrFormatter(formatx))
            axes.yaxis.set_major_formatter(FormatStrFormatter(formaty))

        return fig, axes

    def montage_figure(
        self,
        ncols: int = 1,
        nrows: int = 1,
        aspect: Optional[float] = None,
        width: Optional[float] = None,
        **subplot_kw,
    ) -> Tuple[Figure, Union[Axes, np.ndarray]]:
        """Create a figure sized and styled for the epoch montage (Fig_N.tif).

        All standalone panels (embedding, MLP1, loss, UMAP) that get loaded
        into the 2x2 montage should use this method so that figure height,
        tick sizes, and label sizes are identical across panels.

        Returns ``(fig, axes)`` — same convention as :meth:`figure`.
        """
        h = self.montage_height
        a = aspect or self.default_aspect
        w = width or (h * a * ncols / max(nrows, 1))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(w, h * nrows),
            facecolor=self.background,
            **subplot_kw,
        )

        def _style_ax(ax: Axes) -> None:
            self.clean_ax(ax)
            ax.tick_params(
                axis="both", which="both",
                labelsize=self.montage_tick_font_size,
                colors=self.foreground,
            )

        if isinstance(axes, np.ndarray):
            for ax in axes.flat:
                _style_ax(ax)
        else:
            _style_ax(axes)

        return fig, axes

    def montage_xlabel(self, ax: Axes, text: str, **kwargs) -> None:
        """Set x-label with montage-consistent font size."""
        ax.set_xlabel(
            text,
            fontsize=kwargs.pop("fontsize", self.montage_label_font_size),
            color=kwargs.pop("color", self.foreground),
            **kwargs,
        )

    def montage_ylabel(self, ax: Axes, text: str, **kwargs) -> None:
        """Set y-label with montage-consistent font size."""
        ax.set_ylabel(
            text,
            fontsize=kwargs.pop("fontsize", self.montage_label_font_size),
            color=kwargs.pop("color", self.foreground),
            **kwargs,
        )

    def montage_annotate(self, ax: Axes, text: str, xy: tuple, **kwargs) -> None:
        """Add annotation with montage-consistent font size."""
        ax.text(
            *xy,
            text,
            fontsize=kwargs.pop("fontsize", self.montage_annotation_font_size),
            color=kwargs.pop("color", self.foreground),
            transform=kwargs.pop("transform", ax.transAxes),
            **kwargs,
        )

    def savefig(self, fig: Figure, path: str, close: bool = True, **kwargs) -> None:
        """Save with consistent DPI and tight bbox, then close."""
        defaults = dict(
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor=self.background,
        )
        defaults.update(kwargs)
        fig.savefig(path, **defaults)
        if close:
            plt.close(fig)

    def xlabel(self, ax: Axes, text: str, **kwargs) -> None:
        ax.set_xlabel(
            text,
            fontsize=kwargs.pop("fontsize", self.label_font_size),
            color=kwargs.pop("color", self.foreground),
            **kwargs,
        )

    def ylabel(self, ax: Axes, text: str, **kwargs) -> None:
        ax.set_ylabel(
            text,
            fontsize=kwargs.pop("fontsize", self.label_font_size),
            color=kwargs.pop("color", self.foreground),
            **kwargs,
        )

    def annotate(self, ax: Axes, text: str, xy: tuple, **kwargs) -> None:
        """Add text annotation with consistent font."""
        ax.text(
            *xy,
            text,
            fontsize=kwargs.pop("fontsize", self.annotation_font_size),
            color=kwargs.pop("color", self.foreground),
            transform=kwargs.pop("transform", ax.transAxes),
            **kwargs,
        )


# --------------------------------------------------------------------------- #
#  Module-level singletons
# --------------------------------------------------------------------------- #

default_style = FigureStyle()
dark_style = FigureStyle(foreground="white", background="black", _is_dark=True)
