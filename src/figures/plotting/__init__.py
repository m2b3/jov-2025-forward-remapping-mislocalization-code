import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from ..style import (
    col_title_fontsize,
    lines_kw,
    heatmap_cm,
    saccade_start_col,
    saccade_end_col,
    lip_peak_col,
    common_annotation_line_kw,
    pre_split_fill,
    post_split_fill,
    decoding_color,
    decoded_loc_color
)


def standard_plot(ax: Axes, individual=False, **kw) -> Line2D:
    """
    Create a plot with standard line style.
    """
    kw = lines_kw | {"marker": "o" if individual else None} | kw
    (line,) = ax.plot([], [], **kw)
    return line


def heatmap(ax: Axes) -> AxesImage:
    """
    Make a standard, empty heatmap with our colormap.
    """
    return ax.imshow(np.zeros((1, 1)), aspect="auto", origin="lower", cmap=heatmap_cm)


def hmark(ax, y, **kw):
    return ax.axhline(y, **(common_annotation_line_kw | kw))


def vmark(ax, y, **kw):
    return ax.axvline(y, **(common_annotation_line_kw | kw))


def sac_start(ax, **kw):
    return vmark(ax, 0, color=saccade_start_col, **kw)


def sac_start_marker(ax) -> Text:
    return add_time_marker(ax, "S", 0, saccade_start_col)


def sac_end(ax, end_time=0, **kw):
    return vmark(ax, end_time, color=saccade_end_col, **kw)


def sac_end_marker(ax) -> Text:
    return add_time_marker(ax, "E", 0, saccade_end_col)


def prop_delay(ax, t=0, **kw):
    return vmark(ax, t, color=lip_peak_col, **kw)


# Vertical line to mark the decoding time.
def decoding_time(ax, t=0, **kw):
    return vmark(ax, t, color=decoding_color, **kw)

# Horizontal line to mark the decoded location on a response / activity map.
def decoded_location(ax, y=0, **kw):
    return hmark(ax, y, color=decoded_loc_color, **kw)

def prop_delay_marker(ax) -> Text:
    return add_time_marker(ax, "P", 0, lip_peak_col)


def add_subplot_label(fig, ax, label, x_offset=-0.04, y_offset=0.008, **kwargs):
    """
    Add a paper-style label to a subplot.
    """
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + x_offset,
        bbox.y1 + y_offset,
        label,
        fontsize=col_title_fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
        transform=fig.transFigure,
        **kwargs,
    )


def add_subplot_letter(fig, ax, idx, **kw):
    """
    Label a subplot with a letter (A, B, C...) based on its index.
    """
    add_subplot_label(fig, ax, chr(ord("A") + idx), **kw)


def add_time_marker(
    ax: Axes,
    label: str,
    x: float,
    color: str,
    y: float = 1.02,
) -> Text:
    """Add a time marker (like saccade start 'S' or end 'E') above an axis.

    Args:
        ax: The matplotlib axis to add the marker to
        label: The marker text (typically 'S' or 'E')
        x: The x-position for the marker
        color: The color for the marker
        y: The y-position in axis coordinates (default 1.02)

    Returns:
        The Text object (useful for updating dynamic markers)
    """
    return ax.text(
        x,
        y,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=color,
        fontweight="bold",
    )


def fill_split_regions(ax, x, y, split_x):
    """Fill curve regions before/after split_x with different colors."""
    return [
        ax.fill_between(x, y, where=x <= split_x, color=pre_split_fill, alpha=0.2),
        ax.fill_between(x, y, where=x > split_x, color=post_split_fill, alpha=0.2),
    ]
