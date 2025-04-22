from typing import Dict, Any

# Import to register in matplotlib's colormap system.
# Do not remove
from cmcrameri import cm

_ = cm
heatmap_cm = "cmc.batlowW_r"

# Standard figure width.
fig_width = 10

# On grid plots, smaller fontsize.
grid_axis_label_kw: Dict[str, Any] = dict(fontsize=8)

# Color conventions.
retinal_onset_col = "red"
lip_peak_col = "blue"
saccade_start_col = "black"
saccade_end_col = "green"
barycenter_color = "cyan"
decoding_color = "purple"
# decoded_loc_color = "orange"
decoded_loc_color = decoding_color
pre_split_fill = "lightblue"
post_split_fill = "lightpink"

# Markers.
common_annotation_line_kw: Dict[str, Any] = dict(
    linestyle="--", alpha=0.8, color="black"
)
saccade_start_vline_kw: Dict[str, Any] = common_annotation_line_kw | dict(
    color=saccade_start_col
)
saccade_end_vline_kw: Dict[str, Any] = common_annotation_line_kw | dict(
    color=saccade_end_col
)

# Gridspecs
common_gs_kw: Dict[str, Any] = {"hspace": 0.5, "wspace": 0.3}

lines_kw: Dict[str, Any] = dict(linewidth=2, color="black")

col_title_fontsize = 15
