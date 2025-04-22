# src/digitized/honda.py
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

from .wpd_utils import parse_xy_columns

def bin_time_series(x: np.ndarray, y: np.ndarray, *,
                   bin_width: float = 20.0,
                   t_start: float = -200,
                   t_end: float = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Bin x,y data into fixed time windows with specified bounds."""
    bins = np.arange(t_start, t_end + bin_width, bin_width)
    bin_indices = np.digitize(x, bins)

    x_binned = []
    y_binned = []

    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.any():
            x_binned.append(bins[i-1] + bin_width/2)  # bin center
            y_binned.append(y[mask].mean())

    return np.array(x_binned), np.array(y_binned)

def load_dataset(path: Path, bin_params: Optional[Dict] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Load a single dataset, optionally binning each series"""
    data = parse_xy_columns(pd.read_csv(path))
    if bin_params is not None:
        binned_data = {}
        for name, series in data.items():
            x_binned, y_binned = bin_time_series(series['x'], series['y'], **bin_params)
            binned_data[name] = {'x': x_binned, 'y': y_binned}
        return binned_data
    return data

def load_honda_data() -> Tuple[List[Dict[str, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """Load all Honda datasets, returns (list of series data, aggregated x/y arrays)"""
    base = Path('../digitization')
    bin_params = {'bin_width': 20.0, 't_start': -200, 't_end': 200}
    files = {
        'Honda (1991), Fig. 2': ('honda-1991-time-courses/H1991-TC-Fig2-alldata.csv', None),
        'Honda (1991), Fig. 3': ('honda-1991-time-courses/H1991-TC-Fig3-alldata.csv', bin_params),
        'Honda (1993), Fig. 3': ('honda-1993-contingent-structured-bg/Fig3-H1993-all.csv', None),
        'Honda (1999), Fig. 3': ('honda-1999-modification-contingent-frame/Fig3-Dark-Honda1999-all-datasets.csv', None)
    }

    all_series = []
    x_all, y_all = [], []

    # Use items() so that we capture the paper name.
    for paper, (fname, params) in files.items():
        data = load_dataset(base / fname, params)
        for series in data.values():
            series['paper'] = paper  # Tag each series with its paper.
            all_series.append(series)
            x_all.extend(series['x'])
            y_all.extend(series['y'])

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    sort_idx = np.argsort(x_all)

    return all_series, (x_all[sort_idx], y_all[sort_idx])
