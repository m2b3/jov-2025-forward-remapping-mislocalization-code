import pandas as pd
import numpy as np
from typing import Dict

# WebPlotDigitizer form
def parse_xy_columns(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    df = df.iloc[1:]  # Skip units
    data = {}
    for col in df.columns:
        if col.startswith('Unnamed'): continue
        idx = df.columns.get_loc(col)
        xy = pd.DataFrame({
            'x': pd.to_numeric(df.iloc[:, idx], errors='coerce'),
            'y': pd.to_numeric(df.iloc[:, idx + 1], errors='coerce')
        }).dropna().sort_values('x')
        data[col] = {'x': xy['x'].values, 'y': xy['y'].values}
    return data
