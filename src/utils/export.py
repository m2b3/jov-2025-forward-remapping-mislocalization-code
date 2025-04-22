import warnings
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from git import Repo

from src.figures.base import BaseFigure


def get_git_info():
    try:
        repo = Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha[:7]
        return f"{sha}{'_dirty' if repo.is_dirty() else ''}"
    except Exception:
        warnings.warn("Failed to retrieve git information", RuntimeWarning)
        return "nogit"


def get_output_dirname():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    git_info = get_git_info()
    return timestamp + "_" + git_info


def get_output_dir(prefix="."):
    """
    Create directory for output storage and return its `Path`.
    """

    p = Path(prefix) / "out" / get_output_dirname()
    p = p.absolute()
    p.mkdir(parents=True)
    print(f"Created output dir: {str(p)}")
    return p


def round_floats(obj, decimal_places=2):
    """
    Useful for export preprocessing.
    Recursively rounds floats in dictionaries, lists, tuples.

    FIXME: beta shouldn't be rounded too much. Would be better to handle the decimal places at the Typst / reading level.
    """

    if isinstance(obj, float):
        return round(obj, decimal_places)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimal_places) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [round_floats(x, decimal_places) for x in obj]
    return obj


def check_serializability(obj: Any, path: str = "") -> None:
    """
    Recursively checks each part of a nested object for JSON serializability.

    Args:
        obj: The object to check
        path: Current path in the object (for nested structures)
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else str(key)
            try:
                json.dumps(value)
            except TypeError as e:
                print(f"Error at path: {current_path}")
                print(f"Value type: {type(value)}")
                print(f"Value: {value}")
                print(f"Error: {str(e)}\n")
            check_serializability(value, current_path)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            current_path = f"{path}[{i}]"
            try:
                json.dumps(value)
            except TypeError as e:
                print(f"Error at path: {current_path}")
                print(f"Value type: {type(value)}")
                print(f"Value: {value}")
                print(f"Error: {str(e)}\n")
            check_serializability(value, current_path)


def export_all_parameters(simulator, figures, out_dir=None):
    """Export all parameters needed for paper reproduction."""
    if out_dir is None:
        out_dir = get_output_dir()

    params = {
        "global": simulator.get_parameters(),
        "figures": {
            name: fig.get_parameters()
            for name, fig in figures.items()
            if fig.get_parameters()
        },
    }

    params_fp = out_dir / "params.json"
    with open(params_fp, "w") as f:
        check_serializability(params)
        s = json.dumps(params, indent=2)
        print(f"All parameters:\n{s}")
        f.write(s)

    return params


def export_all_figures(figures: Dict[str, BaseFigure], out_dir: Path, formats=None):
    formats = formats or ["svg", "pdf", "png"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"Saving figures to {fig_dir}")

    for name, fig in figures.items():
        if not fig.active:
            continue

        print(f"  Exporting {name}")
        for fmt in formats:
            path = fig_dir / f"{name}.{fmt}"
            print(f"    - {fmt}")
            kwargs: Any = {"bbox_inches": "tight"}
            if fmt == "png":
                kwargs["dpi"] = 300
            try:
                fig.fig.savefig(path, **kwargs)
            except Exception as e:
                print(f"    Failed: {e}")
                traceback.print_exc()
