# Code for Berreby and Krishna, JoV 2025

Public code release for the Journal of Vision paper [How forward remapping predicts perisaccadic biphasic mislocalization (Berreby and Krishna, 2025)](https://doi.org/10.1167/jov.25.7.4).

## Prerequisites
- [uv](https://github.com/astral-sh/uv) for dependency management - used ubiquitously.
- [Typst](https://github.com/typst/typst), but only for the task diagram (Figure 1 **A**).

Tested on Arch Linux, but should work on other platforms.

## Quickstart

Dependencies are managed using `uv`.

**Please install `uv` before trying to run this project**.

Once you have `uv` installed:

```bash
# Clone the repo
git clone https://github.com/m2b3/jov-2025-forward-remapping-mislocalization-code.git
cd jov-2025-forward-remapping-mislocalization-code

# Install dependencies
# You do _not_ need to explicitly (de)activate the venv,
# as long as you use the `uv` command.
uv sync

# Convert notebook from .py to .ipynb
# You should only have to do this once, upon cloning.
uv run jupytext --to ipynb notebooks/generate_figures.py

# Open the notebook in Jupyter Lab.
# Using `uv run` ensures that the notebook
# will run in the appropriate Python virtual environment.
uv run jupyter-lab notebooks/generate_figures.ipynb

# You can now generate the paper's figures and explore the impact of parameter changes.
# Ensure that the "Pair with percent script" option is enabled for your notebook,
# and any changes to the .ipynb will be automatically applied to the .py.
```
