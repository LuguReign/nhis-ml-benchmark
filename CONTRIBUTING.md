# Contributing to nhisml

Thank you for your interest in contributing. This document covers how to report issues, submit patches, add new prediction tasks, and extend the feature set registry.

## Reporting Issues

Please open a GitHub issue and include:
- A minimal, reproducible example if possible
- The nhisml version (`python -c "import nhisml; print(nhisml.__version__)"`)
- Python version and OS
- Any relevant error messages or stack traces

## Development Setup

```bash
git clone https://github.com/LuguReign/nhis-ml-benchmark.git
cd nhisml
pip install -e ".[dev]"
```

This installs the package in editable mode along with `pytest` and `ruff`.

## Running the Tests

```bash
pytest tests/ -v
```

All tests must pass before a pull request can be merged. Tests use only synthetic data; no NHIS data download is required.

## Code Style

Code is linted with [ruff](https://docs.astral.sh/ruff/) with a line length of 100 characters. Before opening a pull request, please run:

```bash
ruff check src/nhisml/
```

## Adding a New Prediction Task

1. Open `src/nhisml/tasks.py`.

2. Write a label-generation function with the signature:

   ```python
   def _my_task(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
       """
       Returns (y, eligible_mask).
       y            : integer array (0/1) — one value per row
       eligible_mask: boolean array — True for rows to include in training/evaluation,
                      or None to include all rows
       """
       ...
   ```

3. Register the task in `_TASKS`:

   ```python
   _TASKS["my_task"] = Task(
       name="my_task",
       problem_type="binary",
       description="Brief description of the target.",
       required_cols=["TARGET_COL"],
       make_labels=_my_task,
   )
   ```

4. Add corresponding tests in `tests/test_tasks.py` covering label values, eligibility masking, and edge cases (missing codes, absent columns).

5. If the task requires columns not already in the `core` feature set, add them to the `required_cols` list; `build_core_year` will include them automatically.

## Adding or Extending a Feature Set

1. Open `src/nhisml/featuresets.py`.

2. Call `_register()` with a new `FeatureSet` dataclass:

   ```python
   _register(
       FeatureSet(
           name="extended",
           description="Extended NHIS predictors including additional health module items.",
           binary_12=["NEW_BIN_COL_A", ...],
           ordinal=["NEW_ORD_COL_A", ...],
           categorical=["NEW_CAT_COL_A", ...],
       )
   )
   ```

3. Add tests in `tests/test_featuresets.py`.

## Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/`)
- [ ] New functionality has corresponding tests
- [ ] `ruff check src/nhisml/` passes with no errors
- [ ] Docstrings are present on new public functions and classes
- [ ] The `SOURCES.txt` and egg-info do not need to be updated manually — they are generated at build time

## Code of Conduct

This project follows a standard open-source code of conduct: be respectful, constructive, and inclusive in all interactions.
