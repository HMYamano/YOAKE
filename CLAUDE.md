# CLAUDE.md

## Which Python to run

**Run every Python command in this repo through `$YOAKE_PYTHON`**, not through bare
`python`. Bare `python` on this machine is pythoncore 3.14, which has dearpygui and
OpenCV but **no torch** — so anything that touches the model, training, inference or
the SAM sidecar fails there with a message about installing PyTorch.

```bash
"$YOAKE_PYTHON" -m pytest tests/ -q -m "not gui"
"$YOAKE_PYTHON" -c "import torch; print(torch.cuda.is_available())"
"$YOAKE_PYTHON" -m htrtdetr.cli predict source=... weights=... output_dir=...
```

In the PowerShell tool the same interpreter is `& $env:YOAKE_PYTHON`.

`YOAKE_PYTHON` and `YOAKE_SAM_PYTHON` are both set in `.claude/settings.local.json`,
which is gitignored because the path is machine-specific. If either is unset, point
them at a conda env that has torch **and** dearpygui **and** opencv, and say so
rather than silently falling back to bare `python`.

PATH cannot be used for this: Claude Code's `settings.json` `env` values are literal
strings with no `${VAR}` expansion (verified — `${PATH}`, `$PATH` and `%PATH%` all
arrive uninterpolated), so prepending to PATH would replace it and break the shell.

## Why one env holds everything

`docs/ANNOTATION_COPILOT.md` says the GUI's Python cannot have torch, and that is
true of the two interpreters it measured. It is **not** true of the `yoake` env,
which has dearpygui + opencv + torch + CUDA together. The torch-free purity contract
for `src/htrtdetr/annotation/` still stands — it is what lets that package be tested
from either side — but do not repeat "no interpreter here can do both" as a fact
about this machine.

## Running the GUI

Launch the annotator from the same env, so in-process torch features (Auto Annotate
Project's *Generate object candidates*) are available instead of locked:

```bash
"$YOAKE_PYTHON" -m htrtdetr.annotator --project <folder>
```

## Test suite

Do not export `PYTHONIOENCODING` when running the suite; `tests/test_cli_smoke.py`
sets it on its own children and a global value used to break the parent's decoding.
Use `-m "not gui"` to skip the windowed tests and `-m "not live_model"` to skip the
ones that download a checkpoint.
