"""Default location for runtime artifacts outside the source checkout."""

import os
from pathlib import Path

RUNS_ROOT = Path(
    os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")
).expanduser()
