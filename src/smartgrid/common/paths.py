from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smartgrid.common.utils import ensure_dir


@dataclass(slots=True)
class ConsumptionPaths:
    run_dir: Path
    exports_dir: Path
    registry_runs_dir: Path
    registry_current_dir: Path


def build_consumption_paths(root_dir: str | Path, exports_subdir: str, registry_subdir: str, run_id: str) -> ConsumptionPaths:
    root = ensure_dir(root_dir)
    run_dir = ensure_dir(root / "runs" / "consumption" / run_id)
    exports_dir = ensure_dir(root / exports_subdir / run_id)
    registry_runs_dir = ensure_dir(root / registry_subdir / "runs")
    registry_current_dir = ensure_dir(root / registry_subdir / "current")
    return ConsumptionPaths(
        run_dir=run_dir,
        exports_dir=exports_dir,
        registry_runs_dir=registry_runs_dir,
        registry_current_dir=registry_current_dir,
    )
