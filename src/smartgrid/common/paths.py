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


@dataclass(slots=True)
class ForecastPaths:
    current_output_path: Path
    archive_output_path: Path


@dataclass(slots=True)
class ReplayPaths:
    output_csv: Path
    metrics_json: Path
    per_day_dir: Path


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


def build_forecast_paths(root_dir: str | Path, target_date: str, run_id: str) -> ForecastPaths:
    root = ensure_dir(root_dir)
    current_dir = ensure_dir(root / "forecasts" / "consumption" / "current")
    archive_dir = ensure_dir(root / "forecasts" / "consumption" / "archive" / run_id)
    filename = f"forecast_{target_date}.csv"
    return ForecastPaths(
        current_output_path=current_dir / filename,
        archive_output_path=archive_dir / filename,
    )


def build_replay_paths(root_dir: str | Path, start_date: str, end_date: str, stamp: str) -> ReplayPaths:
    root = ensure_dir(root_dir)
    replay_dir = ensure_dir(root / "replays" / "consumption" / f"{stamp}__{start_date}__{end_date}")
    per_day_dir = ensure_dir(replay_dir / "per_day")
    return ReplayPaths(
        output_csv=replay_dir / "replay_forecasts.csv",
        metrics_json=replay_dir / "replay_metrics.json",
        per_day_dir=per_day_dir,
    )
