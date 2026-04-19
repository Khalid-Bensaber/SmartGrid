from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from smartgrid.common.constants import STRICT_DAY_AHEAD_MODE
from smartgrid.common.utils import ensure_dir, load_yaml
from smartgrid.evaluation.metrics import seasonal_naive_weekly


@dataclass(slots=True, frozen=True)
class DemoPaths:
    root: Path
    artifacts_root: Path
    configs_dir: Path
    notebook_export_root: Path


@dataclass(slots=True)
class CliResult:
    command: list[str]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str

    @property
    def command_text(self) -> str:
        return shlex.join(self.command)

    def extract_json(self) -> dict[str, Any]:
        return extract_last_json(self.stdout)


def configure_pandas_display() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 2400)
    pd.set_option("display.max_rows", 200)


def find_repo_root(start: str | Path | None = None) -> Path:
    start_path = Path(start or Path.cwd()).resolve()
    for candidate in [start_path] + list(start_path.parents):
        if (
            (candidate / "Makefile").exists()
            and (candidate / "scripts" / "train_consumption.py").exists()
            and (candidate / "src" / "smartgrid").exists()
        ):
            return candidate
    raise FileNotFoundError(
        f"Unable to find the SmartGrid repo root from {start_path}. "
        "Open the notebook from inside the repository or set the root manually."
    )


def ensure_repo_on_path(root: str | Path) -> None:
    root_path = Path(root).resolve()
    for candidate in [root_path, root_path / "src"]:
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def build_demo_paths(
    root: str | Path,
    notebook_export_dir: str = "artifacts/notebook_exports/cli_demo_v3",
) -> DemoPaths:
    root_path = Path(root).resolve()
    export_root = ensure_dir(root_path / notebook_export_dir)
    return DemoPaths(
        root=root_path,
        artifacts_root=root_path / "artifacts",
        configs_dir=root_path / "configs" / "consumption",
        notebook_export_root=export_root,
    )


def run_cli(
    command: Sequence[str],
    *,
    cwd: str | Path,
    check: bool = True,
    env_overrides: Mapping[str, str] | None = None,
    echo: bool = True,
) -> CliResult:
    cmd = [str(part) for part in command]
    cwd_path = Path(cwd).resolve()
    env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})

    if echo:
        print("$", shlex.join(cmd))
        print("cwd:", cwd_path)
        if env_overrides:
            print("env overrides:", dict(env_overrides))

    completed = subprocess.run(
        cmd,
        cwd=str(cwd_path),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    result = CliResult(
        command=cmd,
        cwd=cwd_path,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )

    if echo and result.stdout.strip():
        print("STDOUT:")
        print(result.stdout)
    if echo and result.stderr.strip():
        print("STDERR:")
        print(result.stderr)

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"CMD: {result.command_text}\n"
            f"CWD: {result.cwd}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result


def extract_last_json(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(stdout) if char == "{"]

    for start in reversed(starts):
        snippet = stdout[start:]
        try:
            payload, end = decoder.raw_decode(snippet)
        except json.JSONDecodeError:
            continue
        if snippet[end:].strip() == "":
            if isinstance(payload, dict):
                return payload
            raise RuntimeError("The trailing JSON payload is not an object.")

    raise RuntimeError("Unable to find a final JSON object in the command stdout.")


def optional_cli_args(
    *,
    dataset_key: str | None = None,
    catalog_path: str | Path | None = None,
    historical_csv: str | Path | None = None,
    weather_csv: str | Path | None = None,
    holidays_xlsx: str | Path | None = None,
    benchmark_csv: str | Path | None = None,
    allow_fallback: bool = False,
) -> list[str]:
    args: list[str] = []
    if dataset_key:
        args += ["--dataset-key", str(dataset_key)]
    if catalog_path:
        args += ["--catalog-path", str(catalog_path)]
    if historical_csv:
        args += ["--historical-csv", str(historical_csv)]
    if weather_csv:
        args += ["--weather-csv", str(weather_csv)]
    if holidays_xlsx:
        args += ["--holidays-xlsx", str(holidays_xlsx)]
    if benchmark_csv:
        args += ["--benchmark-csv", str(benchmark_csv)]
    if allow_fallback:
        args.append("--allow-fallback")
    return args


def make_overrides(
    *,
    dataset_key: str | None = None,
    historical_csv: str | Path | None = None,
    weather_csv: str | Path | None = None,
    holidays_xlsx: str | Path | None = None,
    benchmark_csv: str | Path | None = None,
) -> list[str]:
    args: list[str] = []
    if dataset_key:
        args.append(f"DATASET_KEY={dataset_key}")
    if historical_csv:
        args.append(f"HISTORICAL_CSV={historical_csv}")
    if weather_csv:
        args.append(f"WEATHER_CSV={weather_csv}")
    if holidays_xlsx:
        args.append(f"HOLIDAYS_XLSX={holidays_xlsx}")
    if benchmark_csv:
        args.append(f"BENCHMARK_CSV={benchmark_csv}")
    return args


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return out_path


def _resolve_path(root: Path, path_like: str | Path | None) -> Path | None:
    if path_like in (None, ""):
        return None
    raw = Path(path_like)
    if raw.is_absolute():
        return raw.resolve()
    return (root / raw).resolve()


def _config_name(config_path: str | Path | None) -> str | None:
    if not config_path:
        return None
    return Path(str(config_path)).stem


def _display_label(
    *,
    experiment_name: str | None,
    config_name: str | None,
    run_id: str | None,
) -> str:
    if experiment_name:
        return str(experiment_name)
    if config_name:
        return str(config_name)
    return str(run_id or "unknown_run")


def build_config_inventory(root: str | Path, config_paths: Sequence[str | Path]) -> pd.DataFrame:
    root_path = Path(root).resolve()
    rows: list[dict[str, Any]] = []

    for config_ref in config_paths:
        resolved = _resolve_path(root_path, config_ref)
        if resolved is None or not resolved.exists():
            rows.append(
                {
                    "config_path": str(config_ref),
                    "config_name": Path(str(config_ref)).stem,
                    "experiment_name": None,
                    "forecast_mode": None,
                    "official_eligible": False,
                    "notes": "missing config file",
                }
            )
            continue

        config = load_yaml(resolved)
        features = dict(config.get("features") or {})
        training = dict(config.get("training") or {})
        rows.append(
            {
                "config_path": str(resolved),
                "config_name": resolved.stem,
                "config_file": resolved.name,
                "experiment_name": config.get("experiment_name"),
                "forecast_mode": features.get("forecast_mode"),
                "official_eligible": features.get("forecast_mode") == STRICT_DAY_AHEAD_MODE,
                "include_weather": bool(features.get("include_weather", False)),
                "weather_mode": features.get("weather_mode"),
                "include_cyclical_time": bool(features.get("include_cyclical_time", False)),
                "include_lag_aggregates": bool(features.get("include_lag_aggregates", False)),
                "include_recent_dynamics": bool(features.get("include_recent_dynamics", False)),
                "include_shifted_recent_dynamics": bool(
                    features.get("include_shifted_recent_dynamics", False)
                ),
                "lag_days": features.get("lag_days"),
                "hidden_layers": training.get("hidden_layers"),
                "notes": (
                    "official strict day-ahead candidate"
                    if features.get("forecast_mode") == STRICT_DAY_AHEAD_MODE
                    else "excluded from official replay leaderboard"
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty and "config_name" in frame.columns:
        frame = frame.sort_values("config_name").reset_index(drop=True)
    return frame


def collect_consumption_runs(artifacts_root: str | Path) -> pd.DataFrame:
    artifacts_path = Path(artifacts_root).resolve()
    exports_root = artifacts_path / "exports" / "consumption"
    rows: list[dict[str, Any]] = []

    if not exports_root.exists():
        return pd.DataFrame()

    for summary_path in sorted(exports_root.glob("*/run_summary.json")):
        summary = read_json(summary_path)
        run_id = str(summary.get("run_id") or summary_path.parent.name)
        config_path = summary.get("config_path")
        config_name = _config_name(config_path)
        metrics_model = summary.get("metrics_model") or summary.get("offline_test_metrics") or {}
        metrics_naive = summary.get("metrics_naive_weekly") or summary.get(
            "offline_test_naive_weekly_metrics"
        ) or {}
        run_dir = summary.get("run_dir") or str(
            (artifacts_path / "runs" / "consumption" / run_id).resolve()
        )
        exports_dir = summary.get("exports_dir") or str(summary_path.parent.resolve())
        rows.append(
            {
                "run_id": run_id,
                "summary_json": str(summary_path.resolve()),
                "run_dir": run_dir,
                "exports_dir": exports_dir,
                "config_path": config_path,
                "config_name": config_name,
                "experiment_name": summary.get("experiment_name"),
                "human_label": _display_label(
                    experiment_name=summary.get("experiment_name"),
                    config_name=config_name,
                    run_id=run_id,
                ),
                "forecast_mode": summary.get("forecast_mode"),
                "official_eligible": summary.get("forecast_mode") == STRICT_DAY_AHEAD_MODE,
                "dataset_key": summary.get("dataset_key"),
                "n_features": summary.get("n_features"),
                "selected_analysis_day": summary.get("selected_analysis_day"),
                "train_duration_sec": summary.get("train_duration_sec"),
                "epochs_ran": summary.get("epochs_ran"),
                "offline_MAE": metrics_model.get("MAE"),
                "offline_RMSE": metrics_model.get("RMSE"),
                "offline_MAPE%": metrics_model.get("MAPE%"),
                "offline_InTolerance%": metrics_model.get("InTolerance%"),
                "offline_RampingError_RMSE": metrics_model.get("RampingError_RMSE"),
                "offline_naive_MAE": metrics_naive.get("MAE"),
                "offline_naive_RMSE": metrics_naive.get("RMSE"),
                "backtest_csv": summary.get("offline_test_backtest_csv")
                or summary.get("backtest_csv"),
                "selected_day_csv": summary.get("offline_test_selected_day_csv")
                or summary.get("day_compare_csv"),
                "total_output_csv": summary.get("offline_test_total_csv")
                or summary.get("output_total_csv"),
                "notebook_output_csv": summary.get("output_notebook_csv"),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("run_id", ascending=False).reset_index(drop=True)
    return frame


def select_latest_runs_per_config(
    runs_df: pd.DataFrame,
    config_paths: Sequence[str | Path],
    *,
    root: str | Path,
    official_only: bool = True,
) -> list[str]:
    if runs_df.empty:
        return []

    root_path = Path(root).resolve()
    wanted = {
        str(_resolve_path(root_path, config_path))
        for config_path in config_paths
        if _resolve_path(root_path, config_path) is not None
    }
    work = runs_df.copy()
    if official_only and "official_eligible" in work.columns:
        work = work[work["official_eligible"] == True]  # noqa: E712
    if wanted and "config_path" in work.columns:
        work = work[work["config_path"].astype(str).isin(wanted)]
    if work.empty:
        return []

    latest = (
        work.sort_values("run_id", ascending=False)
        .drop_duplicates(subset=["config_path"], keep="first")
        .reset_index(drop=True)
    )
    return latest["run_id"].astype(str).tolist()


def find_latest_replay_summary(
    artifacts_root: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path | None:
    replay_root = Path(artifacts_root).resolve() / "benchmarks" / "replay"
    if not replay_root.exists():
        return None

    candidates = sorted(replay_root.glob("*/replay_benchmark_summary.csv"))
    if start_date and end_date:
        token = f"__{start_date}__{end_date}"
        candidates = [path for path in candidates if token in path.parent.name]
    return candidates[-1] if candidates else None


def load_replay_summary(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Date" in frame.columns:
        frame = coerce_datetime(frame, "Date")
    return frame


def normalize_replay_summary(
    summary_df: pd.DataFrame,
    *,
    runs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    work = summary_df.copy()
    if "config_name" not in work.columns:
        if "config_path" in work.columns:
            work["config_name"] = work["config_path"].map(_config_name)
        else:
            work["config_name"] = pd.NA
    else:
        work["config_name"] = work["config_name"].map(
            lambda value: Path(str(value)).stem if pd.notna(value) else value
        )

    for column in ["n_requested_days", "n_forecasted_days", "n_skipped_days", "MAE", "RMSE"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    if {"n_skipped_days", "n_requested_days"}.issubset(work.columns):
        work["skip_rate_pct"] = (
            100.0 * work["n_skipped_days"].fillna(0.0) / work["n_requested_days"].replace(0, pd.NA)
        )

    if "forecast_mode" in work.columns:
        work["official_eligible"] = work["forecast_mode"].astype(str).eq(STRICT_DAY_AHEAD_MODE)
    else:
        work["official_eligible"] = False
    work["human_label"] = [
        _display_label(
            experiment_name=experiment_name if pd.notna(experiment_name) else None,
            config_name=config_name if pd.notna(config_name) else None,
            run_id=run_id if pd.notna(run_id) else None,
        )
        for experiment_name, config_name, run_id in zip(
            work.get("experiment_name", pd.Series([None] * len(work))),
            work.get("config_name", pd.Series([None] * len(work))),
            work.get("requested_model_run_id", pd.Series([None] * len(work))),
            strict=False,
        )
    ]

    if runs_df is not None and not runs_df.empty and "requested_model_run_id" in work.columns:
        offline_cols = [
            "run_id",
            "offline_MAE",
            "offline_RMSE",
            "offline_MAPE%",
            "offline_InTolerance%",
            "offline_RampingError_RMSE",
        ]
        offline = runs_df[[col for col in offline_cols if col in runs_df.columns]].copy()
        work = work.merge(
            offline,
            left_on="requested_model_run_id",
            right_on="run_id",
            how="left",
            suffixes=("", "_offline"),
        )
        if "run_id" in work.columns:
            work = work.drop(columns=["run_id"])

    sort_cols = [col for col in ["MAE", "RMSE"] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=True).reset_index(drop=True)
    return work


def build_model_label_map(
    run_ids: Sequence[str],
    *,
    runs_df: pd.DataFrame | None = None,
    replay_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    metadata: dict[str, str] = {}

    if runs_df is not None and not runs_df.empty and "run_id" in runs_df.columns:
        for row in runs_df.to_dict("records"):
            metadata[str(row["run_id"])] = _display_label(
                experiment_name=row.get("experiment_name"),
                config_name=row.get("config_name"),
                run_id=row.get("run_id"),
            )

    if replay_df is not None and not replay_df.empty and "requested_model_run_id" in replay_df.columns:
        for row in replay_df.to_dict("records"):
            metadata[str(row["requested_model_run_id"])] = _display_label(
                experiment_name=row.get("experiment_name"),
                config_name=row.get("config_name"),
                run_id=row.get("requested_model_run_id"),
            )

    label_map: dict[str, str] = {}
    seen: dict[str, int] = {}
    for run_id in run_ids:
        run_key = str(run_id)
        base_label = metadata.get(run_key, run_key)
        seen[base_label] = seen.get(base_label, 0) + 1
        if seen[base_label] == 1:
            label_map[run_key] = base_label
        else:
            label_map[run_key] = f"{base_label} [{run_key[-8:]}]"
    return label_map


def build_skipped_days_audit(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []

    if summary_df.empty or "metrics_json" not in summary_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    for row in summary_df.to_dict("records"):
        metrics_json = row.get("metrics_json")
        if not metrics_json or not Path(str(metrics_json)).exists():
            continue
        metrics = read_json(metrics_json)
        skipped_days = list(metrics.get("skipped_days") or [])
        for skipped in skipped_days:
            detail_rows.append(
                {
                    "requested_model_run_id": row.get("requested_model_run_id"),
                    "human_label": row.get("human_label"),
                    "config_name": row.get("config_name"),
                    "experiment_name": row.get("experiment_name"),
                    "target_date": skipped.get("target_date"),
                    "reason": skipped.get("reason"),
                    "metrics_json": metrics_json,
                    "n_requested_days": metrics.get("n_requested_days"),
                    "n_forecasted_days": metrics.get("n_forecasted_days"),
                    "n_skipped_days": metrics.get("n_skipped_days"),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        empty_counts = pd.DataFrame(
            columns=["target_date", "reason", "n_models_skipping", "models"]
        )
        return detail_df, empty_counts

    detail_df = detail_df.sort_values(["target_date", "human_label"]).reset_index(drop=True)
    counts_df = (
        detail_df.groupby(["target_date", "reason"], dropna=False)
        .agg(
            n_models_skipping=("requested_model_run_id", "nunique"),
            models=("human_label", lambda values: " | ".join(sorted({str(v) for v in values}))),
        )
        .reset_index()
        .sort_values(["target_date", "n_models_skipping", "reason"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return detail_df, counts_df


def coerce_datetime(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return out


def slice_date_range(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    *,
    date_col: str = "Date",
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df.copy()

    out = coerce_datetime(df, date_col=date_col)
    if start_date:
        start_ts = pd.Timestamp(start_date).normalize()
        out = out[out[date_col] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
        out = out[out[date_col] < end_ts]
    return out.reset_index(drop=True)


def slice_single_day(df: pd.DataFrame, target_date: str, *, date_col: str = "Date") -> pd.DataFrame:
    return slice_date_range(df, target_date, target_date, date_col=date_col)


def load_or_run_long_sample_predictions(
    *,
    root: str | Path,
    artifacts_root: str | Path,
    export_root: str | Path,
    run_id: str,
    start_date: str,
    end_date: str,
    dataset_key: str | None = None,
    catalog_path: str | Path | None = None,
    historical_csv: str | Path | None = None,
    weather_csv: str | Path | None = None,
    holidays_xlsx: str | Path | None = None,
    benchmark_csv: str | Path | None = None,
    allow_fallback: bool = False,
    force_recompute: bool = False,
    use_cache: bool = True,
    continue_on_error: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(root).resolve()
    artifacts_path = Path(artifacts_root).resolve()
    export_path = Path(export_root).resolve()
    run_dir = artifacts_path / "runs" / "consumption" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found for {run_id}: {run_dir}")

    cache_dir = ensure_dir(export_path / "long_sample_predict" / run_id / f"{start_date}__{end_date}")
    per_day_dir = ensure_dir(cache_dir / "per_day")
    output_csv = cache_dir / "predict_long_sample.csv"
    metadata_json = cache_dir / "metadata.json"

    if use_cache and not force_recompute and output_csv.exists():
        cached = pd.read_csv(output_csv)
        cached = coerce_datetime(cached, "Date")
        metadata = read_json(metadata_json) if metadata_json.exists() else {
            "run_id": run_id,
            "output_csv": str(output_csv.resolve()),
            "cache_dir": str(cache_dir.resolve()),
            "source": "cache_only",
        }
        return cached, metadata

    days = pd.date_range(start_date, end_date, freq="D")
    frames: list[pd.DataFrame] = []
    day_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for day in days:
        target_date = str(day.date())
        custom_output = per_day_dir / f"forecast_{target_date}.csv"
        cmd = [
            "python",
            "scripts/predict_next_day.py",
            "--current-dir",
            str(run_dir),
            "--target-date",
            target_date,
            "--output-csv",
            str(custom_output),
        ] + optional_cli_args(
            dataset_key=dataset_key,
            catalog_path=catalog_path,
            historical_csv=historical_csv,
            weather_csv=weather_csv,
            holidays_xlsx=holidays_xlsx,
            benchmark_csv=benchmark_csv,
            allow_fallback=allow_fallback,
        )

        result = run_cli(cmd, cwd=root_path, check=not continue_on_error)
        if result.returncode != 0:
            failure = {
                "target_date": target_date,
                "returncode": result.returncode,
                "stderr": result.stderr,
            }
            failures.append(failure)
            if continue_on_error:
                continue
            raise RuntimeError(
                f"Long-sample prediction failed for {target_date}\n"
                f"CMD: {result.command_text}\n"
                f"STDERR:\n{result.stderr}"
            )

        payload = result.extract_json()
        day_frame = pd.read_csv(custom_output)
        day_frame = coerce_datetime(day_frame, "Date")
        day_frame["requested_run_id"] = run_id
        day_frame["requested_target_date"] = target_date
        frames.append(day_frame)
        day_results.append(
            {
                "target_date": target_date,
                "points": int(len(day_frame)),
                "custom_output_csv": str(custom_output.resolve()),
                "archive_output_csv": payload.get("archive_output_csv"),
                "current_output_csv": payload.get("current_output_csv"),
            }
        )

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = coerce_datetime(combined, "Date")
    else:
        combined = pd.DataFrame(
            columns=[
                "Date",
                "Ptot_TOTAL_Forecast",
                "model_run_id",
                "generated_at",
                "target_date",
                "dataset_key",
                "forecast_mode",
                "Ptot_TOTAL_Real",
                "requested_run_id",
                "requested_target_date",
            ]
        )

    combined.to_csv(output_csv, index=False)
    metadata = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "start_date": start_date,
        "end_date": end_date,
        "n_days_requested": int(len(days)),
        "n_days_completed": int(len(day_results)),
        "n_days_failed": int(len(failures)),
        "output_csv": str(output_csv.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "allow_fallback": bool(allow_fallback),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "day_results": day_results,
        "failed_days": failures,
    }
    write_json(metadata_json, metadata)
    return combined, metadata


def build_truth_baseline_frame(
    history_df: pd.DataFrame,
    *,
    date_col: str = "Date",
    real_col: str = "tot",
) -> pd.DataFrame:
    truth = history_df[[date_col, real_col]].copy()
    truth = truth.rename(columns={date_col: "Date", real_col: "real"})
    truth = coerce_datetime(truth, "Date")
    truth = truth.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    indexed = truth.set_index("Date")
    truth["weekly_naive"] = seasonal_naive_weekly(indexed["real"], lag="7D").to_numpy()
    return truth


def prepare_legacy_forecast_frame(
    legacy_df: pd.DataFrame | None,
    *,
    coverage_end_date: str | None,
    date_col: str = "Date",
    legacy_col: str = "OldLegacy_TOTAL_Forecast",
) -> pd.DataFrame:
    if legacy_df is None or legacy_df.empty:
        return pd.DataFrame(columns=["Date", "legacy_forecast"])

    out = legacy_df[[date_col, legacy_col]].copy()
    out = out.rename(columns={date_col: "Date", legacy_col: "legacy_forecast"})
    out = coerce_datetime(out, "Date")

    if coverage_end_date:
        end_ts = pd.Timestamp(coverage_end_date).normalize() + pd.Timedelta(days=1)
        out = out[out["Date"] < end_ts]

    out = out.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return out


def build_wide_comparison_frame(
    *,
    truth_baseline_df: pd.DataFrame,
    model_frames: Mapping[str, pd.DataFrame],
    label_map: Mapping[str, str],
    start_date: str,
    end_date: str,
    legacy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = slice_date_range(truth_baseline_df, start_date, end_date, date_col="Date")
    base = base.copy()

    if legacy_df is not None and not legacy_df.empty:
        legacy_slice = slice_date_range(legacy_df, start_date, end_date, date_col="Date")
        base = base.merge(legacy_slice, on="Date", how="left")
    elif "legacy_forecast" not in base.columns:
        base["legacy_forecast"] = pd.NA

    for run_id, frame in model_frames.items():
        if frame is None or frame.empty:
            continue
        label = str(label_map.get(run_id, run_id))
        work = slice_date_range(frame, start_date, end_date, date_col="Date")
        if "Ptot_TOTAL_Forecast" not in work.columns:
            continue
        keep = ["Date", "Ptot_TOTAL_Forecast"]
        if "Ptot_TOTAL_Real" in work.columns and "real" not in base.columns:
            keep.append("Ptot_TOTAL_Real")
        work = work[keep].drop_duplicates(subset=["Date"], keep="last").rename(
            columns={"Ptot_TOTAL_Forecast": label, "Ptot_TOTAL_Real": "real"}
        )
        base = base.merge(work, on="Date", how="left")

    return coerce_datetime(base, "Date")
