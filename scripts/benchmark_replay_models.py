from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smartgrid.common.constants import STRICT_DAY_AHEAD_MODE
from smartgrid.common.logging import build_log_path, setup_logger
from smartgrid.common.utils import ensure_dir
from smartgrid.evaluation.reporting import evaluate_forecast_frame, save_json
from smartgrid.inference.day_ahead import build_forecast_runtime, replay_forecast_period


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark several registered models on a replay period"
    )
    parser.add_argument("--historical-csv", default=None)
    parser.add_argument("--dataset-key", default=None)
    parser.add_argument("--catalog-path", default=None)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--weather-csv", default=None)
    parser.add_argument("--holidays-xlsx", default=None)
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument(
        "model_refs",
        nargs="+",
        help=(
            "Run ids or bundle directories to benchmark, "
            "e.g. consumption_mlp_... or artifacts/runs/consumption/<run_id>"
        ),
    )
    return parser.parse_args()


def resolve_bundle_dir(model_ref: str, artifacts_root: Path) -> Path:
    raw = Path(model_ref)
    if raw.exists():
        return raw

    candidate = artifacts_root / "runs" / "consumption" / model_ref
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Unable to resolve model ref: {model_ref}")


def resolve_bundle_summary(bundle_dir: Path) -> dict:
    for candidate in [bundle_dir / "run_summary.json", bundle_dir / "summary.json"]:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def resolve_bundle_forecast_mode(bundle_dir: Path) -> str | None:
    summary = resolve_bundle_summary(bundle_dir)
    feature_cfg = summary.get("feature_config") or {}
    forecast_mode = summary.get("forecast_mode") or feature_cfg.get("forecast_mode")
    return str(forecast_mode) if forecast_mode else None


def build_model_metadata(runtime, bundle_dir: Path) -> dict:
    summary = runtime.bundle.summary or {}
    config_path = summary.get("config_path")
    config_name = Path(config_path).name if config_path else None
    feature_cfg = summary.get("feature_config") or {}
    hidden_layers = summary.get("hidden_layers") or []
    return {
        "config_path": config_path,
        "config_name": config_name,
        "experiment_name": summary.get("experiment_name"),
        "dataset_key": summary.get("dataset_key"),
        "feature_config": feature_cfg,
        "forecast_mode": summary.get("forecast_mode") or feature_cfg.get("forecast_mode"),
        "feature_columns": summary.get("feature_columns"),
        "n_features": summary.get("n_features"),
        "hidden_layers": hidden_layers,
        "bundle_dir": str(bundle_dir.resolve()),
    }


def main() -> None:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = setup_logger(
        "smartgrid.replay_benchmark",
        log_file=build_log_path(
            artifacts_root,
            "replay",
            f"{stamp}__benchmark_replay__{args.start_date}__{args.end_date}.log",
        ),
    )

    out_dir = ensure_dir(
        artifacts_root / "benchmarks" / "replay" / f"{stamp}__{args.start_date}__{args.end_date}"
    )
    all_days = pd.date_range(args.start_date, args.end_date, freq="D")
    summary_rows = []
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "allow_fallback": bool(args.allow_fallback),
        "models": [],
        "skipped_models": [],
    }

    for model_ref in args.model_refs:
        bundle_dir = resolve_bundle_dir(model_ref, artifacts_root)
        requested_run_id = bundle_dir.name
        forecast_mode = resolve_bundle_forecast_mode(bundle_dir)
        if forecast_mode not in [None, STRICT_DAY_AHEAD_MODE]:
            logger.warning(
                "Skipping requested_model_run_id=%s bundle_dir=%s forecast_mode=%s: "
                "replay benchmark only supports %s",
                requested_run_id,
                bundle_dir,
                forecast_mode,
                STRICT_DAY_AHEAD_MODE,
            )
            manifest["skipped_models"].append(
                {
                    "requested_model_run_id": requested_run_id,
                    "bundle_dir": str(bundle_dir.resolve()),
                    "forecast_mode": forecast_mode,
                    "skip_reason": f"unsupported forecast_mode for replay benchmark: {forecast_mode}",
                }
            )
            continue
        runtime = build_forecast_runtime(
            historical_csv=args.historical_csv,
            current_dir=bundle_dir,
            artifacts_root=artifacts_root,
            dataset_key=args.dataset_key,
            catalog_path=args.catalog_path,
            weather_csv=args.weather_csv,
            holidays_xlsx=args.holidays_xlsx,
            device_request=args.device,
            benchmark_csv=args.benchmark_csv,
            allow_fallback=args.allow_fallback,
            logger=logger,
        )
        requested_run_id = (runtime.bundle.summary or {}).get("run_id", bundle_dir.name)
        model_metadata = build_model_metadata(runtime, bundle_dir)
        logger.info(
            "Benchmarking replay for requested_model_run_id=%s bundle_dir=%s",
            requested_run_id,
            bundle_dir,
        )

        model_dir = ensure_dir(out_dir / requested_run_id)
        replay = replay_forecast_period(runtime, args.start_date, args.end_date, logger=logger)
        replay_df = replay.replay_df
        skipped_days = replay.skipped_days
        replay_csv = model_dir / "replay_forecasts.csv"
        replay_df.to_csv(replay_csv, index=False)

        overall_metrics = evaluate_forecast_frame(replay_df)
        per_day_metrics = []
        per_day_model_usage = []
        effective_model_run_ids = replay.effective_model_run_ids
        fallback_used = replay.fallback_used

        if not replay_df.empty:
            for target_date, day_df in replay_df.groupby("target_date", dropna=False):
                day_metrics = evaluate_forecast_frame(day_df)
                if day_metrics is not None:
                    per_day_metrics.append({"target_date": target_date, **day_metrics})
                per_day_model_usage.append(
                    {
                        "target_date": target_date,
                        "model_run_ids": sorted(
                            str(x) for x in day_df["model_run_id"].dropna().unique().tolist()
                        ),
                    }
                )

        metrics_payload = {
            "requested_model_run_id": requested_run_id,
            "effective_model_run_ids": effective_model_run_ids,
            "fallback_enabled": bool(args.allow_fallback),
            "fallback_used": fallback_used,
            "n_requested_days": int(len(all_days)),
            "n_forecasted_days": int(len(all_days) - len(skipped_days)),
            "n_skipped_days": int(len(skipped_days)),
            "skipped_days": skipped_days,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "n_days": int(len(all_days)),
            "n_rows": int(len(replay_df)),
            "dataset_key": runtime.data_config.get("dataset_key"),
            "forecast_mode": runtime.forecast_mode,
            "overall_metrics": overall_metrics,
            "per_day_metrics": per_day_metrics,
            "per_day_model_usage": per_day_model_usage,
            "replay_csv": str(replay_csv.resolve()),
            **model_metadata,
        }
        metrics_json = model_dir / "replay_metrics.json"
        save_json(metrics_json, metrics_payload)

        summary_rows.append(
            {
                "requested_model_run_id": requested_run_id,
                "effective_model_run_ids": "|".join(effective_model_run_ids),
                "fallback_enabled": bool(args.allow_fallback),
                "fallback_used": fallback_used,
                "n_requested_days": int(len(all_days)),
                "n_forecasted_days": int(len(all_days) - len(skipped_days)),
                "n_skipped_days": int(len(skipped_days)),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "n_days": int(len(all_days)),
                "n_rows": int(len(replay_df)),
                "replay_csv": str(replay_csv.resolve()),
                "metrics_json": str(metrics_json.resolve()),
                **model_metadata,
                **(overall_metrics or {}),
            }
        )
        manifest["models"].append(
            {
                "requested_model_run_id": requested_run_id,
                "replay_csv": str(replay_csv.resolve()),
                "metrics_json": str(metrics_json.resolve()),
                **model_metadata,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        sort_cols = [col for col in ["MAE", "RMSE"] if col in summary_df.columns]
        if sort_cols:
            summary_df = summary_df.sort_values(sort_cols, ascending=True)
    summary_csv = out_dir / "replay_benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    manifest_path = out_dir / "replay_benchmark_manifest.json"
    save_json(manifest_path, manifest)

    logger.info("Replay benchmark written summary_csv=%s manifest=%s", summary_csv, manifest_path)
    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv.resolve()),
                "manifest_json": str(manifest_path.resolve()),
                "n_models": int(len(summary_df)),
                "start_date": args.start_date,
                "end_date": args.end_date,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
