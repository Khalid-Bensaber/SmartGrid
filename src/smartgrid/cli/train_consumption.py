from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from smartgrid.common.constants import DEFAULT_TARGET_NAME
from smartgrid.common.logging import build_log_path, setup_logger
from smartgrid.common.paths import build_consumption_paths
from smartgrid.common.utils import get_device, load_yaml, parse_hidden_layers, utc_run_id
from smartgrid.data.catalog import resolve_consumption_data_config
from smartgrid.data.loaders import (
    load_history,
    load_holiday_sets,
    load_old_benchmark,
    load_weather_history,
    merge_weather_on_history,
)
from smartgrid.data.splits import make_splits
from smartgrid.evaluation.metrics import compute_basic_metrics
from smartgrid.evaluation.reporting import (
    build_backtest_outputs,
    evaluate_backtest,
    make_notebook_export_legacy_schema,
    make_total_export,
    pick_analysis_day,
    save_json,
)
from smartgrid.features.engineering import build_feature_table
from smartgrid.training.artifacts import promote_bundle, save_training_bundle
from smartgrid.training.trainer import predict_model, train_mlp_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the consumption MLP pipeline")
    parser.add_argument("--config", default="configs/consumption/mlp_baseline.yaml")
    parser.add_argument("--analysis-date", default=None)
    parser.add_argument("--analysis-days", type=int, default=1)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--dataset-key", default=None)
    parser.add_argument("--catalog-path", default=None)
    parser.add_argument("--historical-csv", default=None)
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument("--weather-csv", default=None)
    parser.add_argument("--holidays-xlsx", default=None)
    parser.add_argument("--promote", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    data_cfg = resolve_consumption_data_config(
        config["data"],
        dataset_key=args.dataset_key,
        catalog_path=args.catalog_path,
        overrides={
            "historical_csv": args.historical_csv,
            "benchmark_csv": args.benchmark_csv,
            "weather_csv": args.weather_csv,
            "holidays_xlsx": args.holidays_xlsx,
        },
    )
    split_cfg = config["split"]
    feat_cfg = config["features"]
    train_cfg = config["training"]
    artifacts_cfg = config["artifacts"]
    target_col = data_cfg.get("target_name", DEFAULT_TARGET_NAME)

    run_id = utc_run_id("consumption_mlp")
    logger = setup_logger(
        "smartgrid.train",
        log_file=build_log_path(artifacts_cfg["root_dir"], "train", f"{run_id}.log"),
    )
    paths = build_consumption_paths(
        root_dir=artifacts_cfg["root_dir"],
        exports_subdir=artifacts_cfg["exports_subdir"],
        registry_subdir=artifacts_cfg["registry_subdir"],
        run_id=run_id,
    )

    device = get_device(train_cfg.get("device", "auto"))
    hidden_layers = parse_hidden_layers(train_cfg["hidden_layers"])
    logger.info("Starting consumption training run_id=%s config=%s device=%s", run_id, args.config, device)

    holiday_dates, special_dates = load_holiday_sets(data_cfg["holidays_xlsx"])
    hist = load_history(
        data_cfg["historical_csv"],
        date_col=data_cfg["date_col"],
        target_col=target_col,
    )
    weather = load_weather_history(data_cfg.get("weather_csv"), date_col=data_cfg["date_col"])
    hist = merge_weather_on_history(hist, weather, date_col=data_cfg["date_col"])

    feat_df, feature_cols = build_feature_table(
        hist_df=hist,
        holiday_dates=holiday_dates,
        special_dates=special_dates,
        date_col=data_cfg["date_col"],
        target_col=target_col,
        lag_days=feat_cfg.get("lag_days", [7, 1, 2, 3, 4, 5, 6]),
        include_calendar=feat_cfg.get("include_calendar", True),
        include_temperature=feat_cfg.get("include_temperature", True),
        include_manual_daily_lags=feat_cfg.get("include_manual_daily_lags", True),
        include_cyclical_time=feat_cfg.get("include_cyclical_time", False),
        include_lag_aggregates=feat_cfg.get("include_lag_aggregates", False),
        include_recent_dynamics=feat_cfg.get("include_recent_dynamics", False),
        include_weather=feat_cfg.get("include_weather", False),
        weather_mode=feat_cfg.get("weather_mode"),
        weather_columns=feat_cfg.get("weather_columns"),
    )
    logger.info(
        "Prepared feature table rows=%s n_features=%s feature_columns=%s",
        len(feat_df),
        len(feature_cols),
        feature_cols,
    )
    
    train_df, val_df, test_df = make_splits(
        feat_df,
        date_col=data_cfg["date_col"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        train_end_date=split_cfg.get("train_end_date"),
        val_end_date=split_cfg.get("val_end_date"),
    )
    logger.info(
        "Split dataset train=%s val=%s test=%s",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[[target_col]].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df[[target_col]].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)
    X_test_scaled = x_scaler.transform(X_test)

    train_start = time.time()
    train_result = train_mlp_regressor(
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        X_val=X_val_scaled,
        y_val=y_val_scaled,
        X_test=X_test_scaled,
        hidden_layers=hidden_layers,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        patience=train_cfg["patience"],
        dropout=train_cfg["dropout"],
        seed=train_cfg["seed"],
        num_workers=train_cfg["num_workers"],
        device=device,
        resume_checkpoint=args.resume_checkpoint,
        logger=logger,
    )
    train_duration_sec = time.time() - train_start

    predictions = predict_model(train_result.model, train_result.test_x, y_scaler, device)
    basic_metrics = compute_basic_metrics(y_test, predictions)

    benchmark = load_old_benchmark(data_cfg.get("benchmark_csv"), date_col=data_cfg["date_col"])
    backtest = build_backtest_outputs(
        test_df=test_df,
        date_col=data_cfg["date_col"],
        predictions=predictions,
        benchmark=benchmark,
        target_col=target_col,
    )
    evaluation = evaluate_backtest(
        backtest=backtest,
        date_col=data_cfg["date_col"],
        target_col=target_col,
    )

    analysis_day = pick_analysis_day(backtest, benchmark, data_cfg["date_col"], args.analysis_date)
    start_day = np.datetime64(analysis_day)
    end_day = start_day + np.timedelta64(args.analysis_days, "D")
    mask = (backtest[data_cfg["date_col"]] >= start_day) & (backtest[data_cfg["date_col"]] < end_day)
    day_df = backtest.loc[mask].copy()

    notebook_export = make_notebook_export_legacy_schema(day_df, data_cfg["date_col"])
    notebook_export_path = paths.exports_dir / artifacts_cfg["notebook_output_filename"]
    notebook_export.to_csv(notebook_export_path, index=False)

    total_export = make_total_export(day_df, data_cfg["date_col"], target_col=target_col)
    total_export_path = paths.exports_dir / "total_forecast_consumption.csv"
    total_export.to_csv(total_export_path, index=False)

    backtest_path = paths.exports_dir / "backtest.csv"
    selected_day_path = paths.exports_dir / f"selected_day_{analysis_day}.csv"
    backtest.to_csv(backtest_path, index=False)
    day_df.to_csv(selected_day_path, index=False)
    epochs_ran = len(train_result.history["train_loss"])
    best_val_loss = float(min(train_result.history["val_loss"])) if train_result.history["val_loss"] else None
    final_train_loss = float(train_result.history["train_loss"][-1]) if train_result.history["train_loss"] else None
    final_val_loss = float(train_result.history["val_loss"][-1]) if train_result.history["val_loss"] else None

    summary = {
        "run_id": run_id,
        "problem": "consumption",
        "experiment_name": config.get("experiment_name"),
        "backend": "pytorch",
        "device": str(device),
        "dataset_key": data_cfg.get("dataset_key"),
        "dataset_description": data_cfg.get("dataset_description"),
        "catalog_path": data_cfg.get("catalog_path"),
        "aliases": data_cfg.get("aliases") or {},
        "date_col": data_cfg["date_col"],
        "historical_csv": str(Path(data_cfg["historical_csv"]).resolve()),
        "holidays_xlsx": str(Path(data_cfg["holidays_xlsx"]).resolve()),
        "weather_csv": str(Path(data_cfg["weather_csv"]).resolve()) if data_cfg.get("weather_csv") else None,
        "benchmark_csv": str(Path(data_cfg["benchmark_csv"]).resolve()) if data_cfg.get("benchmark_csv") else None,
        "target_column": target_col,
        "feature_columns": feature_cols,
        "feature_config": feat_cfg,
        "hidden_layers": list(hidden_layers),
        "n_features": int(len(feature_cols)),
        "train_duration_sec": train_duration_sec,
        "epochs_ran": epochs_ran,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "metrics_basic": basic_metrics,
        **evaluation,
        "n_total_rows": int(len(feat_df)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "train_date_min": str(train_df[data_cfg["date_col"]].min()),
        "train_date_max": str(train_df[data_cfg["date_col"]].max()),
        "val_date_min": str(val_df[data_cfg["date_col"]].min()),
        "val_date_max": str(val_df[data_cfg["date_col"]].max()),
        "test_date_min": str(test_df[data_cfg["date_col"]].min()),
        "test_date_max": str(test_df[data_cfg["date_col"]].max()),
        "selected_analysis_day": str(analysis_day),
        "analysis_days": args.analysis_days,
        "config_path": str(Path(args.config).resolve()),
        "history": train_result.history,
        "run_dir": str(paths.run_dir.resolve()),
        "exports_dir": str(paths.exports_dir.resolve()),
        "backtest_csv": str(backtest_path.resolve()),
        "day_compare_csv": str(selected_day_path.resolve()),
        "output_total_csv": str(total_export_path.resolve()),
        "output_notebook_csv": str(notebook_export_path.resolve()),
        "export_roles": {
            "backtest_csv": "full test-period evaluation output",
            "day_compare_csv": "selected analysis day extracted from the backtest",
            "output_total_csv": "compact selected-day comparison export",
            "output_notebook_csv": "legacy notebook compatibility export",
        },
    }

    train_result.model_config["feature_columns"] = feature_cols
    save_training_bundle(
        model=train_result.model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        run_dir=paths.run_dir,
        model_config=train_result.model_config,
        run_summary=summary,
    )

    save_json(paths.exports_dir / artifacts_cfg["summary_filename"], summary)
    logger.info(
        "Wrote outputs run_dir=%s exports_dir=%s backtest_csv=%s selected_day_csv=%s",
        paths.run_dir,
        paths.exports_dir,
        backtest_path,
        selected_day_path,
    )
    if args.promote:
        promote_bundle(paths.run_dir, paths.registry_current_dir)
        logger.info("Promoted run to current registry: %s", paths.registry_current_dir)

    payload = {
        "run_id": run_id,
        "run_dir": str(paths.run_dir.resolve()),
        "exports_dir": str(paths.exports_dir.resolve()),
        "promoted": bool(args.promote),
        "config": str(Path(args.config)),
        "experiment_name": config.get("experiment_name"),
        "selected_analysis_day": str(analysis_day),
        "train_duration_sec": train_duration_sec,
        "n_features": int(len(feature_cols)),
        "epochs_ran": epochs_ran,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "feature_config": feat_cfg,
        "metrics_model": evaluation["metrics_model"],
    }
    logger.info("Training run completed successfully run_id=%s", run_id)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
