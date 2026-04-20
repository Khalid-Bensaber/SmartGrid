from __future__ import annotations

import importlib
import time

from fastapi.testclient import TestClient

from smartgrid.api import app as fastapi_app


app_module = importlib.import_module("smartgrid.api.app")
client = TestClient(fastapi_app)


def test_root_and_health() -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["message"] == "SmartGrid API running"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}


def test_model_info_and_models_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "get_consumption_model_info",
        lambda **kwargs: {
            "model_config": {"hidden_layers": [128, 64]},
            "latest_summary": {"run_id": "run_123"},
            "dataset_key": "demo",
            "forecast_mode": "strict_day_ahead",
            "model_run_id": "run_123",
            "current_dir": "/tmp/current",
        },
    )
    monkeypatch.setattr(
        app_module,
        "list_consumption_models",
        lambda **kwargs: {
            "current_dir": "/tmp/current",
            "artifacts_root": "/tmp/artifacts",
            "models": [
                {
                    "bundle_dir": "/tmp/current",
                    "run_id": "run_123",
                    "experiment_name": "demo",
                    "dataset_key": "demo",
                    "forecast_mode": "strict_day_ahead",
                    "config_path": "configs/demo.yaml",
                    "is_current": True,
                }
            ],
        },
    )

    response = client.get("/consumption/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_config"]["hidden_layers"] == [128, 64]
    assert payload["model_run_id"] == "run_123"

    models = client.get("/consumption/models")
    assert models.status_code == 200
    assert models.json()["models"][0]["is_current"] is True


def test_forecast_and_replay_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "run_consumption_forecast",
        lambda **kwargs: {
            "target_date": "2026-01-15",
            "points": [
                {
                    "Date": "2026-01-15 00:00:00",
                    "Ptot_TOTAL_Forecast": 12345.0,
                    "Ptot_TOTAL_Real": None,
                }
            ],
            "model_run_id": "run_123",
            "requested_model_run_id": "run_123",
            "dataset_key": "demo",
            "forecast_mode": "strict_day_ahead",
            "fallback_used": False,
            "current_output_csv": "/tmp/current.csv",
            "archive_output_csv": "/tmp/archive.csv",
            "custom_output_csv": None,
        },
    )
    monkeypatch.setattr(
        app_module,
        "run_consumption_replay",
        lambda **kwargs: {
            "start_date": "2026-01-01",
            "end_date": "2026-01-03",
            "points": [
                {
                    "Date": "2026-01-01 00:00:00",
                    "Ptot_TOTAL_Forecast": 12345.0,
                    "Ptot_TOTAL_Real": 12000.0,
                }
            ],
            "requested_model_run_id": "run_123",
            "effective_model_run_ids": ["run_123"],
            "dataset_key": "demo",
            "forecast_mode": "strict_day_ahead",
            "overall_metrics": {"MAE": 10.0},
            "skipped_days": [],
            "fallback_used": False,
            "output_csv": "/tmp/replay.csv",
            "metrics_json": "/tmp/replay.json",
            "per_day_dir": "/tmp/per_day",
            "n_days": 3,
            "n_rows": 144,
            "n_requested_days": 3,
            "n_forecasted_days": 3,
            "n_skipped_days": 0,
            "per_day_metrics": [],
            "per_day_model_usage": [],
        },
    )

    forecast = client.post("/consumption/forecast/next-day", json={})
    assert forecast.status_code == 200
    assert forecast.json()["current_output_csv"] == "/tmp/current.csv"

    replay = client.post(
        "/consumption/replay",
        json={"start_date": "2026-01-01", "end_date": "2026-01-03"},
    )
    assert replay.status_code == 200
    assert replay.json()["overall_metrics"]["MAE"] == 10.0


def test_train_and_promote_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "run_consumption_training",
        lambda **kwargs: {
            "run_id": "run_123",
            "run_dir": "/tmp/run",
            "exports_dir": "/tmp/exports",
            "promoted": False,
            "config": "configs/demo.yaml",
            "experiment_name": "demo",
            "selected_analysis_day": "2026-01-15",
            "train_duration_sec": 1.2,
            "n_features": 12,
            "epochs_ran": 3,
            "best_val_loss": 0.1,
            "final_train_loss": 0.2,
            "final_val_loss": 0.3,
            "n_train_rows": 10,
            "n_val_rows": 5,
            "n_test_rows": 5,
            "feature_config": {"forecast_mode": "strict_day_ahead"},
            "forecast_mode": "strict_day_ahead",
            "dataset_key": "demo",
            "device": "cpu",
            "runtime_diagnostics": {"selected_device": "cpu"},
            "batching_strategy": "auto",
            "resident_data_bytes": 1024,
            "test_date_min": "2026-01-01 00:00:00",
            "test_date_max": "2026-01-02 00:00:00",
            "offline_test_metrics": {"MAE": 1.0},
            "metrics_model": {"MAE": 1.0},
            "profiling_enabled": False,
        },
    )
    monkeypatch.setattr(
        app_module,
        "run_consumption_promote",
        lambda **kwargs: {
            "run_id": "run_123",
            "run_dir": "/tmp/run",
            "current_dir": "/tmp/current",
            "promoted": True,
        },
    )

    train = client.post("/consumption/train", json={})
    assert train.status_code == 200
    assert train.json()["run_id"] == "run_123"

    promote = client.post("/consumption/promote", json={"run_id": "run_123"})
    assert promote.status_code == 200
    assert promote.json()["promoted"] is True


def test_async_job_lifecycle(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "run_consumption_training",
        lambda **kwargs: {
            "run_id": "run_async",
            "run_dir": "/tmp/run_async",
            "exports_dir": "/tmp/exports_async",
            "promoted": False,
            "config": "configs/demo.yaml",
            "experiment_name": "demo",
            "selected_analysis_day": "2026-01-15",
            "train_duration_sec": 0.1,
            "n_features": 4,
            "epochs_ran": 1,
            "best_val_loss": 0.1,
            "final_train_loss": 0.1,
            "final_val_loss": 0.1,
            "n_train_rows": 1,
            "n_val_rows": 1,
            "n_test_rows": 1,
            "feature_config": {"forecast_mode": "strict_day_ahead"},
            "forecast_mode": "strict_day_ahead",
            "dataset_key": "demo",
            "device": "cpu",
            "runtime_diagnostics": {"selected_device": "cpu"},
            "batching_strategy": "auto",
            "resident_data_bytes": 1,
            "test_date_min": "2026-01-01 00:00:00",
            "test_date_max": "2026-01-01 23:50:00",
            "offline_test_metrics": {"MAE": 1.0},
            "metrics_model": {"MAE": 1.0},
            "profiling_enabled": False,
        },
    )

    launch = client.post("/consumption/train/async", json={})
    assert launch.status_code == 200
    job_id = launch.json()["job_id"]

    result = None
    for _ in range(20):
        response = client.get(f"/jobs/{job_id}/result")
        assert response.status_code == 200
        result = response.json()
        if result["status"] == "succeeded":
            break
        time.sleep(0.05)

    assert result is not None
    assert result["status"] == "succeeded"
    assert result["result"]["run_id"] == "run_async"
