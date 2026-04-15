from pathlib import Path

import pytest

from smartgrid.data.catalog import list_consumption_dataset_keys, resolve_consumption_data_config


def test_default_catalog_exposes_expected_dataset_keys():
    keys = list_consumption_dataset_keys()
    assert "clean_v1" in keys
    assert "legacy_2020_2025" in keys


def test_resolve_consumption_data_config_uses_catalog_paths():
    data_cfg = resolve_consumption_data_config({"dataset_key": "clean_v1", "date_col": "Date", "target_name": "tot"})

    assert data_cfg["dataset_key"] == "clean_v1"
    assert Path(data_cfg["historical_csv"]).exists()
    assert Path(data_cfg["benchmark_csv"]).exists()
    assert Path(data_cfg["weather_csv"]).exists()
    assert Path(data_cfg["holidays_xlsx"]).exists()
    assert Path(data_cfg["aliases"]["legacy_historical_csv"]).exists()


def test_resolve_consumption_data_config_allows_explicit_override():
    override_path = "data/processed/conso/Historique 2020-2025.csv"

    data_cfg = resolve_consumption_data_config(
        {"dataset_key": "clean_v1"},
        overrides={"historical_csv": override_path},
    )

    assert Path(data_cfg["historical_csv"]).name == Path(override_path).name


def test_resolve_consumption_data_config_rejects_unknown_dataset():
    with pytest.raises(KeyError):
        resolve_consumption_data_config({"dataset_key": "missing_dataset"})
