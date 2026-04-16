from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from smartgrid.common.utils import load_yaml

DEFAULT_DATA_CATALOG_PATH = Path("configs/common/data_sources.yaml")
DEFAULT_CONSUMPTION_DATASET_KEY = "full_2020_2026"
PATH_FIELDS = (
    "historical_csv",
    "forecast_csv",
    "benchmark_csv",
    "weather_csv",
    "holidays_xlsx",
)


@dataclass(slots=True)
class ConsumptionDataset:
    key: str
    description: str | None
    values: dict[str, Any]


def find_project_root(start: str | Path | None = None) -> Path:
    start_path = Path(start or Path.cwd()).resolve()
    for candidate in [start_path] + list(start_path.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "smartgrid").exists():
            return candidate
    raise FileNotFoundError("Unable to find the project root from the current working directory.")


def resolve_project_path(path_like: str | Path | None, project_root: str | Path | None = None) -> Path | None:
    if path_like in (None, ""):
        return None

    raw = Path(path_like).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    root = find_project_root(project_root) if project_root is not None else find_project_root()
    return (root / raw).resolve()


def load_data_catalog(
    *,
    project_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    root = find_project_root(project_root) if project_root is not None else find_project_root()
    path = resolve_project_path(catalog_path or DEFAULT_DATA_CATALOG_PATH, root)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Data catalog not found: {catalog_path or DEFAULT_DATA_CATALOG_PATH}")
    return load_yaml(path) or {}


def list_consumption_dataset_keys(
    *,
    project_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> list[str]:
    catalog = load_data_catalog(project_root=project_root, catalog_path=catalog_path)
    datasets = catalog.get("consumption", {}).get("datasets", {})
    return sorted(str(key) for key in datasets.keys())


def _resolve_named_consumption_dataset(
    dataset_key: str | None,
    *,
    project_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> ConsumptionDataset:
    catalog = load_data_catalog(project_root=project_root, catalog_path=catalog_path)
    consumption_section = catalog.get("consumption", {})
    datasets = dict(consumption_section.get("datasets", {}))
    resolved_key = dataset_key or consumption_section.get("default_dataset") or DEFAULT_CONSUMPTION_DATASET_KEY
    if resolved_key not in datasets:
        available = ", ".join(sorted(datasets)) or "<none>"
        raise KeyError(f"Unknown consumption dataset '{resolved_key}'. Available datasets: {available}")

    values = dict(datasets[resolved_key] or {})
    aliases = dict(values.get("aliases") or {})
    values["aliases"] = aliases
    return ConsumptionDataset(
        key=str(resolved_key),
        description=values.get("description"),
        values=values,
    )


def resolve_consumption_data_config(
    data_cfg: Mapping[str, Any] | None = None,
    *,
    project_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
    dataset_key: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    require_existing: bool = True,
) -> dict[str, Any]:
    raw_cfg = dict(data_cfg or {})
    merged: dict[str, Any] = {}

    requested_dataset_key = dataset_key or raw_cfg.get("dataset_key")
    if requested_dataset_key or raw_cfg.get("catalog_path") or catalog_path:
        dataset = _resolve_named_consumption_dataset(
            requested_dataset_key,
            project_root=project_root,
            catalog_path=catalog_path or raw_cfg.get("catalog_path"),
        )
        merged.update(dataset.values)
        merged["dataset_key"] = dataset.key
        if dataset.description:
            merged["dataset_description"] = dataset.description

    merged.update(raw_cfg)
    if overrides:
        merged.update({key: value for key, value in overrides.items() if value is not None})

    root = find_project_root(project_root) if project_root is not None else find_project_root()
    resolved_catalog_path = resolve_project_path(catalog_path or raw_cfg.get("catalog_path") or DEFAULT_DATA_CATALOG_PATH, root)
    merged["catalog_path"] = str(resolved_catalog_path) if resolved_catalog_path is not None else None

    for field in PATH_FIELDS:
        if field in merged and merged[field] is not None:
            resolved_path = resolve_project_path(merged[field], root)
            if require_existing and resolved_path is not None and not resolved_path.exists():
                raise FileNotFoundError(f"Configured data file not found for {field}: {resolved_path}")
            merged[field] = str(resolved_path) if resolved_path is not None else None

    aliases = dict(merged.get("aliases") or {})
    for key, value in list(aliases.items()):
        if value is None:
            continue
        resolved_alias = resolve_project_path(value, root)
        if require_existing and resolved_alias is not None and not resolved_alias.exists():
            raise FileNotFoundError(f"Configured alias data file not found for {key}: {resolved_alias}")
        aliases[key] = str(resolved_alias) if resolved_alias is not None else None
    merged["aliases"] = aliases

    return merged
