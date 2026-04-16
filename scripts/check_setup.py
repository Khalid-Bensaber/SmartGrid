from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smartgrid.data.catalog import list_consumption_dataset_keys, resolve_consumption_data_config


PATH_FIELDS = (
    "historical_csv",
    "forecast_csv",
    "benchmark_csv",
    "weather_csv",
    "holidays_xlsx",
)


def iter_dataset_paths(dataset_key: str) -> list[tuple[str, Path]]:
    cfg = resolve_consumption_data_config({"dataset_key": dataset_key}, require_existing=False)
    entries: list[tuple[str, Path]] = []

    for field in PATH_FIELDS:
        value = cfg.get(field)
        if value:
            entries.append((field, Path(value)))

    for alias_name, alias_value in sorted((cfg.get("aliases") or {}).items()):
        if alias_value:
            entries.append((f"alias:{alias_name}", Path(alias_value)))

    return entries


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    missing_any = False
    for dataset_key in list_consumption_dataset_keys():
        print(f"[{dataset_key}]")
        missing = []
        present = 0

        for label, path in iter_dataset_paths(dataset_key):
            if path.exists():
                present += 1
                print(f"  OK      {label}: {path.relative_to(PROJECT_ROOT)}")
            else:
                missing.append((label, path))
                print(f"  MISSING {label}: {path.relative_to(PROJECT_ROOT)}")

        print(f"  Summary: {present} present, {len(missing)} missing")
        print()
        missing_any = missing_any or bool(missing)

    if missing_any:
        print("Setup incomplete: pull the missing tracked files or add them locally under data/ before training, replay, or inference.")
        return 1

    print("Setup looks complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
