from __future__ import annotations

import argparse
from pathlib import Path

from smartgrid.common.logging import build_log_path, setup_logger
from smartgrid.training.artifacts import promote_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a finished consumption run to current")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts-root", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(
        "smartgrid.promote",
        log_file=build_log_path(args.artifacts_root, "train", f"promote__{args.run_id}.log"),
    )
    run_dir = Path(args.artifacts_root) / "runs" / "consumption" / args.run_id
    current_dir = Path(args.artifacts_root) / "models" / "consumption" / "current"
    promote_bundle(run_dir, current_dir)
    logger.info("Promoted %s -> %s", run_dir, current_dir)
    print(f"Promoted {run_dir} -> {current_dir}")


if __name__ == "__main__":
    main()
