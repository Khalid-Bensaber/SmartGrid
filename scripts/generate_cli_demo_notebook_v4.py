from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "notebooks" / "experiments" / "SmartGrid_CLI_Demo_Notebook_v4.ipynb"


def build_notebook() -> dict:
    cells: list[dict] = []

    def add_md(text: str) -> None:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": textwrap.dedent(text).strip() + "\n",
            }
        )

    def add_code(text: str) -> None:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": textwrap.dedent(text).strip() + "\n",
            }
        )

    add_md(
        """
        # SmartGrid CLI Demo Notebook V4

        Notebook CLI-first pour :

        - benchmarker officiellement les runs `strict_day_ahead` sur le **long sample replay**,
        - comparer proprement plusieurs modeles sur des sous-periodes utiles,
        - etudier le comportement des modeles avec de vrais graphes d'analyse,
        - produire des exports reutilisables pour d'autres analyses.
        """
    )

    add_md(
        """
        ## 0. Scope & Semantics

        Regles de lecture :

        - **benchmark officiel metier** = `replay`
        - **offline** = diagnostic secondaire
        - **protocol audit** = controle technique de parite
        - **long sample officiel** = `2025-11-20` -> `2026-03-19`
        - **legacy overlap** = `2025-11-20` -> `2025-12-10`
        - le winner final doit etre choisi sur le replay officiel long sample
        """
    )

    add_code(
        """
        from __future__ import annotations

        import sys
        from pathlib import Path


        def find_repo_root(start: Path | None = None) -> Path:
            start_path = (start or Path.cwd()).resolve()
            for candidate in [start_path, *start_path.parents]:
                if (candidate / "Makefile").exists() and (candidate / "src" / "smartgrid").exists():
                    return candidate
            raise RuntimeError(
                f"Impossible de trouver la racine du repo SmartGrid depuis {start_path}. "
                "Ouvre le notebook depuis le repo ou fixe ROOT manuellement."
            )


        ROOT = find_repo_root()
        for candidate in [ROOT, ROOT / "src"]:
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

        from IPython.display import Markdown, display
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        from smartgrid.data.catalog import resolve_consumption_data_config
        from smartgrid.data.loaders import load_history, load_old_benchmark
        from smartgrid.notebooks.cli_demo_utils import (
            build_config_inventory,
            build_demo_paths,
            build_model_label_map,
            build_skipped_days_audit,
            build_truth_baseline_frame,
            build_wide_comparison_frame,
            coerce_datetime,
            collect_consumption_runs,
            compute_series_daily_metrics,
            compute_series_error_frame,
            compute_series_metrics,
            configure_pandas_display,
            find_latest_replay_summary,
            load_replay_summary,
            make_overrides,
            normalize_replay_summary,
            optional_cli_args,
            prepare_legacy_forecast_frame,
            run_cli,
            select_latest_runs_per_config,
            series_columns,
            slice_date_range,
            slice_single_day,
            write_json,
        )

        configure_pandas_display()
        PATHS = build_demo_paths(ROOT, notebook_export_dir="artifacts/notebook_exports/cli_demo_v4")
        ARTIFACTS = PATHS.artifacts_root
        EXPORT_ROOT = PATHS.notebook_export_root

        plt.style.use("seaborn-v0_8-whitegrid")

        print("ROOT:", ROOT)
        print("ARTIFACTS:", ARTIFACTS)
        print("EXPORT_ROOT:", EXPORT_ROOT)
        """
    )

    add_md(
        """
        ## 0.b User Configuration

        Cette cellule pilote les recalculs, la fenetre officielle et les graphes d'etude.
        """
    )

    add_code(
        """
        # ---- Execution toggles
        RUN_ENVIRONMENT_CHECKS = True
        RUN_TRAINING = False
        RUN_REPLAY_LONG_SAMPLE = False
        RUN_PROTOCOL_AUDIT = False
        RUN_PREDICT_DEMO = False
        RUN_PROMOTION = False

        # ---- Loading / cache behavior
        LOAD_EXISTING_RUNS = True
        LOAD_LATEST_LONG_SAMPLE_REPLAY_IF_NOT_RUN = True
        LOAD_CACHED_PREDICT_DEMO = True
        FORCE_RECOMPUTE_PREDICT_DEMO = False

        # ---- Official scope
        ONLY_STRICT_DAY_AHEAD = True

        # ---- Official long sample replay benchmark window
        OFFICIAL_LONG_SAMPLE_START = "2025-11-20"
        OFFICIAL_LONG_SAMPLE_END = "2026-03-19"

        # ---- Legacy overlap
        LEGACY_OVERLAP_START = "2025-11-20"
        LEGACY_OVERLAP_END = "2025-12-10"
        LEGACY_COVERAGE_END_DATE = "2025-12-10"

        # ---- Secondary study windows
        SHORT_TERM_REFERENCE_START = "2025-11-20"
        SHORT_TERM_REFERENCE_END = "2025-12-19"
        YEAR_2026_START = "2026-01-01"
        YEAR_2026_END = "2026-03-19"
        TRANSITION_START = "2025-12-20"
        TRANSITION_END = "2026-01-15"

        # ---- Dataset resolution
        DATASET_KEY = "full_2020_2026"
        CATALOG_PATH = None
        HISTORICAL_CSV = None
        WEATHER_CSV = None
        HOLIDAYS_XLSX = None
        BENCHMARK_CSV = None
        LEGACY_FORECAST_CSV = None

        # ---- Candidate configs
        CONFIG_PATHS = [
            "configs/consumption/mlp_strict_day_ahead_baseline.yaml",
            "configs/consumption/mlp_strict_day_ahead_weather_basic.yaml",
            "configs/consumption/mlp_strict_day_ahead_cyclical_weather_basic.yaml",
            "configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics.yaml",
        ]
        ANALYSIS_DAYS = 1

        # ---- Optional manual run selection
        REPLAY_MODEL_RUN_IDS = []
        LONG_SAMPLE_STUDY_RUN_IDS = []
        SELECTED_RUN_ID = None
        OVERLAY_TOP_K = 4
        WORST_DAYS_TOP_N = 10

        # ---- Plot focus dates
        SELECTED_DAY_PLOTS = [
            "2025-11-20",
            "2025-12-10",
            "2026-01-15",
            "2026-03-19",
        ]

        # ---- Protocol audit / predict demo
        PROTOCOL_AUDIT_RUN_ID = None
        PROTOCOL_AUDIT_DATE = "2026-03-19"
        DEMO_PREDICT_RUN_ID = None
        DEMO_PREDICT_DATE = "2026-03-19"

        # ---- Promotion
        PROMOTE_LONG_SAMPLE_WINNER = False
        """
    )

    add_md(
        """
        ## 0.c Environment And Data Sanity Checks

        On verifie la racine du repo, les scripts critiques, les configs candidates et la resolution du dataset.
        """
    )

    add_code(
        """
        resolved_data_config = resolve_consumption_data_config(
            {
                "dataset_key": DATASET_KEY,
                "date_col": "Date",
                "target_name": "tot",
            },
            project_root=ROOT,
            catalog_path=CATALOG_PATH,
            overrides={
                "historical_csv": HISTORICAL_CSV,
                "weather_csv": WEATHER_CSV,
                "holidays_xlsx": HOLIDAYS_XLSX,
                "benchmark_csv": BENCHMARK_CSV,
            },
        )

        legacy_candidates = [
            LEGACY_FORECAST_CSV,
            resolved_data_config.get("benchmark_csv"),
            resolved_data_config.get("forecast_csv"),
            (resolved_data_config.get("aliases") or {}).get("legacy_prediction_csv"),
        ]
        LEGACY_FORECAST_SOURCE = next((candidate for candidate in legacy_candidates if candidate), None)

        expected_paths = [
            ROOT / "Makefile",
            ROOT / "scripts" / "train_consumption.py",
            ROOT / "scripts" / "benchmark_replay_models.py",
            ROOT / "scripts" / "predict_next_day.py",
            ROOT / "scripts" / "promote_consumption_run.py",
        ] + [
            Path(path).resolve() if Path(path).is_absolute() else (ROOT / path).resolve()
            for path in CONFIG_PATHS
        ]

        checks_df = pd.DataFrame(
            [{"path": str(path), "exists": path.exists()} for path in expected_paths]
        )
        display(checks_df)

        if not checks_df["exists"].all():
            raise FileNotFoundError("Un ou plusieurs fichiers critiques du notebook sont absents.")

        PERIOD_WINDOWS = {
            "long_sample_official": (OFFICIAL_LONG_SAMPLE_START, OFFICIAL_LONG_SAMPLE_END),
            "legacy_overlap": (LEGACY_OVERLAP_START, LEGACY_OVERLAP_END),
            "short_term_reference": (SHORT_TERM_REFERENCE_START, SHORT_TERM_REFERENCE_END),
            "year_2026": (YEAR_2026_START, YEAR_2026_END),
            "transition_window": (TRANSITION_START, TRANSITION_END),
        }

        dataset_view = pd.DataFrame(
            {
                "dataset_key": [resolved_data_config.get("dataset_key")],
                "catalog_path": [resolved_data_config.get("catalog_path")],
                "historical_csv": [resolved_data_config.get("historical_csv")],
                "forecast_csv": [resolved_data_config.get("forecast_csv")],
                "benchmark_csv": [resolved_data_config.get("benchmark_csv")],
                "weather_csv": [resolved_data_config.get("weather_csv")],
                "holidays_xlsx": [resolved_data_config.get("holidays_xlsx")],
                "legacy_forecast_source": [LEGACY_FORECAST_SOURCE],
            }
        ).T
        dataset_view.columns = ["value"]
        display(dataset_view)

        period_view = pd.DataFrame(
            [
                {"period": name, "start_date": start_date, "end_date": end_date}
                for name, (start_date, end_date) in PERIOD_WINDOWS.items()
            ]
        )
        display(period_view)

        if RUN_ENVIRONMENT_CHECKS:
            _ = run_cli(["make", "help"], cwd=ROOT)
            _ = run_cli(["git", "branch", "--show-current"], cwd=ROOT, check=False)
            _ = run_cli(["git", "status", "--short"], cwd=ROOT, check=False)
        """
    )

    add_md(
        """
        ## 1. Inventory Runs / Configs

        On liste les configs candidates, les runs disponibles et les labels lisibles.
        """
    )

    add_code(
        """
        config_df = build_config_inventory(ROOT, CONFIG_PATHS)
        official_config_df = config_df.loc[config_df["official_eligible"] == True].copy()  # noqa: E712

        all_runs_df = collect_consumption_runs(ARTIFACTS) if LOAD_EXISTING_RUNS else pd.DataFrame()
        runs_df = all_runs_df.copy()
        if ONLY_STRICT_DAY_AHEAD and not runs_df.empty:
            runs_df = runs_df.loc[runs_df["official_eligible"] == True].copy()  # noqa: E712

        latest_run_ids_for_configs = select_latest_runs_per_config(
            runs_df,
            CONFIG_PATHS,
            root=ROOT,
            official_only=ONLY_STRICT_DAY_AHEAD,
        )

        display(config_df)
        display(
            runs_df[
                [
                    col
                    for col in [
                        "run_id",
                        "human_label",
                        "config_name",
                        "forecast_mode",
                        "dataset_key",
                        "offline_MAE",
                        "offline_RMSE",
                        "offline_InTolerance%",
                        "summary_json",
                    ]
                    if col in runs_df.columns
                ]
            ]
        )
        print("Latest candidate run_ids for configured YAMLs:", latest_run_ids_for_configs)
        """
    )

    add_md(
        """
        ## 2. Training Orchestration (CLI)

        Cette section appelle le vrai pipeline repo via `make train-consumption`.
        """
    )

    add_code(
        """
        training_payloads = []

        if RUN_TRAINING:
            for config_path in official_config_df["config_path"].tolist():
                cmd = [
                    "make",
                    "train-consumption",
                    f"CONFIG={config_path}",
                    f"ANALYSIS_DAYS={ANALYSIS_DAYS}",
                ] + make_overrides(
                    dataset_key=DATASET_KEY,
                    historical_csv=HISTORICAL_CSV,
                    weather_csv=WEATHER_CSV,
                    holidays_xlsx=HOLIDAYS_XLSX,
                    benchmark_csv=BENCHMARK_CSV,
                )
                result = run_cli(cmd, cwd=ROOT)
                payload = result.extract_json()
                payload["config_path"] = config_path
                payload["config_name"] = Path(config_path).stem
                training_payloads.append(payload)

        training_df = pd.DataFrame(training_payloads)
        if RUN_TRAINING:
            runs_df = collect_consumption_runs(ARTIFACTS)
        display(training_df)
        """
    )

    add_md(
        """
        ## 3. Official Replay Benchmark On Long Sample

        C'est le coeur du notebook. Le classement principal est produit sur :

        - `2025-11-20`
        - `2026-03-19`
        """
    )

    add_code(
        """
        official_replay_payload = None
        official_replay_source = None
        official_replay_summary_df = pd.DataFrame()

        if REPLAY_MODEL_RUN_IDS:
            official_candidate_run_ids = [str(run_id) for run_id in REPLAY_MODEL_RUN_IDS]
        elif not training_df.empty:
            official_candidate_run_ids = training_df["run_id"].astype(str).tolist()
        else:
            official_candidate_run_ids = latest_run_ids_for_configs

        print("Official long-sample replay candidate run_ids:", official_candidate_run_ids)

        if RUN_REPLAY_LONG_SAMPLE and official_candidate_run_ids:
            cmd = [
                "python",
                "scripts/benchmark_replay_models.py",
                "--start-date",
                OFFICIAL_LONG_SAMPLE_START,
                "--end-date",
                OFFICIAL_LONG_SAMPLE_END,
            ] + optional_cli_args(
                dataset_key=DATASET_KEY,
                catalog_path=CATALOG_PATH,
                historical_csv=HISTORICAL_CSV,
                weather_csv=WEATHER_CSV,
                holidays_xlsx=HOLIDAYS_XLSX,
                benchmark_csv=BENCHMARK_CSV,
            ) + official_candidate_run_ids
            result = run_cli(cmd, cwd=ROOT)
            official_replay_payload = result.extract_json()
            official_replay_source = official_replay_payload["summary_csv"]
            official_replay_summary_df = load_replay_summary(official_replay_source)
        elif LOAD_LATEST_LONG_SAMPLE_REPLAY_IF_NOT_RUN:
            latest_replay_summary = find_latest_replay_summary(
                ARTIFACTS,
                start_date=OFFICIAL_LONG_SAMPLE_START,
                end_date=OFFICIAL_LONG_SAMPLE_END,
            )
            if latest_replay_summary is not None:
                official_replay_source = str(latest_replay_summary.resolve())
                official_replay_summary_df = load_replay_summary(latest_replay_summary)

        official_ranking_df = normalize_replay_summary(official_replay_summary_df, runs_df=runs_df)
        if ONLY_STRICT_DAY_AHEAD and not official_ranking_df.empty:
            official_ranking_df = official_ranking_df.loc[
                official_ranking_df["official_eligible"] == True  # noqa: E712
            ].reset_index(drop=True)

        official_leaderboard_export = EXPORT_ROOT / "long_sample_replay_leaderboard.csv"
        if not official_ranking_df.empty:
            official_ranking_df.to_csv(official_leaderboard_export, index=False)
            print("Official leaderboard export:", official_leaderboard_export)
            print("Official replay source:", official_replay_source)

        display(
            official_ranking_df[
                [
                    col
                    for col in [
                        "human_label",
                        "requested_model_run_id",
                        "config_name",
                        "MAE",
                        "RMSE",
                        "Bias(ME)",
                        "InTolerance%",
                        "n_requested_days",
                        "n_forecasted_days",
                        "n_skipped_days",
                        "skip_rate_pct",
                        "metrics_json",
                        "replay_csv",
                    ]
                    if col in official_ranking_df.columns
                ]
            ]
            if not official_ranking_df.empty
            else official_ranking_df
        )
        """
    )

    add_code(
        """
        history_df = load_history(
            resolved_data_config["historical_csv"],
            date_col=resolved_data_config.get("date_col", "Date"),
            target_col=resolved_data_config.get("target_name", "tot"),
        )
        history_df = coerce_datetime(history_df, "Date")
        truth_baseline_df = build_truth_baseline_frame(history_df, date_col="Date", real_col="tot")

        legacy_raw_df = (
            load_old_benchmark(LEGACY_FORECAST_SOURCE, date_col="Date")
            if LEGACY_FORECAST_SOURCE
            else pd.DataFrame()
        )
        legacy_prepared_df = prepare_legacy_forecast_frame(
            legacy_raw_df,
            coverage_end_date=LEGACY_COVERAGE_END_DATE,
        )

        if LONG_SAMPLE_STUDY_RUN_IDS:
            study_run_ids = [str(run_id) for run_id in LONG_SAMPLE_STUDY_RUN_IDS]
        else:
            study_run_ids = (
                official_ranking_df["requested_model_run_id"].astype(str).tolist()
                if not official_ranking_df.empty
                else []
            )

        replay_detail_frames = {}
        for run_id in study_run_ids:
            row = (
                official_ranking_df.loc[
                    official_ranking_df["requested_model_run_id"].astype(str) == str(run_id)
                ].head(1)
                if not official_ranking_df.empty
                else pd.DataFrame()
            )
            if row.empty:
                continue
            replay_csv = row.iloc[0].get("replay_csv")
            if replay_csv and Path(replay_csv).exists():
                replay_detail_frames[run_id] = coerce_datetime(pd.read_csv(replay_csv), "Date")

        if SELECTED_RUN_ID:
            ACTIVE_RUN_ID = str(SELECTED_RUN_ID)
        elif not official_ranking_df.empty:
            ACTIVE_RUN_ID = str(official_ranking_df.iloc[0]["requested_model_run_id"])
        else:
            ACTIVE_RUN_ID = None

        label_map = build_model_label_map(
            study_run_ids,
            runs_df=runs_df,
            replay_df=official_ranking_df,
        )
        model_metadata_df = (
            official_ranking_df[
                [
                    col
                    for col in [
                        "requested_model_run_id",
                        "human_label",
                        "config_name",
                        "experiment_name",
                        "forecast_mode",
                        "start_date",
                        "end_date",
                        "n_requested_days",
                        "n_forecasted_days",
                        "n_skipped_days",
                        "skip_rate_pct",
                        "replay_csv",
                        "metrics_json",
                        "offline_MAE",
                        "offline_RMSE",
                        "offline_InTolerance%",
                    ]
                    if col in official_ranking_df.columns
                ]
            ]
            .rename(
                columns={
                    "requested_model_run_id": "run_id",
                    "human_label": "series",
                }
            )
            .copy()
            if not official_ranking_df.empty
            else pd.DataFrame()
        )


        def export_frame(frame: pd.DataFrame, relative_path: str) -> Path:
            output_path = EXPORT_ROOT / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_path, index=False)
            return output_path


        def attach_model_metadata(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame.copy()
            if model_metadata_df.empty:
                return frame.copy()
            return frame.merge(model_metadata_df, on="series", how="left")


        long_sample_compare_df = pd.DataFrame()
        all_model_series = []
        top_model_series = []
        long_sample_metrics_raw_df = pd.DataFrame()
        long_sample_metrics_by_model_df = pd.DataFrame()
        long_sample_daily_metrics_raw_df = pd.DataFrame()
        long_sample_daily_metrics_by_model_df = pd.DataFrame()
        long_sample_reference_metrics_df = pd.DataFrame()
        long_sample_error_series_by_model_df = pd.DataFrame()

        if replay_detail_frames:
            long_sample_compare_df = build_wide_comparison_frame(
                truth_baseline_df=truth_baseline_df,
                model_frames=replay_detail_frames,
                label_map=label_map,
                start_date=OFFICIAL_LONG_SAMPLE_START,
                end_date=OFFICIAL_LONG_SAMPLE_END,
                legacy_df=legacy_prepared_df,
            )

            all_model_series = [
                label_map[run_id]
                for run_id in study_run_ids
                if run_id in label_map and label_map[run_id] in long_sample_compare_df.columns
            ]
            top_run_ids = (
                official_ranking_df["requested_model_run_id"].astype(str).head(OVERLAY_TOP_K).tolist()
                if not official_ranking_df.empty
                else []
            )
            top_model_series = [
                label_map[run_id]
                for run_id in top_run_ids
                if run_id in label_map and label_map[run_id] in long_sample_compare_df.columns
            ]

            long_sample_compare_export = export_frame(
                long_sample_compare_df,
                "model_range_comparison_long_sample.csv",
            )
            print("Long sample comparison export:", long_sample_compare_export)

            long_sample_metrics_raw_df = compute_series_metrics(
                long_sample_compare_df,
                columns=all_model_series,
                period_label="long_sample_official",
            )
            long_sample_daily_metrics_raw_df = compute_series_daily_metrics(
                long_sample_compare_df,
                columns=all_model_series,
                period_label="long_sample_official",
            )
            long_sample_reference_metrics_df = compute_series_metrics(
                long_sample_compare_df,
                columns=[column for column in ["weekly_naive"] if column in long_sample_compare_df.columns],
                period_label="long_sample_official",
            )
            long_sample_error_series_by_model_df = attach_model_metadata(
                compute_series_error_frame(
                    long_sample_compare_df,
                    columns=all_model_series,
                    period_label="long_sample_official",
                )
            )

            weekly_reference_mae = None
            weekly_reference_rmse = None
            if not long_sample_reference_metrics_df.empty:
                weekly_row = long_sample_reference_metrics_df.loc[
                    long_sample_reference_metrics_df["series"] == "weekly_naive"
                ].head(1)
                if not weekly_row.empty:
                    weekly_reference_mae = float(weekly_row.iloc[0]["MAE"])
                    weekly_reference_rmse = float(weekly_row.iloc[0]["RMSE"])

            long_sample_metrics_by_model_df = attach_model_metadata(long_sample_metrics_raw_df)
            if not long_sample_metrics_by_model_df.empty:
                if weekly_reference_mae not in [None, 0]:
                    long_sample_metrics_by_model_df["MAE_skill_vs_weekly_pct"] = (
                        100.0
                        * (1.0 - long_sample_metrics_by_model_df["MAE"] / weekly_reference_mae)
                    )
                if weekly_reference_rmse not in [None, 0]:
                    long_sample_metrics_by_model_df["RMSE_skill_vs_weekly_pct"] = (
                        100.0
                        * (1.0 - long_sample_metrics_by_model_df["RMSE"] / weekly_reference_rmse)
                    )
                long_sample_metrics_by_model_df = long_sample_metrics_by_model_df.sort_values("MAE").reset_index(drop=True)

            long_sample_daily_metrics_by_model_df = attach_model_metadata(long_sample_daily_metrics_raw_df)

            if not long_sample_metrics_by_model_df.empty:
                export_frame(long_sample_metrics_by_model_df, "long_sample_metrics_by_model.csv")
            if not long_sample_daily_metrics_by_model_df.empty:
                export_frame(long_sample_daily_metrics_by_model_df, "per_day_metrics_by_model_long_sample.csv")
            if not long_sample_error_series_by_model_df.empty:
                export_frame(long_sample_error_series_by_model_df, "long_sample_error_series_by_model.csv")

        coverage_view = pd.DataFrame(
            {
                "truth_start": [truth_baseline_df["Date"].min()],
                "truth_end": [truth_baseline_df["Date"].max()],
                "legacy_start": [legacy_prepared_df["Date"].min() if not legacy_prepared_df.empty else None],
                "legacy_end": [legacy_prepared_df["Date"].max() if not legacy_prepared_df.empty else None],
                "active_run_id": [ACTIVE_RUN_ID],
            }
        ).T
        coverage_view.columns = ["value"]
        display(coverage_view)
        display(long_sample_metrics_by_model_df)
        """
    )

    add_md(
        """
        ## 4. Audit Skipped Days

        Le benchmark ne doit pas etre "beau" juste parce que des jours difficiles ont ete sautes.
        """
    )

    add_code(
        """
        skipped_detail_df, skipped_counts_df = build_skipped_days_audit(official_ranking_df)

        skip_alignment_df = pd.DataFrame()
        if official_ranking_df.empty:
            skip_alignment_df = pd.DataFrame({"message": ["No official ranking loaded."]})
        elif skipped_detail_df.empty:
            skip_alignment_df = pd.DataFrame(
                {
                    "n_models": [int(len(official_ranking_df))],
                    "all_models_same_skipped_days": [True],
                    "notes": ["No skipped days on the loaded long-sample replay benchmark."],
                }
            )
        else:
            signature_df = (
                skipped_detail_df.assign(signature=skipped_detail_df["target_date"].astype(str) + " | " + skipped_detail_df["reason"].astype(str))
                .groupby(["requested_model_run_id", "human_label"], dropna=False)
                .agg(
                    n_skipped_days=("target_date", "count"),
                    skip_signature=("signature", lambda values: " || ".join(sorted(set(values)))),
                )
                .reset_index()
            )
            all_models_same_skipped_days = signature_df["skip_signature"].nunique() == 1
            identical_day_counts = signature_df["n_skipped_days"].nunique() == 1
            skip_alignment_df = pd.DataFrame(
                {
                    "n_models": [int(len(signature_df))],
                    "all_models_same_skipped_days": [bool(all_models_same_skipped_days)],
                    "identical_skip_counts": [bool(identical_day_counts)],
                    "n_unique_skip_signatures": [int(signature_df["skip_signature"].nunique())],
                }
            )
            display(signature_df)

        skipped_detail_export = export_frame(skipped_detail_df, "skipped_days_audit.csv")
        skipped_count_export = export_frame(skipped_counts_df, "skipped_days_counts.csv")
        skip_alignment_export = export_frame(skip_alignment_df, "skipped_days_alignment_summary.csv")

        display(skipped_counts_df if not skipped_counts_df.empty else skip_alignment_df)
        print("Skipped detail export:", skipped_detail_export)
        print("Skipped counts export:", skipped_count_export)
        print("Skipped alignment export:", skip_alignment_export)
        """
    )

    add_md(
        """
        ## 5. Long Sample Multi-Model Study

        La comparaison principale se fait sur le replay officiel long sample, avec overlays multi-modeles et zooms utiles.
        """
    )

    add_code(
        """
        def plot_overlay(
            frame: pd.DataFrame,
            *,
            columns: list[str],
            title: str,
            resample_rule: str | None = None,
            linewidth: float = 1.8,
        ) -> None:
            if frame.empty:
                print(f"No data for {title}")
                return
            work = frame.copy()
            if resample_rule is not None:
                available_cols = [column for column in columns if column in work.columns]
                work = (
                    work.set_index("Date")[available_cols]
                    .resample(resample_rule)
                    .mean()
                    .reset_index()
                )
            fig, ax = plt.subplots(figsize=(16, 5))
            color_map = {
                "real": "black",
                "weekly_naive": "tab:gray",
                "legacy_forecast": "tab:orange",
            }
            style_map = {
                "real": {"linewidth": 2.6, "linestyle": "-"},
                "weekly_naive": {"linewidth": 1.8, "linestyle": "--"},
                "legacy_forecast": {"linewidth": 1.8, "linestyle": ":"},
            }
            for column in columns:
                if column not in work.columns:
                    continue
                if column == "legacy_forecast" and work[column].notna().sum() == 0:
                    continue
                style = style_map.get(column, {"linewidth": linewidth, "linestyle": "-"})
                ax.plot(
                    work["Date"],
                    work[column],
                    label=column,
                    color=color_map.get(column),
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                )
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Consumption")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


        def plot_daily_metric_over_time(
            daily_metrics_df: pd.DataFrame,
            *,
            metric: str,
            series_to_plot: list[str],
            title: str,
            start_date: str | None = None,
            end_date: str | None = None,
        ) -> None:
            if daily_metrics_df.empty:
                print(f"No daily metrics for {title}")
                return
            work = daily_metrics_df.copy()
            work["target_date"] = pd.to_datetime(work["target_date"], errors="coerce")
            work = work.dropna(subset=["target_date"])
            if start_date is not None:
                work = work[work["target_date"] >= pd.Timestamp(start_date)]
            if end_date is not None:
                work = work[work["target_date"] <= pd.Timestamp(end_date)]
            pivot = (
                work[work["series"].isin(series_to_plot)]
                .pivot(index="target_date", columns="series", values=metric)
                .sort_index()
            )
            if pivot.empty:
                print(f"No pivot data for {title}")
                return
            pivot.plot(figsize=(16, 5), title=title)
            plt.xlabel("Date")
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


        def plot_error_distribution(
            error_df: pd.DataFrame,
            *,
            series_to_plot: list[str],
            title: str,
        ) -> None:
            if error_df.empty:
                print(f"No error data for {title}")
                return
            fig, ax = plt.subplots(figsize=(12, 5))
            for series_name in series_to_plot:
                work = error_df.loc[error_df["series"] == series_name, "abs_error"].dropna()
                if work.empty:
                    continue
                ax.hist(work, bins=40, alpha=0.35, label=series_name)
            ax.set_title(title)
            ax.set_xlabel("Absolute error")
            ax.set_ylabel("Frequency")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


        def bar_rank_chart(
            metrics_df: pd.DataFrame,
            *,
            metric: str,
            title: str,
        ) -> None:
            if metrics_df.empty or metric not in metrics_df.columns:
                print(f"No data for {title}")
                return
            work = metrics_df.sort_values(metric, ascending=True).copy()
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.barh(work["series"], work[metric], color="tab:blue")
            ax.invert_yaxis()
            ax.set_title(title)
            ax.set_xlabel(metric)
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            plt.show()


        long_sample_plot_columns = [
            column
            for column in ["real", "weekly_naive", "legacy_forecast"] + top_model_series
            if not long_sample_compare_df.empty and column in long_sample_compare_df.columns
        ]

        if not long_sample_compare_df.empty and long_sample_plot_columns:
            plot_overlay(
                long_sample_compare_df,
                columns=long_sample_plot_columns,
                title=f"Long-sample official replay study (daily mean) - {OFFICIAL_LONG_SAMPLE_START} to {OFFICIAL_LONG_SAMPLE_END}",
                resample_rule="D",
            )

            transition_compare_df = slice_date_range(
                long_sample_compare_df,
                TRANSITION_START,
                TRANSITION_END,
            )
            plot_overlay(
                transition_compare_df,
                columns=long_sample_plot_columns,
                title=f"Transition study (daily mean) - {TRANSITION_START} to {TRANSITION_END}",
                resample_rule="D",
            )

            short_term_compare_df = slice_date_range(
                long_sample_compare_df,
                SHORT_TERM_REFERENCE_START,
                SHORT_TERM_REFERENCE_END,
            )
            plot_overlay(
                short_term_compare_df,
                columns=long_sample_plot_columns,
                title=f"Short-term reference study (daily mean) - {SHORT_TERM_REFERENCE_START} to {SHORT_TERM_REFERENCE_END}",
                resample_rule="D",
            )

            for target_date in SELECTED_DAY_PLOTS:
                day_frame = slice_single_day(long_sample_compare_df, target_date)
                if day_frame.empty:
                    continue
                export_frame(day_frame, f"daily_overlays/model_day_comparison_{target_date}.csv")
                plot_overlay(
                    day_frame,
                    columns=long_sample_plot_columns,
                    title=f"Detailed daily overlay - {target_date}",
                    resample_rule=None,
                )
        else:
            print("No long-sample comparison frame available.")
        """
    )

    add_md(
        """
        ## 6. Legacy Overlap Study

        Focus sur la zone ou le legacy existe encore :

        - `2025-11-20`
        - `2025-12-10`
        """
    )

    add_code(
        """
        legacy_overlap_compare_df = slice_date_range(
            long_sample_compare_df,
            LEGACY_OVERLAP_START,
            LEGACY_OVERLAP_END,
        )

        legacy_overlap_model_metrics_raw_df = compute_series_metrics(
            legacy_overlap_compare_df,
            columns=all_model_series,
            period_label="legacy_overlap",
        )
        legacy_overlap_reference_metrics_df = compute_series_metrics(
            legacy_overlap_compare_df,
            columns=[
                column
                for column in ["weekly_naive", "legacy_forecast"]
                if not legacy_overlap_compare_df.empty and column in legacy_overlap_compare_df.columns
            ],
            period_label="legacy_overlap",
        )
        legacy_overlap_daily_metrics_df = attach_model_metadata(
            compute_series_daily_metrics(
                legacy_overlap_compare_df,
                columns=all_model_series,
                period_label="legacy_overlap",
            )
        )

        legacy_overlap_metrics_by_model_df = attach_model_metadata(legacy_overlap_model_metrics_raw_df)
        if not legacy_overlap_reference_metrics_df.empty and not legacy_overlap_metrics_by_model_df.empty:
            weekly_row = legacy_overlap_reference_metrics_df.loc[
                legacy_overlap_reference_metrics_df["series"] == "weekly_naive"
            ].head(1)
            legacy_row = legacy_overlap_reference_metrics_df.loc[
                legacy_overlap_reference_metrics_df["series"] == "legacy_forecast"
            ].head(1)
            if not weekly_row.empty and float(weekly_row.iloc[0]["MAE"]) != 0:
                legacy_overlap_metrics_by_model_df["MAE_skill_vs_weekly_pct"] = (
                    100.0
                    * (
                        1.0
                        - legacy_overlap_metrics_by_model_df["MAE"] / float(weekly_row.iloc[0]["MAE"])
                    )
                )
            if not legacy_row.empty and float(legacy_row.iloc[0]["MAE"]) != 0:
                legacy_overlap_metrics_by_model_df["MAE_skill_vs_legacy_pct"] = (
                    100.0
                    * (
                        1.0
                        - legacy_overlap_metrics_by_model_df["MAE"] / float(legacy_row.iloc[0]["MAE"])
                    )
                )
        if not legacy_overlap_metrics_by_model_df.empty:
            legacy_overlap_metrics_by_model_df = legacy_overlap_metrics_by_model_df.sort_values("MAE").reset_index(drop=True)

        export_frame(legacy_overlap_compare_df, "legacy_overlap_comparison_frame.csv")
        export_frame(legacy_overlap_metrics_by_model_df, "legacy_overlap_metrics_by_model.csv")
        export_frame(legacy_overlap_reference_metrics_df, "legacy_overlap_reference_metrics.csv")

        display(legacy_overlap_metrics_by_model_df)
        display(legacy_overlap_reference_metrics_df)

        legacy_plot_columns = [
            column
            for column in ["real", "weekly_naive", "legacy_forecast"] + top_model_series
            if not legacy_overlap_compare_df.empty and column in legacy_overlap_compare_df.columns
        ]
        if not legacy_overlap_compare_df.empty and legacy_plot_columns:
            plot_overlay(
                legacy_overlap_compare_df,
                columns=legacy_plot_columns,
                title=f"Legacy overlap study (daily mean) - {LEGACY_OVERLAP_START} to {LEGACY_OVERLAP_END}",
                resample_rule="D",
            )
        """
    )

    add_md(
        """
        ## 7. Errors And Bias Study

        On analyse :

        - erreur absolue dans le temps
        - distribution d'erreur
        - biais signe
        - classement visuel
        - pires jours
        - transition fin 2025 -> debut 2026
        """
    )

    add_code(
        """
        short_term_metrics_raw_df = compute_series_metrics(
            slice_date_range(long_sample_compare_df, SHORT_TERM_REFERENCE_START, SHORT_TERM_REFERENCE_END),
            columns=all_model_series,
            period_label="short_term_reference",
        )
        year_2026_metrics_raw_df = compute_series_metrics(
            slice_date_range(long_sample_compare_df, YEAR_2026_START, YEAR_2026_END),
            columns=all_model_series,
            period_label="year_2026",
        )
        transition_metrics_raw_df = compute_series_metrics(
            slice_date_range(long_sample_compare_df, TRANSITION_START, TRANSITION_END),
            columns=all_model_series,
            period_label="transition_window",
        )

        short_term_metrics_by_model_df = attach_model_metadata(short_term_metrics_raw_df)
        year_2026_metrics_by_model_df = attach_model_metadata(year_2026_metrics_raw_df)
        transition_metrics_by_model_df = attach_model_metadata(transition_metrics_raw_df)

        if not long_sample_daily_metrics_by_model_df.empty and top_model_series:
            plot_daily_metric_over_time(
                long_sample_daily_metrics_by_model_df,
                metric="MAE",
                series_to_plot=top_model_series,
                title="Daily MAE over time - top models on long sample",
            )
            plot_daily_metric_over_time(
                long_sample_daily_metrics_by_model_df,
                metric="Bias(ME)",
                series_to_plot=top_model_series,
                title="Daily signed bias over time - top models on long sample",
            )
            plot_daily_metric_over_time(
                long_sample_daily_metrics_by_model_df,
                metric="MAE",
                series_to_plot=top_model_series,
                title=f"Transition MAE focus - {TRANSITION_START} to {TRANSITION_END}",
                start_date=TRANSITION_START,
                end_date=TRANSITION_END,
            )

        if not long_sample_error_series_by_model_df.empty and top_model_series:
            plot_error_distribution(
                long_sample_error_series_by_model_df,
                series_to_plot=top_model_series,
                title="Absolute error distribution - top models on long sample",
            )

        if not long_sample_metrics_by_model_df.empty:
            bar_rank_chart(
                long_sample_metrics_by_model_df,
                metric="MAE",
                title="Long-sample MAE ranking",
            )
            bar_rank_chart(
                long_sample_metrics_by_model_df,
                metric="RMSE",
                title="Long-sample RMSE ranking",
            )
            bar_rank_chart(
                long_sample_metrics_by_model_df,
                metric="Bias(ME)",
                title="Long-sample bias ranking",
            )

            under_over_df = long_sample_metrics_by_model_df[["series", "UnderShare%", "OverShare%"]].copy()
            under_over_df = under_over_df.set_index("series")
            under_over_df.plot(kind="bar", stacked=True, figsize=(12, 5), title="Under / over prediction share")
            plt.ylabel("Share %")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show()

        worst_days_by_model_df = pd.DataFrame()
        if not long_sample_daily_metrics_by_model_df.empty:
            worst_days_by_model_df = (
                long_sample_daily_metrics_by_model_df.sort_values(["series", "MAE"], ascending=[True, False])
                .groupby("series", dropna=False)
                .head(WORST_DAYS_TOP_N)
                .reset_index(drop=True)
            )
            export_frame(worst_days_by_model_df, "worst_days_by_model_long_sample.csv")
            display(worst_days_by_model_df)

        period_metrics_by_model_df = pd.concat(
            [
                attach_model_metadata(long_sample_metrics_raw_df),
                short_term_metrics_by_model_df,
                legacy_overlap_metrics_by_model_df,
                year_2026_metrics_by_model_df,
                transition_metrics_by_model_df,
            ],
            ignore_index=True,
            sort=False,
        ) if not long_sample_metrics_raw_df.empty else pd.DataFrame()
        if not period_metrics_by_model_df.empty:
            export_frame(period_metrics_by_model_df, "period_metrics_by_model.csv")
            display(period_metrics_by_model_df)
        """
    )

    add_md(
        """
        ## 8. Protocol Comparison Audit

        Cette section reste un garde-fou technique.
        Elle ne doit pas remplacer le benchmark officiel long sample.
        """
    )

    add_code(
        """
        protocol_run_id = str(PROTOCOL_AUDIT_RUN_ID or ACTIVE_RUN_ID) if (PROTOCOL_AUDIT_RUN_ID or ACTIVE_RUN_ID) else None
        protocol_compare_df = pd.DataFrame()
        protocol_metrics_df = pd.DataFrame()
        protocol_parity_summary_df = pd.DataFrame()

        if protocol_run_id:
            protocol_run_row = runs_df.loc[runs_df["run_id"].astype(str) == protocol_run_id].head(1)
            protocol_run_dir = (
                Path(protocol_run_row.iloc[0]["run_dir"])
                if not protocol_run_row.empty
                else ARTIFACTS / "runs" / "consumption" / protocol_run_id
            )
            protocol_output_dir = EXPORT_ROOT / "protocol_audit" / protocol_run_id / PROTOCOL_AUDIT_DATE
            protocol_predict_csv = protocol_output_dir / f"predict_target_day_{PROTOCOL_AUDIT_DATE}.csv"

            if RUN_PROTOCOL_AUDIT or protocol_predict_csv.exists():
                if RUN_PROTOCOL_AUDIT or not protocol_predict_csv.exists():
                    protocol_output_dir.mkdir(parents=True, exist_ok=True)
                    result = run_cli(
                        [
                            "python",
                            "scripts/predict_next_day.py",
                            "--current-dir",
                            str(protocol_run_dir),
                            "--target-date",
                            PROTOCOL_AUDIT_DATE,
                            "--output-csv",
                            str(protocol_predict_csv),
                        ]
                        + optional_cli_args(
                            dataset_key=DATASET_KEY,
                            catalog_path=CATALOG_PATH,
                            historical_csv=HISTORICAL_CSV,
                            weather_csv=WEATHER_CSV,
                            holidays_xlsx=HOLIDAYS_XLSX,
                            benchmark_csv=BENCHMARK_CSV,
                        ),
                        cwd=ROOT,
                    )
                    _ = result.extract_json()
                protocol_predict_df = coerce_datetime(pd.read_csv(protocol_predict_csv), "Date")
            else:
                protocol_predict_df = pd.DataFrame()

            protocol_offline_df = pd.DataFrame()
            if not protocol_run_row.empty:
                backtest_csv = protocol_run_row.iloc[0].get("backtest_csv")
                if backtest_csv and Path(backtest_csv).exists():
                    protocol_offline_df = slice_single_day(
                        coerce_datetime(pd.read_csv(backtest_csv), "Date"),
                        PROTOCOL_AUDIT_DATE,
                    )

            protocol_replay_df = (
                slice_single_day(replay_detail_frames.get(protocol_run_id, pd.DataFrame()), PROTOCOL_AUDIT_DATE)
                if protocol_run_id in replay_detail_frames
                else pd.DataFrame()
            )
            truth_day_df = slice_single_day(truth_baseline_df, PROTOCOL_AUDIT_DATE)[["Date", "real", "weekly_naive"]]
            legacy_day_df = slice_single_day(legacy_prepared_df, PROTOCOL_AUDIT_DATE)

            protocol_compare_df = truth_day_df.copy()
            if not legacy_day_df.empty:
                protocol_compare_df = protocol_compare_df.merge(legacy_day_df, on="Date", how="left")
            else:
                protocol_compare_df["legacy_forecast"] = pd.NA

            if not protocol_offline_df.empty and "Ptot_TOTAL_Forecast" in protocol_offline_df.columns:
                protocol_compare_df = protocol_compare_df.merge(
                    protocol_offline_df[["Date", "Ptot_TOTAL_Forecast"]].rename(
                        columns={"Ptot_TOTAL_Forecast": "offline_diagnostic"}
                    ),
                    on="Date",
                    how="left",
                )
            if not protocol_predict_df.empty and "Ptot_TOTAL_Forecast" in protocol_predict_df.columns:
                protocol_compare_df = protocol_compare_df.merge(
                    protocol_predict_df[["Date", "Ptot_TOTAL_Forecast"]].rename(
                        columns={"Ptot_TOTAL_Forecast": "runtime_predict"}
                    ),
                    on="Date",
                    how="left",
                )
            if not protocol_replay_df.empty and "Ptot_TOTAL_Forecast" in protocol_replay_df.columns:
                protocol_compare_df = protocol_compare_df.merge(
                    protocol_replay_df[["Date", "Ptot_TOTAL_Forecast"]].rename(
                        columns={"Ptot_TOTAL_Forecast": "replay_same_day"}
                    ),
                    on="Date",
                    how="left",
                )

            protocol_compare_df = coerce_datetime(protocol_compare_df, "Date")
            export_frame(
                protocol_compare_df,
                f"protocol_audit/{protocol_run_id}/model_day_comparison_{PROTOCOL_AUDIT_DATE}.csv",
            )

            protocol_series = [
                column
                for column in ["offline_diagnostic", "runtime_predict", "replay_same_day"]
                if column in protocol_compare_df.columns
            ]
            protocol_metrics_df = compute_series_metrics(
                protocol_compare_df,
                columns=protocol_series,
                period_label="protocol_audit",
            )

            parity_rows = []
            pairs = [
                ("runtime_predict", "replay_same_day"),
                ("offline_diagnostic", "runtime_predict"),
                ("offline_diagnostic", "replay_same_day"),
            ]
            for left_col, right_col in pairs:
                if left_col not in protocol_compare_df.columns or right_col not in protocol_compare_df.columns:
                    continue
                valid = protocol_compare_df[["Date", left_col, right_col]].dropna().copy()
                if valid.empty:
                    continue
                diff = (valid[left_col] - valid[right_col]).abs()
                parity_rows.append(
                    {
                        "pair": f"{left_col} vs {right_col}",
                        "n_points": int(len(valid)),
                        "max_abs_diff": float(diff.max()),
                        "mean_abs_diff": float(diff.mean()),
                    }
                )
            protocol_parity_summary_df = pd.DataFrame(parity_rows)
            export_frame(
                protocol_parity_summary_df,
                f"protocol_audit/{protocol_run_id}/parity_summary_{PROTOCOL_AUDIT_DATE}.csv",
            )

            protocol_plot_columns = [
                column
                for column in [
                    "real",
                    "weekly_naive",
                    "legacy_forecast",
                    "offline_diagnostic",
                    "runtime_predict",
                    "replay_same_day",
                ]
                if column in protocol_compare_df.columns
            ]
            if len(protocol_plot_columns) > 1:
                plot_overlay(
                    protocol_compare_df,
                    columns=protocol_plot_columns,
                    title=f"Technical parity audit - {protocol_run_id} - {PROTOCOL_AUDIT_DATE}",
                    resample_rule=None,
                )

            display(protocol_metrics_df)
            display(protocol_parity_summary_df)
        else:
            print("No protocol audit run selected.")
        """
    )

    add_md(
        """
        ## 9. Final Ranking And Period Comparison

        Le winner final est choisi sur le replay officiel long sample.
        On affiche aussi comment le classement bouge selon les sous-periodes.
        """
    )

    add_code(
        """
        period_metric_frames = {
            "long_sample_official": long_sample_metrics_raw_df,
            "short_term_reference": short_term_metrics_raw_df,
            "legacy_overlap": legacy_overlap_model_metrics_raw_df,
            "year_2026": year_2026_metrics_raw_df,
        }


        def build_rank_table(metric_frame: pd.DataFrame, period_name: str) -> pd.DataFrame:
            if metric_frame.empty:
                return pd.DataFrame(columns=["series", f"rank_{period_name}", f"MAE_{period_name}", f"RMSE_{period_name}", f"Bias_{period_name}"])
            work = metric_frame[["series", "MAE", "RMSE", "Bias(ME)", "RampingError_RMSE"]].copy()
            work = work.sort_values("MAE").reset_index(drop=True)
            work[f"rank_{period_name}"] = range(1, len(work) + 1)
            return work.rename(
                columns={
                    "MAE": f"MAE_{period_name}",
                    "RMSE": f"RMSE_{period_name}",
                    "Bias(ME)": f"Bias_{period_name}",
                    "RampingError_RMSE": f"RampingError_RMSE_{period_name}",
                }
            )


        ranking_comparison_df = model_metadata_df.copy()
        for period_name, metric_frame in period_metric_frames.items():
            ranking_comparison_df = ranking_comparison_df.merge(
                build_rank_table(metric_frame, period_name),
                on="series",
                how="left",
            )

        ranking_comparison_export = export_frame(ranking_comparison_df, "period_ranking_comparison.csv")

        def winner_for_period(metric_frame: pd.DataFrame, period_name: str) -> dict:
            if metric_frame.empty:
                return {"period": period_name, "winner_series": None, "winner_mae": None}
            row = metric_frame.sort_values("MAE").iloc[0]
            return {
                "period": period_name,
                "winner_series": row["series"],
                "winner_mae": row["MAE"],
            }


        period_winners_df = pd.DataFrame(
            [winner_for_period(metric_frame, period_name) for period_name, metric_frame in period_metric_frames.items()]
        )

        final_ranking_df = ranking_comparison_df.sort_values("rank_long_sample_official").reset_index(drop=True) if not ranking_comparison_df.empty else pd.DataFrame()
        WINNER_RUN_ID = str(final_ranking_df.iloc[0]["run_id"]) if not final_ranking_df.empty else None
        WINNER_LABEL = str(final_ranking_df.iloc[0]["series"]) if not final_ranking_df.empty else None

        display(final_ranking_df)
        display(period_winners_df)
        print("Ranking comparison export:", ranking_comparison_export)

        metadata_payload = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "official_long_sample_window": {
                "start_date": OFFICIAL_LONG_SAMPLE_START,
                "end_date": OFFICIAL_LONG_SAMPLE_END,
                "summary_csv": official_replay_source,
            },
            "legacy_overlap_window": {
                "start_date": LEGACY_OVERLAP_START,
                "end_date": LEGACY_OVERLAP_END,
            },
            "short_term_reference_window": {
                "start_date": SHORT_TERM_REFERENCE_START,
                "end_date": SHORT_TERM_REFERENCE_END,
            },
            "year_2026_window": {
                "start_date": YEAR_2026_START,
                "end_date": YEAR_2026_END,
            },
            "transition_window": {
                "start_date": TRANSITION_START,
                "end_date": TRANSITION_END,
            },
            "winner_run_id": WINNER_RUN_ID,
            "winner_label": WINNER_LABEL,
            "active_run_id": ACTIVE_RUN_ID,
            "config_paths": official_config_df["config_path"].tolist() if not official_config_df.empty else list(CONFIG_PATHS),
            "export_root": str(EXPORT_ROOT.resolve()),
        }
        metadata_path = write_json(EXPORT_ROOT / "metadata.json", metadata_payload)
        print("Metadata export:", metadata_path)
        """
    )

    add_md(
        """
        ## 10. Predict / Demo Export

        Cette section permet de lancer un `predict_next_day` propre pour une date choisie sur le winner (ou un run force).
        """
    )

    add_code(
        """
        demo_run_id = str(DEMO_PREDICT_RUN_ID or WINNER_RUN_ID) if (DEMO_PREDICT_RUN_ID or WINNER_RUN_ID) else None
        demo_compare_df = pd.DataFrame()

        if demo_run_id:
            demo_run_row = runs_df.loc[runs_df["run_id"].astype(str) == demo_run_id].head(1)
            demo_run_dir = (
                Path(demo_run_row.iloc[0]["run_dir"])
                if not demo_run_row.empty
                else ARTIFACTS / "runs" / "consumption" / demo_run_id
            )
            demo_output_dir = EXPORT_ROOT / "predict_demo" / demo_run_id / DEMO_PREDICT_DATE
            demo_predict_csv = demo_output_dir / f"predict_target_day_{DEMO_PREDICT_DATE}.csv"

            if RUN_PREDICT_DEMO or (LOAD_CACHED_PREDICT_DEMO and demo_predict_csv.exists()):
                if RUN_PREDICT_DEMO and (FORCE_RECOMPUTE_PREDICT_DEMO or not demo_predict_csv.exists()):
                    demo_output_dir.mkdir(parents=True, exist_ok=True)
                    result = run_cli(
                        [
                            "python",
                            "scripts/predict_next_day.py",
                            "--current-dir",
                            str(demo_run_dir),
                            "--target-date",
                            DEMO_PREDICT_DATE,
                            "--output-csv",
                            str(demo_predict_csv),
                        ]
                        + optional_cli_args(
                            dataset_key=DATASET_KEY,
                            catalog_path=CATALOG_PATH,
                            historical_csv=HISTORICAL_CSV,
                            weather_csv=WEATHER_CSV,
                            holidays_xlsx=HOLIDAYS_XLSX,
                            benchmark_csv=BENCHMARK_CSV,
                        ),
                        cwd=ROOT,
                    )
                    _ = result.extract_json()

                if demo_predict_csv.exists():
                    demo_predict_df = coerce_datetime(pd.read_csv(demo_predict_csv), "Date")
                    truth_day_df = slice_single_day(truth_baseline_df, DEMO_PREDICT_DATE)[["Date", "real", "weekly_naive"]]
                    legacy_day_df = slice_single_day(legacy_prepared_df, DEMO_PREDICT_DATE)
                    demo_compare_df = truth_day_df.copy()
                    if not legacy_day_df.empty:
                        demo_compare_df = demo_compare_df.merge(legacy_day_df, on="Date", how="left")
                    else:
                        demo_compare_df["legacy_forecast"] = pd.NA
                    demo_label = build_model_label_map([demo_run_id], runs_df=runs_df, replay_df=official_ranking_df).get(demo_run_id, demo_run_id)
                    demo_compare_df = demo_compare_df.merge(
                        demo_predict_df[["Date", "Ptot_TOTAL_Forecast"]].rename(
                            columns={"Ptot_TOTAL_Forecast": demo_label}
                        ),
                        on="Date",
                        how="left",
                    )
                    export_frame(
                        demo_compare_df,
                        f"predict_demo/{demo_run_id}/model_day_comparison_{DEMO_PREDICT_DATE}.csv",
                    )
                    plot_overlay(
                        demo_compare_df,
                        columns=[
                            column
                            for column in ["real", "weekly_naive", "legacy_forecast", demo_label]
                            if column in demo_compare_df.columns
                        ],
                        title=f"Predict demo - {demo_run_id} - {DEMO_PREDICT_DATE}",
                        resample_rule=None,
                    )
                    display(demo_compare_df.head())
            else:
                print("No predict demo cache available. Set RUN_PREDICT_DEMO=True to generate it.")
        else:
            print("No demo run selected.")
        """
    )

    add_md(
        """
        ## 11. Promotion

        Promotion optionnelle du winner long sample officiel, via la CLI repo.
        """
    )

    add_code(
        """
        print("WINNER_RUN_ID:", WINNER_RUN_ID)
        if RUN_PROMOTION and PROMOTE_LONG_SAMPLE_WINNER and WINNER_RUN_ID:
            _ = run_cli(
                [
                    "python",
                    "scripts/promote_consumption_run.py",
                    "--run-id",
                    WINNER_RUN_ID,
                ],
                cwd=ROOT,
            )
        else:
            print(
                "Promotion not executed. Set RUN_PROMOTION=True and "
                "PROMOTE_LONG_SAMPLE_WINNER=True to promote the official winner."
            )
        """
    )

    add_md(
        """
        ## Final Notes

        Exports principaux ecrits sous `artifacts/notebook_exports/cli_demo_v4/` :

        - `long_sample_replay_leaderboard.csv`
        - `model_range_comparison_long_sample.csv`
        - `long_sample_metrics_by_model.csv`
        - `per_day_metrics_by_model_long_sample.csv`
        - `legacy_overlap_metrics_by_model.csv`
        - `period_ranking_comparison.csv`
        - `worst_days_by_model_long_sample.csv`
        - `skipped_days_audit.csv`
        - `metadata.json`
        """
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    OUT_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
